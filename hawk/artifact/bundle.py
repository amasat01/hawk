# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""From a traced kernel to a deployment unit on disk.

One HAWK artifact directory is: one compiled object per declared target (a
``.ptx`` *or* ``.cubin`` for ``cuda``, whichever the compiled bytes actually
are — a ``.so`` for ``host``), the emitted SOURCE of each, one JSON sidecar
per kernel and one ``manifest.json`` per bundle. Every compile goes through
the content-closure cache, so a rebuild of an unchanged kernel spawns no
compiler. Nothing here launches anything: eagle loads and runs the device
object (``eagle.registry.load_manifest`` -> ``eagle.plan``) and the host
object (``eagle.HostTeam``); HAWK's job ends at the bytes on disk.

Publishing is content-addressed and idempotent — a unit's identity is the
digest of what it contains, not the directory it sits in. Within one
process, :func:`build_bundle` remembers each unit under a key naming every
input its content is a function of: a repeat build into the same directory
is a no-op, a different one gets the published files COPIED there (never a
second compile). Across processes, a digest STAMP (:data:`STAMP_NAME`)
marks a directory as already holding that unit.

Neither memo short-circuits validity: a hit re-verifies every TU's
dependency closure (:func:`~hawk.compile.cache.ClosureWatch`) and the
unit's own published files (:func:`~hawk.compile.digest_file`), both by
content. A changed header, a deleted unit or a corrupted binary falls
through to a real publish; :func:`publish_log` names which and why.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from types import SimpleNamespace
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..compile import digest_file
from ..compile.cache import ClosureWatch, publish_atomic, tmp_sibling
from ..passes import policy
from ..passes.policy import Settings
from ..trace.kernel import derived_axes as _derived_axes
from ..emit import SegmentSpec, scalar_mode
from ..targets.cards import thread_box as _thread_box_card
from .unit_cache import BOX_ENV
from ..emit.render import binding_name as _binding_name
from ..emit.wrapper import ActiveSpec
from ..ext import DEFAULT_KIND
from ..ir import DispatchInfo
from .._errors import HawkError
from ..ir.nodes import LEAF_ROLES
from ..ir import canonical as _canonical
from ..ir import split_segmented as _split_segmented
from ..ir.segment import rewalk as _segment_rewalk
from .. import components as _components
_box = _components.view("lowering", "box")
_wide = _components.view("lowering", "wide")
_split = _components.view("lowering", "split")
_team = _components.view("lowering", "team")
_wt = _components.view("lowering", "wrapper_text")
from .. import options as _options_mod
from ..build_policy import BuildPolicy, apply as _apply_policy, settings_of as _settings_of
from ..backends import BuiltLowered, Lowered
from ..registry import KNOWN, check, denial, missing, traits
from ..targets import DEFAULT_TARGETS, TARGETS, build_spec
from ..types import Slot, TensorType
from . import manifest as _manifest
from .derivative import _per_kernel_derivative
from .sidecar import sidecar_meta
from .unit_cache import (
    _UNIT_MEMO,
    _UNIT_MEMO_LIMIT,
    _closure_watch,
    _first_moved,
    _log,
    _note,
    _stamp_digest,
    _unit_key,
    _write_stamp,
    declaration_digest,
    arch,
)
from .unit_cache import COUNTERS as COUNTERS
from .unit_cache import bump_builds as _bump_builds
from .unit_cache import STAMP_NAME as STAMP_NAME
from .unit_cache import counters_snapshot as counters_snapshot
from .unit_cache import publish_log as publish_log
from .unit_cache import reset_unit_memo as reset_unit_memo
from .unit_cache import unit_stats as unit_stats


@dataclass(frozen=True)
class Entry:
    """One built target of one kernel. An exported entry's ``artifact`` is its
    header, its ``source`` the unit and its ``report`` the
    ``<name>.embedded.json``; ``report`` is ``None`` for every compiled target."""

    target: str
    artifact: Path
    source: Path
    key: str
    hit: bool
    seconds: float
    closure: tuple = field(default=())
    report: Path | None = None


@dataclass(frozen=True)
class Artifact:
    """One kernel's published files + the sidecar it was described by."""

    name: str
    directory: Path
    sidecar_path: Path
    sidecar: dict
    entries: dict


@dataclass(frozen=True)
class Bundle:
    """A deployment unit: one ``manifest.json`` over one or more artifacts.

    ``digest`` is the unit's own content identity — the same value
    :func:`_unit_digest` computes and :data:`STAMP_NAME` records, exposed
    here because a second unit's derivative can name this one as its
    primal's home (:func:`_per_kernel_derivative`'s cross-unit reference).

    ``manifest_path``/``manifest`` are ``None`` for a unit built for the
    ``c`` target alone: a standalone C unit has nothing for eagle to load."""

    directory: Path
    manifest_path: Path | None
    manifest: dict | None
    artifacts: tuple
    digest: str | None = None

    def files(self) -> list:
        """EVERY file this unit published — the set row greps. Each path
        once: a c-only artifact's sidecar IS its ``c`` entry's report."""
        out = [] if self.manifest_path is None else [self.manifest_path]
        for a in self.artifacts:
            out.append(a.sidecar_path)
            for entry in a.entries.values():
                out += [entry.artifact, entry.source]
                if entry.report is not None:
                    out.append(entry.report)
        return list(dict.fromkeys(out))


# -- The unit's CONTENT, computed before anything is written. -------------- #
@dataclass(frozen=True)
class _File:
    """One file of a unit: its name, its bytes, and their digest.

    ``blob`` is carried inline (HAWK-made text) or fetched from the
    content-closure cache on demand (a compiled object, never held in
    memory). Either way the digest is of the bytes on disk, so the unit
    digest is re-computable from the published tree by anyone."""

    name: str
    digest: str
    blob: bytes | None = None
    source_path: Path | None = None

    def write(self, directory: Path) -> Path:
        path = directory / self.name
        # Written whole to a private sibling, then renamed into place: a
        # concurrent reader (or a second builder of the same unit) sees the
        # old file or the new one, never a partial write.
        tmp = tmp_sibling(path)
        try:
            if self.blob is not None:
                tmp.write_bytes(self.blob)
            else:
                shutil.copyfile(self.source_path, tmp)
            publish_atomic(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path


@dataclass(frozen=True)
class _Draft:
    """One kernel's share of a unit, compiled but not yet published."""

    name: str
    sidecar: dict
    sidecar_name: str
    entries: dict                      # target -> (result, art name, src name, fmt)
    files: tuple                       # the _File records this kernel contributes
    report_files: dict                 # target -> its report's file name, or None


@dataclass(frozen=True)
class _Unit:
    """A published unit, remembered: what it is, where it is, and what its
    validity still depends on."""

    bundle: Bundle
    digest: str
    files: tuple                       # (absolute path, content digest) pairs
    closure: ClosureWatch              # every target's validity record, watched
    kind: object                       # kept ALIVE: the key names it by identity


def _blob(text: str) -> bytes:
    """The bytes a text file is published as, hashed as bytes so the digest
    reflects what's on disk, not an encoding assumption."""
    return text.encode("utf-8")


def _digest_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _unit_digest(files) -> str:
    """The unit's identity: its files' NAMES and CONTENT digests, sorted so
    two publishes of the same content agree regardless of build order."""
    h = hashlib.sha256()
    h.update(b"hawk.artifact.unit/1\n")
    for f in sorted(files, key=lambda f: f.name):
        h.update(f.name.encode("utf-8"))
        h.update(b"\0")
        h.update(f.digest.encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _segment_spec(kernel) -> SegmentSpec | None:
    """Build the :class:`~hawk.emit.wrapper.SegmentSpec` a segmented unit's
    source render needs, or ``None`` for an ordinary kernel. Resolves
    ``kernel.segment`` (:class:`hawk.ir.segment.SegmentInfo`) to a C++
    binding name here, once, since ``hawk/emit/`` itself knows nothing of
    ``hawk/ir/segment.py``."""
    info = getattr(kernel, "segment", None)
    if info is None:
        return None
    return SegmentSpec(_binding_name("lookup", info.offsets), info.j)


#: The two ``lookup`` planes an active-set kernel reads
#: (``Guard(active_set=True)``): the ascending live sample indices and their
#: count. Matches eagle's own ``eagle._active_set.MAP_PLANE``/``COUNT_PLANE``.
ACTIVE_MAP = "active_map"
ACTIVE_COUNT = "active_count"
_MAP_TTYPE = TensorType((), "int32")
_MAP64_TTYPE = TensorType((), "i64")
_COUNT_TTYPE = TensorType((), "uint32")


class _ActiveUnit:
    """A kernel seen through its ACTIVE-SET walk: the kernel's own walk plus
    the two declared ``lookup`` planes the map prologue reads; everything
    else falls through to the kernel's own attributes."""

    def __init__(self, kernel, walk) -> None:
        self._kernel = kernel
        self.walk = walk

    def __getattr__(self, attr):
        return getattr(self._kernel, attr)


def _lane_set(kind):
    """The :class:`~hawk.ext.IndexSet` ``kind`` declares, else ``None``."""
    return getattr(kind, "lanes", None)


def _reads_map(kind) -> bool:
    """Whether ``kind`` reads an index map: a thinning guard or a lane set."""
    return bool(getattr(getattr(kind, "guard", None), "active_set", False)
                or _lane_set(kind) is not None)


def _map_bits(kind) -> int:
    lanes = _lane_set(kind)
    return 32 if lanes is None else lanes.bits


def _active_unit(kernel, name: str, kind):
    """``kernel`` as an :class:`_ActiveUnit` when ``kind`` asks for an index
    map (``Guard(active_set=True)`` or ``lanes=IndexSet()``), else ``kernel``
    itself. Refuses a segmented unit and one already using a reserved plane
    name; the thinning guard also refuses a kernel accumulating across
    samples (the map would reorder the sum) and one binding none of its
    masks, which a lane set, a list the caller built, does not."""
    if not _reads_map(kind):
        return kernel
    what = "lanes=IndexSet()" if _lane_set(kind) is not None else "Guard(active_set=True)"
    walk = kernel.walk
    if (getattr(kernel, "segment", None) is not None
            or any(d.policy == "segmented" for d in walk.dispatches)):
        raise HawkError(
            f"{name!r}: {what} cannot read an index map in a "
            "segmented unit — the segmented prologue already addresses the run"
        )
    if _lane_set(kind) is None:
        if any(role == "accum_out" for role, _ in walk.arg_spec):
            raise HawkError(
                f"{name!r}: Guard(active_set=True) refuses a kernel that accumulates "
                "across samples (accum_out): the map would change which samples each "
                "lane sums, so the result would not be the map-free one bit for bit"
            )
        if not any(("terminated", m) in walk.slot_of for m in kind.guard.names):
            raise HawkError(
                f"{name!r}: Guard(active_set=True) needs the kernel to bind the mask "
                f"the map follows ({list(kind.guard.names)}), and it binds none"
            )
    prefix = bool(getattr(_lane_set(kind), "prefix", False))
    taken = [n for _, n in walk.arg_spec if n in ((ACTIVE_COUNT,) if prefix else (ACTIVE_MAP, ACTIVE_COUNT))]
    if taken:
        raise HawkError(
            f"{name!r}: {taken[0]!r} is reserved for the index map "
            f"under {what}; rename that parameter"
        )
    declared = tuple(Slot(role, n, walk.slot_types[(role, n)])
                     for role, n in walk.arg_spec if role in LEAF_ROLES)     # a row sum's companion is no leaf
    declared += (() if prefix else (Slot("lookup", ACTIVE_MAP,
                                         _MAP64_TTYPE if _map_bits(kind) == 64 else _MAP_TTYPE),))
    declared += (Slot("lookup", ACTIVE_COUNT, _COUNT_TTYPE),)
    return _ActiveUnit(kernel, _canonical(kernel.sinks, declared=declared))


def _active_spec(kind, group: int = 1) -> ActiveSpec | None:
    """The map prologue's bindings when ``kind`` asks for one, else ``None``;
    ``group`` is the kernel's declared compaction group."""
    if not _reads_map(kind):
        return None
    narrow = _lane_set(kind) is not None and _map_bits(kind) == 32
    if getattr(_lane_set(kind), "prefix", False):
        return ActiveSpec(None, _binding_name("lookup", ACTIVE_COUNT), group, 32, narrow)
    return ActiveSpec(_binding_name("lookup", ACTIVE_MAP),
                      _binding_name("lookup", ACTIVE_COUNT), group, _map_bits(kind), narrow)


def _comparable(value):
    """A setting as it is compared: a sequence (``launch_bounds``) as a tuple, anything else as it is."""
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _trait(target: str, key: str, default):
    """One datum of a target's manifest row, without importing its backend."""
    return traits(target).get(key, default)


def _native(targets) -> tuple:
    """The targets this module compiles and the manifest describes: every
    one whose row has a compiler. The rest are exported units
    (:mod:`hawk.embedded`)."""
    return tuple(t for t in targets if _trait(t, "builds", True))


def _exported(targets) -> tuple:
    """The targets whose row is written whole and never compiled."""
    return tuple(t for t in targets if not _trait(t, "builds", True))


def _has_device(targets) -> bool:
    """Whether any known target is built for a device."""
    return any(_trait(t, "device", False) for t in targets if t in TARGETS)


def _device_only(targets) -> bool:
    """Whether every known target is built for a device."""
    return all(_trait(t, "device", False) for t in targets if t in TARGETS)


def _lead(entries) -> str:
    """The compiled entry a manifest plugin names: the device object when the
    unit has one, else the first compiled."""
    native = _native(entries)
    return next((t for t in native if _trait(t, "device", False)), native[0])


def _refuse_cells_off_device(name: str, smode, targets) -> None:
    """Refuse a banded build with a target that is not a device, before anything is emitted."""
    for t in targets:
        if t in TARGETS:
            _wt.refuse_cells_off_device(name, smode, t, SimpleNamespace(device=_trait(t, "device", False)))


def _refuse_libc_names(kernels) -> None:
    """Refuse an authored kernel named after a C library function: its ``extern "C"`` entry would redeclare it (a
    chain or a derivative carries no source digest and is named by its builder)."""
    from ..compile.toolchain import libc_symbols

    for k in kernels:
        if getattr(k, "source_digest", None) is not None and k.name in libc_symbols():
            raise HawkError(f"kernel {k.name!r}: the name is a C library function, and the kernel's "
                            "entry would redeclare it; rename the kernel")


def _refuse_targets(name: str, targets) -> None:
    """Refuse an empty ``targets`` or an id that is no row of
    :data:`hawk.targets.TARGETS`, naming the valid ones."""
    check()

    def hint() -> str:
        each = [f"targets=({t!r},)" for t in TARGETS]
        return f"pass {', '.join(each[:-1])} or {each[-1]}"

    if not targets:
        raise HawkError(f"{name!r}: no target to build; {hint()}")
    for target in targets:
        if target not in TARGETS:
            if why := denial("backends", target):
                raise HawkError(f"target {target!r} is refused: {why}")
            ids = [repr(t) for t in TARGETS]
            about = ", ".join(f"{t!r} {traits(t).get('about', '')}" for t in TARGETS)
            raise HawkError(
                f"unknown target {target!r}: the valid targets are "
                f"{', '.join(ids[:-1])} and {ids[-1]} ({about}); {hint()}"
                + (f"; {missing(target)}" if target in KNOWN else ""))


def lower(kernel, name, smode, kind, targets, spec) -> Lowered:
    """What lowering works out ONCE per kernel for every target (:class:`~hawk.backends.Lowered`), with the refusals that
    come with it. No compiler, no disk.

    A kernel with its own ``fused_body`` hook (a composed lane group) renders it ONCE, so every target wraps the same
    string."""
    _refuse_targets(name, targets)
    hook = getattr(kernel, "fused_body", None) if _native(targets) else None
    rows = getattr(getattr(kernel, "team", None), "lanes", 1)         # lanes a block of the warp form serves
    body = None if hook is None else hook(kind, cells=smode.cells, spec=spec.with_settings(staged_rows=rows))
    segment = _segment_spec(kernel)
    if segment is not None and kernel.walk.reads_lane_position:
        raise HawkError(f"{name!r}: hawk.lane_position() is the launch slot, and a segmented "
                        "unit addresses its lanes through the run's offsets; use "
                        "sample_index() there")
    active = _active_spec(kind, getattr(kernel, "group", 1))
    one_step = getattr(kernel, "one_step", None)
    box = _box.thread_box(kernel, smode.id, spec=spec, card=_thread_box_card(smode.id, spec), enabled=os.environ.get(BOX_ENV) != "0",
                          kind=kind, segment=segment, active=active, one_step=one_step, body=body)
    _wt.refuse_team_reductions(getattr(kernel, "team", None), smode)
    return BuiltLowered(kernel, name, smode, kind, body, segment, one_step, box, active, rows, spec)


def _options(*, device_arch: str, options=None, layout_sizes_override=None,
             defines=(), cache_dir=None, host_profile: str = "", opt_level: str = "", block: int | None = None) -> dict:
    """The build inputs backends read (:meth:`hawk.backends.Backend.emit`, ``build``): shared keys flat
    (``device_arch`` is the resolved one), the user's ``options`` under their ``<id>.`` prefix."""
    return {**(options or {}), "device_arch": device_arch, "layout_sizes_override": layout_sizes_override,
            "defines": tuple(defines), "cache_dir": cache_dir, "host_profile": host_profile or "",
            "opt_level": opt_level or "", "block": block}


def _emit(lowered, targets, options) -> dict:
    """Every declared target's emitted TU: the cheap half of a build (no compiler, no disk, no sidecar)."""
    return {target: TARGETS[target].emit(lowered, options) for target in targets}


def _step_ops(kernel, kind, spec) -> int | None:
    """The operation count of one step of a kernel that finishes its own
    samples by stepping (its body cuts into head, one step and tail), else
    ``None``. A fact for the launching runtime: the emitted code does not
    read it."""
    from ..emit.render import render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return None
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kind, spec=spec)
    return None if split is None else int(split.op_count)


def _fp64_ops_ir(kernel, kind, spec) -> int | None:
    """The 64-bit floating-point operations one step of a stepping kernel
    runs, counted in the IR (as :func:`_step_ops`, ``None`` for any other
    kernel): the step cost the launching runtime's chunk length reads."""
    from ..emit.render import render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return None
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kind, spec=spec)
    return None if split is None else int(split.fp64_ops)


def _entry_counts_finished(kernel, kind, spec) -> bool:
    """Whether the kernel's fast entries count a sample finished on entry into
    the finish counter (:func:`hawk.emit.render.entry_counts_finished`), so the
    launching runtime need not count the mask before a run."""
    from ..emit.render import entry_counts_finished, render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return False
    return entry_counts_finished(kernel.walk, render_lane_split(kernel.sinks, kernel.walk, kind=kind, spec=spec))


@dataclass(frozen=True)
class _ExportResult:
    """An exported target's stand-in for a compile result: the fields
    :func:`_artifact_of`, :func:`_served` and :func:`_closure_watch` read.
    ``key`` is the sha256 of the ``.c`` and ``.h`` texts (no compile cache
    entry exists for it, so its closure is empty), ``seconds`` the toolchain
    checks' run time."""

    key: str
    hit: bool
    seconds: float
    closure: tuple = ()


def _emit_arch(targets, device_arch: str) -> str:
    """The device arch the emit may read cards for: resolved (once, memoised) only
    when a target compiles with the device toolchain, so a host-only build never
    probes."""
    return arch(device_arch) if _has_device(targets) else ""


def _sidecar(kernel, *, spec, name, smode, kind, native, fmt, lead, derivative=None, box=None, block=None,
             ptxas=None, local=None, wide=None) -> dict:
    """The sidecar of one kernel's unit. What only a compile knows (``ptxas``, ``local``, the resolved ``wide`` record) is
    optional: :func:`emit_source` leaves it out."""
    segment = getattr(kernel, "segment", None)
    dispatch_override = None if segment is None else (
        DispatchInfo(segment.dispatch_name, segment.K, "segmented", segment.units,
                     selector_reads=segment.selector_reads,
                     static_selector=segment.static_selector),
    )
    return sidecar_meta(
        name, kernel.walk,
        fmt=fmt,
        scalar_type=smode.id,
        exec_targets=[TARGETS[t].entry for t in native],
        host_entry=(f"{name}_host"
                    if any(TARGETS[t].host_entry for t in native) else None),
        derivative=derivative,
        lanes=getattr(kernel, "lanes", None),
        lane_space=getattr(kernel, "lane_space", ((None, None),)),
        group=getattr(kernel, "group", 1),
        box=box,
        block=block,
        index_set=(None if _lane_set(kind) is None else
                   {"width": 32, "prefix": True, "count": ACTIVE_COUNT} if _lane_set(kind).prefix
                   else {"width": _map_bits(kind)}),
        planes=getattr(kernel, "planes", None) or _derived_axes(
            kernel.walk, getattr(kernel.sinks, "source", None)),
        broadcast=getattr(kernel, "broadcast", None),
        dispatch_override=dispatch_override,
        step_ops=_step_ops(kernel, kind, spec),
        fp64_ops_ir=_fp64_ops_ir(kernel, kind, spec),
        entry_counts_finished=_entry_counts_finished(kernel, kind, spec),
        ptxas=ptxas,
        local=local,
        wide=wide,
    )


def _run_facts(kernel, meta: dict, spec) -> dict:
    """The variant numbers a run reads from the unit's sidecar: ``cell_row`` (samples per row of a reduction over samples; a unit with
    cells) and ``scatter_chunks`` (the most chunks of an ordered scatter; a unit with a team stage). A build whose host and device
    variants state different row widths is refused by name: one unit has one row width."""
    out = {}
    stages = meta.get("stages") or ()
    if meta.get("cells"):
        widths = {side: v["cells"]["row"] for side, v in (("host", spec.host), ("device", spec.device)) if v and "row" in v.get("cells", {})}
        if len(set(widths.values())) > 1:
            raise HawkError(f"one unit has one row width: the host variant has cells.row {widths['host']}, the device variant {widths['device']}")
        out["cell_row"] = int(spec.need("cells", "row"))
    if getattr(kernel, "team", None) is not None or meta.get("team_planes") or any("team" in st for st in stages):
        out["scatter_chunks"] = int(spec.need("scatter", "max_chunks"))
    return out


def _draft(kernel, lowered, sources, options, *, targets, derivative, split: dict | None = None) -> _Draft:
    """Build every target and assemble the unit CONTENT for one kernel.

    Everything here is a pure function of the inputs plus the
    content-closure cache; nothing is written, which is what lets the
    publish step ask "is this already on disk?" before "where do I put
    it?".

    Each target's backend builds its own :class:`~hawk.backends.Part` from the one
    ``lowered`` kernel, its ``sources[target]`` and ``options``
    (:meth:`hawk.backends.Backend.build`); ``options["device_arch"]`` is the arch the
    driver resolved once, and only when a target builds for a device, so a host-only
    build never pays for the probe.

    A target that is exported whole is not compiled: its part carries its unit, its
    header and its report. The JSON sidecar describes the NATIVE targets
    only and is written only when there is one; an export-only draft carries
    the report in the sidecar's place and writes no ``<name>.json``."""
    name, smode, kind = lowered.name, lowered.smode, lowered.kind
    native = _native(targets)
    parts = _build_targets(lowered, sources, options, native)
    parts.update({t: TARGETS[t].build(lowered, sources[t], options) for t in _exported(targets)})
    entries, files = {}, []
    report = None
    for target in (*native, *_exported(targets)):
        part = parts[target]
        files += part.files
        entries[target] = part.entry
        report = part.report if part.report is not None else report
    report_files = {t: p.report_file for t, p in parts.items()}
    if not native:
        return _Draft(name, report, report_files[_exported(targets)[0]], entries, tuple(files), report_files)

    lead = _lead(entries)
    meta = _sidecar(
        kernel, spec=lowered.spec, name=name, smode=smode, kind=kind, native=native,
        # The real format: the detected device format when a device object
        # was built, else the first compiled one — mirrors build_bundle's
        # manifest-entry choice so the sidecar and manifest never disagree.
        fmt=entries[lead][3], lead=lead, derivative=derivative,
        box=getattr(sources[lead], "box", None),
        **parts[lead].facts,
        wide=_wide_with_class({**_wide.walk_record(kernel.walk), **_wide.walk_record(kernel.one_step.walk)} if getattr(kernel, "one_step", None) is not None
                              else _wide.walk_record(kernel.walk), lowered.spec),
    )
    if smode.cells:
        for plane in getattr(getattr(kernel, "team", None), "planes", ()):    # a partial-sum plane is 16 bytes an element
            if plane.dtype == _team.PARTIAL and plane.name in meta["arg_dtypes"]:
                meta["arg_dtypes"][plane.name] = "band_partial"
    if split is not None:
        # the first stage of a split kernel names every stage and every cell
        meta.update(split)
    meta.update(_run_facts(kernel, meta, lowered.spec))
    sidecar_blob = _blob(json.dumps(meta, indent=2) + "\n")
    files.append(_File(f"{name}.json", _digest_bytes(sidecar_blob),
                       blob=sidecar_blob))
    return _Draft(name, meta, f"{name}.json", entries, tuple(files), report_files)


def compile_jobs() -> int:
    """How many of one kernel's target compiles may run at once:
    ``$HAWK_COMPILE_JOBS`` (a positive integer), else 2 — the host and the
    device compile side by side. ``1`` compiles them one after the other.
    An invalid value RAISES."""
    raw = os.environ.get("HAWK_COMPILE_JOBS", "").strip()
    if not raw:
        return 2
    try:
        jobs = int(raw)
    except ValueError:
        jobs = 0
    if jobs < 1:
        raise HawkError(
            f"$HAWK_COMPILE_JOBS={raw!r} is not a positive integer (the number "
            "of a kernel's target compiles allowed to run at once; 1 = one "
            "after the other)")
    return jobs


def _build_targets(lowered, sources, options, targets) -> dict:
    """``build`` of every target in ``targets``, side by side when
    there is more than one (:func:`compile_jobs` at most at once): the host
    compiler and the device compiler are separate processes on separate
    cache slots, so neither waits for the other. Parts are keyed by
    target and consumed in ``targets`` order, so the unit is the same
    whichever finishes first; a failure is raised for the FIRST failing
    target in that order — the error a one-after-the-other build reports —
    after every started build has finished."""
    targets = list(targets)
    jobs = min(len(targets), compile_jobs())
    if jobs <= 1:
        return {t: TARGETS[t].build(lowered, sources[t], options) for t in targets}
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="hawk-compile") as pool:
        futures = {t: pool.submit(contextvars.copy_context().run, TARGETS[t].build, lowered, sources[t], options)
                   for t in targets}
        for future in futures.values():
            future.exception()  # wait for every build before reporting any
    return {t: futures[t].result() for t in targets}


def _artifact_of(draft: _Draft, directory: Path) -> Artifact:
    """The public :class:`Artifact` view of a drafted kernel published at
    ``directory``. Built the same way whether the files were just written or
    were already there — a served unit and a fresh one are the same object."""
    entries = {
        target: Entry(target, directory / art_name, directory / src_name,
                      result.key, result.hit, result.seconds,
                      tuple(result.closure),
                      None if draft.report_files[target] is None else directory / draft.report_files[target])
        for target, (result, art_name, src_name, _fmt) in draft.entries.items()
    }
    return Artifact(draft.name, directory, directory / draft.sidecar_name,
                    draft.sidecar, entries)


def _served(bundle: Bundle) -> Bundle:
    """The bundle a MEMO HIT hands back: identical, except every entry is
    marked ``hit`` with zero compile seconds — needed because the
    compile-floor rows assert ``not entry.hit`` to prove they time a
    compiler and not a lookup."""
    artifacts = tuple(
        Artifact(a.name, a.directory, a.sidecar_path, a.sidecar,
                 {t: replace(e, hit=True, seconds=0.0)
                  for t, e in a.entries.items()})
        for a in bundle.artifacts
    )
    return Bundle(bundle.directory, bundle.manifest_path, bundle.manifest, artifacts,
                 digest=bundle.digest)


def _rehome_bundle(bundle: Bundle, directory: Path) -> Bundle:
    """``bundle``, pointed at ``directory`` instead of wherever it was
    actually built: every path field rewritten to the same names under the
    new directory, everything else carried over untouched. Rehoming names
    where the bytes now ARE; it writes nothing itself —
    :func:`_materialize` is what puts them there first."""
    artifacts = tuple(
        Artifact(a.name, directory, directory / a.sidecar_path.name, a.sidecar,
                 {t: replace(e, artifact=directory / e.artifact.name,
                             source=directory / e.source.name,
                             report=_rehomed(e.report, directory))
                  for t, e in a.entries.items()})
        for a in bundle.artifacts
    )
    return Bundle(directory, _rehomed(bundle.manifest_path, directory),
                 bundle.manifest, artifacts, digest=bundle.digest)


def _rehomed(path: Path | None, directory: Path) -> Path | None:
    """``path``'s name under ``directory``; ``None`` (no manifest, no
    report) stays ``None``."""
    return None if path is None else directory / path.name


def _materialize(bundle: Bundle, files, digest: str, directory: Path) -> Bundle:
    """Copy a MEMOISED unit's files into a different ``directory`` than the
    one it was published at, and return it rehomed there
    (:func:`_rehome_bundle`) — a second physical copy, no recompile.

    A no-op when ``directory`` already holds this unit (the same
    cross-process stamp check :func:`build_bundle`'s Phase 3 makes). Every
    file is a literal byte copy; this stays content-correct because the
    manifest/sidecar carry no absolute path, only file names."""
    pairs = [(Path(p).name, p, d) for p, d in files]
    if (_stamp_digest(directory) == digest and _first_moved(
            [(str(directory / n), d) for n, _p, d in pairs]) is None):
        return _rehome_bundle(bundle, directory)
    directory.mkdir(parents=True, exist_ok=True)
    copied = tuple(_File(n, d, source_path=Path(p)) for n, p, d in pairs)
    for f in copied:
        f.write(directory)
    _write_stamp(directory, digest, copied)
    return _rehome_bundle(bundle, directory)


def _resolve_kind(name: str, kernel_kind, explicit_kind):
    """The ``build(kernel, kind=None)`` rule: ``None`` takes ``kernel.kind``;
    an explicit ``kind`` differing from a non-default ``kernel.kind``
    refuses. A bare-``@kernel`` :data:`DEFAULT_KIND` accepts any explicit
    ``kind``."""
    kernel_kind = kernel_kind if kernel_kind is not None else DEFAULT_KIND
    if explicit_kind is None:
        return kernel_kind
    if kernel_kind is not DEFAULT_KIND and explicit_kind != kernel_kind:
        raise HawkError(
            f"{name!r}: build() was given kind={explicit_kind!r} but the "
            f"kernel was traced under kind={kernel_kind!r} (via "
            f"@{kernel_kind.slug} or Kind(...)(fn)); an explicit kind "
            "differing from the kernel's own is refused — no silent override"
        )
    return explicit_kind


def _resolve_bundle_kind(names: list, kernels: list, explicit_kind):
    """The same rule as :func:`_resolve_kind`, for a whole bundle: every
    member's own (non-default) kind must agree with the resolved one, and
    with no explicit ``kind`` the members may not disagree among
    themselves either — a bundle renders under one kind."""
    tagged = [(n, getattr(k, "kind", None) or DEFAULT_KIND)
             for n, k in zip(names, kernels)]
    tagged = [(n, k) for n, k in tagged if k is not DEFAULT_KIND]
    if explicit_kind is not None:
        conflicting = [(n, k) for n, k in tagged if k != explicit_kind]
        if conflicting:
            n, k = conflicting[0]
            raise HawkError(
                f"build_bundle(): kind={explicit_kind!r} was given but {n!r} "
                f"was traced under kind={k!r}; an explicit kind differing "
                "from a member kernel's own is refused — no silent override"
            )
        return explicit_kind
    distinct: list = []
    for _n, k in tagged:
        if not any(k == d for d in distinct):
            distinct.append(k)
    if len(distinct) > 1:
        raise HawkError(
            "build_bundle(): member kernels declare different kinds "
            f"{[d.slug for d in distinct]}; a bundle renders under ONE kind "
            "— pass kind= explicitly to pick one"
        )
    return distinct[0] if distinct else DEFAULT_KIND


# -- The public entry points. ----------------------------------------------- #
def build(kernel, directory, *, mode: str = "float64", targets=DEFAULT_TARGETS,
          kind=None, layout_sizes_override=None, defines=(), cache_dir=None,
          device_arch: str = "", name: str | None = None,
          derivative: dict | None = None, host_profile: str = "",
          opt_level: str = "", literal_table: bool = False,
          embedded_profile=None, options: Mapping | None = None, spec=None,
          build_policy: BuildPolicy | None = None) -> Artifact:
    """Emit, compile and publish ONE kernel into ``directory``.

    ``kernel`` is anything carrying ``.name``/``.sinks``/``.walk`` (a
    :class:`hawk.trace.Kernel`). ``layout_sizes_override=`` is a test-only
    door threaded to the export block.

    This is the PRIMITIVE, not the deployment unit: no ``manifest.json``, no
    stamp, no unit memo — :func:`build_bundle` publishes a unit (one
    manifest per bundle) with content-addressed publishing.

    ``opt_level`` picks the optimisation level
    (:data:`hawk.compile.toolchain.OPT_LEVELS`), exactly like
    ``host_profile`` picks the host profile: empty means
    ``$HAWK_OPT_LEVEL``, else :data:`hawk.compile.toolchain.DEFAULT_OPT_LEVEL`
    (``"O3"``). See :func:`build_bundle`'s own docstring for the full rule.

    ``literal_table`` (default ``False``) and ``embedded_profile``: see
    :func:`build_bundle`.

    Refuses a ``segmented`` dispatch, which needs K artifacts: call
    :func:`build_bundle` with ``[kernel]`` instead."""
    name = name or kernel.name
    options = _user_options(options, literal_table, embedded_profile, device_arch, host_profile)
    device_arch, host_profile = _options_mod.shared(options)
    kernel, lowered, sources, options = _emitted(
        kernel, name, mode, kind, targets, device_arch, "build() publishes", "build_bundle", spec, build_policy=build_policy,
        layout_sizes_override=layout_sizes_override, options=options,
        defines=defines, cache_dir=cache_dir, host_profile=host_profile, opt_level=opt_level)
    directory = Path(directory)
    draft = _draft(kernel, lowered, sources, options, targets=targets, derivative=derivative)
    directory.mkdir(parents=True, exist_ok=True)
    for f in draft.files:
        f.write(directory)
    return _artifact_of(draft, directory)


def _plain_block_sums(kernel, spec):
    """``kernel`` as the build's forms ask: every block sum the ``linalg_form`` does not keep made its plain sum (the tracer
    keeps every block-shaped sum as one :class:`~simt.team.BlockSum`) and, under ``wide_form="thread"``, each store
    under a condition put back as a select over the store (:func:`simt.wide.unstored`); else ``kernel`` itself. A compensated sum gets
    the companion plane the tracer binds for a plain one."""
    from ..trace.kernel import Kernel, _policy_companions

    if type(kernel) is not Kernel:
        return kernel
    sinks = _team.plain_blocks(kernel.sinks, spec.settings.linalg_form, spec)
    if spec.settings.wide_form == "thread":
        sinks = _wide.unstored(sinks)
    one = _plain_block_sums(kernel.one_step, spec) if kernel.one_step is not None else None
    if all(a is b for a, b in zip(sinks, kernel.sinks)) and len(sinks) == len(kernel.sinks) and one is kernel.one_step:
        return kernel
    planes = dict(kernel.planes)
    old = {id(s) for s in kernel.sinks}
    added = tuple(s for s in sinks if id(s) not in old)
    sinks = tuple.__new__(type(kernel.sinks), (*sinks, *_policy_companions(kernel.name, planes, added)))
    if hasattr(kernel.sinks, "__dict__"):
        sinks.__dict__.update(vars(kernel.sinks))
    out = Kernel(kernel.name, sinks, planes, kind=kernel.kind, step=kernel.step, one_step=one,
                 source_digest=kernel.source_digest, returns=kernel.returns)
    out.time, out.time_step, out.jump, out.broadcast = kernel.time, kernel.time_step, kernel.jump, dict(kernel.broadcast)
    return out


def _user_options(options, literal_table, embedded_profile, device_arch, host_profile) -> dict:
    """The user's ``options`` with the old keywords folded in, checked against the manifests (:mod:`hawk.options`)."""
    merged = _options_mod.merge(options, literal_table=literal_table, embedded_profile=embedded_profile,
                                device_arch=device_arch, host_profile=host_profile)
    _options_mod.check(merged)
    return merged


def _emitted(kernel, name, mode, kind, targets, device_arch, does, instead, spec, build_policy=None, **inputs) -> tuple:
    """``(unit, lowered, sources, options)`` of ONE kernel (``inputs`` are the :func:`_options` keywords but the arch): the emit half of :func:`build` and :func:`emit_source`. A kernel that
    needs several launches (a segmented dispatch, a value reduced over samples) is refused: ``does`` says what the caller
    publishes and ``instead`` what to call."""
    _refuse_targets(name, targets)
    emit_arch = _emit_arch(targets, device_arch)
    spec = build_spec(tuple(targets), emit_arch) if spec is None else spec
    spec = _apply_policy(spec, build_policy, targets).with_settings(route_to_ordered=scalar_mode(mode).cells)
    kernel = _plain_block_sums(kernel, spec)
    if any(d.policy == "segmented" for d in kernel.walk.dispatches):
        raise HawkError(f"{name!r} carries a segmented dispatch: {does} ONE artifact, and a segmented kernel needs K "
                        f"— call {instead}([kernel], ...) instead")
    if _split.stages(name, kernel.sinks, spec=spec) is not None:
        raise HawkError(f"{name!r} reads a value reduced over samples, so it runs as several launches: {does} ONE "
                        f"artifact — call {instead}([kernel], ...) instead")
    kind = _resolve_kind(name, getattr(kernel, "kind", None), kind)
    kernel = _active_unit(kernel, name, kind)
    smode = scalar_mode(mode)
    _refuse_libc_names((kernel,))
    _refuse_cells_off_device(name, smode, targets)
    lowered = lower(kernel, name, smode, kind, targets, spec)
    options = _options(device_arch=emit_arch, **inputs)
    sources = _emit(lowered, targets, options)
    return kernel, lowered, sources, options


@dataclass(frozen=True)
class EmitUnit:
    """What :func:`emit_source` returns: ``sources`` (target name to its :class:`hawk.emit.Source`, the ``.text``
    of each as :func:`build` would write it), the ``sidecar`` dict without the figures only a compile gives (``ptxas``,
    ``launch.local_*``, ``launch.wide``, and ``format`` as the target row's default, not the compiled object's), and ``unit``, the kernel as it was emitted (with the lane-set slots appended)."""

    sources: dict
    sidecar: dict
    unit: object


def emit_source(kernel, *, name: str | None = None, mode: str = "float64", targets=DEFAULT_TARGETS, kind=None,
                device_arch: str = "", literal_table: bool = False, options: Mapping | None = None, spec=None,
                build_policy: BuildPolicy | None = None) -> EmitUnit:
    """Emit ONE kernel's source text and sidecar for ``targets`` without compiling or writing anything.

    The text and the sidecar are what :func:`build` publishes (same lane-set lowering, same policy), so a consumer that
    compiles the source itself gets the same translation unit and the same plane list. Honours ``build_policy``
    (:class:`hawk.build_policy.BuildPolicy`, ``launch_bounds`` included). Refuses what :func:`build` refuses, and a target
    that has no compiler (``c``)."""
    name = name or kernel.name
    native = _native(targets)
    if len(native) != len(targets):
        raise HawkError(f"emit_source covers the compiled targets only; {_exported(targets)} has no compiler: use build()")
    options = _user_options(options, literal_table, None, device_arch, "")
    unit, lowered, sources, _ = _emitted(kernel, name, mode, kind, targets, _options_mod.shared(options)[0],
                                         "emit_source returns", "build_bundle", spec, build_policy=build_policy, options=options)
    lead = _lead(dict.fromkeys(native))
    bounds = lowered.spec.settings.launch_bounds
    sidecar = _sidecar(unit, spec=lowered.spec, name=name, smode=lowered.smode, kind=lowered.kind, native=native, fmt=TARGETS[lead].format,
                       lead=lead, box=getattr(sources[lead], "box", None), block=None if bounds is None else bounds[0])
    return EmitUnit(sources, sidecar, unit)


def _expand_segmented(kernels: list, names: list, kinds: list) -> tuple:
    """Replace every segmented member of ``kernels``/``names``/``kinds`` with
    its K :class:`~hawk.ir.segment.SegmentUnit` s (and, for a static selector,
    the :class:`~hawk.ir.segment.ClassifyUnit`), in place; an ordinary
    member passes through untouched. ``_split_segmented`` itself refuses a
    kernel carrying more than one segmented dispatch node, so that refusal
    reaches the caller from here."""
    out_kernels, out_names, out_kinds = [], [], []
    for k, n, kd in zip(kernels, names, kinds):
        units = _split_segmented(n, k.sinks)
        if units is None:
            out_kernels.append(k)
            out_names.append(n)
            out_kinds.append(kd)
            continue
        for u in units:
            out_kernels.append(u)
            out_names.append(u.name)
            out_kinds.append(kd)
    return out_kernels, out_names, out_kinds


def _team_record(spec) -> dict:
    """The sidecar's ``team`` of a team stage. ``grid`` says what a block is: a lane (the launch's
    grid is the lane count) or a key of a block sum (the grid is the key count, ``keyed``
    names the keys as a keyed plane does, and the launch's ``nSamples`` is the lane count)."""
    out = {"threads": spec.threads}
    if spec.lanes > 1:
        out["lanes"] = spec.lanes
    if spec.warps > 1:
        out["warps"] = spec.warps
    if spec.grid == "keys":
        out["grid"] = "keys"
        if spec.groups > 1:
            out["groups"] = spec.groups
        out["keyed"] = _keys_record(spec.keyed)
    elif spec.grid in ("chunks", "targets"):    # an ordered scatter: lane chunks, then the cells of ``of``
        out["grid"], out["of"] = spec.grid, spec.target
        if spec.grid == "chunks":       # the plane whose flags say a key was out of range

            out["status"] = _team.scatter_names(spec.target)[3]
    return out


def _keys_record(keyed) -> dict:
    """The keys of a block sum as the sidecar spells a keyed plane: ``width`` entries over the kept
    axes (:func:`hawk._lane_bind.key_count` gives their number for a lane count)."""
    return {"width": keyed.width, "axes": [list(a) for a in keyed.kept], "payload": list(keyed.payload)}


def _team_scratch(found, cells: bool = False) -> dict:
    """The scratch planes the team stages of ``found`` work on, ``{name: {"dtype", "per_lane"}}``:
    ``dtype`` is ``"real"`` (the bundle's scalar type), ``"int32"`` or, for a partial sum of a ``cells``
    mode, ``"band_partial"`` (16 bytes); ``per_lane`` the elements of each lane's slice. The runtime makes each ``lanes x per_lane`` once per kernel and lane count."""
    out: dict = {}

    def dtype(p):
        return ("band_partial" if cells else "real") if p.dtype == _team.PARTIAL else p.dtype

    for st in found:
        out.update({p.name: {"dtype": dtype(p), "per_lane": p.per_lane, **({"layout": p.layout} if p.layout else {})}
                    for p in st.planes})
        if st.team is not None:
            out.update({p.name: {"dtype": dtype(p), "per_lane": p.per_lane,
                                 **({"keys": _keys_record(st.team.keyed)} if p.unit == "key" else
                                   {"per_target": st.team.target} if p.unit == "target" else {"once": True} if p.unit == "once" else {}),
                                 **({} if p.fill is None else {"fill": p.fill})}
                        for p in st.team.planes})
    return out


def _wide_with_class(record: dict, spec) -> dict | None:
    """``launch.wide``: the placement record and, beside the planes, the device class whose thresholds placed them."""
    return {**record, "class": spec.placement_class} if record else None


def _tile_rows(sinks, real_bytes: int, bodies: int, spec) -> int:
    """The lanes a block of the warp form serves: the most of 8, 4, 2 whose output tiles fit the card's TC bytes (a kernel
    with two bodies declares them twice), else 1 (a warp per block, stores straight to the planes). Only vector outputs are
    staged; a rank-2 one is not."""
    from ..dialects import is_a
    from ..targets import cards

    widths = [s.ttype.shape for s in sinks
              if is_a(s, "warp_store") and s.ttype.shape and s.ttype.shape[0] >= _components.device_view("lane").slice_min(s.ttype.dtype, spec)]
    if not widths or any(len(w) != 1 for w in widths):
        return 1
    for rows in (8, 4, 2):
        if bodies * sum(w[0] for w in widths) * (rows + 1) * real_bytes <= cards.wide(sinks[0].ttype.dtype, spec)["TC"]:
            return rows
    return 1


def _warp_kernel(kernel, name: str, spec, explicit: bool = True, real_bytes: int = 8, plain: bool = True):
    """``kernel`` in the warp form (:func:`simt.lane.slice_sinks`), or ``None`` when it is not one plain launch of wide
    Mutables the form has a rule for."""
    from ..targets import cards
    from .stages import WarpStage

    from ..ir import canonical
    from ..trace.kernel import Kernel, _declared

    one = getattr(kernel, "one_step", None)
    if (type(kernel).__name__ != "Kernel" or getattr(kernel, "group", 1) != 1 or _split.stages(name, kernel.sinks, spec=spec) is not None
            or (one is not None and _split.stages(name, one.sinks, spec=spec) is not None)):
        return None
    sinks = _components.device_view("lane").slice_sinks(_wide.simplified(kernel.sinks), spec, explicit=explicit)
    single = None if one is None else _components.device_view("lane").slice_sinks(_wide.simplified(one.sinks), spec, explicit=explicit)
    if sinks is None or (one is not None and single is None):
        return None
    rows = _tile_rows(sinks, real_bytes, 1 if one is None else 2, spec) if plain else 1
    warp = WarpStage(name, sinks, _team.TeamSpec(_components.device_view("lane").warp_of(spec) * rows, (), (), lanes=rows, warps=cards.team_warps(spec) if rows == 1 else 1))
    if one is not None:             # an automatic kernel: its single step beside it, in the warp form too
        warp.one_step = Kernel(one.name, single, one.planes, kind=one.kind)
    # the planes of the kernel as written stay its planes (a slot only a dropped value read is still bound), and so do its
    # single step's
    held = set(warp.walk.arg_spec)
    own = [(k.walk, key) for k in (kernel, warp.one_step) if k is not None for key in k.walk.arg_spec]
    extra = [Slot(role, slot, walk.slot_types[role, slot]) for walk, (role, slot) in own if (role, slot) not in held]
    if extra:
        warp.walk = canonical(warp.sinks, (*_declared(warp.planes), *extra))
    if one is not None:
        for attr in ("source_digest", "returns", "time", "time_step", "jump", "broadcast", "group"):
            setattr(warp, attr, getattr(kernel, attr))
    return warp


def _wide_ok(kernel, smode, device: bool):
    """Whether the placement pass may run on ``kernel`` (:func:`simt.split.stages` ``wide``): a build
    for the device, of a kernel that is one launch (a plain, grouped or segmented one, or an automatic one's two bodies)."""
    from hawk.passes.vectorise import interchanged_loops

    # a host-only build places too when its loops run interchanged: the wide carries then live in the lane rows
    return ((device or bool(interchanged_loops(kernel.walk))) and type(kernel).__name__ in ("Kernel", "SegmentUnit")
            and getattr(kernel, "fused_body", None) is None)


def _expand_split(kernels: list, names: list, kinds: list, *, spec, smode=None, device: bool = False,
                  device_only: bool = False) -> tuple:
    """Replace every member that reads a value reduced over samples with its
    stages (:func:`simt.split.stages`), named ``<name>``, ``<name>_2``, ...;
    the rest pass through untouched. Also returns, per first-stage name, the
    sidecar block naming every stage and every cell."""
    from ..diff import Derived
    from ..trace.kernel import Kernel
    from .stages import TeamStage
    from ..trace.kernel import _declared as _declared_slots

    out_kernels, out_names, out_kinds, blocks = [], [], [], {}
    for k, n, kd in zip(kernels, names, kinds):
        k = _plain_block_sums(k, spec)
        # a warp per lane when asked, or under ``auto`` when every target is a device (a host target of the same bundle
        # shares the kernel and has no warp, so it keeps the placed form) and the kernel has a rule in the warp form
        form = spec.settings.wide_form
        real = 4 if smode is not None and smode.id == "float32" else 8
        warp = _warp_kernel(k, n, spec, form == "team", real, _active_spec(kd, getattr(k, "group", 1)) is None) if device and (form == "team" or (form == "auto" and device_only)) else None
        if warp is not None:        # a warp per lane: one launch, no stages, no scratch
            out_kernels.append(warp)
            out_names.append(n)
            out_kinds.append(kd)
            blocks[n] = {"cells": warp.walk.cells, "stages": [
                {"kernel": n, "cells_in": [], "cells_out": [], "recomputed": [], "team": _team_record(warp.team)}]}
            continue
        ok, one = _wide_ok(k, smode, device), getattr(k, "one_step", None)
        real = 4 if smode is not None and smode.id == "float32" else 8
        if one is not None and (_split.stages(n, k.sinks, spec=spec) is not None or _split.stages(n, one.sinks, spec=spec) is not None):
            raise HawkError(
                f"{n!r} reads a value reduced over samples inside a stepping or "
                "segmented kernel; a split kernel is one plain step in this release")
        found = (_split.stages(n, k.sinks, spec=spec, wide=ok, real_bytes=real, inplace=_active_spec(kd, getattr(k, "group", 1)) is None)
                 or _chain_stages(k, n, smode, device, spec))
        found1 = None
        if one is not None and ok:      # the single step is a body of its own: placed beside the fused one, in planes of its own
            found1 = _split.stages(n, one.sinks, spec=spec, wide=True, real_bytes=real,
                                   taken={p.name for st in found or () for p in st.planes})
        if found is None and found1 is None:
            out_kernels.append(k)
            out_names.append(n)
            out_kinds.append(kd)
            continue
        fused = found
        found = found or found1
        if (
                getattr(k, "segment", None) is not None and _split.stages(n, k.sinks, spec=spec) is not None):
            raise HawkError(
                f"{n!r} reads a value reduced over samples inside a stepping or "
                "segmented kernel; a split kernel is one plain step in this release")
        if any(st.keyed for st in found) and getattr(k, "group", 1) != 1:
            raise HawkError(
                f"{n!r} reads hawk.sum/mean(x, over=...) in the body and declares "
                "group=: a split kernel runs every lane, so a compaction group is "
                "not built for it in this release")
        cells: dict = {}
        for stage in found:
            sinks = stage.sinks
            if isinstance(k.sinks, Derived):
                sinks = Derived(sinks, primal=k.sinks.primal, kind=k.sinks.kind,
                                wrt=k.sinks.wrt, primal_unit=k.sinks.primal_unit,
                                jump=k.sinks.jump)
                sinks._fused_steps = k.sinks._fused_steps
                for attr in ("lane_space", "group"):     # the stages launch over the derivative's lanes
                    if hasattr(k.sinks, attr):
                        setattr(sinks, attr, getattr(k.sinks, attr))
            if stage.team is not None and stage.team.steps:
                member = TeamStage(stage.name, sinks, stage.team, k.planes if stage is found[0] else None)
            elif getattr(k, "segment", None) is not None:       # a branch unit, its wide values placed: still the unit
                member = _segment_rewalk(k, sinks)
            else:
                member = Kernel(stage.name, sinks if fused or one is None else k.sinks, k.planes, kind=getattr(k, "kind", None))
                if len(found) == 1:         # the kernel itself, its wide values placed: it keeps what it declared
                    for attr in ("source_digest", "returns", "time", "time_step", "jump", "broadcast", "group"):
                        setattr(member, attr, getattr(k, attr))
                if one is not None:         # an automatic kernel: its single step beside it, both bodies reading the planes of both
                    member.one_step = Kernel(one.name, found1[0].sinks if found1 else one.sinks, one.planes, kind=one.kind)
                    held = set(member.walk.arg_spec)
                    extra = [Slot(role, name, member.one_step.walk.slot_types[role, name])
                             for role, name in member.one_step.walk.arg_spec if (role, name) not in held]
                    member.walk = _canonical(member.sinks, (*_declared_slots(member.planes), *extra))
                if stage.team is not None:      # a copy stage: the stage's own sinks, launched a block per lane
                    member.team = stage.team
            cells.update(member.walk.cells)
            out_kernels.append(member)
            out_names.append(stage.name)
            out_kinds.append(kd)
        # a cell only a later stage reads is the kernel's own
        for c in _split.internal(found, getattr(k, "planes", {})):
            cells[c] = {**cells[c], "internal": True}
        blocks[n] = {"cells": cells, "stages": [
            {"kernel": st.name, "cells_in": list(st.cells_in),
             "cells_out": list(st.cells_out), "recomputed": list(st.recomputed),
             **({} if st.team is None else {"team": _team_record(st.team)})}
            for st in found]}
        scratch = _team_scratch((*found, *(found1 or ())), bool(smode and smode.cells))
        if scratch:
            blocks[n]["team_planes"] = scratch
        # the keyed planes the runtime owns between the stages
        keyed = {kd.name: _keys_record(kd)
                 for st in found for kd in st.keyed}
        if keyed:
            blocks[n]["keyed"] = keyed
    return out_kernels, out_names, out_kinds, blocks


#: The lanes the chain rule weighs a cut at: the extent is not known when the kernel is built.
CHAIN_SAMPLES = 1 << 16


def _chain_stages(kernel, name: str, smode, device: bool, spec) -> tuple | None:
    """The stages of a chain (:func:`hawk.chain`) the fusion rule cuts (:func:`hawk.passes.fusion.cuts`, on the device's
    row when the bundle has a device target, else on this host's), named ``<name>``, ``<name>_2``, ...; ``None`` when
    it runs as one kernel. A value crossing a cut travels in a plane the runtime makes per lane."""
    from ..ir import chain as _chain
    from ..passes import fusion
    from ..trace.kernel import Kernel

    members = getattr(kernel, "members", None)
    if members is None or len(members) < 2 or getattr(kernel, "group", 1) != 1:
        return None                 # a compaction group runs as one kernel: the stages run every lane
    if not all(isinstance(m, Kernel) for m in members):
        return None                 # a fused lane group (hawk.emit.fused) lists lanes, not kernels: one kernel
    scalar = smode.id if smode is not None else "float64"
    cut = fusion.cuts(members, fusion.row(scalar, spec, host=not device), scalar, CHAIN_SAMPLES)
    if not cut:
        return None
    return tuple(_split.Stage(name if j == 0 else f"{name}_{j + 1}", sinks, (), (), (),
                              planes=tuple(_team.Plane(nm, dtype, per_lane, layout="plane_major") for nm, dtype, per_lane in planes))
                 for j, (sinks, planes) in enumerate(_chain.segments(members, cut)))


def _member_kind(kernel, kind, explicit: bool):
    """The kind one bundle member renders under: the bundle's, except that a
    member traced without a kind of its own skips the active-set index map
    from another member's guard unless ``kind=`` was given explicitly."""
    own = getattr(kernel, "kind", None) or DEFAULT_KIND
    guard = getattr(kind, "guard", None)
    if explicit or own is not DEFAULT_KIND or not _reads_map(kind):
        return kind
    return replace(kind, lanes=None, guard=replace(guard, active_set=False))


def _manifest_of(drafts) -> dict:
    """The ``manifest.json`` document over ``drafts``' native targets: one
    execution axis (every member must agree), one plugin entry per kernel
    naming its device object when it has one, else its host object."""
    # the stages of one split kernel run in sequence under the first stage's axis
    first = {st["kernel"]: d.sidecar for d in drafts for st in d.sidecar.get("stages", ())[1:]}
    axes = {(tuple(first.get(d.name, d.sidecar)["exec_targets"]),
             first.get(d.name, d.sidecar)["exec_access"],
             first.get(d.name, d.sidecar).get("exec_op")) for d in drafts}
    if len(axes) != 1:
        raise HawkError(
            "a bundle declares ONE execution axis at the manifest level, but "
            f"its members disagree: {sorted(axes)}. Split them into two bundles"
        )
    axis_targets, access, op = axes.pop()
    entries = [
        _manifest.plugin_entry(
            d.name, order,
            d.entries[_lead(d.entries)][1],
            d.sidecar_name,
            # The real format (``ptx``/``cubin``, read off the compiled
            # bytes), not a static "ptx" guess.
            d.entries[_lead(d.entries)][3])
        for order, d in enumerate(drafts)
    ]
    return _manifest.manifest_doc(entries, exec_targets=axis_targets,
                                  exec_access=access, exec_op=op)


def build_bundle(kernels, directory, *, mode: str = "float64",
                 targets=DEFAULT_TARGETS, kind=None, layout_sizes_override=None,
                 defines=(), cache_dir=None, device_arch: str = "",
                 name: str | None = None,
                 derivative: Mapping[str, str] | None = None,
                 host_profile: str = "", opt_level: str = "",
                 literal_table: bool = False, embedded_profile=None, options: Mapping | None = None,
                 emit_form: str | None = None, linalg_form: str | None = None, wide_form: str | None = None,
                 launch_bounds: tuple[int, int] | None = None, spec=None,
                 build_policy: BuildPolicy | None = None) -> Bundle:
    """Build every kernel into ``directory`` and write ONE ``manifest.json``
    over them. An ``emit_form``/``linalg_form``/``wide_form``/``launch_bounds`` left unset is the value of ``build_policy``
    (a :class:`hawk.build_policy.BuildPolicy`; the default: ``"auto"``, no bound); an explicit argument wins. ``emit_form`` is how a product, dot, sum or norm is spelled
    (:mod:`hawk.passes.policy`): ``"auto"`` unrolls a site of at most 16 elements and
    writes a longer one as a loop, ``"static"`` unrolls every site, ``"rolled"`` writes
    every site as a loop, the fast-compile mode (the compiler does no per-element work).
    ``linalg_form`` is how per-lane linear algebra is worked: ``"auto"`` keeps one thread per
    matrix up to ``n = 32`` and uses a team of threads per matrix above it, ``"team"`` uses a
    team at every size, ``"thread"`` refuses a matrix above 32 by name
    (:mod:`simt.team`). ``wide_form`` is where a per-lane value wider than a thread's
    registers lives: ``"auto"`` builds a kernel as a warp per lane when every target is a device
    and the kernel has a rule in that form (:mod:`simt.lane`, folds of 128 elements or more), else places
    the value (:mod:`simt.wide`) -- a bundle with a host target keeps the placed form, as the host has no
    warp; ``"thread"`` keeps every value in the lane's registers and local memory (the text before the pass);
    ``"team"`` asks for the warp form (device targets only). ``launch_bounds=(threads, blocks_per_sm)`` compiles every device kernel with
    ``__launch_bounds__(threads, blocks_per_sm)`` and records ``launch.block = threads`` in the sidecar, so a launch runs
    that block size (``threads`` a multiple of 32, ``threads * blocks_per_sm`` at most 2048; the values do not change).
    Every member must agree on the execution axis (a
    manifest-level, not per-entry, declaration) — a set that doesn't agree is
    two bundles.

    This is the DEPLOYMENT UNIT: publishing is content-addressed (see the
    module docstring) — a repeat build of an already-published definition
    recompiles nothing, copying files into a different directory if asked,
    so the returned :class:`Bundle` always names a directory that
    genuinely holds the unit.

    ``derivative`` names each derivative kernel's primal, per kernel:
    ``{kernel_name: primal_name}``, or leave it (``None``) to use a
    :func:`hawk.diff.vjp`/:func:`~hawk.diff.jvp` kernel's own recorded
    primal. A primal outside this bundle is named by
    ``{kernel_name: (primal_name, primal_unit_digest)}`` via its
    :attr:`Bundle.digest` — needed when the primal and its scattering VJP
    can't share one manifest.

    ``host_profile`` picks the host code-gen profile
    (:data:`hawk.compile.toolchain.HOST_PROFILES`): ``"native-vector-math"``
    (default on x86-64 with a GCC host compiler, and on arm64) or ``"native"`` (libm
    bit-identity) for a bundle built on the machine that runs it,
    ``"portable"`` for one that's prebuilt and shipped. Empty means
    ``$HAWK_HOST_PROFILE``, else
    :func:`hawk.compile.toolchain.default_host_profile`.

    ``opt_level`` picks the optimisation level
    (:data:`hawk.compile.toolchain.OPT_LEVELS`: ``"O0"``..``"O3"``), the
    same way for every target this bundle builds: ``-O<n>`` on the host
    build and on an AOT nvcc device build (nvcc's own ``-O<n>`` plus
    ``-Xptxas -O<n>``); NVRTC device compiles always optimise and ignore it.
    Empty means ``$HAWK_OPT_LEVEL``, else
    :data:`hawk.compile.toolchain.DEFAULT_OPT_LEVEL` (``"O3"``).

    ``literal_table`` (default ``False``) makes the device source read its
    ``double`` literals from one ``__constant__`` table per translation unit
    instead of as operand immediates: the same bits, fewer constant-move
    instructions on architectures that issue a move pair per literal (sm_90),
    unchanged on the others. Off, the device code is byte-identical to a
    build without the option. A ``float32`` build ignores it, and the host
    build is not affected.

    ``"c"`` in ``targets`` adds each kernel's standalone MISRA C unit
    (:mod:`hawk.embedded`: ``<name>.c``, ``<name>.h``,
    ``<name>.embedded.json``, audited and toolchain-checked) under
    ``embedded_profile`` (:class:`hawk.embedded.EmbeddedProfile` when
    ``None``). It never enters the manifest or a sidecar: those describe
    the native targets only, and a unit built for ``("c",)`` alone writes
    neither (``Bundle.manifest`` is ``None``)."""
    _bump_builds()  #: every CALL, attempt or success (one locked increment)
    options = _user_options(options, literal_table, embedded_profile, device_arch, host_profile)
    device_arch, host_profile = _options_mod.shared(options)
    kernels = list(kernels)
    smode = scalar_mode(mode)
    directory = Path(directory)
    targets = tuple(targets)
    names = [name or k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    _refuse_libc_names(kernels)
    _refuse_cells_off_device(names[0] if names else "", smode, targets)
    native = _native(targets)
    exported = _exported(targets)
    explicit = kind is not None
    kind = _resolve_bundle_kind(names, kernels, kind)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]

    # Replace each segmented member in place by its K units, so everything
    # downstream sees K ordinary-looking kernels.
    kernels, names, kinds = _expand_segmented(kernels, names, kinds)
    # A member reading a value reduced over samples becomes its stages, each
    # its own compile (its own cache key: its own source and options).
    # an argument left unset is ``build_policy``'s value (over the settings of a spec the caller passes); an explicit one wins
    given = _settings_of(build_policy, Settings() if spec is None else spec.settings)
    # ONE door: build_policy= (each legacy keyword still works alone; one that disagrees with the policy is refused by name)
    for key, value in (("emit_form", emit_form), ("linalg_form", linalg_form), ("wide_form", wide_form), ("launch_bounds", launch_bounds)):
        held = None if build_policy is None else getattr(build_policy, key)
        if value is not None and held is not None and _comparable(value) != _comparable(held):
            raise HawkError(f"{key}={value!r} disagrees with build_policy.{key}={held!r}: build settings have one door, "
                            "build_policy=; pass the setting once")
    emit_form = given.form if emit_form is None else emit_form
    linalg_form = given.linalg_form if linalg_form is None else linalg_form
    wide_form = given.wide_form if wide_form is None else wide_form
    launch_bounds = given.launch_bounds if launch_bounds is None else launch_bounds
    for key, value in (("emit_form", emit_form), ("linalg_form", linalg_form), ("wide_form", wide_form)):
        policy.refuse_choice(key, value)
    if wide_form == "team" and not _device_only(targets):
        raise HawkError("wide_form='team' (a warp per lane) is built for device targets only; use 'auto' with a host target")
    emit_arch = _emit_arch(targets, device_arch)
    spec = build_spec(tuple(targets), emit_arch) if spec is None else spec
    if launch_bounds is not None and spec.device is None:
        raise ValueError(f"launch_bounds={launch_bounds!r}: launch bounds are a device setting, refused on a build with no device target")
    _apply_policy(spec, build_policy, targets)            # refuses a key no target declares
    limits = None if spec.device is None else spec.device["launch"]      # what launch_bounds is checked against
    spec = spec.with_settings(form=emit_form, linalg_form=linalg_form, wide_form=wide_form, launch_bounds=launch_bounds,
                              contraction=given.contraction, fold_k=given.fold_k, scalar=smode.id, device_only=_device_only(targets),
                              route_to_ordered=smode.cells)
    spec = spec.with_settings(launch_bounds=policy.check(spec.settings, limits).launch_bounds)
    kernels, names, kinds, split = _expand_split(kernels, names, kinds, spec=spec, smode=smode,
                                                 device=_has_device(targets),
                                                 device_only=_device_only(targets))
    if launch_bounds is not None and any(getattr(k, "team", None) is not None for k in kernels):
        raise HawkError(
            f"launch_bounds={launch_bounds!r}: {[n for k, n in zip(kernels, names) if getattr(k, 'team', None) is not None][0]!r} "
            "is built as a warp per lane (a team), whose block the emit fixes; drop launch_bounds, or build it with wide_form='thread'")
    for t in exported:
        TARGETS[t].refuse_kernels(names, kernels)
    kernels = [_active_unit(k, n, kd) for k, n, kd in zip(kernels, names, kinds)]
    key_kind = kind if all(kd is kind for kd in kinds) else tuple(kinds)

    # Phase 1 — resolve `derivative` and hash the RESOLVED blocks into the
    # unit key before anything is emitted: a refusal here is validation.
    if derivative:      # the stage kernels of a split derivative name the primal their parent names
        derivative = {**{st["kernel"]: v for n, v in derivative.items()
                         for st in (split.get(n) or {}).get("stages", ())}, **derivative}
    per_kernel_derivative = _per_kernel_derivative(kernels, names, derivative)
    key = _unit_key(names, [k.walk.digest for k in kernels],
                    [declaration_digest(k) for k in kernels], targets=targets,
                    smode=smode, kind=key_kind, defines=defines, cache_dir=cache_dir,
                    device_arch=device_arch, derivative_blocks=per_kernel_derivative,
                    layout_sizes_override=layout_sizes_override,
                    host_profile=host_profile, opt_level=opt_level,
                    options=options, emit_form=emit_form, linalg_form=linalg_form,
                    wide_form=wide_form, wide_class=spec.wide_class, launch_bounds=launch_bounds, spec=spec, groups=tuple(getattr(k, "group", 1) for k in kernels),
                    source_digests=[getattr(k, "source_digest", None) for k in kernels])

    memo = _UNIT_MEMO.get(key)
    if memo is not None:
        # Still re-verified: compile valid by content, unit still the bytes
        # it was published as — either "no" falls through to a real publish.
        if not memo.closure.valid():
            _log(f"unit {memo.digest[:12]} invalidated: a header its compile "
                 "reached changed by content")
        elif (moved := _first_moved(memo.files)) is not None:
            _log(f"unit {memo.digest[:12]} invalidated: {moved} is no longer the "
                 "bytes the unit was published as")
        elif directory == memo.bundle.directory:
            _note("memo_hits", f"unit {memo.digest[:12]} served from the "
                               f"per-process memo at {memo.bundle.directory}")
            return memo.bundle
        else:
            # Same unit, different directory: no recompile, but the
            # caller's directory must really hold the bytes.
            try:
                bundle = _materialize(memo.bundle, memo.files, memo.digest, directory)
                _note("memo_hits", f"unit {memo.digest[:12]} served from the "
                                   f"per-process memo at {memo.bundle.directory}; "
                                   f"materialised at {directory}")
                return bundle
            except FileNotFoundError as gone:
                # The memo's directory vanished between the check and the
                # copy: fall through to a real publish.
                _log(f"unit {memo.digest[:12]} invalidated: {gone.filename} vanished "
                     "while being materialised")

    # Phase 2 — emit, compile and assemble the unit's content; still nothing
    # written. Each draft gets its own per_kernel_derivative[i] block.

    options = _options(device_arch=emit_arch, options=options, layout_sizes_override=layout_sizes_override, defines=defines, cache_dir=cache_dir, host_profile=host_profile,
                       opt_level=opt_level, block=None if launch_bounds is None else launch_bounds[0])

    def lowered_of(k, n, kd):
        lowered = lower(k, n, smode, kd, targets, spec)
        return lowered, _emit(lowered, targets, options)

    drafts = []
    for k, n, kd, d in zip(kernels, names, kinds, per_kernel_derivative):
        lowered, sources = lowered_of(k, n, kd)
        drafts.append(_draft(k, lowered, sources, options, targets=targets, derivative=d, split=split.get(n)))
    files = tuple(f for d in drafts for f in d.files)
    # The manifest describes the NATIVE targets only; a unit built for the
    # standalone C target alone has nothing for eagle to load and writes none.
    doc = _manifest_of(drafts) if native else None
    if doc is not None:
        manifest_blob = _blob(json.dumps(doc, indent=2) + "\n")
        files += (_File("manifest.json", _digest_bytes(manifest_blob),
                        blob=manifest_blob),)
    digest = _unit_digest(files)

    # Phase 3 — publish idempotently: a directory already holding this
    # exact unit is left alone.
    if _stamp_digest(directory) == digest and _first_moved(
            [(str(directory / f.name), f.digest) for f in files]) is None:
        _note("stamp_hits", f"unit {digest[:12]} already published at {directory}; "
                            "no file written")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        for f in files:
            f.write(directory)
        _write_stamp(directory, digest, files)
        _note("published", f"unit {digest[:12]} published at {directory}")

    bundle = Bundle(directory, None if doc is None else directory / "manifest.json",
                    doc, tuple(_artifact_of(d, directory) for d in drafts),
                    digest=digest)
    if len(_UNIT_MEMO) >= _UNIT_MEMO_LIMIT:
        _UNIT_MEMO.clear()
    _UNIT_MEMO[key] = _Unit(
        _served(bundle), digest,
        tuple((str(directory / f.name), f.digest) for f in files),
        _closure_watch(drafts), kind)
    return bundle
