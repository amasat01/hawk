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

import hashlib
import json
import re
import warnings
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..compile import CompileOptions, compile_source, digest_file
from ..compile.cache import ClosureWatch, publish_atomic, tmp_sibling
from ..emit import policy
from ..ir.cost import working_set_bytes
from ..trace.kernel import derived_axes as _derived_axes
from ..emit import SegmentSpec, literal_table as _literal_table, render_source, scalar_mode
from ..emit.box import thread_box
from ..emit.render import binding_name as _binding_name
from ..emit.wrapper import ActiveSpec
from ..ext import DEFAULT_KIND
from ..ir import DispatchInfo, HawkError
from ..ir.nodes import LEAF_ROLES
from ..ir import canonical as _canonical
from ..ir import split_segmented as _split_segmented
from ..ir.segment import rewalk as _segment_rewalk
from ..ir.split import internal as _split_internal
from ..ir import wide as _wide
from ..ir.split import stages as _split_stages
from ..targets import DEFAULT_TARGETS, ENTRY_DEVICE, ENTRY_HOST, GPU, TARGETS
from ..targets.cards import build_arch
from ..types import Slot, TensorType
from . import layout as _layout
from . import manifest as _manifest
from .derivative import _per_kernel_derivative
from .sidecar import LOCAL_LIMIT_B, sidecar_meta
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

#: The published extension for each real device format
#: :func:`_sniff_device_format` can return (NVRTC/nvcc here never emit a
#: ``fatbin``, the third ``eagle.roles.MANIFEST_FORMATS`` member).
_DEVICE_EXT = {"ptx": ".ptx", "cubin": ".cubin"}

def _sniff_device_format(path: Path) -> str:
    r"""``"cubin"`` if the compiled device object at ``path`` is an ELF
    CUBIN, ``"ptx"`` otherwise — the artifact's own bytes decide, never the
    target recorded at compile time, since the compiler picks PTX or CUBIN
    from the box's own toolchain. A CUBIN opens with the ELF magic
    (``\\x7fELF``); anything else is read as PTX without inspecting further."""
    with open(path, "rb") as f:
        magic = f.read(4)
    return "cubin" if magic == b"\x7fELF" else "ptx"

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


def _native(targets) -> tuple:
    """The targets this module compiles and the manifest describes: every
    one whose row has a compiler. The rest are exported units
    (:mod:`hawk.embedded`)."""
    return tuple(t for t in targets if TARGETS[t].compile is not None)


def _exported(targets) -> tuple:
    """The targets whose row is written whole and never compiled."""
    return tuple(t for t in targets if TARGETS[t].compile is None)


def _lead(entries) -> str:
    """The compiled entry a manifest plugin names: the device object when the
    unit has one, else the first compiled."""
    native = _native(entries)
    return next((t for t in native if TARGETS[t].entry == ENTRY_DEVICE), native[0])


def _refuse_targets(name: str, targets) -> None:
    """Refuse an empty ``targets`` or an id that is no row of
    :data:`hawk.targets.TARGETS`, naming the valid ones."""
    each = [f"targets=({t!r},)" for t in TARGETS]
    hint = f"pass {', '.join(each[:-1])} or {each[-1]}"
    if not targets:
        raise HawkError(f"{name!r}: no target to build; {hint}")
    for target in targets:
        if target not in TARGETS:
            ids = [repr(t) for t in TARGETS]
            about = ", ".join(f"{t!r} {row.about}" for t, row in TARGETS.items())
            raise HawkError(
                f"unknown target {target!r}: the valid targets are "
                f"{', '.join(ids[:-1])} and {ids[-1]} ({about}); {hint}")


def _sources(kernel, *, name, smode, kind, targets, layout_sizes_override,
             literal_table: bool = False, embedded_profile=None) -> dict:
    """Every declared target's emitted TU. No compiler, no disk, no sidecar
    — the cheap half of a build, and what the unit key is computed from, so
    an emitter change is a key change even when the walk is unmoved.

    A kernel with its own ``fused_body`` hook (a composed lane group)
    renders it ONCE and hands it to both wrappers, so the two targets
    provably wrap the same string.

    ``literal_table`` rewrites the device TU's ``double`` literals into one
    ``__constant__`` table (:func:`hawk.emit.literal_table`).

    A row without a wrapper is its exporter's render, under
    ``embedded_profile`` (:class:`~hawk.embedded.EmbeddedProfile` when
    ``None``): for ``c``, :func:`hawk.embedded.render`'s
    :class:`~hawk.embedded.EmbeddedSource`."""
    _refuse_targets(name, targets)
    hook = getattr(kernel, "fused_body", None) if _native(targets) else None
    body = None if hook is None else hook(kind)
    segment = _segment_spec(kernel)
    if segment is not None and kernel.walk.reads_lane_position:
        raise HawkError(f"{name!r}: hawk.lane_position() is the launch slot, and a segmented "
                        "unit addresses its lanes through the run's offsets; use "
                        "sample_index() there")
    active = _active_spec(kind, getattr(kernel, "group", 1))
    one_step = getattr(kernel, "one_step", None)
    box = thread_box(kernel, smode.id, kind=kind, segment=segment, active=active,
                     one_step=one_step, body=body)
    out = {}
    for target in targets:
        row = TARGETS[target]
        if row.wrapper is None:
            export = row.export
            out[target] = export.render(
                kernel, name=name, mode=smode.id, kind=kind,
                profile=(embedded_profile if embedded_profile is not None
                         else export.EmbeddedProfile()))
            continue
        with policy.placing(row.compile == GPU):        # the device text keeps wide values in their planes
            out[target] = render_source(
                name, kernel.sinks, kernel.walk, row, mode=smode, kind=kind,
                body=body, segment=segment, one_step=one_step, box=box,
                active=None if getattr(getattr(kernel, "team", None), "grid", None) in ("keys", "chunks", "targets") else active,
                team=None if getattr(kernel, "team", None) is None else kernel.team.threads,
                exports=_layout.exports(target, layout_sizes_override=layout_sizes_override))
        if literal_table and row.compile == GPU:
            out[target] = _literal_table(out[target])
    return out




def _step_ops(kernel, kind) -> int | None:
    """The operation count of one step of a kernel that finishes its own
    samples by stepping (its body cuts into head, one step and tail), else
    ``None``. A fact for the launching runtime: the emitted code does not
    read it."""
    from ..emit.render import render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return None
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kind)
    return None if split is None else int(split.op_count)


def _fp64_ops_ir(kernel, kind) -> int | None:
    """The 64-bit floating-point operations one step of a stepping kernel
    runs, counted in the IR (as :func:`_step_ops`, ``None`` for any other
    kernel): the step cost the launching runtime's chunk length reads."""
    from ..emit.render import render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return None
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kind)
    return None if split is None else int(split.fp64_ops)


def _entry_counts_finished(kernel, kind) -> bool:
    """Whether the kernel's fast entries count a sample finished on entry into
    the finish counter (:func:`hawk.emit.cuda.entry_counts_finished`), so the
    launching runtime need not count the mask before a run."""
    from ..emit.render import render_lane_split
    from ..emit.cuda import entry_counts_finished

    if getattr(kernel.walk, "finish", None) is None:
        return False
    return entry_counts_finished(kernel.walk, render_lane_split(kernel.sinks, kernel.walk, kind=kind))


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


def _export_part(row, source, profile) -> tuple:
    """An exported target's share of a draft: the audit and the toolchain
    checks (any finding refuses, before anything is written), then the report
    :func:`hawk.embedded.export` writes, built by the same code. Returns
    ``(result, report, files)``."""
    export = row.export
    export.refuse_audit_hits(source)
    checks = export.check(source, profile=profile)
    report = export.finish_report(source, checks)
    name = source.name
    h = hashlib.sha256()
    for text in (source.text, source.header):
        h.update(_blob(text))
        h.update(b"\0")
    result = _ExportResult(h.hexdigest(), False, float(checks.get("seconds", 0.0)))
    files = []
    for fname, text in ((f"{name}{row.source}", source.text),
                        (f"{name}{row.artifact}", source.header),
                        (f"{name}.embedded.json", export.report_text(report))):
        blob = _blob(text)
        files.append(_File(fname, _digest_bytes(blob), blob=blob))
    return result, report, files


def _emit_arch(targets, device_arch: str) -> str:
    """The device arch the emit may read cards for: resolved (once, memoised) only
    when a target compiles with the device toolchain, so a host-only build never
    probes."""
    gpu = any(TARGETS[t].compile == GPU for t in targets if t in TARGETS)
    return arch(device_arch) if gpu else ""


#: A frame this many times the declared working set is reported as a spill.
_FRAME_SLACK = 1.25


def _resource_report(kernel, ptxas: dict | None) -> dict | None:
    """The sidecar's ptxas figures with the kernel's expected working set beside them
    (:func:`hawk.ir.cost.working_set_bytes`) and a ``spills`` flag: registers spilled, or a
    local frame past :data:`_FRAME_SLACK` times that working set. A flagged kernel warns."""
    if not ptxas:
        return ptxas
    working = working_set_bytes(kernel.sinks)
    report = {}
    for entry, row in ptxas.items():
        spills = (row["spill_store_B"] or 0) > 0 or (working > 0 and row["stack_B"] > _FRAME_SLACK * working)
        report[entry] = {**row, "working_set_B": working, "spills": spills}
        if spills:
            warnings.warn(f"hawk: kernel {entry!r} uses {row['regs']} registers and a {row['stack_B']}-byte "
                          f"local frame ({'spills unknown' if row['spill_store_B'] is None else str(row['spill_store_B']) + ' bytes spilled'}) against an expected working set "
                          f"of {working} bytes", stacklevel=2)
    return report


_PTX_DEPOT = re.compile(r"^\s*\.local\s+[^\n;]*__local_depot\d+\[(\d+)\]", re.M)
_PTX_BODY = re.compile(r"^\.(visible\s+)?\.(entry|func)\b", re.M)


def _local_memory(name: str, entries: dict, lead: str) -> tuple | None:
    """``(bytes per thread, source)`` of the device unit, refused by name past the device limit.
    ``source`` is ``"ptxas"`` (the compiler's stack frame of a CUBIN, the real number) or ``"ptx"``
    (the local depots the PTX declares: ptxas can only shrink them, so it is a ceiling). ``None`` for a
    unit with no device object."""
    if TARGETS[lead].entry != ENTRY_DEVICE:
        return None
    result, _, _, fmt = entries[lead]
    if result.ptxas:
        held, source = max(row["stack_B"] for row in result.ptxas.values()), "ptxas"
    elif fmt == "ptx":
        text = Path(result.artifact).read_text()
        cuts = [(m.start(), m.group(2)) for m in _PTX_BODY.finditer(text)] + [(len(text), "")]
        frames = [(kind, sum(int(d.group(1)) for d in _PTX_DEPOT.finditer(text[at:end])))
                  for (at, kind), (end, _) in zip(cuts, cuts[1:])]
        # an entry runs alone; the device functions it may call add to it
        held = (max((b for k, b in frames if k == "entry"), default=0)
                + sum(b for k, b in frames if k == "func"))
        source = "ptx"
    else:
        return None
    if held > LOCAL_LIMIT_B:
        raise HawkError(
            f"kernel {name!r}: {held:,} bytes of local memory per thread ({source}), past the device limit "
            f"of {LOCAL_LIMIT_B:,}; the launch would fail. Hold fewer live factorisations or matrices per "
            "lane, build with linalg_form='team' so a team of threads works the matrix in scratch, or use "
            "a smaller n")
    return held, source


def _draft(kernel, sources, *, name, smode, kind, defines, cache_dir, device_arch,
           targets, derivative, host_profile: str = "", opt_level: str = "",
           embedded_profile=None, split: dict | None = None) -> _Draft:
    """Compile every target and assemble the unit CONTENT for one kernel.

    Everything here is a pure function of the inputs plus the
    content-closure cache; nothing is written, which is what lets the
    publish step ask "is this already on disk?" before "where do I put
    it?".

    The device arch is resolved ONCE here, not per target, and only when a
    target compiles with the device toolchain: the resolver may reach the
    device probe (:func:`hawk.artifact.arch`), so a host-only build must
    never pay for it.

    An exported target is not compiled: its source is audited and checked
    (:func:`_export_part`) and contributes its unit, its header and
    ``<name>.embedded.json``. The JSON sidecar describes the NATIVE targets
    only and is written only when there is one; an export-only draft carries
    the report in the sidecar's place and writes no ``<name>.json``."""
    native = _native(targets)
    entries, files = {}, []
    resolved_arch = (arch(device_arch)
                     if any(TARGETS[t].compile == GPU for t in native) else "")
    options = {target: kind.apply(CompileOptions(
        target=target, mode=smode.id, arch=resolved_arch,
        defines=tuple(defines), cache_dir=cache_dir,
        host_profile=host_profile or "", opt_level=opt_level or ""))
        for target in native}
    results = _compile_targets(sources, name, options) if native else {}
    for target in native:
        row = TARGETS[target]
        art_ext, src_ext, fmt = row.artifact, row.source, row.format
        source = sources[target]
        result = results[target]
        if row.compile == GPU:
            # The real format, read off the compiled bytes, not the row's
            # static ".ptx" default — the same build can land PTX
            # or CUBIN depending on the box's compiler/driver versions.
            fmt = _sniff_device_format(result.artifact)
            art_ext = _DEVICE_EXT[fmt]
        art_name, src_name = f"{name}{art_ext}", f"{name}{src_ext}"
        compiled = digest_file(result.artifact)
        if compiled is None:                                 # pragma: no cover
            raise HawkError(
                f"the content-closure cache says {name!r}'s {target} object is at "
                f"{result.artifact} but it cannot be read; the unit cannot be "
                "published from a store that lost its own artifact")
        files.append(_File(art_name, compiled, source_path=result.artifact))
        blob = _blob(source.text)
        files.append(_File(src_name, _digest_bytes(blob), blob=blob))
        entries[target] = (result, art_name, src_name, fmt)

    report = None
    for target in _exported(targets):
        row = TARGETS[target]
        result, report, part = _export_part(
            row, sources[target], embedded_profile if embedded_profile is not None
            else row.export.EmbeddedProfile())
        files += part
        entries[target] = (result, f"{name}{row.artifact}", f"{name}{row.source}",
                           row.format)
    if not native:
        return _Draft(name, report, f"{name}.embedded.json", entries, tuple(files))

    segment = getattr(kernel, "segment", None)
    dispatch_override = None if segment is None else (
        DispatchInfo(segment.dispatch_name, segment.K, "segmented", segment.units,
                     selector_reads=segment.selector_reads,
                     static_selector=segment.static_selector),
    )
    lead = _lead(entries)
    meta = sidecar_meta(
        name, kernel.walk,
        # The real format: the detected device format when a device object
        # was built, else the first compiled one — mirrors build_bundle's
        # manifest-entry choice so the sidecar and manifest never disagree.
        fmt=entries[lead][3],
        scalar_type=smode.id,
        exec_targets=[TARGETS[t].entry for t in native],
        host_entry=(f"{name}_host"
                    if any(TARGETS[t].entry == ENTRY_HOST for t in native) else None),
        derivative=derivative,
        lanes=getattr(kernel, "lanes", None),
        lane_space=getattr(kernel, "lane_space", ((None, None),)),
        group=getattr(kernel, "group", 1),
        box=getattr(sources[lead], "box", None),
        index_set=(None if _lane_set(kind) is None else
                   {"width": 32, "prefix": True, "count": ACTIVE_COUNT} if _lane_set(kind).prefix
                   else {"width": _map_bits(kind)}),
        planes=getattr(kernel, "planes", None) or _derived_axes(
            kernel.walk, getattr(kernel.sinks, "source", None)),
        broadcast=getattr(kernel, "broadcast", None),
        dispatch_override=dispatch_override,
        step_ops=_step_ops(kernel, kind),
        fp64_ops_ir=_fp64_ops_ir(kernel, kind),
        entry_counts_finished=_entry_counts_finished(kernel, kind),
        ptxas=(_resource_report(kernel, entries[lead][0].ptxas)
               if TARGETS[lead].entry == ENTRY_DEVICE else None),
        local=_local_memory(name, entries, lead),
        wide=_wide_with_class({**_wide.walk_record(kernel.walk), **_wide.walk_record(kernel.one_step.walk)} if getattr(kernel, "one_step", None) is not None
                              else _wide.walk_record(kernel.walk), resolved_arch),
    )
    if split is not None:
        # the first stage of a split kernel names every stage and every cell
        meta.update(split)
    sidecar_blob = _blob(json.dumps(meta, indent=2) + "\n")
    files.append(_File(f"{name}.json", _digest_bytes(sidecar_blob),
                       blob=sidecar_blob))
    return _Draft(name, meta, f"{name}.json", entries, tuple(files))


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


def _compile_targets(sources, name, options) -> dict:
    """``compile_source`` of every target in ``options``, side by side when
    there is more than one (:func:`compile_jobs` at most at once): the host
    compiler and the device compiler are separate processes on separate
    cache slots, so neither waits for the other. Results are keyed by
    target and consumed in ``targets`` order, so the unit is the same
    whichever finishes first; a failure is raised for the FIRST failing
    target in that order — the error a one-after-the-other build reports —
    after every started compile has finished."""
    targets = list(options)
    jobs = min(len(targets), compile_jobs())
    if jobs <= 1:
        return {t: compile_source(sources[t].text, name, options[t]) for t in targets}
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="hawk-compile") as pool:
        futures = {t: pool.submit(compile_source, sources[t].text, name, options[t])
                   for t in targets}
        for future in futures.values():
            future.exception()  # wait for every compile before reporting any
    return {t: futures[t].result() for t in targets}


def _artifact_of(draft: _Draft, directory: Path) -> Artifact:
    """The public :class:`Artifact` view of a drafted kernel published at
    ``directory``. Built the same way whether the files were just written or
    were already there — a served unit and a fresh one are the same object."""
    entries = {
        target: Entry(target, directory / art_name, directory / src_name,
                      result.key, result.hit, result.seconds,
                      tuple(result.closure),
                      directory / f"{draft.name}.embedded.json"
                      if TARGETS[target].compile is None else None)
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
          embedded_profile=None) -> Artifact:
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
    if any(d.policy == "segmented" for d in kernel.walk.dispatches):
        raise HawkError(
            f"{name!r} carries a segmented dispatch: build() publishes "
            "ONE artifact, and a segmented kernel needs K — call "
            f"build_bundle([kernel], ...) instead"
        )
    if _split_stages(name, kernel.sinks) is not None:
        raise HawkError(
            f"{name!r} reads a value reduced over samples, so it runs as several "
            "launches: build() publishes ONE artifact — call "
            f"build_bundle([kernel], ...) instead")
    kind = _resolve_kind(name, getattr(kernel, "kind", None), kind)
    kernel = _active_unit(kernel, name, kind)
    smode = scalar_mode(mode)
    directory = Path(directory)
    with build_arch(_emit_arch(targets, device_arch)):
        sources = _sources(kernel, name=name, smode=smode, kind=kind, targets=targets,
                           layout_sizes_override=layout_sizes_override,
                           literal_table=literal_table, embedded_profile=embedded_profile)
    draft = _draft(kernel, sources, name=name, smode=smode, kind=kind,
                   defines=defines, cache_dir=cache_dir, device_arch=device_arch,
                   targets=targets, derivative=derivative,
                   host_profile=host_profile, opt_level=opt_level,
                   embedded_profile=embedded_profile)
    directory.mkdir(parents=True, exist_ok=True)
    for f in draft.files:
        f.write(directory)
    return _artifact_of(draft, directory)


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
    if spec.grid == "keys":
        out["grid"] = "keys"
        out["keyed"] = _keys_record(spec.keyed)
    elif spec.grid in ("chunks", "targets"):    # an ordered scatter: lane chunks, then the cells of ``of``
        out["grid"], out["of"] = spec.grid, spec.target
        if spec.grid == "chunks":       # the plane whose flags say a key was out of range
            from ..ir.team import scatter_names

            out["status"] = scatter_names(spec.target)[3]
    return out


def _keys_record(keyed) -> dict:
    """The keys of a block sum as the sidecar spells a keyed plane: ``width`` entries over the kept
    axes (:func:`hawk._lane_bind.key_count` gives their number for a lane count)."""
    return {"width": keyed.width, "axes": [list(a) for a in keyed.kept], "payload": list(keyed.payload)}


def _team_scratch(found) -> dict:
    """The scratch planes the team stages of ``found`` work on, ``{name: {"dtype", "per_lane"}}``:
    ``dtype`` is ``"real"`` (the bundle's scalar type) or ``"int32"``, ``per_lane`` the elements of
    each lane's slice. The runtime makes each ``lanes x per_lane`` once per kernel and lane count."""
    out: dict = {}
    for st in found:
        out.update({p.name: {"dtype": p.dtype, "per_lane": p.per_lane} for p in st.planes})
        if st.team is not None:
            out.update({p.name: {"dtype": p.dtype, "per_lane": p.per_lane,
                                 **({"keys": _keys_record(st.team.keyed)} if p.unit == "key" else
                                   {"per_target": st.team.target} if p.unit == "target" else {"once": True} if p.unit == "once" else {}),
                                 **({} if p.fill is None else {"fill": p.fill})}
                        for p in st.team.planes})
    return out


def _wide_with_class(record: dict, resolved_arch: str) -> dict | None:
    """``launch.wide``: the placement record and, beside the planes, the device class whose thresholds placed them."""
    from ..ir import wide_class

    return {**record, "class": wide_class.select(resolved_arch)} if record else None


def _warp_kernel(kernel, name: str):
    """``kernel`` in the warp form (:func:`hawk.ir.lane.slice_sinks`), or ``None`` when it is not one plain launch of wide
    Mutables the form has a rule for."""
    from ..ir.lane import WARP, slice_sinks
    from ..ir.team import TeamSpec
    from ..trace.kernel import WarpStage

    if (type(kernel).__name__ != "Kernel" or getattr(kernel, "one_step", None) is not None
            or getattr(kernel, "group", 1) != 1 or _split_stages(name, kernel.sinks) is not None):
        return None
    sinks = slice_sinks(_wide.simplified(kernel.sinks))
    if sinks is None:
        return None
    return WarpStage(name, sinks, TeamSpec(WARP, (), ()))


def _wide_ok(kernel, smode, device: bool):
    """Whether the placement pass may run on ``kernel`` (:func:`hawk.ir.split.stages` ``wide``): a build
    for the device, of a kernel that is one launch (a plain, grouped or segmented one, or an automatic one's two bodies)."""
    from ..ir.vectorise import interchanged_loops

    # a host-only build places too when its loops run interchanged: the wide carries then live in the lane rows
    return ((device or bool(interchanged_loops(kernel.walk))) and type(kernel).__name__ in ("Kernel", "SegmentUnit")
            and getattr(kernel, "fused_body", None) is None)


def _expand_split(kernels: list, names: list, kinds: list, *, smode=None, device: bool = False,
                  device_only: bool = False) -> tuple:
    """Replace every member that reads a value reduced over samples with its
    stages (:func:`hawk.ir.split.stages`), named ``<name>``, ``<name>_2``, ...;
    the rest pass through untouched. Also returns, per first-stage name, the
    sidecar block naming every stage and every cell."""
    from ..diff import Derived
    from ..trace.kernel import Kernel, TeamStage
    from ..trace.kernel import _declared as _declared_slots

    out_kernels, out_names, out_kinds, blocks = [], [], [], {}
    for k, n, kd in zip(kernels, names, kinds):
        # a warp per lane when asked, or under ``auto`` when every target is a device (a host target of the same bundle
        # shares the kernel and has no warp, so it keeps the placed form) and the kernel has a rule in the warp form
        form = policy.wide_form()
        warp = _warp_kernel(k, n) if device and (form == "team" or (form == "auto" and device_only)) else None
        if warp is not None:        # a warp per lane: one launch, no stages, no scratch
            out_kernels.append(warp)
            out_names.append(n)
            out_kinds.append(kd)
            blocks[n] = {"cells": warp.walk.cells, "stages": [
                {"kernel": n, "cells_in": [], "cells_out": [], "recomputed": [], "team": _team_record(warp.team)}]}
            continue
        ok, one = _wide_ok(k, smode, device), getattr(k, "one_step", None)
        real = 4 if smode is not None and smode.id == "float32" else 8
        if one is not None and (_split_stages(n, k.sinks) is not None or _split_stages(n, one.sinks) is not None):
            raise HawkError(
                f"{n!r} reads a value reduced over samples inside a stepping or "
                "segmented kernel; a split kernel is one plain step in this release")
        found = _split_stages(n, k.sinks, wide=ok, real_bytes=real)
        found1 = None
        if one is not None and ok:      # the single step is a body of its own: placed beside the fused one, in planes of its own
            found1 = _split_stages(n, one.sinks, wide=True, real_bytes=real,
                                   taken={p.name for st in found or () for p in st.planes})
        if found is None and found1 is None:
            out_kernels.append(k)
            out_names.append(n)
            out_kinds.append(kd)
            continue
        fused = found
        found = found or found1
        if (
                getattr(k, "segment", None) is not None and _split_stages(n, k.sinks) is not None):
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
                for attr in ("lane_space", "group"):     # the stages launch over the derivative's lanes
                    if hasattr(k.sinks, attr):
                        setattr(sinks, attr, getattr(k.sinks, attr))
            if stage.team is not None:
                member = TeamStage(stage.name, sinks, stage.team)
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
            cells.update(member.walk.cells)
            out_kernels.append(member)
            out_names.append(stage.name)
            out_kinds.append(kd)
        # a cell only a later stage reads is the kernel's own
        for c in _split_internal(found, getattr(k, "planes", {})):
            cells[c] = {**cells[c], "internal": True}
        blocks[n] = {"cells": cells, "stages": [
            {"kernel": st.name, "cells_in": list(st.cells_in),
             "cells_out": list(st.cells_out), "recomputed": list(st.recomputed),
             **({} if st.team is None else {"team": _team_record(st.team)})}
            for st in found]}
        scratch = _team_scratch((*found, *(found1 or ())))
        if scratch:
            blocks[n]["team_planes"] = scratch
        # the keyed planes the runtime owns between the stages
        keyed = {kd.name: _keys_record(kd)
                 for st in found for kd in st.keyed}
        if keyed:
            blocks[n]["keyed"] = keyed
    return out_kernels, out_names, out_kinds, blocks


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
                 literal_table: bool = False, embedded_profile=None,
                 emit_form: str = "auto", linalg_form: str = "auto", wide_form: str = "auto") -> Bundle:
    """Build every kernel into ``directory`` and write ONE ``manifest.json``
    over them. ``emit_form`` is how a product, dot, sum or norm is spelled
    (:mod:`hawk.emit.policy`): ``"auto"`` unrolls a site of at most 16 elements and
    writes a longer one as a loop, ``"static"`` unrolls every site, ``"rolled"`` writes
    every site as a loop, the fast-compile mode (the compiler does no per-element work).
    ``linalg_form`` is how per-lane linear algebra is worked: ``"auto"`` keeps one thread per
    matrix up to ``n = 32`` and uses a team of threads per matrix above it, ``"team"`` uses a
    team at every size, ``"thread"`` refuses a matrix above 32 by name
    (:mod:`hawk.ir.team`). ``wide_form`` is where a per-lane value wider than a thread's
    registers lives: ``"auto"`` builds a kernel as a warp per lane when every target is a device
    and the kernel has a rule in that form (:mod:`hawk.ir.lane`, folds of 128 elements or more), else places
    the value (:mod:`hawk.ir.wide`) -- a bundle with a host target keeps the placed form, as the host has no
    warp; ``"thread"`` keeps every value in the lane's registers and local memory (the text before the pass);
    ``"team"`` asks for the warp form (device targets only). Every member must agree on the execution axis (a
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
    (default on x86-64 with a GCC host compiler) or ``"native"`` (libm
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
    kernels = list(kernels)
    smode = scalar_mode(mode)
    directory = Path(directory)
    targets = tuple(targets)
    names = [name or k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    native = _native(targets)
    exported = _exported(targets)
    if exported and embedded_profile is None:
        embedded_profile = TARGETS[exported[0]].export.EmbeddedProfile()
    explicit = kind is not None
    kind = _resolve_bundle_kind(names, kernels, kind)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]

    # Replace each segmented member in place by its K units, so everything
    # downstream sees K ordinary-looking kernels.
    kernels, names, kinds = _expand_segmented(kernels, names, kinds)
    # A member reading a value reduced over samples becomes its stages, each
    # its own compile (its own cache key: its own source and options).
    if emit_form not in policy.FORMS:
        raise HawkError(f"emit_form must be one of {policy.FORMS}, got {emit_form!r}")
    if linalg_form not in policy.LINALG_FORMS:
        raise HawkError(f"linalg_form must be one of {policy.LINALG_FORMS}, got {linalg_form!r}")
    if wide_form not in policy.WIDE_FORMS:
        raise HawkError(f"wide_form must be one of {policy.WIDE_FORMS}, got {wide_form!r}")
    if wide_form == "team" and any(TARGETS[t].compile != GPU for t in targets if t in TARGETS):
        raise HawkError("wide_form='team' (a warp per lane) is built for device targets only; use 'auto' with a host target")
    from ..ir import wide_class

    emit_arch = _emit_arch(targets, device_arch)
    cls = wide_class.select(emit_arch) if emit_arch else ""       # the thresholds of the nearest measured device class
    with policy.using(emit_form, linalg_form=linalg_form, wide_form=wide_form), policy.wide_class_scope(cls):
        kernels, names, kinds, split = _expand_split(kernels, names, kinds, smode=smode,
                                                     device=any(TARGETS[t].compile == GPU for t in targets if t in TARGETS),
                                                     device_only=all(TARGETS[t].compile == GPU for t in targets if t in TARGETS))
    if exported and any(getattr(k, "team", None) is not None for k in kernels):
        raise HawkError(
            f"{names[0]!r} has linear algebra worked by a team of threads (a matrix above 32, or "
            "linalg_form='team'): the standalone C target has no team form, and its helpers hold "
            "n <= 32; build the C unit from a kernel whose matrices fit one lane")
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
                    literal_table=literal_table, emit_form=emit_form, linalg_form=linalg_form,
                    wide_form=wide_form, wide_class=cls, groups=tuple(getattr(k, "group", 1) for k in kernels),
                    embedded_profile=repr(embedded_profile) if exported else "",
                    source_digests=([getattr(k, "source_digest", None) for k in kernels]
                                    if exported else ()))

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

    def sources_of(k, n, kd):
        with build_arch(emit_arch), policy.using(emit_form, linalg_form=linalg_form, wide_form=wide_form):
            return _sources(k, name=n, smode=smode, kind=kd, targets=targets,
                            layout_sizes_override=layout_sizes_override,
                            literal_table=literal_table, embedded_profile=embedded_profile)

    drafts = [_draft(k, sources_of(k, n, kd),
                     name=n, smode=smode, kind=kd, defines=defines,
                     cache_dir=cache_dir, device_arch=device_arch, targets=targets,
                     derivative=d, host_profile=host_profile, opt_level=opt_level,
                     embedded_profile=embedded_profile, split=split.get(n))
              for k, n, kd, d in zip(kernels, names, kinds, per_kernel_derivative)]
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
