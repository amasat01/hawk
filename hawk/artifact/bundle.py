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
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..compile import CompileOptions, compile_source, digest_file
from ..compile.cache import ClosureWatch, publish_atomic, tmp_sibling
from ..emit import BACKENDS, SegmentSpec, render_source, scalar_mode
from ..emit.aether import binding_name as _binding_name
from ..emit.backend import ActiveSpec
from ..ext import DEFAULT_KIND
from ..ir import DispatchInfo, HawkError
from ..ir import canonical as _canonical
from ..ir import split_segmented as _split_segmented
from ..types import Slot, TensorType
from . import layout as _layout
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
    arch,
)
from .unit_cache import COUNTERS as COUNTERS
from .unit_cache import bump_builds as _bump_builds
from .unit_cache import STAMP_NAME as STAMP_NAME
from .unit_cache import counters_snapshot as counters_snapshot
from .unit_cache import publish_log as publish_log
from .unit_cache import reset_unit_memo as reset_unit_memo
from .unit_cache import unit_stats as unit_stats

#: Which compiled container each backend produces (the SOURCE extension and
#: the FALLBACK manifest ``format``, used as-is for ``host`` and overridden
#: for ``cuda`` by :func:`_sniff_device_format` below, since the ``cuda``
#: leg can compile to PTX *or* CUBIN depending on the box's toolchain).
_TARGET = {"cuda": (".ptx", ".cu", "ptx"), "host": (".so", ".cpp", "host")}

#: The published extension for each real device format
#: :func:`_sniff_device_format` can return (NVRTC/nvcc here never emit a
#: ``fatbin``, the third ``eagle.roles.MANIFEST_FORMATS`` member).
_DEVICE_EXT = {"ptx": ".ptx", "cubin": ".cubin"}

#: HAWK's backend id -> the manifest's ``exec_targets`` spelling.
_EXEC_TARGET = {"cuda": "device", "host": "host"}


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
    """One compiled target of one kernel."""

    backend: str
    artifact: Path
    source: Path
    key: str
    hit: bool
    seconds: float
    closure: tuple = field(default=())


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
    primal's home (:func:`_per_kernel_derivative`'s cross-unit reference)."""

    directory: Path
    manifest_path: Path
    manifest: dict
    artifacts: tuple
    digest: str | None = None

    def files(self) -> list:
        """EVERY file this unit published — the set row greps."""
        out = [self.manifest_path]
        for a in self.artifacts:
            out.append(a.sidecar_path)
            for entry in a.entries.values():
                out += [entry.artifact, entry.source]
        return out


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
    entries: dict                      # target -> (CompileResult, art name, src name)
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
    """Build the :class:`~hawk.emit.backend.SegmentSpec` a segmented unit's
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
_ACTIVE_TTYPE = TensorType((), "i32")


class _ActiveUnit:
    """A kernel seen through its ACTIVE-SET walk: the kernel's own walk plus
    the two declared ``lookup`` planes the map prologue reads; everything
    else falls through to the kernel's own attributes."""

    def __init__(self, kernel, walk) -> None:
        self._kernel = kernel
        self.walk = walk

    def __getattr__(self, attr):
        return getattr(self._kernel, attr)


def _active_unit(kernel, name: str, kind):
    """``kernel`` as an :class:`_ActiveUnit` when ``kind``'s guard asks for
    an active-set index map, else ``kernel`` itself. Refuses a segmented
    unit, a kernel accumulating across samples (the map would reorder the
    sum), one binding none of the guard's masks, and one already using a
    reserved plane name."""
    if not getattr(getattr(kind, "guard", None), "active_set", False):
        return kernel
    walk = kernel.walk
    if (getattr(kernel, "segment", None) is not None
            or any(d.policy == "segmented" for d in walk.dispatches)):
        raise HawkError(
            f"{name!r}: Guard(active_set=True) cannot read an index map in a "
            "segmented unit — the segmented prologue already addresses the run"
        )
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
    taken = [n for _, n in walk.arg_spec if n in (ACTIVE_MAP, ACTIVE_COUNT)]
    if taken:
        raise HawkError(
            f"{name!r}: {taken[0]!r} is reserved for the active-set index map "
            "under Guard(active_set=True); rename that parameter"
        )
    declared = tuple(Slot(role, n, walk.slot_types[(role, n)])
                     for role, n in walk.arg_spec)
    declared += (Slot("lookup", ACTIVE_MAP, _ACTIVE_TTYPE),
                 Slot("lookup", ACTIVE_COUNT, _ACTIVE_TTYPE))
    return _ActiveUnit(kernel, _canonical(kernel.sinks, declared=declared))


def _active_spec(kind) -> ActiveSpec | None:
    """The map prologue's bindings when ``kind`` asks for one, else ``None``."""
    if not getattr(getattr(kind, "guard", None), "active_set", False):
        return None
    return ActiveSpec(_binding_name("lookup", ACTIVE_MAP),
                      _binding_name("lookup", ACTIVE_COUNT))


def _sources(kernel, *, name, smode, kind, targets, layout_sizes_override) -> dict:
    """Every declared target's emitted TU. No compiler, no disk, no sidecar
    — the cheap half of a build, and what the unit key is computed from, so
    an emitter change is a key change even when the walk is unmoved.

    A kernel with its own ``fused_body`` hook (a composed lane group)
    renders it ONCE and hands it to both backends, so the two targets
    provably wrap the same string."""
    hook = getattr(kernel, "fused_body", None)
    body = None if hook is None else hook(kind)
    segment = _segment_spec(kernel)
    active = _active_spec(kind)
    out = {}
    for target in targets:
        if target not in _TARGET:
            raise HawkError(f"unknown target {target!r}; built are {tuple(_TARGET)}")
        out[target] = render_source(
            name, kernel.sinks, kernel.walk, BACKENDS[target], mode=smode, kind=kind,
            body=body, segment=segment, active=active,
            one_step=getattr(kernel, "one_step", None),
            exports=_layout.exports(target,
                                    layout_sizes_override=layout_sizes_override))
        if active is not None and out[target].reduce_op is not None:
            raise HawkError(
                f"{name!r}: Guard(active_set=True) refuses a reducing kernel "
                f"({out[target].reduce_op}): the map would change which samples "
                "each partial folds, so the result would not be the map-free one "
                "bit for bit"
            )
    return out


def _step_ops(kernel, kind) -> int | None:
    """The operation count of one step of a kernel that finishes its own
    samples by stepping (its body cuts into head, one step and tail), else
    ``None``. A fact for the launching runtime: the emitted code does not
    read it."""
    from ..emit.aether import render_lane_split

    if getattr(kernel.walk, "finish", None) is None:
        return None
    split = render_lane_split(kernel.sinks, kernel.walk, kind=kind)
    return None if split is None else int(split.op_count)


def _entry_counts_finished(kernel, kind) -> bool:
    """Whether the kernel's fast entries count a sample finished on entry into
    the finish counter (:func:`hawk.emit.cuda.entry_counts_finished`), so the
    launching runtime need not count the mask before a run."""
    from ..emit.aether import render_lane_split
    from ..emit.cuda import entry_counts_finished

    if getattr(kernel.walk, "finish", None) is None:
        return False
    return entry_counts_finished(kernel.walk, render_lane_split(kernel.sinks, kernel.walk, kind=kind))


def _draft(kernel, sources, *, name, smode, kind, defines, cache_dir, device_arch,
           targets, derivative, host_profile: str = "", opt_level: str = "") -> _Draft:
    """Compile every target and assemble the unit CONTENT for one kernel.

    Everything here is a pure function of the inputs plus the
    content-closure cache; nothing is written, which is what lets the
    publish step ask "is this already on disk?" before "where do I put
    it?".

    The device arch is resolved ONCE here, not per target, and only when
    ``"cuda"`` is actually one of ``targets``: the resolver may reach the
    device probe (:func:`hawk.artifact.arch`), so a host-only build must
    never pay for it."""
    entries, files = {}, []
    resolved_arch = arch(device_arch) if "cuda" in targets else ""
    options = {target: kind.apply(CompileOptions(
        backend=target, mode=smode.id, arch=resolved_arch,
        defines=tuple(defines), cache_dir=cache_dir,
        host_profile=host_profile or "", opt_level=opt_level or ""))
        for target in targets}
    results = _compile_targets(sources, name, options)
    for target in targets:
        art_ext, src_ext, fmt = _TARGET[target]
        source = sources[target]
        result = results[target]
        if target == "cuda":
            # The real format, read off the compiled bytes, not the static
            # ".ptx" :data:`_TARGET` default — the same build can land PTX
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

    segment = getattr(kernel, "segment", None)
    dispatch_override = None if segment is None else (
        DispatchInfo(segment.dispatch_name, segment.K, "segmented", segment.units),
    )
    meta = sidecar_meta(
        name, kernel.walk,
        # The real format: the detected ``cuda`` format when a device object
        # was built, else ``host`` — mirrors build_bundle's manifest-entry
        # choice so the sidecar and manifest never disagree.
        fmt=entries["cuda"][3] if "cuda" in entries else entries["host"][3],
        scalar_type=smode.id,
        exec_targets=[_EXEC_TARGET[t] for t in targets],
        host_entry=f"{name}_host" if "host" in targets else None,
        derivative=derivative,
        lanes=getattr(kernel, "lanes", None),
        dispatch_override=dispatch_override,
        step_ops=_step_ops(kernel, kind),
        entry_counts_finished=_entry_counts_finished(kernel, kind),
    )
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
                      tuple(result.closure))
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
                 {t: Entry(e.backend, e.artifact, e.source, e.key, True, 0.0,
                           e.closure)
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
                 {t: Entry(e.backend, directory / e.artifact.name,
                           directory / e.source.name, e.key, e.hit, e.seconds,
                           e.closure)
                  for t, e in a.entries.items()})
        for a in bundle.artifacts
    )
    return Bundle(directory, directory / bundle.manifest_path.name, bundle.manifest,
                 artifacts, digest=bundle.digest)


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
def build(kernel, directory, *, mode: str = "float64", targets=("cuda", "host"),
          kind=None, layout_sizes_override=None, defines=(), cache_dir=None,
          device_arch: str = "", name: str | None = None,
          derivative: dict | None = None, host_profile: str = "",
          opt_level: str = "") -> Artifact:
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

    Refuses a ``segmented`` dispatch, which needs K artifacts: call
    :func:`build_bundle` with ``[kernel]`` instead."""
    name = name or kernel.name
    if any(d.policy == "segmented" for d in kernel.walk.dispatches):
        raise HawkError(
            f"{name!r} carries a segmented dispatch: build() publishes "
            "ONE artifact, and a segmented kernel needs K — call "
            f"build_bundle([kernel], ...) instead"
        )
    kind = _resolve_kind(name, getattr(kernel, "kind", None), kind)
    kernel = _active_unit(kernel, name, kind)
    smode = scalar_mode(mode)
    directory = Path(directory)
    sources = _sources(kernel, name=name, smode=smode, kind=kind, targets=targets,
                       layout_sizes_override=layout_sizes_override)
    draft = _draft(kernel, sources, name=name, smode=smode, kind=kind,
                   defines=defines, cache_dir=cache_dir, device_arch=device_arch,
                   targets=targets, derivative=derivative,
                   host_profile=host_profile, opt_level=opt_level)
    directory.mkdir(parents=True, exist_ok=True)
    for f in draft.files:
        f.write(directory)
    return _artifact_of(draft, directory)


def _expand_segmented(kernels: list, names: list, kinds: list) -> tuple:
    """Replace every segmented member of ``kernels``/``names``/``kinds`` with
    its K :class:`~hawk.ir.segment.SegmentUnit` s, in place; an ordinary
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


def _member_kind(kernel, kind, explicit: bool):
    """The kind one bundle member renders under: the bundle's, except that a
    member traced without a kind of its own skips the active-set index map
    from another member's guard unless ``kind=`` was given explicitly."""
    own = getattr(kernel, "kind", None) or DEFAULT_KIND
    guard = getattr(kind, "guard", None)
    if explicit or own is not DEFAULT_KIND or not getattr(guard, "active_set", False):
        return kind
    return replace(kind, guard=replace(guard, active_set=False))


def build_bundle(kernels, directory, *, mode: str = "float64",
                 targets=("cuda", "host"), kind=None, layout_sizes_override=None,
                 defines=(), cache_dir=None, device_arch: str = "",
                 name: str | None = None,
                 derivative: Mapping[str, str] | None = None,
                 host_profile: str = "", opt_level: str = "") -> Bundle:
    """Build every kernel into ``directory`` and write ONE ``manifest.json``
    over them. Every member must agree on the execution axis (a
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
    :data:`hawk.compile.toolchain.DEFAULT_OPT_LEVEL` (``"O3"``)."""
    _bump_builds()  #: every CALL, attempt or success (one locked increment)
    kernels = list(kernels)
    smode = scalar_mode(mode)
    directory = Path(directory)
    targets = tuple(targets)
    names = [name or k.name for k in kernels]
    explicit = kind is not None
    kind = _resolve_bundle_kind(names, kernels, kind)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]

    # Replace each segmented member in place by its K units, so everything
    # downstream sees K ordinary-looking kernels.
    kernels, names, kinds = _expand_segmented(kernels, names, kinds)
    kernels = [_active_unit(k, n, kd) for k, n, kd in zip(kernels, names, kinds)]
    key_kind = kind if all(kd is kind for kd in kinds) else tuple(kinds)

    # Phase 1 — resolve `derivative` and hash the RESOLVED blocks into the
    # unit key before anything is emitted: a refusal here is validation.
    per_kernel_derivative = _per_kernel_derivative(kernels, names, derivative)
    key = _unit_key(names, [k.walk.digest for k in kernels], targets=targets,
                    smode=smode, kind=key_kind, defines=defines, cache_dir=cache_dir,
                    device_arch=device_arch, derivative_blocks=per_kernel_derivative,
                    layout_sizes_override=layout_sizes_override,
                    host_profile=host_profile, opt_level=opt_level)

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
    drafts = [_draft(k, _sources(k, name=n, smode=smode, kind=kd, targets=targets,
                                 layout_sizes_override=layout_sizes_override),
                     name=n, smode=smode, kind=kd, defines=defines,
                     cache_dir=cache_dir, device_arch=device_arch, targets=targets,
                     derivative=d, host_profile=host_profile, opt_level=opt_level)
              for k, n, kd, d in zip(kernels, names, kinds, per_kernel_derivative)]
    axes = {(tuple(d.sidecar["exec_targets"]), d.sidecar["exec_access"],
             d.sidecar.get("exec_op")) for d in drafts}
    if len(axes) != 1:
        raise HawkError(
            "a bundle declares ONE execution axis at the manifest level, but "
            f"its members disagree: {sorted(axes)}. Split them into two bundles"
        )
    axis_targets, access, op = axes.pop()
    entries = [
        _manifest.plugin_entry(
            d.name, order,
            d.entries["cuda"][1] if "cuda" in d.entries else d.entries["host"][1],
            d.sidecar_name,
            # The real format (``ptx``/``cubin``, read off the compiled
            # bytes), not a static "ptx" guess.
            d.entries["cuda"][3] if "cuda" in d.entries else "host")
        for order, d in enumerate(drafts)
    ]
    doc = _manifest.manifest_doc(entries, exec_targets=axis_targets,
                                 exec_access=access, exec_op=op)
    manifest_blob = _blob(json.dumps(doc, indent=2) + "\n")
    files = tuple(f for d in drafts for f in d.files) + (
        _File("manifest.json", _digest_bytes(manifest_blob), blob=manifest_blob),)
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

    bundle = Bundle(directory, directory / "manifest.json", doc,
                    tuple(_artifact_of(d, directory) for d in drafts), digest=digest)
    if len(_UNIT_MEMO) >= _UNIT_MEMO_LIMIT:
        _UNIT_MEMO.clear()
    _UNIT_MEMO[key] = _Unit(
        _served(bundle), digest,
        tuple((str(directory / f.name), f.digest) for f in files),
        _closure_watch(drafts), kind)
    return bundle


