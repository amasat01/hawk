# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""From a traced kernel to a deployment unit on disk.

One HAWK artifact directory is: one compiled object per declared device
target (a ``.ptx`` *or* ``.cubin``, whichever the compiled bytes actually
are), ONE shared library holding every host unit of the build call (with a
link ``<unit>.so`` to it for each unit, which the runtime still reads), the
emitted SOURCE of each, one JSON sidecar per kernel and one ``manifest.json``
per bundle. Every compile goes through
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
from aether.wings.base.wrapper import scalar_mode
from ..ext import DEFAULT_KIND
from .._errors import EMBEDDED_C, HawkError
from .. import wings as _wings
_wt = _wings.view("raptor", "base", "wrapper_text")
from .. import options as _options_mod
from ..build_settings import BuildSettings, apply as _apply_settings, check as _check_settings, default_settings, refuse_choice, settings_of as _settings_of
from ..backends import default_backends
from ..registry import KNOWN, check, denial, missing, traits
from ..targets import DEFAULT_TARGETS, TARGETS, build_spec
from . import manifest as _manifest
from .derivative import _per_kernel_derivative
from . import bridge as _bridge
from . import request as _request
from .sidecar import sidecar_of_unit
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
    """One kernel's published files + the sidecar it was described by. ``library`` is the shared
    library of the build call (every unit of the call is in it; ``None`` when the call made none);
    the ``entries`` of the host target name it, and those of a device target name the unit's image
    file, which the library also holds. ``links`` are the files named for this unit that resolve to
    the library (none when it is itself named for the unit, and none for a device-only unit)."""

    name: str
    directory: Path
    sidecar_path: Path
    sidecar: dict
    entries: dict
    links: tuple = ()
    library: Path | None = None


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
            out += a.links
            if a.library is not None:
                out.append(a.library)
            for entry in a.entries.values():
                out += [entry.artifact, entry.source]
                if entry.report is not None:
                    out.append(entry.report)
        return list(dict.fromkeys(out))


# -- The unit's CONTENT, computed before anything is written. -------------- #
@dataclass(frozen=True)
class UnitFile:
    """One file of a unit: its name, its bytes, and their digest.

    ``blob`` is carried inline (HAWK-made text) or fetched from the
    content-closure cache on demand (a compiled object, never held in
    memory). Either way the digest is of the bytes on disk, so the unit
    digest is re-computable from the published tree by anyone. A file with
    ``link_to`` is a symbolic link to the file of that name beside it; its
    digest is the one of the bytes it resolves to."""

    name: str
    digest: str
    blob: bytes | None = None
    source_path: Path | None = None
    link_to: str | None = None

    def write(self, directory: Path) -> Path:
        path = directory / self.name
        # Written whole to a private sibling, then renamed into place: a
        # concurrent reader (or a second builder of the same unit) sees the
        # old file or the new one, never a partial write.
        tmp = tmp_sibling(path)
        try:
            if self.link_to is not None:
                os.symlink(self.link_to, tmp)
            elif self.blob is not None:
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
    kernel: str                        # the name of the kernel the unit was written for
    meta: dict                         # the unit's record in the library's answer
    sidecar: dict
    sidecar_name: str
    entries: dict                      # target -> (result, art name, src name, fmt)
    files: tuple                       # the UnitFile records this kernel contributes
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


def text_blob(text: str) -> bytes:
    """The bytes a text file is published as, hashed as bytes so the digest
    reflects what's on disk, not an encoding assumption."""
    return text.encode("utf-8")


def digest_bytes(blob: bytes) -> str:
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


def _lane_set(kind):
    """The :class:`~hawk.ext.IndexSet` ``kind`` declares, else ``None``."""
    return getattr(kind, "lanes", None)


def _reads_map(kind) -> bool:
    """Whether ``kind`` reads an index map: a thinning guard or a lane set."""
    return bool(getattr(getattr(kind, "guard", None), "active_set", False)
                or _lane_set(kind) is not None)


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


def _refuse_libc_names(kernels, targets) -> None:
    """Refuse an authored kernel that takes a name a row of the build reserves, with the reason that row gives. The names are a
    machine fact of the build's rows (``reserved_names``: name -> reason); a chain or a derivative carries no source digest and
    is named by its builder."""
    from ..targets import detect_machine

    authored = [k for k in kernels if getattr(k, "source_digest", None) is not None]
    reserved = detect_machine("", tuple(targets)).reserved_names if authored else {}
    for k in authored:
        if k.name in reserved:
            raise HawkError(f"kernel {k.name!r}: {reserved[k.name]}; rename the kernel")


def _refuse_segmented_position(kernels, names) -> None:
    """Refuse a kernel with a segmented dispatch that reads the launch slot: the units of a segmented kernel address their lanes through
    the run's offsets."""
    for kernel, name in zip(kernels, names):
        if kernel.walk.reads_sample_position and any(d.strategy == "segmented" for d in kernel.walk.dispatches):
            raise HawkError(f"{name!r}: hawk.sample_position() is the launch slot, and a segmented unit addresses its lanes through the "
                            "run's offsets; use sample_index() there")


def _refuse_targets(name: str, targets) -> None:
    """Refuse an empty ``targets`` or an id that is no row of
    :data:`hawk.targets.TARGETS`, naming the valid ones."""
    check()
    if "c" in targets:
        raise HawkError(EMBEDDED_C)

    def hint() -> str:
        each = [f"targets=({t!r},)" for t in TARGETS]
        if not each:
            from .. import registry

            return f"no backend is installed; install one of {', '.join(sorted(set(registry.KNOWN.values())))}"
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


_SERVED_ID = {name: target for target, name in _request.SERVED.items()}


def _plans(kernels, names, kinds, *, smode, targets, spec, options, exports_block: bool = False) -> list:
    """``(request, dump, dialects)`` of each kernel: the request the lowering library is asked, the text of the nodes it names and the
    text of their dialects. The nodes are read here and nowhere else (:mod:`hawk.artifact.bridge`). A build asks for texts without the
    self-check block, which the text of its set carries once; a text that is compiled alone (:func:`emit_source`) asks for it."""
    out = []
    for kernel, name, kind in zip(kernels, names, kinds):
        dump, dialects, position = _bridge.dump(_request.sinks_of(kernel))
        out.append((_request.request(kernel, name=name, scalar=smode.id, kind=kind, targets=targets, spec=spec, position=position,
                                     options=options, exports_block=exports_block), dump, dialects))
    return out


def _options(*, device_arch: str, options=None, layout_sizes_override=None,
             defines=(), cache_dir=None, host_profile: str = "", opt_level: str = "", block: int | None = None) -> dict:
    """The build inputs backends read (:meth:`hawk.backends.Backend.emit`, ``build``): shared keys flat
    (``device_arch`` is the resolved one), the user's ``options`` under their ``<id>.`` prefix."""
    return {**(options or {}), "device_arch": device_arch, "layout_sizes_override": layout_sizes_override,
            "defines": tuple(defines), "cache_dir": cache_dir, "host_profile": host_profile or "",
            "opt_level": opt_level or "", "block": block}


def _emit_arch(targets, device_arch: str) -> str:
    """The device arch the emit may read cards for: resolved (once, memoised) only
    when a target compiles with the device toolchain, so a host-only build never
    probes."""
    return arch(device_arch) if _has_device(targets) else ""


def _prepared(unit, lowered, kernel, targets) -> tuple:
    """``(lowered, texts)`` of one answered unit: ``lowered`` with the unit's name and kernel, ``texts`` the text the library wrote
    for each target, by target."""
    name = unit["meta"]["name"]
    found = {_SERVED_ID[text["target"]]: text for text in unit["texts"]}
    if absent := [t for t in targets if t not in found]:
        raise HawkError(f"the lowering library wrote no {absent[0]} text for {name!r}")
    lowered = SimpleNamespace(kernel=kernel, name=name, smode=lowered.smode, kind=lowered.kind, spec=lowered.spec)
    return lowered, {t: SimpleNamespace(**text) for t, text in found.items()}


def _draft(kernel, unit, family, prepared, parts, options, *, targets, derivative) -> _Draft:
    """Assemble the unit CONTENT of one answered unit from the :class:`~hawk.backends.Part` each target built for it.

    Everything here is a pure function of the inputs plus the content-closure cache; nothing is written, which is what lets the
    publish step ask "is this already on disk?" before "where do I put it?". ``unit`` is the unit's record in the library's answer,
    ``family`` the records of every unit the answer holds for ``kernel``, ``prepared`` what :func:`_prepared` made of the unit.

    Each target's backend builds its own :class:`~hawk.backends.Part` from the text the library wrote for it
    (:meth:`hawk.backends.Backend.build`, started by :func:`_submit_targets` or run by :func:`_build_targets`); ``options["device_arch"]``
    is the arch the driver resolved once, and only when a target builds for a device, so a host-only build never pays for the
    probe. The sidecar is the unit's record and the kernel's own declarations (:func:`hawk.artifact.sidecar.sidecar_of_unit`)."""
    lowered = prepared[0]
    name = lowered.name
    entries, files = {}, []
    for target in targets:
        files += parts[target].files
        entries[target] = parts[target].entry
    lead = _lead(entries)
    # The real format: the detected device format when a device object was built, else the first compiled one: mirrors build_bundle's
    # manifest-entry choice so the sidecar and manifest never disagree.
    meta = sidecar_of_unit(kernel, unit, lead_device=_trait(lead, "device", False), fmt=entries[lead][3], scalar_type=lowered.smode.id,
                           exec_targets=[TARGETS[t].entry for t in targets], family=family, derivative=derivative, **parts[lead].facts)
    sidecar_blob = text_blob(json.dumps(meta, indent=2) + "\n")
    files.append(UnitFile(f"{name}.json", digest_bytes(sidecar_blob), blob=sidecar_blob))
    return _Draft(name, kernel.name, unit["meta"], meta, f"{name}.json", entries, tuple(files), {t: p.report_file for t, p in parts.items()})


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


def _submit_targets(pool, lowered, sources, options, targets) -> dict:
    """The ``build`` of every target in ``targets`` started in ``pool``: ``{target: future}``. One pool serves all the units of a
    build call, so the compiles of the call run at most :func:`compile_jobs` at once."""
    return {t: pool.submit(contextvars.copy_context().run, TARGETS[t].build, lowered, sources[t], options) for t in targets}


def _collect(futures: dict) -> dict:
    """The parts of :func:`_submit_targets`, by target, once every one has finished; the failure raised is the one of the first
    target in order."""
    for future in futures.values():
        future.exception()
    return {t: future.result() for t, future in futures.items()}


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


@dataclass(frozen=True)
class _Library:
    """The shared library of one build call: its file name, the host target whose units' objects it
    holds (``None``: a device-only build), its link in the cache and the files it publishes (the
    library, then one link to it for each host unit not named like it)."""

    name: str
    target: str | None
    result: object
    files: tuple


def _host_row(targets):
    """The backend whose compiler links the library of a build: the host row of the build, else the
    host row installed (a build for a device alone still assembles its images and links with the
    host compiler)."""
    row = next((TARGETS[t] for t in targets if TARGETS[t].host_compiled), None)
    if row is not None:
        return row
    for row in default_backends():
        if row.host_compiled:
            return row
    raise HawkError("a device build links its images into one library with the host compiler: "
                    "install the host OSPREY")


def _link_set(drafts, lowered, options, *, targets, stem: str | None) -> "_Library | None":
    """Link the objects of ``drafts`` into one library (:meth:`hawk.backends.Backend.link`): each
    unit's host object and the object that holds its device image, with the text of the set. The
    file is ``<stem>.<ext>`` (``stem`` is the bundle's name; none for a library named for its one
    unit); each host unit also gets a link ``<unit>.<ext>`` to it, which the runtime still opens by
    the unit's name. ``None`` when no target of the build is compiled."""
    if not _native(targets):
        return None
    target = next((t for t in targets if TARGETS[t].host_compiled), None)
    device = next((t for t in targets if TARGETS[t].device), None)
    row = _host_row(targets)
    units = [(None if target is None else d.entries[target][0],
              None if device is None else d.entries[device][0], d.meta) for d in drafts]
    recipe = None if device is None else TARGETS[device].embed_recipe(lowered, options)
    ext = row.artifact if target is None else Path(drafts[0].entries[target][1]).suffix
    name = (stem or drafts[0].name) + ext
    result = row.link(lowered, options, units, name, recipe)
    digest = digest_file(result.artifact)
    if digest is None:                                       # pragma: no cover
        raise HawkError(f"the content-closure cache says the library {name!r} is at {result.artifact} but it cannot be read")
    links = () if target is None else tuple(UnitFile(d.entries[target][1], digest, link_to=name)
                                            for d in drafts if d.entries[target][1] != name)
    return _Library(name, target, result, (UnitFile(name, digest, source_path=result.artifact), *links))


def _refuse_repeated_units(drafts_names) -> None:
    """Refuse two units of one name in one build call (a unit's entry, files and sidecar are named for it): ``drafts_names`` are
    ``(kernel, unit)`` pairs in order."""
    seen = {}
    for kernel, unit in drafts_names:
        if unit in seen:
            raise HawkError(f"the kernels {seen[unit]!r} and {kernel!r} both write a unit named {unit!r}: the units of one build "
                            "have unique names")
        seen[unit] = kernel


def _artifact_of(draft: _Draft, directory: Path, library: "_Library | None") -> Artifact:
    """The public :class:`Artifact` view of a drafted kernel published at
    ``directory``. Built the same way whether the files were just written or
    were already there — a served unit and a fresh one are the same object.
    A host entry names the build's shared library (``library``), and the
    unit's link to it is listed in ``links``."""
    entries = {
        target: Entry(target, directory / (library.name if library is not None and target == library.target else art_name),
                      directory / src_name, result.key, result.hit, result.seconds,
                      tuple(result.closure),
                      None if draft.report_files[target] is None else directory / draft.report_files[target])
        for target, (result, art_name, src_name, _fmt) in draft.entries.items()
    }
    link = None if library is None or library.target is None else draft.entries[library.target][1]
    links = () if link is None or link == library.name else (directory / link,)
    return Artifact(draft.name, directory, directory / draft.sidecar_name,
                    draft.sidecar, entries, links,
                    None if library is None else directory / library.name)


def _served(bundle: Bundle) -> Bundle:
    """The bundle a MEMO HIT hands back: identical, except every entry is
    marked ``hit`` with zero compile seconds — needed because the
    compile-floor rows assert ``not entry.hit`` to prove they time a
    compiler and not a lookup."""
    artifacts = tuple(
        replace(a, entries={t: replace(e, hit=True, seconds=0.0) for t, e in a.entries.items()})
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
                  for t, e in a.entries.items()}, tuple(directory / link.name for link in a.links),
                 _rehomed(a.library, directory))
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
    copied = tuple(UnitFile(n, d, link_to=os.readlink(p)) if Path(p).is_symlink() else UnitFile(n, d, source_path=Path(p))
                   for n, p, d in pairs)
    for f in copied:
        f.write(directory)
    _write_stamp(directory, digest, copied)
    return _rehome_bundle(bundle, directory)


def resolve_kind(name: str, kernel_kind, explicit_kind):
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
    """The same rule as :func:`resolve_kind`, for a whole bundle: every
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
          settings: BuildSettings | None = None) -> Artifact:
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
    lowered, answer, unit, options = _emitted(
        kernel, name, mode, kind, targets, device_arch, "build() publishes", "build_bundle", spec, settings=settings,
        layout_sizes_override=layout_sizes_override, options=options,
        defines=defines, cache_dir=cache_dir, host_profile=host_profile, opt_level=opt_level, exports_block=False)
    directory = Path(directory)
    prepared = _prepared(unit, lowered, kernel, targets)
    parts = _build_targets(prepared[0], prepared[1], options, targets)
    draft = _draft(kernel, unit, answer["units"], prepared, parts, options, targets=targets, derivative=derivative)
    # one kernel is a set of one: its library is named for its unit
    library = _link_set([draft], lowered, options, targets=targets, stem=None)
    directory.mkdir(parents=True, exist_ok=True)
    for f in (*draft.files, *(() if library is None else library.files)):
        f.write(directory)
    return _artifact_of(draft, directory, library)


def _user_options(options, literal_table, embedded_profile, device_arch, host_profile) -> dict:
    """The user's ``options`` with the old keywords folded in, checked against the manifests (:mod:`hawk.options`)."""
    if embedded_profile is not None or any(str(key).partition(".")[0] == "c" for key in options or ()):
        raise HawkError(EMBEDDED_C)
    merged = _options_mod.merge(options, literal_table=literal_table, device_arch=device_arch, host_profile=host_profile)
    _options_mod.check(merged)
    return merged


def _emitted(kernel, name, mode, kind, targets, device_arch, does, instead, spec, settings=None, *, exports_block: bool, **inputs) -> tuple:
    """``(lowered, answer, unit, options)`` of ONE kernel (``inputs`` are the :func:`_options` keywords but the arch): the lowering half of
    :func:`build` and :func:`emit_source`. ``lowered`` holds the scalar mode, the kind and the spec of the build, ``answer`` is the
    library's answer and ``unit`` the one unit it wrote a text for. A kernel that needs several launches (a segmented dispatch, a value
    reduced over samples) is refused: ``does`` says what the caller publishes and ``instead`` what to call."""
    _refuse_targets(name, targets)
    emit_arch = _emit_arch(targets, device_arch)
    spec = build_spec(tuple(targets), emit_arch) if spec is None else spec
    spec = _apply_settings(spec, settings, targets).with_settings(route_to_ordered=scalar_mode(mode).cells)
    if any(d.strategy == "segmented" for d in kernel.walk.dispatches):
        raise HawkError(f"{name!r} carries a segmented dispatch: {does} ONE artifact, and a segmented kernel needs K "
                        f"— call {instead}([kernel], ...) instead")
    kind = resolve_kind(name, getattr(kernel, "kind", None), kind)
    smode = scalar_mode(mode)
    _refuse_libc_names((kernel,), targets)
    _refuse_cells_off_device(name, smode, targets)
    options = _options(device_arch=emit_arch, **inputs)
    answer = _request.lower(*_plans([kernel], [name], [kind], smode=smode, targets=targets, spec=spec, options=options,
                                    exports_block=exports_block)[0])
    written = [u for u in answer["units"] if u["texts"]]
    if len(written) != 1:
        raise HawkError(f"{name!r} reads a value reduced over samples, so it runs as several launches: {does} ONE "
                        f"artifact — call {instead}([kernel], ...) instead")
    return SimpleNamespace(smode=smode, kind=kind, spec=spec), answer, written[0], options


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
                settings: BuildSettings | None = None) -> EmitUnit:
    """Emit ONE kernel's source text and sidecar for ``targets`` without compiling or writing anything.

    The sidecar is what :func:`build` publishes (same lane-set lowering, same settings). The device text is what a build
    publishes; the host text is the one a build publishes with the exports block in it (a build states that block once, in
    the text of its set), so it is a whole translation unit a consumer can compile alone, with the same plane list. Honours ``settings``
    (:class:`hawk.build_settings.BuildSettings`, ``launch_bounds`` included). Refuses what :func:`build` refuses, and a target
    that has no compiler (``c``)."""
    name = name or kernel.name
    _refuse_targets(name, targets)
    native = _native(targets)
    if len(native) != len(targets):
        raise HawkError(f"emit_source covers the compiled targets only; {_exported(targets)} has no compiler: use build()")
    options = _user_options(options, literal_table, None, device_arch, "")
    lowered, answer, unit, _ = _emitted(kernel, name, mode, kind, targets, _options_mod.shared(options)[0],
                                        "emit_source returns", "build_bundle", spec, settings=settings, options=options,
                                        exports_block=True)
    lead = _lead(dict.fromkeys(native))
    bounds = lowered.spec.settings.launch_bounds
    sidecar = sidecar_of_unit(kernel, unit, lead_device=_trait(lead, "device", False), fmt=TARGETS[lead].format, scalar_type=lowered.smode.id,
                              exec_targets=[TARGETS[t].entry for t in native], family=answer["units"], block=None if bounds is None else bounds[0])
    return EmitUnit({_SERVED_ID[text["target"]]: SimpleNamespace(**text) for text in unit["texts"]}, sidecar, kernel)


def _member_kind(kernel, kind, explicit: bool):
    """The kind one bundle member renders under: the bundle's, except that a
    member traced without a kind of its own skips the active-set index map
    from another member's guard unless ``kind=`` was given explicitly."""
    own = getattr(kernel, "kind", None) or DEFAULT_KIND
    guard = getattr(kind, "guard", None)
    if explicit or own is not DEFAULT_KIND or not _reads_map(kind):
        return kind
    return replace(kind, lanes=None, guard=replace(guard, active_set=False))


def _manifest_of(families, library: "_Library | None" = None) -> dict:
    """The ``manifest.json`` document over the units of each kernel
    (``families``: one list of drafts per kernel, in order), on their native
    targets: one execution axis (every kernel must agree), one plugin entry
    per unit naming its device object when it has one, else its host object (the
    link to the build's library, which an entry also names as its ``library``)."""
    drafts = [d for family in families for d in family]
    # the stages of one split kernel run in sequence under the first stage's axis
    first = {st["kernel"]: d.sidecar for d in drafts for st in d.sidecar.get("stages", ())[1:]}
    chain = _manifest.EXEC_ACCESS_CLASSES[:3]

    def axis(family):
        """One kernel's axis. Its own units may differ in access (a guarded branch unit commits across samples, the unit that
        sweeps its stale samples does not): the kernel states the strictest, which a placement check only ever refuses more for."""
        sides = [first.get(d.name, d.sidecar) for d in family]
        mine = {(tuple(side["exec_targets"]), side["exec_access"], side.get("exec_op")) for side in sides}
        if len(mine) > 1 and len({(t, o) for t, _, o in mine}) == 1 and all(a in chain for _, a, _ in mine):
            targets, _, op = next(iter(mine))
            mine = {(targets, max((a for _, a, _ in mine), key=chain.index), op)}
        return mine

    # Two KERNELS that disagree are refused: a bundle is launched as one.
    axes = set().union(*(axis(family) for family in families))
    if len(axes) != 1:
        raise HawkError(
            "a bundle declares ONE execution axis at the manifest level, but "
            f"its members disagree: {sorted(axes)}. Split them into two bundles"
        )
    axis_targets, access, op = next(iter(axes))
    entries = [
        _manifest.plugin_entry(
            d.name, order,
            d.entries[_lead(d.entries)][1],
            d.sidecar_name,
            # The real format (``ptx``/``cubin``, read off the compiled
            # bytes), not a static "ptx" guess.
            d.entries[_lead(d.entries)][3],
            None if library is None else library.name)
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
                 settings: BuildSettings | None = None) -> Bundle:
    """Build every kernel into ``directory`` and write ONE ``manifest.json``
    over them. An ``emit_form``/``linalg_form``/``wide_form``/``launch_bounds`` left unset is the value of ``settings``
    (a :class:`hawk.build_settings.BuildSettings`; the default: ``"auto"``, no bound); an explicit argument wins. ``emit_form`` is how a product, dot, sum or norm is spelled
    (:mod:`raptor.ir.fold`): ``"auto"`` unrolls a site of at most 16 elements and
    writes a longer one as a loop, ``"static"`` unrolls every site, ``"rolled"`` writes
    every site as a loop, the fast-compile mode (the compiler does no per-element work).
    ``linalg_form`` is how per-lane linear algebra is worked: ``"auto"`` keeps one thread per
    matrix up to ``n = 32`` and uses a team of threads per matrix above it, ``"team"`` uses a
    team at every size, ``"thread"`` refuses a matrix above 32 by name
    (:mod:`raptor.wings.simt.team`). ``wide_form`` is where a per-lane value wider than a thread's
    registers lives: ``"auto"`` builds a kernel as a warp per lane when every target is a device
    and the kernel has a rule in that form (:mod:`raptor.wings.simt.lane`, folds of 128 elements or more), else places
    the value (:mod:`raptor.wings.simt.wide`) -- a bundle with a host target keeps the placed form, as the host has no
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
    (:data:`ospreys.cpu.toolchain.HOST_PROFILES`): ``"native-vector-math"``
    (default on x86-64 with a GCC host compiler, and on arm64) or ``"native"`` (libm
    bit-identity) for a bundle built on the machine that runs it,
    ``"portable"`` for one that's prebuilt and shipped. Empty means
    ``$HAWK_HOST_PROFILE``, else
    :func:`ospreys.cpu.toolchain.default_host_profile`.

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

    Only the ``host`` and ``cuda`` targets are built: any other target, ``"c"`` included, is refused by name. ``embedded_profile`` is
    refused with it."""
    _bump_builds()  #: every CALL, attempt or success (one locked increment)
    options = _user_options(options, literal_table, embedded_profile, device_arch, host_profile)
    device_arch, host_profile = _options_mod.shared(options)
    kernels = list(kernels)
    smode = scalar_mode(mode)
    directory = Path(directory)
    targets = tuple(targets)
    names = [name or k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    _refuse_libc_names(kernels, targets)
    _refuse_segmented_position(kernels, names)
    _refuse_cells_off_device(names[0] if names else "", smode, targets)
    native = _native(targets)
    if native and not kernels:
        raise HawkError("no kernel to build: a bundle holds at least one")
    explicit = kind is not None
    kind = _resolve_bundle_kind(names, kernels, kind)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]

    # an argument left unset is ``settings``' value (over the settings of a spec the caller passes); an explicit one wins
    given = _settings_of(settings, default_settings() if spec is None else spec.settings)
    # ONE door: settings= (each legacy keyword still works alone; one that disagrees with it is refused by name)
    for key, value in (("emit_form", emit_form), ("linalg_form", linalg_form), ("wide_form", wide_form), ("launch_bounds", launch_bounds)):
        held = None if settings is None else getattr(settings, key)
        if value is not None and held is not None and _comparable(value) != _comparable(held):
            raise HawkError(f"{key}={value!r} disagrees with settings.{key}={held!r}: build settings have one door, "
                            "settings=; pass the setting once")
    emit_form = given.form if emit_form is None else emit_form
    linalg_form = given.linalg_form if linalg_form is None else linalg_form
    wide_form = given.wide_form if wide_form is None else wide_form
    launch_bounds = given.launch_bounds if launch_bounds is None else launch_bounds
    for key, value in (("emit_form", emit_form), ("linalg_form", linalg_form), ("wide_form", wide_form)):
        refuse_choice(key, value)
    if wide_form == "team" and not _device_only(targets):
        raise HawkError("wide_form='team' (a warp per lane) is built for device targets only; use 'auto' with a host target")
    emit_arch = _emit_arch(targets, device_arch)
    spec = build_spec(tuple(targets), emit_arch) if spec is None else spec
    if launch_bounds is not None and spec.device is None:
        raise ValueError(f"launch_bounds={launch_bounds!r}: launch bounds are a device setting, refused on a build with no device target")
    _apply_settings(spec, settings, targets)            # refuses a key no target declares
    limits = None if spec.device is None else spec.device["launch"]      # what launch_bounds is checked against
    spec = spec.with_settings(form=emit_form, linalg_form=linalg_form, wide_form=wide_form, launch_bounds=launch_bounds,
                              contraction=given.contraction, fold_k=given.fold_k, fast_entries=given.fast_entries, scalar=smode.id, device_only=_device_only(targets),
                              route_to_ordered=smode.cells)
    spec = spec.with_settings(launch_bounds=_check_settings(spec.settings, limits).launch_bounds)
    # Phase 1 — resolve `derivative` and ask for every kernel's request before anything is lowered; the unit key hashes the requests,
    # so a refusal here is validation.
    per_kernel_derivative = _per_kernel_derivative(kernels, names, derivative)
    inputs = _options(device_arch=emit_arch, options=options, layout_sizes_override=layout_sizes_override, defines=defines, cache_dir=cache_dir,
                      host_profile=host_profile, opt_level=opt_level, block=None if launch_bounds is None else launch_bounds[0])
    plans = _plans(kernels, names, kinds, smode=smode, targets=targets, spec=spec, options=inputs)
    key = _unit_key([json.dumps(request, sort_keys=True) + "\0" + dump + "\0" + dialects for request, dump, dialects in plans],
                    [declaration_digest(k) for k in kernels], targets=targets, defines=defines, cache_dir=cache_dir, device_arch=device_arch,
                    derivative_blocks=per_kernel_derivative, opt_level=opt_level, options=options,
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

    # Phase 2 — lower, compile and assemble the unit's content; still nothing
    # written. Each draft gets its own per_kernel_derivative[i] block. One IR context per kernel, made and dropped inside each lowering call, before any compile starts.
    answers = [_request.lower(*plan) for plan in plans]
    _refuse_repeated_units([(k.name, unit["meta"]["name"]) for k, answer in zip(kernels, answers) for unit in answer["units"] if unit["texts"]])
    if launch_bounds is not None:
        for answer in answers:
            teams = [st["unit"]["name"] for unit in answer["units"] for st in unit["meta"]["stages"] if st["team"] is not None]
            if teams:
                raise HawkError(f"launch_bounds={launch_bounds!r}: {teams[0]!r} is built as a warp per lane (a team), whose block the emit "
                                "fixes; drop launch_bounds, or build it with wide_form='thread'")
    # every target compile of every unit of the call runs in ONE pool (compile_jobs at most at once); the drafts are assembled in order
    with ThreadPoolExecutor(max_workers=compile_jobs(), thread_name_prefix="hawk-compile") as pool:
        try:
            started = [[(k, unit, answer["units"], d, _prepared(unit, SimpleNamespace(smode=smode, kind=kd, spec=spec), k, targets))
                        for unit in answer["units"] if unit["texts"]]
                       for k, kd, d, answer in zip(kernels, kinds, per_kernel_derivative, answers)]
            submitted = [[(*one, _submit_targets(pool, one[4][0], one[4][1], inputs, targets)) for one in family] for family in started]
            families = [[_draft(k, unit, family_units, prepared, _collect(futures), inputs, targets=targets, derivative=d)
                         for k, unit, family_units, d, prepared, futures in family] for family in submitted]
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    drafts = [d for family in families for d in family]
    library = _link_set(drafts, SimpleNamespace(smode=smode, kind=kinds[0], spec=spec), inputs,
                        targets=targets, stem=name or "units")
    files = tuple(f for d in drafts for f in d.files) + (() if library is None else library.files)
    doc = _manifest_of(families, library)
    manifest_blob = text_blob(json.dumps(doc, indent=2) + "\n")
    files += (UnitFile("manifest.json", digest_bytes(manifest_blob), blob=manifest_blob),)
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

    bundle = Bundle(directory, directory / "manifest.json",
                    doc, tuple(_artifact_of(d, directory, library) for d in drafts),
                    digest=digest)
    if len(_UNIT_MEMO) >= _UNIT_MEMO_LIMIT:
        _UNIT_MEMO.clear()
    _UNIT_MEMO[key] = _Unit(
        _served(bundle), digest,
        tuple((str(directory / f.name), f.digest) for f in files),
        _closure_watch(drafts), kind)
    return bundle
