# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""From a traced kernel to a deployment unit on disk.

One HAWK artifact directory is: one compiled object per declared device target, ONE shared library holding every host unit of
the build call (with a link ``<unit>.so`` to it for each unit, which the runtime still reads), the emitted SOURCE of each, one
JSON sidecar per kernel and one ``manifest.json`` per bundle. Nothing here launches anything: eagle loads and runs the device
object and the host object; HAWK's job ends at the bytes on disk.

hawk does not compile. A build asks each OSPREY that serves a target for its machine, sends the kernel-side request to the
OSPREYs' lowering, asks each for the build of every unit, asks the host OSPREY to link the set, and publishes what came back with
a sidecar. The OSPREYs keep the compilers, the flags, the headers and the store of built units under the one folder hawk names
(:func:`default_cache_dir`), so a second build of an unchanged unit runs no compiler there.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from .. import _ir
from .. import options as _options_mod
from .. import ospreys
from ..build_settings import BuildSettings, check as _check_settings, default_settings, refuse_choice, settings_of as _settings_of
from ..errors import EMBEDDED_C, HawkError
from ..ext import DEFAULT_KIND
from ..targets import DEFAULT_TARGETS, TARGETS, host_target
from . import manifest as _manifest
from . import request as _request
from .derivative import _per_kernel_derivative
from .sidecar import sidecar_of_unit


@dataclass(frozen=True)
class Entry:
    """One built target of one kernel: ``artifact`` its compiled object (a host entry names the build's library), ``source`` the
    unit's text, ``key`` the OSPREY's key of the build, ``hit`` whether the store served it and ``seconds`` what it took."""

    target: str
    artifact: Path
    source: Path
    key: str
    hit: bool
    seconds: float


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
    :func:`_unit_digest` computes, exposed
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
        """EVERY file this unit published — the set row greps. Each path once."""
        out = [] if self.manifest_path is None else [self.manifest_path]
        for a in self.artifacts:
            out.append(a.sidecar_path)
            out += a.links
            if a.library is not None:
                out.append(a.library)
            for entry in a.entries.values():
                out += [entry.artifact, entry.source]
        return list(dict.fromkeys(out))


# -- The unit's CONTENT, computed before anything is written. -------------- #
@dataclass(frozen=True)
class UnitFile:
    """One file of a unit: its name, its bytes, and their digest.

    ``blob`` is carried inline (HAWK-made text) or fetched from the
    a file of an OSPREY's store on demand (a compiled object, never held in
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
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            if self.link_to is not None:
                os.symlink(self.link_to, tmp)
            elif self.blob is not None:
                tmp.write_bytes(self.blob)
            else:
                shutil.copyfile(self.source_path, tmp)
            os.replace(tmp, path)
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


def text_blob(text: str) -> bytes:
    """The bytes a text file is published as, hashed as bytes so the digest reflects what's on disk, not an encoding assumption."""
    return text.encode("utf-8")


def digest_bytes(blob: bytes) -> str:
    """The sha256 of ``blob``."""
    return hashlib.sha256(blob).hexdigest()


def _digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def _unit_digest(files) -> str:
    """The unit's identity: its files' NAMES and CONTENT digests, sorted so two publishes of the same content agree regardless of
    build order."""
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


def _has_device(targets) -> bool:
    """Whether any target is built for a device."""
    return any(TARGETS[t].device for t in targets)


def _device_only(targets) -> bool:
    """Whether every target is built for a device."""
    return all(TARGETS[t].device for t in targets)


def _lead(entries) -> str:
    """The entry a manifest plugin names: the device object when the unit has one, else the first target's."""
    return next((t for t in entries if TARGETS[t].device), next(iter(entries)))


def _refuse_segmented_position(kernels, names) -> None:
    """Refuse a kernel with a segmented dispatch that reads the launch slot: the units of a segmented kernel address their lanes through
    the run's offsets."""
    for kernel, name in zip(kernels, names):
        reads_position = any(_ir.is_a(n, "sample_position") for n in _ir.walk_order(kernel.walk))
        if reads_position and any(d.strategy == "segmented" for d in _ir.dispatch_groups(kernel.walk)):
            raise HawkError(f"{name!r}: hawk.sample_position() is the launch slot, and a segmented unit addresses its lanes through the "
                            "run's offsets; use sample_index() there")


def _refuse_targets(name: str, targets) -> None:
    """Refuse an empty ``targets``, the embedded C target, and a target no OSPREY serves (by name, with the served ones)."""
    if "c" in targets:
        raise HawkError(EMBEDDED_C)
    if not targets:
        raise HawkError(f"{name!r}: no target to build; the served targets are {tuple(TARGETS) or 'none'}")
    for target in targets:
        ospreys.serving(target)


# -- The store the OSPREYs keep their built units in: hawk names the folder and asks for its state. -------------------------- #
def default_cache_dir() -> Path:
    """The folder the OSPREYs store built units under: ``$HAWK_CACHE_DIR``, else ``$XDG_CACHE_HOME/hawk``, else ``~/.cache/hawk``."""
    override = os.environ.get("HAWK_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "hawk"


def cache_stats(cache_dir=None) -> dict:
    """``{OSPREY name: the counters of its store}`` under ``cache_dir`` (default: :func:`default_cache_dir`), as the libraries state them."""
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    return {name: osprey.state("stats", root) for name, osprey in ospreys.found().items()}


def reset(cache_dir=None) -> None:
    """Empty every OSPREY's store under ``cache_dir`` (default: :func:`default_cache_dir`)."""
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    for osprey in ospreys.found().values():
        osprey.state("reset", root)


def build_key(kernels, *, targets, mode: str = "float64", options: Mapping | None = None, opt_level: str = "") -> str:
    """One opaque text naming everything a build of ``kernels`` for ``targets`` is made of.

    Per kernel: its name, the text of its traced nodes, the request the lowering is asked (declarations, kind, sinks, settings)
    and its derivative tag; per target: the ``key`` the serving OSPREY states for the machine; and ``mode``, ``options`` and
    ``opt_level``. Equal requests give equal texts, and a change in any of these gives another. A caller that has choices of its
    own (a folder name, say) hashes this text together with them.
    """
    options = _user_options(options, False, None, "", "")
    kernels, targets = list(kernels), tuple(targets)
    names = [k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    explicit = False
    kind = _resolve_bundle_kind(names, kernels, None)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]
    cfg = _resolved_settings(None, targets)
    machines = _machines(targets, options, opt_level, default_cache_dir())
    keyed = {t: {"key": machines[t]["key"]} for t in targets}
    plans = _plans(kernels, names, kinds, mode=mode, targets=targets, settings=cfg, options=options, machines=keyed, sizes=None)
    derivative = _per_kernel_derivative(kernels, names, None)
    return hashlib.sha256(json.dumps([plans, derivative, opt_level], sort_keys=True, default=repr).encode()).hexdigest()


# -- The calls to the OSPREYs. ------------------------------------------------------------------------------------------------ #
def _machines(targets, options, opt_level: str, root: Path) -> dict:
    """``{target: the machine answer of its OSPREY}``, stored as given; a machine with no tools still states itself.

    ``root`` is the folder the OSPREY may keep what it finds out about the machine in (the store of the build).
    """
    return {t: ospreys.serving(t).machine({"target": t, "options": _options_mod.of_target(options, t), "opt_level": opt_level,
                                           "root": str(root)})
            for t in targets}


def _plans(kernels, names, kinds, *, mode, targets, settings, options, machines, sizes, exports_block: bool = False) -> list:
    """``(request, dump, dialects)`` of each kernel: the request the lowering is asked, the text of the nodes it names and the
    text of their dialects (empty: the lowering context declares the dialect kinds itself). A build asks for texts without the
    self-check block, which the text of its set carries once; a text that is compiled alone (:func:`emit_source`) asks for it."""
    out = []
    for kernel, name, kind in zip(kernels, names, kinds):
        roots = _request.sinks_of(kernel)
        # the lowering loads the text into an empty context, where a node's handle is its position in the text: so the roots
        # are copied into one first, and the request names the copies' numbers
        copies = _ir.adopt(_ir.new_context(), roots)
        position = dict(zip(roots, _ir.dump_numbers(copies)))
        request = _request.request(kernel, name=name, scalar=mode, kind=kind, targets=targets, settings=settings, position=position,
                                   options=options, machines=machines, sizes=sizes, exports_block=exports_block)
        out.append((request, _ir.dump_text(copies), ""))
    return out


def _restricted(request: dict, names) -> dict:
    """``request`` asking only for the targets spelled ``names`` (the ones one OSPREY serves)."""
    mine = {**request, "targets": list(names), "machine": {n: request["machine"][n] for n in names}}
    reserved = next((m["stated"]["reserved_names"] for m in mine["machine"].values()
                     if isinstance(m.get("stated"), dict) and "reserved_names" in m["stated"]), None)
    if reserved is not None:
        mine["reserved_names"] = reserved
    return mine


def _linker(targets):
    """The OSPREY that links the build: the one serving the first host-entry target, else any host OSPREY."""
    host = next((t for t in targets if TARGETS[t].host_entry), None)
    return ospreys.serving((host_target() if host is None else TARGETS[host]).id)


def _lower(plan, targets) -> dict:
    """The answer to one kernel's request, as the linking OSPREY merges the answers of the OSPREYs lowered.

    The leader (the OSPREY whose target row says ``device``) is lowered first and its request carries ``companions``, the
    target names of the other OSPREYs of the build (none for a build of one). The ``peer`` member of its answer is passed on,
    unread, in the request of every other OSPREY as ``peer: {leader's target name: record}``. All answers go, as given, to
    the linking OSPREY's ``merge``, which states a merged answer with no ``peer``.
    """
    request, dump, dialects = plan
    owned = {}
    for target in targets:
        owned.setdefault(ospreys.serving(target).name, []).append(target)
    leader = next((n for n, ts in owned.items() if any(TARGETS[t].device for t in ts)), None)
    leader_target = None if leader is None else next(TARGETS[t].target for t in owned[leader] if TARGETS[t].device)
    companions = [TARGETS[t].target for n, ts in owned.items() if n != leader for t in ts]
    answers, peer = [], None
    for name in sorted(owned, key=lambda n: n != leader):
        mine = _restricted(request, [TARGETS[t].target for t in owned[name]])
        if name == leader:
            mine["companions"] = companions
        elif leader is not None:
            mine["peer"] = {leader_target: peer}
        answer = ospreys.found()[name].lower(mine, dump, dialects)
        if name == leader:
            peer = answer.get("peer")
        answers.append(answer)
    return _linker(targets).merge(answers)


def _require_tools(machines: dict) -> None:
    """Raise ``ToolchainAbsent`` for the first target whose machine says its tools are absent: a build needs them."""
    for target, machine in machines.items():
        ospreys.serving(target).require_tools(machine)


def _texts(unit: dict, targets) -> dict:
    """``{target: the text record the OSPREYs wrote for it}`` of one answered unit; a target with no text is refused by name."""
    by_request = {TARGETS[t].target: t for t in targets}
    found = {by_request[text["target"]]: text for text in unit["texts"] if text["target"] in by_request}
    if absent := [t for t in targets if t not in found]:
        raise HawkError(f"the lowering wrote no {absent[0]} text for {unit['meta']['name']!r}")
    return found


def _build_one(target: str, texts: dict, unit: dict, *, machines, options, defines, opt_level: str, block, kind, root: Path) -> dict:
    """The build of one unit for one target, as its OSPREY answers: the files, the format, the figures."""
    request = {"v": 1, "name": unit["meta"]["name"], "target": TARGETS[target].target, "text": texts[target]["text"], "meta": unit["meta"],
               "machine": machines[target], "options": _options_mod.of_target(options, target), "defines": list(defines),
               "opt_level": opt_level, "block": block}
    return ospreys.serving(target).build(kind.apply(request), root)


def _built(answer: dict) -> tuple:
    """``(result, artifact name, source name, format)`` of one build answer: what a draft's entry holds."""
    return SimpleNamespace(key=answer["key"], hit=answer["hit"], seconds=answer["seconds"]), answer["artifact"], answer["source"], answer["format"]


def _draft(kernel, unit, family, answers, *, mode, targets, machines, derivative, block, records=None) -> _Draft:
    """Assemble the unit CONTENT of one answered unit from what each target's OSPREY built for it.

    Everything here is a pure function of the answers; nothing is written, which lets the publish step lay the files down at once.
    ``unit`` is the unit's record in the lowering answer, ``family`` the records of every unit the answer holds for ``kernel``,
    ``answers`` ``{target: its build answer}``. The sidecar is the unit's record and the kernel's own declarations, with the
    figures each OSPREY gave and the machine it stated, written as given.
    """
    name = unit["meta"]["name"]
    entries, files = {}, []
    for target in targets:
        # A host-entry target's artifact is the build's linked library: its object is a link input, so only its source is published.
        published = [answers[target]["source"]] if TARGETS[target].host_entry else list(answers[target]["files"])
        files += [UnitFile(file, _digest_file(Path(answers[target]["files"][file])), source_path=Path(answers[target]["files"][file]))
                  for file in published]
        built = _built(answers[target])
        # the runtime opens a host unit as ``<unit>.so``: the file name is the bundle's, the object the library built is link input
        entries[target] = (built[0], f"{name}.so", *built[2:]) if TARGETS[target].host_entry else built
    meta = _sidecar(kernel, unit, family, mode=mode, targets=targets, machines=machines, derivative=derivative, block=block,
                    records=records, built=answers)
    blob = text_blob(json.dumps(meta, indent=2) + "\n")
    files.append(UnitFile(f"{name}.json", digest_bytes(blob), blob=blob))
    return _Draft(name, kernel.name, unit["meta"], meta, f"{name}.json", entries, tuple(files))


def _sidecar(kernel, unit, family, *, mode, targets, machines, derivative, block, records, built=None) -> dict:
    """The sidecar of one answered unit: what the lowering states and the machines stated, and, when ``built`` (``{target: its
    build answer}``) is given, what the builds measured (the unit's launch figures and each tool's figures). :func:`describe` and
    :func:`build` both come through here, so the one document has one writer."""
    lead = _lead(built if built is not None else dict.fromkeys(targets))
    stated = {ospreys.serving(t).name: machines[t]["stated"] for t in targets}
    host = next((t for t in targets if TARGETS[t].host_entry), None)
    measured = tool = None
    if built is not None:
        measured = built[lead]["figures"].get("launch")
        tool = {ospreys.serving(t).name: built[t]["figures"] for t in targets}
    return sidecar_of_unit(kernel, unit, lead_device=TARGETS[lead].device, fmt=built[lead]["format"] if built else TARGETS[lead].format,
                           scalar_type=mode, exec_targets=[TARGETS[t].entry for t in targets], family=family, derivative=derivative,
                           block=block, measured=measured, tool=tool, stated=stated, records=records)


@dataclass(frozen=True)
class _Library:
    """The shared library of one build call: its file name, the host target whose units it holds
    and the files it publishes (the library, then one link to it for each host unit not named like it)."""

    name: str
    target: str | None
    files: tuple


def _link_set(drafts, answers, built, *, targets, stem: str, exports, root: Path) -> "_Library | None":
    """Link the units of ``drafts`` into one library with the OSPREY that serves the host: each unit's host part with the
    text of the set. The set lists every unit of every answer in the answers' order: a unit with no text (the source unit of a split
    kernel, which names its stages) has a record in the library and no part. Each host unit also gets a link ``<unit>.<ext>`` to the
    library, which the runtime still opens by the unit's name. ``None`` for a build of no host target: its units are the device
    artifacts themselves, and a library links host objects only."""
    host = next((t for t in targets if TARGETS[t].host_entry), None)
    if host is None:
        return None
    linker = _linker(targets)
    written = iter(built)
    parts, metas = [], []
    for unit in (u for answer in answers for u in answer["units"]):
        metas.append(unit["meta"])
        if unit["texts"]:
            parts.append(next(written)[host])
    path = Path(linker.link(parts, linker.set_text(metas, exports), f"{stem}.so", root)["library"])
    name = path.name
    digest = _digest_file(path)
    links = tuple(UnitFile(d.entries[host][1], digest, link_to=name) for d in drafts if d.entries[host][1] != name)
    return _Library(name, host, (UnitFile(name, digest, source_path=path), *links))


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
    """The public :class:`Artifact` view of a drafted kernel published at ``directory``. A host entry names the build's shared
    library (``library``), and the unit's link to it is listed in ``links``."""
    entries = {
        target: Entry(target, directory / (library.name if library is not None and target == library.target else art_name),
                      directory / src_name, result.key, result.hit, result.seconds)
        for target, (result, art_name, src_name, _fmt) in draft.entries.items()
    }
    link = None if library is None or library.target is None else draft.entries[library.target][1]
    links = () if link is None or link == library.name else (directory / link,)
    return Artifact(draft.name, directory, directory / draft.sidecar_name, draft.sidecar, entries, links,
                    None if library is None else directory / library.name)


def _publish(files, directory: Path) -> None:
    """Lay ``files`` down in ``directory``, each whole to a private sibling and then renamed into place: a concurrent reader (or a
    second builder of the same unit) sees the old file or the new one, never a partial write."""
    directory.mkdir(parents=True, exist_ok=True)
    for f in files:
        f.write(directory)


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
            # The real format (the OSPREY states it for the compiled
            # bytes), not a static "ptx" guess.
            d.entries[_lead(d.entries)][3],
            None if library is None else library.name)
        for order, d in enumerate(drafts)
    ]
    return _manifest.manifest_doc(entries, exec_targets=axis_targets,
                                  exec_access=access, exec_op=op)


def _resolved_settings(settings, targets, *, emit_form=None, linalg_form=None, wide_form=None, launch_bounds=None):
    """The :class:`~hawk.build_settings.Settings` of a build: ``settings`` over the defaults, an explicit keyword over both, a
    keyword that disagrees with ``settings`` refused by name. Values the targets cannot take are the library's refusal."""
    given = _settings_of(settings, default_settings())
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
    if launch_bounds is not None and not _has_device(targets):
        raise ValueError(f"launch_bounds={launch_bounds!r}: launch bounds are a device setting, refused on a build with no device target")
    bounds = None if launch_bounds is None else tuple(int(n) for n in launch_bounds)
    return _check_settings(replace(given, form=emit_form, linalg_form=linalg_form, wide_form=wide_form, launch_bounds=bounds))


def _user_options(options, literal_table, embedded_profile, device_arch, host_profile) -> dict:
    """The user's ``options`` with the old keywords folded in, checked (:mod:`hawk.options`)."""
    if embedded_profile is not None or any(str(key).partition(".")[0] == "c" for key in options or ()):
        raise HawkError(EMBEDDED_C)
    merged = _options_mod.merge(options, literal_table=literal_table, device_arch=device_arch, host_profile=host_profile)
    _options_mod.check(merged)
    return merged


def _emitted(kernel, name, mode, kind, targets, does, instead, settings, options, opt_level, sizes, root: Path, exports_block: bool) -> tuple:
    """``(answer, unit, kind, machines)`` of ONE kernel: the lowering half of :func:`build` and :func:`emit_source`.
    A kernel that needs several launches (a segmented dispatch, a value reduced over samples) is refused: ``does`` says what the
    caller publishes and ``instead`` what to call."""
    _refuse_targets(name, targets)
    if any(d.strategy == "segmented" for d in _ir.dispatch_groups(kernel.walk)):
        raise HawkError(f"{name!r} carries a segmented dispatch: {does} ONE artifact, and a segmented kernel needs K "
                        f"— call {instead}([kernel], ...) instead")
    kind = resolve_kind(name, getattr(kernel, "kind", None), kind)
    machines = _machines(targets, options, opt_level, root)
    plan = _plans([kernel], [name], [kind], mode=mode, targets=targets, settings=settings, options=options, machines=machines,
                  sizes=sizes, exports_block=exports_block)[0]
    answer = _lower(plan, targets)
    written = [u for u in answer["units"] if u["texts"]]
    if len(written) != 1:
        # what the answer is decides the cause: a split with a team stage is a team kernel, any other split a reduction over samples
        teams = any(stage["team"] is not None for unit in answer["units"] for stage in unit["meta"]["stages"])
        cause = ("is worked by a team per lane, so it runs as several stages, one launch each" if teams
                 else "reads a value reduced over samples, so it runs as several launches")
        raise HawkError(f"{name!r} {cause}: {does} ONE artifact — call {instead}([kernel], ...) instead")
    return answer, written[0], kind, machines


def _lowered(kernels, names, kind, *, mode, targets, cfg, options, opt_level, root: Path, sizes, tools_needed: bool) -> tuple:
    """``(kinds, machines, answers)`` of a set of kernels: each target's machine and each kernel's lowering answer, with every refusal
    the lowering makes. ``tools_needed`` refuses a machine that lacks its tools before anything is lowered; a caller that compiles
    nothing does not ask."""
    explicit = kind is not None
    kind = _resolve_bundle_kind(names, kernels, kind)
    kinds = [_member_kind(k, kind, explicit) for k in kernels]
    machines = _machines(targets, options, opt_level, root)
    plans = _plans(kernels, names, kinds, mode=mode, targets=targets, settings=cfg, options=options, machines=machines, sizes=sizes)
    if tools_needed:
        _require_tools(machines)
    answers = [_lower(plan, targets) for plan in plans]
    _refuse_repeated_units([(k.name, unit["meta"]["name"]) for k, answer in zip(kernels, answers) for unit in answer["units"] if unit["texts"]])
    if cfg.launch_bounds is not None:
        for answer in answers:
            teams = [st["unit"]["name"] for unit in answer["units"] for st in unit["meta"]["stages"] if st["team"] is not None]
            if teams:
                raise HawkError(f"launch_bounds={cfg.launch_bounds!r}: {teams[0]!r} is built as a warp per lane (a team), whose block the "
                                "emit fixes; drop launch_bounds, or build it with wide_form='thread'")
    return kinds, machines, answers


def _build_units(families, *, targets, machines, options, defines, opt_level, block, kinds, root) -> list:
    """The build answers of every unit of every family, ``[[{target: answer}]]``: every call runs on its own thread and the OSPREYs
    bound how many compiles run at once; a failure is the first one in order, raised after every started call has finished."""
    calls = [[[(target, unit, _texts(unit, targets), kind) for target in targets] for unit in answer["units"] if unit["texts"]]
             for answer, kind in zip(families, kinds)]
    flat = [call for family in calls for unit in family for call in unit]
    if not flat:
        return [[] for _ in calls]
    with ThreadPoolExecutor(max_workers=len(flat), thread_name_prefix="hawk-build") as pool:
        futures = [pool.submit(_build_one, target, texts, unit, machines=machines, options=options, defines=defines,
                               opt_level=opt_level, block=block, kind=kind, root=root) for target, unit, texts, kind in flat]
        for future in futures:
            future.exception()
    done = iter(f.result() for f in futures)
    return [[{call[0]: next(done) for call in unit} for unit in family] for family in calls]


# -- The public entry points. ----------------------------------------------- #
def build(kernel, directory, *, mode: str = "float64", targets=DEFAULT_TARGETS,
          kind=None, layout_sizes_override=None, defines=(), cache_dir=None,
          device_arch: str = "", name: str | None = None,
          derivative: dict | None = None, host_profile: str = "",
          opt_level: str = "", literal_table: bool = False,
          embedded_profile=None, options: Mapping | None = None,
          settings: BuildSettings | None = None) -> Artifact:
    """Emit, build and publish ONE kernel into ``directory``.

    ``kernel`` is anything carrying ``.name``/``.sinks``/``.walk`` (a :class:`hawk.trace.Kernel`). ``layout_sizes_override=`` is a
    test-only door threaded to the export block.

    This is the PRIMITIVE, not the deployment unit: no ``manifest.json``; :func:`build_bundle` publishes a unit (one manifest per
    bundle). ``opt_level`` is the optimisation level asked of every OSPREY, as it spells it (empty: its own default).

    Refuses a ``segmented`` dispatch, which needs K artifacts: call :func:`build_bundle` with ``[kernel]`` instead."""
    name = name or kernel.name
    targets = tuple(targets)
    options = _user_options(options, literal_table, embedded_profile, device_arch, host_profile)
    cfg = _resolved_settings(settings, targets)
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    answer, unit, kind, machines = _emitted(kernel, name, mode, kind, targets, "build() publishes", "build_bundle", cfg, options,
                                            opt_level, layout_sizes_override, root, exports_block=False)
    _require_tools(machines)
    built = _build_units([answer], targets=targets, machines=machines, options=options, defines=defines, opt_level=opt_level,
                         block=None if cfg.launch_bounds is None else cfg.launch_bounds[0], kinds=[kind], root=root)[0][0]
    draft = _draft(kernel, unit, answer["units"], built, mode=mode, targets=targets, machines=machines, derivative=derivative,
                   block=None if cfg.launch_bounds is None else cfg.launch_bounds[0], records=answer)
    # one kernel is a set of one: its library is named for its unit
    library = _link_set([draft], [answer], [built], targets=targets, stem=name, exports=layout_sizes_override, root=root)
    _publish((*draft.files, *(library.files if library else ())), Path(directory))
    return _artifact_of(draft, Path(directory), library)


@dataclass(frozen=True)
class EmitUnit:
    """What :func:`emit_source` returns: ``sources`` (target name to the text record its OSPREY wrote, with ``.text``, of each as
    :func:`build` would write it), the ``sidecar`` dict without the figures only a build gives, and ``unit``, the kernel as it
    was emitted."""

    sources: dict
    sidecar: dict
    unit: object


def describe(kernels, *, targets=None, mode: str = "float64", options: Mapping | None = None, opt_level: str = "") -> dict:
    """``{kernel name: the sidecar a build of it would write}``, answered before any build and with no compiler run.

    The document has the sidecar's own shape, less the members that exist only after a compile (the unit files, each tool's
    figures and the unit's measured launch figures), so one reader serves a kernel before and after its build. It asks each target's
    OSPREY for its machine and the lowering and nothing more, so it answers on a machine whose tools are absent. ``targets`` left
    unset is every target an OSPREY serves, as :func:`build` chooses; the refusals are :func:`build`'s up to the compile. A kernel
    that splits into several units answers with its first unit, which lists the stages (``stages``) of the others.
    """
    kernels = list(kernels)
    targets = tuple(DEFAULT_TARGETS if targets is None else targets)
    names = [k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    _refuse_segmented_position(kernels, names)
    if not kernels:
        raise HawkError("no kernel to describe: name at least one")
    options = _user_options(options, False, None, "", "")
    cfg = _resolved_settings(None, targets)
    derivative = _per_kernel_derivative(kernels, names, None)
    _kinds, machines, answers = _lowered(kernels, names, None, mode=mode, targets=targets, cfg=cfg, options=options, opt_level=opt_level,
                                         root=default_cache_dir(), sizes=None, tools_needed=False)
    return {k.name: _sidecar(k, unit, answer["units"], mode=mode, targets=targets, machines=machines, derivative=d, block=None,
                             records=answer)
            for k, answer, d in zip(kernels, answers, derivative)
            for unit in [next(u for u in answer["units"] if u["texts"])]}


def emit_source(kernel, *, name: str | None = None, mode: str = "float64", targets=DEFAULT_TARGETS, kind=None,
                device_arch: str = "", literal_table: bool = False, options: Mapping | None = None,
                settings: BuildSettings | None = None) -> EmitUnit:
    """Emit ONE kernel's source text and sidecar for ``targets`` without building or writing anything.

    The sidecar is what :func:`build` publishes less the figures a build gives. The host text is the one a build publishes with the
    exports block in it (a build states that block once, in the text of its set), so it is a whole translation unit a consumer can
    build alone. Honours ``settings`` (:class:`hawk.build_settings.BuildSettings`). Refuses what :func:`build` refuses; a machine that
    with no tools raises ``ToolchainAbsent`` only for a build."""
    name = name or kernel.name
    targets = tuple(targets)
    options = _user_options(options, literal_table, None, device_arch, "")
    cfg = _resolved_settings(settings, targets)
    answer, unit, kind, machines = _emitted(kernel, name, mode, kind, targets, "emit_source returns", "build_bundle", cfg, options, "",
                                            None, default_cache_dir(), exports_block=True)
    bounds = cfg.launch_bounds
    sidecar = _sidecar(kernel, unit, answer["units"], mode=mode, targets=targets, machines=machines, derivative=None,
                       block=None if bounds is None else bounds[0], records=answer)
    return EmitUnit({t: SimpleNamespace(**text) for t, text in _texts(unit, targets).items()}, sidecar, kernel)


def build_bundle(kernels, directory, *, mode: str = "float64",
                 targets=DEFAULT_TARGETS, kind=None, layout_sizes_override=None,
                 defines=(), cache_dir=None, device_arch: str = "",
                 name: str | None = None,
                 derivative: Mapping[str, str] | None = None,
                 host_profile: str = "", opt_level: str = "",
                 literal_table: bool = False, embedded_profile=None, options: Mapping | None = None,
                 emit_form: str | None = None, linalg_form: str | None = None, wide_form: str | None = None,
                 launch_bounds: tuple[int, int] | None = None,
                 settings: BuildSettings | None = None) -> Bundle:
    """Build every kernel into ``directory`` and write ONE ``manifest.json`` over them.

    An ``emit_form``/``linalg_form``/``wide_form``/``launch_bounds`` left unset is the value of ``settings`` (a
    :class:`hawk.build_settings.BuildSettings`; the default: ``"auto"``, no bound); an explicit argument wins. ``emit_form`` is how a
    product, dot, sum or norm is spelled: ``"auto"`` unrolls a site of at most 16 elements and writes a longer one as a loop,
    ``"static"`` unrolls every site, ``"rolled"`` writes every site as a loop, the fast-compile mode. ``linalg_form`` is how
    per-lane linear algebra is worked: ``"auto"`` keeps one thread per matrix up to ``n = 32`` and uses a team of threads per
    matrix above it, ``"team"`` uses a team at every size, ``"thread"`` refuses a matrix above 32 by name. ``wide_form`` is where a
    per-lane value wider than a thread's registers lives: ``"auto"`` builds a kernel as a warp per lane when every target is a
    device and the kernel has a rule in that form, else places the value; ``"thread"`` keeps every value in the lane's registers and
    local memory; ``"team"`` asks for the warp form (device targets only). ``launch_bounds=(threads, blocks_per_sm)`` bounds every
    device kernel and records ``launch.block = threads`` in the sidecar. Every member must agree on the execution axis (a
    manifest-level declaration): a set that does not agree is two bundles.

    This is the DEPLOYMENT UNIT. The OSPREYs store what they build under ``cache_dir`` (default :func:`default_cache_dir`), so a
    repeat build of an unchanged definition runs no compiler.

    ``derivative`` names each derivative kernel's primal, per kernel: ``{kernel_name: primal_name}``, or leave it (``None``) to use a
    :func:`hawk.diff.vjp`/:func:`~hawk.diff.jvp` kernel's own recorded primal. A primal outside this bundle is named by
    ``{kernel_name: (primal_name, primal_unit_digest)}`` via its :attr:`Bundle.digest`.

    ``host_profile``, ``device_arch`` and ``literal_table`` are the old spellings of the target options ``host.profile``,
    ``cuda.arch`` and ``cuda.literal_table`` (:mod:`hawk.options`); ``opt_level`` is the optimisation level asked of every OSPREY.
    Only the targets an OSPREY serves are built: any other target, ``"c"`` included, is refused by name."""
    options = _user_options(options, literal_table, embedded_profile, device_arch, host_profile)
    kernels = list(kernels)
    directory = Path(directory)
    targets = tuple(targets)
    names = [name or k.name for k in kernels]
    _refuse_targets(names[0] if names else "", targets)
    _refuse_segmented_position(kernels, names)
    if not kernels:
        raise HawkError("no kernel to build: a bundle holds at least one")
    cfg = _resolved_settings(settings, targets, emit_form=emit_form, linalg_form=linalg_form, wide_form=wide_form,
                             launch_bounds=launch_bounds)
    block = None if cfg.launch_bounds is None else cfg.launch_bounds[0]
    per_kernel_derivative = _per_kernel_derivative(kernels, names, derivative)
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    kinds, machines, answers = _lowered(kernels, names, kind, mode=mode, targets=targets, cfg=cfg, options=options, opt_level=opt_level,
                                        root=root, sizes=layout_sizes_override, tools_needed=True)
    built = _build_units(answers, targets=targets, machines=machines, options=options, defines=defines, opt_level=opt_level,
                         block=block, kinds=kinds, root=root)
    families = [[_draft(k, unit, answer["units"], got, mode=mode, targets=targets, machines=machines, derivative=d, block=block, records=answer)
                 for unit, got in zip((u for u in answer["units"] if u["texts"]), family)]
                for k, answer, family, d in zip(kernels, answers, built, per_kernel_derivative)]
    drafts = [d for family in families for d in family]
    library = _link_set(drafts, answers, [got for family in built for got in family], targets=targets, stem=name or "units",
                        exports=layout_sizes_override, root=root)
    files = tuple(f for d in drafts for f in d.files) + (library.files if library else ())
    doc = _manifest_of(families, library)
    manifest_blob = text_blob(json.dumps(doc, indent=2) + "\n")
    files += (UnitFile("manifest.json", digest_bytes(manifest_blob), blob=manifest_blob),)
    _publish(files, directory)
    return Bundle(directory, directory / "manifest.json", doc, tuple(_artifact_of(d, directory, library) for d in drafts),
                  digest=_unit_digest(files))
