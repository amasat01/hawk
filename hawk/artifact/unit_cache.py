# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-process unit memo, the publish counters and log, the cross-process
digest stamp, and the unit key :func:`hawk.artifact.bundle.build_bundle`
memoises under."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from types import MappingProxyType
from pathlib import Path

from .. import _contracts
from .._errors import HawkError
from .. import options as _options_mod
from .. import registry as _registry
from ..compile import digest_file
from ..compile.cache import ClosureWatch, artifact_memo_get, validity_digest, write_atomic
from ..compile.toolchain import host_profile as _host_profile
from ..compile.toolchain import opt_level as _opt_level
#: ``HAWK_BOX=0`` builds every kernel without a box (an A/B switch, not an API).
BOX_ENV = "HAWK_BOX"
from ..targets import TARGETS
from ..targets.cards import thread_box as thread_box_card
from ..trace.decl import Plane

#: The digest stamp a published unit carries. Dot-prefixed and
#: extension-less so it's not a member of the unit in any consumer's
#: eyes — not in :meth:`Bundle.files`, not scanned by the audits, and
#: not part of the unit digest it records (a file can't hash itself).
STAMP_NAME = ".hawk_unit"

#: The per-process DEPLOYMENT-UNIT memo. Bounded, and cleared wholesale
#: rather than evicted entry by entry: the bound exists so a long-lived
#: process can't grow it without limit, not to implement a policy.
_UNIT_MEMO: dict = {}
_UNIT_MEMO_LIMIT = 64

#: Static-graph counter: how many times :func:`build_bundle` was CALLED —
#: an attempt, not a success (a raise still counts). Lives here, not in
#: ``eagle._counters``, since hawk must not import eagle. Deliberately
#: separate from ``_UNIT_STATS`` and NEVER reset by
#: :func:`reset_unit_memo` — a reset would let two concurrent readers
#: race over which observes a given increment.
COUNTERS = {"builds": 0}

#: One lock for every read-modify-write on this module's process state
#: (``COUNTERS``, ``_UNIT_STATS``, ``_PUBLISH_LOG``): a bare ``+= 1`` on a
#: dict item is not atomic on a free-threaded interpreter. Held for a few
#: bytecodes at a time, never across I/O or a build.
_STATE_LOCK = threading.Lock()


def bump_builds() -> None:
    """Count one :func:`~hawk.artifact.bundle.build_bundle` call."""
    with _STATE_LOCK:
        COUNTERS["builds"] += 1


def counters_snapshot() -> dict:
    """A fresh ``dict`` copy of hawk's own static-graph counters."""
    with _STATE_LOCK:
        return dict(COUNTERS)

#: Publish counters, record-only: a row that wants to know WHICH answer
#: the publisher gave reads them instead of inferring it from wall time.
_UNIT_STATS = {"memo_hits": 0, "stamp_hits": 0, "published": 0}

#: A bounded ring of the last publish decisions, each naming the unit
#: digest it was about, so a re-publish can say what it saw.
_PUBLISH_LOG: list[str] = []
_PUBLISH_LOG_LIMIT = 64


def unit_stats() -> dict:
    """A copy of the publish counters (memo hits, stamp hits, real publishes)."""
    with _STATE_LOCK:
        return dict(_UNIT_STATS)


def publish_log() -> tuple[str, ...]:
    """The recent publish decisions, oldest first, each naming a unit digest."""
    with _STATE_LOCK:
        return tuple(_PUBLISH_LOG)


def reset_unit_memo() -> None:
    """Forget every memoised unit, the counters and the log — the door
    back to the disk path, for a row that wants a cold publisher."""
    _UNIT_MEMO.clear()
    with _STATE_LOCK:
        _PUBLISH_LOG.clear()
        for k in _UNIT_STATS:
            _UNIT_STATS[k] = 0


def _log_locked(message: str) -> None:
    if len(_PUBLISH_LOG) >= _PUBLISH_LOG_LIMIT:
        del _PUBLISH_LOG[0]
    _PUBLISH_LOG.append(message)


def _log(message: str) -> None:
    with _STATE_LOCK:
        _log_locked(message)


def _note(kind: str, message: str) -> None:
    with _STATE_LOCK:
        _UNIT_STATS[kind] += 1
        _log_locked(message)


def arch(device_arch: str = "") -> str:
    """The CUDA arch HAWK compiles a device artifact for -- the public name
    for the ONE resolver (:meth:`hawk.backends.Backend.resolve_arch`):
    ``device_arch=`` > ``$HAWK_CUDA_ARCH`` > the running GPU (probed once
    per process) > PTX for the toolkit's oldest architecture, each checked
    against what the toolkit compiles. Kept here, under the door callers already
    use (:mod:`hawk.artifact.bundle` and this module both called a function
    of this name before the resolver grew the env/probe/fallback chain).

    Call only where a target compiles with the device toolchain: every call may
    reach the device probe, so a host-only build must never call this at
    all."""
    from ..backends import need_device

    return need_device().resolve_arch(device_arch)



# -- The stamp and the two validity questions a memo hit must answer. ------ #
def _stamp_digest(directory: Path) -> str | None:
    """The digest the unit at ``directory`` last published under, or
    ``None`` when there's no stamp or it can't be read."""
    try:
        doc = json.loads((directory / STAMP_NAME).read_text())
    except (OSError, ValueError):
        return None
    got = doc.get("unit")
    return got if isinstance(got, str) else None


def _write_stamp(directory: Path, digest: str, files) -> None:
    """Record the unit's identity beside it, with per-file digests."""
    write_atomic(directory / STAMP_NAME, json.dumps(
        {"unit": digest, "aether_abi": _contracts.AETHER_ABI_VERSION,
         "files": {f.name: f.digest for f in sorted(files, key=lambda f: f.name)}},
        indent=2) + "\n")


def _first_moved(pairs) -> str | None:
    """The first published file whose bytes are no longer what the unit
    recorded, or ``None`` if intact.

    By CONTENT, never by mtime alone: a unit rewritten with different
    bytes is a different unit whatever its timestamps say, and one whose
    timestamps moved with its bytes unchanged must stay a hit."""
    for path, want in pairs:
        if digest_file(path) != want:
            return path
    return None


# -- The unit key. ----------------------------------------------------------- #
def _wide_part(scalar_type: str, spec) -> str:
    """The wide-value thresholds of the build when they differ from the defaults (a card row, or an override), else nothing,
    so a build with no row keeps the key it always had."""
    from ..targets import cards

    if spec.device is None:         # a host build has no wide card: it places by the host variant's own table
        return ""
    doc = cards.wide_card(spec)["defaults"]
    got = cards.wide(scalar_type, spec)
    return "" if all(got[k] == doc[k] for k in cards.WIDE_KEYS) else "wide:" + ":".join(str(got[k]) for k in cards.WIDE_KEYS)


def _backend_key(row, options, source_digests, spec, smode, wide_class: str) -> str:
    """The key of ONE built backend: its terms (:func:`_row_terms`) and the facts of ITS OWN variant only: the digest of that variant's
    data and, for a device row, the class its thresholds come from and the card row of its arch; for a host row, nothing of a device."""
    own = spec.by_target.get(row.id)
    terms = [*_row_terms(row, options, source_digests),
             "variant:" + (hashlib.sha256(json.dumps(own, sort_keys=True, default=str).encode("utf-8")).hexdigest() if own else "")]
    if row.device:
        terms += ["wide_class:" + wide_class if wide_class else "", _wide_part(smode.id, spec)]
    h = hashlib.sha256()
    for term in terms:
        h.update(term.encode("utf-8") + b"\0")
    return h.hexdigest()


def _row_terms(row, options, source_digests) -> tuple:
    """One built backend's key terms: its distribution version (which covers its defaults), each option given under its
    prefix, then its own."""
    own = row.own_options(options)
    return ("version:" + row.id + "=" + _registry.version(row.id), *(f"{row.id}.{k}={_options_mod.canon(row.id + '.' + k, own[k])}" for k in sorted(own)),
            *row.key_terms(source_digests, own))


def _plain(v):
    """``v`` as text when it is plain data (numbers, text, and tuples, lists, sets, mappings of them), else ``None``."""
    if v is None or isinstance(v, (str, bytes, int, float, bool)):
        return repr(v)
    if isinstance(v, (tuple, list)):
        parts = [_plain(x) for x in v]
        return None if None in parts else "(" + ",".join(parts) + ")"
    if isinstance(v, (set, frozenset)):
        parts = [_plain(x) for x in v]
        return None if None in parts else "{" + ",".join(sorted(parts)) + "}"
    if isinstance(v, (dict, MappingProxyType)):
        parts = [(_plain(k), _plain(x)) for k, x in v.items()]
        return None if any(None in kv for kv in parts) else "{" + ",".join(f"{k}:{x}" for k, x in sorted(parts)) + "}"
    return None


def _emission_modules(targets) -> list:
    """The modules whose tables ``targets`` emit with, as the manifests DECLARE them (``emission_modules``): hawk's own, each target's
    extension's, and the component's of every component a target's manifest names (``components``). The list is the declaration alone:
    loaded or not, a declared module is in it (:func:`emission_digest` imports it)."""
    from .. import components as _components

    names = set(_registry.shared_emission_modules())
    for t in targets:
        names.update(_registry.emission_lists(t)[0])
        for c in _registry.traits(t).get("components", ()):
            names.update(_components.component(c).MANIFEST.get("emission_modules", ()))
    return sorted(names)


def emission_digest(targets) -> str:
    """The sha256 of the plain module-level tables of the emission modules ``targets`` use (:func:`_emission_modules`): what the unit
    is emitted WITH besides its inputs, so an import-time edit of a shared spelling table or text constant moves the key, and an edit
    of a table only another target emits with does not (sideways identity)."""
    import importlib

    h = hashlib.sha256()
    for name in _emission_modules(targets):
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            raise HawkError(f"the emission module {name!r}, declared by the manifest of a target of this build {tuple(targets)}, cannot be "
                            f"imported ({exc}): its tables are part of the unit key, so the build is refused rather than keyed without it") from exc
        h.update(name.encode() + b"\0")
        for key in sorted(vars(mod)):
            if key.startswith("__"):
                continue
            text = _plain(vars(mod)[key])
            if text is not None:
                h.update(f"{key}={text}\0".encode("utf-8"))
    return h.hexdigest()


def _unit_key(names, walk_digests, declarations, *, targets, smode, kind, defines, cache_dir,
              device_arch, derivative_blocks, layout_sizes_override,
              host_profile: str = "", opt_level: str = "",
              options=None,
              source_digests=(), groups=(), emit_form: str = "auto",
              linalg_form: str = "auto", wide_form: str = "auto", wide_class: str = "",
              launch_bounds: tuple[int, int] | None = None, spec) -> str:
    """The per-process identity of a BUILD REQUEST: every input the
    unit's content is a function of, hashed term by term. Per kernel:
    :attr:`hawk.ir.walk.Walk.digest`, its name and its declarations
    (``declarations``, :func:`declaration_digest`: the walk omits plane
    extents, which reach the sidecar, so one factory body built at two
    sizes has one walk and two units). Bundle-wide: the
    target list, scalar mode, :class:`hawk.ext.Kind` (by ``repr``),
    ``defines``, the RESOLVED arch (only when a target compiles for the device --
    calling the resolver at all may reach the device probe, which a
    host-only key must never pay for) and ``layout_sizes_override``; the
    ``aether_abi`` tag; ``derivative_blocks`` — the RESOLVED per-kernel
    sidecar blocks, hashed rather than the raw override mapping, since a
    primal may live in a different already-published unit named only by a
    digest that never touches the raw argument; ``cache_dir`` (so a
    cold-compile benchmark against an unused cache directory never answers
    from the wrong sample); the two header-root overrides (so re-pointing
    ``$HAWK_AETHER_INCLUDE`` mid-process never hands back a stale unit);
    the resolved host profile, so a ``portable`` build is never served the
    ``native`` unit (host-compiled targets only — the device compile ignores
    it); and the resolved opt level, UNCONDITIONALLY — unlike the host
    profile, it changes the AOT nvcc device compile too (``-O<n>`` /
    ``-Xptxas -O<n>``), so an ``"O0"`` request must never be served an
    ``"O3"`` unit regardless of which targets this bundle builds; and
    the distribution version of each built backend, then every user option of it (``<id>.<key>=<canonical text>``, :mod:`hawk.options`) before that backend's own terms; and
    the thread-box inputs (``$HAWK_BOX`` and the card's ``T`` for the scalar
    mode, :mod:`simt.box`), which change the emitted entry; and
    the terms a backend adds itself (:meth:`hawk.backends.Backend.key_terms`); and each kernel's compaction
    ``group`` (``groups``, hashed only when one exceeds 1: a grouped
    active-set map reads its index prologue differently).

    and the plain tables the emission modules hold (:func:`emission_digest`: spellings, text constants, name maps).

    STATED EXPOSURE, accepted rather than closed: the key names the
    render's INPUTS and its tables, not its code, so an emitter FUNCTION monkeypatched between
    two builds of the same kernel is invisible to it and would be served
    the first build's unit. :func:`reset_unit_memo` is the door out."""
    h = hashlib.sha256()
    h.update(b"hawk.artifact.unit-key/10\n")
    rows = [TARGETS[t] for t in targets]
    for part in (_contracts.AETHER_ABI_VERSION, smode.id, repr(kind),
                 "\x00".join(targets), "\x00".join(str(d) for d in defines),
                 arch(device_arch) if any(r.device for r in rows) else "",
                 str(cache_dir),
                 json.dumps(derivative_blocks, sort_keys=True, default=str),
                 repr(layout_sizes_override),
                 str(os.environ.get("HAWK_AETHER_INCLUDE")),
                 str(os.environ.get("HAWK_EAGLE_INCLUDE")),
                 ("host_profile:" + _host_profile(host_profile or None)
                  if any(r.host_compiled for r in rows) else ""),
                 "opt_level:" + _opt_level(opt_level or None),
                 "emit_form:" + emit_form if emit_form != "auto" else "",
                 "linalg_form:" + linalg_form if linalg_form != "auto" else "",
                 "wide_form:" + wide_form if wide_form != "auto" else "",
                 "launch_bounds:%d,%d" % launch_bounds if launch_bounds else "",
                 f"team_threads:{os.environ.get('HAWK_TEAM_THREADS')}",
                 "emission_tables:" + emission_digest(targets),
                 f"box:{os.environ.get(BOX_ENV)}:{(thread_box_card(smode.id, spec) or {}).get('T')}",
                 *(f"backend:{row.id}:{_backend_key(row, options or {}, source_digests, spec, smode, wide_class)}" for row in rows)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    if any(g > 1 for g in groups):
        h.update(("groups:" + ",".join(map(str, groups))).encode("ascii") + b"\0")
    if not len(names) == len(walk_digests) == len(declarations):
        raise ValueError("unit key: names, walk digests and declarations differ in length")
    for name, walk_digest, decl in zip(names, walk_digests, declarations):
        h.update(f"{name}\0{walk_digest}\0{decl}\0".encode("utf-8"))
    return h.hexdigest()


def declaration_digest(kernel) -> str:
    """The sha256 of ``kernel``'s resolved plane declarations (extents, launch
    shape, sortedness), by name: what its sidecar reads besides the walk."""
    planes = getattr(kernel, "planes", None) or {}
    return hashlib.sha256(repr(sorted(
        (name, p) for name, p in planes.items()
        if isinstance(p, Plane))).encode("utf-8")).hexdigest()


def _closure_watch(drafts) -> ClosureWatch:
    """Every target's validity record, de-duplicated across the unit.

    Pairs come from the artifact memo where it has them (already
    verified, nothing re-hashed), else derived from the closure. A
    two-target unit's ``.cu`` and ``.cpp`` reach most of the same
    headers, so checking one twice buys nothing."""
    seen, out = set(), []
    for draft in drafts:
        for result, _art, _src, _fmt in draft.entries.values():
            pairs = artifact_memo_get(result.key)
            if pairs is None:                                # pragma: no cover
                pairs = validity_digest(result.closure)
            for path, want in pairs:
                if path not in seen:
                    seen.add(path)
                    out.append((path, want))
    return ClosureWatch(out)
