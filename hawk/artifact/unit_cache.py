# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-process unit memo, the publish counters and log, the cross-process
digest stamp, and the unit key :func:`hawk.artifact.bundle.build_bundle`
memoises under."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from .. import _contracts
from ..compile import digest_file
from ..compile.cache import ClosureWatch, artifact_memo_get, validity_digest, write_atomic
from ..compile.toolchain import host_profile as _host_profile
from ..compile.toolchain import opt_level as _opt_level

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
    for the ONE resolver (:func:`hawk.compile.toolchain.resolve_arch`):
    ``device_arch=`` > ``$HAWK_CUDA_ARCH`` > the running GPU (probed once
    per process) > PTX for the toolkit's oldest architecture, each checked
    against what the toolkit compiles. Kept here, under the door callers already
    use (:mod:`hawk.artifact.bundle` and this module both called a function
    of this name before the resolver grew the env/probe/fallback chain).

    Call only where ``"cuda"`` is actually a build target: every call may
    reach the device probe, so a host-only build must never call this at
    all."""
    from ..compile.toolchain import resolve_arch

    return resolve_arch(device_arch)



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
def _unit_key(names, walk_digests, *, targets, smode, kind, defines, cache_dir,
              device_arch, derivative_blocks, layout_sizes_override,
              host_profile: str = "", opt_level: str = "") -> str:
    """The per-process identity of a BUILD REQUEST: every input the
    unit's content is a function of, hashed term by term. Per kernel:
    :attr:`hawk.ir.walk.Walk.digest` and its name. Bundle-wide: the
    target list, scalar mode, :class:`hawk.ext.Kind` (by ``repr``),
    ``defines``, the RESOLVED arch (only when ``"cuda"`` is a target --
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
    ``native`` unit (``host`` targets only — the device backend ignores
    it); and the resolved opt level, UNCONDITIONALLY — unlike the host
    profile, it changes the AOT nvcc device compile too (``-O<n>`` /
    ``-Xptxas -O<n>``), so an ``"O0"`` request must never be served an
    ``"O3"`` unit regardless of which targets this bundle builds.

    STATED EXPOSURE, accepted rather than closed: the key names the
    render's INPUTS, not its output, so an emitter monkeypatched between
    two builds of the same kernel is invisible to it and would be served
    the first build's unit. :func:`reset_unit_memo` is the door out."""
    h = hashlib.sha256()
    h.update(b"hawk.artifact.unit-key/5\n")
    for part in (_contracts.AETHER_ABI_VERSION, smode.id, repr(kind),
                 "\x00".join(targets), "\x00".join(str(d) for d in defines),
                 arch(device_arch) if "cuda" in targets else "", str(cache_dir),
                 json.dumps(derivative_blocks, sort_keys=True, default=str),
                 repr(layout_sizes_override),
                 str(os.environ.get("HAWK_AETHER_INCLUDE")),
                 str(os.environ.get("HAWK_EAGLE_INCLUDE")),
                 ("host_profile:" + _host_profile(host_profile or None)
                  if "host" in targets else ""),
                 "opt_level:" + _opt_level(opt_level or None)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    for name, walk_digest in zip(names, walk_digests):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(walk_digest).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


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
