# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The content-closure compile cache.

Two objects. The **lookup key** is ``H(source text || backend id || scalar
mode || compiler identity + full flag string)`` — every term is known
before any compile. Stored under that key is the **validity record**: the
compiler-reported dependency closure of the last compile (``-MD``/``-MMD``
output) as ``(path, content digest)`` pairs, beside the artifact itself. A
lookup with no entry is a miss; an entry whose closure still re-hashes
identically by content is a hit; anything else is a miss that recompiles
and rewrites the record.

Content, never mtime, decides: mtime appears in neither the key nor the
record, so touching a reached header is a hit, and the whole aether tree
is never walked (fingerprinting every header on every miss would grow
with a library HAWK does not control).

Stated exposure, accepted rather than closed: a ``-MD`` closure records
only the files that were opened, so a new file shadowing a recorded
header earlier on the include path changes the compile while every
recorded file's content stays unchanged — a false hit. The realistic
trigger is an include-path change, which is already a flag (key) change.
A test asserts the case explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

# `import hawk.compile.cache` cannot run before `hawk/__init__` has
# finished, so the self-check has already refused a stale/wrong-arch/
# editable-shadowed `_core` by the time this line runs — which is why
# `ClosureWatch` below calls `_core.stat_many` with no Python fallback.
from .. import _core

#: The ``(path, want)`` pair sentinel an NVRTC-compiled entry's "closure"
#: uses in place of a real file: ``want`` is the sealed payload's own
#: digest, not a file's content digest. Shared between
#: :mod:`hawk.compile.drivers` (which builds it) and :class:`ClosureWatch`
#: below (which must recognise it rather than ``os.stat``-ing a name that
#: was never a file).
PAYLOAD_CLOSURE_TAG = "__hawk_nvrtc_payload__"

#: Per-process closure-verdict memo: once :meth:`ClosureWatch.valid` has
#: verified a closure by content in this process, a later ask about the
#: same closure (keyed on its sorted ``(path, want)`` pairs, never on
#: ``id(self)``) is trusted rather than re-stat'd. Cleared by
#: :func:`reset_artifact_memo`. A header edited between two checks of the
#: same closure within one process is not seen until that reset runs.
_CLOSURE_VERDICT_MEMO: dict[str, bool] = {}
_CLOSURE_VERDICT_MEMO_LIMIT = 256

#: Hit/miss counters, record-only — row reads them to say WHICH answer the
#: cache gave, rather than inferring it from a wall time.
_STATS = {"hits": 0, "misses": 0}

#: Guards the counters and memos above and below: hawk builds the host and
#: device targets of one bundle on two threads.
_LOCK = threading.Lock()

#: The bounded per-process content memo. Keyed on the OS metadata that
#: makes a re-read pointless within one process, valued the content digest.
_MEMO: dict[tuple[str, int, int], str] = {}
_MEMO_LIMIT = 4096

#: The bounded per-process artifact memo, keyed by the lookup key. A
#: lookup already answered in this process still pays for the whole
#: answer again (record read and JSON-parsed twice, slot re-made, source
#: re-written) for nothing: the key is a pure function of the
#: source/backend/mode/compiler. It does not short-circuit the validity
#: record: :func:`hawk.compile.drivers.compile_source` re-verifies every
#: stored pair by content on a memo hit, through the same
#: :func:`closure_unchanged` the on-disk record uses — the memo replaces
#: the two JSON reads, never the check they carry.
_ARTIFACT_MEMO: dict[str, tuple] = {}
_ARTIFACT_MEMO_LIMIT = 256


def artifact_memo_get(key: str):
    """The stored ``(artifact, source, closure pairs)`` for ``key``, or ``None``."""
    return _ARTIFACT_MEMO.get(key)


def artifact_memo_put(key: str, payload: tuple) -> None:
    """Remember one built artifact under its lookup key (bounded)."""
    with _LOCK:
        if len(_ARTIFACT_MEMO) >= _ARTIFACT_MEMO_LIMIT:
            _ARTIFACT_MEMO.clear()
        _ARTIFACT_MEMO[key] = payload


def reset_artifact_memo() -> None:
    """Forget every memoised artifact and closure verdict (an honest re-check)."""
    with _LOCK:
        _ARTIFACT_MEMO.clear()
        _CLOSURE_VERDICT_MEMO.clear()


def cache_stats() -> dict:
    """A copy of the hit/miss counters."""
    with _LOCK:
        return dict(_STATS)


def reset_cache_stats() -> None:
    """Reset the counters (cached artifacts on disk are untouched)."""
    with _LOCK:
        _STATS["hits"] = 0
        _STATS["misses"] = 0


def default_cache_dir() -> Path:
    """``$HAWK_CACHE_DIR``, else ``$XDG_CACHE_HOME/hawk``, else ``~/.cache/hawk``."""
    override = os.environ.get("HAWK_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "hawk"


def _hash_bytes(path: str) -> str | None:
    """THE content rule, in one place: sha256 of a file's bytes, or
    ``None`` — the one answer every reader below goes through."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def digest_file(path: str | os.PathLike) -> str | None:
    """The sha256 of ``path``'s bytes, or ``None`` if it cannot be read —
    a miss at every call site, never an ignored entry."""
    p = os.fspath(path)
    try:
        st = os.stat(p)
    except OSError:
        return None
    memo_key = (p, st.st_size, st.st_mtime_ns)
    hit = _MEMO.get(memo_key)
    if hit is not None:
        return hit
    digest = _hash_bytes(p)
    if digest is None:
        return None
    with _LOCK:
        if len(_MEMO) >= _MEMO_LIMIT:
            _MEMO.clear()
        _MEMO[memo_key] = digest
    return digest


def closure_unchanged(pairs) -> bool:
    """Is every ``(path, content digest)`` pair still true.

    The same verdict :func:`digest_file` would give pair by pair —
    content, never mtime — written as one inlined loop rather than a
    comprehension, since on a warm hit this is Python-call-bound, not
    filesystem-bound. Short-circuits on the first disagreement, so a
    changed header is reported without stat'ing the rest."""
    memo = _MEMO
    stat = os.stat
    for path, want in pairs:
        try:
            st = stat(path)
        except OSError:
            return False                      # missing or unreadable => a miss
        got = memo.get((path, st.st_size, st.st_mtime_ns))
        if got is None:
            got = digest_file(path)
        if got != want:
            return False
    return True


def _closure_digest(pairs) -> str:
    """The closure's own content identity: sha256 of its sorted ``(path,
    want)`` pairs — order-independent, so any :class:`ClosureWatch`
    wrapping the same recorded closure shares one verdict slot."""
    h = hashlib.sha256()
    h.update(b"hawk.compile.cache/closure-verdict/1\n")
    for path, want in sorted(pairs):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update((want or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class ClosureWatch:
    """One compile's validity record, shaped for a cheap repeated re-check.

    Same rule as :func:`closure_unchanged` (content, with a ``(size,
    mtime_ns)`` shortcut), but the stat pass itself is one
    ``hawk._core.stat_many`` crossing over the whole closure instead of
    one Python ``os.stat`` per row, since :meth:`valid` is the one
    re-verification the warm path pays on every re-authored kernel in the
    same process. ``_paths`` is built once in :meth:`__init__` and reused
    on every later :meth:`valid` call; a fall-through content check
    re-syncs the record's ``(size, mtime_ns)`` so a touched header is
    paid for once.

    :meth:`valid` also wraps the stat pass in a per-process verdict memo
    keyed on the closure's own content (:func:`_closure_digest`): the
    first call for a given closure pays the crossing above, and every
    later call on the same closure is trusted without one. See
    :data:`_CLOSURE_VERDICT_MEMO` for the exposure this trades for, and
    :func:`reset_artifact_memo` for the door back to an honest check."""

    __slots__ = ("_rows", "_paths", "_digest", "_payload_digest")

    def __init__(self, pairs) -> None:
        rows = []
        payload_digest = None
        for path, want in pairs:
            if path == PAYLOAD_CLOSURE_TAG:
                # An NVRTC entry's pair is not a real file: `want` IS the
                # payload digest it was compiled against. Recorded
                # separately, never stat'd.
                payload_digest = want
                continue
            try:
                st = os.stat(path)
                rows.append([path, st.st_size, st.st_mtime_ns, want])
            except OSError:
                # Unreadable NOW: recorded with a signature nothing can match,
                # so the check falls through to the content and says "miss".
                rows.append([path, -1, -1, want])
        self._rows = rows
        # The stable object `valid()` hands to `_core.stat_many` on every
        # later check, built once from the rows' own path objects.
        self._paths = [row[0] for row in rows]
        self._digest = _closure_digest(pairs)
        self._payload_digest = payload_digest

    def __len__(self) -> int:
        return len(self._rows) + (1 if self._payload_digest is not None else 0)

    def pairs(self) -> tuple:
        """The ``(path, content digest)`` record, for a reader or a diagnostic."""
        out = tuple((row[0], row[3]) for row in self._rows)
        if self._payload_digest is not None:
            out = ((PAYLOAD_CLOSURE_TAG, self._payload_digest),) + out
        return out

    def valid(self, *, payload_digest: str | None = None) -> bool:
        """Whether every recorded file is still the content it was
        recorded as — the first call pays one ``_core.stat_many``
        crossing; a later call on the same closure content is served
        from the per-process verdict memo (:data:`_CLOSURE_VERDICT_MEMO`)
        until :func:`reset_artifact_memo`.

        A watch carrying a payload sentinel (see :data:`PAYLOAD_CLOSURE_TAG`)
        is valid iff the caller's ``payload_digest`` equals the recorded
        one — a plain string compare, not routed through the verdict memo
        (memoising on ``self._digest`` alone would ignore a changed
        ``payload_digest`` between calls). A watch mixing a payload
        sentinel with real file rows requires both to hold."""
        if self._payload_digest is not None:
            if payload_digest is None or payload_digest != self._payload_digest:
                return False
            if not self._rows:
                return True
            # fall through: real rows mixed into the same watch still need
            # their own by-content check below.
        cached = _CLOSURE_VERDICT_MEMO.get(self._digest)
        if cached is not None:
            return cached
        verdict = self._check()
        with _LOCK:
            if len(_CLOSURE_VERDICT_MEMO) >= _CLOSURE_VERDICT_MEMO_LIMIT:
                _CLOSURE_VERDICT_MEMO.clear()
            _CLOSURE_VERDICT_MEMO[self._digest] = verdict
        return verdict

    def _check(self) -> bool:
        """The real, by-content check (the verdict memo wraps this; the
        microbench calls it directly to keep timing the crossing itself)."""
        sizes, mtimes_ns = _core.stat_many(self._paths)
        rows = self._rows
        for i in range(len(rows)):
            row = rows[i]
            size, mtime = sizes[i], mtimes_ns[i]
            if size < 0:
                return False               # missing or unreadable => a miss
            if size == row[1] and mtime == row[2]:
                continue
            if digest_file(row[0]) != row[3]:
                return False
            row[1], row[2] = size, mtime
        return True


def tmp_sibling(path: Path) -> Path:
    """A private temporary name beside ``path`` in the same slot
    (``<slot>/.tmp.<pid>.<rand><suffix>``): written in full, then moved
    into place with :func:`publish_atomic`. Same directory, so the final
    ``os.replace`` is a rename on one filesystem -- atomic."""
    return path.parent / f".tmp.{os.getpid()}.{secrets.token_hex(8)}{path.suffix}"


def publish_atomic(tmp: Path, path: Path) -> None:
    """Move a fully written ``tmp`` (:func:`tmp_sibling`) onto ``path``:
    a concurrent reader sees the old file or the new one, never a partial
    write."""
    os.replace(tmp, path)


def write_atomic(path: Path, data: bytes | str) -> None:
    """Write ``data`` to ``path`` through a :func:`tmp_sibling` and
    :func:`publish_atomic`; the temporary is removed on failure."""
    tmp = tmp_sibling(path)
    try:
        if isinstance(data, str):
            tmp.write_text(data)
        else:
            tmp.write_bytes(data)
        publish_atomic(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def lookup_key(source: str, backend: str, mode: str, compiler: str,
               flags) -> str:
    """The lookup key. Every term is available before any compile; the two
    resolved header roots ride inside ``flags`` (``-I`` entries), so
    moving either root is a key change, not a validity question."""
    h = hashlib.sha256()
    h.update(b"hawk.compile.cache/1\n")
    for part in (source, backend, mode, compiler, "\x00".join(str(f) for f in flags)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def closure_of(depfile: str | os.PathLike) -> tuple[str, ...]:
    """Parse a ``-MD``/``-MMD`` makefile fragment into its dependency
    paths. The compiler reports the files it opened; HAWK never guesses a
    closure or walks a header tree to approximate one."""
    text = Path(depfile).read_text()
    text = text.replace("\\\n", " ")
    paths: list[str] = []
    for line in text.splitlines():
        body = line.split(":", 1)[1] if ":" in line else line
        paths += [tok for tok in body.split() if tok and not tok.endswith(":")]
    seen, out = set(), []
    for p in paths:
        real = os.path.abspath(p)
        if real not in seen:
            seen.add(real)
            out.append(real)
    return tuple(out)


def validity_digest(closure) -> tuple[tuple[str, str | None], ...]:
    """The validity record for ``closure``: ``(path, content digest)``
    pairs, in closure order."""
    return tuple((p, digest_file(p)) for p in closure)


@dataclass(frozen=True)
class CacheEntry:
    """One cached compile: where its files live and whether the lookup HIT."""

    key: str
    directory: Path
    artifact: Path
    source: Path
    hit: bool


class Cache:
    """A content-closure cache rooted at ``directory``: the compiled
    object, its emitted source and its validity record all live under the
    lookup key, and a previous generation is never evicted implicitly."""

    def __init__(self, directory: str | os.PathLike | None = None) -> None:
        self.root = Path(directory) if directory is not None else default_cache_dir()

    def slot(self, key: str) -> Path:
        return self.root / key[:2] / key

    def check(self, key: str, *, payload_digest: str | None = None) -> bool:
        """Is the entry at ``key`` valid? Absent record or missing
        artifact => ``False``.

        Two validity rules, by what the stored record carries: no
        ``payload_digest`` field (the nvcc/g++ path) re-hashes the
        recorded closure by content; a ``payload_digest`` field (an
        NVRTC-compiled entry) is valid iff it matches the caller's own —
        one string compare, no stat at all. A caller that omits
        ``payload_digest`` gets ``False`` rather than trusting an empty
        closure list — a miss is always the safe default."""
        slot = self.slot(key)
        record = slot / "validity.json"
        if not record.is_file():
            return False
        try:
            data = json.loads(record.read_text())
        except (OSError, ValueError):                        # pragma: no cover
            return False
        artifact = slot / data.get("artifact", "")
        if not artifact.is_file():
            return False
        recorded_payload_digest = data.get("payload_digest")
        if recorded_payload_digest is not None:
            return (payload_digest is not None
                    and recorded_payload_digest == payload_digest)
        return closure_unchanged(data.get("closure", ()))

    def store(self, key: str, *, artifact: Path, closure, meta: dict) -> None:
        """Write the validity record for ``key`` beside the artifact,
        atomically (:func:`write_atomic`): written AFTER the artifact is in
        place, so a reader that finds a record finds a whole artifact."""
        slot = self.slot(key)
        slot.mkdir(parents=True, exist_ok=True)
        write_atomic(slot / "validity.json", json.dumps({
            "artifact": artifact.name,
            "closure": [list(pair) for pair in validity_digest(closure)],
            **meta,
        }, indent=2))

    def record(self, key: str) -> dict:
        """The stored validity record (for a diagnostic or a row's assertion)."""
        return json.loads((self.slot(key) / "validity.json").read_text())

    @staticmethod
    def note(hit: bool) -> None:
        with _LOCK:
            _STATS["hits" if hit else "misses"] += 1
