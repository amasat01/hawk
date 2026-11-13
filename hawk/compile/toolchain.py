# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Include roots, compilers and the flag strings both drivers use.

One resolution of the two header roots, for every consumer: the emitted TUs,
the cache's flag string (a moved root is a key change), the sidecar's
recorded ``eagle_include``, and ``hawk/_core``'s compile line. The rule lives
in :mod:`hawk._roots`, a leaf module with no hawk imports, so
``hawk/CMakeLists.txt`` can load it by path without importing the hawk
package (whose self-check would import the extension CMake is building).
This module raises :class:`~hawk.HawkError` instead.

Nothing here is skipped when a tool is missing: an unresolvable root or
compiler raises and names the variable that would fix it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
from pathlib import Path

from .. import _roots
from .._errors import HawkError
from .cache import default_cache_dir

#: The workspace root the ``*-abi`` worktrees and ``aether`` hang off
#: (:data:`hawk._roots.WORKSPACE`, re-exported under its historical name).
WORKSPACE = _roots.WORKSPACE


def probe_stdout(compiler: str, args: tuple[str, ...], *, timeout: float,
                  persist: bool = True) -> tuple[int, str]:
    """``(returncode, stdout)`` of ``<compiler> <args>``, answered from the cache
    directory when this exact binary (path, size, mtime_ns) already answered the
    same arguments in ANY process: a toolchain probe is paid once per toolchain,
    never per process. A replaced binary (new size or mtime) is a new key, so a
    stale toolchain never reads an old answer. Only a returncode-0 answer is kept.
    A name resolved through ``PATH`` (no ``stat``) is not persisted."""
    path = None
    try:
        st = os.stat(compiler)
        key = hashlib.sha256("\0".join(
            [compiler, str(st.st_size), str(st.st_mtime_ns), *args]).encode()).hexdigest()[:24]
        path = default_cache_dir() / "toolchain" / f"{key}.txt"
        if persist:
            return 0, path.read_text()
    except OSError:
        pass
    out = subprocess.run([compiler, *args], capture_output=True, text=True, timeout=timeout,
                         check=False)
    if persist and path is not None and out.returncode == 0:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(out.stdout)
            os.replace(tmp, path)
        except OSError:
            pass
    return out.returncode, out.stdout


#: The bounded per-process memo behind :func:`compiler_identity`, keyed on
#: the compiler binary's ``(path, size, mtime_ns)``. Needed because identity
#: is asked for on every compile — including cache hits — and costs a
#: ``--version`` subprocess; per-process only, like the content memo in
#: :mod:`hawk.compile.cache`.
_IDENTITY_MEMO: dict = {}
_IDENTITY_MEMO_LIMIT = 32

#: Seconds :func:`compiler_identity` waits for ``<compiler> --version``.
#:
#: Must tolerate a COLD filesystem, not a slow compiler: on ZFS, the first
#: exec of a compiler binary after it falls out of page cache is pure I/O
#: and can take far longer than any warm run — a too-short timeout fails a
#: suite's first identity query with "cannot identify compiler", which reads
#: like a broken toolchain and is not one. Costs nothing ordinarily: the
#: answer is MEMOISED per process (:data:`_IDENTITY_MEMO`).
IDENTITY_TIMEOUT_S = 180


def _strict_aether_include() -> str:
    """The aether header root, or a :class:`~hawk.HawkError` when no real
    tree resolves (:func:`hawk._roots.aether_include`)."""
    return _resolve(_roots.aether_include)


def _strict_eagle_include() -> str:
    """The eagle header root, or a :class:`~hawk.HawkError` when no real
    tree resolves (:func:`hawk._roots.eagle_include`)."""
    return _resolve(_roots.eagle_include)


def aether_include() -> str:
    """The aether header root: the resolved tree, else the sealed payload's
    served root (a wheel install carries no C++ header tree), like
    :func:`include_flags`."""
    return _resolved_or_sealed(_strict_aether_include)


def eagle_include() -> str:
    """The eagle header root — the directory ``plugin/gref_abi.h`` hangs off:
    the resolved tree, else the sealed payload's served root, like
    :func:`include_flags`."""
    return _resolved_or_sealed(_strict_eagle_include)


def _resolve(fn):
    """Run one :mod:`hawk._roots` resolver, re-raising its stdlib-only
    :class:`~hawk._roots.RootError` as HAWK's own refusal type."""
    try:
        return fn()
    except _roots.RootError as exc:
        raise HawkError(str(exc)) from None


#: The one process-lifetime directory a pip-only compile serves the sealed
#: payload into when neither real root resolves — set by
#: :func:`_pip_only_include_root`, read by :func:`pip_only_root`. Lives for
#: the whole process (``atexit``-cleaned), unlike
#: :mod:`hawk.compile.drivers`'s per-compile fallback, since
#: :func:`include_flags` has no natural "done with it" moment.
_PIP_ONLY_ROOT: str | None = None

#: Serialises the one-time materialisation of :data:`_PIP_ONLY_ROOT`, so two
#: threads that both find it unset make ONE served directory and ONE ``atexit``
#: hook, not two of each.
_PIP_ONLY_LOCK = threading.Lock()


def pip_only_root() -> str | None:
    """The pip-only served root this process is using as a real-root
    fallback, or ``None`` if nothing has needed one yet. Only reports; never
    triggers the fallback itself (see :func:`_pip_only_include_root`)."""
    return _PIP_ONLY_ROOT


def _pip_only_include_root() -> str:
    """Materialise :func:`~hawk.compile.payload.current_payload` into one
    private directory for the rest of this process's life, and hand back its
    path — used as both the aether and the eagle root, since the payload's
    own header names (``aether/...`` and ``plugin/...``) already sit at the
    top of one served tree.

    Entered once (``atexit``-cleaned): a compile that needs this twice in one
    process reuses the first directory rather than paying `serve()`'s
    copy-to-tmpfs cost again."""
    global _PIP_ONLY_ROOT
    root = _PIP_ONLY_ROOT
    if root is not None:
        return root
    with _PIP_ONLY_LOCK:
        if _PIP_ONLY_ROOT is None:
            import atexit

            from . import payload as _payload_module
            served = _payload_module.current_payload().serve()
            path = served.__enter__()
            atexit.register(served.__exit__, None, None, None)
            _PIP_ONLY_ROOT = str(path)
        return _PIP_ONLY_ROOT


def _resolved_or_sealed(resolver) -> str:
    """``resolver()`` (a strict root resolver), or the pip-only served root
    when it raises: the one place a compiler needs an ``-I`` root and has
    nowhere else to get one. Sealing the payload and the provenance marker
    use the strict resolvers directly."""
    try:
        return resolver()
    except HawkError:
        return _pip_only_include_root()


def include_flags(*, aether_root: str | None = None,
                  eagle_root: str | None = None) -> list[str]:
    """``-I`` flags in (b)'s mandated order: aether first, eagle second.

    ``aether_root``/``eagle_root`` bypass the resolvers entirely when given:
    a caller that already knows its root (a sealed payload's served
    directory) must never call the root resolvers
    to confirm it, since a wheel-only install carries no C++ header tree for
    those resolvers to find. Omitted, this falls back to the sealed
    payload's own served directory for whichever root does not resolve."""
    resolved_aether = (aether_root if aether_root is not None
                      else _resolved_or_sealed(_strict_aether_include))
    resolved_eagle = (eagle_root if eagle_root is not None
                      else _resolved_or_sealed(_strict_eagle_include))
    return ["-I", resolved_aether, "-I", resolved_eagle]


def compile_hint(cache_dir: Path | str | None = None) -> str:
    """The tail every compile error ends with: where the build lives and how
    to pick another compiler."""
    return (f"\nbuild directory: {cache_dir or default_cache_dir()}; "
            "another host compiler: set HAWK_CXX=/path/to/c++")


def announce_build(kernel: str, target: str, cache_dir: Path | str | None = None) -> None:
    """The one INFO line (logger ``hawk``) a first build of ``kernel`` for
    ``target`` logs; silent unless the application enables INFO."""
    logging.getLogger("hawk").info(
        "building %s (%s) in %s — delete the directory to rebuild",
        kernel, target, cache_dir or default_cache_dir())


#: The optimisation levels a compile accepts, host or device alike.
OPT_LEVELS = ("O0", "O1", "O2", "O3")
#: The level used when neither the compile options nor ``$HAWK_OPT_LEVEL``
#: name one.
DEFAULT_OPT_LEVEL = "O3"


def opt_level(requested: str | None = None) -> str:
    """The optimisation level in effect: ``requested`` when given (the
    compile options' ``opt_level``), else ``$HAWK_OPT_LEVEL``, else
    :data:`DEFAULT_OPT_LEVEL`. An unknown name RAISES — mirrors
    :func:`host_profile`'s own resolution order exactly.

    Feeds the host build's ``-O<n>`` (:func:`host_codegen_flags`, replacing
    what used to be a hard-coded ``-O3``) and the AOT device build's
    ``-O<n>``/``-Xptxas -O<n>`` (:func:`device_flags`, nvcc only — there was
    no explicit ``-O`` there before, so the default is now spelled out
    rather than left to nvcc's own). NVRTC always optimises and takes no
    ``-O`` option at all: this level does not reach it (see
    :func:`hawk.compile.nvrtc.options`)."""
    name = requested or os.environ.get("HAWK_OPT_LEVEL") or DEFAULT_OPT_LEVEL
    if name not in OPT_LEVELS:
        raise HawkError(
            f"unknown opt level {name!r} (from "
            f"{'the compile options' if requested else '$HAWK_OPT_LEVEL'}); "
            f"the levels are {OPT_LEVELS}")
    return name


#: ``nvcc --version``'s own release line, e.g. ``"Cuda compilation tools,
#: release 12.6, V12.6.77"`` (see :func:`compiler_identity` for why the
#: version lives on this line and not the first).
_NVCC_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")

#: The memo behind :func:`nvcc_version`, keyed and bounded like
#: :data:`_IDENTITY_MEMO` — a separate dict because it holds a parsed
#: version tuple rather than an opaque digest string.
_NVCC_VERSION_MEMO: dict = {}


def nvcc_version(compiler: str) -> tuple[int, int] | None:
    """``(major, minor)`` parsed from ``<compiler> --version``'s "release"
    line, or ``None`` when the text carries no such line or the subprocess
    fails — an nvcc whose format this does not recognise is an unknown
    version to the caller (:func:`hawk.compile.nvrtc.select_target`'s own
    conservative answer), not a guess, and the real compile attempt right
    after this is what reports an actual failure loudly."""
    memo_key = None
    try:
        st = os.stat(compiler)
        memo_key = (compiler, st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    if memo_key is not None and memo_key in _NVCC_VERSION_MEMO:
        return _NVCC_VERSION_MEMO[memo_key]
    try:
        out = probe_stdout(compiler, ("--version",), timeout=IDENTITY_TIMEOUT_S)[1]
    except (OSError, subprocess.SubprocessError):
        return None
    m = _NVCC_RELEASE_RE.search(out)
    version = (int(m.group(1)), int(m.group(2))) if m else None
    if memo_key is not None:
        if len(_NVCC_VERSION_MEMO) >= _IDENTITY_MEMO_LIMIT:
            _NVCC_VERSION_MEMO.clear()
        _NVCC_VERSION_MEMO[memo_key] = version
    return version


def compiler_identity(compiler: str) -> str:
    """``<path> || <first --version line> || <digest of the whole --version
    text>`` — the compiler term of the lookup key. A binary/version change
    is a key change.

    The digest, not just the first line, is what makes that true: nvcc's
    first ``--version`` line is version-invariant (the actual release lives
    on a later line), so keying on the first line alone gave two different
    nvcc releases the same identity, and the cache served one version's PTX
    to the other's run. Hashing the full text catches that line for nvcc and
    whatever line any other compiler puts its version on, without special-
    casing the word "release".

    Answered from :data:`_IDENTITY_MEMO` when this process has already asked
    about the same binary at the same size and mtime."""
    memo_key = None
    try:
        st = os.stat(compiler)
        memo_key = (compiler, st.st_size, st.st_mtime_ns)
    except OSError:
        pass                       # a name resolved through PATH, or gone: no memo
    if memo_key is not None:
        hit = _IDENTITY_MEMO.get(memo_key)
        if hit is not None:
            return hit
    try:
        out = probe_stdout(compiler, ("--version",), timeout=IDENTITY_TIMEOUT_S)[1]
    except (OSError, subprocess.SubprocessError) as exc:   # pragma: no cover
        raise HawkError(f"cannot identify compiler {compiler!r}: {exc}") from exc
    first = out.splitlines()[0].strip() if out.strip() else ""
    version_digest = hashlib.sha256(out.encode()).hexdigest()[:16]
    identity = f"{compiler}||{first}||{version_digest}"
    if memo_key is not None:
        if len(_IDENTITY_MEMO) >= _IDENTITY_MEMO_LIMIT:  # pragma: no cover - bound
            _IDENTITY_MEMO.clear()
        _IDENTITY_MEMO[memo_key] = identity
    return identity


def reset_identity_memo() -> None:
    """Forget every memoised compiler identity (:data:`_IDENTITY_MEMO`) AND
    every memoised nvcc version (:data:`_NVCC_VERSION_MEMO`) — a compiler
    binary swapped mid-process should invalidate both alike."""
    _IDENTITY_MEMO.clear()
    _NVCC_VERSION_MEMO.clear()


def subprocess_env(targets: tuple = ()) -> dict:
    """The environment a driver subprocess runs under: this process's, as each of the ``targets`` being built adjusts it
    (:meth:`hawk.backends.Backend.process_env`), for those whose manifest declares ``process_env``. No other backend is imported."""
    from .. import registry
    from ..backends import need

    env = dict(os.environ)
    for target in targets:
        if registry.has("backends", target) and registry.traits(target).get("process_env"):
            env = need(target).process_env(env)
    return env
