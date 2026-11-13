# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The compile drivers, both through the content-closure cache.

``g++`` builds the host ``.so`` (flags from one host profile, see
:data:`osprey_cpu.toolchain.HOST_PROFILES`). The device toolchain has two
compilers, chosen by :func:`hawk.compile.toolchain.device_compiler_kind`:
``nvcc`` ahead of time on the developer/CI tree (the only path for
LTO/relocatable fatbins), or an in-memory NVRTC compile
(:mod:`hawk.compile.nvrtc`) on the shipped Python tier, serving the sealed
:mod:`aether_dsc` payload as NVRTC's header array — NVRTC has no host
stdlib, so the payload supplies the seven std headers it reaches as
``cuda::std`` shims instead.

The nvcc/g++ path asks the compiler for its dependency closure (``-MD -MF``)
and hands it to :class:`~hawk.compile.cache.Cache` as the compile's validity
record, so the next lookup is decided by header content, never a walk of a
tree HAWK does not own. The NVRTC path has no such closure to discover: the
payload's digest IS the closure, known before the compile runs, so
re-validating a hit is one string compare, never a stat walk.

A driver is a pure function of its inputs: source, target, scalar mode,
compiler and flags produce the same key and slot.

Every compiler subprocess runs under an address-space cap — a machine
guard, not a codegen policy, so a pathological compile fails loudly instead
of taking the box down (a 64x23 spline cell once reached ptxas as one function
and took 25-27 GB of RSS on a 31 GB machine, with nothing bounding it).
:data:`ADDRESS_SPACE_CAP_BYTES` is that bound, applied with ``RLIMIT_AS`` in
the child before the compiler ``exec``s; ``$HAWK_COMPILE_ADDRESS_CAP`` overrides it
(bytes; ``0`` disables it).

Two more guards keep a compile from running away unseen: a translation unit
over :data:`SOURCE_BYTES_BUDGET` is refused BEFORE it is compiled
(``$HAWK_COMPILE_SOURCE_BUDGET``, bytes, ``0`` disables), because compile time
grows with the source (about 20-30 KB/s on the device compiler), and the
compiler subprocess gets a wall-clock timeout, :data:`COMPILE_TIMEOUT_SECONDS`
(``$HAWK_COMPILE_TIMEOUT``, seconds, ``0`` disables).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from .._errors import HawkError
from . import payload as _payload
from . import toolchain as tc
from .cache import (
    PAYLOAD_CLOSURE_TAG as _PAYLOAD_MEMO_TAG,
)
from .cache import (
    Cache,
    artifact_memo_get,
    artifact_memo_put,
    closure_of,
    closure_unchanged,
    lookup_key,
    publish_atomic,
    tmp_sibling,
    validity_digest,
    write_atomic,
)

#: The default ceiling on one compiler subprocess's address space, in bytes
#: (12 GiB) — chosen against the 31 GB machine HAWK is developed on, where a
#: single compile wanting a third of it has stopped being a build and
#: started being an outage. A machine fact, not a HAWK one: see this
#: module's docstring for why the guard lives here and not in the emitter.
ADDRESS_SPACE_CAP_BYTES = 12 * 1024 * 1024 * 1024


def address_space_cap() -> int:
    """The cap this process applies, in bytes; ``0`` means unbounded."""
    raw = os.environ.get("HAWK_COMPILE_ADDRESS_CAP")
    if raw is None or raw.strip() == "":
        return ADDRESS_SPACE_CAP_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise HawkError(
            f"$HAWK_COMPILE_ADDRESS_CAP={raw!r} is not an integer number of bytes "
            "(0 disables the compiler address-space cap)") from None
    if value < 0:
        raise HawkError(
            f"$HAWK_COMPILE_ADDRESS_CAP={value} is negative; it is a byte count, "
            "and 0 disables the cap")
    return value


#: The trampoline's program: lower ``RLIMIT_AS`` to the cap in argv[1], then
#: ``exec`` the real command (argv[2:]). It runs in a fresh interpreter
#: (``-I``), so no thread state of the calling process is involved.
_TRAMPOLINE = (
    "import os, resource, sys\n"
    "cap = int(sys.argv[1])\n"
    "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
    "limit = cap if hard == resource.RLIM_INFINITY else min(cap, hard)\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))\n"
    "except (ValueError, OSError):\n"
    "    pass  # macOS cannot cap the address space; the timeout and source budget still bound the compile\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)


#: The POSIX trampoline: lower the SOFT ``RLIMIT_AS`` to the cap (KiB, ``$1``)
#: with the shell's ``ulimit -S -v``, then ``exec`` the real command. A failed
#: ``ulimit`` (a cap above the inherited hard limit, or a platform that cannot
#: cap the address space) leaves the limit as it was and the command still runs.
#: No second interpreter starts: ``sh`` costs about a millisecond, the Python
#: trampoline above about 20 ms per compile.
_SH_TRAMPOLINE = 'ulimit -S -v "$1" 2>/dev/null; shift; exec "$@"'

#: The largest translation unit, in bytes, a driver compiles. A kernel whose
#: source is bigger than this is too wide for one compile (a 3.6 MB source took
#: 163 s on the device compiler); split it or narrow it. A machine guard like
#: :data:`ADDRESS_SPACE_CAP_BYTES`, not a codegen policy.
SOURCE_BYTES_BUDGET = 8 * 1024 * 1024

#: The wall-clock limit, in seconds, on one compiler subprocess.
COMPILE_TIMEOUT_SECONDS = 1800


def _env_number(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise HawkError(f"${name}={raw!r} is not a non-negative integer (0 disables it)")
    return value


def check_source_budget(name: str, source: str) -> None:
    """Refuse a translation unit over the byte budget, before any compile."""
    budget = _env_number("HAWK_COMPILE_SOURCE_BUDGET", SOURCE_BYTES_BUDGET)
    size = len(source.encode("utf-8"))
    if budget and size > budget:
        raise HawkError(
            f"the translation unit of {name!r} is {size} bytes, over the compile "
            f"budget of {budget} bytes ($HAWK_COMPILE_SOURCE_BUDGET, 0 disables): "
            "a source this large is too wide for one compile. Narrow the kernel "
            "(roll an unrolled loop, read a wide plane by runtime index) or raise "
            "the budget for one build")


def run_compiler(argv: list[str], what: str, target: str) -> subprocess.CompletedProcess:
    """Run one compiler command for the ``target`` row (its :meth:`~hawk.backends.Backend.process_env` applies) under the
    address-space cap and the wall-clock timeout; a timeout raises a :class:`HawkError` that names the limit."""
    limit = _env_number("HAWK_COMPILE_TIMEOUT", COMPILE_TIMEOUT_SECONDS) or None
    try:
        return subprocess.run(limited(argv), capture_output=True, text=True,
                              env=tc.subprocess_env((target,)), timeout=limit)
    except subprocess.TimeoutExpired:
        raise HawkError(
            f"{what} did not finish within {limit} s ($HAWK_COMPILE_TIMEOUT, "
            "0 disables): the translation unit is too large for one compile") from None


def limited(argv: list[str]) -> list[str]:
    """``argv`` wrapped so the child runs under the address-space cap, or
    ``argv`` itself when the cap is disabled.

    The cap is installed by a small exec trampoline (``sh -c 'ulimit -S -v N;
    exec ...'``; ``python -I -c`` with ``resource.setrlimit`` where there is
    no ``/bin/sh``) rather than a fork-time hook (``preexec_fn``), which is
    unsafe to run from a multi-threaded process.
    The limit is inherited by every process the compiler driver spawns —
    ``nvcc`` runs ``cicc`` and ``ptxas``, and ``ptxas`` is the one that ran
    the machine out of memory. The soft limit is raised no higher than the
    inherited hard limit."""
    cap = address_space_cap()
    if not cap:
        return list(argv)
    if os.path.exists("/bin/sh"):
        return ["/bin/sh", "-c", _SH_TRAMPOLINE, "sh", str(-(-cap // 1024)), *argv]
    return [sys.executable, "-I", "-c", _TRAMPOLINE, str(cap), *argv]


@dataclass(frozen=True)
class CompileOptions:
    """Everything outside the source text that changes the output — and so
    every term of the LOOKUP key beyond the source itself."""

    #: The id of the :data:`hawk.targets.TARGETS` row to compile for.
    target: str
    mode: str = "float64"
    arch: str = ""
    defines: tuple[str, ...] = ()
    cache_dir: str | None = None
    extra: tuple[str, ...] = ()
    #: The host code-generation profile
    #: (:data:`osprey_cpu.toolchain.HOST_PROFILES`). Empty means
    #: ``$HAWK_HOST_PROFILE``, else
    #: :func:`~osprey_cpu.toolchain.default_host_profile`. Ignored by the
    #: device compile.
    host_profile: str = ""
    #: The optimisation level (:data:`hawk.compile.toolchain.OPT_LEVELS`).
    #: Empty means ``$HAWK_OPT_LEVEL``, else
    #: :data:`~hawk.compile.toolchain.DEFAULT_OPT_LEVEL`. Used by the host
    #: host compile's ``-O<n>`` and the device AOT nvcc compile
    #: (``-O<n>`` / ``-Xptxas -O<n>``); ignored by the NVRTC device compile,
    #: which always optimises and takes no ``-O`` option.
    opt_level: str = ""


@dataclass(frozen=True)
class CompileResult:
    """One compiled TU: where it landed, whether the cache HIT, and — on a
    miss — how long the compiler took (the card's raw material)."""

    artifact: Path
    source: Path
    key: str
    hit: bool
    argv: tuple[str, ...]
    seconds: float = 0.0
    closure: tuple[str, ...] = field(default=())
    #: ``{kernel: {regs, spill_store_B, spill_load_B, stack_B}}`` ptxas reported
    #: for a device compile that ran ptxas (a CUBIN); ``{}`` otherwise.
    ptxas: dict = field(default_factory=dict)


#: Guards :data:`_INFLIGHT`.
_LOCK = threading.Lock()
#: Lookup key -> the event the thread compiling that key sets when it is done.
_INFLIGHT: dict[str, threading.Event] = {}


def single_flight(key: str, lookup, build) -> CompileResult:
    """In-process single-flight per lookup key: ``lookup()`` returns a hit's
    :class:`CompileResult` or ``None``; ``build()`` compiles. The first
    thread to miss a key builds; a second thread asking for the same key
    waits for it and then takes the hit (if the build failed it retries as
    the builder itself, so it raises the same refusal). There is no
    cross-process lock: the publish is an atomic rename, so a duplicate
    across processes costs one compile and nothing else."""
    while True:
        found = lookup()
        if found is not None:
            return found
        with _LOCK:
            event = _INFLIGHT.get(key)
            leader = event is None
            if leader:
                event = _INFLIGHT[key] = threading.Event()
        if not leader:
            event.wait()
            continue
        try:
            found = lookup()               # a build may have landed since the first look
            return found if found is not None else build()
        finally:
            with _LOCK:
                _INFLIGHT.pop(key, None)
            event.set()


def define_args(opts: CompileOptions) -> list[str]:
    """``-D`` arguments of ``opts.defines``."""
    return [f"-D{d}" for d in opts.defines]


def _argv(compiler: str, flags, src: Path, out: Path, dep: Path) -> list[str]:
    return [compiler, *flags, "-MD", "-MF", str(dep), str(src), "-o", str(out)]


def key_flags(flags) -> tuple:
    """``flags`` as the compile key reads them: a flag that carries a ``key_term`` (the variant's include directory:
    :class:`hawk.variants.IncludeFlag`) stands for it, so the key does not depend on where that directory is. The compile
    line itself keeps the flag."""
    return tuple(getattr(f, "key_term", "") or f for f in flags)


def _stable_served(flags) -> tuple[tuple, tuple]:
    """``flags`` with the pip-only served ``-I`` root replaced by the sealed
    payload's digest, and that root's path prefixes (empty when this process
    serves none)."""
    flags = key_flags(flags)
    root = tc.pip_only_root()
    if root is None:
        return tuple(flags), ()
    prefixes = tuple({os.path.abspath(root) + os.sep, os.path.realpath(root) + os.sep})
    term = f"payload:{_payload.current_payload().digest}"
    return tuple(term if f == root else f for f in flags), prefixes


def _compiler_not_runnable(argv0: str, returncode: int) -> bool:
    """True when ``argv0`` was never a running compiler: the shell's 127
    (not found) or 126 (not executable), or an argv[0] that is no executable
    file on PATH."""
    return returncode in (126, 127) or shutil.which(argv0) is None


def _allocation_failure(stderr: str) -> bool:
    """True when the compiler's stderr reads like an out-of-memory failure. Asked
    only under a cap, where a compiler killed by a signal is the cap too: g++
    may crash in cc1plus rather than report the failed allocation."""
    text = stderr.lower()
    return any(m in text for m in ("bad_alloc", "cannot allocate memory",
                                   "out of memory", "unable to allocate",
                                   "memory exhausted", "failed to map segment",
                                   "signal terminated program"))


def _run_and_publish(argv, label, failed_tail, tmp_artifact, artifact, tmp_dep, dep,
                     cache_root, target):
    """Run the compiler into the private temporaries and move the artifact and
    its dependency file into place; ``(done, seconds)``, or a :class:`HawkError`
    carrying the compiler's stderr when it failed."""
    start = time.perf_counter()
    try:
        done = run_compiler(argv, label, target)
        if done.returncode == 0:
            publish_atomic(tmp_artifact, artifact)
            publish_atomic(tmp_dep, dep)
    finally:
        tmp_artifact.unlink(missing_ok=True)
        tmp_dep.unlink(missing_ok=True)
    seconds = time.perf_counter() - start
    if done.returncode != 0:
        _refuse(done, argv, label, failed_tail, cache_root)
    return done, seconds


def _refuse(done, argv, label: str, failed_tail: str, cache_root) -> None:
    """Raise the refusal of a failed compiler command: its stderr, the reason (the cap, a missing compiler) and the hint."""
    missing = _compiler_not_runnable(argv[0], done.returncode)
    cap = address_space_cap()
    if cap and _allocation_failure(done.stderr):
        reason = (f"; the compiler ran under a {cap}-byte address-space cap "
                  "($HAWK_COMPILE_ADDRESS_CAP), so an allocation failure here means "
                  "this translation unit is too large for one compile and not that "
                  "the machine is out of memory")
    elif missing:
        reason = (f"; compiler not found or not executable: {argv[0]} "
                  f"(exit {done.returncode}); point $HAWK_CXX at a working "
                  "C++ compiler, e.g. HAWK_CXX=/path/to/c++")
    else:
        reason = ""
    raise HawkError(
        f"{label} failed{failed_tail} (rc={done.returncode}){reason}"
        + f".\n$ {' '.join(map(str, argv))}\n{done.stderr[:4000]}"
        + tc.compile_hint(cache_root)
    )


def compile_source(source: str, name: str, opts: CompileOptions) -> CompileResult:
    """Compile ``source`` for the ``opts.target`` row, through the content-closure
    cache: a valid lookup is a hit and no compiler runs; anything else is a
    miss that compiles into the key's own slot and rewrites its validity
    record from the compiler-reported closure.

    A backend with a plan (:meth:`hawk.backends.Backend.compile_plan`) is compiled by :func:`_drive`;
    one without still compiles itself (:meth:`hawk.backends.Backend.compile`, through :func:`compile_cached`).
    """
    from ..registry import backend_ids, traits
    from ..targets import TARGETS

    row = TARGETS.get(opts.target)
    if row is None or not row.builds:
        built = tuple(t for t in backend_ids(default=True) if traits(t).get("builds", True))
        raise HawkError(f"unknown target {opts.target!r}; built are {built}")
    plan = row.compile_plan(opts, _Inputs())
    if plan is None:
        return row.compile(source, name, opts)
    return _drive(source, name, opts, row, plan)


def compile_cached(source: str, name: str, opts: CompileOptions, row, *, compiler: str,
                   argv_flags, pick_target, terms, parse) -> CompileResult:
    """The cached compile of ``source`` with ``compiler`` for ``row``: ``pick_target(compiler)`` names the
    artifact kind, ``argv_flags(opts, device_target)`` the compiler's flags, ``terms(flags, device_target)`` the key's flag terms, ``parse(stderr)`` the resource
    figures the compiler reports."""
    identity = tc.compiler_identity(compiler)
    art_ext, src_ext = row.artifact, row.source
    cache = Cache(opts.cache_dir)

    # The device artifact that actually loads on the driver present; "ptx"
    # for every host compile, which carries no such choice.
    device_target = pick_target(compiler)

    # The key is computed from a nominal compile's flags; -o/-MF are
    # excluded (they say where output goes, not what it is). The chosen
    # target is folded in explicitly, beside the flags it already changes
    # (-ptx vs -cubin): the same source to two targets must never share a
    # slot.
    probe = _argv(compiler, argv_flags(opts, device_target), Path("<src>"), Path("<out>"), Path("<dep>"))
    # A device key also carries nvcc's host compiler (-ccbin): it compiles
    # the TU's host side and its version gates what nvcc accepts.
    # A pip-only compile serves the sealed payload into a per-process
    # directory (tc.pip_only_root): its path differs in every process, so the
    # key names the payload by digest and the closure leaves those headers out.
    flags, served = _stable_served(probe[1:-4])
    key_terms = terms(flags, device_target)
    key = lookup_key(source, opts.target, opts.mode, identity, key_terms)
    slot = cache.slot(key)
    artifact, src_path, dep = (slot / f"{name}{art_ext}", slot / f"{name}{src_ext}",
                               slot / f"{name}.d")

    def lookup():
        memo = artifact_memo_get(key)
        if memo is not None and memo_still_valid(artifact, memo):
            # Per-process memo hit: same key, same recorded closure, still valid
            # by content — counted as a cache hit; it only skips re-reading the
            # on-disk record.
            cache.note(True)
            return CompileResult(artifact, src_path, key, True, tuple(probe),
                                 closure=tuple(p for p, _ in memo),
                                 ptxas=cache.record(key).get("ptxas", {}))

        if cache.check(key):
            cache.note(True)
            record = cache.record(key)
            closure = tuple(record.get("closure", ()))
            artifact_memo_put(key, tuple((p, d) for p, d in closure))
            return CompileResult(artifact, src_path, key, True, tuple(probe),
                                 closure=tuple(p for p, _ in closure),
                                 ptxas=record.get("ptxas", {}))
        return None

    def build():
        cache.note(False)
        tc.announce_build(name, opts.target, cache.root)
        check_source_budget(name, source)
        slot.mkdir(parents=True, exist_ok=True)
        write_atomic(src_path, source)
        # Compile into private temporaries, then move into place: a concurrent
        # reader of this slot sees the old artifact or the new one, never a
        # partial write.
        tmp_artifact, tmp_dep = tmp_sibling(artifact), tmp_sibling(dep)
        argv = _argv(compiler, argv_flags(opts, device_target), src_path, tmp_artifact, tmp_dep)
        done, seconds = _run_and_publish(
            argv, f"{opts.target} compile of {name!r}", "",
            tmp_artifact, artifact, tmp_dep, dep, cache.root, opts.target)
        closure = tuple(p for p in closure_of(dep) if not p.startswith(served))
        ptxas = parse(done.stderr)
        cache.store(key, artifact=artifact, closure=closure,
                    meta={"target": opts.target, "mode": opts.mode,
                          "compiler": identity, "flags": argv[1:-4],
                          "key_terms": list(key_terms),
                          "seconds": seconds, "name": name,
                          "device_target": device_target, "ptxas": ptxas})
        artifact_memo_put(key, validity_digest(closure))
        return CompileResult(artifact, src_path, key, False, tuple(argv), seconds,
                             closure, ptxas)

    return single_flight(key, lookup, build)


def memo_still_valid(artifact: Path, closure,
                       current_payload_digest: str | None = None) -> bool:
    """Whether a memoised answer is still the right one: the memo remembers
    WHAT the last compile at this key opened and what those files hashed
    to, never that they are still that way, so a hit re-verifies the whole
    recorded closure by content (:func:`~hawk.compile.cache.closure_unchanged`)
    and requires the artifact still be on disk.

    A ``closure`` led by the pair ``(_PAYLOAD_MEMO_TAG, payload_digest)`` is
    a payload-backed entry: the payload's headers are that digest (one
    string compare), and any pairs after it are USER headers, re-verified by
    content like any other closure (an NVRTC entry has none)."""
    if not artifact.is_file():
        return False
    if (isinstance(closure, tuple) and closure
            and closure[0][0] == _PAYLOAD_MEMO_TAG):
        return (current_payload_digest is not None
                and closure[0][1] == current_payload_digest
                and closure_unchanged(closure[1:]))
    return closure_unchanged(closure)


class _KeyedFlag(str):
    """A flag the compile line keeps and the key reads as its ``key_term`` (the per-process served root)."""

    key_term: str = ""


class _Inputs:
    """What hawk knows of the headers a compile reads, handed to :meth:`hawk.backends.Backend.compile_plan`. Each
    field is resolved when read: resolving a root or the payload has a cost or a side effect."""

    @property
    def sealed(self) -> bool:
        """Whether a compile builds against the shipped sealed payload rather than a real include root: the condition
        :func:`hawk.compile.payload.current_payload` uses to prefer ``aether_dsc.payload()`` over a live re-seal (no
        ``$HAWK_AETHER_INCLUDE`` override, and a blob exists)."""
        if "HAWK_AETHER_INCLUDE" in os.environ:
            return False
        import aether_dsc
        try:
            aether_dsc.payload()
            return True
        except FileNotFoundError:
            return False

    @property
    def payload(self):
        """The sealed payload of this process (:func:`hawk.compile.payload.current_payload`)."""
        return _payload.current_payload()

    def include_flags(self) -> tuple:
        """The aether and eagle ``-I`` flags; a per-process served root carries the payload digest as its key term."""
        flags = tc.include_flags()
        root = tc.pip_only_root()
        if root is None:
            return tuple(flags)
        keyed = _KeyedFlag(root)
        keyed.key_term = f"payload:{_payload.current_payload().digest}"
        return tuple(keyed if f == root else f for f in flags)


@dataclass(frozen=True)
class _Job:
    """One compile as hawk hands it to :meth:`hawk.backends.Backend.compile_hook`: the source hawk wrote, the
    temporaries the artifact and the dependency file go to (hawk publishes them), the served payload root (``None``: none)."""

    name: str
    src: Path
    out: Path
    dep: Path
    served: object


def _served_prefixes(*roots) -> tuple:
    return tuple({p for r in roots if r for p in (os.path.abspath(str(r)) + os.sep, os.path.realpath(str(r)) + os.sep)})


def _drive(source: str, name: str, opts: CompileOptions, row, plan: dict) -> CompileResult:
    """The cached compile of ``source`` for ``row`` from its ``plan`` (data) and its
    :meth:`~hawk.backends.Backend.compile_hook` (code). hawk owns the key, the slot, the memo and its validity, the
    single flight, the payload serving, the machine guards, the closure and the record.

    A ``plan["payload"]`` digest makes the entry valid by that digest: the payload's headers are keyed by it (the
    backend puts it in ``key_terms``), and only what the compiler opened outside every root hawk served is a recorded
    closure, validated by content."""
    cache = Cache(opts.cache_dir)
    identity, key_terms, digest = plan["identity"], tuple(plan["key_terms"]), plan["payload"]
    key = lookup_key(source, opts.target, opts.mode, identity, key_terms)
    slot = cache.slot(key)
    artifact, src_path, dep = (slot / f"{name}{row.artifact}", slot / f"{name}{row.source}", slot / f"{name}.d")
    tagged = () if digest is None else ((_PAYLOAD_MEMO_TAG, digest),)
    hit_argv = tuple(plan["hit_argv"])

    def lookup():
        memo = artifact_memo_get(key)
        if memo is not None and memo_still_valid(artifact, memo, digest):
            # per-process memo hit: same key, same recorded closure, still valid by content
            cache.note(True)
            return CompileResult(artifact, src_path, key, True, hit_argv, closure=tagged or tuple(p for p, _ in memo),
                                 ptxas=cache.record(key).get("ptxas", {}))
        if cache.check(key, payload_digest=digest):
            cache.note(True)
            record = cache.record(key)
            closure = tuple(record.get("closure", ()))
            artifact_memo_put(key, tuple((p, d) for p, d in closure))
            return CompileResult(artifact, src_path, key, True, hit_argv, closure=tagged or tuple(p for p, _ in closure),
                                 ptxas=record.get("ptxas", {}))
        return None

    def build():
        cache.note(False)
        tc.announce_build(name, opts.target, cache.root)
        check_source_budget(name, source)
        slot.mkdir(parents=True, exist_ok=True)
        write_atomic(src_path, source)
        # compile into private temporaries, then move into place: a reader of this slot never sees a partial write
        tmp_artifact, tmp_dep = tmp_sibling(artifact), tmp_sibling(dep)
        label = f"{opts.target} compile of {name!r}"

        def run(argv):
            done = run_compiler(argv, label, opts.target)
            if done.returncode != 0:
                _refuse(done, argv, label, plan["tail"], cache.root)
            return done

        start = time.perf_counter()
        try:
            with (_payload.current_payload().serve() if plan["serve"] else nullcontext()) as served_root:
                out = row.compile_hook(plan, _Job(name, src_path, tmp_artifact, tmp_dep, served_root), run)
                publish_atomic(tmp_artifact, artifact)
                if plan["deps"]:
                    publish_atomic(tmp_dep, dep)
                seconds = time.perf_counter() - start
                reached = closure_of(dep) if plan["deps"] else ()
                prefixes = _served_prefixes(served_root, tc.pip_only_root())
                closure = tuple(p for p in reached if not p.startswith(prefixes))
        finally:
            tmp_artifact.unlink(missing_ok=True)
            tmp_dep.unlink(missing_ok=True)
        meta = {"target": opts.target, "mode": opts.mode, "compiler": identity, "key_terms": list(key_terms),
                "seconds": seconds, "name": name, **plan["meta"], "ptxas": out["figures"]}
        if digest is not None:
            meta.update(payload_digest=digest, served_closure_card=list(reached))
        cache.store(key, artifact=artifact, closure=closure, meta=meta)
        artifact_memo_put(key, validity_digest(closure))
        return CompileResult(artifact, src_path, key, False, tuple(out["argv"]), seconds, tagged or closure, out["figures"])

    return single_flight(key, lookup, build)


def publish(result: CompileResult, destination: Path) -> Path:
    """Copy a cached artifact to ``destination`` (an artifact DIRECTORY is a
    deployment unit; the cache is a build-time store, never the unit itself)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.artifact, destination)
    return destination
