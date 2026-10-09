# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The compile drivers, both through the content-closure cache.

``g++`` builds the host ``.so`` (flags from one host profile, see
:data:`hawk.compile.toolchain.HOST_PROFILES`). The device backend has two
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

A driver is a pure function of its inputs: source, backend, scalar mode,
compiler and flags produce the same key and slot.

Every compiler subprocess runs under an address-space cap — a machine
guard, not a codegen policy, so a pathological compile fails loudly instead
of taking the box down (a 64x23 KAN cell once reached ptxas as one function
and took 25-27 GB of RSS on a 31 GB machine, with nothing bounding it).
:data:`ADDRESS_SPACE_CAP_BYTES` is that bound, applied with ``RLIMIT_AS`` in
the child before the compiler ``exec``s; ``$HAWK_COMPILE_ADDRESS_CAP`` overrides it
(bytes; ``0`` disables it).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..ir import HawkError
from . import nvrtc as _nvrtc_driver
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
    "resource.setrlimit(resource.RLIMIT_AS, (limit, hard))\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)


def limited(argv: list[str]) -> list[str]:
    """``argv`` wrapped so the child runs under the address-space cap, or
    ``argv`` itself when the cap is disabled.

    The cap is installed by a small exec trampoline (``python -I -c``: set
    ``RLIMIT_AS``, then ``os.execvp`` the real command) rather than a
    a fork-time hook, which is unsafe to run from a multi-threaded process.
    The limit is inherited by every process the compiler driver spawns —
    ``nvcc`` runs ``cicc`` and ``ptxas``, and ``ptxas`` is the one that ran
    the machine out of memory. The soft limit is raised no higher than the
    inherited hard limit."""
    cap = address_space_cap()
    if not cap:
        return list(argv)
    return [sys.executable, "-I", "-c", _TRAMPOLINE, str(cap), *argv]


#: Artifact suffix + source suffix per backend id.
_SUFFIX = {"host": (".so", ".cpp"), "cuda": (".ptx", ".cu")}
HOST = "host"
DEVICE = "cuda"


@dataclass(frozen=True)
class CompileOptions:
    """Everything outside the source text that changes the output — and so
    every term of the LOOKUP key beyond the source itself."""

    backend: str = HOST
    mode: str = "float64"
    arch: str = ""
    defines: tuple[str, ...] = ()
    cache_dir: str | None = None
    extra: tuple[str, ...] = ()
    #: The host code-generation profile
    #: (:data:`hawk.compile.toolchain.HOST_PROFILES`). Empty means
    #: ``$HAWK_HOST_PROFILE``, else
    #: :data:`~hawk.compile.toolchain.DEFAULT_HOST_PROFILE`. Ignored by the
    #: device backend.
    host_profile: str = ""
    #: The optimisation level (:data:`hawk.compile.toolchain.OPT_LEVELS`).
    #: Empty means ``$HAWK_OPT_LEVEL``, else
    #: :data:`~hawk.compile.toolchain.DEFAULT_OPT_LEVEL`. Used by the host
    #: backend's ``-O<n>`` and the device backend's AOT nvcc compile
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


#: Guards :data:`_INFLIGHT`.
_LOCK = threading.Lock()
#: Lookup key -> the event the thread compiling that key sets when it is done.
_INFLIGHT: dict[str, threading.Event] = {}


def _single_flight(key: str, lookup, build) -> CompileResult:
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


def _argv(compiler: str, opts: CompileOptions, src: Path, out: Path,
          dep: Path, *, device_target: str = "ptx") -> list[str]:
    defines = [f"-D{d}" for d in opts.defines]
    if opts.backend == HOST:
        flags = tc.host_flags(extra=(*defines, *opts.extra),
                              profile=opts.host_profile or None,
                              opt=opts.opt_level or None)
    else:
        flags = tc.device_flags(arch=opts.arch, target=device_target,
                                extra=(*defines, *opts.extra),
                                opt=opts.opt_level or None)
    return [compiler, *flags, "-MD", "-MF", str(dep), str(src), "-o", str(out)]


def _host_key_terms(flags, opts: CompileOptions) -> tuple:
    """The host key's flag terms plus, for the two ``-march=native``
    profiles, the CPU it targets
    (:func:`~hawk.compile.toolchain.host_target_identity`): ``-march=native``
    reads the same on every machine, so without this term a shared cache
    directory would serve one CPU's binary to another's. The profile name
    and the resolved opt level are folded in too, so two profiles — or two
    opt levels — never share a slot even where their flag strings coincide.
    The C library (``platform.libc_ver()``) is folded in as well: a host
    ``.so`` links against the libc it was built on."""
    profile = tc.host_profile(opts.host_profile or None)
    level = tc.opt_level(opts.opt_level or None)
    libc = "-".join(part for part in platform.libc_ver() if part) or "unknown"
    terms = (*flags, f"host_profile:{profile}", f"opt_level:{level}", f"libc:{libc}")
    if profile in ("native", "native-vector-math"):
        terms = (*terms, f"host_target:{tc.host_target_identity()}")
    return terms


def _device_target(compiler: str, arch: str) -> str:
    """``"cubin"`` or ``"ptx"`` for an AOT nvcc device compile at the
    resolved ``arch``. A virtual ``compute_<N>`` arch (the arch policy's
    no-GPU and above-the-toolkit answers) is always PTX
    (:func:`hawk.compile.nvrtc.forced_target`); otherwise the device
    artifact that loads on the driver present
    (:func:`hawk.compile.nvrtc.select_target`), compared against nvcc's own
    parsed version rather than NVRTC's."""
    forced = _nvrtc_driver.forced_target(arch)
    if forced is not None:
        return forced
    return _nvrtc_driver.select_target(
        tc.nvcc_version(compiler), _probed(_nvrtc_driver.driver_cuda_version))


def _probed(fn):
    """``fn()``, or ``None`` on a :class:`HawkError` — see
    :func:`_device_target`'s own docstring for why this never raises."""
    try:
        return fn()
    except HawkError:
        return None


def compile_source(source: str, name: str, opts: CompileOptions) -> CompileResult:
    """Compile ``source`` for ``opts.backend``, through the content-closure
    cache: a valid lookup is a hit and no compiler runs; anything else is a
    miss that compiles into the key's own slot and rewrites its validity
    record from the compiler-reported closure.

    Two forks: the device backend forks on
    :func:`~hawk.compile.toolchain.device_compiler_kind` — ``"nvrtc"`` goes
    to :func:`_compile_nvrtc_source`. The host backend forks on
    :func:`_host_uses_sealed_payload` — a host compile against the shipped
    sealed payload goes to :func:`_compile_host_sealed`, serving the payload
    into a private directory for the compile's lifetime instead of pointing
    ``-I`` at a real include root.
    """
    if opts.backend not in _SUFFIX:
        raise HawkError(f"unknown backend {opts.backend!r}; built are {tuple(_SUFFIX)}")
    if opts.backend == HOST and _host_uses_sealed_payload():
        return _compile_host_sealed(source, name, opts)
    if opts.backend == DEVICE and tc.device_compiler_kind() == "nvrtc":
        return _compile_nvrtc_source(source, name, opts)
    compiler = tc.host_compiler() if opts.backend == HOST else tc.device_compiler()
    identity = tc.compiler_identity(compiler)
    art_ext, src_ext = _SUFFIX[opts.backend]
    cache = Cache(opts.cache_dir)

    # The device artifact that actually loads on the driver present; "ptx"
    # for every host compile, which carries no such choice.
    device_target = (_device_target(compiler, tc.resolve_arch(opts.arch))
                     if opts.backend == DEVICE else "ptx")

    # The key is computed from a nominal compile's flags; -o/-MF are
    # excluded (they say where output goes, not what it is). The chosen
    # target is folded in explicitly, beside the flags it already changes
    # (-ptx vs -cubin): the same source to two targets must never share a
    # slot.
    probe = _argv(compiler, opts, Path("<src>"), Path("<out>"), Path("<dep>"),
                  device_target=device_target)
    # A device key also carries nvcc's host compiler (-ccbin): it compiles
    # the TU's host side and its version gates what nvcc accepts.
    key_terms = (_host_key_terms(probe[1:-4], opts) if opts.backend == HOST
                else (*probe[1:-4], f"device_target:{device_target}",
                      tc.device_host_compiler_identity()))
    key = lookup_key(source, opts.backend, opts.mode, identity, key_terms)
    slot = cache.slot(key)
    artifact, src_path, dep = (slot / f"{name}{art_ext}", slot / f"{name}{src_ext}",
                               slot / f"{name}.d")

    def lookup():
        memo = artifact_memo_get(key)
        if memo is not None and _memo_still_valid(artifact, memo):
            # Per-process memo hit: same key, same recorded closure, still valid
            # by content — counted as a cache hit; it only skips re-reading the
            # on-disk record.
            cache.note(True)
            return CompileResult(artifact, src_path, key, True, tuple(probe),
                                 closure=tuple(p for p, _ in memo))

        if cache.check(key):
            cache.note(True)
            record = cache.record(key)
            closure = tuple(record.get("closure", ()))
            artifact_memo_put(key, tuple((p, d) for p, d in closure))
            return CompileResult(artifact, src_path, key, True, tuple(probe),
                                 closure=tuple(p for p, _ in closure))
        return None

    def build():
        cache.note(False)
        slot.mkdir(parents=True, exist_ok=True)
        write_atomic(src_path, source)
        # Compile into private temporaries, then move into place: a concurrent
        # reader of this slot sees the old artifact or the new one, never a
        # partial write.
        tmp_artifact, tmp_dep = tmp_sibling(artifact), tmp_sibling(dep)
        argv = _argv(compiler, opts, src_path, tmp_artifact, tmp_dep,
                     device_target=device_target)
        start = time.perf_counter()
        try:
            done = subprocess.run(limited(argv), capture_output=True, text=True,
                                  env=tc.subprocess_env())
            if done.returncode == 0:
                publish_atomic(tmp_artifact, artifact)
                publish_atomic(tmp_dep, dep)
        finally:
            tmp_artifact.unlink(missing_ok=True)
            tmp_dep.unlink(missing_ok=True)
        seconds = time.perf_counter() - start
        if done.returncode != 0:
            cap = address_space_cap()
            raise HawkError(
                f"{opts.backend} compile of {name!r} failed (rc={done.returncode})"
                + (f"; the compiler ran under a {cap}-byte address-space cap "
                   "($HAWK_COMPILE_ADDRESS_CAP), so an allocation failure here means "
                   "this translation unit is too large for one compile and not that "
                   "the machine is out of memory" if cap else "")
                + f".\n$ {' '.join(argv)}\n{done.stderr[:4000]}"
            )
        closure = closure_of(dep)
        cache.store(key, artifact=artifact, closure=closure,
                    meta={"backend": opts.backend, "mode": opts.mode,
                          "compiler": identity, "flags": argv[1:-4],
                          "key_terms": list(key_terms),
                          "seconds": seconds, "name": name,
                          "device_target": device_target})
        artifact_memo_put(key, validity_digest(closure))
        return CompileResult(artifact, src_path, key, False, tuple(argv), seconds, closure)

    return _single_flight(key, lookup, build)


def _memo_still_valid(artifact: Path, closure,
                       current_payload_digest: str | None = None) -> bool:
    """Whether a memoised answer is still the right one: the memo remembers
    WHAT the last compile at this key opened and what those files hashed
    to, never that they are still that way, so a hit re-verifies the whole
    recorded closure by content (:func:`~hawk.compile.cache.closure_unchanged`)
    and requires the artifact still be on disk.

    A ``closure`` shaped as the one pair ``((_PAYLOAD_MEMO_TAG,
    payload_digest))`` is an NVRTC entry: its closure IS the payload digest
    it was compiled against, so re-validating it is one string compare."""
    if not artifact.is_file():
        return False
    if (isinstance(closure, tuple) and len(closure) == 1
            and closure[0][0] == _PAYLOAD_MEMO_TAG):
        return (current_payload_digest is not None
                and closure[0][1] == current_payload_digest)
    return closure_unchanged(closure)


def _compile_nvrtc_source(source: str, name: str,
                          opts: CompileOptions) -> CompileResult:
    """The NVRTC device compile, parallel to :func:`compile_source`'s body
    but with an NVRTC-shaped key and validity record: the compiler term is
    :func:`hawk.compile.nvrtc.identity`, the flags term is
    :func:`hawk.compile.nvrtc.options` (no ``-I <aether root>``), and the
    closure is the sealed payload's own digest rather than a
    compiler-reported file list — the payload is immutable and known before
    the compile runs, so a changed header is already subsumed by "a
    different payload is a different key". The target is picked the same
    way :func:`hawk.compile.nvrtc.device` does
    (:func:`~hawk.compile.nvrtc.select_target`), and is also folded into the
    key, for the same never-share-a-slot reason.

    ``opts.opt_level`` is NOT read here: NVRTC always optimises and takes no
    ``-O`` option at all (:func:`hawk.compile.nvrtc.options` carries none),
    so the level has nothing to change and folding it into this key would
    only fragment the cache for free.
    """
    arch = tc.resolve_arch(opts.arch)
    art_ext, src_ext = _SUFFIX[DEVICE]
    cache = Cache(opts.cache_dir)

    payload = _payload.current_payload()
    identity = _nvrtc_driver.identity()
    target = _nvrtc_driver.forced_target(arch) or _nvrtc_driver.select_target(
        _probed(_nvrtc_driver.nvrtc_version),
        _probed(_nvrtc_driver.driver_cuda_version))
    device_opts = _nvrtc_driver.options(arch, virtual_arch=(target == "ptx"))
    key = lookup_key(source, DEVICE, opts.mode, identity,
                      (*device_opts, f"device_target:{target}",
                       f"payload:{payload.digest}"))
    slot = cache.slot(key)
    artifact = slot / f"{name}{art_ext}"
    src_path = slot / f"{name}{src_ext}"       # never written (no file)
    argv = tuple(device_opts)                  # no compiler subprocess argv exists

    payload_closure = ((_PAYLOAD_MEMO_TAG, payload.digest),)

    def lookup():
        memo = artifact_memo_get(key)
        if memo is not None and _memo_still_valid(artifact, memo, payload.digest):
            cache.note(True)
            return CompileResult(artifact, src_path, key, True, argv,
                                 closure=payload_closure)

        if cache.check(key, payload_digest=payload.digest):
            cache.note(True)
            artifact_memo_put(key, payload_closure)
            return CompileResult(artifact, src_path, key, True, argv,
                                 closure=payload_closure)
        return None

    def build():
        cache.note(False)
        slot.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        compile_fn = (_nvrtc_driver.compile_cubin if target == "cubin"
                     else _nvrtc_driver.compile_ptx)
        result = compile_fn(source, name, arch, headers=payload.device_headers)
        seconds = time.perf_counter() - start
        if not result.ok:
            raise HawkError(
                f"cuda compile of {name!r} failed under NVRTC (target={target}, "
                f"rc != NVRTC_SUCCESS).\n$ nvrtc {' '.join(argv)}\n{result.log[:4000]}"
            )
        write_atomic(artifact, result.ptx)
        cache.store(key, artifact=artifact, closure=(),
                    meta={"backend": DEVICE, "mode": opts.mode, "compiler": identity,
                          "flags": list(argv), "seconds": seconds, "name": name,
                          "payload_digest": payload.digest, "device_target": target})
        artifact_memo_put(key, payload_closure)
        return CompileResult(artifact, src_path, key, False, argv, seconds,
                             closure=payload_closure)

    return _single_flight(key, lookup, build)


def _host_uses_sealed_payload() -> bool:
    """Whether a host compile should build against the shipped sealed
    payload rather than a real include root on disk — the same condition
    :func:`hawk.compile.payload.current_payload` uses to prefer
    ``aether_dsc.payload()`` over a live re-seal: no
    ``$HAWK_AETHER_INCLUDE`` override, and a blob actually exists."""
    if "HAWK_AETHER_INCLUDE" in os.environ:
        return False
    import aether_dsc
    try:
        aether_dsc.payload()
        return True
    except FileNotFoundError:
        return False


def _compile_host_sealed(source: str, name: str, opts: CompileOptions) -> CompileResult:
    """The host ``g++`` compile against the sealed payload: ``-I`` points at
    a private directory :meth:`~aether_dsc.Payload.serve` materialises for
    exactly this compile's lifetime, removed afterwards on success or
    failure (`serve()`'s own ``finally``).

    The lookup key cannot include the served directory's own path (a fresh
    ``tempfile.mkdtemp`` every compile, never repeating) — the payload's
    digest is the stable substitute, the same one the NVRTC device path
    uses. ``-MD`` still runs and its closure is recorded for the
    compile-time card's sake, but validity is decided by payload digest
    alone: re-stat'ing paths inside an already-removed served directory
    would be a guaranteed miss.
    """
    art_ext, src_ext = _SUFFIX[HOST]
    cache = Cache(opts.cache_dir)
    compiler = tc.host_compiler()
    identity = tc.compiler_identity(compiler)
    payload = _payload.current_payload()

    defines = [f"-D{d}" for d in opts.defines]
    stable_flags = [*tc.host_codegen_flags(opts.host_profile or None,
                                           opts.opt_level or None),
                    f"-std={tc.HOST_STD}", "-shared", "-fPIC",
                    "-DAETHER_CPP_MODE", *defines, *opts.extra]
    key_terms = (*_host_key_terms(stable_flags, opts), f"payload:{payload.digest}")
    key = lookup_key(source, HOST, opts.mode, identity, key_terms)
    slot = cache.slot(key)
    artifact, src_path, dep = (slot / f"{name}{art_ext}", slot / f"{name}{src_ext}",
                               slot / f"{name}.d")
    payload_closure = ((_PAYLOAD_MEMO_TAG, payload.digest),)

    def lookup():
        memo = artifact_memo_get(key)
        if memo is not None and _memo_still_valid(artifact, memo, payload.digest):
            cache.note(True)
            return CompileResult(artifact, src_path, key, True, tuple(stable_flags),
                                 closure=payload_closure)

        if cache.check(key, payload_digest=payload.digest):
            cache.note(True)
            artifact_memo_put(key, payload_closure)
            return CompileResult(artifact, src_path, key, True, tuple(stable_flags),
                                 closure=payload_closure)
        return None

    def build():
        cache.note(False)
        slot.mkdir(parents=True, exist_ok=True)
        write_atomic(src_path, source)
        tmp_artifact, tmp_dep = tmp_sibling(artifact), tmp_sibling(dep)
        with payload.serve() as served_root:
            argv = [compiler, *stable_flags, f"-I{served_root}", "-MD", "-MF", str(tmp_dep),
                    str(src_path), "-o", str(tmp_artifact)]
            start = time.perf_counter()
            try:
                done = subprocess.run(limited(argv), capture_output=True, text=True,
                                      env=tc.subprocess_env())
                if done.returncode == 0:
                    publish_atomic(tmp_artifact, artifact)
                    publish_atomic(tmp_dep, dep)
            finally:
                tmp_artifact.unlink(missing_ok=True)
                tmp_dep.unlink(missing_ok=True)
            seconds = time.perf_counter() - start
            if done.returncode != 0:
                raise HawkError(
                    f"host compile of {name!r} failed against the sealed payload "
                    f"(rc={done.returncode}).\n$ {' '.join(argv)}\n{done.stderr[:4000]}"
                )
            card_closure = closure_of(dep)  # card material only, see docstring
        cache.store(key, artifact=artifact, closure=(),
                    meta={"backend": HOST, "mode": opts.mode, "compiler": identity,
                          "flags": stable_flags, "seconds": seconds, "name": name,
                          "key_terms": list(key_terms),
                          "payload_digest": payload.digest,
                          "served_closure_card": list(card_closure)})
        artifact_memo_put(key, payload_closure)
        return CompileResult(artifact, src_path, key, False, tuple(argv), seconds,
                             closure=payload_closure)

    return _single_flight(key, lookup, build)


def publish(result: CompileResult, destination: Path) -> Path:
    """Copy a cached artifact to ``destination`` (an artifact DIRECTORY is a
    deployment unit; the cache is a build-time store, never the unit itself)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.artifact, destination)
    return destination
