# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device compile drivers: nvcc through the content-closure cache, and NVRTC in memory."""

from __future__ import annotations

import re
import time

from hawk.compile import payload as _payload
from hawk.compile import toolchain as tc
from hawk.compile.cache import (
    PAYLOAD_CLOSURE_TAG as _PAYLOAD_MEMO_TAG,
    Cache,
    artifact_memo_get,
    artifact_memo_put,
    lookup_key,
    write_atomic,
)
from hawk.compile.drivers import CompileOptions, CompileResult, _memo_still_valid, _single_flight
from raptor.errors import RaptorError

from . import nvrtc as _nvrtc_driver
from . import toolchain as dtc

_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")


_PTXAS_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


_PTXAS_REGS = re.compile(r"Used (\d+) registers")


def parse_ptxas_info(stderr: str) -> dict:
    """``{kernel: {regs, spill_store_B, spill_load_B, stack_B}}`` from the
    ``ptxas info`` lines of an nvcc ``-Xptxas -v`` stderr; ``{}`` when it carries
    none. A kernel compiled for several archs keeps the largest figures."""
    found: dict = {}
    entry = None
    for line in stderr.splitlines():
        if (m := _PTXAS_ENTRY.search(line)):
            entry = found.setdefault(m.group(1), {})
        elif entry is not None and (m := _PTXAS_FRAME.search(line)):
            _grow(entry, stack_B=m.group(1), spill_store_B=m.group(2),
                  spill_load_B=m.group(3))
        elif entry is not None and (m := _PTXAS_REGS.search(line)):
            _grow(entry, regs=m.group(1))
    return found


def _grow(entry: dict, **values: str) -> None:
    for key, value in values.items():
        entry[key] = max(entry.get(key, 0), int(value))


def _device_target(compiler: str, arch: str) -> str:
    """``"cubin"`` or ``"ptx"`` for an AOT nvcc device compile at the
    resolved ``arch``. A virtual ``compute_<N>`` arch (the arch policy's
    no-GPU and above-the-toolkit answers) is always PTX
    (:func:`hawk_cuda.nvrtc.forced_target`); otherwise the device
    artifact that loads on the driver present
    (:func:`hawk_cuda.nvrtc.select_target`), compared against nvcc's own
    parsed version rather than NVRTC's."""
    forced = _nvrtc_driver.forced_target(arch)
    if forced is not None:
        return forced
    return _nvrtc_driver.select_target(
        tc.nvcc_version(compiler),
        _nvrtc_driver._probed(_nvrtc_driver.driver_cuda_version))


def _compile_nvrtc_source(source: str, name: str, opts: CompileOptions,
                          row) -> CompileResult:
    """The NVRTC device compile, parallel to :func:`compile_source`'s body
    but with an NVRTC-shaped key and validity record: the compiler term is
    :func:`hawk_cuda.nvrtc.identity`, the flags term is
    :func:`hawk_cuda.nvrtc.options` (no ``-I <aether root>``), and the
    closure is the sealed payload's own digest rather than a
    compiler-reported file list — the payload is immutable and known before
    the compile runs, so a changed header is already subsumed by "a
    different payload is a different key". The target is picked the same
    way :func:`hawk_cuda.nvrtc.device` does
    (:func:`~hawk_cuda.nvrtc.select_target`), and is also folded into the
    key, for the same never-share-a-slot reason. ``opts.defines`` and
    ``opts.extra`` are passed to NVRTC and are terms of the key, as they are
    for nvcc: a build with a define and one without never share a slot.

    ``opts.opt_level`` is NOT read here: NVRTC always optimises and takes no
    ``-O`` option at all (:func:`hawk_cuda.nvrtc.options` carries none),
    so the level has nothing to change and folding it into this key would
    only fragment the cache for free.
    """
    arch = dtc.resolve_arch(opts.arch)
    art_ext, src_ext = row.artifact, row.source
    cache = Cache(opts.cache_dir)

    payload = _payload.current_payload()
    identity = _nvrtc_driver.identity()
    target = _nvrtc_driver.forced_target(arch) or _nvrtc_driver.select_target(
        _nvrtc_driver._probed(_nvrtc_driver.nvrtc_version),
        _nvrtc_driver._probed(_nvrtc_driver.driver_cuda_version))
    # the build's defines and extra flags reach NVRTC exactly as they reach
    # nvcc (:func:`_argv`), and through ``device_opts`` the key as well
    extra = (*(f"-D{d}" for d in opts.defines), *opts.extra)
    device_opts = _nvrtc_driver.options(arch, virtual_arch=(target == "ptx"),
                                        extra=extra)
    key = lookup_key(source, opts.target, opts.mode, identity,
                      (*device_opts, f"device_target:{target}",
                       f"payload:{payload.digest}", "ptxas-info:v1"))
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
                                 closure=payload_closure,
                                 ptxas=cache.record(key).get("ptxas", {}))

        if cache.check(key, payload_digest=payload.digest):
            cache.note(True)
            artifact_memo_put(key, payload_closure)
            return CompileResult(artifact, src_path, key, True, argv,
                                 closure=payload_closure,
                                 ptxas=cache.record(key).get("ptxas", {}))
        return None

    def build():
        cache.note(False)
        slot.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        compile_fn = (_nvrtc_driver.compile_cubin if target == "cubin"
                     else _nvrtc_driver.compile_ptx)
        result = compile_fn(source, name, arch, headers=payload.device_headers,
                            extra=extra)
        seconds = time.perf_counter() - start
        if not result.ok:
            raise RaptorError(
                f"cuda compile of {name!r} failed under NVRTC (target={target}, "
                f"rc != NVRTC_SUCCESS).\n$ nvrtc {' '.join(argv)}\n{result.log[:4000]}"
            )
        write_atomic(artifact, result.ptx)
        # registers and stack frame read off the CUBIN itself; spills are not knowable (None)
        ptxas = _nvrtc_driver.cubin_resources(result.ptx) if target == "cubin" else {}
        cache.store(key, artifact=artifact, closure=(),
                    meta={"target": opts.target, "mode": opts.mode, "compiler": identity,
                          "flags": list(argv), "seconds": seconds, "name": name,
                          "payload_digest": payload.digest, "device_target": target,
                          "ptxas": ptxas})
        artifact_memo_put(key, payload_closure)
        return CompileResult(artifact, src_path, key, False, argv, seconds,
                             closure=payload_closure, ptxas=ptxas)

    return _single_flight(key, lookup, build)
