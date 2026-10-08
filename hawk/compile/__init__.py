# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""AOT compile drivers + the content-closure cache.

The shipped Python tier's device compile is NVRTC, in memory, from a
sealed :mod:`aether_dsc` payload (:func:`current_payload`) — no aether
header touches disk. :func:`.toolchain.device_compiler_kind` picks it
whenever no `nvcc` is findable; the lookup key folds in the payload's
own digest instead of a compiler-reported closure. A host compile
against the same sealed payload goes through
`aether_dsc.Payload.serve()` instead of a real include root.
"""

from __future__ import annotations

from . import nvrtc
from . import payload as _payload_module
from .cache import (
    Cache,
    cache_stats,
    default_cache_dir,
    digest_file,
    lookup_key,
    reset_cache_stats,
)
from .cache import (
    reset_artifact_memo as _cache_reset_artifact_memo,
)
from .drivers import (
    DEVICE,
    HOST,
    CompileOptions,
    compile_source,
    publish,
)
from .nvrtc import (
    DeviceImage,
    cubin,
    cubin_available,
    device,
)
from .payload import current_payload
from .toolchain import (
    HOST_PROFILES,
    OPT_LEVELS,
    aether_include,
    compiler_identity,
    device_compiler,
    device_compiler_kind,
    eagle_include,
    host_codegen_flags,
    host_compiler,
    host_flags,
    host_profile,
    opt_level,
)

__all__ = ["Cache", "CompileOptions", "DeviceImage", "DEVICE", "HOST",
           "HOST_PROFILES", "OPT_LEVELS", "aether_include", "cache_stats",
           "compile_source", "compiler_identity", "cubin", "cubin_available",
           "current_payload", "default_cache_dir", "device", "device_compiler",
           "device_compiler_kind", "digest_file", "eagle_include",
           "host_codegen_flags", "host_compiler", "host_flags", "host_profile",
           "lookup_key", "nvrtc", "opt_level", "publish", "reset_cache_stats"]


def _reset_artifact_memo() -> None:
    """Forget every memoised artifact/closure verdict and the memoised
    :func:`current_payload` — one door back to a fresh, honest check."""
    _cache_reset_artifact_memo()
    _payload_module.reset_payload_memo()
