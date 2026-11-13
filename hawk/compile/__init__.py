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
    CompileOptions,
    compile_source,
    publish,
)
from .payload import current_payload
from .toolchain import (
    OPT_LEVELS,
    aether_include,
    compiler_identity,
    eagle_include,
    opt_level,
)

#: Doors that are a device backend's names: ``hawk.compile.<door>`` is that backend's ``api`` attribute.
_CUDA_DOORS = ("nvrtc", "DeviceImage", "cubin", "cubin_available", "device", "device_compiler", "device_compiler_kind")

__all__ = ["Cache", "CompileOptions", "OPT_LEVELS", "aether_include", "cache_stats",
           "compile_source", "compiler_identity",
           "current_payload", "default_cache_dir", "digest_file", "eagle_include",
           "lookup_key", "opt_level", "publish", "reset_cache_stats", *_CUDA_DOORS]


def __getattr__(name):
    if name in _CUDA_DOORS:
        from ..backends import need_device

        return getattr(need_device().api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _reset_artifact_memo() -> None:
    """Forget every memoised artifact/closure verdict and the memoised
    :func:`current_payload` — one door back to a fresh, honest check."""
    _cache_reset_artifact_memo()
    _payload_module.reset_payload_memo()
