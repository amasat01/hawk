# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The hawk-cuda extension: the GPU target of hawk (device source, NVRTC/nvcc compile, arch, cards tools).

``hawk.compile.nvrtc``, ``.device``, ``.cubin`` and the device-compiler doors are this package's names."""

from __future__ import annotations

from . import nvrtc
from .nvrtc import DeviceImage, cubin, cubin_available, device
from .toolchain import device_compiler, device_compiler_kind

__all__ = ["DeviceImage", "cubin", "cubin_available", "device", "device_compiler", "device_compiler_kind", "nvrtc"]
