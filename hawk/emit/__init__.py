# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's emitters: ONE aether renderer, two built backends.

An INTERNAL package: its ``__all__`` is the small "IR access" surface a
downstream package may read; every other name is implementation detail.
Importing it is not free, and :mod:`hawk` itself never does.
:mod:`hawk.emit.aether` produces the body STRING; :mod:`hawk.emit.backend`
carries the two seams and the shared translation-unit assembly;
:mod:`hawk.emit.cuda`/:mod:`hawk.emit.host` are the two built backends
and :mod:`hawk.emit.fused` the fused-lane emitter's body path.
"""

# Internal names (`X as X` marks a deliberate re-export not on `__all__`).
from .aether import Body as Body
from .aether import render_body
from .backend import FAST_ENTRIES_MACRO as FAST_ENTRIES_MACRO
from .backend import FLOAT32 as FLOAT32
from .backend import FLOAT64 as FLOAT64
from .backend import SCALAR_MODES as SCALAR_MODES
from .backend import RANGE_SUFFIX as RANGE_SUFFIX
from .backend import Backend as Backend
from .backend import ScalarMode as ScalarMode
from .backend import SegmentSpec as SegmentSpec
from .backend import Source as Source
from .backend import render_source, scalar_mode
from .cuda import CUDA as CUDA
from .cuda import CudaBackend as CudaBackend
from .fused import FusedKernel as FusedKernel
from .fused import Lane as Lane
from .fused import compose
from .fused import render_lane_body as render_lane_body
from .host import HOST as _HOST
from .host import HostBackend as HostBackend

#: The built backends, by id. ``metal`` is a later slot; no dead stub.
BACKENDS = {b.id: b for b in (CUDA, _HOST)}

__all__ = ["BACKENDS", "compose", "render_body", "render_source", "scalar_mode"]
