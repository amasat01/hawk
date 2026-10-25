# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's emitters: ONE renderer, a writer per language, a wrapper per execution model.

An INTERNAL package: its ``__all__`` is the small "IR access" surface a
downstream package may read; every other name is implementation detail.
Importing it is not free, and :mod:`hawk` itself never does.
:mod:`hawk.emit.render` produces the body STRING through the spelling seam
(:mod:`hawk.emit.writer`; :mod:`hawk.emit.writer_aether` is the aether C++
writer every built wrapper wraps); :mod:`hawk.emit.wrapper`
carries the two seams and the shared translation-unit assembly;
:mod:`hawk.emit.cuda`/:mod:`hawk.emit.host` are the two built wrappers
and :mod:`hawk.emit.fused` the fused-lane emitter's body path. The targets
that pair them are :data:`hawk.targets.TARGETS`.
"""

# Internal names (`X as X` marks a deliberate re-export not on `__all__`).
from .cuda import CUDA as CUDA
from .cuda import CudaWrapper as CudaWrapper
from .fused import FusedKernel as FusedKernel
from .fused import Lane as Lane
from .fused import compose
from .fused import render_lane_body as render_lane_body
from .host import HostWrapper as HostWrapper
from .render import Body as Body
from .render import render_body
from .wrapper import FAST_ENTRIES_MACRO as FAST_ENTRIES_MACRO
from .wrapper import FLOAT32 as FLOAT32
from .wrapper import FLOAT64 as FLOAT64
from .wrapper import RANGE_SUFFIX as RANGE_SUFFIX
from .wrapper import SCALAR_MODES as SCALAR_MODES
from .wrapper import ScalarMode as ScalarMode
from .wrapper import SegmentSpec as SegmentSpec
from .wrapper import Source as Source
from .wrapper import Wrapper as Wrapper
from .wrapper import literal_table, render_source, scalar_mode
from .writer import Writer as Writer
from .writer_aether import AETHER as AETHER

__all__ = ["compose", "literal_table", "render_body", "render_source", "scalar_mode"]
