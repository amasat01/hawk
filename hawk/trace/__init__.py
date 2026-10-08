# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The authoring surface: Python -> IR by operator-overload tracing.

``@kernel`` traces a decorated function whose PARAMETERS are its declared
vocabulary and whose body is rewritten by the ``ast`` pass into ``select``
nodes; ``@raw_device`` is the bounded escape hatch for opaque text.
Importing this package imports :mod:`hawk.ir`, so :mod:`hawk` itself binds
these names LAZILY and a light consumer keeps paying only for
:mod:`hawk.types`.

LOOPS: a body loops with ``for k in range(...)``, a compile-time cap. A
sample can leave early with ONE ``if cond: break`` as a top-level loop
statement, keeping the values its carried names held at the break, as in
Python. ``while`` is refused — write
``for _ in range(CAP): if lnot(cond): break`` instead. ``continue``, a
nested ``break``, and a second ``break`` in one loop are refused too.

The ``break`` is a real per-sample exit:

* on the CPU, a data-dependent exit can stop the compiler vectorising
  across samples (measure both forms — whether it does depends on the
  body);
* on the GPU, each thread stops issuing work at its ``break``, but a warp
  waits for its slowest lane, so the saving is per WARP — largest when
  samples that stop together sit together.

Forward mode (``jvp``) follows each sample's own iterations. Reverse mode
(``vjp``) supports a ``break`` only as the loop body's FIRST or LAST
statement, refusing one in the middle by name — the derivative is
undefined where a sample's exit step changes.

STEPS PER LAUNCH: ``hawk.steps(kernel, K)`` (or ``@hawk.kernel(steps=K)``)
derives a kernel that runs ``K`` steps of a finishing step kernel per
launch — see :mod:`hawk.trace.steps` for the loop, bit-identity and
``"auto"`` mechanics.
"""

# Every import below is "redundant"-aliased (`X as X`) ON PURPOSE: each is
# imported here for PLUMBING, never a public re-export — `hawk/__init__.py`'s
# lazy `__getattr__` resolves `hawk.Quantity`/`hawk.Table` as
# `getattr(trace, name)`, so the attribute must exist here, but its PUBLIC
# path is the top level, never `hawk.trace.<name>` (hence absent from
# `__all__`). The self-alias is the standard "yes, this is unused by this
# file, on purpose" static-analysis idiom.
from ..ir import Quantity as Quantity
from .decl import Accum as Accum
from .decl import Index as Index
from .decl import Matrix as Matrix
from .decl import Mutable as Mutable
from .decl import Param as Param
from .decl import Quat as Quat
from .decl import Reduce as Reduce
from .decl import Scalar as Scalar
from .decl import Staged as Staged
from .decl import Table as Table
from .decl import Terminated as Terminated
from .decl import Vector as Vector
from .decl import Wide as Wide
from .decl import WideOut as WideOut
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .kernel import trace_value as trace_value
from .raw import RawBlock as RawBlock
from .raw import raw_device as raw_device
from .value import Value as Value
from .value import random_normal as random_normal
from .value import random_uniform as random_uniform
from .value import vec as vec

# Nothing here is public: the authoring names live at the top level
# (`hawk.<name>`) and the math functions at `hawk.math`.
__all__ = []
