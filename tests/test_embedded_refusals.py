# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C target refuses every unbuilt feature BY NAME, before anything renders.

One planted kernel per :data:`hawk.embedded.verify.UNSUPPORTED` key, each
tripping only its own row; the message names the C target, the kernel, the
feature (``unsupported.<key>``) and the alternative an author can take.
"""

from __future__ import annotations

import _deployable as D
import _kernels as K
import pytest

import hawk
import hawk.math as hm
from hawk import Index, Mutable, Param, Scalar, Terminated, Vector, all_samples
from hawk.embedded.verify import UNSUPPORTED, refuse_unsupported
from hawk.emit import compose
from hawk.ext import DEFAULT_KIND, Guard, Kind
from hawk.ir import Assign, HawkError, Leaf, canonical
from hawk.ir.loop_nodes import Loop, LoopCarry, LoopIndex, LoopValue
from hawk.ir.nodes import AccumWrite, Const
from hawk.ir.ops import make
from hawk.trace.kernel import Kernel
from hawk.types import TensorType

F64 = TensorType((), "f64")


@hawk.kernel
def draw(seed: Index, counter: Index, y: Mutable[Scalar]):
    y = hm.random.uniform(seed, counter)


_COPY = hawk.raw_device("mut_y = psc_x;", access="sample_local", returns=Scalar)


@hawk.kernel
def spliced(x: Scalar, y: Mutable[Scalar]):
    y = _COPY(x)


ACTIVE = Kind("verify_active_rows", guard=Guard(active_set=True))


@hawk.kernel(kind=ACTIVE)
def active(omega: Scalar, dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt * omega


@hawk.kernel
def segmented(k: Index, b0: Scalar, b1: Scalar, y: Mutable[Scalar]):
    y = hm.dispatch(k, [b0, b1], policy="segmented")


@hawk.kernel
def lane_a(x: Scalar, p: Mutable[Scalar]):
    p = x * 2.0


@hawk.kernel
def lane_b(x: Scalar, q: Mutable[Scalar]):
    q = x + 2.0


@hawk.kernel(steps=1)
def advance(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + dt
    x = xn
    terminated = xn >= t_final


@hawk.kernel
def int_hypot(a: Index, b: Index, y: Mutable[Index]):
    y = hm.hypot(a, b)


@hawk.kernel
def int_min_max_abs(a: Index, b: Index, y: Mutable[Index]):
    y = hm.minimum(a, b) + hm.maximum(a, b) + abs(a)


def _vector_scatter():
    """A scattered accumulate of a ``Vector[3]``, as IR: hawk's tracer refuses
    ``all_samples(acc)[lane] += v`` for a vector itself (one plane per
    component), so no ``@hawk.kernel`` reaches the node; the C target keeps
    its own refusal behind that one."""
    v = Leaf("vocab_read", "vec_in", "v", TensorType((3,), "f64"))
    lane = Leaf("vocab_read", "per_sample", "lane", TensorType((), "i64"))
    return Kernel("vector_scatter",
                  (AccumWrite("acc", v, lane, TensorType((3,), "f64")),))


@hawk.kernel
def gather_other(x: Scalar, i: Index, y: Mutable[Scalar]):
    y = all_samples(x)[i]


def _zero_trip():
    """A ``range(0)`` loop as IR. The frontend collapses ``for _ in range(0)``
    to the loop's initial values before a ``Loop`` is minted
    (``hawk/trace/astpass.py``), so no ``@hawk.kernel`` reaches the node; the
    IR constructor does, and the C target must still refuse it (MISRA 14.3)."""
    x = Leaf("vocab_read", "per_sample", "x", F64)
    carry = LoopCarry(0, 0, "s", F64)
    loop = Loop(LoopIndex(0, "k"), (carry,), (x,),
                (make("mul", (carry, Const(2.0, F64))),), start=0, stop=0, step=1)
    return Kernel("zero_trip", (Assign("y", LoopValue(loop, 0), F64),))


@hawk.kernel
def tensor_select(v: Vector[3], y: Mutable[Vector[3]]):
    y = hm.where(v > 0.0, v, -v)


def _case(kernel, kind=None):
    name = kernel.name
    kind = kind if kind is not None else getattr(kernel, "kind", DEFAULT_KIND)
    return name, kernel, kind, kernel.walk, kernel.sinks


#: feature -> (planted case, the FIX text the message must name).
CASES = {
    "random": (lambda: _case(draw), "no counter-based draw is built for C yet"),
    "raw_device": (lambda: _case(spliced), "spell the block with hawk.math"),
    "sample_reduction": (lambda: _case(K.energy),
                         "commit the own column and reduce in the caller"),
    "cross_sample_at": (lambda: _case(gather_other),
                        "pass the values the kernel needs as a Table plane"),
    "active_set": (lambda: _case(active), "run without the index map"),
    "segmented": (lambda: _case(segmented), "export each branch as its own kernel"),
    "compensated": (lambda: _case(D.scatter_c, D.COMPENSATED),
                    "use the plain sink policy"),
    "fused_lanes": (lambda: _case(compose([lane_a, lane_b], "fused"), DEFAULT_KIND),
                    "export each lane as its own kernel"),
    "two_path": (lambda: _case(hawk.steps(advance, "auto")),
                 "export `kernel.step` (the single step) or `hawk.steps(step, K)` "
                 "with an integer K"),
    "int_libm": (lambda: _case(int_hypot),
                 "use maximum(minimum(i, hi), lo) (Int min/max are allowed) or "
                 "convert to Real explicitly"),
    "scattered_tensor_accum": (lambda: _case(_vector_scatter()),
                               "declare one scalar accumulate plane per component"),
    "zero_trip_loop": (lambda: _case(_zero_trip()), "give the loop at least one trip"),
    "tensor_select_cond": (lambda: _case(tensor_select), "select per component"),
}


def test_every_unsupported_feature_has_a_planted_kernel():
    assert list(CASES) == list(UNSUPPORTED)


@pytest.mark.parametrize("feature", list(UNSUPPORTED))
def test_each_row_carries_a_why_and_a_specific_fix(feature):
    why, fix = UNSUPPORTED[feature][1]
    assert why and fix and CASES[feature][1] in fix


@pytest.mark.parametrize("feature", list(UNSUPPORTED))
def test_the_c_target_refuses_by_name(feature):
    build, alternative = CASES[feature]
    name, kernel, kind, walk, sinks = build()
    with pytest.raises(HawkError) as err:
        refuse_unsupported(name, kernel, kind, walk, sinks)
    message = str(err.value)
    for needle in ("the C target", f"unsupported.{feature}", f"{name!r}", alternative):
        assert needle in message, (needle, message)


@pytest.mark.parametrize("kernel", [K.drag, K.gather, D.scatter, int_min_max_abs,
                                    advance, hawk.steps(advance, 4)],
                         ids=lambda k: k.name)
def test_a_supported_kernel_passes(kernel):
    refuse_unsupported(kernel.name, kernel, kernel.kind, kernel.walk, kernel.sinks)


def test_a_derivative_twin_passes():
    from hawk.diff import vjp

    sinks = vjp(K.drag)
    refuse_unsupported("drag_vjp", sinks, DEFAULT_KIND, canonical(sinks), sinks)
