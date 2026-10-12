# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-branch cost vector of a ``dispatch`` group, counted from the IR.

Every expected vector below is worked out by hand from the counting rules in
``hawk/ir/cost.py``: ``(arith32, arith64, special32, special64, loads,
gathers)`` per branch.
"""

from __future__ import annotations

import hawk
import hawk.math as M
from hawk.ir import Assign, Dispatch, Leaf, canonical, dispatch_costs
from hawk.ir.cost import kernel_cost
from hawk.ir import make as mk
from hawk.types import TensorType

F32 = TensorType((), "f32")
I32 = TensorType((), "i32")


def _costs(kernel):
    return dispatch_costs(canonical(kernel.sinks))["dispatch0"]


@hawk.kernel
def _mixed(k: hawk.Index, a: hawk.Scalar, x: hawk.Scalar, b: hawk.Scalar,
           y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [a * x, M.exp(x), M.log(x * x + 1.0) * b], policy="switch")


@hawk.kernel
def _vectors(k: hawk.Index, v: hawk.Vector[3], s: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [M.norm(v), M.dot(v, v) * s], policy="switch")


@hawk.kernel
def _shared(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [x * x + 1.0, x * x + 2.0], policy="switch")


@hawk.kernel
def _selector_reused(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [x * k, x], policy="switch")


@hawk.kernel
def _gather(k: hawk.Index, table: hawk.Table[hawk.Scalar], where: hawk.Index, x: hawk.Scalar,
            y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, [table.at(where), x * 2.0], policy="switch")


@hawk.kernel
def _looped(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    acc = 0.0
    for _ in range(4):
        acc = acc * x + 1.0
    y = M.dispatch(k, [acc, x], policy="switch")


def test_scalar_branches_split_into_arithmetic_and_library_calls():
    cost = _costs(_mixed)
    assert [b.as_tuple() for b in cost.branches] == [
        (0, 1, 0, 0, 2, 0),   # a*x: one multiply, two loads
        (0, 0, 0, 1, 1, 0),   # exp(x): one library call, one load
        (0, 3, 0, 1, 2, 0),   # x*x, +1, *b: three ops, log, loads of x and b
    ]


def test_vector_ops_count_their_operation_totals():
    cost = _costs(_vectors)
    assert [b.as_tuple() for b in cost.branches] == [
        (0, 7, 0, 0, 3, 0),   # norm of 3: 2*3 + sqrt, three loads
        (0, 7, 0, 0, 4, 0),   # dot of 3: 2*3, one multiply, loads 3 + s
    ]


def test_a_value_two_branches_share_is_counted_in_each():
    assert [b.as_tuple() for b in _costs(_shared).branches] == [(0, 2, 0, 0, 1, 0)] * 2


def test_what_the_selector_reads_is_left_out():
    cost = _costs(_selector_reused)
    assert [b.as_tuple() for b in cost.branches] == [(0, 1, 0, 0, 1, 0), (0, 0, 0, 0, 1, 0)]
    assert ("per_sample", "k") in cost.selector_reads


def test_an_indexed_read_is_a_gather_and_never_loads_the_plane():
    cost = _costs(_gather)
    assert [b.as_tuple() for b in cost.branches] == [
        (0, 0, 0, 0, 1, 1),   # table.at(where): the index load and one gather
        (0, 1, 0, 0, 1, 0),   # x*2
    ]


def test_a_loop_counts_its_body_once_per_trip():
    cost = _costs(_looped)
    # four trips of (acc*x, +1); x is loaded once, outside the loop
    assert cost.branches[0].as_tuple() == (0, 8, 0, 0, 1, 0)
    assert cost.branches[1].as_tuple() == (0, 0, 0, 0, 1, 0)


def test_32_bit_work_lands_in_the_32_bit_fields():
    a = Leaf("vocab_read", "per_sample", "a", F32)
    k = Leaf("vocab_read", "per_sample", "k", I32)
    d = Dispatch(k, [mk("mul", (a, a)), mk("sin", (a,))], "switch", F32)
    cost = dispatch_costs(canonical((Assign("out", d, F32),)))["dispatch0"]
    assert [b.as_tuple() for b in cost.branches] == [(1, 0, 0, 0, 1, 0), (0, 0, 1, 0, 1, 0)]


def test_the_sidecar_facts_ride_along():
    cost = _costs(_selector_reused)
    assert cost.static_selector is True
    assert cost.selector_reads == (("per_sample", "k"),)


def test_every_group_of_a_kernel_is_costed_under_its_published_name():
    walk = canonical(_mixed.sinks)
    assert set(dispatch_costs(walk)) == {d.name for d in walk.dispatches}


def test_a_library_call_vectorises_the_fused_loop_only_under_the_vector_math_profile():
    walk = canonical(_mixed.sinks)
    assert not dispatch_costs(walk, "native")["dispatch0"].fused_vectorised
    assert not dispatch_costs(walk, "portable")["dispatch0"].fused_vectorised
    assert dispatch_costs(walk, "native-vector-math")["dispatch0"].fused_vectorised


def test_branches_without_a_library_call_vectorise_under_every_profile():
    walk = canonical(_shared.sinks)
    for profile in ("native", "portable", "native-vector-math"):
        assert dispatch_costs(walk, profile)["dispatch0"].fused_vectorised


@hawk.kernel
def _two_chains(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    a = 0.0
    b = 0.0
    for _ in range(8):
        a = a * x + 1.0
        b = b * x + 2.0
    y = M.dispatch(k, [a + b, x], policy="switch")


def test_a_chain_is_the_critical_path_not_the_op_count():
    # two independent carries of 8 trips: 32 multiplies/adds in the loop, but each
    # iteration is a multiply then an add, so the path is 8 x 2, plus the final add
    cost = _costs(_two_chains).branches[0]
    assert cost.arith64 == 33
    assert (cost.chain32, cost.chain64, cost.chainSpecial32, cost.chainSpecial64) == (0, 17, 0, 0)
    assert cost.chain_tuple() == cost.as_tuple() + (0, 17, 0, 0)
    assert _costs(_two_chains).branches[1].chain_tuple()[6:] == (0, 0, 0, 0)


def test_a_library_call_lands_in_the_special_chain():
    cost = _costs(_mixed).branches
    assert [b.chainSpecial64 for b in cost] == [0, 1, 1]
    assert [b.chain64 for b in cost] == [1, 0, 3]   # x*x, +1, then log, then *b: arith path 3


def test_the_loop_chain_is_trip_times_the_carry_depth():
    assert _costs(_looped).branches[0].chain64 == 8


def test_kernel_cost_covers_a_kernel_without_a_dispatch():
    @hawk.kernel
    def plain(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x * x + 1.0

    cost = kernel_cost(canonical(plain.sinks), "native")
    assert (cost.arith64, cost.chain64, cost.vectorised) == (2, 2, True)


def test_work_that_does_not_depend_on_a_carry_adds_nothing_to_the_chain():
    @hawk.kernel
    def gathered(k: hawk.Index, table: hawk.Table[hawk.Scalar], where: hawk.Index,
                 x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = x
        for i in range(8):
            acc = acc + table.at(where + i)    # the index arithmetic is off the recurrence
        y = M.dispatch(k, [acc, x], policy="switch")

    cost = _costs(gathered).branches[0]
    assert (cost.chain32, cost.chain64) == (0, 8)
