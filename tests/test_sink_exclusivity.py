# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A ``mapreduce_partial`` sink may not coexist with any other sink in one
kernel.

``exec_access`` is ONE scalar and ``exec_op`` ONE value per manifest, so a
kernel carrying a
reduce sink beside a mutable or scatter sink would be declared under some
OTHER class, eagle would never fold the partials (``eagle.exec.fold``) and
``Plan.run`` would return un-combined per-partition partials — a WRONG ANSWER,
not a placement cost. ``canonical()`` refuses the mixed set at DAG-validation
time, naming BOTH sinks. A kernel that needs both is two kernels, sequenced by
the plan.

This test previously failed when run against a walk with no sink-set
check — the mixed set built a Walk and inferred ``sample_local`` for a kernel
whose sink was a reduce partial.
"""

from __future__ import annotations

import pytest
from _dags import F64, V3

from hawk.ir import Assign, HawkError, Leaf, MapreducePartial, canonical, infer

_X = Leaf("vocab_read", "vec_in", "x", V3)
_S = Leaf("vocab_read", "per_sample", "s", F64)


def test_mixed_sink_set_is_refused_naming_both_sinks():
    sinks = (Assign("acc", _X, V3), MapreducePartial("loss", _S, "sum"))
    with pytest.raises(HawkError) as excinfo:
        canonical(sinks)
    msg = str(excinfo.value)
    assert "'loss'" in msg and "'acc'" in msg, (
        "the refusal must name BOTH sinks, got: " + msg
    )


def test_mixed_sink_set_is_refused_in_either_declaration_order():
    sinks = (MapreducePartial("loss", _S, "sum"), Assign("acc", _X, V3))
    with pytest.raises(HawkError, match="may not coexist"):
        canonical(sinks)


def test_two_mapreduce_sinks_are_refused():
    sinks = (MapreducePartial("a", _S, "sum"), MapreducePartial("b", _S, "max"))
    with pytest.raises(HawkError, match="may not coexist"):
        canonical(sinks)


def test_lone_mapreduce_sink_is_accepted_and_carries_its_op():
    walk = canonical((MapreducePartial("loss", _S, "sum"),))
    assert walk.access == ("mapreduce", "sum")
    assert infer(walk) == ("mapreduce", "sum")


def test_undeclared_reduction_op_is_refused_at_construction():
    """The op is a DECLARED aether functor, never an inferred ``sum``."""
    with pytest.raises(HawkError, match="not a declared aether functor"):
        MapreducePartial("loss", _S, "mean")
