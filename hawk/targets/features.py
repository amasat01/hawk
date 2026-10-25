# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The shared feature table: what a kernel can use that a target may not build.

One entry per feature, each a predicate over the kernel and the sentence a
refusal says. A target builds the features it lists in
:attr:`hawk.targets.Target.features` and refuses the rest, by name, before
anything renders. The table starts from what the C target refuses.
"""

from __future__ import annotations

from collections.abc import Callable

from ..emit.spelling import _MATH1, _MATH2, _MATH3
from ..ir.loop_nodes import Loop
from ..ir.nodes import (
    GATHER_ROLES,
    RAW_KIND,
    AccumWrite,
    At,
    Dispatch,
    Op,
    ReduceSamples,
    SampleSum,
    Select,
)
from ..ir.ops import RANDOM_OPS

#: The dtypes of an index Int.
INDEX_INTS = ("i32", "i64")
#: libm kinds that take an integer operand without a double round-trip.
_INT_LIBM_OK = ("abs", "min", "max")


def _first(nodes, test) -> tuple | None:
    return next(((pos, node.kind) for pos, node, _ in nodes if test(node)), None)


def _int_libm(node) -> bool:
    return (isinstance(node, Op) and node.kind not in _INT_LIBM_OK
            and (node.kind in _MATH1 or node.kind in _MATH2 or node.kind in _MATH3)
            and any(o.ttype.dtype in INDEX_INTS for o in node.operands))


def _segmented(kernel, nodes):
    found = _first(nodes, lambda n: isinstance(n, Dispatch) and n.policy == "segmented")
    if found is None and getattr(kernel, "segment", None) is not None:
        return "kernel", "segment"
    return found


def _sample_reduction(walk, nodes):
    found = _first(nodes, lambda n: isinstance(n, (SampleSum, ReduceSamples)))
    if found is None and walk.access.cls == "mapreduce":
        return "kernel", "mapreduce"
    return found


def _cross_sample_at(nodes):
    return _first(nodes, lambda n: isinstance(n, At)
                  and n.operands[0].role not in GATHER_ROLES)


def _whole(flag: bool, what: str):
    return ("kernel", what) if flag else None


#: feature -> (predicate(kernel, kind, walk, nodes) -> (pos, node kind) | None,
#: (why, fix)). Checked in this order; the first hit refuses. A target that
#: lists the feature in its ``features`` builds it; one that does not refuses it.
FEATURES: dict[str, tuple[Callable, tuple[str, str]]] = {
    "random": (
        lambda k, kind, w, nodes: _first(
            nodes, lambda n: isinstance(n, Op) and n.kind in RANDOM_OPS),
        ("a random draw lowers to aether's counter-based generator",
         "no counter-based draw is built for C yet; draw in the caller and pass "
         "the values in as a plane")),
    "raw_device": (
        lambda k, kind, w, nodes: _first(nodes, lambda n: n.kind == RAW_KIND),
        ("a raw_device block splices C++ text the C target cannot check",
         "spell the block with hawk.math")),
    "sample_reduction": (
        lambda k, kind, w, nodes: _sample_reduction(w, nodes),
        ("a reduction over samples needs the wrapper's cross-sample fold, which a "
         "standalone build does not have",
         "commit the own column and reduce in the caller")),
    "cross_sample_at": (
        lambda k, kind, w, nodes: _cross_sample_at(nodes),
        ("an at() read of a per-sample plane reads another sample's value, which a "
         "partition call may not hold",
         "pass the values the kernel needs as a Table plane")),
    "active_set": (
        lambda k, kind, w, nodes: _whole(
            bool(getattr(getattr(kind, "guard", None), "active_set", False)), "guard"),
        ("an active-set guard reads an index map the C entry does not take",
         "run without the index map (the plain mask guard)")),
    "segmented": (
        lambda k, kind, w, nodes: _segmented(k, nodes),
        ("a segmented dispatch is a per-unit split the C target does not run",
         "export each branch as its own kernel")),
    "compensated": (
        lambda k, kind, w, nodes: _whole(
            getattr(getattr(kind, "sink", None), "id", None) == "compensated",
            "sink policy"),
        ("the compensated sink policy commits a correction plane the C target "
         "does not lower", "use the plain sink policy")),
    "fused_lanes": (
        lambda k, kind, w, nodes: _whole(
            getattr(k, "fused_body", None) is not None
            or getattr(k, "lanes", None) is not None, "fused lanes"),
        ("a fused lane group is composed for a launching runtime",
         "export each lane as its own kernel")),
    "two_path": (
        lambda k, kind, w, nodes: _whole(
            getattr(k, "one_step", None) is not None, "automatic kernel"),
        ("an automatic kernel branches once per launch on a runtime step word",
         "export `kernel.step` (the single step) or `hawk.steps(step, K)` with "
         "an integer K")),
    "int_libm": (
        lambda k, kind, w, nodes: _first(nodes, _int_libm),
        ("a libm function on an Int operand takes an implicit double round-trip "
         "a safety build must not rely on",
         "use maximum(minimum(i, hi), lo) (Int min/max are allowed) or convert "
         "to Real explicitly")),
    "scattered_tensor_accum": (
        lambda k, kind, w, nodes: _first(
            nodes, lambda n: isinstance(n, AccumWrite) and n.index is not None
            and len(n.ttype.shape) >= 1),
        ("a scattered accumulate of a rank>=1 value is not lowered",
         "declare one scalar accumulate plane per component")),
    "zero_trip_loop": (
        lambda k, kind, w, nodes: _first(
            nodes, lambda n: isinstance(n, Loop) and n.trip == 0),
        ("a loop that never runs is dead code (MISRA C:2012 Rule 14.3)",
         "give the loop at least one trip")),
    "tensor_select_cond": (
        lambda k, kind, w, nodes: _first(
            nodes, lambda n: isinstance(n, Select)
            and len(n.operands[0].ttype.shape) >= 1),
        ("a select on a rank>=1 condition picks per component, which is not "
         "lowered", "select per component with a rank-0 condition")),
}
