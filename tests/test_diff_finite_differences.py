# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The numeric CHECK of every DiffRule-21, gate).

A derivative rule table is the one place where a wrong sign or a swapped
cofactor produces a graph that walks, hashes and emits perfectly and computes
the wrong number, so this row does not inspect the rules — it EVALUATES them.
For every op kind in ``hawk.ir.ops.OP_KINDS`` a small primal DAG is built, run
through :func:`hawk.diff.vjp` and :func:`hawk.diff.jvp`, and both derivative IRs
are evaluated by the scratch numpy interpreter (``tests/_eval.py``) against a
CENTRAL DIFFERENCE of the primal at the same point, to 1e-6 relative. The
coverage assertion at the end is what makes it a gate rather than a sample: the
kinds actually exercised must be the whole vocabulary (``at`` is covered by
``test_diff_cross_sample.py``, whose reverse is a scatter, not a value).

This test previously failed when ``_BIN_VJP['div']``'s second
contribution sign flipped in ``hawk/hawk/diff/rules.py``; the rule was restored immediately after.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

from hawk.diff import jvp, vjp
from hawk.ir import Assign, Const, Leaf, Node, Op, Select
from hawk.ir import make as mk
from hawk.ir.ops import OP_KINDS
from hawk.types import TensorType

RNG = np.random.default_rng(20260904)
H = 1e-5
S = TensorType((), "f64")
I32 = TensorType((), "i32")
V3 = TensorType((3,), "f64")
Q = TensorType((4,), "f64", "quaternion")
M33 = TensorType((3, 3), "f64")
M32 = TensorType((3, 2), "f64")


def _sample(code: str):
    if code == "s":
        return float(RNG.normal())
    if code == "sp":
        return float(RNG.uniform(0.5, 2.0))
    if code == "su":
        return float(RNG.uniform(-0.7, 0.7))
    if code == "sg1":
        return float(RNG.uniform(1.5, 3.0))
    if code == "v3":
        return RNG.normal(size=3)
    if code == "v3p":
        return RNG.uniform(0.5, 2.0, size=3)
    if code == "q":
        return RNG.normal(size=4) + 2.0
    if code == "m33":
        return RNG.normal(size=(3, 3))
    if code == "m32":
        return RNG.normal(size=(3, 2))
    raise AssertionError(code)


_TTYPE = {"s": S, "sp": S, "su": S, "sg1": S, "v3": V3, "v3p": V3, "q": Q, "m33": M33,
          "m32": M32}
_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}


def _leaf(name: str, code: str) -> Leaf:
    t = _TTYPE[code]
    return Leaf("vocab_read", _ROLE[len(t.shape)], name, t)


def _u(kind):
    return lambda x: mk(kind, (x,))


def _b(kind):
    return lambda a, b: mk(kind, (a, b))


#: ``(id, operand codes, builder)`` — one primal per rule under test.
CASES = [
    ("add", ("v3", "v3"), _b("add")),
    ("sub", ("v3", "v3"), _b("sub")),
    ("mul", ("v3", "v3"), _b("mul")),
    ("div", ("v3", "v3p"), _b("div")),
    ("mul_broadcast", ("s", "v3"), _b("mul")),
    ("div_broadcast", ("v3", "sp"), _b("div")),
    # The un-broadcast path (hawk/diff/transform._reduce_to) is what turns a
    # rank-0 operand's adjoint into a `sum` of every contribution its broadcast
    # produced, and the RANK of that sum is the rank it broadcast INTO. Rank 1
    # is the pair above; rank 2 is these two, and they are here because the
    # rank-2 `sum` they mint is the node aether has no `Expression::sum()` for
    # (`hawk/emit/aether._matrix_sum` spells it `matDot` against a ones matrix).
    # The rank axis of a broadcast is not a detail of one op: it decides which
    # aether reduction the emitter must reach for.
    ("mul_broadcast_rank2", ("s", "m33"), _b("mul")),
    ("div_broadcast_rank2", ("m33", "sp"), _b("div")),
    ("pow", ("sp", "sp"), _b("pow")),
    ("min", ("v3", "v3"), _b("min")),
    ("max", ("v3", "v3"), _b("max")),
    ("atan2", ("s", "sp"), _b("atan2")),
    ("neg", ("v3",), _u("neg")),
    ("abs", ("s",), _u("abs")),
    ("sqrt", ("sp",), _u("sqrt")),
    ("rsqrt", ("sp",), _u("rsqrt")),
    ("exp_rank1", ("v3",), _u("exp")),
    ("log", ("sp",), _u("log")),
    ("sin", ("s",), _u("sin")),
    ("cos", ("s",), _u("cos")),
    ("tan", ("su",), _u("tan")),
    ("tanh", ("s",), _u("tanh")),
    ("asin", ("su",), _u("asin")),
    ("acos", ("su",), _u("acos")),
    ("atan", ("s",), _u("atan")),
    ("dot", ("v3", "v3"), _b("dot")),
    ("cross", ("v3", "v3"), _b("cross")),
    ("norm", ("v3",), _u("norm")),
    ("sum", ("v3",), _u("sum")),
    ("sum_rank2", ("m33",), _u("sum")),
    ("splat", ("s",), lambda x: mk("splat", (x,), (3,))),
    ("component", ("v3",), lambda v: mk("component", (v,), 1)),
    # The RUNTIME-indexed pair. Their index is a rank-0 INTEGER node,
    # so a `Const(_, i32)` stands in for a lowered loop's own symbolic index:
    # the derivative rules never read the index's VALUE (`component_at`'s
    # reverse builds a one-hot by comparing it against every static position),
    # so a constant index exercises the same arithmetic a symbolic one does
    # while keeping the finite difference a function of the traced leaves alone.
    ("component_at", ("v3",),
     lambda v: mk("component_at", (v, Const(1, TensorType((), "i32"))))),
    ("set_component_at", ("v3", "s"),
     lambda v, e: mk("set_component_at",
                     (v, Const(2, TensorType((), "i32")), e))),
    ("vec", ("s", "s", "s"), lambda a, b, c: mk("vec", (a, b, c))),
    ("mv", ("m32", "v3"), lambda m, v: mk("mv", (mk("transpose", (m,)), v))),
    ("mm", ("m33", "m32"), _b("mm")),
    ("transpose", ("m32",), _u("transpose")),
    ("outer", ("v3", "v3"), _b("outer")),
    ("select_compare", ("s", "s"),
     lambda a, b: Select(mk("lt", (a, b)), mk("mul", (a, b)), mk("add", (a, b)), S)),
    ("select_logical", ("s", "s"),
     lambda a, b: Select(
         mk("land", (mk("le", (a, b)), mk("lnot", (mk("gt", (a, b)),)))),
         mk("tanh", (a,)), mk("exp", (b,)), S)),
    ("select_ne_eq", ("s", "s"),
     lambda a, b: Select(mk("lor", (mk("ne", (a, b)), mk("eq", (a, b)))),
                         mk("mul", (a, a)), b, S)),
    ("select_ge", ("s", "s"),
     lambda a, b: Select(mk("ge", (a, b)), mk("sin", (a,)), mk("cos", (b,)), S)),
    ("quat_mul", ("q", "q"), _b("quat_mul")),
    ("quat_conj", ("q",), _u("quat_conj")),
    ("quat_recip", ("q",), _u("quat_recip")),
    ("quat_rotate", ("q", "v3"), _b("quat_rotate")),
    ("as_pure", ("v3",), _u("as_pure")),
    ("as_vec3", ("q",), _u("as_vec3")),
    # lever 2: `floor` joins the explicit-zero bucket (COMPARE/LOGICAL's
    # own treatment, `hawk/diff/rules.py`'s `_zero_vjp`/`_zero_jvp`) — its
    # own output is REAL-typed (unlike a comparison's bool), so, unlike
    # `select_compare`/`select_logical` above, a BARE case is meaningful: away
    # from an integer knot the central difference of `floor(x)` is exactly
    # 0.0, matching the zero rule's `bar_x0`/tangent exactly rather than only
    # approximately. Appended LAST so its RNG draw does not perturb any
    # earlier case's sampled point (the module's `RNG` is shared and drawn
    # from in CASES order).
    ("floor", ("sp",), _u("floor")),
    # `random_uniform`/`random_normal` carry the explicit ZERO rule
    # (COMPARE/LOGICAL/floor's own bucket, hawk/diff/rules.py) — seed/counter
    # are i32 Consts here, so there is no leaf to check a `bar_`/`dot_` plane
    # for either way ( already keeps an integer leaf out of `_wrt`'s
    # selection, exercised on a REAL leaf by `tests/test_random_op.py`'s own
    # row). What this case checks is the SURROUNDING arithmetic: `x`'s
    # derivative through a `mul`/`add` whose OTHER operand is a random draw
    # differentiates exactly as it would through any other rank-0 constant.
    ("random_uniform_mul", ("s",),
     lambda x: mk("mul", (x, mk("random_uniform",
                                (Const(11, I32), Const(4, I32)))))),
    ("random_normal_add", ("s",),
     lambda x: mk("add", (x, mk("random_normal",
                                (Const(11, I32), Const(4, I32)))))),
    # The numpy-parity set, appended after every earlier case so their
    # sampled points stay put. Piecewise kinds get one case per branch the
    # sampled point can reach; the rounding family, `sign` and the class
    # tests are the zero rule, checked bare like `floor`.
    ("exp2", ("s",), _u("exp2")),
    ("expm1", ("s",), _u("expm1")),
    ("log2", ("sp",), _u("log2")),
    ("log10", ("sp",), _u("log10")),
    ("log1p", ("sp",), _u("log1p")),
    ("cbrt", ("sp",), _u("cbrt")),
    ("sinh", ("s",), _u("sinh")),
    ("cosh", ("s",), _u("cosh")),
    ("asinh", ("s",), _u("asinh")),
    ("acosh", ("sg1",), _u("acosh")),
    ("atanh", ("su",), _u("atanh")),
    ("erf", ("s",), _u("erf")),
    ("erfc", ("s",), _u("erfc")),
    ("ceil", ("sp",), _u("ceil")),
    ("trunc", ("sp",), _u("trunc")),
    ("round", ("sp",), _u("round")),
    ("rint", ("sp",), _u("rint")),
    ("sign", ("s",), _u("sign")),
    ("hypot", ("s", "s"), _b("hypot")),
    ("copysign_pos", ("s", "sp"), _b("copysign")),
    ("copysign_neg", ("sp", "s"),
     lambda a, b: mk("copysign", (a, mk("neg", (mk("abs", (b,)),))))),
    ("fmod", ("sg1", "sp"), _b("fmod")),
    ("fmod_negative", ("sg1", "sp"),
     lambda a, b: mk("fmod", (mk("neg", (a,)), b))),
    ("remainder", ("sg1", "sp"), _b("remainder")),
    ("remainder_negative", ("sg1", "sp"),
     lambda a, b: mk("remainder", (a, mk("neg", (b,))))),
    ("fdim_above", ("sg1", "su"), _b("fdim")),
    ("fdim_below", ("su", "sg1"), _b("fdim")),
    ("fma", ("s", "s", "s"), lambda a, b, c: mk("fma", (a, b, c))),
    # clip's three branches: inside, below lo, above hi (lo/hi offset from x).
    ("clip_inside", ("su", "sp", "sp"),
     lambda x, lo, hi: mk("clip", (x, mk("neg", (lo,)), hi))),
    ("clip_below", ("su", "sp", "sp"),
     lambda x, lo, hi: mk("clip", (x, mk("add", (lo, Const(1.0, S))),
                                   mk("add", (hi, Const(3.0, S)))))),
    ("clip_above", ("sp", "su", "su"),
     lambda x, lo, hi: mk("clip", (mk("add", (x, Const(1.0, S))), lo,
                                   mk("mul", (hi, Const(0.1, S)))))),
    ("isnan_isinf_isfinite", ("s", "s"),
     lambda a, b: Select(mk("lor", (mk("isnan", (a,)), mk("isinf", (b,)))), b,
                         Select(mk("isfinite", (a,)), mk("mul", (a, b)), a, S), S)),
    # Appended last, like the rows above. A matrix entry and a matrix
    # assembled from entries: what a function aether spells at rank 0 only
    # is mapped through on a matrix operand.
    ("component_rank2", ("m32",), lambda m: mk("component", (m,), (2, 1))),
    ("vec_rank2", ("s",) * 6, lambda *xs: mk("vec", xs, (3, 2))),
    # The masked rules on a matrix: the mask is a matrix of rank-0 compares.
    ("abs_rank2", ("m33",), _u("abs")),
    ("max_rank2", ("m33", "m33"), _b("max")),
    ("min_broadcast_rank2", ("s", "m33"), _b("min")),
]


def _kinds_of(node: Node, seen: set) -> set:
    if isinstance(node, (Op, Select)) and not isinstance(node, Const):
        seen.add(node.kind)
    for child in node.operands:
        _kinds_of(child, seen)
    return seen


def _primal(codes, build):
    leaves = [_leaf(f"x{i}", c) for i, c in enumerate(codes)]
    value = build(*leaves)
    return leaves, (Assign("out", value, value.ttype),)


def _objective(sinks, env, seed):
    return float(np.sum(np.asarray(evaluate(sinks, env)["out"]) * seed))


def _perturbed(env, name, index, delta):
    shifted = dict(env)
    if index is None:
        shifted[name] = env[name] + delta
    else:
        arr = np.array(env[name], dtype=float)
        arr[index] += delta
        shifted[name] = arr
    return shifted


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_vjp_matches_finite_differences(case):
    _, codes, build = case
    leaves, sinks = _primal(codes, build)
    env = {leaf.name: _sample(code) for leaf, code in zip(leaves, codes)}
    seed = RNG.normal(size=sinks[0].ttype.shape or ())
    derived = vjp(sinks)
    got = evaluate(derived, {**env, "bar_out": seed})
    for leaf in leaves:
        value = np.asarray(env[leaf.name], dtype=float)
        indices = [None] if value.ndim == 0 else list(np.ndindex(value.shape))
        for index in indices:
            h = H * max(1.0, abs(float(value[index] if index else value)))
            plus = _objective(sinks, _perturbed(env, leaf.name, index, h), seed)
            minus = _objective(sinks, _perturbed(env, leaf.name, index, -h), seed)
            expect = (plus - minus) / (2 * h)
            actual = np.asarray(got[f"bar_{leaf.name}"], dtype=float)
            actual = float(actual if index is None else actual[index])
            assert actual == pytest.approx(expect, rel=1e-6, abs=1e-7), (
                f"VJP of {leaf.name}{'' if index is None else list(index)} disagrees "
                f"with the central difference: {actual} vs {expect}"
            )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_jvp_matches_finite_differences(case):
    _, codes, build = case
    leaves, sinks = _primal(codes, build)
    env = {leaf.name: _sample(code) for leaf, code in zip(leaves, codes)}
    tangents = {leaf.name: (RNG.normal() if np.ndim(env[leaf.name]) == 0
                            else RNG.normal(size=np.shape(env[leaf.name])))
                for leaf in leaves}
    derived = jvp(sinks)
    got = np.asarray(evaluate(derived, {**env, **{f"dot_{k}": v
                                                  for k, v in tangents.items()}})
                     ["dot_out"], dtype=float)
    plus = {k: np.asarray(v, dtype=float) + H * tangents[k] for k, v in env.items()}
    minus = {k: np.asarray(v, dtype=float) - H * tangents[k] for k, v in env.items()}
    expect = (np.asarray(evaluate(sinks, plus)["out"], dtype=float)
              - np.asarray(evaluate(sinks, minus)["out"], dtype=float)) / (2 * H)
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-7), (
        f"JVP disagrees with the central difference: {got} vs {expect}"
    )


def test_every_op_kind_is_numerically_checked():
    exercised: set = set()
    for _, codes, build in CASES:
        leaves, sinks = _primal(codes, build)
        _kinds_of(sinks[0], exercised)
    # the reverse of "at" is a scatter, not a value (test_diff_cross_sample.py).
    # "dispatch" has its OWN dedicated FD rows (tests/test_dispatch.py):
    # its selector is a non-differentiable i32 leaf, so this table's per-leaf
    # loop -- which assumes every leaf gets a checked `bar_<name>` plane --
    # never gets one for it ("kind receives no adjoint" means it never does).
    # The case lives beside the rest of the dispatch rows instead of bending
    # that uniform assumption for one kind.
    unchecked = sorted(set(OP_KINDS) - exercised - {"at", "dispatch"})
    assert not unchecked, (
        "every op kind must have a case whose derivative is checked against a "
        f"finite difference; unchecked: {unchecked}"
    )
