# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The numeric CHECK of every DiffRule-21, gate).

A derivative rule table is the one place where a wrong sign or a swapped
cofactor produces a graph that walks, hashes and emits perfectly and computes
the wrong number, so this row does not inspect the rules — it EVALUATES them.
For every op kind in ``hawk._ir.OP_KINDS`` a small primal DAG is built, run
through :func:`hawk.diff.vjp` and :func:`hawk.diff.jvp`, and both derivative IRs
are evaluated by the scratch numpy interpreter (``tests/_eval.py``) against a
CENTRAL DIFFERENCE of the primal at the same point, to 1e-6 relative. The
coverage assertion at the end is what makes it a gate rather than a sample: the
kinds actually exercised must be the whole vocabulary (``at`` is covered by
``test_diff_cross_sample.py``, whose reverse is a scatter, not a value).

This test previously failed when ``_BIN_VJP['div']``'s second
contribution sign flipped in ``hawk/diff/rules.py``; the rule was restored immediately after.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk import _ir
from hawk.diff import jvp, vjp

#: One context holds every node these rows build, so a row's leaves and constants meet.
CTX = _ir.new_context()
TensorType = _ir.TensorType

RNG = np.random.default_rng(20260904)
H = 1e-5
S = TensorType((), "f64")
I32 = TensorType((), "i32")
#: The type of a factorisation status (``info``): the status compares against a status-typed zero.
INT32 = TensorType((), "int32")
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
    if code == "gen3":
        return 3.0 * np.eye(3) + 0.6 * RNG.normal(size=(3, 3))
    if code == "spd3":
        g = RNG.normal(size=(3, 3))
        return g @ g.T + 3.0 * np.eye(3)
    if code in _RAY_POINTS:
        return (
            np.array(_RAY_POINTS[code])
            if isinstance(_RAY_POINTS[code], list)
            else _RAY_POINTS[code]
        )
    raise AssertionError(code)


#: Fixed scenes for the ray ops: the ray HITS each primitive, well inside it, so
#: the central difference stays on the live branch (no RNG draw: no other row moves).
_RAY_POINTS = {
    "ro": [0.2, 0.1, -2.0],
    "rd": [0.05, -0.02, 1.0],
    "ta": [-1.0, -1.0, 0.0],
    "tb": [1.0, -1.0, 0.2],
    "tc": [0.0, 1.5, -0.1],
    "bo": [-3.0, 0.2, 0.1],
    "bd": [1.0, 0.1, 0.05],
    "blo": [-1.0, -1.0, -1.0],
    "bhi": [1.2, 1.0, 0.9],
    "sc": [0.1, 0.0, 0.0],
    "sr": 1.3,
}

_TTYPE = {
    "s": S,
    "sp": S,
    "su": S,
    "sg1": S,
    "v3": V3,
    "v3p": V3,
    "q": Q,
    "m33": M33,
    "m32": M32,
    "spd3": M33,
    "gen3": M33,
    **{k: (S if k == "sr" else V3) for k in _RAY_POINTS},
}
_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}


def _leaf(name: str, code: str):
    t = _TTYPE[code]
    return _ir.leaf(CTX, "vocab_read", _ROLE[len(t.shape)], name, t)


def _const(value, ttype):
    return _ir.const(CTX, value, ttype)


def _select(cond, on_true, on_false):
    return _ir.make_node("Select", _ir.type_of(on_true), [cond, on_true, on_false])


def _u(kind):
    return lambda x: _ir.make(kind, (x,))


def _b(kind):
    return lambda a, b: _ir.make(kind, (a, b))


def _lane(info_kind, f, use):
    """``use(f)`` where the factorisation ``f`` reports success, zeros elsewhere: the
    user-level read of a status. The status and ``use`` read the SAME factor node, as a
    kernel that factors once does."""
    value = use(f)
    ok = _ir.make("eq", (_ir.make(info_kind, (f,)), _const(0, INT32)))
    return _select(ok, value, _ir.make("splat", (_const(0.0, S),), _ir.type_of(value).shape))


#: The rows of the linear-algebra kinds (each a SPD or general 3x3 at a point
#: the finite difference stays inside the same branch of).
LINALG_CASES = [
    (
        "lu_solve_vector",
        ("gen3", "v3"),
        lambda a, b: _ir.make("lu_solve", (_ir.make("lu", (a,)), b)),
    ),
    (
        "lu_solve_matrix",
        ("gen3", "m32"),
        lambda a, b: _ir.make("lu_solve", (_ir.make("lu", (a,)), b)),
    ),
    (
        "lu_solve_t_vector",
        ("gen3", "v3"),
        lambda a, b: _ir.make("lu_solve_t", (_ir.make("lu", (a,)), b)),
    ),
    (
        "lu_solve_t_matrix",
        ("gen3", "m32"),
        lambda a, b: _ir.make("lu_solve_t", (_ir.make("lu", (a,)), b)),
    ),
    ("lu_det", ("gen3",), lambda a: _ir.make("lu_det", (_ir.make("lu", (a,)),))),
    (
        "lu_info",
        ("gen3", "v3"),
        lambda a, b: _lane("lu_info", _ir.make("lu", (a,)), lambda f: _ir.make("lu_solve", (f, b))),
    ),
    (
        "cho_solve_vector",
        ("spd3", "v3"),
        lambda a, b: _ir.make("cho_solve", (_ir.make("chol", (a,)), b)),
    ),
    (
        "cho_solve_matrix",
        ("spd3", "m32"),
        lambda a, b: _ir.make("cho_solve", (_ir.make("chol", (a,)), b)),
    ),
    ("chol_factor", ("spd3",), lambda a: _ir.make("chol_factor", (_ir.make("chol", (a,)),))),
    ("chol_as_matrix", ("spd3", "m33"), lambda a, b: _ir.make("mm", (_ir.make("chol", (a,)), b))),
    (
        "chol_info",
        ("spd3",),
        lambda a: _lane("chol_info", _ir.make("chol", (a,)), lambda f: _ir.make("chol_factor", (f,))),
    ),
    (
        "chol_from_solve",
        ("gen3", "m32"),
        lambda a, b: _ir.make("cho_solve", (_ir.make("chol_from", (a,)), b)),
    ),
    (
        "chol_from_factor",
        ("gen3",),
        lambda a: _ir.make("chol_factor", (_ir.make("chol_from", (a,)),)),
    ),
    (
        "chol_from_tri",
        ("gen3", "v3"),
        lambda a, b: _ir.make("tri_lower", (_ir.make("chol_from", (a,)), b)),
    ),
    ("tri_lower", ("spd3", "m32"), lambda a, b: _ir.make("tri_lower", (_ir.make("chol", (a,)), b))),
    (
        "tri_upper_t",
        ("spd3", "m32"),
        lambda a, b: _ir.make("tri_upper_t", (_ir.make("chol", (a,)), b)),
    ),
    (
        "tri_lower_vector",
        ("spd3", "v3"),
        lambda a, b: _ir.make("tri_lower", (_ir.make("chol", (a,)), b)),
    ),
    (
        "tri_upper_t_vector",
        ("spd3", "v3"),
        lambda a, b: _ir.make("tri_upper_t", (_ir.make("chol", (a,)), b)),
    ),
    ("eigh_w", ("gen3",), lambda a: _ir.make("eigh_w", (_ir.make("eigh", (a,)),))),
    ("eigh_v", ("gen3",), lambda a: _ir.make("eigh_v", (_ir.make("eigh", (a,)),))),
    (
        "eigh_info",
        ("gen3",),
        lambda a: _lane("eigh_info", _ir.make("eigh", (a,)), lambda f: _ir.make("eigh_w", (f,))),
    ),
    (
        "qr_solve_vector",
        ("m32", "v3"),
        lambda j, r: _ir.make("qr_solve", (_ir.make("qr", (j,)), r)),
    ),
    (
        "qr_solve_matrix",
        ("m32", "m32"),
        lambda j, r: _ir.make("qr_solve", (_ir.make("qr", (j,)), r)),
    ),
    (
        "qr_gram_solve_vector",
        ("m32", "v3"),
        lambda j, r: _ir.make(
            "qr_gram_solve",
            (_ir.make("qr", (j,)), _ir.make("mv", (_ir.make("transpose", (j,)), r))),
        ),
    ),
    (
        "qr_gram_solve_matrix",
        ("m32", "m32"),
        lambda j, b: _ir.make(
            "qr_gram_solve",
            (_ir.make("qr", (j,)), _ir.make("mm", (_ir.make("transpose", (j,)), b))),
        ),
    ),
    ("lstsq_vector", ("m32", "v3"), lambda j, r: _ir.make("lstsq", (j, r))),
    ("lstsq_matrix", ("m32", "m32"), lambda j, r: _ir.make("lstsq", (j, r))),
    (
        "lstsq_gram_solve_vector",
        ("m32", "v3"),
        lambda j, r: _ir.make(
            "lstsq_gram_solve", (j, _ir.make("mv", (_ir.make("transpose", (j,)), r)))
        ),
    ),
    (
        "lstsq_gram_solve_matrix",
        ("m32", "m32"),
        lambda j, b: _ir.make(
            "lstsq_gram_solve", (j, _ir.make("mm", (_ir.make("transpose", (j,)), b)))
        ),
    ),
    (
        "lstsq_info",
        ("m32", "v3"),
        lambda j, r: (
            lambda x: _select(
                _ir.make("eq", (_ir.make("lstsq_info", (j,)), _const(0, INT32))),
                x,
                _ir.make("splat", (_const(0.0, S),), _ir.type_of(x).shape),
            )
        )(_ir.make("lstsq", (j, r))),
    ),
    (
        "qr_info",
        ("m32", "v3"),
        lambda j, r: _lane("qr_info", _ir.make("qr", (j,)), lambda f: _ir.make("qr_solve", (f, r))),
    ),
    # pathwise mvn_sample: mean + L z, z the counter generator's draws
    (
        "mvn_composition",
        ("v3", "spd3"),
        lambda mean, cov: _ir.make(
            "add",
            (
                mean,
                _ir.make(
                    "mv",
                    (
                        _ir.make("chol", (cov,)),
                        _ir.make(
                            "vec",
                            tuple(
                                _ir.make("random_normal", (_const(11, I32), _const(k, I32)))
                                for k in range(3)
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
]

_TMAX = _const(100.0, S)

#: The ray ops: ``[hit, a, b, c]``, checked on a scene that hits.
RAY_CASES = [
    (
        "ray_tri",
        ("ro", "rd", "ta", "tb", "tc"),
        lambda o, d, a, b, c: _ir.make("ray_tri", (o, d, a, b, c, _TMAX)),
    ),
    (
        "ray_aabb",
        ("bo", "bd", "blo", "bhi"),
        lambda o, d, lo, hi: _ir.make("ray_aabb", (o, d, lo, hi, _TMAX)),
    ),
    (
        "ray_sphere",
        ("bo", "bd", "sc", "sr"),
        lambda o, d, c, r: _ir.make("ray_sphere", (o, d, c, r, _TMAX)),
    ),
]

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
    # (`staging/aether/wings/base/render._matrix_sum` spells it `matDot` against a ones matrix).
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
    ("splat", ("s",), lambda x: _ir.make("splat", (x,), (3,))),
    ("component", ("v3",), lambda v: _ir.make("component", (v,), 1)),
    # The RUNTIME-indexed pair. Their index is a rank-0 INTEGER node,
    # so a `_const(_, i32)` stands in for a lowered loop's own symbolic index:
    # the derivative rules never read the index's VALUE (`component_at`'s
    # reverse builds a one-hot by comparing it against every static position),
    # so a constant index exercises the same arithmetic a symbolic one does
    # while keeping the finite difference a function of the traced leaves alone.
    (
        "component_at",
        ("v3",),
        lambda v: _ir.make("component_at", (v, _const(1, TensorType((), "i32")))),
    ),
    (
        "set_component_at",
        ("v3", "s"),
        lambda v, e: _ir.make("set_component_at", (v, _const(2, TensorType((), "i32")), e)),
    ),
    (
        "add_at",
        ("v3", "s"),
        lambda v, e: _ir.make(
            "mul",
            (
                _ir.make("add_at", (v, _const(2, TensorType((), "i32")), e)),
                _ir.make("add_at", (v, _const(0, TensorType((), "i32")), e)),
            ),
        ),
    ),
    ("vec", ("s", "s", "s"), lambda a, b, c: _ir.make("vec", (a, b, c))),
    ("mv", ("m32", "v3"), lambda m, v: _ir.make("mv", (_ir.make("transpose", (m,)), v))),
    ("mm", ("m33", "m32"), _b("mm")),
    ("transpose", ("m32",), _u("transpose")),
    ("outer", ("v3", "v3"), _b("outer")),
    (
        "select_compare",
        ("s", "s"),
        lambda a, b: _select(
            _ir.make("lt", (a, b)), _ir.make("mul", (a, b)), _ir.make("add", (a, b))
        ),
    ),
    (
        "select_logical",
        ("s", "s"),
        lambda a, b: _select(
            _ir.make("land", (_ir.make("le", (a, b)), _ir.make("lnot", (_ir.make("gt", (a, b)),)))),
            _ir.make("tanh", (a,)),
            _ir.make("exp", (b,)),
        ),
    ),
    (
        "select_ne_eq",
        ("s", "s"),
        lambda a, b: _select(
            _ir.make("lor", (_ir.make("ne", (a, b)), _ir.make("eq", (a, b)))),
            _ir.make("mul", (a, a)),
            b,
        ),
    ),
    (
        "select_ge",
        ("s", "s"),
        lambda a, b: _select(_ir.make("ge", (a, b)), _ir.make("sin", (a,)), _ir.make("cos", (b,))),
    ),
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
    (
        "random_uniform_mul",
        ("s",),
        lambda x: _ir.make(
            "mul", (x, _ir.make("random_uniform", (_const(11, I32), _const(4, I32))))
        ),
    ),
    (
        "random_normal_add",
        ("s",),
        lambda x: _ir.make(
            "add", (x, _ir.make("random_normal", (_const(11, I32), _const(4, I32))))
        ),
    ),
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
    ("spacing", ("sp",), _u("spacing")),
    ("hypot", ("s", "s"), _b("hypot")),
    ("copysign_pos", ("s", "sp"), _b("copysign")),
    (
        "copysign_neg",
        ("sp", "s"),
        lambda a, b: _ir.make("copysign", (a, _ir.make("neg", (_ir.make("abs", (b,)),)))),
    ),
    ("fmod", ("sg1", "sp"), _b("fmod")),
    ("fmod_negative", ("sg1", "sp"), lambda a, b: _ir.make("fmod", (_ir.make("neg", (a,)), b))),
    ("remainder", ("sg1", "sp"), _b("remainder")),
    (
        "remainder_negative",
        ("sg1", "sp"),
        lambda a, b: _ir.make("remainder", (a, _ir.make("neg", (b,)))),
    ),
    ("fdim_above", ("sg1", "su"), _b("fdim")),
    ("fdim_below", ("su", "sg1"), _b("fdim")),
    ("fma", ("s", "s", "s"), lambda a, b, c: _ir.make("fma", (a, b, c))),
    # clip's three branches: inside, below lo, above hi (lo/hi offset from x).
    (
        "clip_inside",
        ("su", "sp", "sp"),
        lambda x, lo, hi: _ir.make("clip", (x, _ir.make("neg", (lo,)), hi)),
    ),
    (
        "clip_below",
        ("su", "sp", "sp"),
        lambda x, lo, hi: _ir.make(
            "clip",
            (x, _ir.make("add", (lo, _const(1.0, S))), _ir.make("add", (hi, _const(3.0, S)))),
        ),
    ),
    (
        "clip_above",
        ("sp", "su", "su"),
        lambda x, lo, hi: _ir.make(
            "clip",
            (_ir.make("add", (x, _const(1.0, S))), lo, _ir.make("mul", (hi, _const(0.1, S)))),
        ),
    ),
    (
        "isnan_isinf_isfinite",
        ("s", "s"),
        lambda a, b: _select(
            _ir.make("lor", (_ir.make("isnan", (a,)), _ir.make("isinf", (b,)))),
            b,
            _select(_ir.make("isfinite", (a,)), _ir.make("mul", (a, b)), a),
        ),
    ),
    # Appended last, like the rows above. A matrix entry and a matrix
    # assembled from entries: what a function aether spells at rank 0 only
    # is mapped through on a matrix operand.
    ("component_rank2", ("m32",), lambda m: _ir.make("component", (m,), (2, 1))),
    ("vec_rank2", ("s",) * 6, lambda *xs: _ir.make("vec", xs, (3, 2))),
    # The masked rules on a matrix: the mask is a matrix of rank-0 compares.
    ("abs_rank2", ("m33",), _u("abs")),
    ("max_rank2", ("m33", "m33"), _b("max")),
    ("min_broadcast_rank2", ("s", "m33"), _b("min")),
    # The linear-algebra kinds: every consumer and projection reads ONE
    # factorisation node, and the checked derivative flows through it.
    *LINALG_CASES,
    *RAY_CASES,
]


def _name(leaf) -> str:
    return leaf.field_text("name")


def _kinds_of(node, seen: set) -> set:
    if _ir.is_a(node, "op"):
        seen.add(node.field_text("kind"))
    elif _ir.is_a(node, "select"):
        seen.add("select")
    for child in node.operands:
        _kinds_of(child, seen)
    return seen


def _primal(codes, build):
    leaves = [_leaf(f"x{i}", c) for i, c in enumerate(codes)]
    value = build(*leaves)
    return leaves, (_ir.make_node("Assign", _ir.type_of(value), [value], name="out"),)


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
    env = {_name(leaf): _sample(code) for leaf, code in zip(leaves, codes)}
    seed = RNG.normal(size=_ir.type_of(sinks[0]).shape or ())
    derived = vjp(sinks)
    got = evaluate(derived, {**env, "bar_out": seed})
    for leaf in leaves:
        value = np.asarray(env[_name(leaf)], dtype=float)
        indices = [None] if value.ndim == 0 else list(np.ndindex(value.shape))
        for index in indices:
            h = H * max(1.0, abs(float(value[index] if index else value)))
            plus = _objective(sinks, _perturbed(env, _name(leaf), index, h), seed)
            minus = _objective(sinks, _perturbed(env, _name(leaf), index, -h), seed)
            expect = (plus - minus) / (2 * h)
            actual = np.asarray(got[f"bar_{_name(leaf)}"], dtype=float)
            actual = float(actual if index is None else actual[index])
            assert actual == pytest.approx(expect, rel=1e-6, abs=1e-7), (
                f"VJP of {_name(leaf)}{'' if index is None else list(index)} disagrees "
                f"with the central difference: {actual} vs {expect}"
            )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_jvp_matches_finite_differences(case):
    _, codes, build = case
    leaves, sinks = _primal(codes, build)
    env = {_name(leaf): _sample(code) for leaf, code in zip(leaves, codes)}
    tangents = {
        _name(leaf): (
            RNG.normal()
            if np.ndim(env[_name(leaf)]) == 0
            else RNG.normal(size=np.shape(env[_name(leaf)]))
        )
        for leaf in leaves
    }
    derived = jvp(sinks)
    got = np.asarray(
        evaluate(derived, {**env, **{f"dot_{k}": v for k, v in tangents.items()}})["dot_out"],
        dtype=float,
    )
    plus = {k: np.asarray(v, dtype=float) + H * tangents[k] for k, v in env.items()}
    minus = {k: np.asarray(v, dtype=float) - H * tangents[k] for k, v in env.items()}
    expect = (
        np.asarray(evaluate(sinks, plus)["out"], dtype=float)
        - np.asarray(evaluate(sinks, minus)["out"], dtype=float)
    ) / (2 * H)
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
    # The integer kinds carry no derivative (an integer plane is never
    # differentiated); a float through `cast` has its own finite-difference
    # rows beside the integer-plane rows (tests/test_int_planes.py).
    integer = set(_ir.INT_BINARY + _ir.INT_UNARY + _ir.CAST)
    unchecked = sorted(set(_ir.OP_KINDS) - exercised - {"at", "dispatch"} - integer)
    assert not unchecked, (
        "every op kind must have a case whose derivative is checked against a "
        f"finite difference; unchecked: {unchecked}"
    )


# --------------------------------------------------------------------------
# A subexpression written twice is two nodes made apart that are one value: its
# derivative must count both uses, as it counts a node shared by two consumers.


@hawk.kernel
def _scalar_written_twice(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = x * x + m.sin(x * x)


@hawk.kernel
def _pair_read_twice(a: hawk.Vector[2], b: hawk.Vector[2], y: hawk.Mutable[hawk.Scalar]):
    z = m.cmul(a, b)
    y = m.abs2(z) + m.abs2(m.conj(z))


#: (kernel, a point for each input)
_TWICE = {
    "scalar": (_scalar_written_twice, {"x": np.array(0.7)}),
    "pair": (_pair_read_twice, {"a": np.array([1.3, 0.6]), "b": np.array([0.8, 1.9])}),
}


def _central(kernel, env, name, index, delta):
    """The central difference of the kernel's output in input ``name`` at ``index``."""
    out = []
    for sign in (1, -1):
        shifted = {k: np.array(v, dtype=float) for k, v in env.items()}
        shifted[name][index] += sign * delta
        out.append(float(evaluate(kernel.sinks, shifted)["y"]))
    return (out[0] - out[1]) / (2 * delta)


@pytest.mark.parametrize("which", _TWICE)
def test_a_subexpression_written_twice_has_the_reverse_derivative_of_its_value(which):
    kernel, env = _TWICE[which]
    got = evaluate(vjp(kernel.sinks), {**env, "bar_y": 1.0})
    for name, value in env.items():
        for index in np.ndindex(value.shape):
            assert float(got[f"bar_{name}"][index]) == pytest.approx(
                _central(kernel, env, name, index, H), rel=1e-6, abs=1e-7
            ), (name, index)


@pytest.mark.parametrize("which", _TWICE)
def test_a_subexpression_written_twice_has_the_forward_derivative_of_its_value(which):
    kernel, env = _TWICE[which]
    for name, value in env.items():
        for index in np.ndindex(value.shape):
            direction = {k: np.zeros(v.shape) for k, v in env.items()}
            direction[name][index] = 1.0
            got = evaluate(
                jvp(kernel.sinks),
                {**env, **{f"dot_{k}": d for k, d in direction.items()}},
            )["dot_y"]
            assert float(got) == pytest.approx(
                _central(kernel, env, name, index, H), rel=1e-6, abs=1e-7
            ), (name, index)
