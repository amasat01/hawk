# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Slotwise form (``hawk._ir.slotwise``): a wide cotangent as a ``vec`` of
scalar parts, committed where each part is ready.

Three rows. The VALUE row evaluates a derived IR in its dense form and in its
slotwise form at the same point and compares every sink (``-0 == +0``,
NaN == NaN): the pass may change the shape of the graph and nothing it
computes. The SHAPE row pins what the pass does and does not touch (no-op,
idempotence, a non-finite seed). The EMIT row pins where the own-column stores
land."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate
from _spline import spline_edge_cell

from hawk.diff import jvp, vjp
from hawk import _ir

#: One context holds every node these rows build, so a row's leaves meet in one graph.
CTX = _ir.new_context()
#: The pass itself: the rows switch it off at the derivative's call sites and run it by hand.
slotwise = _ir.slotwise
TensorType = _ir.TensorType
S = TensorType((), "f64")
I32 = TensorType((), "i32")
WIDE = 6
VW = TensorType((WIDE,), "f64")
V3 = TensorType((3,), "f64")


def _leaf(name, t, role=None):
    role = role or ("plane_in" if not t.shape else "plane_in")
    return _ir.leaf(CTX, "vocab_read", role, name, t)


def _comp(v, k):
    return _ir.make("component", (v,), k)


def _assign(name, value, ttype):
    return _ir.make_node("Assign", ttype, (value,), name=name)


def _const(value, ttype):
    return _ir.const(CTX, value, ttype)


def _select(cond, on_true, on_false, ttype):
    return _ir.make_node("Select", ttype, [cond, on_true, on_false])


def _dispatch(selector, branches, ttype):
    return _ir.make_node("Dispatch", ttype, [selector, *branches])


# --------------------------------------------------------------------------
# the corpus: each entry builds its derived sinks (dense, slotwise off)


def _wide_dispatch():
    """A spline-shaped read: three cases each reading some components of ONE wide
    leaf, so the cotangent of ``theta`` is a rank-1 dispatch of one-hot sums."""
    theta, kd, x = _leaf("theta", VW), _leaf("kd", I32), _leaf("x", S)
    b0 = _ir.make("mul", (_comp(theta, 0), x))
    b1 = _ir.make("add", (_ir.make("mul", (_comp(theta, 1), x)),
                    _ir.make("mul", (_comp(theta, 2), _ir.make("mul", (x, x))))))
    b2 = _ir.make("add", (_ir.make("sin", (_ir.make("mul", (_comp(theta, 3), x)),)), _comp(theta, 0)))
    y = _dispatch(kd, [b0, b1, b2], S)
    return vjp((_assign("y", y, S),), wrt=("theta", "x"))


def _dispatch_table():
    from test_diff_dispatch_table_vjp import _table_dispatch_primal

    return vjp(_table_dispatch_primal())


def _runtime_index():
    v, idx, s = _leaf("v", V3), _leaf("idx", I32), _leaf("s", S)
    picked = _ir.make("mul", (_ir.make("component_at", (v, idx)), s))
    swapped = _ir.make("mul", (_ir.make("set_component_at", (v, idx, _ir.make("mul", (s, s)))), s))
    return vjp((_assign("y", picked, S), _assign("w", swapped, V3)), wrt=("v", "s"))


def _select_in_select():
    v, s, p, q = _leaf("v", V3), _leaf("s", S), _leaf("p", V3), _leaf("q", V3)
    c1, c2 = _ir.make("gt", (s, _const(0.0, S))), _ir.make("lt", (s, _const(1.0, S)))
    inner = _select(c2, _comp(v, 0), _comp(v, 1), S)
    y = _ir.make("mul", (_select(c1, inner, _comp(v, 2), S), s))
    z = _select(c1, _ir.make("mul", (p, _const(2.0, S))), q, V3)
    return vjp((_assign("y", y, S), _assign("z", z, V3)), wrt=("v", "p", "q", "s"))


def _loop_derived():
    return vjp(spline_edge_cell(3, 4, 2), wrt=("theta", "g"))


def _component_fold():
    """``component`` of vec-valued ``add``/``mul``/``select``/``dispatch``/``sum``,
    each part a different leaf so a misplaced index is a different number."""
    a, b, c, d, e, f = (_leaf(n, S) for n in "abcdef")
    s, kd = _leaf("s", S), _leaf("kd", I32)
    one, two = _ir.make("vec", (a, b, c)), _ir.make("vec", (d, e, f))
    both = _ir.make("add", (_ir.make("mul", (one, s)), _ir.make("sub", (two, one))))
    pick = _dispatch(kd, [both, _ir.make("neg", (two,)), one], V3)
    chosen = _select(_ir.make("gt", (s, _const(0.0, S))), pick, two, V3)
    return (_assign("c0", _comp(chosen, 0), S), _assign("c2", _comp(chosen, 2), S),
            _assign("t", _ir.make("sum", (chosen,)), S))


def _jvp_of_vjp():
    return jvp(_wide_dispatch(), wrt=("theta", "x"))


CORPUS = {"wide_dispatch": _wide_dispatch, "dispatch_table": _dispatch_table,
          "runtime_index": _runtime_index, "select_in_select": _select_in_select,
          "loop_derived": _loop_derived, "jvp_of_vjp": _jvp_of_vjp,
          "component_fold": _component_fold}


def _dense(build):
    """``build()`` with the pass switched off at both call sites."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(_ir, "slotwise", lambda roots: tuple(roots))
        return tuple(build())
    finally:
        mp.undo()


def _extents(leaf):
    """The shape of the array bound to a plane: its declared axes, a run-time axis being 64 long."""
    return tuple(64 if extent is None else extent for _label, extent in _ir.field_pairs(leaf, "axes")) or (64,)


def _shapes(sinks):
    """The shape of each sink stored at several indices (a loop's body sinks included), from its axes."""
    shapes = {}
    for node in _ir.canonical_nodes(tuple(sinks))[0]:
        stored = node.field_nodes("body_sinks") if _ir.is_a(node, "loop") else [node]
        for sink in stored:
            if _ir.is_a(sink, "update") and len(sink.operands) > 1:
                axes = _ir.field_pairs(sink, "axes")
                shapes[sink.field_text("name")] = tuple(64 if e is None else e for _label, e in axes) or (64,)
    return shapes


def _env(sinks, rng):
    """A random value for every leaf the sinks read (an integer leaf in 0..2)."""
    env = {}
    walk = _ir.canonical(tuple(sinks))
    leaves = {n.field_text("name"): n for n in _ir.canonical_nodes(tuple(sinks))[0] if _ir.is_a(n, "leaf")}
    for role, name in _ir.arg_spec(walk):
        t = _ir.slot_type(walk, role, name)
        if role in ("lookup", "wide_in"):
            env[name] = rng.normal(size=_extents(leaves[name]))
        elif t.dtype in ("i32", "i64"):
            env[name] = int(rng.integers(0, 3)) if not t.shape \
                else rng.integers(0, 3, size=t.shape)
        else:
            env[name] = float(rng.normal()) if not t.shape \
                else rng.normal(size=t.shape)
    return env


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.mark.parametrize("name", CORPUS)
def test_slotwise_form_evaluates_as_the_dense_form(name):
    dense = _dense(CORPUS[name])
    wide = slotwise(dense)
    rng = np.random.default_rng(20261021)
    for _trial in range(4):
        env = _env(dense, rng)
        shapes = _shapes(dense)
        want = evaluate(dense, env, lanes=64, shapes=shapes)
        got = evaluate(wide, env, lanes=64, shapes=shapes)
        assert want.keys() == got.keys()
        for key in want:
            assert _same(want[key], got[key]), f"{name}: sink {key!r}"


@pytest.mark.parametrize("name", ["wide_dispatch", "runtime_index", "select_in_select",
                                  "component_fold"])
def test_the_pass_changes_the_wide_rows(name):
    """Or A1 compares a graph with itself."""
    dense = _dense(CORPUS[name])
    assert any(a != b for a, b in zip(dense, slotwise(dense)))


def test_a_non_finite_factor_leaves_an_untouched_slot_zero_not_nan():
    """The one admitted behavioural change: a one-hot ``vec`` scaled by an
    ``inf`` was NaN in its zero slots, a structural zero is 0."""
    g, scale = _leaf("g", S), _leaf("scale", S)
    zero = _const(0.0, S)
    onehot = _ir.make("vec", (zero, g, zero))
    dense = (_assign("o", _ir.make("mul", (onehot, scale)), V3),)
    env = {"g": 2.0, "scale": np.inf}
    with np.errstate(invalid="ignore"):  # 0 * inf in the dense form is the NaN this test shows
        old, new = evaluate(dense, env)["o"], evaluate(slotwise(dense), env)["o"]
    assert np.isnan(old[0]) and new[0] == 0.0 and np.isinf(new[1])


def test_a_loop_is_not_entered():
    dense = _dense(_loop_derived)
    loops = [s for s in dense if _ir.is_a(s, "loop")]
    kept = [s for s in slotwise(dense) if _ir.is_a(s, "loop")]
    assert loops and all(a == b for a, b in zip(loops, kept))


# --------------------------------------------------------------------------
# the emitted stores


# --------------------------------------------------------------------------
# the runtime form: an ``add_at`` chain over a slot-form base


def _chains():
    """Chains met by every rule the pass pushes them through, and by the ones
    that force a static slot (``component``, ``sum``, a vector product)."""
    k1, k2, kd = _leaf("k1", I32), _leaf("k2", I32), _leaf("kd", I32)
    e1, e2, s, p, q, r = (_leaf(n, S) for n in ("e1", "e2", "s", "p", "q", "r"))
    zero3 = _ir.make("splat", (_const(0.0, S),), (3,))
    a = _ir.make("add_at", (zero3, k1, e1))
    b = _ir.make("add_at", (_ir.make("add_at", (_ir.make("vec", (p, q, r)), k2, e2)), k1, e1))
    cond = _ir.make("gt", (s, _const(0.0, S)))
    out = {
        "add": _ir.make("add", (a, b)), "sub": _ir.make("sub", (a, b)), "neg": _ir.make("neg", (b,)),
        "mul": _ir.make("mul", (a, s)), "mul_left": _ir.make("mul", (s, a)),
        "div": _ir.make("div", (a, s)), "select": _select(cond, a, b, V3),
        "dispatch": _dispatch(kd, [a, b, _ir.make("vec", (p, q, r))], V3),
        "mask": _select(_ir.make("lt", (s, _const(1.0, S))), zero3, b, V3),
    }
    sinks = [_assign(f"o_{n}", v, V3) for n, v in out.items()]
    sinks += [_assign("c1", _comp(b, 1), S), _assign("sum", _ir.make("sum", (a,)), S),
              _assign("prod", _ir.make("mul", (a, b)), V3)]
    return tuple(sinks)


def test_a_chain_evaluates_as_the_dense_graph_through_every_rule():
    sinks = _chains()
    wide = slotwise(sinks)
    rng = np.random.default_rng(11)
    for _trial in range(6):
        env = _env(sinks, rng)
        want, got = evaluate(sinks, env), evaluate(wide, env)
        assert want.keys() == got.keys()
        for key in want:
            # a slot hit by a static part and links sums them in readiness order
            # (the ULP contract of three or more contributions)
            a, b = np.ravel(want[key]), np.ravel(got[key])
            assert all(x == y or abs(int(np.float64(x).view(np.int64))
                                     - int(np.float64(y).view(np.int64))) <= 4
                       for x, y in zip(a, b)), key


def test_a_chain_is_kept_where_a_commit_can_take_it_and_lowered_where_not():
    wide = {s.field_text("name"): s.operands[0] for s in slotwise(_chains())}
    for name in ("o_add", "o_sub", "o_neg", "o_mul", "o_div", "o_select",
                 "o_dispatch", "o_mask"):
        assert wide[name].field_text("kind") == "add_at", name
    for name in ("c1", "sum", "prod"):
        kinds = {_kind(n) for n in _nodes((_assign("x", wide[name], _ir.type_of(wide[name])),))}
        assert "add_at" not in kinds, name


def _nodes(sinks):
    return _ir.post_order(sinks)


def _kind(node):
    """The op kind of ``node``, or an empty string for a node that is not an op."""
    return node.field_text("kind") if _ir.is_a(node, "op") else ""


def test_a_chain_pass_is_idempotent():
    once = slotwise(_chains())
    assert all(a == b for a, b in zip(once, slotwise(once)))


def test_a_runtime_read_gradient_is_a_chain_not_a_one_hot_per_read():
    """The reverse of ``component_at`` is ``add_at`` over a zero: a kernel with
    many runtime reads of one wide plane has a gradient of a few nodes per read,
    not ``width`` selects per read."""
    kinds = [_kind(n) for n in _nodes(_dense(_runtime_index))]
    assert "add_at" in kinds and "eq" not in kinds, kinds


def test_a_lowered_chain_sums_its_links_in_link_order():
    """Three links on ONE slot whose sum depends on the order (``1 + 1e16 - 1e16``
    is 0 left to right, 1 reversed): the lowering keeps the chain's order."""
    k = _const(0, I32)
    chain = _ir.make("splat", (_const(0.0, S),), (3,))
    for value in (1.0, 1e16, -1e16):
        chain = _ir.make("add_at", (chain, k, _const(value, S)))
    sinks = (_assign("c0", _comp(chain, 0), S),)
    assert evaluate(sinks, {})["c0"] == evaluate(slotwise(sinks), {})["c0"] == 0.0
