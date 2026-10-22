# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Slotwise form (:mod:`hawk.ir.slotwise`): a wide cotangent as a ``vec`` of
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
from _kan import kan_edge_cell
from _kernels import drag_vjp

import hawk.diff.loops as diff_loops
import hawk.diff.transform as diff_transform
from hawk import Index, Mutable, Scalar, Terminated, Vector, kernel
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import Assign, Const, Dispatch, Leaf, Loop, Select, WideWrite, canonical
from hawk.ir import make as mk
from hawk.ir.slotwise import slotwise
from hawk.math import dispatch, sin
from hawk.types import TensorType

S = TensorType((), "f64")
I32 = TensorType((), "i32")
WIDE = 6
VW = TensorType((WIDE,), "f64")
V3 = TensorType((3,), "f64")


def _leaf(name, t, role=None):
    role = role or ("per_sample" if not t.shape else "vec_in")
    return Leaf("vocab_read", role, name, t)


def _comp(v, k):
    return mk("component", (v,), k)


# --------------------------------------------------------------------------
# the corpus: each entry builds its derived sinks (dense, slotwise off)


def _wide_dispatch():
    """A KAN-shaped read: three cases each reading some components of ONE wide
    leaf, so the cotangent of ``theta`` is a rank-1 dispatch of one-hot sums."""
    theta, kd, x = _leaf("theta", VW), _leaf("kd", I32), _leaf("x", S)
    b0 = mk("mul", (_comp(theta, 0), x))
    b1 = mk("add", (mk("mul", (_comp(theta, 1), x)),
                    mk("mul", (_comp(theta, 2), mk("mul", (x, x))))))
    b2 = mk("add", (mk("sin", (mk("mul", (_comp(theta, 3), x)),)), _comp(theta, 0)))
    y = Dispatch(kd, [b0, b1, b2], "switch", S)
    return vjp((Assign("y", y, S),), wrt=("theta", "x"))


def _dispatch_table():
    from test_diff_dispatch_table_vjp import _table_dispatch_primal

    return vjp(_table_dispatch_primal())


def _runtime_index():
    v, idx, s = _leaf("v", V3), _leaf("idx", I32), _leaf("s", S)
    picked = mk("mul", (mk("component_at", (v, idx)), s))
    swapped = mk("mul", (mk("set_component_at", (v, idx, mk("mul", (s, s)))), s))
    return vjp((Assign("y", picked, S), Assign("w", swapped, V3)), wrt=("v", "s"))


def _select_in_select():
    v, s, p, q = _leaf("v", V3), _leaf("s", S), _leaf("p", V3), _leaf("q", V3)
    c1, c2 = mk("gt", (s, Const(0.0, S))), mk("lt", (s, Const(1.0, S)))
    inner = Select(c2, _comp(v, 0), _comp(v, 1), S)
    y = mk("mul", (Select(c1, inner, _comp(v, 2), S), s))
    z = Select(c1, mk("mul", (p, Const(2.0, S))), q, V3)
    return vjp((Assign("y", y, S), Assign("z", z, V3)), wrt=("v", "p", "q", "s"))


def _loop_derived():
    return vjp(kan_edge_cell(3, 4, 2), wrt=("theta", "g"))


def _component_fold():
    """``component`` of vec-valued ``add``/``mul``/``select``/``dispatch``/``sum``,
    each part a different leaf so a misplaced index is a different number."""
    a, b, c, d, e, f = (_leaf(n, S) for n in "abcdef")
    s, kd = _leaf("s", S), _leaf("kd", I32)
    one, two = mk("vec", (a, b, c)), mk("vec", (d, e, f))
    both = mk("add", (mk("mul", (one, s)), mk("sub", (two, one))))
    pick = Dispatch(kd, [both, mk("neg", (two,)), one], "switch", V3)
    chosen = Select(mk("gt", (s, Const(0.0, S))), pick, two, V3)
    return (Assign("c0", _comp(chosen, 0), S), Assign("c2", _comp(chosen, 2), S),
            Assign("t", mk("sum", (chosen,)), S))


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
        mp.setattr(diff_transform, "slotwise", lambda roots: tuple(roots))
        mp.setattr(diff_loops, "slotwise", lambda roots: tuple(roots))
        return tuple(build())
    finally:
        mp.undo()


def _env(sinks, rng):
    """A random value for every leaf the sinks read (an integer leaf in 0..2)."""
    env = {}
    for leaf in canonical(tuple(sinks)).leaves:
        t = leaf.ttype
        if leaf.role in ("lookup", "wide_in"):
            env[leaf.name] = rng.normal(size=64)
        elif t.dtype in ("i32", "i64"):
            env[leaf.name] = int(rng.integers(0, 3)) if not t.shape \
                else rng.integers(0, 3, size=t.shape)
        else:
            env[leaf.name] = float(rng.normal()) if not t.shape \
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
        want, got = evaluate(dense, env, lanes=64), evaluate(wide, env, lanes=64)
        assert want.keys() == got.keys()
        for key in want:
            assert _same(want[key], got[key]), f"{name}: sink {key!r}"


@pytest.mark.parametrize("name", ["wide_dispatch", "runtime_index", "select_in_select",
                                  "component_fold"])
def test_the_pass_changes_the_wide_rows(name):
    """Or A1 compares a graph with itself."""
    dense = _dense(CORPUS[name])
    assert any(a is not b for a, b in zip(dense, slotwise(dense)))


def test_a_non_finite_factor_leaves_an_untouched_slot_zero_not_nan():
    """The one admitted behavioural change: a one-hot ``vec`` scaled by an
    ``inf`` was NaN in its zero slots, a structural zero is 0."""
    g, scale = _leaf("g", S), _leaf("scale", S)
    zero = Const(0.0, S)
    onehot = mk("vec", (zero, g, zero))
    dense = (Assign("o", mk("mul", (onehot, scale)), V3),)
    env = {"g": 2.0, "scale": np.inf}
    with np.errstate(invalid="ignore"):  # 0 * inf in the dense form is the NaN this test shows
        old, new = evaluate(dense, env)["o"], evaluate(slotwise(dense), env)["o"]
    assert np.isnan(old[0]) and new[0] == 0.0 and np.isinf(new[1])


def test_no_vec_combination_is_a_no_op_and_the_pass_is_idempotent():
    plain = tuple(drag_vjp())
    assert all(a is b for a, b in zip(slotwise(plain), plain))
    assert (render_body(slotwise(plain), canonical(slotwise(plain))).text
            == render_body(plain, canonical(plain)).text)
    once = slotwise(_dense(_wide_dispatch))
    again = slotwise(once)
    assert all(a is b for a, b in zip(once, again))


def test_a_loop_is_not_entered():
    dense = _dense(_loop_derived)
    loops = [s for s in dense if isinstance(s, Loop)]
    kept = [s for s in slotwise(dense) if isinstance(s, Loop)]
    assert loops and all(a is b for a, b in zip(loops, kept))


# --------------------------------------------------------------------------
# the emitted stores


def _body(sinks):
    return render_body(tuple(sinks), canonical(tuple(sinks))).text


def test_a_wide_cotangent_is_committed_per_component():
    sinks = _wide_dispatch()
    text = _body(sinks)
    stores = [ln for ln in text.splitlines() if "mut_bar_theta(" in ln]
    assert len(stores) == WIDE, text
    assert "mut_bar_theta[i]" not in text
    assert "aether::Item" not in text.split("mut_bar_theta(")[0].split("switch")[0]


def _two_edges():
    """Two independent switches, each owning a different pair of slots of one
    wide leaf: the KAN shape at two edges."""
    theta, x = _leaf("theta", VW), _leaf("x", S)
    k1, k2 = _leaf("k1", I32), _leaf("k2", I32)
    one = Dispatch(k1, [mk("mul", (_comp(theta, 0), x)),
                        mk("mul", (_comp(theta, 1), x))], "switch", S)
    two = Dispatch(k2, [mk("mul", (_comp(theta, 2), x)),
                        mk("mul", (_comp(theta, 3), x))], "switch", S)
    return vjp((Assign("y", mk("add", (one, two)), S),), wrt=("theta",))


def test_each_store_follows_its_part_not_the_end_of_the_body():
    """Edge one's slots are stored before edge two's switch runs; with the
    commits left to the end of the body, no store would precede it."""
    lines = _body(_two_edges()).splitlines()
    switches = [n for n, ln in enumerate(lines) if "switch (" in ln]
    store = next(n for n, ln in enumerate(lines) if "mut_bar_theta(0," in ln)
    assert len(switches) == 2 and switches[0] < store < switches[1], (switches, store)


@kernel
def _kan_pair(theta: Vector[6], x: Vector[3], k0: Index, k1: Index, k2: Index,
              terminated: Terminated, y: Mutable[Scalar]):
    """Three edges, each a two-case switch over its own slots of ``theta`` and
    its own ``x``: the cotangents of ``theta`` and ``x`` read the same switches,
    behind the termination mask."""
    e0 = dispatch(k0, [theta[0] * x[0], theta[1] * sin(x[0])])
    e1 = dispatch(k1, [theta[2] * x[1], theta[3] * sin(x[1])])
    e2 = dispatch(k2, [theta[4] * x[2], theta[5] * sin(x[2])])
    y = e0 + e1 + e2


def test_a_store_whose_part_is_ready_early_is_not_left_to_its_turn():
    """The part ``mask ? 0 : switch_result`` of ``bar_x`` sits, in walk order,
    after the whole subtree of ``bar_theta``; it is ready as soon as its
    switch has run, and its store follows that switch (as ``bar_theta``'s do)
    rather than waiting for the end of the body."""
    lines = _body(vjp(_kan_pair, wrt=("theta", "x"))).splitlines()
    switches = [n for n, ln in enumerate(lines) if "switch (" in ln]
    store = next(n for n, ln in enumerate(lines) if "mut_bar_x(0," in ln)
    assert len(switches) == 3 and store < switches[1], (switches, store)


def test_a_plane_that_is_read_keeps_its_whole_store():
    """The plane's prior value is read after the early store would have
    overwritten it: such a sink commits as one value, as before."""
    theta, x = _leaf("theta", VW), _leaf("x", S)
    prior = Leaf("prior_read", "mutable", "acc", VW)
    wide = mk("vec", tuple(mk("mul", (x, Const(float(k), S))) for k in range(WIDE)))
    sinks = (Assign("acc", mk("add", (prior, wide)), VW),)
    del theta
    assert "acc(" not in _body(sinks)
    sinks = (Assign("acc", wide, VW),)
    assert _body(sinks).count("mut_acc(") == WIDE


def test_a_wide_write_is_committed_per_component_too():
    x = _leaf("x", S)
    wide = mk("vec", tuple(mk("mul", (x, Const(float(k), S))) for k in range(3)))
    text = _body((WideWrite("w", wide, None, V3),))
    assert text.count("out_w(") + text.count("wout_w(") + text.count("w(") >= 3


def test_a_matrix_is_committed_per_entry_with_both_indices():
    x = _leaf("x", S)
    wide = mk("vec", tuple(mk("mul", (x, Const(float(k), S))) for k in range(6)),
              literal=(2, 3))
    text = _body((Assign("m", wide, TensorType((2, 3), "f64")),))
    assert "mut_m(1, 2, i.global())" in text and "mut_m[i]" not in text


# --------------------------------------------------------------------------
# the runtime form: an ``add_at`` chain over a slot-form base


def _chains():
    """Chains met by every rule the pass pushes them through, and by the ones
    that force a static slot (``component``, ``sum``, a vector product)."""
    k1, k2, kd = _leaf("k1", I32), _leaf("k2", I32), _leaf("kd", I32)
    e1, e2, s, p, q, r = (_leaf(n, S) for n in ("e1", "e2", "s", "p", "q", "r"))
    zero3 = mk("splat", (Const(0.0, S),), (3,))
    a = mk("add_at", (zero3, k1, e1))
    b = mk("add_at", (mk("add_at", (mk("vec", (p, q, r)), k2, e2)), k1, e1))
    cond = mk("gt", (s, Const(0.0, S)))
    out = {
        "add": mk("add", (a, b)), "sub": mk("sub", (a, b)), "neg": mk("neg", (b,)),
        "mul": mk("mul", (a, s)), "mul_left": mk("mul", (s, a)),
        "div": mk("div", (a, s)), "select": Select(cond, a, b, V3),
        "dispatch": Dispatch(kd, [a, b, mk("vec", (p, q, r))], "switch", V3),
        "mask": Select(mk("lt", (s, Const(1.0, S))), zero3, b, V3),
    }
    sinks = [Assign(f"o_{n}", v, V3) for n, v in out.items()]
    sinks += [Assign("c1", _comp(b, 1), S), Assign("sum", mk("sum", (a,)), S),
              Assign("prod", mk("mul", (a, b)), V3)]
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
    wide = {s.name: s.value for s in slotwise(_chains())}
    for name in ("o_add", "o_sub", "o_neg", "o_mul", "o_div", "o_select",
                 "o_dispatch", "o_mask"):
        assert wide[name].kind == "add_at", name
    for name in ("c1", "sum", "prod"):
        kinds = {n.kind for n in _nodes((Assign("x", wide[name], wide[name].ttype),))}
        assert "add_at" not in kinds, name


def _nodes(sinks):
    from hawk.ir.walk import canonical_nodes

    return canonical_nodes(tuple(sinks))[0]


def test_a_chain_pass_is_idempotent():
    once = slotwise(_chains())
    assert all(a is b for a, b in zip(once, slotwise(once)))


def test_a_runtime_read_gradient_is_a_chain_not_a_one_hot_per_read():
    """The reverse of ``component_at`` is ``add_at`` over a zero: a kernel with
    many runtime reads of one wide plane has a gradient of a few nodes per read,
    not ``width`` selects per read."""
    kinds = [getattr(n, "kind", "") for n in _nodes(_dense(_runtime_index))]
    assert "add_at" in kinds and "eq" not in kinds, kinds


def test_a_lowered_chain_sums_its_links_in_link_order():
    """Three links on ONE slot whose sum depends on the order (``1 + 1e16 - 1e16``
    is 0 left to right, 1 reversed): the lowering keeps the chain's order."""
    k = Const(0, I32)
    chain = mk("splat", (Const(0.0, S),), (3,))
    for value in (1.0, 1e16, -1e16):
        chain = mk("add_at", (chain, k, Const(value, S)))
    sinks = (Assign("c0", _comp(chain, 0), S),)
    assert evaluate(sinks, {})["c0"] == evaluate(slotwise(sinks), {})["c0"] == 0.0
