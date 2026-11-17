# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative rules and the primitive rules over ``_ir`` against the old Python functions.

Each entry shape of the old ``raptor.rules`` is run on one graph built twice (old classes and
``_ir``) and the two results are compared as plain data. The kind-by-kind equality of the library's
templates is pinned in the library's own tests and is not repeated here.
"""

from __future__ import annotations

import pytest
from raptor import ir as old
from raptor.ir import kinds as old_kinds
from raptor.ir import nodes as old_nodes
from raptor.ir import types as old_types
from raptor.rules import core as old_core
from raptor.rules import primitive as old_prim
from test_ir_module import OLD_F64, new_shape, old_shape

from hawk import _ir
from hawk._errors import HawkError
from hawk.diff import rules
from hawk.ext import _primitive_rules as prim

F64 = ((), "f64", None)


def vec(n):
    """Return the old and the ``_ir`` type of an ``n``-vector of f64."""
    return old_types.TensorType((n,), "f64"), ((n,), "f64", None)


@pytest.fixture
def ctx():
    """Make a fresh context."""
    return _ir.new_context()


def leaves(ctx, names, ttype=None):
    """Make the same input leaves on both sides; ``ttype`` is an (old, new) pair of types."""
    old_t, new_t = ttype or (OLD_F64, F64)
    olds = [old.Leaf("vocab_read", "vec_in", n, old_t) for n in names]
    news = [_ir.leaf(ctx, "vocab_read", "vec_in", n, new_t) for n in names]
    return olds, news


def contributions(pairs):
    """Return a contribution list as plain data: the child and its adjoint expression."""
    return [(new_shape(child), new_shape(adjoint)) for child, adjoint in pairs]


def old_contributions(pairs):
    """Return an old contribution list as the same plain data."""
    return [(old_shape(child), old_shape(adjoint)) for child, adjoint in pairs]


def test_a_unary_kind_has_the_old_reverse_and_forward_results(ctx):
    """``sin``: the pair ``(x, g * cos x)`` and the tangent ``t * cos x``."""
    (ox, og, ot), (nx, ng, nt) = leaves(ctx, "x g t".split())[0], leaves(ctx, "x g t".split())[1]
    on, nn = old.make("sin", [ox]), _ir.make("sin", [nx])
    orule, nrule = old_core.rule_for("sin", 0), rules.rule_for("sin", 0)
    assert contributions(nrule.vjp(nn, ng)) == old_contributions(orule.vjp(on, og))
    assert new_shape(nrule.jvp(nn, lambda c: nt)) == old_shape(orule.jvp(on, lambda c: ot))
    assert nrule.mode == orule.mode == "rule"


@pytest.mark.parametrize("kind", ["mul", "div"])
def test_a_binary_kind_has_the_old_results_with_one_tangent_missing(ctx, kind):
    """A binary: the reverse pairs and the tangent when only the first operand has one."""
    olds, news = leaves(ctx, "a b g ta".split())
    (oa, ob, og, ota), (na, nb, ng, nta) = olds, news
    on, nn = old.make(kind, [oa, ob]), _ir.make(kind, [na, nb])
    orule, nrule = old_core.rule_for(kind, 0), rules.rule_for(kind, 0)
    assert contributions(nrule.vjp(nn, ng)) == old_contributions(orule.vjp(on, og))
    got = nrule.jvp(nn, lambda c: nta if c == na else None)
    assert new_shape(got) == old_shape(orule.jvp(on, lambda c: ota if c is oa else None))
    assert nrule.jvp(nn, lambda c: None) is None and orule.jvp(on, lambda c: None) is None


def test_a_core_kind_has_the_old_results(ctx):
    """``dot`` reverse and ``norm`` forward: programs the library runs."""
    (ov, og, ot), (nv, ng, nt) = (
        leaves(ctx, ["v"], vec(3))[0] + leaves(ctx, ["g"])[0] + leaves(ctx, ["t"], vec(3))[0],
        leaves(ctx, ["v"], vec(3))[1] + leaves(ctx, ["g"])[1] + leaves(ctx, ["t"], vec(3))[1],
    )
    od, nd = old.make("dot", [ov, ov]), _ir.make("dot", [nv, nv])
    assert contributions(rules.rule_for("dot", 0).vjp(nd, ng)) == old_contributions(
        old_core.rule_for("dot", 0).vjp(od, og)
    )
    on, nn = old.make("norm", [ov]), _ir.make("norm", [nv])
    got = rules.rule_for("norm", 0).jvp(nn, lambda c: nt)
    assert new_shape(got) == old_shape(old_core.rule_for("norm", 0).jvp(on, lambda c: ot))


def test_a_linalg_kind_has_the_old_results(ctx):
    """``lu_det`` over an ``lu`` factor: reverse pairs and forward tangent."""
    mat = (old_types.TensorType((3, 3), "f64"), ((3, 3), "f64", None))
    (om, og, otm), (nm, ng, ntm) = (
        leaves(ctx, ["m"], mat)[0] + leaves(ctx, ["g"])[0] + leaves(ctx, ["tm"], mat)[0],
        leaves(ctx, ["m"], mat)[1] + leaves(ctx, ["g"])[1] + leaves(ctx, ["tm"], mat)[1],
    )
    olu, nlu = old.make("lu", [om]), _ir.make("lu", [nm])
    od, nd = old.make("lu_det", [olu]), _ir.make("lu_det", [nlu])
    assert contributions(rules.rule_for("lu_det", 0).vjp(nd, ng)) == old_contributions(
        old_core.rule_for("lu_det", 0).vjp(od, og)
    )
    got = rules.rule_for("lu_det", 0).jvp(nd, lambda c: ntm if c == nlu else None)
    assert new_shape(got) == old_shape(
        old_core.rule_for("lu_det", 0).jvp(od, lambda c: otm if c is olu else None)
    )


def test_a_ray_kind_has_the_old_results(ctx):
    """``ray_sphere``: the reverse pairs from the program of the library."""
    v3, f = vec(3), (OLD_F64, F64)
    parts = [("o", v3), ("d", v3), ("c", v3), ("r", f), ("tm", f)]
    olds = [leaves(ctx, [n], t)[0][0] for n, t in parts]
    news = [leaves(ctx, [n], t)[1][0] for n, t in parts]
    og, ng = leaves(ctx, ["g"], vec(4))[0][0], leaves(ctx, ["g"], vec(4))[1][0]
    on, nn = old.make("ray_sphere", olds), _ir.make("ray_sphere", news)
    assert contributions(rules.rule_for("ray_sphere", 0).vjp(nn, ng)) == old_contributions(
        old_core.rule_for("ray_sphere", 0).vjp(on, og)
    )


def test_the_sample_sum_reduction_has_the_old_results_and_a_max_is_refused_alike(ctx):
    """A sum over samples answers like the old rule; a max is refused with the same text."""
    olds, news = leaves(ctx, "x g t".split())
    (ox, og, ot), (nx, ng, nt) = olds, news
    ored = old_nodes.ReduceSamples("sum", ox)
    nred = ctx.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [nx])
    orule, nrule = old_core.rule_for(old_nodes.REDUCE_KIND, 0), rules.rule_for(_ir.REDUCE_KIND, 0)
    assert contributions(nrule.vjp(nred, ng)) == old_contributions(orule.vjp(ored, og))
    assert new_shape(nrule.jvp(nred, lambda c: nt)) == old_shape(orule.jvp(ored, lambda c: ot))
    assert nrule.jvp(nred, lambda c: None) is None
    omax = old_nodes.ReduceSamples("max", ox)
    nmax = ctx.new("ReduceSamples", "(('op', 'max'), ('value', #0))", [nx])
    with pytest.raises(HawkError) as old_refusal:
        orule.vjp(omax, og)
    with pytest.raises(HawkError) as new_refusal:
        nrule.vjp(nmax, ng)
    assert str(new_refusal.value) == str(old_refusal.value)


def test_zero_and_walker_kinds_and_an_unknown_kind_answer_as_the_old_table_does():
    """A comparison is a zero rule, ``at`` a walker, and a kind nobody knows is refused by name."""
    for kind, mode in (("lt", "zero"), ("floor", "zero"), ("at", "walker"), ("dispatch", "walker")):
        assert rules.rule_for(kind, 0).mode == old_core.rule_for(kind, 0).mode == mode
    assert rules.rule_for("lt", 0).vjp(None, None) == () and rules.rule_for("at", 0).vjp is None
    with pytest.raises(HawkError) as old_refusal:
        old_core.rule_for("no_such_kind", 0)
    with pytest.raises(HawkError) as new_refusal:
        rules.rule_for("no_such_kind", 0)
    assert str(new_refusal.value).split(":")[0] == str(old_refusal.value).split(":")[0]
    assert set(rules.ZERO_KINDS) >= {"lt", "floor"} and rules.WALKER_KINDS == ("at", "dispatch")


@pytest.mark.parametrize("dtype", ["f64", "f32", "i32"])
@pytest.mark.parametrize("shape", [(), (3,), (2, 2)])
def test_zero_like_equals_the_old_zero_for_each_dtype_and_shape(ctx, dtype, shape):
    """A rank-0 constant, else a splat of it, in the dtype of the type."""
    got = _ir.zero_like(ctx, (shape, dtype, None))
    assert new_shape(got) == old_shape(old_core.zero_like(old_types.TensorType(shape, dtype)))


def test_entrywise_widens_over_a_vector_and_is_the_op_over_scalars(ctx):
    """Check the documented behaviour: the old package has no public ``entrywise``."""
    nv, ns = leaves(ctx, ["v"], vec(3))[1][0], leaves(ctx, ["s"])[1][0]
    assert _ir.entrywise("add", [nv, ns]).field_text("kind") == "vec"
    assert _ir.entrywise("add", [ns, ns]).field_text("kind") == "add"
    assert _ir.entrywise("add", [nv, ns]).ttype == ((3,), "f64", None)


# ---- primitives ----


def primitive_pair(ctx):
    """Build a one-input primitive ``p(x) = x * x`` with a registered rule on both sides."""
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    oprim = old.Primitive("sq", old.make("mul", [ox, ox]), (ox,))
    nprim = ctx.new(
        "Primitive", "(('name', 'sq'), ('value', #0), ('inputs', (#1, )))",
        [_ir.make("mul", [nx, nx]), nx],
    )  # fmt: skip
    return (ox, oprim), (nx, nprim)


def test_a_primitive_with_a_registered_rule_instantiates_as_the_old_one_does(ctx):
    """Register a reverse and a forward rule, instantiate both, and compare the results."""
    (ox, oprim), (nx, nprim) = primitive_pair(ctx)
    oreg, nreg = old_kinds.Registry(), prim.PrimitiveRegistry()
    old_prim.declare_primitive(oreg, "sq", ext="user", directions=("vjp", "jvp"))
    prim.declare_primitive(nreg, "sq", ext="user", directions=("vjp", "jvp"))
    oin, ocot = old_prim.placeholder("in0", OLD_F64), old_prim.placeholder("cot", OLD_F64)
    nin, ncot = prim.placeholder(ctx, "in0", F64), prim.placeholder(ctx, "cot", F64)
    odot, ndot = old_prim.placeholder("dot0", OLD_F64), prim.placeholder(ctx, "dot0", F64)
    two = old.Const(2.0, OLD_F64), _ir.const(ctx, 2.0, F64)
    ovjp = (old.make("mul", [ocot, old.make("mul", [two[0], oin])]),)
    nvjp = (_ir.make("mul", [ncot, _ir.make("mul", [two[1], nin])]),)
    ojvp, njvp = old.make("mul", [odot, oin]), _ir.make("mul", [ndot, nin])
    old_prim.register_rule(oreg, "sq", old_prim.signature_of(oprim), "vjp", ovjp)
    old_prim.register_rule(oreg, "sq", old_prim.signature_of(oprim), "jvp", ojvp)
    prim.register_rule(nreg, "sq", prim.signature_of(nprim), "vjp", nvjp)
    prim.register_rule(nreg, "sq", prim.signature_of(nprim), "jvp", njvp)
    assert prim.has_rule(nreg, nprim, "vjp") == old_prim.has_rule(oreg, oprim, "vjp") is True
    og, ng = leaves(ctx, ["g"])[0][0], leaves(ctx, ["g"])[1][0]
    assert contributions(prim.vjp_contributions(nreg, nprim, ng)) == old_contributions(
        old_prim.vjp_contributions(oreg, oprim, og)
    )
    got = prim.jvp_tangent(nreg, nprim, [None])
    assert new_shape(got) == old_shape(old_prim.jvp_tangent(oreg, oprim, [None]))
    # a second registration of the same rule is a no-op; a different one is refused alike
    prim.register_rule(nreg, "sq", prim.signature_of(nprim), "jvp", njvp)
    with pytest.raises(HawkError) as old_refusal:
        old_prim.register_rule(oreg, "sq", old_prim.signature_of(oprim), "jvp", ovjp[0])
    with pytest.raises(HawkError) as new_refusal:
        prim.register_rule(nreg, "sq", prim.signature_of(nprim), "jvp", nvjp[0])
    assert str(new_refusal.value) == str(old_refusal.value)


def test_a_primitive_without_the_asked_rule_or_with_a_free_leaf_is_refused_alike(ctx):
    """Refuse a direction never supplied, a wrong arity and a free leaf with the old words."""
    (ox, oprim), (nx, nprim) = primitive_pair(ctx)
    oreg, nreg = old_kinds.Registry(), prim.PrimitiveRegistry()
    old_prim.declare_primitive(oreg, "sq", ext="user", directions=("jvp",))
    prim.declare_primitive(nreg, "sq", ext="user", directions=("jvp",))
    og, ng = leaves(ctx, ["g"])[0][0], leaves(ctx, ["g"])[1][0]
    with pytest.raises(HawkError) as old_refusal:
        old_prim.vjp_contributions(oreg, oprim, og)
    with pytest.raises(HawkError) as new_refusal:
        prim.vjp_contributions(nreg, nprim, ng)
    assert str(new_refusal.value) == str(old_refusal.value)
    old_sig, new_sig = old_prim.signature_of(oprim), prim.signature_of(nprim)
    for label, old_call, new_call in (
        ("arity", lambda: old_prim.register_rule(oreg, "sq", old_sig, "vjp", ()),
         lambda: prim.register_rule(nreg, "sq", new_sig, "vjp", ())),
        ("free leaf", lambda: old_prim.register_rule(oreg, "sq", old_sig, "jvp", ox),
         lambda: prim.register_rule(nreg, "sq", new_sig, "jvp", nx)),
    ):  # fmt: skip
        with pytest.raises(HawkError) as old_refusal:
            old_call()
        with pytest.raises(HawkError) as new_refusal:
            new_call()
        assert str(new_refusal.value) == str(old_refusal.value), label


def test_substitute_replaces_placeholders_by_name_and_leaves_the_rest(ctx):
    """``substitute`` replaces the named placeholders and keeps a subgraph that reads none."""
    oin, nin = old_prim.placeholder("in0", OLD_F64), prim.placeholder(ctx, "in0", F64)
    (ox,), (nx,) = leaves(ctx, ["x"])[0], leaves(ctx, ["x"])[1]
    otemplate = old.make("add", [oin, old.make("mul", [oin, old.Const(2.0, OLD_F64)])])
    ntemplate = _ir.make("add", [nin, _ir.make("mul", [nin, _ir.const(ctx, 2.0, F64)])])
    got = prim.substitute(ntemplate, {"$in0": nx, "$unused": nx})
    assert new_shape(got) == old_shape(old_prim.substitute(otemplate, {"$in0": ox}))
    assert prim.substitute(nx, {"$in0": nx}) == nx
