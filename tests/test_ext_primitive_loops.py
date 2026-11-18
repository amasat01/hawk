"""A user primitive whose derivative rules contain a lowered loop (fixed-count accumulation): the rule is a template with a loop
in it, registered as data, instantiated by substitution through the loop body, checked against finite differences on the host."""
import importlib

import pytest
from _eval import evaluate

import hawk
from hawk.diff import jvp, vjp
from hawk import HawkError, _ir
from hawk.ext import _primitive_rules as rules
from hawk.ext import primitive

# the seam's module, which the name `primitive` of the package hides
RULES = importlib.import_module("hawk.ext.primitive")._RULES

H = 1e-5


def _vjp_rule(x, bar):
    acc = 0.0 * bar
    for k in range(3):
        acc = acc + bar * x
    return acc


def _jvp_rule(x, dx):
    acc = 0.0 * dx
    for k in range(3):
        acc = acc + x * dx
    return acc


def _make(name):
    @primitive(name, vjp=_vjp_rule, jvp=_jvp_rule)
    def cube_half(x):                # 1.5 x^2: derivative 3 x, which both rules sum as three terms of x
        return 1.5 * x * x
    return cube_half


looped = _make("w1_loop_rule")


@hawk.kernel
def uses_looped(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = looped(x)


def _templates(name, direction):
    (inst,) = RULES.primitives[name][1].values()
    return inst[direction][1]


def _has_loop(node):
    return any(_ir.is_a(n, "loop") for n in _ir.post_order([node]))


def test_both_rules_register_a_template_with_a_loop_in_it():
    vjp(uses_looped), jvp(uses_looped)
    assert all(_has_loop(t) for t in _templates("w1_loop_rule", "vjp") + _templates("w1_loop_rule", "jvp"))


@pytest.mark.parametrize("x0", [-1.3, 0.4, 2.6])
def test_the_loop_rules_match_finite_differences_on_the_host(x0):
    sinks = uses_looped.sinks
    h = H * max(1.0, abs(x0))
    ref = (float(evaluate(sinks, {"x": x0 + h})["y"]) - float(evaluate(sinks, {"x": x0 - h})["y"])) / (2 * h)
    assert ref == pytest.approx(3.0 * x0, rel=1e-6)
    assert float(evaluate(vjp(sinks), {"x": x0, "bar_y": 1.0})["bar_x"]) == pytest.approx(ref, rel=1e-6, abs=1e-9)
    assert float(evaluate(jvp(sinks), {"x": x0, "dot_x": 1.0})["dot_y"]) == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_a_free_leaf_inside_a_loop_body_is_refused_by_name():
    t = _ir.TensorType((), "f64")
    vjp(uses_looped)
    (inst,) = RULES.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    stray = _ir.leaf(tpl.ctx, "uniform", "uniform", "stray", t)
    foreign = rules.substitute(tpl, {"$in0": stray})        # the substitution reaches the loop body: it now reads `stray`
    rules.declare_primitive(RULES, tpl.ctx, "w1_loop_bad", ext="user", directions=("vjp",))
    with pytest.raises(HawkError, match="reads uniform 'stray' inside a loop body"):
        rules.register_rule(RULES, "w1_loop_bad", ((t,), t), "vjp", (foreign,))


def test_a_loop_that_writes_a_plane_or_reads_a_foreign_loop_local_is_refused_by_name():
    t = _ir.TensorType((), "f64")
    vjp(uses_looped)
    (inst,) = RULES.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    loop = next(n for n in _ir.post_order([tpl]) if _ir.is_a(n, "loop"))
    (index,), (carry, *_) = loop.field_nodes("index"), loop.field_nodes("carries")
    rules.declare_primitive(RULES, tpl.ctx, "w1_loop_bad2", ext="user", directions=("vjp",))
    with pytest.raises(HawkError, match="loop outside that loop"):
        rules.register_rule(
            RULES, "w1_loop_bad2", ((t,), t), "vjp", (index if _ir.type_of(index) == t else carry,)
        )


def test_substitution_gives_each_instantiation_its_own_loop_locals():
    vjp(uses_looped)
    (inst,) = RULES.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    t = _ir.TensorType((), "f64")
    one, two = (
        rules.substitute(
            tpl, {"$in0": rules.placeholder(tpl.ctx, "a", t), "$cot": rules.placeholder(tpl.ctx, "b", t)}
        )
        for _ in range(2)
    )
    la, lb = (next(n for n in _ir.post_order([r]) if _ir.is_a(n, "loop")) for r in (one, two))
    assert la != lb
    assert la.field_nodes("index") != lb.field_nodes("index")
    assert la.field_nodes("carries") != lb.field_nodes("carries")


def test_the_digest_of_a_kernel_using_the_primitive_is_stable_across_two_registrations():
    first = _ir.digest(uses_looped.walk)
    again = _make("w1_loop_rule")                                        # an equal re-registration (a notebook cell run twice)
    assert again is looped

    @hawk.kernel
    def uses_looped2(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = looped(x)
    vjp(uses_looped), vjp(uses_looped2)
    assert _ir.digest(uses_looped2.walk) == first == _ir.digest(uses_looped.walk)
