"""A user primitive whose derivative rules contain a lowered loop (fixed-count accumulation): the rule is a template with a loop
in it, registered as data, instantiated by substitution through the loop body, checked against finite differences on the host."""
import pytest
from _eval import evaluate

import hawk
from hawk.diff import jvp, vjp
from hawk.dialects import KINDS
from hawk.ext import primitive
from hawk.ir import Loop
from hawk import HawkError
from hawk.ir.nodes import Leaf
from hawk.types import TensorType
from raptor.rules import primitive as rules
from raptor.rules.primitive import placeholder, substitute

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
    (inst,) = KINDS.primitives[name][1].values()
    return inst[direction][1]


def _has_loop(node):
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Loop):
            return True
        stack.extend(n.operands)
    return False


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
    t = TensorType((), "f64")
    vjp(uses_looped)
    (inst,) = KINDS.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    stray = Leaf("uniform", "uniform", "stray", t)
    foreign = substitute(tpl, {"$in0": stray})              # the substitution reaches the loop body: it now reads `stray`
    rules.declare_primitive(KINDS, "w1_loop_bad", ext="user", directions=("vjp",))
    with pytest.raises(HawkError, match="reads uniform 'stray' inside a loop body"):
        rules.register_rule(KINDS, "w1_loop_bad", ((t,), t), "vjp", (foreign,))


def test_a_loop_that_writes_a_plane_or_reads_a_foreign_loop_local_is_refused_by_name():
    t = TensorType((), "f64")
    vjp(uses_looped)
    (inst,) = KINDS.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    loop = next(n for n in _walk(tpl) if isinstance(n, Loop))
    rules.declare_primitive(KINDS, "w1_loop_bad2", ext="user", directions=("vjp",))
    with pytest.raises(HawkError, match="loop outside that loop"):
        rules.register_rule(KINDS, "w1_loop_bad2", ((t,), t), "vjp", (loop.index if loop.index.ttype == t else loop.carries[0],))


def test_substitution_gives_each_instantiation_its_own_loop_locals():
    vjp(uses_looped)
    (inst,) = KINDS.primitives["w1_loop_rule"][1].values()
    (tpl,) = inst["vjp"][1]
    t = TensorType((), "f64")
    one, two = (substitute(tpl, {"$in0": placeholder("a", t), "$cot": placeholder("b", t)}) for _ in range(2))
    la, lb = (next(n for n in _walk(r) if isinstance(n, Loop)) for r in (one, two))
    assert la is not lb and la.index is not lb.index and la.carries[0] is not lb.carries[0]


def _walk(root):
    stack, seen, out = [root], set(), []
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        stack.extend(n.operands)
    return out


def test_the_digest_of_a_kernel_using_the_primitive_is_stable_across_two_registrations():
    first = uses_looped.walk.digest
    again = _make("w1_loop_rule")                                        # an equal re-registration (a notebook cell run twice)
    assert again is looped

    @hawk.kernel
    def uses_looped2(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = looped(x)
    vjp(uses_looped), vjp(uses_looped2)
    assert uses_looped2.walk.digest == first == uses_looped.walk.digest
