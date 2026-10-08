# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An extension seam: a custom primitive with a SUPPLIED rule.

WHAT THIS ROW HAS TO PROVE, and why each half is here.

A primitive is two claims at once, and they pull in opposite directions. The
FORWARD claim is that nothing changed: the decorated function is inlined, so the
kernel that calls it computes and emits exactly what the same arithmetic written
longhand computes and emits — no call, no second entry point, no extra
translation unit. The DERIVATIVE claim is that everything changed: the 
table's rule for that subgraph is not used, the author's is. A test that only
checked the forward would pass with the rule ignored; a test that only checked
the derivative against finite differences would pass with a CORRECT supplied
rule that was never consulted, because the table's rule is correct too.

So the load-bearing row is neither: it is
:func:`test_a_wrong_supplied_vjp_is_the_one_that_is_used`, which registers a
primitive whose declared reverse rule is deliberately WRONG by a factor of three
and requires the derived gradient to be wrong by exactly three. That is the
positive control — an in-suite one, not a plant — and it is the only shape of
evidence that separates "the supplied rule is used" from "some correct rule is
used". The finite-difference rows beside it then say the machinery around it is
sound, and the emitted-body row says the forward paid nothing for it.

"""

from __future__ import annotations

import _oracle as O
import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as hm
from hawk.artifact import build
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ext import primitive, primitives
from hawk.ir import HawkError, Primitive
from hawk.math import dot, exp, log, vec, vsum

RNG = np.random.default_rng(20260905)
N = 7
H = 1e-5


# --------------------------------------------------------------------------- #
# The subject: softplus, whose derivative HAS a closed form the table would
# otherwise reach through log/exp/div — the shape a real primitive is declared
# for (a downstream KAN B-spline basis is the same shape at a larger size).
# --------------------------------------------------------------------------- #
@primitive("hw7a_softplus",
           vjp=lambda x, bar: bar / (1.0 + exp(-x)),
           jvp=lambda x, dx: dx / (1.0 + exp(-x)))
def softplus(x):
    """``log(1 + exp(x))`` — traced INLINE at every call site."""
    return log(1.0 + exp(x))


def plain_softplus(x):
    """The same arithmetic, undecorated: the forward row's reference."""
    return log(1.0 + exp(x))


@primitive("hw7a_triple_wrong", vjp=lambda x, bar: 3.0 * bar)
def wrongly_declared(x):
    """``x`` — an identity whose declared reverse rule is WRONG by 3x.

    Not a mistake: it is the positive control. The identity's true gradient is
    the incoming adjoint, the table would compute exactly that, and the declared
    rule computes three times it — so a derived gradient of ``3 * bar`` is proof
    the SUPPLIED rule ran and a gradient of ``bar`` is proof it did not."""
    return x + 0.0


@primitive("hw7a_reverse_only", vjp=lambda x, bar: bar)
def reverse_only(x):
    """A primitive with no forward-mode rule — the refusal row's subject."""
    return x + 0.0


@hawk.kernel
def primitive_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = softplus(x) * 2.0


@hawk.kernel
def longhand_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = plain_softplus(x) * 2.0


@hawk.kernel
def wrong_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = wrongly_declared(x)


# --------------------------------------------------------------------------- #
# The forward is INLINED: same text, same numbers.
# --------------------------------------------------------------------------- #
def test_the_boundary_node_is_recorded_in_the_dag():
    """The trace carries a Primitive node whose operands are the forward and the
    declared inputs — without it there is nothing for a rule to attach to."""
    found = [n for n in primitive_kernel.walk.order if isinstance(n, Primitive)]
    assert len(found) == 1, f"expected ONE primitive boundary, got {found}"
    assert found[0].name == "hw7a_softplus"
    assert found[0].input_count == 1
    assert not any(isinstance(n, Primitive) for n in longhand_kernel.walk.order)


def test_the_emitted_body_is_byte_identical_to_the_longhand_body():
    """Needs the ``Primitive`` branch in ``hawk/emit/aether._Renderer._expr``,
    or the boundary falls through to the op dispatch::

        hawk.ir.nodes.HawkError: op 'primitive' has no aether spelling at rank 0
    """
    mine = render_body(primitive_kernel.sinks, primitive_kernel.walk).text
    theirs = render_body(longhand_kernel.sinks, longhand_kernel.walk).text
    assert mine == theirs, (
        "a primitive is INLINED: the emitted body must be the body "
        f"the same arithmetic written longhand emits.\n--- primitive ---\n{mine}\n"
        f"--- longhand ---\n{theirs}")


def test_the_forward_runs_bit_identically_on_the_host_path(tmp_path, cache_dir):
    """Not `allclose`: the two bodies are the same text, so the two artifacts
    must give the same bits, and a tolerance here would hide a re-association."""
    x = RNG.uniform(-2.0, 2.0, size=N)
    mine = _run(primitive_kernel, tmp_path / "prim", cache_dir, x=x)
    theirs = _run(longhand_kernel, tmp_path / "long", cache_dir, x=x)
    np.testing.assert_array_equal(mine, theirs)


# --------------------------------------------------------------------------- #
# The SUPPLIED rule is the one that runs — the positive control first.
# --------------------------------------------------------------------------- #
def test_a_wrong_supplied_vjp_is_the_one_that_is_used():
    """The control that makes every other derivative row mean something.

    ``wrongly_declared`` is the identity; the table's rule for that DAG
    gives ``bar`` and the declared rule gives ``3 * bar``. If the transform ever
    differentiated THROUGH the primitive, this row would report ``bar`` and say
    so — which is exactly the failure a correct-but-ignored rule hides."""
    env = {"x": 1.7, "bar_y": 0.9}
    got = evaluate(vjp(wrong_kernel), env)["bar_x"]
    assert float(got) == pytest.approx(3.0 * env["bar_y"], rel=1e-12), (
        "the SUPPLIED vjp must be the one applied: expected the "
        f"declared 3x rule ({3.0 * env['bar_y']}), got {got} — which is the "
        "table's own derivative of the inlined forward")


@pytest.mark.parametrize("x0", [-1.3, -0.2, 0.4, 2.6])
def test_the_supplied_vjp_matches_finite_differences(x0):
    """Needs the ``Primitive`` branch in ``hawk/diff/transform``'s reverse
    loop, or the boundary reaches the closed-world table::

        hawk.ir.nodes.HawkError: no derivative rule for node kind 'primitive'
        (rank 0): HAWK's rule table is closed-world ...
    """
    sinks = primitive_kernel.sinks
    env = {"x": x0, "bar_y": 0.7}
    got = float(evaluate(vjp(sinks), env)["bar_x"])
    h = H * max(1.0, abs(x0))
    plus = float(evaluate(sinks, {**env, "x": x0 + h})["y"])
    minus = float(evaluate(sinks, {**env, "x": x0 - h})["y"])
    assert got == pytest.approx(env["bar_y"] * (plus - minus) / (2 * h),
                                rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("x0", [-1.3, -0.2, 0.4, 2.6])
def test_the_supplied_jvp_matches_finite_differences(x0):
    sinks = primitive_kernel.sinks
    got = float(evaluate(jvp(sinks), {"x": x0, "dot_x": 1.0})["dot_y"])
    h = H * max(1.0, abs(x0))
    plus = float(evaluate(sinks, {"x": x0 + h})["y"])
    minus = float(evaluate(sinks, {"x": x0 - h})["y"])
    assert got == pytest.approx((plus - minus) / (2 * h), rel=1e-6, abs=1e-9)


def test_the_derived_reverse_ir_deploys_and_runs(tmp_path, cache_dir):
    """The supplied rule builds ORDINARY HAWK IR, so it must survive the whole
    toolchain — walk, emitter, compiler — not merely the interpreter."""
    from types import SimpleNamespace

    from hawk.ir import canonical

    sinks = vjp(primitive_kernel)
    derived = SimpleNamespace(name="primitive_kernel_vjp", sinks=sinks,
                              walk=canonical(sinks))
    x = RNG.uniform(-2.0, 2.0, size=N)
    bar = RNG.uniform(0.2, 1.4, size=N)
    got = _run(derived, tmp_path / "vjp", cache_dir, x=x, bar_y=bar)
    np.testing.assert_allclose(got, 2.0 * bar / (1.0 + np.exp(-x)),
                               rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# The two refusals.
# --------------------------------------------------------------------------- #
def test_a_missing_direction_is_refused_naming_the_primitive_and_the_direction():
    """No fallback: differentiating through the forward would produce the table's
    number for a kernel whose author declared the table's number was not wanted."""
    @hawk.kernel
    def uses_reverse_only(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = reverse_only(x)

    assert vjp(uses_reverse_only), "the direction that IS supplied must still work"
    with pytest.raises(HawkError) as excinfo:
        jvp(uses_reverse_only)
    message = str(excinfo.value)
    assert "hw7a_reverse_only" in message and "forward (jvp)" in message, message


def test_a_registry_name_collision_is_refused():
    """The name is the boundary node's dedup identity and therefore part of
    ``Walk.digest``; two primitives sharing it would give two different kernels
    one content hash, which is the failure a content-addressed cache cannot see."""
    name = "hw7a_collision_subject"

    @primitive(name)
    def first(x):
        return x + 1.0

    assert primitives()[name] is first
    with pytest.raises(HawkError, match="already registered"):
        @primitive(name)
        def second(x):
            return x + 2.0


def test_a_rule_of_the_wrong_arity_or_type_is_refused():
    """A supplied rule is checked against what it stands in for: a short tuple
    would silently leave an input's gradient at zero, and a rank mismatch would
    reach the emitter far from the declaration that caused it."""
    @primitive("hw7a_bad_arity", vjp=lambda a, b, bar: 1.0 * bar)
    def two_in(a, b):
        return a * b

    @hawk.kernel
    def uses_two(a: hawk.Scalar, b: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = two_in(a, b)

    with pytest.raises(HawkError, match="ONE adjoint per"):
        vjp(uses_two)

    @primitive("hw7a_bad_type", vjp=lambda v, bar: hm.norm(v) * hm.vsum(bar))
    def wrong_rank(v):
        return v * 2.0

    @hawk.kernel
    def uses_rank(v: hawk.Vector[3], y: hawk.Mutable[hawk.Vector[3]]):
        y = wrong_rank(v)

    with pytest.raises(HawkError, match="was owed"):
        vjp(uses_rank)


# --------------------------------------------------------------------------- #
def _run(kernel, directory, cache, **planes):
    """Emit, compile and RUN one kernel on the host; its single output plane."""
    build(kernel, directory, targets=("host",), cache_dir=cache)
    loaded = O.load(directory, kernel.name)
    wanted = {name: planes[name] for role, name in loaded.arg_spec
              if role in O.INPUT_ROLES or role == "uniform"}
    return O.run_kernel(loaded, N, **wanted)


# --------------------------------------------------------------------------- #
# A REUSED rank-1 primitive result: the boundary is transparent to the
# emitter's NAMING, not merely to its dispatch.
# --------------------------------------------------------------------------- #
@primitive("hw7a_pair")
def pair(x):
    """A rank-1 forward: an aether LEAF value the emitter names unconditionally."""
    return vec(exp(x), log(1.0 + exp(x)))


def plain_pair(x):
    return vec(exp(x), log(1.0 + exp(x)))


@hawk.kernel
def reused_primitive_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    p = pair(x)
    y = dot(p, p) + vsum(p)


@hawk.kernel
def reused_longhand_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    p = plain_pair(x)
    y = dot(p, p) + vsum(p)


def test_a_reused_primitive_result_emits_no_alias_of_its_forward():
    """A reused primitive's boundary node must not be named like any node
    with fan-out >= 2: since its rank-1 forward was
    already named as a leaf value the artifact carried ``const auto t60 = t59;``
    -- an alias the longhand body does not have, so the two bodies were not
    byte-identical. A primitive is not a node the emitter renders: every
    reference to it is a reference to its forward, and its fan-out is the
    forward's."""
    mine = render_body(reused_primitive_kernel.sinks, reused_primitive_kernel.walk).text
    theirs = render_body(reused_longhand_kernel.sinks, reused_longhand_kernel.walk).text
    assert mine == theirs, (
        "a reused primitive result must not alias its forward.\n"
        f"--- primitive ---\n{mine}\n--- longhand ---\n{theirs}")
