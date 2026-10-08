# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``v[k]`` and ``v[k] = …`` at a lowered loop's own index.

Lowering a bounded ``for`` to a real loop makes the author's ``k`` a
symbolic :class:`~hawk.ir.loop_nodes.LoopIndex` instead of a Python integer, so
every way of touching a rank-1 value one component at a time needs its own
spelling: ``v[k]`` (a static-integer-component read only), ``v[k] = …`` (not
a form on its own), and a comprehension over a symbolic index (banned
outright, since the body runs once). This covers the remaining shape: a
state vector advanced component by component inside a loop.

THE TWO OPS, AND WHY THEY ARE OPS AND NOT STATEMENTS. ``component_at(v, k)``
reads and ``set_component_at(v, k, e)`` returns ``v`` with one component
replaced. Both are VALUES, because HAWK's IR has no mutable object: a body's
``v[k] = e`` is rewritten by the ``ast`` pass into ``v = set_component(v, k,
e)``, which makes ``v`` an ordinary assigned name — and therefore an ordinary
loop-CARRIED value, with the loop's existing rules and no new statement form
anywhere below.

THE AETHER FACT THIS TURNS ON. ``aether/view/Item.h``
declares BOTH a compile-time ``get<Is...>`` and a RUNTIME
``operator()(Idxs... idxs)`` (:227-239) in const and non-const overloads, the
non-const returning ``T&``; it casts each index to ``std::size_t`` itself and
folds row-major over the static extents. So the answer to "does a local
``Real[W]`` array copy have to stand in?" is NO: the read is ``item(k)`` and
the write is ``item(k) = e`` on a register-resident value. What the emitter
does have to do is MATERIALISE the operand into an ``Item`` first — a general
aether expression carries only the ``Expression`` CRTP's compile-time
``eval<Is...>(SampleIndex)``, and ``operator()`` is ``Item``'s own — which it
does through ``Item``'s converting constructor, once per value per scope.

Three mechanisms, each with its own refusal:

* the runtime read: ``HawkError: a traced value is indexed by a STATIC integer
  component, got <Value loop_index TensorType(shape=, dtype='i32', tag=None)>
  (a runtime index reads a Table with at())``;
* the component write: ``HawkError: …:125: this `for` assigns no name that is
  bound before it and commits no sink, so it computes nothing observable`` —
  ``nxt[c] = …`` was a SUBSCRIPT store, which is not a name assignment, so the
  loop rewrite saw a body that carried nothing at all;
* the rank-1 carry both of the above lead to: ``HawkError: Loop: carried slot 0
  ('nxt') is a rank-1 value typed TensorType(shape=(3,), dtype='f64',
  tag=None), and a loop-carried value must be rank-0`` — and the same refusal
  named ``'bar_v'`` for the VJP of the READ-only kernel, because a vector
  adjoint accumulated across iterations is exactly a rank-1 carry (see
  the loop-lowering primal rows' own row on that refusal and what
  replaced it).
"""

from __future__ import annotations

import pathlib

import _toolchain as TC
import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from hawk.emit import BACKENDS, CUDA, FLOAT64, render_body, render_source
from hawk.ir import HawkError, canonical

#: Classical RK4 (Kutta 1901), the stage and combine weights
#: a downstream rollout kernel carries as structural constants.
RK4_C = (0.0, 0.5, 0.5, 1.0)
RK4_B = (1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0)

W = 3
H = 1e-6


def _rk4_step(x, d, dt):
    """ONE RK4 step of ``dx/dt = -d tanh(x)`` for a single component.

    A plain-python trace-time helper — the four stages are compile-time
    constants, so they belong in a helper whose body the ``ast`` pass never
    sees. It is called from INSIDE a lowered loop's body with ``x``/``d`` read at that
    loop's symbolic index, which is the whole point of the file."""
    acc, stage = 0.0, x
    for c, b in zip(RK4_C, RK4_B):
        slope = -d * m.tanh(x + c * dt * stage)
        acc = acc + b * slope
        stage = slope
    return x + dt * acc


def _rk4_reference(x: float, d: float, dt: float) -> float:
    """The same step in numpy — the third arm, independent of both the IR
    interpreter and the compiled kernel."""
    acc, stage = 0.0, x
    for c, b in zip(RK4_C, RK4_B):
        slope = -d * np.tanh(x + c * dt * stage)
        acc = acc + b * slope
        stage = slope
    return x + dt * acc


# --------------------------------------------------------------------------- #
# The subjects: the rollout shape, once per spelling.
# --------------------------------------------------------------------------- #
@hawk.kernel
def rollout_read(v: hawk.Vector[W], k1: hawk.Vector[W], dt: hawk.Param,
                 y: hawk.Mutable[hawk.Scalar]):
    """The READ half: an RK4 step per component, summed over a lowered loop.

    ``v[c]`` and ``k1[c]`` are read at the loop's own symbolic index; the loop
    carries one rank-0 accumulator, so nothing here needs the write spelling."""
    total = 0.0
    for c in range(W):
        total = total + _rk4_step(v[c], k1[c], dt)
    y = total


@hawk.kernel
def rollout_write(v: hawk.Vector[W], k1: hawk.Vector[W], dt: hawk.Param,
                  out: hawk.Mutable[hawk.Vector[W]]):
    """The write half: the advanced state, built component by component and
    committed once.

    ``nxt`` is bound before the loop, so it is loop-CARRIED, and it is rank-1,
    so its emitted local is an ``aether::Item``. Each iteration replaces one
    component; the plane is committed whole after the loop, because a mutable
    plane is a per-sample row the body owns entirely."""
    nxt = v
    for c in range(W):
        nxt[c] = _rk4_step(v[c], k1[c], dt)
    out = nxt


ENV = {"v": np.array([0.3, -0.7, 1.1]), "k1": np.array([0.5, 0.25, -0.4]),
       "dt": 0.1}


# --------------------------------------------------------------------------- #
# The IR and the emission.
# --------------------------------------------------------------------------- #
def test_a_read_at_the_loop_index_lowers_to_aethers_runtime_item_access():
    """The aether fact, read off the generated body: ``Item::operator()`` at the
    loop's own index, on a value MATERIALISED into an ``Item`` first.

    Both halves are asserted because both are load-bearing. Without the
    materialisation the subscript would be taken on an aether EXPRESSION, which
    carries only the compile-time ``eval<Is...>`` and would not compile; and if
    the emitter had reached for a local ``Real[W]`` array copy instead — the
    other option named — the body would carry an array declaration and a
    copy loop that aether does not need."""
    text = render_body(rollout_read.sinks, rollout_read.walk).text
    assert "const aether::Item<Real, 3> hawk_it" in text, text
    index = [ln for ln in text.splitlines() if "for (Int hawk_k" in ln]
    assert len(index) == 1, text
    variable = index[0].split("Int ")[1].split(" ")[0]
    assert f"hawk_it0({variable})" in text, text
    assert "[3];" not in text, ("a Real[W] array copy was emitted; aether's Item "
                                f"carries a runtime operator()\n{text}")


def test_a_write_at_the_loop_index_is_a_carried_Item_the_body_reassigns():
    """The WRITE half's emission. The carried vector is an ``aether::Item``
    declared before the loop; each iteration takes a NON-const copy, writes one
    component through ``operator()``, and the copy becomes the next iteration's
    carried value. The copy is what keeps the IR's functional meaning — the node
    produces a new value and leaves its operand alone — and it is registers, not
    memory, because an ``Item`` has all-static extents."""
    text = render_body(rollout_write.sinks, rollout_write.walk).text
    assert "aether::Item<Real, 3> hawk_v" in text, text
    stores = [ln.strip() for ln in text.splitlines() if "hawk_sv" in ln]
    # the copy, the component write, and the carried local taking the result
    assert len(stores) == 3, text
    assert stores[0].startswith("aether::Item<Real, 3> hawk_sv"), stores
    assert not stores[0].startswith("const"), (
        "the copy a component is written into must be NON-const", stores)
    assert ") = static_cast<Real>(" in stores[1], stores
    assert "mut_out[i] = hawk_v" in text, text


def test_both_backends_wrap_the_same_body():
    """ONE renderer, and a runtime component access is not a host
    construct the device path re-derives."""
    for kernel in (rollout_read, rollout_write):
        host = render_source("k", kernel.sinks, kernel.walk, BACKENDS["host"])
        device = render_source("k", kernel.sinks, kernel.walk, CUDA)
        assert host.body == device.body
        assert "for (" in host.body


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("case", ["read", "write", "read_vjp", "write_vjp"],
                         ids=["read", "write", "read_vjp", "write_vjp"])
def test_the_emitted_translation_unit_compiles(backend, case, tmp_path):
    """The compiler is the judge of whether ``Item::operator()`` was spelled
    correctly, at the right constness, on a value of the right type — including
    in the DERIVED kernels, whose loops carry rank-1 ``Item`` locals. Never
    skipped: an unresolvable toolchain FAILS (``tests/_toolchain.py``)."""
    primal = rollout_read if case.startswith("read") else rollout_write
    if case.endswith("vjp"):
        sinks = vjp(primal, wrt=("v", "k1", "dt"))
        walk = canonical(sinks)
    else:
        sinks, walk = primal.sinks, primal.walk
    src = render_source("k", sinks, walk, BACKENDS[backend], mode=FLOAT64,
                        exports=exports(backend))
    path = pathlib.Path(tmp_path) / ("k.cpp" if backend == "host" else "k.cu")
    path.write_text(src.text)
    rc, err, argv = TC.compile_check(path, backend)
    assert rc == 0, f"$ {' '.join(argv)}\n{err[-4000:]}"


# --------------------------------------------------------------------------- #
# The numbers.
# --------------------------------------------------------------------------- #
def test_the_interpreter_reproduces_the_rk4_step_component_by_component():
    """Both spellings against numpy. The interpreter walks the lowered loop
    iteration by iteration, so it is the unrolled reading of the same body —
    which is what makes it the oracle for a lowering whose whole claim is that
    it changed WHERE the arithmetic is written and not what it is."""
    want = np.array([_rk4_reference(ENV["v"][i], ENV["k1"][i], ENV["dt"])
                     for i in range(W)])
    got_write = np.asarray(evaluate(rollout_write.sinks, ENV)["out"], dtype=float)
    np.testing.assert_allclose(got_write, want, rtol=0, atol=0)
    got_read = float(evaluate(rollout_read.sinks, ENV)["y"])
    assert got_read == pytest.approx(float(np.sum(want)), rel=1e-15)


@pytest.mark.parametrize("name", ["rollout_read", "rollout_write"])
def test_the_reverse_matches_a_central_difference(name):
    """The gate. A vector adjoint accumulated across a loop's iterations is a
    rank-1 carry, and a one-hot built at a runtime index is easy to get subtly
    wrong (an off-by-one in the comparison chain would put the gradient in the
    neighbouring component and still look plausible), so the rules are
    EVALUATED against the primal rather than inspected."""
    kernel = {"rollout_read": rollout_read, "rollout_write": rollout_write}[name]
    sink = kernel.sinks[0].name
    seed = (1.0 if sink == "y" else np.array([1.0, -0.5, 0.25]))
    derived = vjp(kernel, wrt=("v", "k1", "dt"))
    got = evaluate(derived, {**ENV, f"bar_{sink}": seed})

    def objective(env: dict) -> float:
        return float(np.sum(np.asarray(evaluate(kernel.sinks, env)[sink]) * seed))

    for plane in ("v", "k1"):
        for i in range(W):
            up, down = dict(ENV), dict(ENV)
            a, b = np.array(ENV[plane], float), np.array(ENV[plane], float)
            a[i] += H
            b[i] -= H
            up[plane], down[plane] = a, b
            want = (objective(up) - objective(down)) / (2 * H)
            actual = float(np.asarray(got[f"bar_{plane}"], dtype=float)[i])
            assert actual == pytest.approx(want, rel=1e-6, abs=1e-7), (
                f"d{sink}/d{plane}[{i}]: {actual} vs {want}")
    up, down = {**ENV, "dt": ENV["dt"] + H}, {**ENV, "dt": ENV["dt"] - H}
    want = (objective(up) - objective(down)) / (2 * H)
    assert float(got["bar_dt"]) == pytest.approx(want, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("name", ["rollout_read", "rollout_write"])
def test_the_forward_matches_a_central_difference(name):
    """Forward mode through the same two spellings: the tangent of a runtime
    read is the same component of the tangent, and the tangent of a replacement
    is the replacement of the tangents."""
    kernel = {"rollout_read": rollout_read, "rollout_write": rollout_write}[name]
    sink = kernel.sinks[0].name
    tangent = {"v": np.array([1.0, -0.5, 0.25]),
               "k1": np.array([0.2, 0.7, -1.0]), "dt": 0.3}
    derived = jvp(kernel, wrt=("v", "k1", "dt"))
    got = np.asarray(evaluate(derived, {**ENV,
                                        **{f"dot_{k}": v
                                           for k, v in tangent.items()}})
                     [f"dot_{sink}"], dtype=float)
    up = {k: np.asarray(ENV[k], float) + H * tangent[k] for k in ENV}
    down = {k: np.asarray(ENV[k], float) - H * tangent[k] for k in ENV}
    want = ((np.asarray(evaluate(kernel.sinks, up)[sink], dtype=float)
             - np.asarray(evaluate(kernel.sinks, down)[sink], dtype=float))
            / (2 * H))
    assert got == pytest.approx(want, rel=1e-6, abs=1e-7)


def test_the_accumulate_spelling_writes_the_same_component():
    """``v[k] += e`` is the same store over ``v[k] <op> e``.

    It needs its own rewrite rather than Python's: Python lowers an augmented
    subscript store into ``__getitem__`` then ``__setitem__``, and a traced
    value's ``__setitem__`` can only refuse (there is nothing to mutate). The
    ``ast`` pass therefore writes the read out itself and reuses the plain
    store, so an author who reaches for ``+=`` — the natural spelling for an
    accumulate — gets the arithmetic rather than a message about immutability."""
    @hawk.kernel
    def accumulate(v: hawk.Vector[W], out: hawk.Mutable[hawk.Vector[W]]):
        held = v
        for c in range(W):
            held[c] += v[c] * 2.0
        out = held

    value = np.array([1.0, 2.0, 4.0])
    got = np.asarray(evaluate(accumulate.sinks, {"v": value})["out"], dtype=float)
    np.testing.assert_allclose(got, value * 3.0, rtol=0, atol=0)


def _mutate_in_a_helper(v):
    """A plain-python helper that tries to write a component. Module level, so
    it is a genuine helper and not a closure the ``ast`` pass could reach."""
    v[0] = 1.0
    return v


def test_a_component_store_in_a_plain_helper_names_the_kernel_body():
    """The ``ast`` pass rewrites a kernel BODY; a plain-python helper's body it
    never sees. So ``v[k] = …`` in a helper reaches ``Value.__setitem__``, which
    refuses by name instead of letting Python say "object does not support item
    assignment" — a message that would send the author looking for a bug in
    their own code rather than at the one rule that applies."""
    with pytest.raises(HawkError, match="only works inside a KERNEL body"):
        @hawk.kernel
        def host(v: hawk.Vector[W], out: hawk.Mutable[hawk.Vector[W]]):
            out = _mutate_in_a_helper(v)


def test_a_write_leaves_the_value_it_was_given_alone():
    """``set_component_at`` is a VALUE, not a mutation: a body that reads ``v``
    after writing a component of a copy must read the value ``v`` had. If the
    emitter had written through the operand instead of through a copy, this
    kernel would read back the WRITTEN vector and the row would say so."""
    @hawk.kernel
    def both(v: hawk.Vector[W], out: hawk.Mutable[hawk.Scalar]):
        held = v
        for c in range(W):
            held[c] = v[c] * 10.0
        out = m.vsum(held) + m.vsum(v)

    value = np.array([1.0, 2.0, 4.0])
    got = float(evaluate(both.sinks, {"v": value})["out"])
    assert got == pytest.approx(float(np.sum(value) * 10 + np.sum(value)))


# --------------------------------------------------------------------------- #
# The refusals — each names the spelling that works.
# --------------------------------------------------------------------------- #
def test_a_comprehension_names_the_component_write():
    """A comprehension stays banned — the body is traced ONCE, so a
    comprehension over a lowered loop's index would build one element — but the
    refusal now SAYS what to write instead. A "no" with no route is how a body
    ends up building a python list and getting a width error three statements
    later."""
    with pytest.raises(HawkError) as excinfo:
        @hawk.kernel
        def wrong(v: hawk.Vector[W], out: hawk.Mutable[hawk.Vector[W]]):
            out = m.vec([v[i] * 2.0 for i in range(W)])

    message = str(excinfo.value)
    assert "list comprehension" in message, message
    assert "`v[k] = …`" in message, message


def test_a_python_list_mutated_inside_a_loop_names_the_component_write():
    """The other route to the same mistake: bind a list before the loop and
    append to it inside. The body runs ONCE against a symbolic index, so the
    list would hold exactly one element — a bare call statement inside a lowered
    ``for`` discards its value and can therefore only be mutating a python
    object, which is refused by NAME rather than left to surface as a width
    error later."""
    with pytest.raises(HawkError) as excinfo:
        @hawk.kernel
        def wrong(v: hawk.Vector[W], out: hawk.Mutable[hawk.Vector[W]]):
            parts = []
            for c in range(W):
                parts.append(v[c])
            out = m.vec(parts)

    message = str(excinfo.value)
    assert "bare call statement" in message, message
    assert "`v[k] = …`" in message, message


def test_a_component_write_directly_on_a_mutable_plane_matches_the_bound_local_vector():
    """A mutable plane's bare name is an ordinary local, so writing one
    component of it directly inside the loop -- with no separate local bound
    first -- traces: the loop carries ``out`` itself, seeded from its own
    launch-start value (the component a given iteration does not touch yet
    still has to come from somewhere), the same way :func:`rollout_write`
    seeds its separate carried local from ``v``. Every component is written
    exactly once over the full range, so both reach the same answer -- the
    launch-start seed's own numbers never survive to either result."""
    @hawk.kernel
    def direct(v: hawk.Vector[W], k1: hawk.Vector[W], dt: hawk.Param,
              out: hawk.Mutable[hawk.Vector[W]]):
        for c in range(W):
            out[c] = _rk4_step(v[c], k1[c], dt)

    env = {**ENV, "out": np.zeros(W)}
    got = np.asarray(evaluate(direct.sinks, env)["out"], dtype=float)
    want = np.asarray(evaluate(rollout_write.sinks, ENV)["out"], dtype=float)
    np.testing.assert_array_equal(got, want)


def test_a_float_index_is_refused_rather_than_rounded():
    """An index is an ADDRESS. A rank-0 floating value would reach the emitter
    as a cast and would silently read the neighbouring component when the
    arithmetic that produced it landed a hair below an integer."""
    with pytest.raises(HawkError, match="traced INTEGER index"):
        @hawk.kernel
        def wrong(v: hawk.Vector[W], s: hawk.Param, out: hawk.Mutable[hawk.Scalar]):
            out = v[s]


def test_a_component_write_on_a_rank_zero_value_is_refused():
    """``x[k] = e`` on a scalar has no meaning: there is no component to write.
    Refused naming the value's own type rather than failing inside the type
    rule with a rank message the author cannot place."""
    with pytest.raises(HawkError, match="no components to write"):
        @hawk.kernel
        def wrong(s: hawk.Param, out: hawk.Mutable[hawk.Scalar]):
            acc = s
            for c in range(W):
                acc[c] = s
            out = acc
