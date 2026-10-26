# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A read-only per-lane input of a ``Grid`` kernel binds broadcast.

Its shape is any that broadcasts, right-aligned, against ``(head, *lane shape)``
(NumPy's rule) and it binds as the caller's own buffer, compact. The pattern (the
lane axes it varies along) picks a variant of the kernel, built once per pattern;
the exact shape runs the kernel untouched. Both hawk doors classify with the one
module (:mod:`hawk._broadcast`): ``k(...)`` runs the variant; a loaded artifact is
one kernel's and refuses a broadcast shape in the same words.

* **Values** against NumPy's broadcast, through ``k(...)`` (eagle's host plan) and
  through ``hawk.runtime.run`` on the variant's own artifact.
* **Refusals** name the given shape and the ``(head, *lane shape)`` it was checked
  against; written, masked and matrix planes take the exact shape.
* **No copy**: the variant binds the caller's memory, asserted by address.
* **The reverse** of a broadcast input is the sum over the axes it does not vary
  along, in its own compact shape, compensated.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as hm
from hawk import Grid, HawkError, Scalar, Terminated, Vector
from hawk import _broadcast as B
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
import eagle
from hawk.diff.transform import _masked_sink
from hawk.ir import AccumWrite, Leaf
from hawk.ir.nodes import Assign
from hawk.types import TensorType
from hawk.trace.kernel import Kernel

A, NB, NC = 3, 4, 6
LANE = (A, NB, NC)


@hawk.kernel
def affine(lane: Grid["a", "b":4, "c":6], x: Scalar) -> Scalar:
    return 2.0 * x + 1.0


@hawk.kernel
def mix(lane: Grid["a", "b":4, "c":6], v: Vector[3]) -> Scalar:
    return v[0] + 2.0 * v[1] + 3.0 * v[2]


@hawk.kernel
def square(lane: Grid["a", "b":4, "c":6], x: Scalar) -> Scalar:
    return x * x


@hawk.kernel
def vmix(lane: Grid["a", "b":4, "c":6], v: Vector[3]) -> Scalar:
    return v[0] * v[1] + 2.0 * v[2]


@hawk.kernel
def switched(lane: Grid["a", "b":4, "c":6], x: Scalar) -> Scalar:
    k = hm.int32(x * 3.0)
    return hm.dispatch(k, [x * x, 3.0 * x + 1.0, hm.sin(x)], policy="switch")


@hawk.kernel
def stopped(lane: Grid["a", "b":4, "c":6], x: Scalar, terminated: Terminated) -> Scalar:
    return x * x * x


@pytest.fixture
def run_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))


def _full(x, shape=LANE):
    return np.broadcast_to(np.asarray(x), shape)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(11)


def _artifact(kernel, root, cache_dir):
    bundle = build_bundle([kernel], pathlib.Path(root) / kernel.name, targets=("host",),
                          cache_dir=cache_dir, host_profile="native")
    return rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), bundle


# 1. a scalar binds as a uniform, no lane-shaped buffer

def test_scalar_binds_as_uniform(run_cache):
    for given in (0.5, np.float64(0.5), np.array(0.5)):
        run, args, kw = B.route(affine, (), {"x": given, "a": A})
        assert run.planes["x"].form == "uniform"
        assert isinstance(kw["x"], float) and kw["x"] == 0.5
    np.testing.assert_array_equal(affine(0.5, a=A), np.full(LANE, 2.0))


# 2. a size-1 middle axis

def test_size1_middle_axis_broadcasts(run_cache, rng):
    x = rng.random((A, 1, NC))
    np.testing.assert_array_equal(affine(x, a=A), 2.0 * _full(x) + 1.0)
    run, _args, _kw = B.route(affine, (x,), {"a": A})
    assert run.planes["x"].form == "table" and run.planes["x"].dims == ()   # the runtime axis is kept
    named, _a, _k = B.route(affine, (rng.random((1, NB, NC)),), {"a": A})
    assert [d[0] for d in named.planes["x"].dims] == ["b", "c"]


# 3. fewer axes align to the right

@pytest.mark.parametrize("shape", [(NC,), (NB, NC)])
def test_rank_lower_is_right_aligned(run_cache, rng, shape):
    x = rng.random(shape)
    np.testing.assert_array_equal(affine(x, a=A), 2.0 * _full(x) + 1.0)


# 4. a column view keeps the runtime axis, through a flat table

def test_column_view_keeps_runtime_axis(run_cache, rng):
    col = rng.random(A)
    x = col[:, None, None]
    np.testing.assert_array_equal(affine(x, a=A), 2.0 * _full(x) + 1.0)
    run, _args, _kw = B.route(affine, (x,), {"a": A})
    assert run.planes["x"].form == "table" and run.planes["x"].dims == ()


# 5. every axis of size 1 is a uniform

def test_all_size1_is_uniform(run_cache):
    x = np.array([[[0.25]]])
    run, _args, kw = B.route(affine, (x,), {"a": A})
    assert run.planes["x"].form == "uniform" and kw["x"] == 0.25
    np.testing.assert_array_equal(affine(x, a=A), np.full(LANE, 1.5))


# 6. a Vector keeps its component axis exact; the lane axes broadcast

def test_vector_head_exact_lanes_broadcast(run_cache, rng):
    v = rng.random((3, 1, NC))
    want = v[0] + 2.0 * v[1] + 3.0 * v[2]
    np.testing.assert_allclose(mix(v, a=A), _full(want))
    one = rng.random(3)
    np.testing.assert_allclose(mix(one, a=A), np.full(LANE, one[0] + 2.0 * one[1] + 3.0 * one[2]))
    with pytest.raises((HawkError, ValueError), match=r"\(1, 3, 4, 6\)"):
        mix(rng.random((1, A, NB, NC)), a=A)


# 7. a shape that does not broadcast names both shapes, at both doors

def test_incompatible_shape_names_both_shapes(run_cache, rng, tmp_path_factory, cache_dir):
    with pytest.raises(HawkError, match=r"shape \(3, 2, 6\).*against \(a, 4, 6\)"):
        affine(rng.random((A, 2, NC)), a=A)
    with pytest.raises(HawkError, match=r"shape \(5, 1, 6\).*against \(a, 4, 6\)"):
        affine(rng.random((5, 1, NC)), a=A)
    art, _bundle = _artifact(affine, tmp_path_factory.mktemp("bc_refuse"), cache_dir)
    with pytest.raises(HawkError, match=r"shape \(3, 2, 6\).*against \(a, 4, 6\)"):
        rt.run(art, x=rng.random((A, 2, NC)), out=np.zeros(LANE), a=A)
    with pytest.raises(HawkError, match=r"broadcasts against \(a, 4, 6\) along \[a, c\]"):
        rt.run(art, x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)


# 8. written planes take the exact shape

def test_mutable_and_output_refuse_broadcast(run_cache, rng, tmp_path_factory, cache_dir):
    x = rng.random(LANE)
    with pytest.raises(HawkError, match=r"'out' has shape \(3, 1, 6\).*exact shape \(a, 4, 6\)"):
        affine(x, a=A, out=np.zeros((A, 1, NC)))
    art, _bundle = _artifact(affine, tmp_path_factory.mktemp("bc_exact"), cache_dir)
    with pytest.raises(HawkError, match=r"'out' has shape \(3, 1, 6\).*exact shape \(a, 4, 6\)"):
        rt.run(art, x=x, out=np.zeros((A, 1, NC)), a=A)


# 9. the variant binds the caller's memory

def test_broadcast_binds_caller_buffer(run_cache, rng):
    x = rng.random((A, 1, NC))
    _run, _args, kw = B.route(affine, (x,), {"a": A})
    assert kw["x"].__array_interface__["data"][0] == x.__array_interface__["data"][0]
    assert np.shares_memory(kw["x"], x)
    strided = rng.random((A, 2, NC))[:, :1, :]
    with pytest.raises(HawkError, match="C-contiguous"):
        B.route(affine, (strided,), {"a": A})


# 10. one variant per pattern; the exact shape is the kernel itself

def test_one_variant_per_pattern(run_cache, rng):
    before = len(affine._variants)
    first, _a, _k = B.route(affine, (rng.random((A, 1, NC)),), {"a": A})
    again, _a, _k = B.route(affine, (rng.random((A, 1, NC)),), {"a": A})
    other, _a, _k = B.route(affine, (rng.random(NC),), {"a": A})
    assert first is again and first is not other and first is not affine
    exact, _a, _k = B.route(affine, (rng.random(LANE),), {"a": A})
    assert exact is affine
    assert len(affine._variants) - before <= 2


# 11. a plan bound to one pattern is not rebound to another

def test_rebind_new_pattern_refused(rng):
    axes = affine.lane_space
    held = B.kept_axes((A, 1, NC), (), axes, A)
    B.check_rebind("x", held, B.kept_axes((A, 1, NC), (), axes, A), axes)
    with pytest.raises(HawkError, match=r"varying along \[a, c\].*along \[c\]"):
        B.check_rebind("x", held, B.kept_axes((NC,), (), axes, A), axes)


# 12. the reverse of a broadcast input sums the axes it does not vary along

def test_vjp_broadcast_input_sums_broadcast_axes(tmp_path_factory, cache_dir, rng):
    axes = square.lane_space
    kept = B.kept_axes((A, 1, NC), (), axes, A)
    var = B.variant(square, {"x": kept})
    d = Kernel("square_vjp", vjp(var, wrt=("x",)))
    assert [s.policy for s in d.sinks if isinstance(s, AccumWrite) and s.name == "bar_x"] == ["compensated"]
    root = tmp_path_factory.mktemp("bc_vjp")
    art = _derived(var, vjp, "x", root, cache_dir)
    assert art.sidecar["companions"] == {"bar_x_c": "bar_x"}
    x = rng.random((A, 1, NC)) * 1e8
    g = rng.random(LANE) * 1e-8 - 0.5e-8
    got = np.zeros((A, 1, NC))
    rt.run(art, x=x.reshape(-1), bar_out=g.reshape(-1), bar_x=got.reshape(-1), a=A)
    np.testing.assert_allclose(got, np.sum(2.0 * x * g, axis=1, keepdims=True), rtol=1e-12)
    # the sum over b is correctly rounded: within 1 ulp of the exact one, which a plain sum is not owed
    exact = np.array([[[math.fsum(2.0 * x[i, 0, j] * g[i, k, j] for k in range(NB))
                        for j in range(NC)]] for i in range(A)])
    assert np.max(np.abs(got - exact) / np.abs(exact)) < 4e-16


# -- autodiff through a broadcast input: exact, both modes ----------------------

#: The patterns of a scalar input: ``(a, 1, c)``, ``(c,)``, a scalar, all size 1.
PATTERNS = [(A, 1, NC), (NC,), (), (1, 1, 1)]
FD_RTOL = 1e-6      # f64 central differences at h = 1e-6, away from a switch's edge


def _derived(primal, make, wrt, root, cache_dir):
    """The artifact of ``make(primal)`` for ``wrt``, the primal built beside it."""
    d = Kernel(primal.name + "_" + make.__name__, make(primal, wrt=(wrt,)))
    kw = dict(targets=("host",), cache_dir=cache_dir, host_profile="native")
    home = build_bundle([primal], pathlib.Path(root) / "primal", **kw)
    bundle = build_bundle([d], pathlib.Path(root) / d.name,
                          derivative={d.name: (primal.name, home.digest)}, **kw)
    return rt.load(bundle.directory, d.name, sidecar_of(bundle, d.name))


def _bound(kernel, name, value, head=(), cell=False):
    """``(variant, kept, the buffer it binds)`` for ``value`` given to ``name``."""
    kept = B.kept_axes(np.shape(value), head, kernel.lane_space, A)
    var = B.variant(kernel, {name: kept}, reverse=(name,) if cell else ())
    if cell and not any(kept) and not head:
        return var, kept, np.ravel(value)[:1].copy()
    return var, kept, B.compact(name, np.asarray(value), kept, head, "test")


def _summed(g, shape):
    """``g`` (lane-shaped) summed over the axes a broadcast ``shape`` does not vary along."""
    padded = (1,) * (3 - len(shape)) + tuple(shape)
    axes = tuple(i for i, d in enumerate(padded) if d == 1)
    return np.sum(g, axis=axes, keepdims=True).reshape(shape)


@pytest.mark.parametrize("shape", PATTERNS)
def test_jvp_broadcast_input_equals_dense_jvp_of_broadcast_tangent(tmp_path_factory, cache_dir, rng, shape):
    x, t = rng.random(shape) + 0.5, rng.random(shape) - 0.5
    var, kept, xb = _bound(square, "x", x)
    art = _derived(var, jvp, "x", tmp_path_factory.mktemp("bc_jvp"), cache_dir)
    tb = B.compact("dot_x", t, kept, (), "test")
    out = np.zeros(A * NB * NC)
    rt.run(art, x=xb, dot_x=tb, dot_out=out, a=A)
    np.testing.assert_allclose(out.reshape(LANE), 2.0 * _full(x) * _full(t), rtol=1e-14)


@pytest.mark.parametrize("shape", PATTERNS)
def test_vjp_broadcast_input_sums_broadcast_axes_for_every_pattern(tmp_path_factory, cache_dir, rng, shape):
    x, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    var, kept, xb = _bound(square, "x", x, cell=True)
    art = _derived(var, vjp, "x", tmp_path_factory.mktemp("bc_vjp_all"), cache_dir)
    bar = np.zeros(1 if not any(kept) else np.size(x))
    rt.run(art, x=xb, bar_out=g.reshape(-1), bar_x=bar, a=A)
    want = _summed(2.0 * _full(x) * g, np.shape(x))
    np.testing.assert_allclose(bar.reshape(np.shape(x)), want, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("shape", [(3, 1, NC), (3,)])
def test_vjp_broadcast_vector_input_sums_broadcast_axes(tmp_path_factory, cache_dir, rng, shape):
    v, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    var, kept, vb = _bound(vmix, "v", v, head=(3,))
    art = _derived(var, vjp, "v", tmp_path_factory.mktemp("bc_vjp_vec"), cache_dir)
    bar = np.zeros(v.size)
    rt.run(art, v=vb, bar_out=g.reshape(-1), bar_v=bar, a=A)
    lane = np.broadcast_to(v.reshape(3, *(1,) * (3 - (v.ndim - 1)), *v.shape[1:]), (3, *LANE))
    parts = np.stack([lane[1] * g, lane[0] * g, 2.0 * g])
    want = np.stack([_summed(p, v.shape[1:]) for p in parts])
    np.testing.assert_allclose(bar.reshape(v.shape), want, rtol=1e-12, atol=1e-13)


def _central(f, x, h=1e-6):
    """``df/dx`` elementwise by central differences."""
    out = np.zeros(x.size)
    for i in range(x.size):
        up, down = x.copy().reshape(-1), x.copy().reshape(-1)
        up[i] += h
        down[i] -= h
        out[i] = (f(up.reshape(x.shape)) - f(down.reshape(x.shape))) / (2 * h)
    return out.reshape(x.shape)


def _switched_ref(x):
    full = _full(x)
    return np.where(full < 1 / 3, full * full, np.where(full < 2 / 3, 3.0 * full + 1.0, np.sin(full)))


def test_switch_reads_a_broadcast_input_vjp_and_jvp_match_finite_differences(tmp_path_factory, cache_dir, rng):
    """The input sits well inside each switch branch, so the central difference
    sees one branch; relative tolerance ``FD_RTOL``."""
    x = np.array([[[0.15, 0.5, 0.85, 0.2, 0.45, 0.9]]] * A)
    t, g = rng.random(x.shape) - 0.5, rng.random(LANE) - 0.5
    var, kept, xb = _bound(switched, "x", x)
    root = tmp_path_factory.mktemp("bc_switch")
    bar = np.zeros(x.size)
    rt.run(_derived(var, vjp, "x", root, cache_dir), x=xb, bar_out=g.reshape(-1), bar_x=bar,
           a=A)
    want = _central(lambda z: np.sum(g * _switched_ref(z)), x)
    np.testing.assert_allclose(bar.reshape(x.shape), want, rtol=FD_RTOL, atol=1e-9)
    dot = np.zeros(g.size)
    rt.run(_derived(var, jvp, "x", root, cache_dir), x=xb, dot_x=B.compact("dot_x", t, kept, (), "test"),
           dot_out=dot, a=A)
    h = 1e-6
    fd = (_switched_ref(x + h * t) - _switched_ref(x - h * t)) / (2 * h)
    np.testing.assert_allclose(dot.reshape(LANE), fd, rtol=FD_RTOL, atol=1e-9)


def test_per_lane_stop_reads_a_broadcast_input_vjp_and_jvp_match_finite_differences(
        tmp_path_factory, cache_dir, rng):
    """A terminated lane commits nothing, so it contributes nothing to either mode."""
    x = rng.random((A, 1, NC)) + 0.5
    t, g = rng.random(x.shape) - 0.5, rng.random(LANE) - 0.5
    stop = rng.random(LANE) < 0.4
    var, kept, xb = _bound(stopped, "x", x)
    root = tmp_path_factory.mktemp("bc_stop")

    def ref(z):
        return np.where(stop, 0.0, _full(z) ** 3)

    bar = np.zeros(x.size)
    rt.run(_derived(var, vjp, "x", root, cache_dir), x=xb, terminated=stop.reshape(-1),
           bar_out=g.reshape(-1), bar_x=bar, a=A)
    want = _central(lambda z: np.sum(g * ref(z)), x)
    np.testing.assert_allclose(bar.reshape(x.shape), want, rtol=FD_RTOL, atol=1e-9)
    dot = np.zeros(g.size)
    rt.run(_derived(var, jvp, "x", root, cache_dir), x=xb, terminated=stop.reshape(-1),
           dot_x=B.compact("dot_x", t, kept, (), "test"), dot_out=dot, a=A)
    h = 1e-6
    fd = (ref(x + h * t) - ref(x - h * t)) / (2 * h)
    np.testing.assert_allclose(dot.reshape(LANE), fd, rtol=FD_RTOL, atol=1e-9)


# -- vjp(k) / jvp(k) called directly with a broadcast input ----------------------

def _differentiated(kernel, make, wrt):
    return Kernel(f"{kernel.name}_{make.__name__}", make(kernel, wrt=(wrt,)))


@pytest.mark.parametrize("shape", PATTERNS)
def test_vjp_called_directly_returns_the_gradient_in_the_inputs_own_shape(run_cache, rng, shape):
    x, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    got = _differentiated(square, vjp, "x")(x=x, bar_out=g, a=A)
    assert got.shape == np.shape(x)
    np.testing.assert_allclose(got, _summed(2.0 * _full(x) * g, np.shape(x)), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("shape", PATTERNS)
def test_jvp_called_directly_equals_the_dense_jvp_of_the_broadcast_tangent(run_cache, rng, shape):
    x, t = rng.random(shape) + 0.5, rng.random(shape) - 0.5
    got = _differentiated(square, jvp, "x")(x=x, dot_x=t, a=A)
    np.testing.assert_allclose(got, 2.0 * _full(x) * _full(t), rtol=1e-14)


def test_a_tangent_must_vary_along_its_inputs_axes(run_cache, rng):
    with pytest.raises(HawkError, match=r"'dot_x' must vary along the axes of 'x', \[a, c\]"):
        _differentiated(square, jvp, "x")(x=rng.random((A, 1, NC)), dot_x=rng.random(NC), a=A)


@pytest.mark.parametrize("shape", [(3, 1, NC), (3,)])
def test_vjp_called_directly_with_a_broadcast_vector(run_cache, rng, shape):
    v, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    got = _differentiated(vmix, vjp, "v")(v=v, bar_out=g, a=A)
    assert got.shape == v.shape
    lane = np.broadcast_to(v.reshape(3, *(1,) * (4 - v.ndim), *v.shape[1:]), (3, *LANE))
    want = np.stack([_summed(p, v.shape[1:]) for p in (lane[1] * g, lane[0] * g, 2.0 * g)])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_a_compensated_scatter_stays_compensated_under_a_terminated_mask():
    mask = Leaf("terminated", "terminated", "terminated", TensorType((), "bool"))
    value = Leaf("vocab_read", "per_sample", "v", TensorType((), "f64"))
    index = Leaf("vocab_read", "per_sample", "k", TensorType((), "i32"))
    masked = _masked_sink(AccumWrite("acc", value, index, value.ttype, "compensated"), mask)
    assert masked.policy == "compensated"
    assert isinstance(_masked_sink(Assign("y", value, value.ttype), mask), Assign)


def test_rebinding_a_plan_to_another_pattern_is_refused(run_cache, rng):
    var = B.variant(affine, {"x": B.kept_axes((A, 1, NC), (), affine.lane_space, A)})
    bound = eagle.deploy(var).bind(x=rng.random(A * NC), out=np.zeros(LANE), a=A)
    bound.rebind(x=rng.random(A * NC))
    with pytest.raises(HawkError, match=r"varying along \[a, c\].*rebound varying along \[c\]"):
        bound.rebind(x=rng.random(NC))
