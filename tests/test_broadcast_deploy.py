# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.deploy(k)`` takes a per-lane input given broadcast, as ``k(...)`` does.

The plan keeps the hawk kernel it was built from: a broadcast shape runs the
kernel's variant of that pattern (:mod:`hawk._broadcast`), built once per pattern
with the plan's own build options, the caller's buffer bound compact. A loaded
artifact has no source, so it refuses by name; the rebind and graph rules are the
variant plan's own.

* **Values** against NumPy's broadcast through ``plan(...)``, ``plan.run`` and
  ``plan.bind``, for every pattern (a runtime-axis keep, a uniform, a Vector head).
* **One variant per pattern**, built with the plan's ``scalar_type`` and cache.
* **Rebinding** a bound broadcast plane: the compact buffer, the same pattern only;
  a captured graph swaps a same-shape buffer and refuses any other.
* **Reverse** of a broadcast input through a recorded graph is the sum over the axes
  it does not vary along, compensated, equal to the hawk door's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import eagle
import hawk
from eagle import diff as ediff
from hawk import Grid, HawkError, Scalar, Vector
from hawk import _broadcast as B
from hawk.diff import vjp
from hawk.trace.kernel import Kernel

A, NB, NC = 3, 4, 6
LANE = (A, NB, NC)


@hawk.kernel
def daffine(lane: Grid["a", "b":4, "c":6], x: Scalar) -> Scalar:
    return 2.0 * x + 1.0


@hawk.kernel
def dcube(lane: Grid["a", "b":4, "c":6], x: Scalar) -> Scalar:
    return x * x * x


@hawk.kernel
def dmix(lane: Grid["a", "b":4, "c":6], v: Vector[3]) -> Scalar:
    return v[0] * v[1] + 2.0 * v[2]


@pytest.fixture
def run_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def plans(cache_dir):
    return {k.name: eagle.deploy(k, targets=("host",), cache_dir=cache_dir)
            for k in (daffine, dcube, dmix)}


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(23)


def _full(x, shape=LANE):
    return np.broadcast_to(np.asarray(x), shape)


def _summed(g, shape):
    padded = (1,) * (3 - len(shape)) + tuple(shape)
    axes = tuple(i for i, d in enumerate(padded) if d == 1)
    return np.sum(g, axis=axes, keepdims=True).reshape(shape)


#: runtime-axis kept [a, c], [c], a uniform, a uniform, [a, b]
PATTERNS = [(A, 1, NC), (NC,), (), (1, 1, 1), (A, NB, 1)]


# 1. values, through every door of the plan

@pytest.mark.parametrize("shape", PATTERNS)
def test_deploy_run_call_and_bind_match_numpy_for_each_pattern(plans, rng, shape):
    plan = plans["daffine"]
    x = rng.random(shape)
    want = 2.0 * _full(x) + 1.0
    np.testing.assert_array_equal(np.asarray(plan(x, a=A)), want)
    np.testing.assert_array_equal(np.asarray(plan(x=x, a=A)), want)
    np.testing.assert_array_equal(np.asarray(plan.run(x=x, out=np.zeros(LANE), a=A)), want)
    out = np.zeros(LANE)
    plan.bind(x=x, out=out, a=A).launch()
    np.testing.assert_array_equal(out, want)


@pytest.mark.parametrize("shape", [(3, 1, NB, NC), (3,)])
def test_deploy_broadcast_vector_input_keeps_its_head(plans, rng, shape):
    v = rng.random(shape) + 0.5
    lane = np.broadcast_to(v.reshape(3, *(1,) * (4 - v.ndim), *v.shape[1:]) if v.ndim > 1
                           else v.reshape(3, 1, 1, 1), (3, *LANE))
    want = lane[0] * lane[1] + 2.0 * lane[2]
    np.testing.assert_allclose(np.asarray(plans["dmix"](v, a=A)), want, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(plans["dmix"].run(v=v, out=np.zeros(LANE), a=A)), want, rtol=1e-15)


def test_a_dense_shape_runs_the_plan_itself(plans, rng):
    plan = plans["daffine"]
    x = rng.random(LANE)
    routed, _args, _kw = plan._routed((x,), {"a": A})
    assert routed is plan
    np.testing.assert_array_equal(np.asarray(plan(x, a=A)), 2.0 * x + 1.0)


# 2. one variant per pattern

def test_variant_plan_memoised_per_pattern(plans, rng):
    plan = plans["daffine"]
    first, _a, _k = plan._routed((rng.random((A, 1, NC)),), {"a": A})
    again, _a, _k = plan._routed((rng.random((A, 1, NC)),), {"a": A})
    other, _a, _k = plan._routed((rng.random(NC),), {"a": A})
    exact, _a, _k = plan._routed((rng.random(LANE),), {"a": A})
    assert first is again and first is not other and first is not plan and exact is plan
    assert first.kernel.broadcast and first._routed((rng.random(NC),), {"a": A})[0] is first   # no second routing
    assert first.kernel is B.variant(daffine, dict(first.kernel.broadcast))


# 3. the plan's build options reach the variant

def test_deploy_options_reach_the_variant(tmp_path, rng):
    cache = str(tmp_path / "opts")
    plan = eagle.deploy(daffine, targets=("host",), cache_dir=cache, scalar_type="float32")
    x = rng.random((A, 1, NC)).astype(np.float32)
    got = np.asarray(plan(x, a=A))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, 2.0 * _full(x) + 1.0, rtol=1e-6)
    variant_plan, _a, _k = plan._routed((x,), {"a": A})
    assert variant_plan._build == plan._build == (("host",), cache, "float32")
    assert variant_plan.plugin._hawk_source[2] == plan.plugin._hawk_source[2]


# 4. rebinding a bound broadcast plane

def test_bind_then_rebind_the_compact_buffer_only(plans, rng):
    plan = plans["daffine"]
    x = rng.random((A, 1, NC))
    out = np.zeros(LANE)
    bound = plan.bind(x=x, out=out, a=A)
    bound.launch()
    np.testing.assert_array_equal(out, 2.0 * _full(x) + 1.0)
    x2 = rng.random(A * NC)
    bound.rebind(x=x2).launch()
    np.testing.assert_array_equal(out, 2.0 * _full(x2.reshape(A, 1, NC)) + 1.0)
    with pytest.raises(HawkError, match=r"varying along \[a, c\].*rebound varying along \[c\]"):
        bound.rebind(x=rng.random(NC))
    with pytest.raises(HawkError, match=r"varying along \[a, c\].*rebound the exact lane shape"):
        bound.rebind(x=rng.random(LANE))
    with pytest.raises(HawkError, match=r"was bound with shape \(18,\).*cannot be rebound with shape \(3, 1, 6\)"):
        bound.rebind(x=rng.random((A, 1, NC)))


# 5. a captured graph swaps the compact buffer

def test_graph_swaps_a_same_shape_buffer_and_refuses_any_other(plans, rng):
    plan = plans["daffine"]
    out = np.zeros(LANE)
    bound = plan.bind(x=rng.random((A, 1, NC)), out=out, a=A)
    x0 = bound._planes["x"]                 # the compact view the graph holds
    assert x0.shape == (A * NC,)
    graph = eagle.Graph().add(bound).build()
    graph.launch().synchronize()
    np.testing.assert_array_equal(out, 2.0 * _full(x0.reshape(A, 1, NC)) + 1.0)
    x1 = rng.random(A * NC)
    graph._rebind([(x0, x1)])
    graph.launch().synchronize()
    np.testing.assert_array_equal(out, 2.0 * _full(x1.reshape(A, 1, NC)) + 1.0)
    for other in (rng.random(NC), rng.random((A, 1, NC))):
        with pytest.raises(ValueError, match="call the builder again"):
            graph._rebind([(x1, other)])


# 6. a loaded artifact and a prebuilt plugin refuse by name

def test_prebuilt_plugin_refuses_a_broadcast_shape_naming_the_variant(plans, rng):
    prebuilt = eagle.plan.auto(plans["daffine"].plugin)
    with pytest.raises(ValueError, match=r"lane shape \(3, 1, 6\).*contradicts.*binds through the hawk kernel"):
        prebuilt.run(x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)


def test_artifact_door_refusal_is_unchanged(tmp_path_factory, cache_dir, rng):
    import pathlib

    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    bundle = build_bundle([daffine], pathlib.Path(tmp_path_factory.mktemp("bd_art")) / "affine",
                          targets=("host",), cache_dir=cache_dir, host_profile="native")
    art = rt.load(bundle.directory, "daffine", sidecar_of(bundle, "daffine"))
    with pytest.raises(HawkError, match=r"broadcasts against \(a, 4, 6\) along \[a, c\].*hawk._broadcast.variant"):
        rt.run(art, x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)


# 7. exact planes stay exact; a bad shape names both shapes

def test_written_plane_stays_exact_through_deploy_run(plans, rng):
    with pytest.raises(ValueError, match=r"'out' has shape \(3, 1, 6\).*exact shape \(a, 4, 6\)"):
        plans["daffine"].run(x=rng.random(LANE), out=np.zeros((A, 1, NC)), a=A)


def test_incompatible_shape_names_both_shapes_through_deploy(plans, rng):
    with pytest.raises(ValueError, match=r"shape \(3, 2, 6\).*against \(a, 4, 6\)"):
        plans["daffine"](rng.random((A, 2, NC)), a=A)


# 8. the reverse of a broadcast input through a recorded graph

def _reverse(plan, name, value, cot, head=()):
    uniform = not head and np.size(value) == 1
    bound = plan.bind(**{name: float(np.ravel(value)[0]) if uniform else value},
                      out=np.zeros(LANE), a=A)
    graph = eagle.Graph(record=True).add(bound).build()
    graph.launch().synchronize()
    return np.asarray(ediff.reverse(graph.record, {"out": cot.reshape(-1)})[name])


@pytest.mark.parametrize("shape", [(A, 1, NC), (NC,), (), (A, NB, 1)])
def test_recorded_reverse_sums_the_broadcast_axes(plans, rng, shape):
    x, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    got = _reverse(plans["dcube"], "x", x, g)
    want = _summed(3.0 * _full(x) ** 2 * g, np.shape(x))
    assert got.shape == np.shape(x) or got.size == want.size
    np.testing.assert_allclose(got.reshape(np.shape(x)), want, rtol=1e-12, atol=1e-13)


def test_recorded_reverse_of_a_broadcast_vector(plans, rng):
    v, g = rng.random((3, 1, NB, NC)) + 0.5, rng.random(LANE)
    got = _reverse(plans["dmix"], "v", v, g, head=(3,))
    lane = np.broadcast_to(v, (3, *LANE))
    parts = np.stack([lane[1] * g, lane[0] * g, 2.0 * g])
    want = np.stack([_summed(p, v.shape[1:]) for p in parts])
    np.testing.assert_allclose(got.reshape(v.shape), want, rtol=1e-12, atol=1e-13)


def test_recorded_reverse_is_compensated_and_equals_the_hawk_door(plans, rng, run_cache):
    x = np.full((A, 1, NC), 1e8)
    g = (rng.random(LANE) - 0.5) * 1e-8
    got = _reverse(plans["dcube"], "x", x, g).reshape(A, 1, NC)
    door = Kernel("dcube_vjp", vjp(dcube, wrt=("x",)))(x=x, bar_out=g, a=A)
    np.testing.assert_array_equal(got, np.asarray(door).reshape(A, 1, NC))
    exact = np.array([[[math.fsum(3.0 * x[i, 0, j] ** 2 * g[i, k, j] for k in range(NB))
                        for j in range(NC)]] for i in range(A)])
    assert np.max(np.abs(got - exact) / np.abs(exact)) < 4e-15      # the products round once, the sum does not


def test_a_derivative_plan_routes_a_broadcast_input_as_the_hawk_door_does(rng, run_cache):
    x, g = rng.random((A, 1, NC)) + 0.5, rng.random(LANE) - 0.5
    d = Kernel("dcube_vjp2", vjp(dcube, wrt=("x",)))
    got = np.asarray(d._deploy()(x=x, bar_out=g, a=A))
    np.testing.assert_allclose(got, _summed(3.0 * _full(x) ** 2 * g, x.shape), rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(got, np.asarray(d(x=x, bar_out=g, a=A)))


# 9. a device plan routes the same way (L1 runs it through the pool)

@pytest.mark.gpu
def test_deploy_run_broadcast_on_the_device(cache_dir, rng):
    cp = pytest.importorskip("cupy")
    plan = eagle.deploy(daffine, cache_dir=cache_dir)
    x = rng.random((A, 1, NC))
    got = plan(cp.asarray(x), a=A)
    np.testing.assert_allclose(cp.asnumpy(got), 2.0 * _full(x) + 1.0, rtol=1e-15)
    out = cp.zeros(LANE)
    plan.bind(x=cp.asarray(x), out=out, a=A).launch()
    np.testing.assert_allclose(cp.asnumpy(out), 2.0 * _full(x) + 1.0, rtol=1e-15)
