# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.deploy(k)`` takes a per-lane input given broadcast, as ``k(...)`` does.

The plan keeps the hawk kernel it was built from: a broadcast shape runs the
kernel's twin of that pattern (:mod:`hawk._broadcast`), built once per pattern
with the plan's own build options, the caller's buffer bound compact. A plan over a
loaded artifact has no source, so it refuses by name.

* **Values** against NumPy's broadcast through ``plan(...)``, ``plan.run`` and
  ``plan.bind``, for every pattern (a runtime-axis keep, a uniform, a Vector head).
* **One twin per pattern**, built with the plan's ``scalar_type`` and cache.
* **Rebinding** a bound broadcast plane: any shape of its pattern and the compact
  buffer, never another pattern.
* **Reverse** of a broadcast input is the sum over the axes it does not vary along,
  compensated: the direct call of the derivative against ``math.fsum`` closed forms.
  The same through a recorded graph is the graph builder's rows (left, listed with the run).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hawk
from hawk import Grid, HawkError, Scalar, Vector
from hawk import _broadcast as B
from hawk import plans as plans_module
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
    """The three kernels deployed: a plan routes a broadcast input by itself."""
    return {k.name: hawk.deploy(k, targets=("host",), cache_dir=cache_dir) for k in (daffine, dcube, dmix)}


@pytest.fixture
def rng():
    """A generator seeded afresh for each test: a test's data does not depend on which tests ran before it."""
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
    patterns, _shown, _planes = B.routed(daffine, {"x": x, "a": A}, "daffine")
    assert patterns == {}
    np.testing.assert_array_equal(np.asarray(plan(x, a=A)), 2.0 * x + 1.0)


# 2. one twin per pattern

@pytest.fixture
def deploys(monkeypatch):
    """The ``(kernel, settings)`` of every deploy asked while the test runs; a plan's twins are among them."""
    asked = []
    deploy = plans_module.deploy

    def recording(kernel, **settings):
        asked.append((kernel, settings))
        return deploy(kernel, **settings)

    monkeypatch.setattr(plans_module, "deploy", recording)
    return asked


def test_twin_plan_memoised_per_pattern(cache_dir, rng, deploys):
    plan = hawk.deploy(daffine, targets=("host",), cache_dir=cache_dir)
    twins = [kernel for kernel, _settings in deploys if kernel is not daffine]
    for shape, count in [((A, 1, NC), 1), ((A, 1, NC), 1), ((NC,), 2), (LANE, 2)]:
        plan(rng.random(shape), a=A)
        twins = [kernel for kernel, _settings in deploys if kernel is not daffine]
        assert len(twins) == count          # the repeat and the exact shape build nothing
    assert twins[0] is not twins[1]
    for kernel in twins:
        assert kernel is B.twin(daffine, dict(kernel.broadcast))


# 3. the plan's build options reach the twin

def test_deploy_options_reach_the_twin(tmp_path, rng, deploys):
    cache = str(tmp_path / "opts")
    plan = hawk.deploy(daffine, targets=("host",), cache_dir=cache, scalar_type="float32")
    x = rng.random((A, 1, NC)).astype(np.float32)
    got = np.asarray(plan(x, a=A))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, 2.0 * _full(x) + 1.0, rtol=1e-6)
    (_twin, settings), = deploys
    assert settings["scalar_type"] == "float32" and settings["cache_dir"] == cache and settings["targets"] == ("host",)


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
    x3 = rng.random((A, 1, NC))                  # any shape of the bound pattern is taken, as the caller's own buffer
    bound.rebind(x=x3).launch()
    np.testing.assert_array_equal(out, 2.0 * _full(x3) + 1.0)
    assert bound.planes["x"] is x3


def test_rebind_of_a_number_and_of_a_one_cell_array_keep_their_kinds(plans, rng):
    plan = plans["daffine"]
    out = np.zeros(LANE)
    bound = plan.bind(x=0.5, out=out, a=A)
    bound.launch()
    np.testing.assert_array_equal(out, np.full(LANE, 2.0))
    bound.rebind(x=1.5).launch()
    np.testing.assert_array_equal(out, np.full(LANE, 4.0))
    with pytest.raises(HawkError, match=r"was bound by value.*cannot be rebound varying along \[\]"):
        bound.rebind(x=np.array([2.5]))
    cell = np.array([0.5])
    bound = plan.bind(x=cell, out=out, a=A)
    cell[0] = 2.0                                 # the caller's memory: an update in place is what the next launch reads
    bound.launch()
    np.testing.assert_array_equal(out, np.full(LANE, 5.0))
    with pytest.raises(HawkError, match=r"was bound varying along \[\].*rebound by value"):
        bound.rebind(x=1.5)


# 5. a captured graph swaps the compact buffer


@pytest.mark.skip(reason="needs the graph builder rows: hawk.Graph.bind and hawk.reverse over a graph record")
def test_graph_swaps_a_same_shape_buffer_and_refuses_any_other(plans, rng):
    eagle = pytest.importorskip("eagle")
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


# 6. a plan over a loaded artifact refuses by name

def _loaded_plan(tmp_path_factory, cache_dir):
    import pathlib

    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    bundle = build_bundle([daffine], pathlib.Path(tmp_path_factory.mktemp("bd_art")) / "affine",
                          targets=("host",), cache_dir=cache_dir, host_profile="native")
    return rt.load(bundle.directory, "daffine", sidecar_of(bundle, "daffine"))


R5 = r"daffine: argument 'x' has shape \(3, 1, 6\), which broadcasts against \(a, 4, 6\) along \[a, c\] only.*loaded artifact.*hawk.deploy"


def test_a_plan_over_a_loaded_artifact_refuses_a_broadcast_shape_by_name(tmp_path_factory, cache_dir, rng):
    plan = hawk.plan(_loaded_plan(tmp_path_factory, cache_dir))
    with pytest.raises(HawkError, match=R5):
        plan.bind(x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)
    with pytest.raises(HawkError, match=R5):
        plan.run(x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)
    with pytest.raises(HawkError, match="loaded artifact, which carries no call signature"):
        plan(rng.random(LANE), a=A)


def test_artifact_door_refusal_is_the_plan_refusal(tmp_path_factory, cache_dir, rng):
    from hawk import runtime as rt

    with pytest.raises(HawkError, match=R5):
        rt.run(_loaded_plan(tmp_path_factory, cache_dir), x=rng.random((A, 1, NC)), out=np.zeros(LANE), a=A)


# 7. exact planes stay exact; a bad shape names both shapes

def test_written_plane_stays_exact_through_deploy_run(plans, rng):
    with pytest.raises(HawkError, match=r"'out' has shape \(3, 1, 6\).*exact shape \(a, 4, 6\)"):
        plans["daffine"].run(x=rng.random(LANE), out=np.zeros((A, 1, NC)), a=A)


def test_incompatible_shape_names_both_shapes_through_deploy(plans, rng):
    with pytest.raises(HawkError, match=r"shape \(3, 2, 6\).*against \(a, 4, 6\)"):
        plans["daffine"](rng.random((A, 2, NC)), a=A)


# 8. the reverse of a broadcast input through a recorded graph

def _reverse(plan, name, value, cot, head=()):
    eagle = pytest.importorskip("eagle")
    from eagle import diff as ediff

    uniform = not head and np.size(value) == 1
    bound = plan.bind(**{name: float(np.ravel(value)[0]) if uniform else value},
                      out=np.zeros(LANE), a=A)
    graph = eagle.Graph(record=True).add(bound).build()
    graph.launch().synchronize()
    return np.asarray(ediff.reverse(graph.record, {"out": cot.reshape(-1)})[name])


@pytest.mark.skip(reason="needs the graph builder rows: hawk.Graph.bind and hawk.reverse over a graph record")
@pytest.mark.parametrize("shape", [(A, 1, NC), (NC,), (), (A, NB, 1)])
def test_recorded_reverse_sums_the_broadcast_axes(plans, rng, shape):
    x, g = rng.random(shape) + 0.5, rng.random(LANE) - 0.5
    got = _reverse(plans["dcube"], "x", x, g)
    want = _summed(3.0 * _full(x) ** 2 * g, np.shape(x))
    assert got.shape == np.shape(x) or got.size == want.size
    np.testing.assert_allclose(got.reshape(np.shape(x)), want, rtol=1e-12, atol=1e-13)


@pytest.mark.skip(reason="needs the graph builder rows: hawk.Graph.bind and hawk.reverse over a graph record")
def test_recorded_reverse_of_a_broadcast_vector(plans, rng):
    v, g = rng.random((3, 1, NB, NC)) + 0.5, rng.random(LANE)
    got = _reverse(plans["dmix"], "v", v, g, head=(3,))
    lane = np.broadcast_to(v, (3, *LANE))
    parts = np.stack([lane[1] * g, lane[0] * g, 2.0 * g])
    want = np.stack([_summed(p, v.shape[1:]) for p in parts])
    np.testing.assert_allclose(got.reshape(v.shape), want, rtol=1e-12, atol=1e-13)


@pytest.mark.skip(reason="needs the graph builder rows: hawk.Graph.bind and hawk.reverse over a graph record")
def test_recorded_reverse_is_compensated_and_equals_the_hawk_door(plans, rng, run_cache):
    x = np.full((A, 1, NC), 1e8)
    g = (rng.random(LANE) - 0.5) * 1e-8
    got = _reverse(plans["dcube"], "x", x, g).reshape(A, 1, NC)
    door = Kernel("dcube_vjp", vjp(dcube, wrt=("x",)))(x=x, bar_out=g, a=A)
    np.testing.assert_array_equal(got, np.asarray(door).reshape(A, 1, NC))
    terms = 3.0 * x[:, :, None, :] ** 2 * g[:, None, :, :]         # (A, 1, NB, NC): the products one adjoint cell sums
    exact = np.array([[[math.fsum(terms[i, 0, :, j]) for j in range(NC)]] for i in range(A)])
    # each product is three rounded multiplications and the compensated sum rounds once: four half-ulps of the terms'
    # magnitudes. The terms have both signs, so the bound is on their magnitudes, not on the sum they cancel to.
    bound = 4 * (np.finfo(np.float64).eps / 2) * np.abs(terms).sum(axis=2)
    assert np.all(np.abs(got - exact) <= bound), np.max(np.abs(got - exact) / bound)


def test_a_derivative_plan_routes_a_broadcast_input(rng, run_cache):
    x, g = rng.random((A, 1, NC)) + 0.5, rng.random(LANE) - 0.5
    d = Kernel("dcube_vjp2", vjp(dcube, wrt=("x",)))
    got = np.asarray(hawk.deploy(d)(x=x, bar_out=g, a=A))
    np.testing.assert_allclose(got, _summed(3.0 * _full(x) ** 2 * g, x.shape), rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(got, np.asarray(d(x=x, bar_out=g, a=A)))


def test_the_direct_vjp_call_is_compensated_against_fsum(rng, run_cache):
    """One adjoint cell sums the NB lane products it serves; the compensated sum rounds once."""
    x = np.full((A, 1, NC), 1e8)
    g = (rng.random(LANE) - 0.5) * 1e-8
    got = Kernel("dcube_vjp3", vjp(dcube, wrt=("x",)))(x=x, bar_out=g, a=A)
    assert got.shape == x.shape
    terms = 3.0 * x[:, :, None, :] ** 2 * g[:, None, :, :]         # (A, 1, NB, NC): the products one adjoint cell sums
    exact = np.array([[[math.fsum(terms[i, 0, :, j]) for j in range(NC)]] for i in range(A)])
    # each product is three rounded multiplications and the compensated sum rounds once: four half-ulps of the terms'
    # magnitudes. The terms have both signs, so the bound is on their magnitudes, not on the sum they cancel to.
    bound = 4 * (np.finfo(np.float64).eps / 2) * np.abs(terms).sum(axis=2)
    assert np.all(np.abs(got - exact) <= bound), np.max(np.abs(got - exact) / bound)


@pytest.mark.parametrize("shape", PATTERNS)
def test_a_broadcast_call_equals_the_serial_run_of_the_expanded_input_byte_for_byte(plans, tmp_path_factory, cache_dir,
                                                                                    rng, shape):
    """The team plan of the twin and the serial plan of the dense kernel over ``numpy.broadcast_to`` of the same input."""
    x = rng.random(shape)
    team = np.asarray(plans["daffine"](x, a=A))
    serial = hawk.plan(_loaded_plan(tmp_path_factory, cache_dir), hawk.HostSerial).run(
        x=np.ascontiguousarray(_full(x)), out=np.zeros(LANE), a=A)
    assert team.tobytes() == np.asarray(serial).tobytes()


@pytest.mark.parametrize("scalar_type, dtype", [("float32", np.float32), ("float64", np.float64)])
def test_a_derivative_deployed_in_each_precision_names_the_primal_built_in_that_precision(tmp_path, rng, scalar_type, dtype):
    x, g = (rng.random((A, 1, NC)) + 0.5).astype(dtype), (rng.random(LANE) - 0.5).astype(dtype)
    d = Kernel("dcube_vjp_" + scalar_type, vjp(dcube, wrt=("x",)))
    plan = hawk.deploy(d, scalar_type=scalar_type, cache_dir=str(tmp_path / scalar_type))
    got = np.asarray(plan(x=x, bar_out=g, a=A))
    assert got.dtype == dtype
    np.testing.assert_allclose(got, _summed(3.0 * _full(x) ** 2 * g, x.shape), rtol=1e-5 if dtype == np.float32 else 1e-12,
                               atol=1e-5 if dtype == np.float32 else 1e-13)


# 9. a device plan routes the same way (L1 runs it through the pool)

@pytest.mark.gpu
def test_deploy_run_broadcast_on_the_device(cache_dir, rng):
    cp = pytest.importorskip("cupy")
    plan = hawk.deploy(daffine, cache_dir=cache_dir)
    x = rng.random((A, 1, NC))
    got = plan(cp.asarray(x), a=A)
    np.testing.assert_array_equal(cp.asnumpy(got), 2.0 * _full(x) + 1.0)
    out = cp.zeros(LANE)
    plan.bind(x=cp.asarray(x), out=out, a=A).launch()
    np.testing.assert_array_equal(cp.asnumpy(out), 2.0 * _full(x) + 1.0)
