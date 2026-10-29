# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ray ops (``hawk.math.ray_tri`` / ``ray_aabb`` / ``ray_sphere``): typing,
emitted text and values.

Each op is ONE node typed ``(4,)``, ``[hit, a, b, c]``, over ``aether::geometry``. The
rows below check the type rule, the emitted call, the host values in both
precisions against NumPy (a separate implementation of the same formulas,
``_eval.ray_*``), the finite results on degenerate input, and the f32 accuracy at
orbit scale."""

from __future__ import annotations

import pathlib

import _eval as E
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import HawkError, Mutable, Scalar, Vector
from hawk.artifact import build_bundle
from hawk.emit import render_body
from hawk.ir import canonical
from hawk.types import TensorType

V3 = Vector[3]


@hawk.kernel
def tri(o: V3, d: V3, a: V3, b: V3, c: V3, tmax: Scalar, y: Mutable[Vector[4]]):
    y = m.ray_tri(o, d, a, b, c, tmax)


@hawk.kernel
def box(o: V3, d: V3, lo: V3, hi: V3, tmax: Scalar, y: Mutable[Vector[4]]):
    y = m.ray_aabb(o, d, lo, hi, tmax)


@hawk.kernel
def ball(o: V3, d: V3, c: V3, r: Scalar, tmax: Scalar, y: Mutable[Vector[4]]):
    y = m.ray_sphere(o, d, c, r, tmax)


KERNELS = {"tri": tri, "box": box, "ball": ball}
REF = {"tri": E.ray_tri, "box": E.ray_aabb, "ball": E.ray_sphere}
N = 64


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("hawk_rays")
    out = {}
    for mode, dtype in (("float64", np.float64), ("float32", np.float32)):
        for name, k in KERNELS.items():
            bundle = build_bundle([k], pathlib.Path(root) / f"{name}_{mode}", targets=("host",),
                                  cache_dir=cache_dir, host_profile="native", mode=mode)
            out[name, dtype] = (bundle, k)
    return out


def _scene(rng, kind: str):
    """Rays from a shell around the origin towards points near it, ``(3, N)`` planes."""
    o = rng.normal(size=(N, 3)) * 3.0
    d = rng.normal(size=(N, 3)) * 0.5 - o * 0.3
    if kind == "tri":
        planes = [o, d, *(rng.normal(size=(N, 3)) for _ in range(3))]
    elif kind == "box":
        lo = rng.normal(size=(N, 3)) - 1.5
        planes = [o, d, lo, lo + rng.uniform(0.5, 3.0, (N, 3))]
    else:
        planes = [o, d, rng.normal(size=(N, 3)) * 0.5]
    return planes


VECTORS = {"tri": ("o", "d", "a", "b", "c"), "box": ("o", "d", "lo", "hi"), "ball": ("o", "d", "c")}


def _run(loaded, kind: str, dtype, planes, scalars):
    """One host run: ``planes`` are ``(n, 3)`` vectors in call order, ``scalars`` the rank-0
    inputs (``r`` for a sphere, then ``tmax``)."""
    bundle, _k = loaded[kind, dtype]
    given = {nm: p.T.copy() for nm, p in zip(VECTORS[kind], planes)}
    given.update(zip(("r", "tmax") if kind == "ball" else ("tmax",), scalars))
    return O.run(bundle.directory, kind, len(planes[0]), sidecar_of(bundle, kind), **given)


def test_each_op_is_one_node_typed_four():
    for k in KERNELS.values():
        (sink,) = k.sinks
        node = sink.value
        assert node.kind.startswith("ray_") and node.ttype == TensorType((4,), "f64")
    text = render_body(tri.sinks, canonical(tri.sinks)).text
    assert text.count("aether::geometry::rayTriangle<Real>(") == 1
    assert "aether::geometry::rayAabb<Real>(" in render_body(box.sinks, canonical(box.sinks)).text
    assert "aether::geometry::raySphere<Real>(" in render_body(ball.sinks, canonical(ball.sinks)).text


def test_the_type_rule_refuses_the_wrong_shapes():
    with pytest.raises(HawkError, match=r"ray_tri: a length-3 vector is wanted"):
        @hawk.kernel
        def bad(a: Vector[2], y: Mutable[Vector[4]]):
            y = m.ray_tri(a, a, a, a, a)

    with pytest.raises(HawkError, match=r"operand 3 is a rank-0"):
        @hawk.kernel
        def bad2(a: V3, y: Mutable[Vector[4]]):
            y = m.ray_sphere(a, a, a, a)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("kind", ["tri", "box", "ball"])
def test_host_values_equal_numpy(loaded, kind, dtype):
    rng = np.random.default_rng(11)
    planes = _scene(rng, kind)
    tmax = np.full(N, 6.0)
    scalars = ([rng.uniform(0.5, 2.0, N)] if kind == "ball" else []) + [tmax]
    got = _run(loaded, kind, dtype, planes, scalars)
    args = [p.astype(dtype) for p in planes] + ([scalars[0].astype(dtype)] if kind == "ball" else [])
    want = REF[kind](*args, np.asarray(6.0, dtype), dtype=dtype)
    tol = 1e-5 if dtype is np.float32 else 1e-12
    np.testing.assert_allclose(got.T, want, rtol=tol, atol=tol * 10)
    assert 0 < got[0].sum() < N, "the scene has hits and misses"


DEGENERATE = {
    "tri": [
        ([0, 0, -1], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]),       # parallel
        ([0, 0, -1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [2, 0, 0]),       # zero area
        ([0, 0, -1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]),       # zero direction
        ([0.5, 0.0, -1], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]),   # exactly on an edge
        ([0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]),        # origin on a vertex
    ],
    "box": [
        ([0.5, 0.5, -1], [0, 0, 1], [0, 0, 0], [1, 1, 1]),              # d_x = d_y = 0, inside slabs
        ([2.5, 0.5, -1], [0, 0, 1], [0, 0, 0], [1, 1, 1]),              # d_x = 0, outside
        ([0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0], [1, 1, 1]),             # zero direction, inside
        ([0.5, 0.5, -1], [-0.0, 0, 1], [0, 0, 0], [1, 1, 1]),           # negative zero
        ([0.5, 0.5, -1], [0, 0, 1], [0, 0, 0], [1, 1, 1]),              # a plain hit
    ],
    "ball": [
        ([0, 0, 5], [0, 0, 0], [0, 0, 0], 1.0),                         # zero direction
        ([0, 2, 0], [1, 0, 0], [0, 0, 0], 1.0),                         # misses
        ([0, 1, 0], [1, 0, 0], [0, 0, 0], 1.0),                         # tangent
        ([0, 0, 0], [1, 0, 0], [0, 0, 0], 1.0),                         # origin inside
        ([0, 0, -3], [0, 0, 1], [0, 0, 0], 1.0),                        # a plain hit
    ],
}


@pytest.mark.parametrize("kind", ["tri", "box", "ball"])
def test_degenerate_input_gives_finite_values_equal_to_numpy(loaded, kind):
    rows = DEGENERATE[kind]
    n, nv = len(rows), len(VECTORS[kind])
    planes = [np.array([r[k] for r in rows], dtype=float) for k in range(nv)]
    scalars = ([np.array([r[3] for r in rows], dtype=float)] if kind == "ball" else []) + [np.full(n, 10.0)]
    got = _run(loaded, kind, np.float64, planes, scalars)
    assert np.all(np.isfinite(got))
    want = REF[kind](*planes, *scalars[:-1], np.asarray(10.0))
    np.testing.assert_allclose(got.T, want, rtol=1e-12, atol=1e-12)


# --- f32 at orbit scale (D11) --------------------------------------------------

ORBIT = 7.0e6


def _orbit_scene(rng, n=20000):
    """Rays from ``7e6 m`` at an Earth-radius triangle, offsets well off the edges."""
    radius = 6.4e6
    tri_pts = np.array([[radius, -radius, -radius], [radius, radius, -radius], [radius, 0.0, radius]])
    target = tri_pts.mean(axis=0) + rng.uniform(-0.3, 0.3, (n, 3)) * radius * np.array([0, 1, 1])
    o = np.tile([ORBIT * 1.5, 0.0, 0.0], (n, 1)) + rng.normal(size=(n, 3)) * 1e3
    d = target - o
    return o, d / np.linalg.norm(d, axis=1, keepdims=True), tri_pts


def test_f32_at_orbit_scale_meets_the_thresholds(loaded):
    rng = np.random.default_rng(3)
    o, d, pts = _orbit_scene(rng)
    n = len(o)
    far = np.full(n, np.inf)
    got = {}
    for dtype in (np.float64, np.float32):
        planes = [o, d, *(np.broadcast_to(p, o.shape) for p in pts)]
        got[dtype] = _run(loaded, "tri", dtype, planes, [far]).T.astype(np.float64)
    w64, w32 = got[np.float64], got[np.float32]
    live = w64[:, 0] > 0
    assert live.sum() > 1000
    rel = np.abs(w32[live, 1] - w64[live, 1]) / w64[live, 1]
    assert rel.max() < 1e-6
    # rays whose barycentric position is not within 1e-6 of an edge keep their verdict
    edge = np.minimum.reduce([w64[:, 2], w64[:, 3], 1 - w64[:, 2] - w64[:, 3]])
    firm = np.abs(edge) > 1e-6
    assert np.array_equal(w32[firm, 0], w64[firm, 0])
    assert np.all(np.isfinite(w32))


def test_a_one_metre_facet_at_orbit_scale_is_better_in_scene_local_coordinates(loaded):
    """The same facet and rays in raw coordinates (``7e6 m`` from the origin) and shifted
    to the facet: f32 keeps ``7e6`` to half a metre, so the raw verdicts and weights are off
    by a large part of the facet; the local ones are not."""
    rng = np.random.default_rng(5)
    n = 4000
    centre = np.array([4.0e6, 4.0e6, 3.7e6])                  # |centre| = 6.9e6 m
    facet = np.array([[0.0, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.0, 0.5]])
    o_local = np.column_stack([np.full(n, -100.0), rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n)])
    d = np.tile([1.0, 0.0, 0.0], (n, 1))
    truth = E.ray_tri(o_local, d, *(np.broadcast_to(f, o_local.shape) for f in facet), np.asarray(np.inf))
    far = np.full(n, np.inf)

    def run(shift):
        planes = [o_local + shift, d, *(np.broadcast_to(f + shift, o_local.shape) for f in facet)]
        return _run(loaded, "tri", np.float32, planes, [far]).T.astype(np.float64)

    local, raw = run(0.0), run(centre)
    flips = lambda w: int(np.sum(w[:, 0] != truth[:, 0]))
    assert flips(local) <= flips(raw)
    err = lambda w: np.abs(w[:, 2:] - truth[:, 2:]).max()      # the barycentric weights
    assert err(local) < 1e-5 < err(raw)


def _device(kernel, name, arrays, n, cache_dir, tmp, mode="float64"):
    import cupy as cp
    import eagle.exec as eexec
    from _deploy import device_plugin
    from eagle import plan as eplan
    from hawk.compile.toolchain import resolve_arch

    bundle = build_bundle([kernel], tmp, targets=("cuda",), cache_dir=cache_dir,
                          device_arch=resolve_arch(), mode=mode)
    plugin = device_plugin(bundle.directory, name, sidecar_of(bundle, name))
    planes = {k: cp.asarray(v) for k, v in arrays.items()}
    eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=1).run(**planes)
    return bundle, {k: cp.asnumpy(v) for k, v in planes.items()}


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["tri", "box", "ball"])
def test_device_values_equal_the_host_in_both_precisions(loaded, kind, cache_dir, tmp_path):
    for mode, dtype in (("float64", np.float64), ("float32", np.float32)):
        rng = np.random.default_rng(31)
        planes = _scene(rng, kind)
        scalars = ([rng.uniform(0.5, 2.0, N)] if kind == "ball" else []) + [np.full(N, 6.0)]
        host = _run(loaded, kind, dtype, planes, scalars)
        given = {nm: p.T.astype(dtype).copy() for nm, p in zip(VECTORS[kind], planes)}
        given.update({nm: s.astype(dtype) for nm, s in zip(("r", "tmax") if kind == "ball" else ("tmax",), scalars)})
        given["y"] = np.zeros((4, N), dtype)
        _bundle, got = _device(KERNELS[kind], kind, given, N, cache_dir, tmp_path / f"{kind}_{mode}", mode)
        tol = 1e-5 if dtype is np.float32 else 1e-12
        np.testing.assert_array_equal(got["y"][0], host[0])
        np.testing.assert_allclose(got["y"], host, rtol=tol, atol=tol * 10)
