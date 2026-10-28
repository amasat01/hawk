# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The public tree and its traversal (``hawk.geometry``): orbit line of sight.

* **Build**: flat tables, deterministic, every triangle in exactly one leaf; a bad
  input is refused by name.
* **Traversal**: the nearest hit of 1e4 random triangles equals NumPy brute force on
  the host (f64 exactly, f32 to its rounding), and the depth the caller gives must
  hold the tree: a shorter stack is refused, never clamped.
* **Device** (``gpu`` rows): the three ray ops and the traversal equal the host, and
  ptxas reports no spills for the f32 traversal at stack depth 32.

Correctness only: no speed is claimed."""

from __future__ import annotations

import pathlib

import _eval as E
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of
from test_ir_rays import _device

import hawk
from hawk import HawkError, Index, Mutable, Scalar, Table, Vector
from hawk.artifact import build_bundle
from hawk.geometry import PLANES, build_bvh, closest_hit

RNG = np.random.default_rng(29)
N_TRI, N_RAY, RANGE = 10_000, 512, 40.0


def _scene(n=N_TRI):
    """Small triangles scattered in a box, so a ray meets few of them."""
    centre = RNG.uniform(-6, 6, (n, 1, 3))
    return centre + RNG.normal(size=(n, 3, 3)) * 0.12


def _rays(n=N_RAY):
    o = RNG.uniform(-8, 8, (n, 3))
    d = RNG.normal(size=(n, 3))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


def _brute(tris, o, d, tmax, dtype=np.float64):
    """Nearest hit by testing every triangle: ``(hit, t, input index)``."""
    best = np.full(len(o), tmax, dtype=dtype)
    where = np.full(len(o), -1)
    for k, t in enumerate(tris.astype(dtype)):
        r = E.ray_tri(o, d, *(np.broadcast_to(v, o.shape) for v in t), np.asarray(tmax, dtype), dtype=dtype)
        better = (r[:, 0] > 0) & (r[:, 1] < best)
        best = np.where(better, r[:, 1], best)
        where = np.where(better, k, where)
    return where >= 0, np.where(where >= 0, best, 0.0), where


def _kernel(bvh, depth=32):
    @hawk.kernel
    def nearest(o: Vector[3], d: Vector[3], tmax: Scalar, y: Mutable[Vector[5]],
                lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
                first: Table[Index], size: Table[Index], tri: Table[Scalar]):
        y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), o, d, tmax, depth=depth)

    return nearest


# --- build ---------------------------------------------------------------------

def test_the_tables_cover_every_triangle_once_and_are_deterministic():
    tris = _scene(300)
    a, b = build_bvh(tris, leaf=4), build_bvh(tris, leaf=4)
    assert sorted(a.order.tolist()) == list(range(300))
    for name in PLANES:
        assert np.array_equal(getattr(a, name), getattr(b, name))
    leaves = a.size > 0
    assert a.size[leaves].sum() == 300 and a.size.max() <= 4
    assert np.all(a.left[leaves] == 0) and np.all(a.size[~leaves] == 0)
    assert np.all(a.lo <= a.hi)
    np.testing.assert_array_equal(a.tri.reshape(-1, 3, 3), tris[a.order])
    assert 1 <= a.depth < 300


def test_a_node_bounds_every_triangle_below_it():
    tris = _scene(200)
    bvh = build_bvh(tris, leaf=3)
    stack = [(0, bvh.lo[0], bvh.hi[0])]
    while stack:
        node, lo, hi = stack.pop()
        assert np.all(bvh.lo[node] >= lo - 1e-12) and np.all(bvh.hi[node] <= hi + 1e-12)
        if bvh.size[node]:
            v = bvh.tri[bvh.first[node]:bvh.first[node] + bvh.size[node]].reshape(-1, 3)
            assert np.all(v >= bvh.lo[node]) and np.all(v <= bvh.hi[node])
        else:
            stack += [(bvh.left[node], bvh.lo[node], bvh.hi[node]),
                      (bvh.right[node], bvh.lo[node], bvh.hi[node])]


def test_a_bad_scene_is_refused_by_name():
    with pytest.raises(HawkError, match=r"build_bvh: an \(n, 3, 3\) array of triangles"):
        build_bvh(np.zeros((4, 3)))
    with pytest.raises(HawkError, match="leaf must be at least 1"):
        build_bvh(_scene(4), leaf=0)


# --- depth ---------------------------------------------------------------------

def test_a_stack_shorter_than_the_tree_is_refused_never_clamped():
    bvh = build_bvh(_scene(500), leaf=2)
    assert bvh.depth > 4
    with pytest.raises(HawkError, match=rf"the tree is {bvh.depth} levels deep, past depth=4"):
        _kernel(bvh, depth=4)
    assert _kernel(bvh, depth=bvh.depth) is not None


# --- host ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory, cache_dir):
    tris, (o, d) = _scene(), _rays()
    bvh = build_bvh(tris, leaf=4)
    assert bvh.depth <= 32
    root = tmp_path_factory.mktemp("hawk_bvh")
    units = {}
    for mode, dtype in (("float64", np.float64), ("float32", np.float32)):
        k = _kernel(bvh)
        b = build_bundle([k], pathlib.Path(root) / mode, targets=("host",), cache_dir=cache_dir,
                         host_profile="native", mode=mode)
        units[dtype] = (b, k)
    return tris, o, d, bvh, units


def _host(world, dtype):
    tris, o, d, bvh, units = world
    bundle, _k = units[dtype]
    return O.run(bundle.directory, "nearest", len(o), sidecar_of(bundle, "nearest"),
                 o=o.T.copy(), d=d.T.copy(), tmax=np.full(len(o), RANGE), **bvh.planes(dtype))


def test_host_nearest_hit_equals_numpy_brute_force_in_f64(world):
    tris, o, d, bvh, _ = world
    got = _host(world, np.float64)
    hit, t, where = _brute(tris, o, d, RANGE)
    assert 20 < hit.sum() < len(o) - 20, "the rays both hit and miss"
    np.testing.assert_array_equal(got[0] > 0, hit)
    np.testing.assert_allclose(got[1], t, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(bvh.order[got[4][hit].astype(int)], where[hit])


def test_host_nearest_hit_equals_numpy_brute_force_in_f32(world):
    tris, o, d, _bvh, _ = world
    got = _host(world, np.float32)
    hit, t, _where = _brute(tris, o, d, RANGE, np.float32)
    np.testing.assert_array_equal(got[0] > 0, hit)
    np.testing.assert_allclose(got[1], t, rtol=2e-5, atol=1e-5)


# --- device ------------------------------------------------------------------------

@pytest.mark.gpu
def test_device_traversal_equals_host_and_brute_force(world, cache_dir, tmp_path):
    tris, o, d, bvh, _ = world
    n = len(o)
    arrays = dict(o=o.T.copy(), d=d.T.copy(), tmax=np.full(n, RANGE), y=np.zeros((5, n)), **bvh.planes())
    _bundle, got = _device(_kernel(bvh), "nearest", arrays, n, cache_dir, tmp_path / "dev")
    hit, t, where = _brute(tris, o, d, RANGE)
    host = _host(world, np.float64)
    np.testing.assert_array_equal(got["y"][0] > 0, hit)
    np.testing.assert_allclose(got["y"][1], t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["y"], host, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_the_f32_traversal_at_depth_32_has_no_spills(world, tmp_path):
    """ptxas on the device unit at the real arch: the stack lives in a local frame by
    design, and nothing is spilled."""
    import subprocess

    from hawk.artifact.layout import exports
    from hawk.compile import device_compiler
    from hawk.compile.drivers import parse_ptxas_info
    from hawk.compile.toolchain import device_flags, resolve_arch
    from hawk.emit import FLOAT32, render_source
    from hawk.targets import TARGETS

    _t, _o, _d, bvh, _ = world
    k = _kernel(bvh, depth=32)
    src = tmp_path / "nearest.cu"
    src.write_text(render_source(k.name, k.sinks, k.walk, TARGETS["cuda"], mode=FLOAT32,
                                 exports=exports("cuda")).text)
    done = subprocess.run([device_compiler(), *device_flags(arch=resolve_arch(), target="cubin"),
                           str(src), "-o", str(tmp_path / "nearest.cubin")],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    figures = parse_ptxas_info(done.stderr)["nearest"]
    assert figures["spill_store_B"] == figures["spill_load_B"] == 0, figures
