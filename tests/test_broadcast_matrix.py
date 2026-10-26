# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A per-lane ``Matrix`` input of a ``Grid`` kernel may be given broadcast.

The variant of its pattern reads the caller's compact ``(rows*cols, m)`` plane with
``At`` at the flat index over the axes the matrix varies along; eagle binds that
plane by its own column count (sidecar ``gather_mats``), not the lane count.

* **Values** against ``np.broadcast_to`` for every pattern, a rolled ``M @ v``
  (R = 8, 20), and the run-time first axis (two sizes, one build).
* **Refusals** name both shapes; a compact plane of the wrong row count is refused at bind.
* **jvp** equals the dense jvp of the broadcast tangent.
* **vjp** wrt the matrix is the sum over the axes it does not vary along, equal to the
  per-entry ``fsum`` of the dense adjoints and to central differences, in the input's
  own shape and layout; no lane writes outside its kept index.
* **Recorded reverse** over a variant plan with a broadcast matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import eagle
import hawk
import hawk.math as hm
from eagle import diff as ediff
from hawk import Grid, HawkError, Matrix, Scalar, Vector
from hawk import _broadcast as B
from hawk.diff import jvp, vjp
from hawk.trace.kernel import Kernel

A, NB, NC = 3, 4, 6
LANE = (A, NB, NC)
R = 2


@hawk.kernel
def mscal(lane: Grid["a", "b":4, "c":6], M: Matrix[2, 2], x: Scalar) -> Scalar:
    return M[0, 0] * M[1, 1] * x + 10.0 * M[0, 1] + 100.0 * M[1, 0] + 1000.0 * M[1, 1] * x


def _mv(rows):
    @hawk.kernel
    def mv(lane: Grid["a", "b":4, "c":6], M: Matrix[rows, rows], v: Vector[rows]) -> Scalar:
        return hm.sum(M @ v)
    return mv


mv8, mv20 = _mv(8), _mv(20)


@hawk.kernel
def hand(lane: Grid["a", "b":4, "c":6], M: Matrix[2, 2]) -> Scalar:
    m = M.at(lane.a * 6 + lane.c)           # a hand-written read at another lane
    return m[0, 0] + 2.0 * m[1, 1]


@pytest.fixture
def run_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(41)


#: [a, c], [c], [a], none, [b, c]
SHAPES = [(R, R, A, 1, NC), (R, R, NC), (R, R, A, 1, 1), (R, R), (R, R, 1, NB, NC)]


def _full(m, rows=R, a=A):
    """``m`` (``(rows, rows, *lane)``) right-aligned against ``(a, NB, NC)``."""
    lane = m.shape[2:]
    return np.broadcast_to(m.reshape(rows, rows, *(1,) * (3 - len(lane)), *lane), (rows, rows, a, NB, NC))


def _scal(m, x):
    f = _full(m)
    return f[0, 0] * f[1, 1] * x + 10.0 * f[0, 1] + 100.0 * f[1, 0] + 1000.0 * f[1, 1] * x


def _adjoint(m, x, g):
    f = _full(m)
    d = np.zeros(f.shape)
    d[0, 0] = f[1, 1] * x * g
    d[1, 1] = (f[0, 0] * x + 1000.0 * x) * g
    d[0, 1], d[1, 0] = 10.0 * g, 100.0 * g
    return d


def _fsum(d, shape):
    """``d`` (``(R, R, *LANE)``) summed per kept index with ``math.fsum``, in ``shape``."""
    lane = shape[2:]
    padded = (1,) * (3 - len(lane)) + tuple(lane)
    out = np.zeros((d.shape[0], d.shape[1], *padded))
    for idx in np.ndindex(*padded):
        sel = tuple(slice(None) if p == 1 else i for p, i in zip(padded, idx))
        for r in range(d.shape[0]):
            for c in range(d.shape[1]):
                out[(r, c, *idx)] = math.fsum(d[r, c][sel].reshape(-1))
    return out.reshape(shape)


# 1. values

@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_broadcast_to_through_every_door(run_cache, cache_dir, rng, shape):
    m, x = rng.standard_normal(shape), rng.random(LANE)
    want = _scal(m, x)
    np.testing.assert_allclose(np.asarray(mscal(m, x, a=A)), want, rtol=1e-13)
    plan = eagle.deploy(mscal, targets=("host",), cache_dir=cache_dir)
    np.testing.assert_allclose(np.asarray(plan.run(M=m, x=x, out=np.zeros(LANE), a=A)), want, rtol=1e-13)
    out = np.zeros(LANE)
    plan.bind(M=m, x=x, out=out, a=A).launch()
    np.testing.assert_allclose(out, want, rtol=1e-13)


@pytest.mark.parametrize("kernel,rows", [(mv8, 8), (mv20, 20)])
def test_rolled_matvec_reads_the_compact_plane(run_cache, rng, kernel, rows):
    m, v = rng.standard_normal((rows, rows, A, 1, NC)), rng.standard_normal((rows, *LANE))
    want = np.einsum("rcabk,cabk->abk", _full(m, rows), v)
    np.testing.assert_allclose(np.asarray(kernel(m, v, a=A)), want, rtol=1e-12, atol=1e-12)


def test_runtime_first_axis_two_sizes_one_build(run_cache, rng):
    before = len(mscal._variants)
    got = {}
    for a in (3, 5):
        m, x = rng.standard_normal((R, R, a, 1, NC)), rng.random((a, NB, NC))
        got[a] = len(mscal._variants)
        np.testing.assert_allclose(np.asarray(mscal(m, x, a=a)), _scal_a(m, x, a), rtol=1e-13)
    assert got[3] == got[5] and got[3] - before <= 1


def _scal_a(m, x, a):
    f = _full(m, R, a)
    return f[0, 0] * f[1, 1] * x + 10.0 * f[0, 1] + 100.0 * f[1, 0] + 1000.0 * f[1, 1] * x


# 2. refusals

def test_a_shape_that_does_not_broadcast_names_both_shapes(run_cache, rng):
    with pytest.raises(HawkError, match=r"shape \(2, 2, 3, 2, 6\).*does not broadcast against \(2, 2, a, 4, 6\)"):
        mscal(rng.standard_normal((R, R, A, 2, NC)), rng.random(LANE), a=A)


def test_a_compact_plane_of_the_wrong_row_count_is_refused_at_bind(run_cache, cache_dir, rng):
    kept = B.kept_axes((R, R, A, 1, NC), (R, R), mscal.lane_space, A)
    var = B.variant(mscal, {"M": kept})
    plan = eagle.deploy(var, targets=("host",), cache_dir=cache_dir)
    with pytest.raises(ValueError, match=r"binds compact as \(4, 18\) at n=72; got shape \(3, 18\)"):
        plan.bind(M=np.zeros((3, A * NC)), x=rng.random(LANE), out=np.zeros(LANE), a=A)
    for columns in (A * NC - 1, A * NC + 1, A * NB * NC):     # short, long, the lane count
        with pytest.raises(ValueError, match=rf"binds compact as \(4, 18\) at n=72; got shape \(4, {columns}\)"):
            plan.bind(M=np.zeros((R * R, columns)), x=rng.random(LANE), out=np.zeros(LANE), a=A)
    plan.bind(M=np.zeros((R * R, A * NC)), x=rng.random(LANE), out=np.zeros(LANE), a=A)   # the right count binds


def test_a_loaded_compact_matrix_artifact_binds_and_checks_its_columns(tmp_path_factory, cache_dir, rng):
    import pathlib

    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    shape = SHAPES[0]
    kept = B.kept_axes(shape, (R, R), mscal.lane_space, A)
    var = B.variant(mscal, {"M": kept})
    bundle = build_bundle([var], pathlib.Path(tmp_path_factory.mktemp("bm_art")) / "k",
                          targets=("host",), cache_dir=cache_dir, host_profile="native")
    sidecar = sidecar_of(bundle, "mscal")
    assert sidecar["gather_mats"] == {"M": [True, False, True]}
    art = rt.load(bundle.directory, "mscal", sidecar)
    m, x = rng.standard_normal(shape), rng.random(LANE)
    out = np.zeros(A * NB * NC)
    rt.run(art, M=m.reshape(R * R, -1), x=x.reshape(-1), out=out, a=A)
    np.testing.assert_allclose(out.reshape(LANE), _scal(m, x), rtol=1e-13)
    for columns in (A * NC - 1, A * NC + 1):
        with pytest.raises(HawkError, match=rf"bound compact as \(4, 18\) at n_samples=72, got shape \(4, {columns}\)"):
            rt.run(art, M=np.zeros((R * R, columns)), x=x.reshape(-1), out=out, a=A)


def test_a_hand_written_read_at_another_lane_has_no_matrix_vjp(run_cache):
    d = Kernel("hand_vjp", vjp(hand, wrt=("M",)))
    padded = np.zeros((R, R, A * NB * NC))
    with pytest.raises(HawkError, match=r"scattered accumulate of a rank-2 value is not built"):
        d(M=padded.reshape(R, R, A, NB, NC), bar_out=np.zeros(LANE), a=A)


# 3. jvp

@pytest.mark.parametrize("shape", SHAPES)
def test_jvp_equals_the_dense_jvp_of_the_broadcast_tangent(run_cache, rng, shape):
    m, t, x = rng.standard_normal(shape), rng.standard_normal(shape), rng.random(LANE)
    got = np.asarray(Kernel("mscal_jvp", jvp(mscal, wrt=("M",)))(M=m, dot_M=t, x=x, a=A))
    f, d = _full(m), _full(t)
    want = (d[0, 0] * f[1, 1] + f[0, 0] * d[1, 1]) * x + 10.0 * d[0, 1] + 100.0 * d[1, 0] + 1000.0 * d[1, 1] * x
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


# 4. vjp

@pytest.mark.parametrize("shape", SHAPES)
def test_vjp_is_the_fsum_of_the_dense_adjoints_in_the_inputs_shape(run_cache, rng, shape):
    m, x, g = rng.standard_normal(shape), rng.random(LANE), rng.standard_normal(LANE)
    got = np.asarray(Kernel("mscal_vjp", vjp(mscal, wrt=("M",)))(M=m, x=x, bar_out=g, a=A))
    assert got.shape == shape
    want = _fsum(_adjoint(m, x, g), shape)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_vjp_equals_central_differences(run_cache, rng):
    shape = SHAPES[0]
    m, x, g = rng.standard_normal(shape), rng.random(LANE), rng.standard_normal(LANE)
    got = np.asarray(Kernel("mscal_vjp_fd", vjp(mscal, wrt=("M",)))(M=m, x=x, bar_out=g, a=A))
    h = 1e-4                                # the output is quadratic in M: no truncation, only rounding
    for idx in np.ndindex(*shape):
        up, down = m.copy(), m.copy()
        up[idx] += h
        down[idx] -= h
        fd = (np.sum(g * _scal(up, x)) - np.sum(g * _scal(down, x))) / (2 * h)
        assert abs(got[idx] - fd) <= 1e-7 * max(1.0, abs(fd)), idx


@pytest.mark.parametrize("kernel,rows", [(mv8, 8), (mv20, 20)])
def test_vjp_through_a_rolled_matvec(run_cache, rng, kernel, rows):
    m, v, g = rng.standard_normal((rows, rows, A, 1, NC)), rng.standard_normal((rows, *LANE)), rng.standard_normal(LANE)
    got = np.asarray(Kernel(kernel.name + "_vjp", vjp(kernel, wrt=("M",)))(M=m, v=v, bar_out=g, a=A))
    want = np.broadcast_to(np.einsum("abk,cabk->cabk", g, v)[None], (rows, rows, *LANE)).sum(axis=3, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_no_lane_writes_outside_its_kept_index(run_cache, rng):
    """One lane carries the cotangent: only its compact column holds anything."""
    shape = SHAPES[0]
    m, x = rng.standard_normal(shape), rng.random(LANE)
    g = np.zeros(LANE)
    g[1, 2, 4] = 1.0
    got = np.asarray(Kernel("mscal_vjp_one", vjp(mscal, wrt=("M",)))(M=m, x=x, bar_out=g, a=A))
    column = got.reshape(R * R, A * NC)
    own = 1 * NC + 4                          # a = 1, c = 4
    assert np.count_nonzero(np.delete(column, own, axis=1)) == 0
    np.testing.assert_allclose(column[:, own], _adjoint(m, x, g)[:, :, 1, 2, 4].reshape(-1), rtol=1e-14)


# 5. recorded reverse over a variant plan

def test_recorded_reverse_with_a_broadcast_matrix(run_cache, cache_dir, rng):
    shape = SHAPES[0]
    m, x, g = rng.standard_normal(shape), rng.random(LANE), rng.standard_normal(LANE)
    kept = B.kept_axes(shape, (R, R), mscal.lane_space, A)
    plan = eagle.deploy(B.variant(mscal, {"M": kept}), targets=("host",), cache_dir=cache_dir)
    bound = plan.bind(M=B.compact("M", m, kept, (R, R), "test"), x=x, out=np.zeros(LANE), a=A)
    graph = eagle.Graph(record=True).add(bound).build()
    graph.launch().synchronize()
    got = np.asarray(ediff.reverse(graph.record, {"out": g.reshape(-1)})["M"])
    np.testing.assert_allclose(got.reshape(shape), _fsum(_adjoint(m, x, g), shape), rtol=1e-12, atol=1e-13)


# 6. the device

@pytest.mark.gpu
@pytest.mark.parametrize("kernel,rows", [(mv8, 8), (mv20, 20)])
def test_device_forward_and_vjp(cache_dir, rng, kernel, rows):
    cp = pytest.importorskip("cupy")
    m, v, g = rng.standard_normal((rows, rows, A, 1, NC)), rng.standard_normal((rows, *LANE)), rng.standard_normal(LANE)
    plan = eagle.deploy(kernel, cache_dir=cache_dir)
    out = cp.asnumpy(plan(cp.asarray(m), cp.asarray(v), a=A))
    np.testing.assert_allclose(out, np.einsum("rcabk,cabk->abk", _full(m, rows), v), rtol=1e-12, atol=1e-12)
    got = cp.asnumpy(Kernel(kernel.name + "_vjp_dev", vjp(kernel, wrt=("M",)))(
        M=cp.asarray(m), v=cp.asarray(v), bar_out=cp.asarray(g), a=A))
    want = np.broadcast_to(np.einsum("abk,cabk->cabk", g, v)[None], (rows, rows, *LANE)).sum(axis=3, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_device_scalar_kernel_vjp_through_the_deploy_door(cache_dir, rng):
    cp = pytest.importorskip("cupy")
    m, x, g = rng.standard_normal(SHAPES[0]), rng.random(LANE), rng.standard_normal(LANE)
    got = cp.asnumpy(Kernel("mscal_vjp_dev", vjp(mscal, wrt=("M",)))(
        M=cp.asarray(m), x=cp.asarray(x), bar_out=cp.asarray(g), a=A))
    np.testing.assert_allclose(got, _fsum(_adjoint(m, x, g), SHAPES[0]), rtol=1e-12, atol=1e-13)
