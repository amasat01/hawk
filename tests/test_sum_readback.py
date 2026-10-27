# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.sum/mean(x, over=...)`` read back in the same kernel.

``x / hawk.sum(x, over="col")`` needs every lane of a row before any lane of it,
so the kernel runs in stages like ``x / hawk.sum(x)``: the first accumulates
into a keyed plane the runtime owns and zeroes for every run, the next reads it
at its own lane's key.

* **forward** — matches numpy, f64 and f32; a mean; a vector payload read per
  component; ``auto`` and ``compensated`` within the per-key ``math.fsum`` contract;
* **derivative** — the vjp has three stages and, like the jvp, matches central
  differences; the derivative of a sum keeps its key, count and policy;
* **runs** — two runs are identical (the plane is zeroed each run);
* **refused by name** — ``ordered`` and a keyed max/min read back, a stepping
  kernel, ``group=``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hawk
import hawk.diff
import hawk.math as m
from hawk import Grid, HawkError, Index, Kernel, Mutable, Scalar, Terminated, Vector, kernel, runtime
from hawk.artifact import build_bundle
from hawk.ir.nodes import ReduceSamples
from hawk.ir.split import _post_order, stages

R, COL = 4, 64
N = R * COL


@kernel
def share(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
    return x / hawk.sum(x, over="col")


@kernel
def share_c(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
    return x / hawk.sum(x, over="col", policy="compensated")


@kernel
def centred(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
    return x - hawk.mean(x, over="col")


@kernel
def pair(lane: Grid["row", "col":COL], x: Scalar) -> Vector[2]:
    v = m.vec(x, x * x)
    return v / hawk.sum(v, over="col")


@kernel
def whole(x: Scalar) -> Scalar:
    return x / hawk.sum(x, policy="ordered")


def _bundle(kernels, tmp, mode, cache_dir):
    return build_bundle(kernels, tmp, mode=mode, targets=("host",), cache_dir=cache_dir,
                        host_profile="native")


def _x(mode, lo=0.5):
    return np.random.default_rng(3).uniform(lo, 2.0, N).astype(mode)


def _run(bundle, name, **planes):
    runtime.run(runtime.load(bundle.directory, name), row=R, **planes)


def _tol(mode):
    return 1e-12 if mode == "float64" else 2e-5


# -- forward ----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["float64", "float32"])
@pytest.mark.parametrize("which", [share, share_c])
def test_x_over_its_row_sum_matches_numpy(tmp_path, cache_dir, mode, which):
    bundle = _bundle([which], tmp_path, mode, cache_dir)
    assert [a.name for a in bundle.artifacts] == [which.name, f"{which.name}_2"]
    assert list(bundle.artifacts[0].sidecar["keyed"]) == ["cut0"]
    x = _x(mode)
    z = np.zeros(N, dtype=mode)
    _run(bundle, which.name, x=x, out=z)
    rows = x.astype(np.float64).reshape(R, COL)
    want = (rows / rows.sum(axis=1, keepdims=True)).ravel()
    np.testing.assert_allclose(z, want, rtol=_tol(mode))


@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_x_minus_its_row_mean_matches_numpy(tmp_path, cache_dir, mode):
    bundle = _bundle([centred], tmp_path, mode, cache_dir)
    x = _x(mode)
    z = np.zeros(N, dtype=mode)
    _run(bundle, "centred", x=x, out=z)
    rows = x.astype(np.float64).reshape(R, COL)
    np.testing.assert_allclose(z, (rows - rows.mean(axis=1, keepdims=True)).ravel(),
                               atol=_tol(mode))


@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_a_vector_payload_is_read_back_per_component(tmp_path, cache_dir, mode):
    bundle = _bundle([pair], tmp_path, mode, cache_dir)
    assert bundle.artifacts[0].sidecar["keyed"]["cut0"]["width"] == 2
    x = _x(mode)
    z = np.zeros((2, N), dtype=mode)
    _run(bundle, "pair", x=x, out=z)
    rows = x.astype(np.float64).reshape(R, COL)
    for k, v in enumerate((rows, rows * rows)):
        np.testing.assert_allclose(z[k], (v / v.sum(axis=1, keepdims=True)).ravel(),
                                   rtol=_tol(mode))


@pytest.mark.parametrize("mode", ["float64", "float32"])
@pytest.mark.parametrize("which", [share, share_c])
def test_the_row_sum_is_within_the_fsum_contract(tmp_path, cache_dir, mode, which):
    """``z = x / s``: the sum behind it is within eps * sum|x| of ``math.fsum``."""
    bundle = _bundle([which], tmp_path, mode, cache_dir)
    x = (np.random.default_rng(7).uniform(-1, 1, N) * np.exp(
        np.random.default_rng(8).uniform(0, 8, N))).astype(mode)
    z = np.zeros(N, dtype=mode)
    _run(bundle, which.name, x=x, out=z)
    rows = x.astype(np.float64).reshape(R, COL)
    for r in range(R):
        total = math.fsum(rows[r])
        got = x[r * COL] / z[r * COL]
        assert abs(float(got) - total) <= 4 * np.finfo(mode).eps * np.abs(rows[r]).sum()


# -- derivative -------------------------------------------------------------

def test_the_vjp_has_three_stages_and_matches_central_differences(tmp_path, cache_dir):
    fwd = _bundle([share], tmp_path / "f", "float64", cache_dir)
    grad = Kernel("share_vjp", hawk.diff.vjp(share, primal_unit=fwd.digest))
    assert len(stages(grad.name, grad.sinks)) == 3
    bundle = _bundle([grad], tmp_path / "g", "float64", cache_dir)
    x = _x("float64")
    w = np.random.default_rng(5).uniform(-1, 1, N)
    bar_x = np.zeros(N)
    _run(bundle, "share_vjp", x=x, bar_out=w, bar_x=bar_x)

    def loss(v):
        z = np.zeros(N)
        _run(fwd, "share", x=v, out=z)
        return float(w @ z)

    h, fd = 1e-6, np.zeros(N)
    for i in range(0, N, 7):
        e = np.zeros(N)
        e[i] = h
        fd[i] = (loss(x + e) - loss(x - e)) / (2 * h)
    np.testing.assert_allclose(bar_x[::7], fd[::7], rtol=1e-6, atol=1e-9)


def test_the_jvp_matches_central_differences(tmp_path, cache_dir):
    fwd = _bundle([share], tmp_path / "f", "float64", cache_dir)
    tan = Kernel("share_jvp", hawk.diff.jvp(share, primal_unit=fwd.digest))
    bundle = _bundle([tan], tmp_path / "t", "float64", cache_dir)
    x = _x("float64")
    dx = np.random.default_rng(5).uniform(-1, 1, N)
    dz = np.zeros(N)
    planes = dict(runtime.load(bundle.directory, "share_jvp").arg_spec)
    assert len(stages(tan.name, tan.sinks)) == 2, planes
    _run(bundle, "share_jvp", x=x, dot_x=dx, dot_out=dz)
    h = 1e-6
    lo, hi = np.zeros(N), np.zeros(N)
    _run(fwd, "share", x=x - h * dx, out=lo)
    _run(fwd, "share", x=x + h * dx, out=hi)
    np.testing.assert_allclose(dz, (hi - lo) / (2 * h), rtol=1e-6, atol=1e-9)


def _reductions(sinks):
    return [n for n in _post_order(tuple(s.value for s in sinks)) if isinstance(n, ReduceSamples)]


def test_the_derivative_of_a_sum_keeps_its_key_count_and_policy():
    keyed = [n for n in _reductions(hawk.diff.vjp(share_c)) if n.key is not None]
    assert keyed and all(n.count is not None and n.policy == "compensated" for n in keyed)
    for derive in (hawk.diff.vjp, hawk.diff.jvp):
        found = _reductions(derive(whole))
        assert found and {n.policy for n in found} == {"ordered"}


# -- runs -------------------------------------------------------------------

def test_two_runs_are_identical(tmp_path, cache_dir):
    """The plane the runtime owns is zeroed for every run."""
    bundle = _bundle([share_c], tmp_path, "float64", cache_dir)
    x = _x("float64")
    first, second = np.zeros(N), np.zeros(N)
    _run(bundle, "share_c", x=x, out=first)
    _run(bundle, "share_c", x=x, out=second)
    assert first.tobytes() == second.tobytes()


# -- refused by name --------------------------------------------------------

def test_ordered_and_a_keyed_max_are_refused_by_name():
    with pytest.raises(HawkError, match="policy='ordered' over some of the lane axes"):
        @kernel
        def ordered(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
            return x / hawk.sum(x, over="col", policy="ordered")
    with pytest.raises(HawkError, match="a max over some of the lane axes"):
        @kernel
        def peak(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
            return x / hawk.max(x, over="col")


def test_a_stepping_kernel_and_group_are_refused_by_name(tmp_path, cache_dir):
    @kernel
    def reading(lane: Grid["row", "col":COL], x: Scalar, terminated: Terminated,
                y: Mutable[Scalar]):
        y = x / hawk.sum(x, over="col")
        terminated = x > 1.0
    with pytest.raises(HawkError, match="stepping or segmented"):
        _bundle([hawk.steps(reading, "auto")], tmp_path / "s", "float64", cache_dir)

    @kernel(group="col")
    def grouped(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
        return x / hawk.sum(x, over="col")
    with pytest.raises(HawkError, match="group="):
        _bundle([grouped], tmp_path, "float64", cache_dir)


def test_a_segmented_kernel_is_refused_by_name(tmp_path, cache_dir):
    @hawk.kernel(steps=1)
    def segmented(lane: Grid["row", "col":COL], k: Index, bs: Vector[2], y: Mutable[Scalar]):
        y = y + m.dispatch(k, [bs[0], bs[1]], policy="segmented") / hawk.sum(bs[0], over="col")
    with pytest.raises(HawkError, match="stepping or segmented"):
        _bundle([segmented], tmp_path, "float64", cache_dir)
