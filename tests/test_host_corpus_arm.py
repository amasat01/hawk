# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host values of the small kernels the old bit-identity corpus ran, read back against an independent reference.

The old rows compared an embedded C unit with the host build bit for bit; the C unit is gone, so each kernel's host
output is compared with its own reference instead: the ``tests/_eval.py`` interpreter, numpy, or exact Python integers.
Tolerances follow the neighbouring kept host rows: ``rtol = 1e-12`` for float64 (``test_emit_sample_subscript``) and
``1e-5`` for float32 (``test_ir_rays``); integer planes are compared exactly."""

from __future__ import annotations

import _deployable as D
import _kernels as K
import _oracle as O
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Index, Mutable, Vector
from hawk.artifact import build_bundle
from hawk import _ir
from hawk.trace.kernel import Kernel
from hawk import Terminated

N = 40
SEED = 20261016
MODES = ("float64", "float32")
_NP = {"float64": np.float64, "float32": np.float32}
_TOL = {"float64": 1e-12, "float32": 1e-5}


def _host(tmp_path, cache_dir, kernel, mode="float64", **kw):
    bundle = build_bundle([kernel], tmp_path / f"{kernel.name}_{mode}", mode=mode, targets=("host",),
                          host_profile="native", cache_dir=cache_dir, **kw)
    return O.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))


def _run(loaded, count, **planes):
    O.runtime.run(loaded, base=0, count=count, n_samples=count, **planes)
    return planes


def _sample(values, i):
    """Sample ``i`` of every plane of ``values`` in the interpreter's own shape."""
    return {k: (v if not isinstance(v, np.ndarray) else (v[i] if v.ndim == 1 else v[:, i]))
            for k, v in values.items()}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["drag", "scale", "chain6"])
def test_the_host_values_of_the_small_kernels_equal_the_interpreter(name, mode, tmp_path, cache_dir):
    kernel, real = K.CORPUS[name], _NP[mode]
    rng = np.random.default_rng(SEED)
    if name == "scale":
        given = {"x": rng.uniform(-2.0, 2.0, N).astype(real), "a": float(real(1.25)), "b": float(real(-0.5)),
                 "y": np.zeros(N, real)}
        out = "y"
    else:
        param = "rho" if name == "drag" else "k"
        given = {"state": rng.uniform(-2.0, 2.0, (3, N)).astype(real), param: float(real(0.75)),
                 "out": np.zeros((3, N), real)}
        out = "out"
    got = _run(_host(tmp_path, cache_dir, kernel, mode), N, **given)[out]
    for i in range(N):
        env = {k: v for k, v in _sample(given, i).items() if k != out}
        want = np.asarray(evaluate(kernel.sinks, env)[out], dtype=float)
        mine = np.asarray(got[i] if got.ndim == 1 else got[:, i], dtype=float)
        np.testing.assert_allclose(mine, want, rtol=_TOL[mode], atol=_TOL[mode], err_msg=f"{name} sample {i}")


@pytest.mark.parametrize("mode", MODES)
def test_the_rank_one_vocabulary_equals_numpy(mode, tmp_path, cache_dir):
    real = _NP[mode]
    rng = np.random.default_rng(SEED)
    v, k = rng.uniform(0.2, 2.0, (3, N)).astype(real), float(real(0.5))
    got = _run(_host(tmp_path, cache_dir, D.vocab, mode), N, v=v, k=k, y=np.zeros((3, N), real))["y"]
    want = D.ref_vocab(v.astype(np.float64), k)
    np.testing.assert_allclose(got, want, rtol=_TOL[mode], atol=_TOL[mode])


@hawk.kernel
def vdiv(a: Vector[3], b: Vector[3], y: Mutable[Vector[3]]):
    """Rank>=1 over rank>=1: ``a_k / b_k``."""
    y = a / b


@pytest.mark.parametrize("mode", MODES)
def test_a_vector_over_a_vector_is_the_componentwise_quotient(mode, tmp_path, cache_dir):
    real = _NP[mode]
    rng = np.random.default_rng(SEED)
    a, b = (rng.uniform(0.5, 2.0, (3, N)).astype(real) for _ in range(2))
    got = _run(_host(tmp_path, cache_dir, vdiv, mode), N, a=a, b=b, y=np.zeros((3, N), real))["y"]
    np.testing.assert_allclose(got, a / b, rtol=_TOL[mode], atol=0)


@hawk.kernel
def scaled(n: hawk.Int32, x: hawk.Scalar, a: hawk.Param, y: Mutable[hawk.Scalar], c: Mutable[hawk.Int32]):
    y = m.real(n) * a * m.sin(x) + m.real(m.int32(x * 3.0))
    c = n * 2


@pytest.mark.parametrize("mode", MODES)
def test_a_kernel_mixing_integer_and_real_planes_equals_the_interpreter(mode, tmp_path, cache_dir):
    real = _NP[mode]
    rng = np.random.default_rng(SEED)
    given = {"n": rng.integers(-1000, 1000, N).astype(np.int32), "x": rng.uniform(-2.0, 2.0, N).astype(real),
             "a": float(real(1.25)), "y": np.zeros(N, real), "c": np.zeros(N, np.int32)}
    got = _run(_host(tmp_path, cache_dir, scaled, mode), N, **given)
    for i in range(N):
        want = evaluate(scaled.sinks, {"n": int(given["n"][i]), "x": float(given["x"][i]), "a": given["a"]})
        assert float(got["y"][i]) == pytest.approx(float(want["y"]), rel=_TOL[mode], abs=_TOL[mode])
        assert int(got["c"][i]) == int(want["c"]) == 2 * int(given["n"][i])


# --- the Index kernels on their edge grid ------------------------------------------------------------------
@hawk.kernel
def index_arith(a: Index[np.int64], b: Index[np.int64], s: Mutable[Index[np.int64]], p: Mutable[Index[np.int64]],
                ng: Mutable[Index[np.int64]]):
    s = a + b - 3
    p = a * b
    ng = -a


@hawk.kernel
def index_div(a: Index[np.int64], b: Index[np.int64], q: Mutable[Index[np.int64]]):
    q = a / b


@hawk.kernel
def index_divshift(a: Index[np.int64], b: Index[np.int64], q: Mutable[Index[np.int64]], r: Mutable[Index[np.int64]],
                   sh: Mutable[Index[np.int64]]):
    q = a // b
    r = a % b
    sh = (a << b) >> 1


#: Small values, shift counts at and past the width, and magnitudes past 2^31 and 2^40 (so products overflow).
_EDGES = [0, 1, -1, 2, -2, 7, -7, 31, 63, 64, 1000, -1000, 2**31 - 1, -2**31, 2**40, -2**40]
_I64 = (-(2**63), 2**63 - 1)


def _grid():
    a, b = np.meshgrid(np.array(_EDGES, dtype=np.int64), np.array(_EDGES, dtype=np.int64))
    return a.reshape(-1).copy(), b.reshape(-1).copy()


def _keep(a, b, ok):
    keep = [k for k, (x, y) in enumerate(zip(a.tolist(), b.tolist())) if ok(x, y)]
    return a[keep].copy(), b[keep].copy()


def test_index_plus_minus_times_and_negation_equal_exact_integers(tmp_path, cache_dir):
    a, b = _keep(*_grid(), lambda x, y: all(_I64[0] <= v <= _I64[1] for v in (x + y, x + y - 3, x * y, -x)))
    n = a.size
    got = _run(_host(tmp_path, cache_dir, index_arith), n, a=a, b=b, s=np.zeros(n, np.int64),
               p=np.zeros(n, np.int64), ng=np.zeros(n, np.int64))
    np.testing.assert_array_equal(got["s"], np.array([x + y - 3 for x, y in zip(a.tolist(), b.tolist())], np.int64))
    np.testing.assert_array_equal(got["p"], np.array([x * y for x, y in zip(a.tolist(), b.tolist())], np.int64))
    np.testing.assert_array_equal(got["ng"], -a)


def test_index_division_equals_truncating_integer_division_where_it_is_defined(tmp_path, cache_dir):
    a, b = _keep(*_grid(), lambda x, y: y != 0 and not (x == _I64[0] and y == -1))
    n = a.size
    got = _run(_host(tmp_path, cache_dir, index_div), n, a=a, b=b, q=np.zeros(n, np.int64))["q"]
    want = np.array([abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1) for x, y in zip(a.tolist(), b.tolist())], np.int64)
    np.testing.assert_array_equal(got, want)


def _wrap64(x: int) -> int:
    """``x`` reduced mod 2^64 into the signed int64 range."""
    return (x + 2**63) % 2**64 - 2**63


def test_index_floor_division_modulo_and_shifts_equal_numpy_and_exact_integers(tmp_path, cache_dir):
    """``//`` and ``%`` take the whole grid, numpy's way (a zero divisor gives 0, as in ``test_int_planes``);
    ``(a << b) >> 1`` is compared where the shift count is inside the word, in exact wrapped integers."""
    a, b = _grid()
    n = a.size
    got = _run(_host(tmp_path, cache_dir, index_divshift), n, a=a, b=b, q=np.zeros(n, np.int64),
               r=np.zeros(n, np.int64), sh=np.zeros(n, np.int64))
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(got["q"], a // b, err_msg="q")
        np.testing.assert_array_equal(got["r"], a % b, err_msg="r")
    inside = (b >= 0) & (b < 64)
    want = np.array([_wrap64(_wrap64(x << y) >> 1) for x, y in zip(a[inside].tolist(), b[inside].tolist())], np.int64)
    np.testing.assert_array_equal(got["sh"][inside], want, err_msg="sh")


# --- masks ----------------------------------------------------------------------------------------------
def test_a_finishing_kernel_under_two_masks_keeps_a_flagged_samples_value(tmp_path, cache_dir):
    rng = np.random.default_rng(SEED)
    x, lim = rng.uniform(-2.0, 2.0, N), np.full(N, 0.5)
    terminated, rejected = np.zeros(N, bool), np.zeros(N, bool)
    terminated[: N // 4] = True
    rejected[N // 4: N // 2] = True
    y0 = rng.uniform(-9.0, 9.0, N)
    y = y0.copy()
    done = terminated.copy()
    loaded = _host(tmp_path, cache_dir, D.two_fin, kind=D.TWO_MASKS)
    _run(loaded, N, x=x, lim=lim, terminated=done, rejected=rejected, y=y)
    live = ~(terminated | rejected)
    np.testing.assert_array_equal(y[~live], y0[~live])               # a flagged sample keeps the caller's value
    np.testing.assert_allclose(y[live], 2.0 * x[live], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(done[live], (2.0 * x)[live] > 0.5)   # the finish sets terminated only where live
    np.testing.assert_array_equal(done[~live], terminated[~live])


def _gp() -> Kernel:
    """Own-column partials under the default mask guard, built as IR (the ``AccumWrite`` a sum sink makes)."""
    x = _ir.leaf(_ir.new_context(), "vocab_read", "plane_in", "x", _ir.TensorType((), "f64"))
    return Kernel("gp", (_ir.accum_write("acc", _ir.make("mul", (x, x))),), {"terminated": Terminated})


def test_an_own_column_partial_is_written_where_live_and_kept_where_masked(tmp_path, cache_dir):
    gp = _gp()
    rng = np.random.default_rng(SEED)
    x, acc0 = rng.uniform(-2.0, 2.0, N), rng.uniform(-9.0, 9.0, N)
    stopped = rng.random(N) < 0.3
    assert stopped.any() and not stopped.all()
    acc = acc0.copy()
    _run(_host(tmp_path, cache_dir, gp), N, x=x, terminated=stopped.copy(), acc=acc)
    want = np.array([acc0[i] if stopped[i] else float(evaluate(gp.sinks, {"x": float(x[i])})["acc"])
                     for i in range(N)])
    np.testing.assert_allclose(acc, want, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(acc[stopped], acc0[stopped])


# --- the stepping, finishing and dispatching kernels: one launch against a hand-written numpy reference ---------
def _osc_steps(dt, t_final, x, v, t, terminated, k):
    """``test_steps.osc`` applied ``k`` times to every live sample, the stop rule read after each step."""
    x, v, t, done = x.copy(), v.copy(), t.copy(), terminated.copy()
    for _ in range(k):
        live = ~done
        tn = t + dt
        x, v = np.where(live, x + dt * v, x), np.where(live, v + dt * -x, v)
        t = np.where(live, tn, t)
        done = done | (live & (tn >= t_final))
    return x, v, t, done


@pytest.mark.parametrize("k", [1, 4])
def test_the_oscillator_step_equals_numpy_for_one_and_for_four_steps_per_launch(k, tmp_path, cache_dir):
    import test_steps as S

    rng = np.random.default_rng(SEED)
    dt = 0.0625
    x, v = rng.uniform(-1.0, 1.0, N), rng.uniform(-1.0, 1.0, N)
    t, t_final = np.zeros(N), rng.choice([0.1, 0.2, 0.5, 2.0], N)
    stopped = rng.random(N) < 0.2
    kernel = S.osc if k == 1 else hawk.steps(S.osc, k)
    planes = dict(t_final=t_final, terminated=stopped.copy(), x=x.copy(), v=v.copy(), t=t.copy(), dt=dt)
    _run(_host(tmp_path, cache_dir, kernel), N, **planes)
    wx, wv, wt, wdone = _osc_steps(dt, t_final, x, v, t, stopped, k)
    np.testing.assert_allclose(planes["x"], wx, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(planes["v"], wv, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(planes["t"], wt, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(planes["terminated"], wdone)


def test_a_finish_inside_one_side_of_an_if_stops_only_where_that_side_ran(tmp_path, cache_dir):
    import test_finish_own_sample as F

    rng = np.random.default_rng(SEED)
    x = rng.uniform(-2.0, 2.0, N)
    stopped = rng.random(N) < 0.2
    y0 = rng.uniform(-9.0, 9.0, N)
    y, done = y0.copy(), stopped.copy()
    _run(_host(tmp_path, cache_dir, F.branch_finish), N, x=x, terminated=done, y=y)
    live = ~stopped
    np.testing.assert_allclose(y[live], 2.0 * x[live], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(y[stopped], y0[stopped])
    np.testing.assert_array_equal(done, stopped | (live & (x > 0.0) & (x > 1.0)))


@pytest.mark.parametrize("name", ["mix_auto", "mix_switch"])
def test_a_dispatch_over_a_vector_picks_its_scaled_branch_per_sample(name, tmp_path, cache_dir):
    import test_dispatch_mix_and_guard as G

    kernel = getattr(G, name)
    rng = np.random.default_rng(3)
    n = G.N
    k = rng.integers(-2, G.K + 2, n).astype(np.int32)           # out of range clamps
    bs = rng.uniform(0.5, 2.0, (G.K, n))
    stopped = rng.random(n) < 0.2
    y0 = np.full(n, -7.0)
    y = y0.copy()
    planes = dict(k=k, bs=bs, terminated=stopped.copy(), y=y)
    if name == "mix_auto":
        planes["dispatch0_mix"] = np.zeros(G.K, np.uint32)
    _run(_host(tmp_path, cache_dir, kernel), n, **planes)
    cls = np.clip(k, 0, G.K - 1)
    want = np.where(stopped, y0, bs[cls, np.arange(n)] * (cls + 1.0))
    np.testing.assert_allclose(y, want, rtol=1e-14, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_the_reverse_of_the_drag_kernel_equals_the_interpreter(mode, tmp_path, cache_dir):
    from hawk.diff import vjp

    real = _NP[mode]
    kernel = Kernel("drag_vjp", vjp(K.drag))
    rng = np.random.default_rng(SEED)
    given = {"state": rng.uniform(-2.0, 2.0, (3, N)).astype(real), "rho": float(real(0.75)),
             "bar_out": rng.uniform(-2.0, 2.0, (3, N)).astype(real)}
    names = [nm for role, nm in kernel.arg_spec if role in O.OUTPUT_ROLES]
    outs = {nm: np.zeros(given["state"].shape if nm == "bar_state" else (N,), real) for nm in names}
    bundle = build_bundle([K.drag, kernel], tmp_path / mode, mode=mode, targets=("host",), host_profile="native",
                          cache_dir=cache_dir)       # a derivative is built beside its primal
    loaded = O.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    got = _run(loaded, N, **given, **outs)
    for i in range(N):
        want = evaluate(kernel.sinks, _sample(given, i))
        for nm in names:
            have = np.asarray(got[nm][:, i] if got[nm].ndim == 2 else got[nm][i], dtype=float)
            np.testing.assert_allclose(have, np.asarray(want[nm], dtype=float), rtol=_TOL[mode], atol=_TOL[mode], err_msg=nm)
