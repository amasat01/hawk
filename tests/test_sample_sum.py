# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``sample_sum`` sinks — a Param gradient summed over every 32 samples.

Rows (host; the ``cuda`` arm is marked ``gpu``):

* **reference** — each ``bar_<name>`` row plus its ``_comp`` row equals a
  numpy float64 sum of that run's 32 per-sample terms, for ragged and
  aligned ``n``, float32 and float64; every row is written (the planes
  start as NaN) and the per-sample ``bar_x`` is unchanged;
* **ten columns** — a rational law's ten parameter gradients (uneven
  column splits in the warp) equal the plain kernel's per-sample gradients
  summed per 32 samples;
* **bitwise** — repeated device launches give the same bits;
* **twins** — host rows equal device rows within the sum contract;
* **sidecar** — ``cells`` names the planes; the access class is the
  primal's, so both share one bundle;
* **refusals** — the segmented, active-set and ``_persist`` entries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import _launch, _xp, refuse_device_row, sidecar_of

import hawk
from hawk import Kernel, Mutable, Param, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.diff import param_planes, vjp
from hawk.errors import HawkError
from hawk import _ir

SIZES = (1, 31, 32, 33, 1000, 65536)
#: Row tolerance, relative to the run's sum of |term|.
TOL = {"float32": 1e-6, "float64": 1e-14}


@kernel
def law(x: Scalar, k: Param, b: Param, y: Mutable[Scalar]):
    y = k * x * x + b * hawk.math.sin(x)


PROMOTED = param_planes(law)
VJP = vjp(PROMOTED, wrt=("x", "k", "b"))
COEFFS = ("a0", "a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4")


@kernel
def rat(x: Scalar, a0: Param, a1: Param, a2: Param, a3: Param, a4: Param, a5: Param,
        b1: Param, b2: Param, b3: Param, b4: Param, y: Mutable[Scalar]):
    p = ((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0
    q = (((b4 * x + b3) * x + b2) * x + b1) * x
    y = p / (1.0 + hawk.math.abs(q))


RAT_PROMOTED = param_planes(rat)


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp(f"sample_sum_{request.param}")
    bundle = build_bundle([PROMOTED, Kernel("law_ss", VJP), rat, RAT_PROMOTED,
                           Kernel("rat_ss", vjp(RAT_PROMOTED, wrt=("x",) + COEFFS)),
                           Kernel("rat_vjp", vjp(rat, wrt=("x",) + COEFFS))],
                          root, mode=request.param,
                          targets=("cuda", "host"), cache_dir=cache_dir,
                          host_profile="native")
    return bundle, request.param


def _inputs(n, mode):
    rng = np.random.default_rng(n)
    dt = np.dtype(mode)
    return (rng.uniform(-2.0, 2.0, n).astype(dt), rng.standard_normal(n).astype(dt),
            dt.type(1.3), dt.type(-0.7))


def _run(bundle, target, n, mode):
    xp = _xp(target)
    x, gy, k, b = _inputs(n, mode)
    rows = 1 if target == "host" else -(-n // 32)       # the host library folds the rows into the one cell
    planes = {name: xp.full(rows, np.nan, dtype=mode)
              for name in ("bar_k", "bar_k_comp", "bar_b", "bar_b_comp") if target != "host" or not name.endswith("_comp")}
    planes.update(x=xp.asarray(x), bar_y=xp.asarray(gy), k=xp.asarray([k]),
                  b=xp.asarray([b]), bar_x=xp.zeros(n, dtype=mode))
    bound = _launch(bundle, "law_ss", target, planes)
    bound.launch()
    return _read(planes), (bound, planes)


def _read(planes):
    return {name: np.array(a.get() if hasattr(a, "get") else a) for name, a in planes.items()}


def _terms(n, mode):
    x, gy, _, _ = (np.asarray(v, dtype=np.float64) for v in _inputs(n, mode))
    return {"bar_k": gy * x * x, "bar_b": gy * np.sin(x)}


def _per_row(t):
    pad = np.zeros(-(-t.size // 32) * 32)
    pad[:t.size] = t
    return pad.reshape(-1, 32)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rows_equal_the_float64_reference(built, target, n):
    bundle, mode = built
    got, _ = _run(bundle, target, n, mode)
    for name, t in _terms(n, mode).items():
        ref, scale = _per_row(t).sum(1), np.abs(_per_row(t)).sum(1)
        if target == "host":
            ref, scale = ref.sum(keepdims=True), scale.sum(keepdims=True)
        row = got[name].astype(np.float64) + (got[name + "_comp"].astype(np.float64) if name + "_comp" in got else 0.0)
        assert np.all(np.isfinite(got[name])) and np.all(np.isfinite(got.get(name + "_comp", 0.0)))
        assert np.all(np.abs(row - ref) <= TOL[mode] * scale + 1e-300), name
    x, gy, k, b = (np.asarray(v, dtype=np.float64) for v in _inputs(n, mode))
    np.testing.assert_allclose(got["bar_x"], gy * (2 * k * x + b * np.cos(x)),
                               rtol=1e-5 if mode == "float32" else 1e-13, atol=1e-6)


@pytest.mark.parametrize("n", (33, 1000))
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_ten_columns_equal_the_per_sample_gradients_summed(built, target, n):
    bundle, mode = built
    xp, rng = _xp(target), np.random.default_rng(n)
    rows = -(-n // 32)
    common = {"x": xp.asarray(rng.uniform(-1.5, 1.5, n).astype(mode)),
              "bar_y": xp.asarray(rng.standard_normal(n).astype(mode))}
    cells = {c: rng.standard_normal() * 0.3 for c in COEFFS}
    summed = dict(common, bar_x=xp.zeros(n, dtype=mode))
    summed.update({c: xp.asarray(np.array([v], dtype=mode)) for c, v in cells.items()})
    cell_rows = 1 if target == "host" else rows     # the host library folds the rows into the one cell
    summed.update({f"bar_{c}{t}": xp.full(cell_rows, np.nan, dtype=mode)
                   for c in COEFFS for t in ("", "_comp") if target != "host" or not t})
    plain = dict(common, bar_x=xp.zeros(n, dtype=mode), **cells)
    plain.update({f"bar_{c}": xp.zeros(n, dtype=mode) for c in COEFFS})
    _launch(bundle, "rat_ss", target, summed).launch()
    _launch(bundle, "rat_vjp", target, plain).launch()
    got, want = _read(summed), _read(plain)
    # the device compiler may contract the cell and by-value spellings apart
    np.testing.assert_allclose(got["bar_x"], want["bar_x"], rtol=0,
                               atol=64 * np.finfo(mode).eps * np.abs(want["bar_x"]).max())
    for c in COEFFS:
        t = want[f"bar_{c}"].astype(np.float64)
        row = got[f"bar_{c}"].astype(np.float64) + (got[f"bar_{c}_comp"] if f"bar_{c}_comp" in got else 0.0)
        ref, scale = _per_row(t).sum(1), np.abs(_per_row(t)).sum(1)
        if target == "host":
            ref, scale = ref.sum(keepdims=True), scale.sum(keepdims=True)
        assert np.all(np.abs(row - ref) <= TOL[mode] * scale + 1e-300), c


@pytest.mark.gpu
def test_device_rows_are_bitwise_across_launches(built):
    bundle, mode = built
    first, (bound, planes) = _run(bundle, "cuda", 65536 + 17, mode)
    for _ in range(5):
        for name in ("bar_k", "bar_k_comp", "bar_b", "bar_b_comp"):
            planes[name].fill(np.nan)
        bound.launch()
        again = _read(planes)
        for name in ("bar_k", "bar_k_comp", "bar_b", "bar_b_comp"):
            assert again[name].tobytes() == first[name].tobytes(), name


@pytest.mark.gpu
@pytest.mark.parametrize("n", (33, 65536 + 17))
def test_host_rows_equal_device_rows_within_the_contract(built, n):
    bundle, mode = built
    host, _ = _run(bundle, "host", n, mode)
    dev, _ = _run(bundle, "cuda", n, mode)
    for name, t in _terms(n, mode).items():
        # The host library folds the rows into one cell and hands back no companion plane, so the two are compared as totals:
        # the device total is its rows (value plus companion) added in float64, the host total is its cell. Each is within
        # TOL * sum|term| of the exact sum (a row is within TOL * its own scale, and the row scales add up to the total's), so
        # the two are within twice that of each other: the bound is the contract's, not fitted to a run.
        scale = np.abs(t).sum()
        h = float(host[name][0])
        d = float(np.sum(dev[name].astype(np.float64) + dev[name + "_comp"]))
        assert abs(h - d) <= 2 * TOL[mode] * scale + 1e-300, name


def test_sidecar_lists_the_cells(built):
    sidecar = sidecar_of(built[0], "law_ss")
    assert "sample_sums" not in sidecar
    assert list(sidecar["cells"]) == ["bar_k", "bar_b"]
    assert {"bar_k", "bar_k_comp", "bar_b", "bar_b_comp"} <= set(sidecar["accum_outputs"])
    assert sidecar["exec_access"] == sidecar_of(built[0], "law_params")["exec_access"]


# ---- the rows a warp folds in one call (ospreys.cuda.emit._fold_shape) ----

@kernel
def fold1(x: Scalar, s: Mutable[Scalar]):
    s = hawk.sum(x)


@kernel
def fold2(x: Scalar, s: Mutable[Scalar], c: Mutable[Scalar]):
    s = hawk.sum(x)
    c = hawk.sum(x * x)


@kernel
def foldv(v: hawk.Vector[6], t: Mutable[hawk.Vector[6]]):
    t = hawk.sum(v)


@kernel
def foldmax(x: Scalar, hi: Mutable[Scalar]):
    hi = hawk.max(x, where=x > 0.0)


@kernel
def foldmin(x: Scalar, lo: Mutable[Scalar]):
    lo = hawk.min(x, where=x < 0.0)


FOLDS = {"fold1": (fold1, ("s",)), "fold2": (fold2, ("s", "c")), "foldv": (foldv, tuple(f"t_{k}" for k in range(6))),
         "foldmax": (foldmax, ("hi",)), "foldmin": (foldmin, ("lo",))}
FOLD_SIZES = (1, 31, 32, 33, 127, 128, 129, 1000, 65536 + 17)
FOLD_KS = (None, 1, 8)      # the settings, the one-row warp, the card kit's wide switch


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    """``{k: bundle}`` of every fold kernel, built under the settings (``None``) and under the forced ``k``."""
    from hawk.build_settings import BuildSettings
    out = {}
    for k in FOLD_KS:
        out[k] = build_bundle([kern for kern, _ in FOLDS.values()], tmp_path_factory.mktemp(f"fold_{k}"), mode="float64",
                              targets=("cuda", "host"), cache_dir=str(tmp_path_factory.mktemp(f"fold_cache_{k}")),
                              host_profile="native", settings=None if k is None else BuildSettings(fold_rows=k))
    return out


def _fold_rows_run(bundle, name, target, n):
    xp, rng = _xp(target), np.random.default_rng(n)
    _, cols = FOLDS[name]
    rows = 1 if target == "host" else -(-n // 32)       # the host library folds the rows into the one cell
    planes = {c + s: xp.full(rows, np.nan) for c in cols for s in ("", "_comp")
              if (not name.startswith(("foldm",)) or not s) and (target != "host" or not s)}
    wide = name == "foldv"
    if target == "host" and wide:       # the host library takes the six columns of a vector sum as one cell
        planes = {"t": xp.full(6, np.nan)}
    planes["v" if wide else "x"] = xp.asarray(rng.uniform(-2, 2, (6, n) if wide else n))
    _launch(bundle, name, target, planes).launch()
    out = {c: np.array(a.get() if hasattr(a, "get") else a) for c, a in planes.items() if c not in ("x", "v")}
    if target == "host" and wide:
        out = {f"t_{k}": out["t"][k:k + 1] for k in range(6)}
    return out


@pytest.mark.parametrize("k, rows", [(None, 4), (1, 1), (8, 8)])
def test_a_forced_k_reaches_the_built_text(folded, k, rows):
    """A bundle built under ``fold_rows(k)`` carries that k (a shared compile cache would hand every k the first text)."""
    refuse_device_row()
    text = (folded[k].directory / "fold1.cu").read_text()
    assert f"sampleSumWarp_<{rows}," in text


@pytest.mark.gpu
@pytest.mark.parametrize("name", FOLDS)
@pytest.mark.parametrize("n", FOLD_SIZES)
def test_device_rows_are_bit_identical_for_every_k(folded, name, n):
    ref = _fold_rows_run(folded[1], name, "cuda", n)         # k = 1: the one-row warp
    for k in (None, 8):
        got = _fold_rows_run(folded[k], name, "cuda", n)
        for c, rows in got.items():
            assert not np.isnan(rows).any(), (c, k)          # every row, the ragged last one too
            assert rows.view(np.uint64).tobytes() == ref[c].view(np.uint64).tobytes(), (c, k)


@pytest.mark.parametrize("name", FOLDS)
@pytest.mark.parametrize("n", FOLD_SIZES)
def test_the_host_twin_rows_equal_the_float64_reference(folded, name, n):
    got = _fold_rows_run(folded[None], name, "host", n)
    x = np.random.default_rng(n).uniform(-2, 2, (6, n) if name == "foldv" else n)
    rows = -(-n // 32)

    def per_row(t, fill=0.0):
        return np.pad(t, (0, rows * 32 - n), constant_values=fill).reshape(rows, 32)
    if name == "foldmax":
        assert np.array_equal(got["hi"], [per_row(np.where(x > 0, x, -np.inf), -np.inf).max()])
    elif name == "foldmin":
        assert np.array_equal(got["lo"], [per_row(np.where(x < 0, x, np.inf), np.inf).min()])
    else:
        want = {"fold1": lambda: {"s": x}, "fold2": lambda: {"s": x, "c": x * x},
                "foldv": lambda: {f"t_{k}": x[k] for k in range(6)}}[name]()
        for c, t in want.items():
            assert np.all(np.abs(got[c] - per_row(t).sum()) <= 1e-14 * np.abs(per_row(t)).sum() + 1e-300), c


def _awkward(n):
    """Mixed magnitudes, both zeros, a NaN, +-Inf and an overflow in distinct rows of 32."""
    rng = np.random.default_rng(n + 5)
    x = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-300, 300, n)
    x[::7] = rng.uniform(-1, 1, len(x[::7]))
    if n > 200:
        x[3], x[4] = 0.0, -0.0                     # zeros of both signs
        x[40 + 1] = np.nan                         # a NaN row
        x[70], x[71] = np.inf, -np.inf             # Inf - Inf is NaN
        x[100], x[101] = np.inf, 1.0               # an Inf row
        x[130:134] = 1.7e308                       # the sum overflows to Inf
        x[160:164] = -1.7e308
    return x


@pytest.mark.gpu
@pytest.mark.parametrize("name", ("fold1", "fold2", "foldv"))
@pytest.mark.parametrize("n", (1, 33, 1000, 4099))
def test_the_device_combine_on_awkward_values_equals_the_host_twin(folded, name, n):
    """Fast2Sum with an integer magnitude pick on the device, Knuth's two-sum on the host: the same total, NaN and Inf
    decided the same way, and a finite total within the sum contract of the device's (the host keeps one cell, no rows)."""
    x = _awkward(n)
    cols = FOLDS[name][1]
    rows = -(-n // 32)
    data = np.vstack([x] * 6) if name == "foldv" else x
    key = "v" if name == "foldv" else "x"
    # the device keeps one (sum, companion) row per 32 samples
    dev_planes = {c + s: _xp("cuda").full(rows, np.nan) for c in cols for s in ("", "_comp")}
    dev_planes[key] = _xp("cuda").asarray(data)
    _launch(folded[None], name, "cuda", dev_planes).launch()
    device = {k: np.array(a.get() if hasattr(a, "get") else a) for k, a in dev_planes.items() if k != key}
    # the host library folds the rows into one cell and keeps no companion plane: the cells of a vector sum are bound as one
    # cell of six values, a scalar sum by its name
    host_planes = {"t": np.full(6, np.nan)} if name == "foldv" else {c: np.full(1, np.nan) for c in cols}
    host_planes[key] = data
    _launch(folded[None], name, "host", host_planes).launch()
    host = {f"t_{k}": host_planes["t"][k] for k in range(6)} if name == "foldv" else {c: host_planes[c][0] for c in cols}
    terms = {"fold1": {"s": x}, "fold2": {"s": x, "c": x * x}, "foldv": {f"t_{k}": data[k] for k in range(6)}}[name]
    for c in cols:
        rows_sum = device[c]
        if not np.isfinite(np.sum(rows_sum)):
            # a NaN or an Inf in any row decides the total the same way in every order of the additions, so the host cell
            # must be the one the plain float64 addition of the device rows gives (a row's companion is not read: it is NaN
            # beside an Inf sum)
            np.testing.assert_array_equal(host[c], np.sum(rows_sum), err_msg=c)
        else:
            # finite: each total is within TOL * sum|x| of the exact sum, so the two are within twice that of each other
            total = math.fsum(rows_sum) + math.fsum(device[c + "_comp"])
            bound = 2 * TOL["float64"] * float(np.abs(terms[c]).sum())
            assert abs(float(host[c]) - total) <= bound + 1e-300, c


def test_a_sum_and_a_plane_named_like_its_companion_collide_by_name():
    """The compensation plane ``<name>_comp`` of a summing ``sample_sum`` may not be the name of another sink of the same role:
    refused when the kernel is declared, before any text."""
    with pytest.raises(HawkError, match="sample_sum 't': its compensation plane 't_comp' collides with a plane of the same name"):
        @hawk.kernel
        def clash(x: Scalar, t: Mutable[Scalar], t_comp: Mutable[Scalar]):
            t = hawk.sum(x * x)
            t_comp = hawk.sum(x)


def test_the_companion_plane_is_a_slot_of_the_walk():
    @hawk.kernel
    def one(x: Scalar, t: Mutable[Scalar]):
        t = hawk.sum(x * x)
    assert ("accum_out", "t_comp") in _ir.arg_spec(one.walk)
