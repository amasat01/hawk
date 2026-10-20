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

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Kernel, Mutable, Param, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.diff import param_planes, vjp
from hawk.emit.backend import ActiveSpec, SegmentSpec, render_source
from hawk.emit.cuda import CudaBackend
from hawk.ir import HawkError

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


def _launch(bundle, name, target, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(**planes)


def _xp(target):
    if target == "host":
        return np
    import cupy

    return cupy


def _run(bundle, target, n, mode):
    xp = _xp(target)
    x, gy, k, b = _inputs(n, mode)
    rows = -(-n // 32)
    planes = {name: xp.full(rows, np.nan, dtype=mode)
              for name in ("bar_k", "bar_k_comp", "bar_b", "bar_b_comp")}
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
        row = got[name].astype(np.float64) + got[name + "_comp"].astype(np.float64)
        assert np.all(np.isfinite(got[name])) and np.all(np.isfinite(got[name + "_comp"]))
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
    summed.update({f"bar_{c}{t}": xp.full(rows, np.nan, dtype=mode)
                   for c in COEFFS for t in ("", "_comp")})
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
        row = got[f"bar_{c}"].astype(np.float64) + got[f"bar_{c}_comp"]
        err = np.abs(row - _per_row(t).sum(1))
        assert np.all(err <= TOL[mode] * np.abs(_per_row(t)).sum(1) + 1e-300), c


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
        scale = np.abs(_per_row(t)).sum(1)
        h = host[name].astype(np.float64) + host[name + "_comp"]
        d = dev[name].astype(np.float64) + dev[name + "_comp"]
        assert np.all(np.abs(h - d) <= 2 * TOL[mode] * scale + 1e-300), name


def test_sidecar_lists_the_cells(built):
    sidecar = sidecar_of(built[0], "law_ss")
    assert "sample_sums" not in sidecar
    assert list(sidecar["cells"]) == ["bar_k", "bar_b"]
    assert {"bar_k", "bar_k_comp", "bar_b", "bar_b_comp"} <= set(sidecar["accum_outputs"])
    assert sidecar["exec_access"] == sidecar_of(built[0], "law_params")["exec_access"]


@pytest.mark.parametrize("entry, kw", [
    ("segmented", {"segment": SegmentSpec("offsets", 0)}),
    ("active-set", {"active": ActiveSpec("map", "count")}),
    ("_persist", {"one_step": object()}),
])
def test_entries_whose_warp_is_not_32_samples_refuse_by_name(entry, kw):
    derived = Kernel("law_ss", VJP)
    with pytest.raises(HawkError, match=rf"sample_sum \['bar_k', 'bar_b'\].*{entry} entry"):
        render_source(derived.name, derived.sinks, derived.walk, CudaBackend(), **kw)
