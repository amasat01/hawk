# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.math.pair_index()``: one lane per pair ``(i, j)``, ``i < j``, of ``N`` items.

Lane ``k`` is the pair with ``k = j(j-1)/2 + i`` (``j`` the slow axis, ``i`` the fast
one): ``Grid["pair"]`` declared, ``pair=N*(N-1)//2`` passed. The map is a float
estimate of ``j`` put right by one step each way, in 64-bit integers on every
target; the rows pin it

* exhaustively for small ``N`` and across ``float32``'s first wrong estimate
  (``k = 10619135``, ``N = 5000``), on the host and the device;
* at the edges ``k = T(j) - 1, T(j)`` and at the top of the range (``float32``:
  ``k < 2**31``; ``float64``: ``k`` up to ``2**53``) through a window of 64 lanes
  run at a high ``base``;
* in the emitted C unit (strict compile, cppcheck, and the host's integer and
  float results bit for bit) and the CUDA source (compiles);
* with ``Table.at(i)`` / ``.at(j)`` reads, a per-pair output, a keyed ``+=`` row
  sum and an OR into a bit mask, against numpy; and ``vjp`` / ``jvp`` against
  central differences.
"""

from __future__ import annotations

import _deploy as L
import _embedded as E
import numpy as np
import pytest
from _toolchain import compile_check
from conftest import sidecar_of

import hawk
from hawk import Grid, Index, Kernel, Mutable, Scalar, Table, UInt32, kernel
from hawk import math as m
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.embedded import EmbeddedProfile, check, export, render
from hawk.emit import FLOAT32, FLOAT64, render_source
from hawk.targets import TARGETS

MODES = ("float64", "float32")
WORDS = 4          # mask words per row: j >> 5 < 4 for N <= 128


@kernel
def pairs(lane: Grid["pair"], oi: Mutable[Index], oj: Mutable[Index]):
    i, j = m.pair_index()
    oi = i
    oj = j


@kernel
def window(lane: Grid["pair"], si: Mutable["w"], sj: Mutable["w"]):
    """64 consecutive lanes at any ``base``: lane ``k`` lands in cell ``k mod 64``."""
    i, j = m.pair_index()
    k = m.sample_index()
    cell = k - ((k >> 6) << 6)
    hawk.scatter(si, policy="plain")[cell] += m.real(i)
    hawk.scatter(sj, policy="plain")[cell] += m.real(j)


@kernel
def energy(lane: Grid["pair"], x: Table["node"], d: Mutable[Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = x.at(i) * x.at(j) * x.at(j) + x.at(i) * x.at(i) * x.at(j) + m.sin(x.at(i) - x.at(j))
    d = t
    hawk.scatter(rs)[j] += t


@kernel
def poly(lane: Grid["pair"], x: Table["node"], d: Mutable[Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = x.at(i) * x.at(j) * x.at(j) * x.at(i)       # products only: nothing for a host FMA to fuse
    d = t
    hawk.scatter(rs)[j] += t


@kernel
def near(lane: Grid["pair"], x: Table["node"], mask: Mutable[UInt32]):
    """Bit ``j`` of row ``i`` of a symmetric-free adjacency mask: ``x_i * x_j > 0.5``."""
    i, j = m.pair_index()
    bit = m.select(x.at(i) * x.at(j) > 0.5, m.uint32(1) << m.uint32(j & 31), m.uint32(0))
    hawk.scatter(mask, policy="or")[i * WORDS + (j >> 5)] |= bit


def _pairs(n_items):
    """``(i, j)`` of every pair in lane order, from numpy alone."""
    j = np.repeat(np.arange(n_items, dtype=np.int64), np.arange(n_items))
    k = np.arange(j.size, dtype=np.int64)
    return k - j * (j - 1) // 2, j


def _tri(j):
    return j * (j - 1) // 2


def _true(k):
    """The exact ``(i, j)`` of lane ``k`` (Python integers)."""
    j = int((1 + (8 * k + 1) ** 0.5) / 2)
    while _tri(j) > k:
        j -= 1
    while _tri(j + 1) <= k:
        j += 1
    return k - _tri(j), j


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    """``{(mode, kernel name): bundle}``: a bundle has ONE execution axis, so the
    lane-local kernel and the keyed ones build apart."""
    out = {}
    for mode in MODES:
        for ks in ([pairs], [window, energy, poly, near]):
            b = build_bundle(ks, tmp_path_factory.mktemp(f"pair_{mode}"), mode=mode,
                             targets=("cuda", "host"), cache_dir=cache_dir)
            out.update({(mode, k.name): b for k in ks})
    return out


def _host(bundles, mode, k):
    b = bundles[mode, k.name]
    return rt.load(b.directory, k.name, sidecar_of(b, k.name))


# ---- the map, on the host -------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_items", [2, 3, 4, 33, 300, 5000])
def test_every_lane_is_its_pair_on_the_host(bundles, mode, n_items):
    """``N = 5000`` is 12.5 million lanes, past ``float32``'s first wrong estimate."""
    n = n_items * (n_items - 1) // 2
    oi, oj = np.zeros(n, np.int64), np.zeros(n, np.int64)
    rt.run(_host(bundles, mode, pairs), oi=oi, oj=oj, pair=n)
    want_i, want_j = _pairs(n_items)
    assert np.array_equal(oj, want_j) and np.array_equal(oi, want_i)
    assert (oi < oj).all()


def _edge_bases(limit):
    """Window starts around ``T(j) - 1, T(j)`` for ``j`` up to the ``limit``, the first
    ``float32`` miss and the top of the range."""
    js = [2, 3, 4, 5, 63, 64, 65, 1000, 4608, 4609, 4610, 46340, 46341, 65535, 65536]
    js += [j for j in (2**20 + 1, 2**24, 10**8, 134217728, 134217729) if _tri(j) < limit]
    bases = [max(_tri(j) - 32, 0) for j in js if _tri(j) < limit]
    bases += [10619135 - 32, limit - 64]
    return bases


@pytest.mark.parametrize("mode, limit", [("float64", 2**53), ("float32", 2**31)])
def test_edge_lanes_are_exact_at_a_high_base(bundles, mode, limit):
    a = _host(bundles, mode, window)
    real = np.float32 if mode == "float32" else np.float64
    for base in _edge_bases(limit):
        si, sj = np.zeros(64, real), np.zeros(64, real)
        rt.run(a, si=si, sj=sj, pair=base + 64, base=base, count=64, w=64)
        want = np.array([_true(base + c) for c in range(64)], dtype=np.float64)
        got = np.stack([si, sj], axis=1)
        assert np.array_equal(got[(np.arange(base, base + 64) % 64)], want), (mode, base)


# ---- Table reads, per-pair output, keyed sum, OR mask -------------------------

def _x(n_items):
    return np.linspace(0.3, 1.7, n_items)


def test_per_pair_output_and_row_sum_equal_numpy(bundles):
    n_items = 40
    n = n_items * (n_items - 1) // 2
    x = _x(n_items)
    d, rs = np.zeros(n), np.zeros(n_items)
    rt.run(_host(bundles, "float64", energy), x=x, d=d, rs=rs, pair=n, node=n_items)
    i, j = _pairs(n_items)
    want = x[i] * x[j] ** 2 + x[i] ** 2 * x[j] + np.sin(x[i] - x[j])
    np.testing.assert_allclose(d, want, rtol=1e-14, atol=1e-14)
    row = np.zeros(n_items)
    np.add.at(row, j, want)
    np.testing.assert_allclose(rs, row, rtol=1e-13, atol=1e-13)
    assert rs[0] == 0.0


@pytest.mark.parametrize("n_items", [2, 33, 100])
def test_an_or_mask_written_from_the_pair_grid_equals_numpy(bundles, n_items):
    n = n_items * (n_items - 1) // 2
    x = _x(n_items)
    mask = np.zeros(n_items * WORDS, np.uint32)
    rt.run(_host(bundles, "float64", near), x=x, mask=mask, pair=n)
    i, j = _pairs(n_items)
    on = x[i] * x[j] > 0.5
    want = np.zeros(n_items * WORDS, np.uint32)
    np.bitwise_or.at(want, (i * WORDS + (j >> 5))[on],
                     (np.uint32(1) << (j & 31).astype(np.uint32))[on])
    assert mask.tobytes() == want.tobytes()
    assert on.any()


# ---- derivatives: the lane axis map only ---------------------------------------

def _energy_np(x, n_items):
    i, j = _pairs(n_items)
    t = x[i] * x[j] ** 2 + x[i] ** 2 * x[j] + np.sin(x[i] - x[j])
    rs = np.zeros(n_items)
    np.add.at(rs, j, t)
    return t, rs


N_D = 9
_JVP = Kernel("energy_jvp", jvp(energy, wrt=("x",)))
_VJP = Kernel("energy_vjp", vjp(energy, wrt=("x",)))


@pytest.mark.parametrize("n_items", [N_D, 40])
def test_jvp_of_a_per_pair_kernel_matches_central_differences(n_items):
    """One build serves both sizes: ``node=N`` sizes the tangent of the keyed sum."""
    n = n_items * (n_items - 1) // 2
    x = _x(n_items)
    v = np.random.default_rng(1).normal(size=n_items)
    got = _JVP(x=x, dot_x=v, pair=n, node=n_items)
    assert got["dot_d"].shape == (n,) and got["dot_rs"].shape == (n_items,)
    h = 1e-6
    td, trs = _energy_np(x + h * v, n_items)
    md, mrs = _energy_np(x - h * v, n_items)
    np.testing.assert_allclose(got["dot_d"], (td - md) / (2 * h), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(got["dot_rs"], (trs - mrs) / (2 * h), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("n_items", [N_D, 40])
def test_vjp_of_a_per_pair_kernel_matches_central_differences(n_items):
    n = n_items * (n_items - 1) // 2
    x = _x(n_items)
    rng = np.random.default_rng(2)
    u_d, u_rs = rng.normal(size=n), rng.normal(size=n_items)
    got = _VJP(x=x, bar_d=u_d, bar_rs=u_rs, pair=n, node=n_items)
    assert got.shape == (n_items,)
    h = 1e-6
    want = np.zeros(n_items)
    for e in range(n_items):
        step = np.zeros(n_items)
        step[e] = h
        td, trs = _energy_np(x + step, n_items)
        md, mrs = _energy_np(x - step, n_items)
        want[e] = (u_d @ (td - md) + u_rs @ (trs - mrs)) / (2 * h)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-8)


# ---- the emitted sources ------------------------------------------------------

@pytest.mark.parametrize("target", ["cuda", "host"])
@pytest.mark.parametrize("mode", [FLOAT64, FLOAT32], ids=["f64", "f32"])
def test_the_emitted_sources_compile(tmp_path, target, mode):
    for k in (pairs, energy, near):
        src = render_source(k.name, k.sinks, k.walk, TARGETS[target], mode=mode, kind=k.kind)
        path = tmp_path / f"{k.name}.{'cu' if target == 'cuda' else 'cpp'}"
        path.write_text(src.text)
        rc, stderr, argv = compile_check(path, target)
        assert rc == 0, f"$ {' '.join(argv)}\n{stderr[:2000]}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [pairs, poly, near], ids=lambda k: k.name)
def test_the_c_unit_is_strict_clean_and_equals_the_host(bundles, tmp_path, mode, k):
    strict = EmbeddedProfile(require_cppcheck=True)
    checks = check(render(k, mode=mode), profile=strict)
    assert all(run["rc"] == 0 for run in checks["cc"])
    assert checks["cppcheck"]["findings"] == []
    art = export(k, tmp_path, mode=mode)
    lib = E.cc_shared(art.c_path, tmp_path)
    host = _host(bundles, mode, k)
    n_items = 100
    n = n_items * (n_items - 1) // 2
    if k is pairs:
        c = E.run_c(lib, "pairs", art.report, n, mode=mode,
                    oi=np.zeros(n, np.int64), oj=np.zeros(n, np.int64))
        h = (np.zeros(n, np.int64), np.zeros(n, np.int64))
        rt.run(host, oi=h[0], oj=h[1], pair=n)
        assert c[0].tobytes() == h[0].tobytes() and c[1].tobytes() == h[1].tobytes()
        assert np.array_equal(c[1], _pairs(n_items)[1])
    elif k is near:
        x = _x(n_items).astype(np.float32 if mode == "float32" else np.float64)
        c = E.run_c(lib, "near", art.report, n, mode=mode, x=x,
                    mask=np.zeros(n_items * WORDS, np.uint32))
        h = np.zeros(n_items * WORDS, np.uint32)
        rt.run(host, x=x, mask=h, pair=n)
        assert c.tobytes() == h.tobytes()
    else:
        dt = np.float32 if mode == "float32" else np.float64
        x = _x(n_items).astype(dt)
        c = E.run_c(lib, "poly", art.report, n, mode=mode, x=x, d=np.zeros(n, dt),
                    rs=np.zeros(n_items, dt))
        hd, hrs = np.zeros(n, dt), np.zeros(n_items, dt)
        rt.run(host, x=x, d=hd, rs=hrs, pair=n, node=n_items)
        c_d = c[0] if isinstance(c, tuple) else c
        assert c_d.tobytes() == hd.tobytes()


# ---- the device ---------------------------------------------------------------

def _launch(bundles, mode, k, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = bundles[mode, k.name]
    plugin = L.device_plugin(bundle.directory, k.name, sidecar_of(bundle, k.name))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes).launch()


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_items", [300, 5000])
def test_every_lane_is_its_pair_on_the_device(bundles, mode, n_items):
    import cupy as cp

    n = n_items * (n_items - 1) // 2
    oi, oj = cp.zeros(n, cp.int64), cp.zeros(n, cp.int64)
    _launch(bundles, mode, pairs, {"oi": oi, "oj": oj, "pair": n})
    cp.cuda.Device().synchronize()
    want_i, want_j = _pairs(n_items)
    assert np.array_equal(oj.get(), want_j) and np.array_equal(oi.get(), want_i)


@pytest.mark.gpu
def test_an_or_mask_and_a_row_sum_on_the_device_equal_the_host(bundles):
    import cupy as cp

    n_items = 100
    n = n_items * (n_items - 1) // 2
    x = _x(n_items)
    mask = cp.zeros(n_items * WORDS, cp.uint32)
    _launch(bundles, "float64", near, {"x": cp.asarray(x), "mask": mask, "pair": n})
    h = np.zeros(n_items * WORDS, np.uint32)
    rt.run(_host(bundles, "float64", near), x=x, mask=h, pair=n)
    cp.cuda.Device().synchronize()
    assert mask.get().tobytes() == h.tobytes()
    d, rs = cp.zeros(n), cp.zeros(n_items)
    _launch(bundles, "float64", energy, {"x": cp.asarray(x), "d": d, "rs": rs,
                                         "pair": n})
    cp.cuda.Device().synchronize()
    want_d, want_rs = _energy_np(x, n_items)
    np.testing.assert_allclose(d.get(), want_d, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rs.get(), want_rs, rtol=1e-12, atol=1e-12)
