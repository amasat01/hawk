# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra kinds in the standalone C target.

The C helpers (``hawk/embedded/linalg.py``) repeat ``aether::linalg``'s order of
floating-point operations, so a lane comes back with the SAME BITS from the
host build and from the C unit: values and ``info`` both, in both scalar modes,
failing lanes (zeros, partial factors) included. The unit also answers to the
whole toolchain check -- the C compiler under ``-std=c99`` and ``-std=c11`` with
the strict flags, then cppcheck's MISRA addon -- and to the emitted-text audit.
The stack a kernel holds is the cost formulas' figure, so a profile cap refuses
a dimension that does not fit.
"""

from __future__ import annotations

import pathlib
import re

import _embedded as E
import _kernels as K
import numpy as np
import pytest
from _linalg_kernels import chol_from_kernel, chol_kernel, eigh_kernel, lu_kernel
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.embedded import EmbeddedProfile, export, render
from hawk.ir import HawkError
from hawk.ir.linalg import factor_bytes
from hawk.trace.kernel import Kernel

LANES = 8
BAD = 3
SIZES = (1, 2, 5, 22, 32)
MODES = ("float64", "float32")
KERNELS = {"lu": lu_kernel, "chol": chol_kernel, "eigh": eigh_kernel, "chol_from": chol_from_kernel}
OUTPUTS = {"lu": {"x": 1, "X": 2, "det": 1, "info": 1},
           "chol": {"x": 1, "X": 2, "L": 1, "info": 1},
           "chol_from": {"x": 1, "X": 2, "L": 1, "info": 1},
           "eigh": {"w": 1, "V": 1, "info": 1}}
_DTYPE = {"float64": np.float64, "float32": np.float32}


def _widths(which, n):
    scale = {"x": n, "X": 2 * n, "det": 1, "info": 1, "L": n * n, "w": n, "V": n * n}
    return {name: scale[name] for name in OUTPUTS[which]}


def _matrices(which, n, rng):
    mats = []
    for lane in range(LANES):
        g = rng.standard_normal((n, n))
        if which == "chol":
            g = g @ g.T + n * np.eye(n)
            if lane == BAD:
                g[n // 2, n // 2] = -1.0
            if lane == BAD + 1 and n > 1:
                g[np.tril_indices(n, -1)] += 0.3          # not symmetric
        elif which == "chol_from":
            g = np.tril(g, -1) + np.diag(np.abs(np.diag(g)) + 0.5) + 0.7 * np.triu(g, 1)   # garbage above
            if lane == BAD:
                g[n // 2, n // 2] = -1.0
            if lane == BAD + 1:
                g[0, 0] = 0.0
            if lane in (BAD + 2, BAD + 3):
                g[n - 1, n - 1] = np.inf if lane == BAD + 2 else np.nan
        elif which == "eigh":
            g = (g + g.T) / 2 if lane != BAD else g       # lane BAD is not symmetric
        elif lane == BAD:
            g[min(1, n - 1)] = 0.0                        # an exactly zero row
        mats.append(g)
    return np.stack(mats)


def _inputs(which, n, mode):
    rng = np.random.default_rng(1000 * n + (mode == "float32"))
    dt = _DTYPE[mode]
    mats = _matrices(which, n, rng).astype(dt)
    return {"A": mats, "b": rng.standard_normal((LANES, n)).astype(dt),
            "B": rng.standard_normal((LANES, n, 2)).astype(dt)}


def _plane(a):
    plane = np.ascontiguousarray(a.reshape(LANES, -1).T)
    return plane[0] if plane.shape[0] == 1 else plane


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("la_c_host")

    def run(kernel, inputs, outs, mode, primal=None):
        kw = dict(targets=("host",), cache_dir=cache_dir, host_profile="native", mode=mode)
        extra = {}
        if primal is not None:
            home = build_bundle([primal], pathlib.Path(root) / f"{primal.name}_{mode}", **kw)
            extra = {"derivative": {kernel.name: (primal.name, home.digest)}}
        bundle = build_bundle([kernel], pathlib.Path(root) / f"{kernel.name}_{mode}",
                              **kw, **extra)
        loaded = rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
        sink = {name: np.zeros(LANES if width == 1 else (width, LANES),
                               dtype=np.int32 if name == "info" else _DTYPE[mode])
                for name, width in outs.items()}
        rt.run(loaded, **{k: _plane(v) for k, v in inputs.items()}, **sink)
        return {name: plane.reshape(outs[name], LANES) for name, plane in sink.items()}
    return run


@pytest.fixture(scope="module")
def clib(tmp_path_factory):
    root = tmp_path_factory.mktemp("la_c_unit")

    def run(kernel, inputs, outs, mode):
        art = export(kernel, root / f"{kernel.name}_{mode}", mode=mode, check=False)
        so = E.cc_shared(art.c_path, art.directory)
        written = [r["name"] for r in art.report["interface"]
                   if r["direction"] != "in" and r["role"] != "nsamples"]
        planes = {k: _plane(v) for k, v in inputs.items()
                  if k in {r["name"] for r in art.report["interface"]}}
        got = E.run_c(so, kernel.name, art.report, LANES, mode=mode, **planes)
        got = got if isinstance(got, tuple) else (got,)
        return {name: np.asarray(plane).reshape(outs[name], LANES)
                for name, plane in zip(written, got)}
    return run


def _same_bits(c, h, what):
    for name in h:
        assert c[name].dtype == h[name].dtype and c[name].tobytes() == h[name].tobytes(), (
            f"{what}: {name} differs between the C unit and the host build")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("which", sorted(KERNELS))
def test_the_c_unit_computes_the_hosts_bits(host, clib, which, n, mode):
    kernel = KERNELS[which](n)
    inputs, outs = _inputs(which, n, mode), _widths(which, n)
    _same_bits(clib(kernel, inputs, outs, mode), host(kernel, inputs, outs, mode),
               f"{kernel.name} {mode}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
@pytest.mark.parametrize("which", sorted(KERNELS))
def test_a_derivative_kernel_computes_the_hosts_bits(host, clib, which, derive, n, mode):
    primal = KERNELS[which](n)
    kernel = Kernel(f"{primal.name}_{derive.__name__}", derive(primal))
    inputs = _inputs(which, n, mode)
    rng = np.random.default_rng(7)
    dt = _DTYPE[mode]
    if derive is vjp:
        widths = {name: width for name, width in _widths(which, n).items() if name != "info"}
        outs = {s.name: int(np.prod(s.ttype.shape or (1,))) for s in kernel.sinks}
        seeds = {f"bar_{name}": rng.standard_normal((LANES, width)).astype(dt)
                 for name, width in widths.items()}
    else:
        outs = {s.name: int(np.prod(s.ttype.shape or (1,))) for s in kernel.sinks}
        seeds = {f"dot_{name}": rng.standard_normal(value.shape).astype(dt)
                 for name, value in inputs.items()}
    planes = {**inputs, **seeds}                  # the derivative reads its primal's inputs too
    _same_bits(clib(kernel, planes, outs, mode),
               host(kernel, planes, outs, mode, primal=primal), f"{kernel.name} {mode}")


# --- the toolchain: compiler under c99 and c11, cppcheck MISRA, the text audit ------

@pytest.mark.parametrize("n,mode", [(n, m) for n in (1, 2, 5) for m in MODES]
                         + [(22, "float64")])
@pytest.mark.parametrize("which", sorted(KERNELS))
def test_the_unit_passes_the_compiler_cppcheck_and_the_audit(tmp_path, which, n, mode):
    art = export(KERNELS[which](n), tmp_path, mode=mode)
    cppcheck = art.report["checks"]["cppcheck"]
    assert "skipped" not in cppcheck, "cppcheck did not run ($HAWK_CPPCHECK)"
    assert cppcheck["findings"] == [] and art.report["audit"] == {"hits": []}


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("which", sorted(KERNELS))
def test_a_derivative_passes_the_compiler_cppcheck_and_the_audit(tmp_path, which, n, derive):
    primal = KERNELS[which](n)
    art = export(Kernel(f"{primal.name}_{derive.__name__}", derive(primal)), tmp_path)
    cppcheck = art.report["checks"]["cppcheck"]
    assert "skipped" not in cppcheck and cppcheck["findings"] == []


# --- the report ----------------------------------------------------------------

def test_the_unit_names_its_helpers_and_the_singular_test():
    from hawk.embedded import finish_report

    source = render(lu_kernel(5))
    assert {"hawk_lu_n5_f64", "hawk_lu_solve_n5_v_f64", "hawk_lu_solve_n5_k2_f64",
            "hawk_lu_det_n5_f64"} <= set(source.report["helpers"])
    assert source.report["fp_equality_sites"], "the pivot test compares reals for equality"
    notes = finish_report(source, {"skipped": "not requested"})["deviations"]
    assert any(d["rule"] == "float-equality" for d in notes)


@pytest.mark.parametrize("which", sorted(KERNELS))
def test_a_dimension_of_one_is_straight_code(which):
    text = render(KERNELS[which](1)).text
    helpers = re.findall(r"/\* hawk: la_.*?\n}\n", text, re.S)
    assert helpers and not [h for h in helpers if "for (" in h]


# --- a status nobody reads --------------------------------------------------------

@hawk.kernel
def reads_status(A: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]],
                 info: Mutable[hawk.Int32]):
    x, info = m.solve(A, b)


@hawk.kernel
def drops_status(A: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]]):
    x, _ = m.solve(A, b)


def test_a_factorisation_whose_status_is_never_read_is_reported():
    assert render(reads_status).report["linalg.status_dropped"] == []
    dropped = render(drops_status).report["linalg.status_dropped"]
    assert len(dropped) == 1
    assert "linalg.status_dropped" not in render(K.drag).report


# --- the stack budget -------------------------------------------------------------

@pytest.mark.parametrize("mode,w", [("float64", 8), ("float32", 4)])
@pytest.mark.parametrize("which", sorted(KERNELS))
def test_the_stack_holds_the_factor_at_the_cost_formulas_figure(which, mode, w):
    n = 6
    kernel = KERNELS[which](n)
    report = render(kernel, mode=mode).report
    factor = next(node for node in kernel.walk.order if getattr(node, "kind", None) == which)
    held = factor_bytes(factor.ttype, w)
    assert report["budgets"]["static_bytes"]["temporaries"] >= held


@pytest.mark.parametrize("which", sorted(KERNELS))
def test_a_32_wide_f64_kernel_is_refused_under_four_kilobytes(which):
    kernel = KERNELS[which](32)
    with pytest.raises(HawkError, match=r"budget\.stack"):
        render(kernel, profile=EmbeddedProfile(max_stack_bytes=4096))
    render(KERNELS[which](5), profile=EmbeddedProfile(max_stack_bytes=8192))
