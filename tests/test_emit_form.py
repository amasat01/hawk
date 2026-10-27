# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``emit_form``: how a product, dot, sum or norm is spelled, and where a library result goes.

``auto`` unrolls a site of at most 16 elements and writes a longer one as a loop
(``aether::rt``), ``static`` unrolls every site, ``rolled`` writes every site as a loop.
All three compute the same bits (host rows here, device rows are run by hand on nvcc and
NVRTC). A library result an own-column store takes unchanged is written into the plane
through its proxy; the resource report puts a kernel's frame beside its working set.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Scalar, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.artifact.bundle import _resource_report
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from hawk.emit import render_source
from hawk.emit import policy
from hawk.emit.wrapper import scalar_mode
from hawk.ir.cost import working_set_bytes
from hawk.ir import HawkError
from hawk.targets import TARGETS
from hawk.trace.kernel import Kernel

LANES = 12


def reductions(n):
    @hawk.kernel
    def k_red(a: Vector[n], b: Vector[n], s: Mutable[Scalar], r: Mutable[Scalar], t: Mutable[Scalar]):
        s = m.dot(a, b)
        r = m.norm(a)
        t = m.sum(b)
    return k_red


def product(r, k):
    @hawk.kernel
    def k_mv(A: Matrix[r, k], v: Vector[k], y: Mutable[Vector[r]]):
        y = A @ v
    return k_mv


def solve(n):
    @hawk.kernel
    def k_solve(A: Matrix[n, n], b: Vector[n], x: Mutable[Vector[n]], i_x: Mutable[hawk.Int32]):
        x, i_x = m.solve(A, b)
    return k_solve


def solve_in_place(n):
    @hawk.kernel
    def k_inplace(A: Matrix[n, n], x: Mutable[Vector[n]]):
        x, _ = m.solve(A, x)
    return k_inplace


def eigen(n):
    @hawk.kernel
    def k_eigh(A: Matrix[n, n], w: Mutable[Vector[n]], V: Mutable[Matrix[n, n]]):
        w, V, _ = m.eigh(A)
    return k_eigh


def _text(kernel, form="auto", target="cuda"):
    with policy.using(form):
        return render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                             mode=scalar_mode("float64"), exports=exports(target)).text


def test_the_three_forms_spell_a_reduction_three_ways():
    k = reductions(20)
    assert _text(k, "auto").count("aether::rt::") == 3 and _text(k, "rolled").count("aether::rt::") == 3
    assert "aether::rt::" not in _text(k, "static")
    assert "aether::rt::" not in _text(reductions(16), "auto")
    assert _text(reductions(16), "rolled").count("aether::rt::") == 3


def test_rolled_is_every_site_and_static_is_none():
    assert "aether::rt::matVecItems(" in _text(product(3, 3), "rolled")
    assert "aether::rt::" not in _text(product(32, 32), "static")


def test_an_unknown_form_is_refused_by_name(tmp_path):
    with pytest.raises(HawkError, match="emit_form must be one of"):
        build_bundle([product(3, 3)], tmp_path, targets=("host",), emit_form="fast")


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_kernel_goes_through_the_same_policy(derive):
    k = product(20, 20)
    d = Kernel(f"{k.name}_d", derive(k))
    text = _text(d)
    assert "aether::rt::matVecItems(" in text
    assert "aether::rt::" not in _text(d, "static")


def test_the_forms_compute_the_same_bits_on_the_host(tmp_path, cache_dir):
    n = 20
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, LANES, n)) * np.exp(rng.uniform(-8, 8, (2, LANES, n)))
    a[0, 3], a[1, :], b[2, 0], b[4, :2] = np.nan, -0.0, np.inf, (-0.0, -np.inf)
    got = {}
    for form in ("static", "rolled"):
        bundle = build_bundle([reductions(n)], pathlib.Path(tmp_path) / form, targets=("host",), cache_dir=cache_dir,
                              host_profile="native", emit_form=form)
        loaded = rt.load(bundle.directory, "k_red", sidecar_of(bundle, "k_red"))
        out = {k: np.zeros(LANES) for k in "srt"}
        rt.run(loaded, a=np.ascontiguousarray(a.T), b=np.ascontiguousarray(b.T), **out)
        got[form] = out
    for k in "srt":
        s_, r_ = got["static"][k], got["rolled"][k]
        both_nan = np.isnan(s_) & np.isnan(r_)
        assert (s_.view(np.uint64) == r_.view(np.uint64))[~both_nan].all(), k


def test_a_solve_is_written_into_the_plane_it_is_stored_to():
    text = _text(solve(5))
    assert "aether::linalg::luSolve<1>(hawk_lf" in text and ", mut_x[i]);" in text
    assert "mut_x[i] = " not in text and "aether::Item<Real, 5> hawk_la" not in text


def test_an_eigh_result_is_written_into_both_planes():
    text = _text(eigen(6))
    assert "mut_w[i].at(hawk_lk) = " in text and "mut_V[i].at(hawk_lk / 6, hawk_lk % 6) = " in text


def test_a_plane_read_as_prior_keeps_the_result_in_a_local():
    text = _text(solve_in_place(5))
    assert "mut_x[i] = hawk_la" in text


def test_the_working_set_counts_the_factor_and_the_result_of_a_library_call():
    assert working_set_bytes(solve(8).sinks) == (8 * 8 * 8 + 4 * 8 + 4) + 8 * 8
    assert working_set_bytes(reductions(20).sinks) == 0


def test_a_frame_past_the_working_set_is_reported_and_warned():
    k = solve(8)
    ws = working_set_bytes(k.sinks)
    fine = {"k": {"regs": 32, "stack_B": ws, "spill_store_B": 0, "spill_load_B": 0}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resource_report(k, fine)["k"]["spills"] is False
    for row in ({"stack_B": ws, "spill_store_B": 16}, {"stack_B": 2 * ws, "spill_store_B": 0}):
        with pytest.warns(UserWarning, match="local frame"):
            report = _resource_report(k, {"k": {"regs": 255, "spill_load_B": 0, **row}})
        assert report["k"]["spills"] is True and report["k"]["working_set_B"] == ws
