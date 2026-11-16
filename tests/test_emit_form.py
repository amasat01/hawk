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
from _specs import spec_for, default_kind
from _specs import DEVICE

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
from ospreys.cuda.backend import _resource_report
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from aether.wings.base.wrapper import scalar_mode
from raptor.errors import RaptorError as HawkError
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


def test_an_unknown_form_is_refused_by_name(tmp_path):
    with pytest.raises(HawkError, match="emit_form must be one of"):
        build_bundle([product(3, 3)], tmp_path, targets=("host",), emit_form="fast")


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


