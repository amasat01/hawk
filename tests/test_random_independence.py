# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Independence of hawk's ``(seed, counter)`` draws.

aether gives every ``(seed, global, counter)`` its own Philox block, and the
emitted draw XORs a per-kind salt into the seed, so consecutive counters (and
a uniform and a normal on the same pair) are unrelated. The rows here fail on
the old word-offset positioning, where a 12-step geometric Asian call priced
off consecutive counters sat 3.8 SE from its exact discrete closed form.
"""

from __future__ import annotations

import math
import os
import subprocess
import textwrap

import numpy as np
import pytest
import _oracle as O

import hawk
import hawk.math as m
from hawk.artifact import build

N = 1 << 21


def _plane(n: int, value) -> np.ndarray:
    return np.full(n, value, dtype=np.int64)


@hawk.kernel
def asian12(seed: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    # 12-step geometric Asian call, S0 = K = 100, r = 5 %, sigma = 20 %, T = 1.
    drift = (0.05 - 0.5 * 0.2 * 0.2) / 12.0
    vol = 0.2 * math.sqrt(1.0 / 12.0)
    ln_s = math.log(100.0) + drift + vol * m.random.normal(seed, 0)
    acc = ln_s
    for j in range(1, 12):
        ln_s = ln_s + drift + vol * m.random.normal(seed, j)
        acc = acc + ln_s
    y = math.exp(-0.05) * m.max(m.exp(acc / 12.0) - 100.0, 0.0)


@hawk.kernel
def uniform_and_normal(seed: hawk.Index, counter: hawk.Index,
                       u: hawk.Mutable[hawk.Scalar], z: hawk.Mutable[hawk.Scalar]):
    u = m.random.uniform(seed, counter)
    z = m.random.normal(seed, counter)


@hawk.kernel
def mvn_identity(seed: hawk.Index, counter: hawk.Index, chol: hawk.Matrix[3, 3],
                 y: hawk.Mutable[hawk.Vector[3]]):
    y = m.random.multivariate_normal(seed, counter, m.vec(0.0, 0.0, 0.0), chol)


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _asian_exact() -> float:
    n, s0, k, r, sg, t = 12, 100.0, 100.0, 0.05, 0.2, 1.0
    mu = math.log(s0) + (r - 0.5 * sg * sg) * t * (n + 1) / (2.0 * n)
    s = sg * math.sqrt(t * (n + 1) * (2.0 * n + 1) / (6.0 * n * n))
    d2 = (mu - math.log(k)) / s
    return math.exp(-r * t) * (math.exp(mu + 0.5 * s * s) * _phi(d2 + s) - k * _phi(d2))


def test_geometric_asian_with_consecutive_counters_matches_its_closed_form(tmp_path, cache_dir):
    build(asian12, tmp_path / "asian", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "asian", "asian12")
    y = np.asarray(O.run_kernel(loaded, N, **O.planes(loaded, N, {"seed": _plane(N, 7)})),
                   dtype=np.float64)
    mean, se = y.mean(), y.std(ddof=1) / math.sqrt(N)
    exact = _asian_exact()
    assert abs(mean - exact) < 4.0 * se, f"MC {mean} vs exact {exact}: {(mean - exact) / se:.2f} SE"


def test_uniform_and_normal_on_the_same_pair_are_unrelated(tmp_path, cache_dir):
    build(uniform_and_normal, tmp_path / "un", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "un", "uniform_and_normal")
    u, z = (np.asarray(p, dtype=np.float64) for p in O.run_kernel(
        loaded, N, **O.planes(loaded, N, {"seed": _plane(N, 11), "counter": _plane(N, 5)})))
    assert u.min() > 0.0 and u.max() < 1.0
    # Independent: E[z^2 * (-2 ln u)] = 1 * 2. A shared radius word would give 4.
    q = z * z * (-2.0 * np.log(u))
    se = q.std(ddof=1) / math.sqrt(N)
    assert abs(q.mean() - 2.0) < 5.0 * se, f"{q.mean()} +- {se}"


def test_mvn_components_0_and_2_have_an_independent_fourth_moment(tmp_path, cache_dir):
    build(mvn_identity, tmp_path / "mvn", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "mvn", "mvn_identity")
    chol = np.repeat(np.eye(3).reshape(9, 1), N, axis=1)
    y = np.asarray(O.run_kernel(loaded, N, **O.planes(
        loaded, N, {"seed": _plane(N, 11), "counter": _plane(N, 0), "chol": chol})),
        dtype=np.float64)
    y = y.reshape(3, N) if y.shape[0] == 3 else y.reshape(N, 3).T
    q = y[0] ** 2 * y[2] ** 2
    se = q.std(ddof=1) / math.sqrt(N)
    assert abs(q.mean() - 1.0) < 5.0 * se, f"{q.mean()} +- {se}"


# --------------------------------------------------------------------------- #
# Boundary words: hawk's exponential and uniform_int formulas, fed the extreme
# uniforms aether can return (words 0 and 0xFFFFFFFF), stay finite / in range.
# --------------------------------------------------------------------------- #
_AETHER = os.environ.get("HAWK_AETHER_INCLUDE", "")


@pytest.mark.skipif(not os.path.isfile(_AETHER + "/aether/random/detail/Backend.h"),
                    reason="aether headers not on this box")
def test_boundary_words_keep_exponential_finite_and_uniform_int_in_range(tmp_path):
    src = tmp_path / "edge.cpp"
    src.write_text(textwrap.dedent("""
        #include <cstdio>
        #include <initializer_list>
        #include "aether/random/detail/Backend.h"
        int main() {
            using namespace aether::random::detail;
            for (unsigned w : {0u, 0xFFFFFFFFu})
                std::printf("%.17g %.9g\\n", uniformDoubleFromWord(w), (double)uniformFloatFromWord(w));
        }
    """))
    exe = tmp_path / "edge"
    subprocess.run(["g++", "-std=c++23", "-DAETHER_CPP_MODE=1", f"-I{_AETHER}", str(src), "-o", str(exe)],
                   check=True)
    rows = [tuple(map(float, ln.split())) for ln in subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True).stdout.splitlines()]
    for column in (0, 1):
        for row in rows:
            u = row[column]
            assert 0.0 < u < 1.0
            assert math.isfinite(-math.log(1.0 - u) / 2.0)  # random_exponential(rate=2)
            assert -2 <= math.floor(u * (5 - -2 + 1) + -2) <= 5  # random_uniform_int(-2, 5)
