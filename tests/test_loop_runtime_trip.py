# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A ``for`` bounded by a run-time value: ``for k in range(n, max=K)``.

``n`` is a rank-0 integer index (``Index``, or ``hawk.math.index(x, K)``), per
sample. The loop is the ordinary lowered ``for`` whose header reads the count,
``n`` held to ``[0, K]``: never unrolled, the same text on every target. Without
``max=`` the bound is the int range, so forward and ``jvp`` build (they keep no
tape) and reverse mode refuses by name; with ``max=K`` the tape is ``K`` long and
``vjp`` runs the last ``n`` iterations back.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
from hawk import Index, Mutable, Scalar
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import HawkError, Loop
from hawk.trace.kernel import Kernel

K = 6


@hawk.kernel
def powers(x: Scalar, n: Index, y: Mutable[Scalar]):
    s = 1.0
    for k in range(n, max=K):
        s = s * x + 0.5
    y = s


@hawk.kernel
def powers_open(x: Scalar, n: Index, y: Mutable[Scalar]):
    s = 1.0
    for k in range(n):
        s = s * x + 0.5
    y = s


def _ref(x, n, top=K):
    s = 1.0
    for _ in range(max(0, min(n, top))):
        s = s * x + 0.5
    return s


def _loop(kernel):
    return next(n for n in kernel.walk.order if isinstance(n, Loop))


@pytest.mark.parametrize("n", [-3, 0, 1, 4, K, K + 5])
def test_the_count_is_held_to_zero_and_max(n):
    got = evaluate(powers.sinks, {"x": 0.7, "n": float(n), "terminated": False}, lanes=1)["y"]
    assert abs(got - _ref(0.7, n)) < 1e-14


def test_the_loop_says_how_it_is_bounded_and_is_never_unrolled():
    assert _loop(powers).dynamic == "max" and _loop(powers).trip == K
    assert _loop(powers_open).dynamic == "open"
    for kernel in (powers, powers_open):
        text = render_body(kernel.sinks, kernel.walk).text
        assert "for (int hawk_k" in text and "hawk_clamp_count(" in text


def test_max_enters_the_walk_digest():
    @hawk.kernel
    def powers7(x: Scalar, n: Index, y: Mutable[Scalar]):
        s = 1.0
        for k in range(n, max=K + 1):
            s = s * x + 0.5
        y = s

    assert powers7.walk.digest != powers.walk.digest != powers_open.walk.digest


def test_the_jvp_of_a_loop_without_max_builds_and_matches_central_differences(tmp_path, cache_dir):
    """The evaluator would walk the int range; the compiled loop runs the count."""
    tangent = Kernel("powers_jvp", jvp(powers_open, wrt=("x",)), {})
    bundle = build_bundle([powers_open, tangent], tmp_path, targets=("host",), cache_dir=cache_dir,
                          host_profile="native", derivative={"powers_jvp": "powers_open"})
    run = rt.load(bundle.directory, "powers_jvp", sidecar_of(bundle, "powers_jvp"))
    counts = np.array([0, 3, 9, 40], dtype=np.int64)
    x, v = np.full(4, 0.7), np.ones(4)
    y, dot_y = np.zeros(4), np.zeros(4)
    rt.run(run, x=x, dot_x=v, n=counts, dot_y=dot_y)
    rt.run(rt.load(bundle.directory, "powers_open", sidecar_of(bundle, "powers_open")),
           x=x, n=counts, y=y)
    h = 1e-6
    want = [(_ref(0.7 + h, int(n), 99) - _ref(0.7 - h, int(n), 99)) / (2 * h) for n in counts]
    np.testing.assert_allclose(dot_y, want, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(y, [_ref(0.7, int(n), 99) for n in counts], rtol=1e-13)


def test_the_vjp_with_max_matches_central_differences():
    d = vjp(powers, wrt=("x",))
    for n in (0, 2, K, K + 3):
        got = evaluate(d, {"x": 0.7, "n": float(n), "terminated": False, "bar_y": 1.0},
                       lanes=1)["bar_x"]
        want = (_ref(0.7 + 1e-6, n) - _ref(0.7 - 1e-6, n)) / 2e-6
        assert abs(got - want) < 1e-7


def test_the_vjp_without_max_is_refused_by_name():
    with pytest.raises(HawkError, match="give `max=` so the tape has a size"):
        vjp(powers_open, wrt=("x",))


def test_the_bound_must_be_an_index_and_max_a_compile_time_int():
    with pytest.raises(HawkError, match="rank-0 integer index"):
        @hawk.kernel
        def bad(x: Scalar, y: Mutable[Scalar]):
            s = 1.0
            for k in range(x):
                s = s * x
            y = s

    with pytest.raises(HawkError, match="already a compile-time one"):
        @hawk.kernel
        def static_with_max(x: Scalar, y: Mutable[Scalar]):
            s = 1.0
            for k in range(3, max=4):
                s = s * x
            y = s

    with pytest.raises(HawkError, match="`max=` is a compile-time int >= 1"):
        @hawk.kernel
        def zero_max(x: Scalar, n: Index, y: Mutable[Scalar]):
            s = 1.0
            for k in range(n, max=0):
                s = s * x
            y = s


def _header_lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("for (")]


def test_the_header_is_the_count_form_on_cuda_host_and_c():
    from hawk.emit import FLOAT64, render_source
    from hawk.embedded import render as render_c
    from hawk.targets import TARGETS

    for target in ("cuda", "host"):
        src = render_source("powers", powers.sinks, powers.walk, TARGETS[target], mode=FLOAT64,
                            kind=powers.kind)
        heads = _header_lines(src.body)
        assert len(heads) == 1 and "hawk_clamp_count(" in heads[0], heads
        assert str(K) not in heads[0] or "hawk_clamp_count" in heads[0]
    c_text = render_c(powers, mode="float64").text
    heads = [h for h in _header_lines(c_text) if "hawk_k" in h]       # past the sample loop
    assert len(heads) == 1 and "hawk_cnt" in heads[0], heads
    assert "clamp_trips" in c_text and f"INT32_C({K})" in c_text


@pytest.fixture(scope="module")
def reverse_bundle(tmp_path_factory, cache_dir):
    adjoint = Kernel("powers_vjp", vjp(powers, wrt=("x",)), {})
    return build_bundle([powers, adjoint], tmp_path_factory.mktemp("rev"), targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native",
                        derivative={"powers_vjp": "powers"})


COUNTS = np.array([0, 2, K, K + 3], dtype=np.int64)


def _want():
    h = 1e-6
    return [(_ref(0.7 + h, int(n)) - _ref(0.7 - h, int(n))) / (2 * h) for n in COUNTS]


def test_the_reverse_loop_runs_the_last_n_iterations_of_the_tape_on_the_host(reverse_bundle):
    bar = np.zeros(4)
    rt.run(rt.load(reverse_bundle.directory, "powers_vjp", sidecar_of(reverse_bundle, "powers_vjp")),
           x=np.full(4, 0.7), n=COUNTS, bar_y=np.ones(4), bar_x=bar)
    np.testing.assert_allclose(bar, _want(), rtol=1e-6, atol=1e-7)


@pytest.mark.gpu
def test_the_reverse_loop_runs_the_last_n_iterations_of_the_tape_on_the_device(reverse_bundle):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    bar = cp.zeros(4)
    planes = {"x": cp.full(4, 0.7), "n": cp.asarray(COUNTS), "bar_y": cp.ones(4), "bar_x": bar}
    plugin = L.device_plugin(reverse_bundle.directory, "powers_vjp",
                             sidecar_of(reverse_bundle, "powers_vjp"))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes).launch()
    cp.cuda.Device().synchronize()
    np.testing.assert_allclose(bar.get(), _want(), rtol=1e-6, atol=1e-7)
