# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Functions on the device: a function is one definition in a unit's device text and a call writes its symbol over the argument texts.

Each row (the ``gpu`` mark) builds a kernel for the card, runs it and compares with NumPy; the text rows read the device text:

* a function used sixteen times is one ``__device__`` definition and sixteen calls, and gives the values of the same kernel written longhand;
* a vector argument and result, a loop inside a function, a function with supplied derivative rules (forward and reverse) run on the card;
* a native function with a device text is written between line directives and runs; one without refuses by name; a body that
  does not compile fails with the function's name and the author's line.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from test_function_calls import LANES, USES, _drag, _sixteen, _text, _vectors, drag, horner, reference
from test_function_supplied_rules import softplus_kernel, wrong_kernel
from test_function_loop_rules import uses_looped
from test_helper_defs import side

import hawk
import hawk.math as m
from hawk import HawkError, Kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp

STEPS = 5


def _device(kernel_or_kernels, directory, cache_dir, name, **planes):
    """``kernels`` built for the card and the kernel ``name`` run over ``planes`` (cupy arrays); returns the plane map."""
    kernels = kernel_or_kernels if isinstance(kernel_or_kernels, list) else [kernel_or_kernels]
    bundle = build_bundle(kernels, directory, targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
    bound = hawk.plan(hawk.load(bundle.directory, name), hawk.DeviceKernel).bind(**planes)
    bound.launch()
    bound.wait()
    return planes


def _cupy():
    return pytest.importorskip("cupy")


def test_a_function_used_sixteen_times_is_one_device_definition_and_sixteen_calls(tmp_path):
    text = _text(_sixteen(tmp_path, True), target="cuda")
    assert len(re.findall(r"^__device__ __forceinline__ \w+ hawk_fn_helper_\w+\(", text, re.M)) == 1
    assert len(re.findall(r"hawk_fn_helper_\w+\(", text)) == USES + 1
    assert text.index("hawk_fn_helper_") < text.index("__global__")


@pytest.mark.gpu
def test_the_called_helper_runs_on_the_card_and_equals_the_longhand_kernel_and_numpy(tmp_path, cache_dir):
    cupy = _cupy()
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0.5, 1.5, LANES), rng.uniform(-1.0, 1.0, LANES)
    results = {}
    for decorated in (True, False):
        (tmp_path / str(decorated)).mkdir()
        kern = _sixteen(tmp_path / str(decorated), decorated)
        planes = {"x": cupy.asarray(x), "y": cupy.asarray(y), "out": cupy.zeros(LANES)}
        results[decorated] = _device(kern, tmp_path / f"b{decorated}", cache_dir, kern.name, **planes)["out"].get()
    want = sum(reference(x * (1.0 + 0.1 * k), y + 0.05 * k) for k in range(USES))
    np.testing.assert_allclose(results[True], want, rtol=1e-12)
    np.testing.assert_allclose(results[True], results[False], rtol=1e-13)


@pytest.mark.gpu
def test_a_vector_argument_and_result_run_on_the_card_and_equal_numpy(tmp_path, cache_dir):
    cupy = _cupy()

    @hawk.kernel
    def kern(v: hawk.Vector[3], rho: hawk.Scalar, out: hawk.Mutable[hawk.Vector[3]]):
        out = drag(v, rho) + drag(v, 2.0)

    text = _text(kern, target="cuda")
    assert len(re.findall(r"^__device__ __forceinline__ aether::Item<Real, 3> hawk_fn_drag_\w+\(", text, re.M)) == 1
    v, rho = _vectors()
    planes = {"v": cupy.asarray(v), "rho": cupy.asarray(rho), "out": cupy.zeros((3, LANES))}
    got = _device(kern, tmp_path, cache_dir, "kern", **planes)["out"].get()
    np.testing.assert_allclose(got, _drag(v, rho) + _drag(v, 2.0), rtol=1e-12)


@pytest.mark.gpu
def test_a_function_whose_body_holds_a_loop_runs_on_the_card_and_equals_numpy(tmp_path, cache_dir):
    cupy = _cupy()

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = horner(x) + horner(x * 0.5)

    text = _text(kern, target="cuda")
    assert len(re.findall(r"^__device__ __forceinline__ \w+ hawk_fn_horner_\w+\(", text, re.M)) == 1
    x = np.linspace(0.1, 0.9, LANES)
    got = _device(kern, tmp_path, cache_dir, "kern", x=cupy.asarray(x), out=cupy.zeros(LANES))["out"].get()

    def series(t):
        return sum(t**j for j in range(STEPS))

    np.testing.assert_allclose(got, series(x) + series(x * 0.5), rtol=1e-13)


@pytest.mark.gpu
def test_the_supplied_rules_of_a_function_run_on_the_card_forward_and_reverse(tmp_path, cache_dir):
    cupy = _cupy()
    reverse = Kernel("softplus_kernel_vjp", vjp(softplus_kernel))
    forward = Kernel("softplus_kernel_jvp", jvp(softplus_kernel))
    x = np.random.default_rng(9).uniform(-2.0, 2.0, LANES)
    seed = np.random.default_rng(10).uniform(0.2, 1.4, LANES)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    planes = {"x": cupy.asarray(x), "bar_y": cupy.asarray(seed), "bar_x": cupy.zeros(LANES)}
    got = _device([softplus_kernel, reverse], tmp_path / "r", cache_dir, "softplus_kernel_vjp", **planes)["bar_x"].get()
    np.testing.assert_allclose(got, 2.0 * seed * sigmoid, rtol=1e-12)
    planes = {"x": cupy.asarray(x), "dot_x": cupy.asarray(seed), "dot_y": cupy.zeros(LANES)}
    got = _device([softplus_kernel, forward], tmp_path / "f", cache_dir, "softplus_kernel_jvp", **planes)["dot_y"].get()
    np.testing.assert_allclose(got, 2.0 * seed * sigmoid, rtol=1e-12)


#: A native function with a device text, and one without.
fma_sq = hawk.native(
    "fma_sq_device",
    reads={"x": hawk.Scalar, "y": hawk.Scalar},
    returns=hawk.Scalar,
    text={"host": "return std::fma(x, x, y);", "cuda": "return fma(x, x, y);"},
)
host_only = hawk.native(
    "host_only_device",
    reads={"x": hawk.Scalar},
    returns=hawk.Scalar,
    text={"host": "return x;"},
)


def test_a_native_function_with_a_device_text_is_written_between_line_directives_once(tmp_path):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(x, y) + fma_sq(y, x)

    text = _text(kern, target="cuda")
    assert len(re.findall(r"^__device__ __forceinline__ \w+ hawk_fn_fma_sq_device_\w+\(", text, re.M)) == 1
    assert '#line 1 "native function fma_sq_device"' in text
    assert "return fma(x, x, y);" in text
    assert "HAWK_RESUME_LINE" not in text


@pytest.mark.gpu
def test_a_native_function_runs_on_the_card_and_equals_numpy(tmp_path, cache_dir):
    cupy = _cupy()

    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(x, y) + fma_sq(y, x)

    rng = np.random.default_rng(11)
    x, y = rng.uniform(-1.0, 1.0, LANES), rng.uniform(-1.0, 1.0, LANES)
    planes = {"x": cupy.asarray(x), "y": cupy.asarray(y), "out": cupy.zeros(LANES)}
    got = _device(kern, tmp_path, cache_dir, "kern", **planes)["out"].get()
    np.testing.assert_allclose(got, x * x + y + y * y + x, rtol=1e-14)


def test_a_native_function_without_a_device_text_is_refused_by_name_for_the_device(tmp_path):
    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = host_only(x)

    with pytest.raises(HawkError, match=r'native function "host_only_device" has no text for target "cuda"'):
        _text(kern, target="cuda")


@pytest.mark.gpu
def test_a_device_text_that_names_a_plane_fails_to_compile_and_the_failure_names_the_function_and_the_line(
    tmp_path, cache_dir
):
    peek = hawk.native(
        "peek_device",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"cuda": "double scaled = x * 2.0;\nreturn scaled * out;"},
    )

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = peek(x)

    with pytest.raises(HawkError) as refused:
        build_bundle([kern], tmp_path, targets=("cuda",), cache_dir=cache_dir)
    message = str(refused.value)
    assert 'compile of "kern" failed' in message
    assert re.search(r"native function peek_device[:(]2[:)]", message), message


@pytest.mark.gpu
def test_a_wrong_supplied_reverse_rule_is_the_one_used_on_the_card(tmp_path, cache_dir):
    """The positive control of the parity rows, run on the card: a rule declared three times too large gives three times the seed."""
    cupy = _cupy()
    reverse = Kernel("wrong_kernel_vjp", vjp(wrong_kernel))
    x = np.linspace(-1.0, 1.0, LANES)
    seed = np.linspace(0.2, 1.4, LANES)
    planes = {"x": cupy.asarray(x), "bar_y": cupy.asarray(seed), "bar_x": cupy.zeros(LANES)}
    got = _device([wrong_kernel, reverse], tmp_path, cache_dir, "wrong_kernel_vjp", **planes)["bar_x"].get()
    np.testing.assert_allclose(got, 3.0 * seed, rtol=1e-13)


@pytest.mark.gpu
def test_the_rules_of_a_function_with_loops_run_on_the_card_and_equal_three_x(tmp_path, cache_dir):
    cupy = _cupy()
    reverse, forward = Kernel("uses_looped_vjp", vjp(uses_looped)), Kernel("uses_looped_jvp", jvp(uses_looped))
    x = np.linspace(-1.3, 2.6, LANES)
    planes = {"x": cupy.asarray(x), "bar_y": cupy.ones(LANES), "bar_x": cupy.zeros(LANES)}
    got = _device([uses_looped, reverse], tmp_path / "r", cache_dir, "uses_looped_vjp", **planes)["bar_x"].get()
    np.testing.assert_allclose(got, 3.0 * x, rtol=1e-12)
    planes = {"x": cupy.asarray(x), "dot_x": cupy.ones(LANES), "dot_y": cupy.zeros(LANES)}
    got = _device([uses_looped, forward], tmp_path / "f", cache_dir, "uses_looped_jvp", **planes)["dot_y"].get()
    np.testing.assert_allclose(got, 3.0 * x, rtol=1e-12)


@pytest.mark.gpu
def test_a_bare_function_differentiates_through_its_body_on_the_card(tmp_path, cache_dir):
    cupy = _cupy()

    @hawk.function
    def dev_softplus(x):
        return m.log(1.0 + m.exp(x))

    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = dev_softplus(x)

    reverse = Kernel("kern_vjp", vjp(kern))
    x = np.linspace(-2.0, 2.0, LANES)
    planes = {"x": cupy.asarray(x), "bar_y": cupy.ones(LANES), "bar_x": cupy.zeros(LANES)}
    got = _device([kern, reverse], tmp_path, cache_dir, "kern_vjp", **planes)["bar_x"].get()
    np.testing.assert_allclose(got, 1.0 / (1.0 + np.exp(-x)), rtol=1e-12)


@pytest.mark.gpu
def test_a_function_with_a_rule_and_a_helper_inside_it_runs_on_the_card(tmp_path, cache_dir):
    cupy = _cupy()

    @hawk.function(name="dev_branchy", vjp=lambda x, bar: 7.0 * bar)
    def branchy(x):
        return side(x)

    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = branchy(x)

    reverse = Kernel("kern_vjp", vjp(kern))
    x = np.linspace(0.1, 0.9, LANES)
    planes = {"x": cupy.asarray(x), "y": cupy.zeros(LANES)}
    got = _device(kern, tmp_path / "p", cache_dir, "kern", **planes)["y"].get()
    np.testing.assert_allclose(got, np.where(x > 0.5, x * x, x), rtol=1e-13)  # the helper inside is rewritten to a select
    planes = {"x": cupy.asarray(x), "bar_y": cupy.ones(LANES), "bar_x": cupy.zeros(LANES)}
    got = _device([kern, reverse], tmp_path / "r", cache_dir, "kern_vjp", **planes)["bar_x"].get()
    np.testing.assert_allclose(got, 7.0 * np.ones(LANES), rtol=1e-13)
