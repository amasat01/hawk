# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.function``: one definition in the host text, a call per use.

* A function used sixteen times is written once; a plain helper stays inlined.
* Values equal the same helper inlined, within a stated bound, and equal NumPy on the closed form.
* A function may call a function; a wrong count, a wrong type, an annotation that is no payload type and recursion are refused by name.
* The device text refuses a call by name.
"""

from __future__ import annotations

import importlib.util
import re

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import HawkError
from hawk.artifact import build_bundle, emit_source

USES = 16
LANES = 64

#: The helper, large enough that its text matters: products, a square root and a branch-free select.
HELPER = """
def helper(a: hawk.Scalar, b: hawk.Scalar) -> hawk.Scalar:
    s = m.sqrt(a * a + b * b + 1.0)
    r = a * s - b * 0.25 + s * 0.5
    return r if r > 0.0 else -r * 0.5
"""


@hawk.function
def blend_sqrt(a: hawk.Scalar, b: hawk.Scalar) -> hawk.Scalar:
    """The helper as a function, for the tests that call it from a kernel written in this file."""
    s = m.sqrt(a * a + b * b + 1.0)
    r = a * s - b * 0.25 + s * 0.5
    return r if r > 0.0 else -r * 0.5


def reference(a, b):
    """The closed form of the helper, in NumPy."""
    s = np.sqrt(a * a + b * b + 1.0)
    r = a * s - b * 0.25 + s * 0.5
    return np.where(r > 0.0, r, -r * 0.5)


def _sixteen(tmp_path, decorated: bool):
    """A kernel with sixteen written-out uses of the helper (a ``for`` would be one loop and one use), decorated or not.

    The kernel is written to a module, because the tracer reads a kernel's own text."""
    uses = "\n".join(
        f"    acc = acc + helper(x * {1.0 + 0.1 * k!r}, y + {0.05 * k!r})"
        for k in range(USES)
    )
    head = "@hawk.function\n" if decorated else ""
    source = (
        "import hawk\nimport hawk.math as m\n\n"
        + head
        + HELPER.lstrip("\n")
        + "\n\n@hawk.kernel\ndef kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):\n"
        f"    acc = 0.0\n{uses}\n    out = acc\n"
    )
    path = tmp_path / f"kern_{'called' if decorated else 'inlined'}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kern


def _text(kernel, target="host"):
    return emit_source(kernel, targets=(target,)).sources[target].text


def _run(kernel, cache_dir, x, y):
    out = np.zeros_like(x)
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(x=x, y=y, out=out)
    return out


def _inputs():
    rng = np.random.default_rng(3)
    return rng.uniform(0.5, 1.5, LANES), rng.uniform(-1.0, 1.0, LANES)


def _ulps(a, b):
    """The distance in units in the last place between two float64 arrays of one sign."""
    return np.abs(a.view(np.int64) - b.view(np.int64))


def test_a_function_used_sixteen_times_is_defined_once_and_called_sixteen_times(
    tmp_path,
):
    text = _text(_sixteen(tmp_path, True))
    assert len(re.findall(r"^static inline Real hawk_fn_helper_\w+\(", text, re.M)) == 1
    assert len(re.findall(r"hawk_fn_helper_\w+\(", text)) == USES + 1


def test_a_plain_helper_stays_inlined(tmp_path):
    assert "hawk_fn_" not in _text(_sixteen(tmp_path, False))


def test_the_called_text_is_smaller_than_the_inlined_text(tmp_path):
    assert len(_text(_sixteen(tmp_path, True))) < 0.75 * len(
        _text(_sixteen(tmp_path, False))
    )


def _run_built(kernel, directory, cache_dir, profile, x, y):
    """``kernel`` built for the host under ``profile`` ("native" contracts nothing; "" is the default, which fuses
    multiply-adds) and run over ``x`` and ``y``."""
    bundle = build_bundle(
        [kernel],
        directory,
        targets=("host",),
        cache_dir=cache_dir,
        host_profile=profile,
    )
    out = np.zeros_like(x)
    bound = hawk.plan(hawk.load(bundle.directory, kernel.name)).bind(x=x, y=y, out=out)
    bound.launch()
    bound.wait()
    return out


def test_the_called_helper_equals_the_inlined_one_bit_for_bit_without_contraction(
    tmp_path, cache_dir
):
    x, y = _inputs()
    called = _run_built(
        _sixteen(tmp_path, True), tmp_path / "called", cache_dir, "native", x, y
    )
    inlined = _run_built(
        _sixteen(tmp_path, False), tmp_path / "inlined", cache_dir, "native", x, y
    )
    # with no multiply-add fused, identical operations round identically whether or not the call is kept
    assert np.array_equal(called, inlined)


def test_the_called_helper_stays_within_the_fusion_bound_of_the_inlined_one_and_equals_numpy(
    tmp_path, cache_dir
):
    """Under the default profile a multiply-add the compiler fuses in one text and not in the other is the only difference.

    The helper has three products that are not exact: ``a * a`` or ``b * b`` (one of them is fused into the radicand) and
    ``a * s`` (fused into the difference with ``b * 0.25``); ``b * 0.25`` and ``s * 0.5`` are exact. Fusing leaves a product
    unrounded, so it changes a sum by at most ``u`` times that product, ``u = 2**-53``. The radicand changes by at most
    ``u (a^2 + b^2)``; the square root ``s`` by at most half of that over ``s``; the result ``r`` reads ``s`` with the
    coefficient ``a + 0.5``, and the select has slope at most one. One use therefore differs by at most
    ``u ((a^2 + b^2) |a + 0.5| / (2 s) + |a| s)``. The sum adds the sixteen differences, and each of its sixteen additions may
    round one unit in the last place (``2 u |sum|``) differently because its operand moved. The bound is computed per lane
    from the inputs; the value must also equal the closed form in NumPy, which is the contraction-free arithmetic.
    """
    x, y = _inputs()
    called = _run_built(
        _sixteen(tmp_path, True), tmp_path / "called", cache_dir, "", x, y
    )
    inlined = _run_built(
        _sixteen(tmp_path, False), tmp_path / "inlined", cache_dir, "", x, y
    )
    u = 2.0**-53
    bound = np.zeros_like(x)
    want = np.zeros_like(x)
    for k in range(USES):
        a, b = x * (1.0 + 0.1 * k), y + 0.05 * k
        s = np.sqrt(a * a + b * b + 1.0)
        bound += u * ((a * a + b * b) * np.abs(a + 0.5) / (2.0 * s) + np.abs(a) * s)
        want += reference(a, b)
    bound += USES * 2.0 * u * np.abs(want)
    assert np.all(np.abs(called - inlined) <= bound)
    np.testing.assert_allclose(called, want, rtol=1e-13)


@hawk.function
def scaled(a: hawk.Scalar, b: hawk.Scalar) -> hawk.Scalar:
    return blend_sqrt(a, b) * 2.0 + 1.0


def test_a_function_may_call_a_function_and_each_is_defined_once(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = scaled(x, y) + scaled(y, x)

    text = _text(kern)
    assert len(re.findall(r"^static inline Real hawk_fn_", text, re.M)) == 2
    assert text.index("hawk_fn_blend_sqrt_") < text.index("hawk_fn_scaled_")
    x, y = _inputs()
    out = _run(kern, cache_dir, x, y)
    np.testing.assert_allclose(
        out, reference(x, y) * 2 + 1 + reference(y, x) * 2 + 1, rtol=1e-13
    )


def test_a_wrong_argument_count_is_refused_by_the_function_name():
    with pytest.raises(HawkError, match="function 'blend_sqrt' takes 2 argument"):

        @hawk.kernel
        def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = blend_sqrt(x)


def test_a_wrong_argument_type_is_refused_by_the_function_name():
    with pytest.raises(HawkError, match="call of function 'blend_sqrt': argument 0"):

        @hawk.kernel
        def kern(v: hawk.Vector[3], y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = blend_sqrt(v, y)


def test_an_annotation_that_is_not_a_payload_type_is_refused_by_name():
    with pytest.raises(
        HawkError, match="function 'bare': parameter 'a' needs a payload type"
    ):

        @hawk.function
        def bare(a: int):
            return a


def test_a_result_that_differs_from_the_declared_one_is_refused_by_name():
    @hawk.function
    def liar(a: hawk.Scalar) -> hawk.Vector[3]:
        return a * 2.0

    with pytest.raises(HawkError, match="function 'liar' is declared to return"):

        @hawk.kernel
        def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = liar(x)


def test_a_function_that_calls_itself_is_refused_by_name():
    @hawk.function
    def again(a: hawk.Scalar) -> hawk.Scalar:
        return again(a) + 1.0

    with pytest.raises(HawkError, match="function 'again' calls itself"):

        @hawk.kernel
        def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = again(x)


def test_the_device_text_writes_a_call_as_a_force_inlined_definition_and_the_symbol_over_the_arguments(tmp_path):
    """RESTATED (the device refused a call by name; it now writes the function once, force-inlined, and the call over its arguments)."""
    text = _text(_sixteen(tmp_path, True), target="cuda")
    assert len(re.findall(r"^__device__ __forceinline__ \w+ hawk_fn_helper_\w+\(", text, re.M)) == 1
    assert len(re.findall(r"hawk_fn_helper_\w+\(", text)) == USES + 1


@hawk.function
def drag(v: hawk.Vector[3], rho: hawk.Scalar) -> hawk.Vector[3]:
    return v * (rho * m.sqrt(m.dot(v, v)))


@hawk.function
def trace2(a: hawk.Matrix[2, 2]) -> hawk.Scalar:
    return a[0, 0] + a[1, 1]


@hawk.function
def twice_drag(v: hawk.Vector[3], rho: hawk.Scalar) -> hawk.Vector[3]:
    return drag(drag(v, rho), rho)


def _drag(v, rho):
    """NumPy closed form of ``drag`` over the lanes of ``v`` (3, N)."""
    return v * (rho * np.sqrt((v * v).sum(axis=0)))


def _run_planes(kernel, cache_dir, out_shape, **planes):
    out = np.zeros(out_shape)
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(out=out, **planes)
    return out


def _vectors():
    rng = np.random.default_rng(5)
    return rng.uniform(-1.0, 1.0, (3, LANES)), rng.uniform(0.5, 1.5, LANES)


def test_a_vector_argument_and_a_vector_result_are_one_definition_and_equal_numpy(
    cache_dir,
):
    @hawk.kernel
    def kern(v: hawk.Vector[3], rho: hawk.Scalar, out: hawk.Mutable[hawk.Vector[3]]):
        out = drag(v, rho) + drag(v, 2.0)

    text = _text(kern)
    assert (
        len(
            re.findall(
                r"^static inline aether::Item<Real, 3> hawk_fn_drag_\w+\(const aether::Item<Real, 3>& v, ",
                text,
                re.M,
            )
        )
        == 1
    )
    v, rho = _vectors()
    got = _run_planes(kern, cache_dir, (3, LANES), v=v, rho=rho)
    np.testing.assert_allclose(got, _drag(v, rho) + _drag(v, 2.0), rtol=1e-13)


def test_a_matrix_argument_is_one_definition_and_equals_numpy(cache_dir):
    @hawk.kernel
    def kern(a: hawk.Matrix[2, 2], out: hawk.Mutable[hawk.Scalar]):
        out = trace2(a) * 2.0 + trace2(a)

    assert (
        len(
            re.findall(
                r"^static inline Real hawk_fn_trace2_\w+\(const aether::Item<Real, 2, 2>& a\)",
                _text(kern),
                re.M,
            )
        )
        == 1
    )
    a = np.random.default_rng(6).uniform(
        -1.0, 1.0, (4, LANES)
    )  # a matrix plane is its components in row order
    got = _run_planes(kern, cache_dir, (LANES,), a=a)
    np.testing.assert_allclose(got, 3.0 * (a[0] + a[3]), rtol=1e-13)


def test_a_function_calling_a_function_over_vectors_defines_each_once_and_equals_numpy(
    cache_dir,
):
    @hawk.kernel
    def kern(v: hawk.Vector[3], rho: hawk.Scalar, out: hawk.Mutable[hawk.Vector[3]]):
        out = twice_drag(v, rho) + twice_drag(v, 2.0)

    text = _text(kern)
    assert (
        len(re.findall(r"^static inline aether::Item<Real, 3> hawk_fn_", text, re.M))
        == 2
    )
    assert text.index("hawk_fn_drag_") < text.index("hawk_fn_twice_drag_")
    v, rho = _vectors()
    got = _run_planes(kern, cache_dir, (3, LANES), v=v, rho=rho)
    want = _drag(_drag(v, rho), rho) + _drag(_drag(v, 2.0), 2.0)
    np.testing.assert_allclose(got, want, rtol=1e-12)


STEPS = 5


@hawk.function
def horner(x: hawk.Scalar) -> hawk.Scalar:
    acc = 0.0
    for _ in range(STEPS):
        acc = acc * x + 1.0
    return acc


def test_a_function_whose_body_holds_a_loop_is_one_definition_with_the_loop_and_equals_numpy(
    cache_dir,
):
    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = horner(x) + horner(x * 0.5)

    text = _text(kern)
    assert len(re.findall(r"^static inline Real hawk_fn_horner_\w+\(", text, re.M)) == 1
    definition = text[text.index("static inline Real hawk_fn_horner_") :]
    assert (
        "for (Int" in definition.split("extern")[0]
    )  # the loop is in the function, not unrolled
    x = np.linspace(0.1, 0.9, LANES)
    got = _run_planes(kern, cache_dir, (LANES,), x=x)

    def series(t):
        return sum(t**j for j in range(STEPS))

    np.testing.assert_allclose(got, series(x) + series(x * 0.5), rtol=1e-13)


def test_a_banded_kernel_that_calls_a_function_is_refused_by_the_host_and_written_by_the_device():
    """RESTATED. Banded runs on device paths only: the host refuses every banded kernel, and the device now writes the function."""

    @hawk.function
    def square_plus_one(a: hawk.Scalar) -> hawk.Scalar:
        return a * a + 1.0

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = square_plus_one(x) + square_plus_one(x * 2.0)

    with pytest.raises(HawkError, match="banded runs on device execution paths"):
        emit_source(kern, mode="banded", targets=("host",))
    text = emit_source(kern, mode="banded", targets=("cuda",)).sources["cuda"].text
    assert len(re.findall(r"^__device__ __forceinline__ \w+ hawk_fn_square_plus_one_\w+\(", text, re.M)) == 1
