# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.native``: a function whose body is the author's text per target, written once and called.

* Values equal NumPy on the closed form; a function used sixteen times is one definition in the text.
* A target the function has no text for is refused by the function's and the target's names.
* A text that names anything but its parameters fails to compile, and the failure names the function.
* A native function and a traced one may call each other's results in one kernel.
* The device text refuses a native function by name.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import hawk
import hawk.diff
from hawk import HawkError
from hawk.artifact import emit_source

USES = 16
LANES = 64

fma_sq = hawk.native(
    "fma_sq",
    reads={"x": hawk.Scalar, "y": hawk.Scalar},
    returns=hawk.Scalar,
    text={"host": "return std::fma(x, x, y);", "cuda": "return fma(x, x, y);"},
)

#: The squared length of a vector, read component by component off the item the signature passes by reference.
length_sq = hawk.native(
    "length_sq",
    reads={"v": hawk.Vector[3]},
    returns=hawk.Scalar,
    text={"host": "return v(0) * v(0) + v(1) * v(1) + v(2) * v(2);"},
)


def _text(kernel, target="host"):
    return emit_source(kernel, targets=(target,)).sources[target].text


def _run(kernel, cache_dir, out_shape=(LANES,), **planes):
    out = np.zeros(out_shape)
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(out=out, **planes)
    return out


def _xy():
    rng = np.random.default_rng(7)
    return rng.uniform(0.5, 1.5, LANES), rng.uniform(-1.0, 1.0, LANES)


def test_a_native_function_equals_numpy_on_its_closed_form(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(x, y)

    x, y = _xy()
    # the oracle is x * x + y in float64, rounded twice; the text rounds once (a fused multiply-add). Each rounding is off by at
    # most u times the value it rounds, u = 2**-53: the product once, the sum twice (once per side), so the two differ by at most
    # u * (x * x + 2 * |x * x + y|)
    want = x * x + y
    bound = 2.0**-53 * (x * x + 2.0 * np.abs(want))
    assert np.all(np.abs(_run(kern, cache_dir, x=x, y=y) - want) <= bound)


def test_a_vector_parameter_is_passed_by_reference_and_equals_numpy(cache_dir):
    @hawk.kernel
    def kern(v: hawk.Vector[3], out: hawk.Mutable[hawk.Scalar]):
        out = length_sq(v) + length_sq(v * 2.0)

    v = np.random.default_rng(8).uniform(-1.0, 1.0, (3, LANES))
    text = _text(kern)
    assert re.search(
        r"^static inline Real hawk_fn_length_sq_\w+\(const aether::Item<Real, 3>& v\)",
        text,
        re.M,
    )
    np.testing.assert_allclose(
        _run(kern, cache_dir, v=v), 5.0 * (v * v).sum(axis=0), rtol=1e-14
    )


def test_a_native_function_used_sixteen_times_is_defined_once_and_called_sixteen_times(
    tmp_path,
):
    uses = "\n".join(
        f"    acc = acc + fma_sq(x * {1.0 + 0.1 * k!r}, y)" for k in range(USES)
    )
    source = (
        "import hawk\n\nfma_sq = hawk.native('fma_sq', reads={'x': hawk.Scalar, 'y': hawk.Scalar}, "
        "returns=hawk.Scalar, text={'host': 'return x * x + y;'})\n\n"
        "@hawk.kernel\ndef kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):\n"
        f"    acc = 0.0\n{uses}\n    out = acc\n"
    )
    # the tracer reads a kernel's own text, so the kernel is written to a module
    module = tmp_path / "sixteen.py"
    module.write_text(source)
    namespace: dict = {}
    exec(compile(source, str(module), "exec"), namespace)  # noqa: S102 - the file's own text, written just above
    text = _text(namespace["kern"])
    assert len(re.findall(r"^static inline Real hawk_fn_fma_sq_\w+\(", text, re.M)) == 1
    assert len(re.findall(r"hawk_fn_fma_sq_\w+\(", text)) == USES + 1


def test_a_target_without_text_is_refused_by_the_function_and_the_target_names():
    host_only = hawk.native(
        "host_only",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x;"},
    )

    @hawk.kernel
    def other(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = host_only(x)

    with pytest.raises(
        HawkError, match=r'native function "host_only" has no text for target "cuda"'
    ):
        _text(other, target="cuda")


def test_a_text_that_names_a_plane_fails_to_compile_and_the_failure_names_the_function_and_the_line(
    cache_dir,
):
    peek = hawk.native(
        "peek",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "double scaled = x * 2.0;\nreturn scaled * out;"},
    )

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = peek(x)

    with pytest.raises(HawkError) as refused:
        _run(kern, cache_dir, x=np.ones(LANES))
    message = str(refused.value)
    assert 'compile of "kern" failed' in message
    # the compiler reads the author's own numbering under the function's name: the plane is named on the second line
    assert re.search(
        r"native function peek:2:\d+: error: [^\n]*'out'",
        message,
    )


def test_the_text_goes_into_the_unit_byte_for_byte_between_line_directives():
    text = 'return x; // a line comment, a backslash \\ and "quotes"'
    verbatim = hawk.native(
        "verbatim", reads={"x": hawk.Scalar}, returns=hawk.Scalar, text={"host": text}
    )

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = verbatim(x)

    unit = _text(kern).split("\n")
    start = unit.index('#line 1 "native function verbatim"')
    assert unit[start + 1] == text
    # the unit's own numbering resumes on the line after the directive, under the name the compiler was given
    assert unit[start + 2] == f"#line {start + 4} __BASE_FILE__"


@hawk.function
def blended(a: hawk.Scalar, b: hawk.Scalar) -> hawk.Scalar:
    return fma_sq(a, b) * 2.0


def test_a_traced_function_may_call_a_native_one_and_each_is_defined_once(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = blended(x, y) + blended(y, x)

    text = _text(kern)
    assert len(re.findall(r"^static inline Real hawk_fn_", text, re.M)) == 2
    assert text.index("hawk_fn_fma_sq_") < text.index("hawk_fn_blended_")
    x, y = _xy()
    np.testing.assert_allclose(
        _run(kern, cache_dir, x=x, y=y),
        2.0 * (x * x + y) + 2.0 * (y * y + x),
        rtol=1e-14,
    )


def test_a_native_function_may_take_the_result_of_a_traced_one(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(blended(x, y), y)

    x, y = _xy()
    inner = 2.0 * (x * x + y)
    # a few units in the last place of the magnitudes summed: the sum may cancel, so the bound is not relative to its result
    np.testing.assert_allclose(
        _run(kern, cache_dir, x=x, y=y),
        inner * inner + y,
        rtol=0,
        atol=1e-14 * (inner * inner + np.abs(y)).max(),
    )


def test_natives_of_different_text_built_one_after_the_other_each_run_their_own_text(
    tmp_path, cache_dir
):
    """The unit a kernel is served from depends on the function's text, so a changed text is never served the old build."""
    x = np.arange(1.0, 5.0)
    for factor in (2, 3, 4):
        scale = hawk.native(
            "scale",
            reads={"x": hawk.Scalar},
            returns=hawk.Scalar,
            text={"host": f"return x * {factor}.0;"},
        )

        @hawk.kernel
        def rk(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = scale(x)

        y = np.zeros(4)
        hawk.deploy(rk, targets=("host",), cache_dir=cache_dir).run(x=x, y=y)
        np.testing.assert_allclose(y, factor * x)


def test_two_native_functions_of_one_name_and_different_text_in_one_kernel_are_refused_by_name():
    a = hawk.native(
        "scale",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x * 2.0;"},
    )
    b = hawk.native(
        "scale",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x * 3.0;"},
    )

    with pytest.raises(HawkError, match="scale"):

        @hawk.kernel
        def two(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = a(x) + b(x)


def test_the_device_text_writes_a_native_function_with_a_device_text_between_line_directives():
    """RESTATED (the device refused a native function by name; one with a device text is now written like the host's, the author's
    text between line directives, and one without a device text is still refused by its name and the target's)."""

    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(x, y)

    text = _text(kern, target="cuda")
    assert '#line 1 "native function fma_sq"' in text
    assert "return fma(x, x, y);" in text


def test_a_wrong_argument_count_is_refused_by_the_function_name():
    with pytest.raises(HawkError, match="function 'fma_sq' takes 2 argument"):

        @hawk.kernel
        def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = fma_sq(x)


@pytest.mark.parametrize(
    ("declare", "match"),
    [
        (
            lambda: hawk.native(
                "not a name",
                reads={},
                returns=hawk.Scalar,
                text={"host": "return 1.0;"},
            ),
            "identifier",
        ),
        (
            lambda: hawk.native(
                "f", reads={"x": 3}, returns=hawk.Scalar, text={"host": "return 1.0;"}
            ),
            "payload type",
        ),
        (
            lambda: hawk.native("f", reads={}, returns=3, text={"host": "return 1.0;"}),
            "returns needs a payload type",
        ),
        (
            lambda: hawk.native("f", reads={}, returns=hawk.Scalar, text={}),
            "at least one target",
        ),
        (
            lambda: hawk.native("f", reads={}, returns=hawk.Scalar, text={"host": 1}),
            "at least one target",
        ),
    ],
)
def test_a_malformed_declaration_is_refused_by_the_function_name(declare, match):
    with pytest.raises(HawkError, match=match):
        declare()


@pytest.mark.parametrize(
    "derivative, direction", [(hawk.diff.vjp, "vjp"), (hawk.diff.jvp, "jvp")]
)
def test_a_native_function_without_the_rule_of_a_direction_is_refused_by_name(
    derivative, direction
):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = fma_sq(x, y)

    with pytest.raises(
        HawkError,
        match=f"native function 'fma_sq' has no {direction} rule; pass {direction}=",
    ):
        derivative(kern)
