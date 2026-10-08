# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``hawk.trace.trace_value`` — tracing an authored EXPRESSION, not a kernel.

A host that builds kernels from authored predicates or loss terms traces the
expression here and commits the returned value in a kernel of its own making.
The load-bearing observation is equivalence: that host-built kernel must emit the
SAME body as the kernel an author would have written with the expression inline.
"""

from __future__ import annotations

import pytest

from hawk import Mutable, Scalar, Terminated, kernel
from hawk.emit import render_body
from hawk.ir import HawkError
from hawk.trace import trace_value
from hawk.trace.kernel import Kernel
from hawk.trace.value import MutableRef


def _branchy(x, v):
    if x > 0.0:
        a = x * v
    else:
        a = x - v
    return a + 1.0


@kernel
def _inline(x: Scalar, v: Scalar, y: Mutable[Scalar]):
    if x > 0.0:
        a = x * v
    else:
        a = x - v
    y = a + 1.0


def _host_kernel(value) -> Kernel:
    """What a host does with a traced value: commit it to an output it declares."""
    out = MutableRef("y", Mutable[Scalar].ttype)
    out[...] = value
    return Kernel("_inline", tuple(out.sinks),
                  {"x": Scalar, "v": Scalar, "y": Mutable[Scalar]})


def test_a_traced_value_commits_to_the_same_body_as_the_inline_kernel():
    host = _host_kernel(trace_value(_branchy, {"x": Scalar, "v": Scalar}))
    assert host.arg_spec == _inline.arg_spec
    assert render_body(host.sinks, host.walk).text == \
        render_body(_inline.sinks, _inline.walk).text


def test_the_bindings_are_the_declarations_and_annotations_are_not_read():
    def doubled(x: NotAName):  # noqa: F821 - deliberately unresolvable
        return x * 2.0

    value = trace_value(doubled, {"x": Scalar})
    assert value is not None


def test_a_mask_declaration_binds_as_a_readable_plane():
    def alive(t):
        return t

    assert trace_value(alive, {"t": Terminated}) is not None


@pytest.mark.parametrize("bindings, match", [
    ({"x": Scalar}, "missing \\['v'\\]"),
    ({"x": Scalar, "v": Scalar, "w": Scalar}, "not a parameter \\['w'\\]"),
    ({"x": Scalar, "v": Mutable[Scalar]}, "declare an output plane"),
])
def test_malformed_bindings_are_refused_by_name(bindings, match):
    with pytest.raises(HawkError, match=match):
        trace_value(_branchy, bindings)


def test_a_body_that_does_not_end_in_a_returned_expression_is_refused():
    def no_return(x):
        _ = x * 2.0

    with pytest.raises(HawkError, match="must end with `return <expression>`"):
        trace_value(no_return, {"x": Scalar})


def test_a_returned_constant_is_refused():
    def constant(x):
        return 2.0

    with pytest.raises(HawkError, match="not a traced value"):
        trace_value(constant, {"x": Scalar})
