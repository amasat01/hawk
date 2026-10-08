# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Keyword-only parameters are bound BY NAME when a body is traced.

A kernel may split its parameters into a positional group and a keyword-only
group (``def k(x, *, scale, y)``). How a parameter is passed is a matter of the
author's signature, never of the artifact: the slot order comes from the traced
walk, so the keyword-only spelling must emit the same slots and the same body as
the all-positional one.
"""

from __future__ import annotations

import pytest

from hawk import Mutable, Scalar, kernel
from hawk.emit import render_body
from hawk.ir import HawkError
from hawk.trace import trace_value


@kernel
def _positional(x: Scalar, scale: Scalar, y: Mutable[Scalar]):
    y = x * scale


@kernel
def _keyword_only(x: Scalar, *, scale: Scalar, y: Mutable[Scalar]):
    y = x * scale


def test_a_keyword_only_signature_emits_the_same_slots_and_body():
    assert _keyword_only.arg_spec == _positional.arg_spec
    assert render_body(_keyword_only.sinks, _keyword_only.walk).text == \
        render_body(_positional.sinks, _positional.walk).text


def test_trace_value_binds_keyword_only_parameters_by_name():
    def scaled(x, *, scale):
        return x * scale

    def plain(x, scale):
        return x * scale

    a = trace_value(scaled, {"x": Scalar, "scale": Scalar})
    b = trace_value(plain, {"x": Scalar, "scale": Scalar})
    assert repr(a.node) == repr(b.node)


def test_variadic_parameters_are_refused_by_name():
    def variadic(x, *rest):
        return x

    with pytest.raises(HawkError, match=r"\*rest"):
        trace_value(variadic, {"x": Scalar, "rest": Scalar})
