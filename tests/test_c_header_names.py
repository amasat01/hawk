# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel named like a standard C header is refused for a C build, because the unit writes ``<name>.h``; a C library
function name is not (the C unit never uses the bare kernel name as an identifier)."""

from __future__ import annotations

import pytest

import hawk
from hawk import Scalar
from hawk.artifact import build_bundle
from raptor.errors import RaptorError as HawkError


@hawk.kernel
def math(x: Scalar) -> Scalar:
    return x * 2.0


@hawk.kernel
def stdint(x: Scalar) -> Scalar:
    return x * 2.0


@hawk.kernel
def string(x: Scalar) -> Scalar:
    return x * 2.0


@hawk.kernel
def sinh(x: Scalar) -> Scalar:
    return x * 2.0


@hawk.kernel
def read(x: Scalar) -> Scalar:
    return x * 2.0


@pytest.mark.parametrize("kernel", [math, stdint, string], ids=lambda k: k.name)
def test_a_standard_header_stem_is_refused_by_name_for_a_c_build(tmp_path, kernel):
    with pytest.raises(HawkError, match=rf"'{kernel.name}'.*standard C header.*rename"):
        build_bundle([kernel], tmp_path / kernel.name, targets=("c",))


@pytest.mark.parametrize("kernel", [sinh, read], ids=lambda k: k.name)
def test_a_library_function_name_is_accepted_for_a_c_build(tmp_path, kernel):
    assert build_bundle([kernel], tmp_path / kernel.name, targets=("c",)) is not None
