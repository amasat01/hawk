# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel named after a C library function is refused at trace.

The entry is ``extern "C"`` under the kernel's name, so ``brk`` or ``read``
would redeclare the library's function; the names come from the toolchain's own
headers, not from a list kept by hand."""

from __future__ import annotations

import pytest

import hawk
from hawk import Scalar
from hawk.compile.toolchain import libc_symbols
from hawk.ir import HawkError

pytestmark = pytest.mark.skipif(not libc_symbols(), reason="no host compiler to read the C headers")


def test_the_derived_list_holds_the_library_functions():
    assert libc_symbols() and "read" in libc_symbols()


def test_a_kernel_named_after_a_library_function_is_refused():
    with pytest.raises(HawkError, match=r"'brk'.*C library.*rename"):
        @hawk.kernel
        def brk(x: Scalar) -> Scalar:
            return x * 2.0


def test_an_ordinary_name_traces():
    @hawk.kernel
    def doubled_input(x: Scalar) -> Scalar:
        return x * 2.0

    assert doubled_input.name == "doubled_input"
