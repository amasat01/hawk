# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusals a caller of the public names meets: the kernel name a C library owns, and the one refusal type."""

from __future__ import annotations

import pytest

import hawk
from hawk import HawkError, Scalar, _ir, errors, ospreys
from hawk.artifact import build_bundle


def test_a_kernel_named_after_a_c_library_function_is_refused_naming_it_and_asking_for_another_name(
    tmp_path,
):
    """``brk``, ``read``, ``malloc`` and ``qsort`` are functions of the C library; the entry is ``extern "C"`` under the
    kernel's name, so each would redeclare one."""

    @hawk.kernel
    def brk(x: Scalar) -> Scalar:
        return x * 2.0

    @hawk.kernel
    def read(x: Scalar) -> Scalar:
        return x * 2.0

    @hawk.kernel
    def malloc(x: Scalar) -> Scalar:
        return x * 2.0

    @hawk.kernel
    def qsort(x: Scalar) -> Scalar:
        return x * 2.0

    for kernel in (brk, read, malloc, qsort):
        with pytest.raises(
            HawkError, match=rf"'{kernel.name}'.*reserved.*name the kernel otherwise"
        ):
            build_bundle([kernel], tmp_path / kernel.name, targets=("host",))


def test_an_ordinary_kernel_name_traces():
    """A name no C library owns is kept as the kernel's name."""

    @hawk.kernel
    def doubled_input(x: Scalar) -> Scalar:
        return x * 2.0

    assert doubled_input.name == "doubled_input"


def test_the_public_refusal_name_is_the_one_class_every_refusal_is():
    """``hawk.HawkError`` is the class of ``hawk.errors``, and the toolchain refusal is one of it carrying the tool's name."""
    assert hawk.HawkError is errors.HawkError
    assert issubclass(hawk.ToolchainAbsent, hawk.HawkError)
    refusal = hawk.ToolchainAbsent("no compiler", toolchain="cc")
    assert refusal.toolchain == "cc" and str(refusal) == "no compiler"


def test_a_refusal_raised_inside_the_library_is_caught_by_the_public_name_as_that_exact_class():
    """A float16 constant is refused by the library; the caller sees ``hawk.HawkError`` itself, not a subclass."""
    with pytest.raises(hawk.HawkError) as refused:
        _ir.const(_ir.new_context(), 1.0, ((), "f16", None))
    assert type(refused.value) is hawk.HawkError


def test_a_build_for_no_target_is_refused_naming_the_served_targets_and_publishes_nothing(
    tmp_path,
):
    """``targets=()`` would publish an empty unit; the refusal names the kernel and lists what is served."""

    @hawk.kernel
    def scaled(x: Scalar) -> Scalar:
        return x * 2.0

    with pytest.raises(
        HawkError,
        match=r"'scaled': no target to build; the served targets are \('cuda', 'host'\)",
    ):
        build_bundle(
            [scaled], tmp_path / "bundle", targets=(), cache_dir=tmp_path / "cache"
        )
    assert not (tmp_path / "bundle").exists()


def test_a_target_no_osprey_serves_is_refused_naming_it_and_where_the_search_looked(
    tmp_path,
):
    """The refusal quotes the unknown target, lists the served ones and names the two places OSPREYs are found and the folders
    that were searched."""

    @hawk.kernel
    def scaled(x: Scalar) -> Scalar:
        return x * 2.0

    with pytest.raises(
        HawkError, match=r"target 'nosuch' is served by no OSPREY found"
    ) as refused:
        build_bundle(
            [scaled],
            tmp_path / "bundle",
            targets=("nosuch",),
            cache_dir=tmp_path / "cache",
        )
    assert "the hawk.ospreys entry points and $HAWK_OSPREY_PATH" in str(refused.value)
    assert "(searched: " in str(refused.value)
    assert all(str(osprey.folder) in str(refused.value) for osprey in ospreys.found().values())
    assert not (tmp_path / "bundle").exists()
