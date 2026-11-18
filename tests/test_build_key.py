# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The build key names what a build is made of; ``buildable`` asks the serving OSPREY whether its tools are present."""

from __future__ import annotations

import pytest
from hawk import Mutable, Param, Scalar, kernel
from hawk.artifact import build_key
from hawk.errors import HawkError
from hawk.targets import TARGETS, buildable

HOST = ("host",)


def make_axpb():
    """A fresh traced kernel ``y = a * x + b``."""

    @kernel
    def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
        y = a * x + b

    return axpb


def make_axpb_other_body():
    """The same signature as :func:`make_axpb` with another body."""

    @kernel
    def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
        y = a * x - b

    return axpb


def make_axpb_renamed():
    """The body of :func:`make_axpb` under another name."""

    @kernel
    def axpb_again(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
        y = a * x + b

    return axpb_again


def test_equal_requests_give_equal_keys():
    assert build_key([make_axpb()], targets=HOST) == build_key([make_axpb()], targets=HOST)


def test_the_key_changes_when_the_body_changes():
    assert build_key([make_axpb()], targets=HOST) != build_key([make_axpb_other_body()], targets=HOST)


def test_the_key_changes_when_the_name_changes():
    assert build_key([make_axpb()], targets=HOST) != build_key([make_axpb_renamed()], targets=HOST)


def test_the_key_changes_when_a_declaration_changes():
    from hawk import Matrix, Vector

    @kernel
    def apply(m: Matrix[3, 3], v: Vector[3], y: Mutable[Vector[3]]):
        y = m @ v

    @kernel
    def apply_wide(m: Matrix[4, 4], v: Vector[4], y: Mutable[Vector[4]]):
        y = m @ v

    wide = apply_wide
    assert build_key([apply], targets=HOST) != build_key([wide], targets=HOST)


def test_the_key_changes_when_the_mode_changes():
    one = make_axpb()
    assert build_key([one], targets=HOST) != build_key([one], targets=HOST, mode="float32")


def test_the_key_changes_when_the_opt_level_changes():
    one = make_axpb()
    assert build_key([one], targets=HOST, opt_level="O0") != build_key([one], targets=HOST, opt_level="O3")


def test_the_key_changes_when_the_targets_change():
    served = tuple(TARGETS)
    if len(served) < 2:
        pytest.skip("one served target: nothing to compare")
    one = make_axpb()
    assert build_key([one], targets=served[:1]) != build_key([one], targets=served)


def test_a_key_for_an_unserved_target_is_refused_by_name():
    with pytest.raises(HawkError, match="target 'tpu' is served by no OSPREY"):
        build_key([make_axpb()], targets=("tpu",))


def test_a_served_target_answers_whether_its_tools_are_present():
    assert isinstance(buildable("host"), bool)


def test_an_unserved_target_is_refused_by_name_when_asked_if_buildable():
    with pytest.raises(HawkError, match="target 'tpu' is served by no OSPREY"):
        buildable("tpu")
