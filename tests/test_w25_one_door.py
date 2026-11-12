# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build settings have one door, ``build_policy=``: a legacy keyword still works alone (it is folded into the policy at the
entry), an equal one beside a policy is accepted, a disagreeing one is refused by name with both values, and the unit is the same
whichever way the setting arrived."""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector
from hawk.artifact import build_bundle
from hawk.build_policy import BuildPolicy
from hawk import HawkError

#: setting -> (value the keyword and the policy agree on, a different allowed value)
SETTINGS = {"emit_form": ("rolled", "static"), "linalg_form": ("team", "thread"), "wide_form": ("thread", "auto")}


@hawk.kernel
def solve3(A: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]]):
    x, _ = m.solve(A, b)


def _build(tmp_path, tag, cache_dir, **kw):
    return build_bundle([solve3], tmp_path / tag, targets=("host",), cache_dir=cache_dir, **kw)


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_an_agreeing_keyword_beside_the_policy_is_accepted(tmp_path, cache_dir, key):
    value, _ = SETTINGS[key]
    _build(tmp_path, "agree", cache_dir, build_policy=BuildPolicy(**{key: value}), **{key: value})


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_a_disagreeing_keyword_is_refused_by_name_with_both_values(tmp_path, cache_dir, key):
    value, other = SETTINGS[key]
    with pytest.raises(HawkError, match=rf"{key}={other!r} disagrees with build_policy\.{key}={value!r}"):
        _build(tmp_path, "clash", cache_dir, build_policy=BuildPolicy(**{key: value}), **{key: other})


def test_a_disagreeing_launch_bounds_keyword_is_refused_by_name_with_both_values(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"launch_bounds=\(128, 2\) disagrees with build_policy\.launch_bounds=\(256, 1\)"):
        _build(tmp_path, "lb", cache_dir, build_policy=BuildPolicy(launch_bounds=(256, 1)), launch_bounds=(128, 2))


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_a_keyword_alone_and_the_policy_alone_make_the_same_unit(tmp_path, cache_dir, key):
    value, _ = SETTINGS[key]
    by_keyword = _build(tmp_path, "kw", cache_dir, **{key: value})
    by_policy = _build(tmp_path, "pol", cache_dir, build_policy=BuildPolicy(**{key: value}))
    assert by_keyword.digest == by_policy.digest
    assert (by_keyword.directory / "solve3.cpp").read_text() == (by_policy.directory / "solve3.cpp").read_text()


def test_the_form_sets_are_validated_in_one_place():
    from hawk.passes import policy

    for key in SETTINGS:
        with pytest.raises(HawkError, match=f"{key} must be one of"):
            BuildPolicy(**{key: "bogus"})
        with pytest.raises(ValueError, match=f"{key} must be one of"):
            policy.refuse_choice(key, "bogus")
