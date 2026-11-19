# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every user option of a built target moves the build key, and the same options in another order give the same key.

``options={"<target id>.<key>": value}`` is stated to the OSPREY that serves the target, which puts it into the key it answers
for the machine; ``hawk.artifact.build_key`` reads that key without compiling anything. The options are those the shipped
OSPREYs declare: ``host.profile`` and ``cuda.literal_table`` / ``cuda.arch``.
"""

from __future__ import annotations

import pytest

import hawk
from hawk import Scalar
from hawk.artifact import build_key


@hawk.kernel
def scaled(x: Scalar) -> Scalar:
    return x * 2.0


def key(targets, options=None):
    """The build key of ``scaled`` for ``targets`` under ``options``."""
    return build_key([scaled], targets=targets, options=options)


@pytest.mark.parametrize(
    ("targets", "first", "second"),
    [
        (("host",), {"host.profile": "native"}, {"host.profile": "portable"}),
        (("cuda",), {"cuda.literal_table": False}, {"cuda.literal_table": True}),
        (("cuda",), {"cuda.arch": "sm_61"}, {"cuda.arch": "sm_89"}),
    ],
    ids=["host.profile", "cuda.literal_table", "cuda.arch"],
)
def test_an_option_of_a_built_target_moves_the_key_and_a_repeat_of_it_does_not(
    targets, first, second
):
    """Another value gives another key, an option given at all is hashed even at a default, and the same option twice agrees."""
    base = key(targets)
    assert key(targets, first) != base and key(targets, second) != base
    assert key(targets, first) != key(targets, second)
    assert key(targets, first) == key(targets, dict(first))


def test_options_in_another_order_give_the_same_key():
    """Two options stated in either order are the same build and name the same unit."""
    one = {"cuda.literal_table": True, "cuda.arch": "sm_61"}
    other = dict(reversed(list(one.items())))
    assert list(one) != list(other)
    assert key(("cuda",), one) == key(("cuda",), other)


def test_an_option_of_a_target_this_build_does_not_build_is_not_hashed():
    """A ``cuda`` option stated for a host-only build leaves the host key as it was."""
    assert key(("host",), {"cuda.literal_table": True}) == key(("host",))
