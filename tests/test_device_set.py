# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where an array is placed: ``hawk.asarray(x, device=...)`` names the member, the library answers, a call follows its planes.

hawk passes the member on and decides nothing about it: the rows assert on what the door is given, and on the host library's
reason reaching the caller as a ``HawkError``. What needs two devices (a node over two members refused, a copy from one member to
the other, values equal to the one-device run) runs on a machine with two cards.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import hawk
from hawk import HawkError
from hawk import _place as place
from hawk.targets import TARGETS, runs_here


class _Door:
    """A door that records the plane it is asked to make and stops there."""

    def __init__(self):
        self.asked = []

    def array_layout(self, array):
        return {"shape": array.shape, "dtype": (2, 64, 1)}

    def plane_wrap(self, array, stream=None):
        return "source"

    def plane_new(self, shape, code, bits, **kwargs):
        self.asked.append(kwargs)
        raise HawkError("recorded")

    def plane_free(self, plane):
        pass


def test_a_plane_made_for_a_device_states_the_member_of_its_place():
    """The maker of device ``(2, 1)`` asks its door for a plane on member 1; the host maker states no member."""
    door = _Door()
    with pytest.raises(HawkError, match="recorded"):
        place.Maker(door, (2, 1)).copy_of(np.zeros(3))
    with pytest.raises(HawkError, match="recorded"):
        place.Maker(door, (2, 1)).zeros((3,), "float64")
    with pytest.raises(HawkError, match="recorded"):
        place.Maker(door, place.HOST).zeros((3,), "float64")
    assert [asked["device"] for asked in door.asked] == [
        1,
        1,
        place._osprey.EG_DEVICE_DEFAULT,
    ]


@pytest.mark.skipif(
    any(runs_here(t.id)[0] for t in TARGETS.values() if t.device),
    reason="a device runs here",
)
def test_a_call_follows_its_planes_and_the_library_answers_for_a_member_that_does_not_run_here():
    class Elsewhere:
        """An array that exports DLPack over host memory and states member 1 of a device."""

        def __init__(self, array):
            self._array = array

        def __dlpack__(self, **kwargs):
            return self._array.__dlpack__(**kwargs)

        def __dlpack_device__(self):
            return (2, 1)

    device = next(t.id for t in TARGETS.values() if t.device)
    with pytest.raises(HawkError, match=re.escape(runs_here(device)[1])):
        hawk.reduce(Elsewhere(np.zeros((3, 6))), "sum")


def test_a_copy_between_host_planes_is_the_one_member_copy():
    src, dst = np.arange(6.0), np.zeros(6)
    assert hawk.copy(src, dst) is dst
    np.testing.assert_array_equal(dst, src)
