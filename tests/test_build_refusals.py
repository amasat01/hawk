# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Kernels a build refuses before anything is compiled, asked through ``build_bundle`` on the host: each is refused by name."""

import hawk
import hawk.math as M
import pytest
from hawk import Index, Mutable, Scalar, Terminated
from hawk.errors import HawkError
from hawk.artifact import build_bundle
from hawk.ext import Guard, IndexSet, Kind

SET = Kind("lane_set", lanes=IndexSet())
MAP = Kind("active_rows", guard=Guard(active_set=True))


def _lane_set_in_a_segmented_unit(x: Scalar, kind: Scalar, y: Mutable[Scalar]):
    y = M.dispatch(M.int32(kind), [x, x * 2.0], strategy="segmented")


def _a_guard_with_an_accumulate(x: Scalar, lane: Index, terminated: Terminated, acc: Mutable[Scalar]):
    hawk.scatter(acc)[lane] += x


def _a_guard_without_its_mask(omega: Scalar, x: Mutable[Scalar]):
    x = x * omega


def _a_reserved_plane_name(active_map: Scalar, terminated: Terminated, x: Mutable[Scalar]):
    x = x + active_map


def _two_segmented_dispatches(k: Index, x: Scalar, y: Mutable[Scalar]):
    a = M.dispatch(k, [x, x * 2.0], strategy="segmented")
    b = M.dispatch(k, [x + 1.0, x * 3.0], strategy="segmented")
    y = a + b


def _a_lane_set_reading_the_launch_slot_in_a_segmented_unit(k: Index, slot: Mutable[Index], y: Mutable[Scalar]):
    y = M.dispatch(k, [1.0, 2.0], strategy="segmented")
    slot = hawk.sample_position()


@pytest.mark.parametrize("body, kind", [
    (_lane_set_in_a_segmented_unit, SET), (_a_guard_with_an_accumulate, MAP), (_a_guard_without_its_mask, MAP),
    (_a_reserved_plane_name, MAP), (_two_segmented_dispatches, None), (_a_lane_set_reading_the_launch_slot_in_a_segmented_unit, None)],
    ids=["lane set in a segmented unit", "guard with an accumulate", "guard without its mask", "reserved plane name",
         "two segmented dispatches", "launch slot in a segmented unit"])
def test_the_build_refuses_the_kernel(body, kind, tmp_path, cache_dir):
    kernel = hawk.kernel(body, kind=kind) if kind is not None else hawk.kernel(body)
    with pytest.raises(HawkError):
        build_bundle([kernel], tmp_path, targets=("host",), cache_dir=cache_dir)
