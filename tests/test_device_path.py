# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The code a run over device memory shares with a run over host memory, proven on the host door.

Arrays cross by DLPack; a plan runs on the door of the memory its arrays are in; an output the caller leaves out is made by that
door and read back by the protocol; the one-call forms wait and ``hawk.synchronize`` waits on every open door. Whether a device
runs here is the device library's own answer, so the rows that need a machine without one skip on a machine with one.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar
from hawk import _place as place
from hawk.targets import TARGETS, runs_here

N = 16


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


class _Elsewhere:
    """An array that exports DLPack over host memory and states a device nothing here serves."""

    def __init__(self, array, device=(2, 0)):
        self._array, self._device = array, device

    def __dlpack__(self, **kwargs):
        return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._device


def _no_device_runs_here() -> bool:
    return not any(runs_here(t.id)[0] for t in TARGETS.values() if t.device)


@pytest.fixture(scope="module")
def plan(cache_dir):
    return hawk.deploy(axpb, targets=("host",), cache_dir=cache_dir)


def test_an_output_left_out_is_made_by_the_door_and_comes_back_as_a_numpy_array(plan):
    x = np.arange(N, dtype=np.float64)
    first = plan.run(x=x, a=2.0, b=1.0)
    second = plan.run(x=x, a=3.0, b=0.0)
    assert isinstance(first, np.ndarray) and first.dtype == np.float64 and first.shape == (N,)
    np.testing.assert_array_equal(first, 2.0 * x + 1.0)
    np.testing.assert_array_equal(second, 3.0 * x)          # each call made its own memory
    assert not np.shares_memory(first, second)


def test_a_made_array_is_writable_zeroed_memory_of_the_hosts_door():
    made = place.Maker.at(place.HOST).zeros((3, 4), "float32")
    assert made.shape == (3, 4) and made.dtype == np.float32 and not made.any()
    made[1, 2] = 5.0
    assert made.sum() == 5.0
    assert place.Maker.at(place.HOST).zeros((2,), "bool").dtype == np.bool_


def test_a_made_array_is_read_once_and_the_dtype_words_are_the_sidecars():
    assert [place.dlpack_dtype(w) for w in ("float64", "float32", "int64", "uint32", "bool")] == [
        (2, 64), (2, 32), (0, 64), (1, 32), (6, 8)]
    with pytest.raises(HawkError, match="has no DLPack form"):
        place.dlpack_dtype("complex128")


def test_planes_of_one_launch_in_two_places_are_refused_naming_both(plan):
    x = np.arange(N, dtype=np.float64)
    with pytest.raises(HawkError, match=r"live in one place; 'x' is on the DLPack device \(1, 0\) and 'y' is on \(2, 0\)"):
        plan.bind(x=x, y=_Elsewhere(np.zeros(N)), a=1.0, b=0.0)


@pytest.mark.skipif(not _no_device_runs_here(), reason="a device runs here")
def test_the_place_of_a_launch_is_its_device_and_the_first_array_there():
    here, there = np.zeros(3), _Elsewhere(np.zeros(3))
    assert place.place_of({"a": 1.0, "x": here}, "k") == (place.HOST, here)
    device, array = place.place_of({"x": there}, "k")
    assert device == (2, 0) and array is there
    assert place.place_of({"a": 1.0}, "k") is None


class _StreamDoor:
    """A door that states a stream and keeps what ``plane_wrap`` was given."""

    def __init__(self):
        self.wrapped = []

    def ctx_info(self):
        return {"stream": 7}

    def plane_wrap(self, array, **keywords):
        self.wrapped.append(keywords)


def test_only_an_array_in_device_memory_is_wrapped_behind_the_doors_stream():
    door = _StreamDoor()
    place.wrap(door, np.zeros(3))
    place.wrap(door, _Elsewhere(np.zeros(3)))
    assert door.wrapped == [{}, {"stream": 7}]


def test_an_array_on_a_device_no_library_runs_on_is_refused_with_each_librarys_reason(plan):
    with pytest.raises(HawkError, match=r"no installed OSPREY runs on the DLPack device \(2, 0\): .*does not run here"):
        plan.run(x=_Elsewhere(np.arange(N, dtype=np.float64)), a=1.0, b=0.0)


def test_synchronize_waits_on_the_host_door_once_it_is_open(plan):
    plan.run(x=np.arange(N, dtype=np.float64), a=1.0, b=0.0)
    from hawk import ospreys

    assert any(osprey.open_doors() for osprey in ospreys.found().values())
    assert hawk.synchronize() is None


def test_a_launch_is_waited_for_by_the_plan_and_by_the_one_call_forms(plan):
    x, out = np.arange(N, dtype=np.float64), np.zeros(N)
    bound = plan.bind(x=x, a=2.0, b=0.0, y=out)
    bound.launch()
    bound.wait()
    np.testing.assert_array_equal(out, 2.0 * x)


@pytest.mark.skipif(runs_here("cuda")[0], reason="a device runs here")
def test_deploy_for_a_stated_device_target_carries_the_librarys_reason_unchanged(cache_dir):
    why = runs_here("cuda")[1]
    with pytest.raises(HawkError) as refusal:
        hawk.deploy(axpb, targets=("host", "cuda"), cache_dir=cache_dir)
    assert why in str(refusal.value) and "does not run here" in str(refusal.value)


def test_deploy_without_targets_builds_the_targets_that_run_here(plan):
    assert plan._deployed.settings["targets"] == tuple(t for t in TARGETS if runs_here(t)[0])


@pytest.mark.skipif(not _no_device_runs_here(), reason="a device runs here")
def test_asarray_onto_a_device_that_does_not_run_here_carries_the_reason():
    device = next(t.id for t in TARGETS.values() if t.device)
    with pytest.raises(HawkError, match=re.escape(runs_here(device)[1])):
        hawk.asarray(np.arange(3.0), device=f"{device}:0")


@pytest.mark.parametrize("device", ["", "host:0", "nowhere:0", "cuda:x"])
def test_asarray_names_the_form_of_a_device(device):
    with pytest.raises(HawkError, match="is not '<target>:<ordinal>' of a device target"):
        hawk.asarray(np.arange(3.0), device=device)


class _Viewable:
    """An array of another library, over host memory: it exports DLPack, is no NumPy array, and has the ``reshape`` and slicing
    every array library has, each giving another such array over the same memory (a view)."""

    def __init__(self, array):
        self._array = array
        self.shape, self.dtype = array.shape, array.dtype

    def __dlpack__(self, **kwargs):
        return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()

    def reshape(self, *shape):
        return _Viewable(self._array.reshape(*shape))

    def __getitem__(self, key):
        return _Viewable(self._array[key])


@hawk.kernel
def column_sums(v: hawk.Vector[6], t: Mutable[hawk.Vector[6]]):
    t = hawk.sum(v)


def test_a_cell_of_several_columns_in_another_library_is_viewed_per_column_by_its_own_slicing(cache_dir):
    v = np.random.default_rng(3).uniform(-2.0, 2.0, (6, N))
    cell = np.full(6, np.nan)
    hawk.deploy(column_sums, targets=("host",), cache_dir=cache_dir).run(v=v, t=_Viewable(cell))
    np.testing.assert_allclose(cell, v.sum(axis=1), rtol=1e-13)


def test_a_cell_with_the_wrong_number_of_values_or_no_slicing_is_refused_by_name(cache_dir):
    v = np.random.default_rng(3).uniform(-2.0, 2.0, (6, N))
    plan = hawk.deploy(column_sums, targets=("host",), cache_dir=cache_dir)
    with pytest.raises(HawkError, match=r"the cell 't' holds 3 value\(s\); it is one value per column, 6 here"):
        plan.run(v=v, t=_Viewable(np.zeros(3)))
    with pytest.raises(HawkError, match=r"the cell 't' \(_Elsewhere\) cannot be viewed per column"):
        plan.run(v=v, t=_Elsewhere(np.zeros(6), device=(1, 0)))
