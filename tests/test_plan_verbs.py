# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host verbs of a plan: ``hawk.plan``, ``Plan.bind`` / ``run``, ``BoundPlan.launch`` / ``rebind`` / ``counters``,
``hawk.deploy``, the array verbs and the banded pack. Every value is held against NumPy or a closed form, and the one-call
form ``hawk.run`` is the reference a plan's launch must equal."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated, Vector
from hawk.artifact import build_bundle

N = 37


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def two_outputs(x: Scalar, y: Mutable[Scalar], v: Mutable[Vector[3]]):
    y = x + x
    v = hawk.math.vec(x, x * 2.0, x * 3.0)


@hawk.kernel
def total(x: Scalar, tot: Mutable[Scalar]):
    tot = hawk.sum(x)


@hawk.kernel
def share(x: Scalar, y: Mutable[Scalar]):
    y = x / hawk.sum(x)


@hawk.kernel
def tally(x: Scalar, lane: hawk.Index, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="plain")[lane] += x


@hawk.kernel
def ramp(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    """``{kernel name: loaded kernel}``: one host bundle, and a second for the kernel that writes across samples (a bundle
    declares one execution axis)."""
    out = {}
    for kernels in ([axpb, two_outputs, total, ramp, share], [tally]):
        bundle = build_bundle(kernels, tmp_path_factory.mktemp("plan_verbs"), targets=("host",), cache_dir=cache_dir)
        out |= {k.name: hawk.load(bundle.directory, k.name) for k in kernels}
    return out


def _axpb_planes():
    return {"x": np.arange(N, dtype=np.float64), "a": 3.0, "b": -2.0, "y": np.zeros(N)}


def _by_run(kernel) -> np.ndarray:
    planes = _axpb_planes()
    hawk.run(kernel, **planes)
    return planes["y"]


def test_a_plan_bound_and_launched_equals_hawk_run_bit_for_bit(loaded):
    planes = _axpb_planes()
    hawk.plan(loaded["axpb"], hawk.HostSerial).bind(**planes).launch()
    assert planes["y"].tobytes() == _by_run(loaded["axpb"]).tobytes()


def test_a_launch_writes_the_callers_own_memory_and_sees_a_changed_input(loaded):
    planes = _axpb_planes()
    bound = hawk.plan(loaded["axpb"]).bind(**planes)
    bound.launch()
    np.testing.assert_array_equal(planes["y"], 3.0 * planes["x"] - 2.0)
    planes["x"] += 1.0
    bound.launch()
    np.testing.assert_array_equal(planes["y"], 3.0 * (np.arange(N) + 1.0) - 2.0)


@pytest.mark.parametrize("structure", [hawk.HostTeam, hawk.HostSerial], ids=["team", "serial"])
def test_every_host_structure_gives_the_values_of_the_closed_form(loaded, structure):
    planes = _axpb_planes()
    hawk.plan(loaded["axpb"], structure).bind(**planes).launch()
    np.testing.assert_array_equal(planes["y"], 3.0 * np.arange(N) - 2.0)


def test_the_team_and_the_serial_structure_give_equal_values(loaded):
    team, serial = _axpb_planes(), _axpb_planes()
    hawk.plan(loaded["axpb"], hawk.HostTeam).bind(**team).launch()
    hawk.plan(loaded["axpb"], hawk.HostSerial).bind(**serial).launch()
    assert team["y"].tobytes() == serial["y"].tobytes()


@pytest.mark.parametrize("parts", [1, 3, 5, N, N + 4])
def test_npartitions_gives_the_values_of_the_whole_range(loaded, parts):
    planes = _axpb_planes()
    hawk.plan(loaded["axpb"], hawk.HostSerial, npartitions=parts).bind(**planes).launch()
    assert planes["y"].tobytes() == _by_run(loaded["axpb"]).tobytes()


def test_explicit_partitions_that_cover_the_range_give_the_values_of_the_whole_range(loaded):
    planes = _axpb_planes()
    cut = [hawk.Partition(0, 10, N), hawk.Partition(10, 20, N), hawk.Partition(30, N - 30, N)]
    hawk.plan(loaded["axpb"], hawk.HostSerial, partitions=cut).bind(**planes).launch()
    assert planes["y"].tobytes() == _by_run(loaded["axpb"]).tobytes()


def test_one_partition_alone_writes_only_its_samples(loaded):
    planes = _axpb_planes()
    hawk.plan(loaded["axpb"], hawk.HostSerial, partitions=[hawk.Partition(5, 7, N)]).bind(**planes).launch()
    want = np.zeros(N)
    want[5:12] = 3.0 * np.arange(5, 12) - 2.0
    np.testing.assert_array_equal(planes["y"], want)


def test_a_partition_outside_the_range_is_refused_by_name(loaded):
    with pytest.raises(HawkError, match=r"partition 1 .*leaves the range \[0, 37\)"):
        hawk.plan(loaded["axpb"], partitions=[hawk.Partition(0, 10, N), hawk.Partition(10, N, N)]).bind(**_axpb_planes())


def test_partitions_over_another_sample_count_than_the_planes_hold_are_refused(loaded):
    with pytest.raises(HawkError, match=r"partitions are over n_samples=10, the planes hold 37"):
        hawk.plan(loaded["axpb"], partitions=[hawk.Partition.whole(10)]).bind(**_axpb_planes())


def test_partitions_and_npartitions_together_are_refused(loaded):
    with pytest.raises(HawkError, match="two ways to partition"):
        hawk.plan(loaded["axpb"], npartitions=2, partitions=[hawk.Partition.whole(N)])


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_an_npartitions_that_is_not_a_positive_integer_is_refused(loaded, bad):
    with pytest.raises(HawkError, match="npartitions must be a positive integer"):
        hawk.plan(loaded["axpb"], npartitions=bad)


def test_the_whole_partition_is_the_whole_range():
    assert hawk.Partition.whole(9) == hawk.Partition(0, 9, 9)


def test_launch_returns_the_host_tiles_it_ran(loaded):
    planes = _axpb_planes()
    assert hawk.plan(loaded["axpb"], hawk.HostSerial).bind(**planes).launch() == 1
    big = {**planes, "x": np.arange(1 << 20, dtype=np.float64), "y": np.zeros(1 << 20)}
    assert hawk.plan(loaded["axpb"], hawk.HostTeam).bind(**big).launch() >= 1


def test_rebind_moves_one_plane_and_keeps_the_others(loaded):
    planes = _axpb_planes()
    bound = hawk.plan(loaded["axpb"]).bind(**planes)
    other = np.zeros(N)
    assert bound.rebind(y=other) is bound
    bound.launch()
    np.testing.assert_array_equal(other, 3.0 * np.arange(N) - 2.0)
    np.testing.assert_array_equal(planes["y"], np.zeros(N))
    assert bound.planes["y"] is other and bound.planes["x"] is planes["x"]


def test_rebind_changes_a_uniform_and_checks_what_it_is_given(loaded):
    planes = _axpb_planes()
    bound = hawk.plan(loaded["axpb"]).bind(**planes)
    bound.rebind(a=5.0).launch()
    np.testing.assert_array_equal(planes["y"], 5.0 * np.arange(N) - 2.0)
    with pytest.raises(HawkError, match="'x'"):
        bound.rebind(x=np.zeros(N, dtype=np.float32))


def test_a_rebind_cannot_change_the_sample_count(loaded):
    bound = hawk.plan(loaded["axpb"]).bind(**_axpb_planes())
    with pytest.raises(HawkError, match="rebind cannot change the sample count, bound for 37, got 8"):
        bound.rebind(x=np.zeros(8), y=np.zeros(8))


def test_the_planes_of_a_bound_plan_are_read_only(loaded):
    bound = hawk.plan(loaded["axpb"]).bind(**_axpb_planes())
    with pytest.raises(TypeError):
        bound.planes["x"] = np.zeros(N)


def test_a_closed_plan_refuses_to_launch_and_closing_twice_is_harmless(loaded):
    bound = hawk.plan(loaded["axpb"]).bind(**_axpb_planes())
    bound.close()
    bound.close()
    with pytest.raises(HawkError, match="this plan is closed"):
        bound.launch()


def test_the_counters_of_a_plan_state_the_finished_count_after_a_launch(loaded):
    n = 16
    rate = np.where(np.arange(n) < 6, 2.0, 0.5)
    planes = {"rate": rate, "goal": 1.0, "terminated": np.zeros(n, dtype=bool), "x": np.zeros(n)}
    bound = hawk.plan(loaded["ramp"]).bind(**planes)
    assert bound.counters()["finished"] in (None, 0)
    bound.launch()
    assert bound.counters()["finished"] == int(np.count_nonzero(planes["terminated"])) == 6
    assert set(bound.counters()) == {"finished", "steps", "mix"}


def test_a_finished_counter_the_caller_binds_is_filled_after_every_launch(loaded):
    n = 16
    planes = {"rate": np.full(n, 2.0), "goal": 1.0, "terminated": np.zeros(n, dtype=bool), "x": np.zeros(n),
              "finished_count": np.zeros(1, dtype=np.uint32)}
    bound = hawk.plan(loaded["ramp"]).bind(**planes)
    bound.launch()
    assert int(planes["finished_count"][0]) == n


def test_run_returns_the_plane_itself_for_one_output(loaded):
    x = np.arange(N, dtype=np.float64)
    y = hawk.plan(loaded["axpb"]).run(x=x, a=3.0, b=-2.0)
    assert isinstance(y, np.ndarray) and y.shape == (N,)
    np.testing.assert_array_equal(y, 3.0 * x - 2.0)


def test_run_returns_a_dict_by_name_for_several_outputs(loaded):
    x = np.arange(N, dtype=np.float64)
    got = hawk.plan(loaded["two_outputs"]).run(x=x)
    assert sorted(got) == ["v", "y"]
    np.testing.assert_array_equal(got["y"], 2.0 * x)
    np.testing.assert_array_equal(got["v"], np.stack([x, 2.0 * x, 3.0 * x]))


def test_run_makes_a_vector_output_of_a_batch_as_wide_as_itself(loaded):
    got = hawk.plan(loaded["two_outputs"]).run(x=np.arange(3.0))
    np.testing.assert_array_equal(got["v"], np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6.0]]))


def test_run_writes_an_output_the_caller_binds_in_place_and_returns_it_as_given(loaded):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    got = hawk.plan(loaded["axpb"]).run(x=x, a=3.0, b=-2.0, y=y)
    assert got is y
    np.testing.assert_array_equal(y, 3.0 * x - 2.0)


def test_run_returns_the_cell_of_a_sum_over_samples(loaded):
    x = np.arange(N, dtype=np.float64)
    got = hawk.plan(loaded["total"]).run(x=x)
    assert got.shape == (1,) and got[0] == x.sum()


def test_run_makes_a_scatter_target_as_long_as_the_lanes(loaded):
    x = np.arange(12, dtype=np.float64)
    lane = (np.arange(12) % 5).astype(np.int32)
    got = hawk.plan(loaded["tally"], hawk.HostSerial).run(x=x, lane=lane)
    np.testing.assert_array_equal(got, np.bincount(lane, weights=x, minlength=12))


@pytest.mark.parametrize("structure", [hawk.HostSerial, hawk.HostTeam], ids=["serial", "team"])
def test_every_launch_of_a_split_kernel_equals_the_oracle_and_a_fresh_bind(loaded, structure):
    """``share`` divides by a sum over every sample, so it runs as stages around a cell the caller never binds. The cell
    the plan keeps between launches must not leak: each launch, and the launch after a rebind, equals ``x / sum(x)`` for
    the inputs of that launch and equals a fresh bind, bit for bit."""
    assert loaded["share"].sidecar.get("stages")
    plan = hawk.plan(loaded["share"], structure)
    x, y = np.arange(1, N + 1, dtype=np.float64), np.zeros(N)
    bound = plan.bind(x=x, y=y)
    launches = [x, x, x * 3.0]
    for i, inputs in enumerate(launches):
        if i == 2:
            bound.rebind(x=inputs)
        bound.launch()
        fresh = np.zeros(N)
        plan.bind(x=inputs, y=fresh).launch()
        np.testing.assert_array_equal(y, inputs / inputs.sum())
        assert y.tobytes() == fresh.tobytes()


def test_run_hands_numpy_arrays_back_on_request(loaded):
    got = hawk.plan(loaded["two_outputs"]).run(x=np.arange(N, dtype=np.float64), returns="numpy")
    assert all(type(value) is np.ndarray for value in got.values())


def test_run_refuses_a_conversion_this_build_does_not_have(loaded):
    with pytest.raises(HawkError, match="returns='torch'"):
        hawk.plan(loaded["axpb"]).run(x=np.zeros(N), a=1.0, b=0.0, returns="torch")


def test_run_refuses_an_input_the_caller_leaves_out_by_name(loaded):
    with pytest.raises(HawkError, match="argument 'x'"):
        hawk.plan(loaded["axpb"]).run(a=1.0, b=0.0, n_samples=4)


def test_a_plan_over_a_plan_is_refused_by_name(loaded):
    with pytest.raises(HawkError, match="already carries its structure"):
        hawk.plan(hawk.plan(loaded["axpb"]), structure=hawk.HostSerial)


def test_a_plan_over_something_that_is_not_a_loaded_kernel_is_refused_by_name():
    with pytest.raises(HawkError, match="takes a loaded kernel"):
        hawk.plan(axpb)


def test_the_device_structure_is_refused_by_name(loaded):
    with pytest.raises(HawkError, match="hawk.DeviceKernel: the device runtime is not in this build"):
        hawk.plan(loaded["axpb"], hawk.DeviceKernel)


def test_a_structure_that_is_no_marker_is_refused_by_name(loaded):
    with pytest.raises(HawkError, match="is none of hawk.HostTeam, hawk.HostSerial, hawk.DeviceKernel"):
        hawk.plan(loaded["axpb"], "host_team")


# -- deploy ------------------------------------------------------------------------------------------------------------
def test_deploy_of_one_kernel_returns_a_plan_that_runs_it(cache_dir):
    plan = hawk.deploy(axpb, cache_dir=cache_dir)
    x = np.arange(N, dtype=np.float64)
    np.testing.assert_array_equal(plan.run(x=x, a=3.0, b=-2.0), 3.0 * x - 2.0)


def test_deploy_of_a_list_returns_a_tuple_of_plans_in_order(cache_dir):
    plans = hawk.deploy([total, axpb, two_outputs], cache_dir=cache_dir)
    assert isinstance(plans, tuple)
    assert [plan.kernel.sidecar["kernel"] for plan in plans] == ["total", "axpb", "two_outputs"]
    x = np.arange(N, dtype=np.float64)
    assert plans[0].run(x=x)[0] == x.sum()
    np.testing.assert_array_equal(plans[2].run(x=x)["y"], 2.0 * x)


def test_a_second_deploy_of_the_same_kernel_reuses_the_bundle(cache_dir):
    first = hawk.deploy(axpb, cache_dir=cache_dir)
    second = hawk.deploy(axpb, cache_dir=cache_dir)
    assert first.kernel.path == second.kernel.path


def test_a_deployed_plan_takes_no_structure(cache_dir):
    with pytest.raises(TypeError, match="structure"):
        hawk.deploy(axpb, cache_dir=cache_dir, structure=hawk.HostSerial)
    with pytest.raises(HawkError, match="already carries its structure"):
        hawk.plan(hawk.deploy(axpb, cache_dir=cache_dir), structure=hawk.HostSerial)


def test_deploy_for_a_device_target_is_refused_by_name(cache_dir):
    with pytest.raises(HawkError, match="target 'cuda': the device runtime is not in this build"):
        hawk.deploy(axpb, targets=("cuda",), cache_dir=cache_dir)


def test_deploy_in_the_banded_type_is_refused_on_the_host(cache_dir):
    with pytest.raises(HawkError, match="'banded' is a device type"):
        hawk.deploy(axpb, scalar_type="banded", cache_dir=cache_dir)


def test_deploy_refuses_the_contraction_setting_by_name(cache_dir):
    with pytest.raises(HawkError, match="contraction is not a build setting"):
        hawk.deploy(axpb, contraction="gemm", cache_dir=cache_dir)


# -- arrays ------------------------------------------------------------------------------------------------------------
class _HostCapsule:
    """An array that exports DLPack and nothing else, over host memory."""

    def __init__(self, array):
        self._array = array

    def __dlpack__(self, **kwargs):
        return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


class _Foreign(_HostCapsule):
    """An array of another library: it exports the buffer protocol AND DLPack, and its ``reshape`` is not NumPy's."""

    def __buffer__(self, flags):
        return memoryview(self._array)

    def reshape(self, *, shape):
        raise AssertionError("an array that is not a NumPy array is handed over as it is: NumPy's reshape is never called on it")


class _OnADevice(_HostCapsule):
    def __dlpack_device__(self):
        return (2, 0)


def test_an_array_of_another_library_that_exports_both_protocols_runs_through_dlpack(loaded):
    base = np.arange(N, dtype=np.float64)
    y = np.zeros(N)
    hawk.run(loaded["axpb"], x=_Foreign(base), a=3.0, b=-2.0, y=_Foreign(y))
    np.testing.assert_array_equal(y, 3.0 * base - 2.0)
    out = np.zeros(N)
    hawk.plan(loaded["axpb"]).bind(x=_Foreign(base), a=3.0, b=-2.0, y=_Foreign(out)).launch()
    np.testing.assert_array_equal(out, 3.0 * base - 2.0)


def test_a_plain_buffer_is_refused_by_name(loaded):
    import array

    with pytest.raises(HawkError, match=r"a plain buffer — wrap it in numpy.asarray\(x"):
        hawk.run(loaded["axpb"], x=array.array("d", [1.0, 2.0]), a=1.0, b=0.0, y=np.zeros(2))


def test_asarray_shares_the_memory_of_an_array_and_copies_a_sequence():
    base = np.arange(4.0)
    assert hawk.asarray(base) is base
    made = hawk.asarray([1.0, 2.0])
    np.testing.assert_array_equal(made, [1.0, 2.0])


def test_to_numpy_reads_an_array_that_exports_only_dlpack_without_a_copy():
    base = np.arange(5.0)
    view = hawk.to_numpy(_HostCapsule(base))
    base[2] = 99.0
    assert view[2] == 99.0 and np.shares_memory(view, base)


def test_to_numpy_refuses_a_device_array_and_a_non_array_by_name():
    with pytest.raises(HawkError, match="lives on a device; the device runtime is not in this build"):
        hawk.to_numpy(_OnADevice(np.arange(2.0)))
    with pytest.raises(HawkError, match="exports no DLPack"):
        hawk.to_numpy([1, 2])


def test_synchronize_returns_with_nothing_pending():
    assert hawk.synchronize() is None


@pytest.mark.parametrize("shape", [(), (7,), (3, 4)])
def test_banded_unpack_of_banded_pack_returns_the_float32_exact_values_in_their_shape(shape):
    rng = np.random.default_rng(5)
    values = rng.uniform(-8.0, 8.0, shape).astype(np.float32).astype(np.float64)
    cells = hawk.banded_pack(values)
    assert cells.dtype == np.uint64 and cells.shape == shape
    back = hawk.banded_unpack(cells)
    assert back.dtype == np.float64 and back.shape == shape
    np.testing.assert_array_equal(back, values)


def _mix_after(plan_dir, words, preload):
    """The mix plane of the one-trip-per-word walker after one launch per word, over a plane preloaded with ``preload``."""
    from test_dispatch_mix_and_guard import K
    from test_dispatch_mix_and_guard import N as lanes

    rng = np.random.default_rng(12)
    classes = rng.integers(0, K, lanes).astype(np.int32)
    mix = np.full(K, preload, dtype=np.uint32)
    word = np.zeros(1, dtype=np.int64)
    bound = hawk.plan(hawk.load(plan_dir, "walker")).bind(
        k=classes, bs=rng.uniform(0.5, 1.0, (K, lanes)), terminated=np.zeros(lanes, dtype=bool), x=np.zeros(lanes),
        finished_count=np.zeros(1, dtype=np.uint32), fused_steps=word, dispatch0_mix=mix)
    for w in words:
        word[0] = w
        bound.launch()
    return mix, np.bincount(classes, minlength=K)


@pytest.fixture(scope="module")
def walker_dir(tmp_path_factory, cache_dir):
    from test_dispatch_mix_and_guard import walker

    return build_bundle([walker], tmp_path_factory.mktemp("walker_mix"), targets=("host",), cache_dir=cache_dir).directory


@pytest.mark.parametrize("preload", [0, 5])
def test_the_mix_plane_adds_what_every_launch_counted_to_its_preload(walker_dir, preload):
    one, hist = _mix_after(walker_dir, [7], preload)
    seven, _ = _mix_after(walker_dir, [1] * 7, preload)
    np.testing.assert_array_equal(one, preload + 7 * hist)       # oracle: the classes' bincount, once per trip
    np.testing.assert_array_equal(seven, one)
