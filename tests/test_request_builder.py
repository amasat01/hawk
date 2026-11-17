# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The request one kernel sends to the lowering library, against a hand-written request for each body kind.

The facts answers and the compiler identity are stood in for (they come from the library and the compile driver); every other
member is read from the kernel and the build spec by the builder under test. A sink is the position of its node in the dump the
bridge writes, so each test also checks the handles it states are nodes of that dump."""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from _specs import default_kind, validation_spec
from hawk import Mutable, Param, Scalar, Terminated, Vector, kernel
from hawk.artifact import bridge, request
from hawk.ext import Guard, Kind
from hawk.targets import build_spec

pytest.importorskip("_osprey")

KIND = {"active_set": False, "guard": "Default", "lanes": None, "sink": "Plain"}
BODY = {"exempt": ["fused_steps"]}
TERMINATED = {"dtype": "Bool", "name": "terminated", "role": "Terminated", "shape": []}


@kernel
def axpb(x: Scalar, y: Mutable[Scalar]):
    y = 2.0 * x + 1.0


@kernel
def first(x: Scalar, a: Mutable[Vector[3]]):
    a = m.vec(m.sin(x), m.cos(x), x * 0.5)


@kernel
def second(a: Vector[3], x: Scalar, b: Mutable[Scalar]):
    b = m.exp(-m.dot(a, a)) + x


@kernel
def lane_energy(x: Vector[3], sq: Mutable[Scalar]):
    sq = m.dot(x, x)


@kernel
def lane_speed(x: Vector[3], k: Param, spd: Mutable[Scalar]):
    spd = k * m.dot(x, x)


@kernel
def standard(x: Scalar, mu: Mutable[Scalar], z: Mutable[Scalar], t: Mutable[Scalar]):
    mu = hawk.mean(x)
    v = hawk.mean((x - mu) * (x - mu))
    z = (x - mu) / m.sqrt(v)
    t = hawk.sum(z * z)


@kernel
def masked(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x + 1.0


@kernel
def stepping(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def stepping_one(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@pytest.fixture(autouse=True)
def stood_in(monkeypatch):
    monkeypatch.setattr(request, "_facts", lambda target, compiler, stated: "FACTS-" + target)
    monkeypatch.setattr(request, "_compile_plan", lambda target, spec, options: {"identity": "cc||line||digest", "meta": {"device_target": "cubin"}})


def built(target_kernel, *, kind=None, targets=("host",), spec=None, name=None, **options):
    """``(request, handles)`` of the kernel: the request the builder makes and the position of each of the kernel's sinks."""
    spec = (spec or build_spec(targets, "")).with_settings(scalar="float64")
    dump, _, position = bridge.dump(request.sinks_of(target_kernel))
    out = request.request(target_kernel, name=name or target_kernel.name, scalar="float64", kind=kind or target_kernel.kind,
                          targets=targets, spec=spec, position=position, options=options)
    assert all(0 <= at < len(dump.splitlines()) for at in position.values())
    return out


def test_a_plain_kernel():
    assert built(axpb) == {
        "v": 1, "name": "axpb", "scalar": "Float64", "targets": ["Host"], "kind": KIND,
        "body": {"Plain": {"name": "axpb", "sinks": [5], "declared": [], **BODY}},
        "planes": ["x", "y"], "steps": 1, "lane_space": [{}], "facts": {"Host": "FACTS-host"},
    }


def test_a_chain_names_each_member_as_a_body():
    assert built(hawk.chain(first, second, name="c12")) == {
        "v": 1, "name": "c12", "scalar": "Float64", "targets": ["Host"], "kind": KIND,
        "body": {"Chain": [{"name": "first", "sinks": [12], "declared": [], **BODY},
                           {"name": "second", "sinks": [18], "declared": [], **BODY}]},
        "planes": ["x", "b"], "steps": 1, "lane_space": [{}], "facts": {"Host": "FACTS-host"},
    }


def test_a_kernel_that_splits_declares_the_companion_plane_of_each_compensated_sum_and_the_row_it_compensates():
    assert built(standard) == {
        "v": 1, "name": "standard", "scalar": "Float64", "targets": ["Host"], "kind": KIND,
        "body": {"Plain": {"name": "standard", "sinks": [4, 16, 18], "exempt": ["fused_steps"], "declared": [
            {"role": "AccumOut", "name": "mu_comp", "dtype": "F64", "shape": [], "compensates": "mu"},
            {"role": "AccumOut", "name": "t_comp", "dtype": "F64", "shape": [], "compensates": "t"}]}},
        "planes": ["x", "mu", "z", "t"], "steps": 1, "lane_space": [{}], "facts": {"Host": "FACTS-host"},
    }


@kernel
def add_vec(key: hawk.Index, v: Vector[3], acc: Mutable[Vector[3], "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += v


@kernel
def add_one(key: hawk.Index, x: Scalar, acc: Mutable[Scalar, "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += x


def test_a_sink_with_named_axes_states_its_payload():
    """The plane is bound (payload, then named axes): the payload is in the declaration only, so the request states it."""
    assert built(add_vec)["sink_axes"] == [{"name": "acc", "axes": [{"label": "bin", "extent": 9}], "payload": [3]}]
    assert built(add_one)["sink_axes"] == [{"name": "acc", "axes": [{"label": "bin", "extent": 9}], "payload": []}]


def test_a_placed_build_states_its_wide_form():
    spec = build_spec(("host",), "").with_settings(wide_form="thread", fold_k=4, warps=2, team_threads=64)
    assert built(axpb, spec=spec, layout_sizes_override=(1, 2, 3, 4, 5))["settings"] == {
        "wide_form": "Thread", "fold_k": 4, "warps": 2, "team_threads": 64, "exports": [1, 2, 3, 4, 5]}


def test_an_active_set_kind_names_its_masks_and_the_map():
    got = built(masked, kind=Kind("active_probe", guard=Guard(active_set=True)))
    assert got["kind"] == {"active_set": True, "guard": {"Masks": ["terminated"]}, "lanes": None, "sink": "Plain"}
    assert got["body"] == {"Plain": {"name": "masked", "sinks": [3], "declared": [TERMINATED], **BODY}}


def test_a_two_path_kernel_sends_its_single_step_as_a_second_body():
    got = built(stepping)
    assert got["body"] == {"Plain": {"name": "stepping", "sinks": [28, 30, 32, 36], "declared": [TERMINATED], **BODY}}
    assert got["one_step"] == {"name": "stepping", "sinks": [41, 44, 47, 49], "declared": [TERMINATED], **BODY}
    assert (got["steps"], got["steps_max"]) == ("auto", hawk.trace.steps.AUTO_K_MAX)


def test_a_finishing_kernel_leaves_the_finished_count_plane_to_the_library():
    got = built(stepping_one)
    assert got["body"] == {"Plain": {"name": "stepping_one", "sinks": [5, 8, 11, 14], "declared": [TERMINATED], **BODY}}
    assert got["steps"] == 1 and "one_step" not in got and got["lane_space"] == [{}]


def test_a_device_build_states_the_image_format_and_the_device_facts():
    assert built(axpb, targets=("cuda",)) == {
        "v": 1, "name": "axpb", "scalar": "Float64", "targets": ["Device"], "kind": KIND, "image_format": "Cubin",
        "body": {"Plain": {"name": "axpb", "sinks": [5], "declared": [], **BODY}},
        "planes": ["x", "y"], "steps": 1, "lane_space": [{}], "facts": {"Device": "FACTS-cuda"},
    }


def test_the_c_target_is_refused_by_name():
    with pytest.raises(hawk.HawkError, match="target 'c' is refused"):
        built(axpb, targets=("c",))


def test_a_build_under_a_card_of_its_own_is_refused_by_name():
    import dataclasses

    spec = build_spec(("host",), "")
    held = spec.by_target["host"]
    own = dataclasses.replace(spec, by_target={"host": {**held, "form": {**held["form"], "lazy_leaf_width": held["form"]["lazy_leaf_width"] + 1}}})
    with pytest.raises(hawk.HawkError, match="card of its own"):
        built(axpb, spec=own)


def test_thresholds_forced_by_wide_override_are_a_card_of_its_own():
    with pytest.raises(hawk.HawkError, match="card of its own"):
        built(axpb, targets=("cuda",), spec=build_spec(("cuda",), "sm_80", wide_override={"WARPS": 2}))


def test_the_host_compiler_line_is_the_real_one_unless_the_spec_names_another_machine(monkeypatch):
    import dataclasses
    from types import MappingProxyType

    monkeypatch.undo()
    monkeypatch.setattr(request.TARGETS["host"], "compiler", lambda: "cc")
    monkeypatch.setattr(request._tc, "compiler_identity", lambda compiler: "cc||line||digest")
    spec = build_spec(("host",), "")
    assert request._host_line(spec, {}) == "cc || line || digest"
    other = "x86_64" if spec.machine.isa == "arm64" else "arm64"
    machine = dataclasses.replace(spec.machine, isa=other, facts=MappingProxyType({**spec.machine.facts, "host_profile": "native-vector-math"}))
    line = request._host_line(dataclasses.replace(spec, machine=machine), {})
    assert line == ("g++ || g++ (GCC) 13.2.0 || 0" if other == "x86_64" else "cc || line || digest")


#: The first line the device compiler driver prints for ``--version``: the same for every release (the release is on a later line).
DRIVER_LINE = "/usr/local/cuda/bin/nvcc||nvcc: NVIDIA (R) Cuda compiler driver||0f3c"


def test_a_device_build_states_the_compilers_release_from_its_plan(monkeypatch):
    plan = {"identity": DRIVER_LINE, "release": "12.9", "meta": {"device_target": "fatbin"}}
    monkeypatch.setattr(request, "_compile_plan", lambda target, spec, options: plan)
    spec = build_spec(("cuda",), "")
    stated, line, image = request._device(spec, {"device_arch": "sm_61"})
    assert stated == {"device_arch": spec.machine.arch or "sm_61", "device_release": "12.9"}
    assert line == "/usr/local/cuda/bin/nvcc || nvcc: NVIDIA (R) Cuda compiler driver || 0f3c" and image == "fatbin"
    # a plan whose probe could not tell the release states none: the library then refuses the line by name
    monkeypatch.setattr(request, "_compile_plan", lambda target, spec, options: {**plan, "release": None})
    assert "device_release" not in request._device(spec, {})[0]


def test_the_library_answers_a_driver_line_with_a_stated_release_and_refuses_it_without(monkeypatch):
    import json

    monkeypatch.undo()                                  # the real facts question
    line = request._identity(DRIVER_LINE)
    said = json.loads(request._facts("cuda", line, json.dumps({"device_release": "12.6"})))
    named = json.loads(request._facts("cuda", request._identity("nvcc||Cuda compilation tools, release 12.6, V12.6.77||0"), "{}"))
    assert said["device_build"] == named["device_build"] and said["device_build"]["compiler"] == "nvcc 12.6"
    with pytest.raises(Exception, match="names no tool or no release, and no device_release is stated"):
        request._facts("cuda", line, "{}")


def test_the_device_plans_release_is_major_minor_or_nothing():
    from ospreys.cuda.backend import _release

    assert _release((12, 9)) == "12.9" and _release(None) is None
