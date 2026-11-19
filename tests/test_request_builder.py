# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The request one kernel sends to the lowering library, against a hand-written request for each body kind.

The machine answers are stood in for by plain text (they come from the OSPREYs); every other member is read from the kernel and
the build settings by the builder under test. A sink is the position of its node in the text of the dump, so each test also
checks that the handles a request states are lines of that dump.
"""

from __future__ import annotations

import dataclasses


import hawk
import hawk.math as m
from hawk import Mutable, Param, Scalar, Terminated, Vector, _ir, kernel
from hawk.artifact import request
from hawk.build_settings import default_settings
from hawk.ext import Guard, Kind

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
def stepping(
    dt: Param,
    t_final: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
    t: Mutable[Scalar],
):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def stepping_one(
    dt: Param,
    t_final: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
    t: Mutable[Scalar],
):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel
def add_vec(key: hawk.Index, v: Vector[3], acc: Mutable[Vector[3], "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += v


@kernel
def add_one(key: hawk.Index, x: Scalar, acc: Mutable[Scalar, "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += x


def built(target, *, targets=("host",), settings=None, kind=None, sizes=None):
    """The request of ``target`` for ``targets``, with each machine answer stood in for by ``"M-<target>"``."""
    roots = request.sinks_of(target)
    copies = _ir.adopt(_ir.new_context(), roots)
    position = dict(zip(roots, _ir.dump_numbers(copies)))
    lines = len(_ir.dump_text(copies).splitlines())
    assert all(0 <= at < lines for at in position.values())
    return request.request(
        target,
        name=target.name,
        scalar="float64",
        kind=kind or target.kind,
        targets=targets,
        settings=settings or default_settings(),
        position=position,
        machines={t: "M-" + t for t in targets},
        sizes=sizes,
    )


def sinks_at(target, member):
    """The positions, in the dump of the request of ``target``, of the sinks of ``member``."""
    roots = request.sinks_of(target)
    copies = _ir.adopt(_ir.new_context(), roots)
    position = dict(zip(roots, _ir.dump_numbers(copies)))
    return [position[sink] for sink in member.sinks]


def plain(name, sinks, declared=()):
    """The ``Plain`` body a request states for a kernel ``name`` whose sinks sit at ``sinks``."""
    return {"Plain": {"name": name, "sinks": sinks, "declared": list(declared), **BODY}}


def test_a_plain_kernel_states_one_body_its_planes_one_step_and_its_machine():
    assert built(axpb) == {
        "v": 1, "name": "axpb", "scalar": "Float64", "targets": ["Host"], "kind": KIND,
        "body": plain("axpb", [5]), "planes": ["x", "y"], "steps": 1, "lane_space": [{}],
        "machine": {"Host": "M-host"},
    }  # fmt: skip


def test_a_chain_names_each_member_as_a_body():
    chained = hawk.chain(first, second, name="c12")
    members = chained.members
    assert built(chained) == {
        "v": 1, "name": "c12", "scalar": "Float64", "targets": ["Host"], "kind": KIND,
        "body": {"Chain": [
            {"name": "first", "sinks": sinks_at(chained, members[0]), "declared": [], **BODY},
            {"name": "second", "sinks": sinks_at(chained, members[1]), "declared": [], **BODY},
        ]},
        "planes": ["x", "b"], "steps": 1, "lane_space": [{}], "machine": {"Host": "M-host"},
    }  # fmt: skip


def test_a_kernel_with_compensated_sums_states_no_companion_plane_and_no_compensated_row():
    """The library derives the companion plane of a compensated sum from the walk: the request declares nothing for it."""
    assert built(standard)["planes"] == ["x", "mu", "z", "t"]
    assert built(standard)["body"] == plain("standard", sinks_at(standard, standard))


def test_a_sink_with_named_axes_states_its_payload():
    """The plane is bound (payload, then named axes): the payload is in the declaration only, so the request states it."""
    assert built(add_vec)["sink_axes"] == [
        {"name": "acc", "axes": [{"label": "bin", "extent": 9}], "payload": [3]}
    ]
    assert built(add_one)["sink_axes"] == [
        {"name": "acc", "axes": [{"label": "bin", "extent": 9}], "payload": []}
    ]


def test_a_placed_build_states_the_settings_that_differ_from_the_defaults_and_the_layout_sizes():
    settings = dataclasses.replace(
        default_settings(), wide_form="thread", fold_k=4, warps=2, team_threads=64
    )
    assert built(axpb, settings=settings, sizes=(1, 2, 3, 4, 5))["settings"] == {
        "wide_form": "Thread", "fold_k": 4, "warps": 2, "team_threads": 64, "exports": [1, 2, 3, 4, 5],
    }  # fmt: skip
    assert "settings" not in built(axpb)


def test_an_active_set_kind_names_its_masks_and_the_map():
    got = built(masked, kind=Kind("active_probe", guard=Guard(active_set=True)))
    assert got["kind"] == {
        "active_set": True,
        "guard": {"Masks": ["terminated"]},
        "lanes": None,
        "sink": "Plain",
    }
    assert got["body"] == plain("masked", [3], [TERMINATED])


def test_a_two_path_kernel_sends_its_single_step_as_a_second_body():
    got = built(stepping)
    assert got["body"] == plain("stepping", sinks_at(stepping, stepping), [TERMINATED])
    assert got["one_step"] == {
        "name": "stepping", "sinks": sinks_at(stepping, stepping.one_step), "declared": [TERMINATED], **BODY,
    }  # fmt: skip
    assert (got["steps"], got["steps_max"]) == ("auto", hawk.trace.steps.AUTO_K_MAX)


def test_a_finishing_kernel_leaves_the_finished_count_plane_to_the_library():
    got = built(stepping_one)
    assert got["body"] == plain(
        "stepping_one", sinks_at(stepping_one, stepping_one), [TERMINATED]
    )
    assert got["steps"] == 1 and "one_step" not in got and got["lane_space"] == [{}]


def test_a_device_build_states_the_device_target_and_its_machine():
    got = built(axpb, targets=("cuda",))
    assert (got["targets"], got["machine"]) == (["Device"], {"Device": "M-cuda"})
    assert got["body"] == plain("axpb", [5])
