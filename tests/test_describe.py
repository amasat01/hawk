# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``describe`` answers, before any build, the sidecar a build writes.

Per kernel shape (an auto dispatch group, a kernel that splits into stages, a fused-steps kernel, a derived kernel) the
document equals the built sidecar on every member both have; the members only a compile gives (``tool``, the unit's
measured launch figures) are absent from it. One row runs ``describe`` with every tool hidden.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from conftest import sidecar_of

import hawk
import hawk.math as M
from hawk import HawkError, Index, Mutable, Scalar, Terminated, Vector, runtime
from hawk.artifact import build_bundle, describe
from hawk.diff import vjp

#: The sidecar members that exist only after a compile.
AFTER_A_COMPILE = ("tool",)


@hawk.kernel(steps=1)
def branchy(k: Index, bs: Vector[3], y: Mutable[Scalar]):
    y = y + M.dispatch(
        k, [bs[0] * 1.0, M.sqrt(bs[1]) * 2.0, M.exp(bs[2]) * 3.0], strategy="auto"
    )


@hawk.kernel
def share(x: Scalar, z: Mutable[Scalar]):
    z = x / hawk.sum(x)


@hawk.kernel(steps=1)
def osc(
    dt: hawk.Param,
    limit: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
):
    xn = x + dt * v
    v = v - dt * x
    x = xn
    terminated = xn >= limit


@hawk.kernel
def square(x: Scalar, y: Mutable[Scalar]):
    y = x * x


#: The reverse-mode derivative of ``square``, named as a kernel of its own.
DERIVED = hawk.Kernel("square_vjp", vjp(square))

#: Each shape: the kernels one bundle holds (a derivative is built with its primal) and the kernel that is described.
STEPPED = hawk.steps(osc, 4)
SUBJECTS = {
    "an auto dispatch group": ([branchy], "branchy"),
    "a kernel that splits into stages": ([share], "share"),
    "a fused-steps kernel": ([STEPPED], STEPPED.name),
    "a derived kernel": ([square, DERIVED], "square_vjp"),
}


def _same_where_both_have(described: dict, built: dict) -> None:
    for member, value in described.items():
        if member in built:
            assert built[member] == value, member
    for member in built:
        assert member in described or member in AFTER_A_COMPILE or member == "launch", (
            member
        )


@pytest.mark.parametrize("shape", SUBJECTS)
def test_describe_equals_the_built_sidecar_on_every_member_both_have(
    shape, tmp_path, cache_dir
):
    kernels, name = SUBJECTS[shape]
    described = describe(kernels, targets=("host",))[name]
    bundle = build_bundle(
        kernels, tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    _same_where_both_have(described, sidecar_of(bundle, name))
    assert not set(AFTER_A_COMPILE) & set(described)


def test_an_auto_group_states_the_cost_of_each_branch():
    (group,) = describe([branchy], targets=("host",))["branchy"]["dispatch"]
    cost = group["cost"]
    assert "fused_vectorised" not in cost  # stated only together with each branch's own verdict
    assert len(cost["branches"]) == 3
    assert all(
        branch["arith32"] >= 0 and branch["stores"] is not None
        for branch in cost["branches"]
    )


def test_the_planes_a_kernel_writes_and_the_element_count_of_each_value_are_stated():
    document = describe([osc], targets=("host",))["osc"]
    assert document["planes_written"] == ["terminated", "v", "x"]
    assert document["element_counts"] == {
        "dt": 1,
        "limit": 1,
        "terminated": 1,
        "x": 1,
        "v": 1,
        "finished_count": 1,
    }


def test_a_derived_kernel_states_the_group_and_wire_of_each_joined_plane_name():
    names = describe([square, DERIVED], targets=("host",))["square_vjp"].get(
        "plane_names", {}
    )
    for plane, parts in names.items():
        assert plane.endswith(parts["wire"]) and parts["group"] in plane


def test_a_host_only_build_states_the_dispatch_cost_record_with_nulls(
    tmp_path, cache_dir
):
    bundle = build_bundle(
        [square], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    assert runtime.dispatch_costs(sidecar_of(bundle, "square")) == {
        "launch_us": None,
        "plane_us_per_byte": None,
        "local_over_plane": None,
    }


def test_describe_refuses_what_build_refuses_by_the_same_words():
    with pytest.raises(HawkError, match="served by no OSPREY"):
        describe([square], targets=("nowhere",))
    with pytest.raises(HawkError, match="at least one"):
        describe([])


def test_describe_answers_with_every_tool_hidden(tmp_path):
    """A caller that constructs kernels has no toolchain: the lowering takes the machine's stated facts, no compiler is run."""
    code = textwrap.dedent(
        """
        import hawk
        from hawk import Mutable, Scalar
        from hawk.artifact import describe

        @hawk.kernel
        def square(x: Scalar, y: Mutable[Scalar]):
            y = x * x

        print(sorted(describe([square], targets=("host",))["square"])[:3])
        """
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(hawk.__file__)))
    env = {
        **os.environ,
        "PATH": str(tmp_path),
        "HAWK_CACHE_DIR": str(tmp_path / "cache"),
        "PYTHONPATH": os.pathsep.join(
            filter(None, [root, os.environ.get("PYTHONPATH")])
        ),
    }
    for tool in ("CXX", "HAWK_CXX", "CC", "CUDACXX", "CUDA_PATH"):
        env.pop(tool, None)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr[-1500:]
