# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``describe`` answers, before any build, the sidecar a build writes.

Per kernel shape (an auto dispatch group, a kernel that splits into stages, a fused-steps kernel, a derived kernel) the
document equals the built sidecar on every member both have; the members only a compile gives (``tool``, the unit's
measured launch figures) are absent from it. One row runs ``describe`` with every tool hidden.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
from conftest import sidecar_of

import hawk
import hawk.math as M
from hawk import HawkError, Index, Matrix, Mutable, Scalar, Terminated, Vector, runtime
from hawk.artifact import build_bundle, describe
from hawk.artifact.bundle import digest_bytes
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


def _without_stage_documents(value):
    """``stages`` as the sidecar file states them: ``describe`` adds each stage's own unit document, a build writes it as its own file."""
    if isinstance(value, list) and all(
        isinstance(v, dict) and "sidecar" in v for v in value
    ):
        return [{k: v for k, v in stage.items() if k != "sidecar"} for stage in value]
    return value


def _same_where_both_have(described: dict, built: dict) -> None:
    for member, value in described.items():
        if member in built:
            assert built[member] == _without_stage_documents(value), member
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
    assert (
        "fused_vectorised" not in cost
    )  # stated only together with each branch's own verdict
    assert len(cost["branches"]) == 3
    assert all(
        branch["arith32"] >= 0 and branch["stores"] is not None
        for branch in cost["branches"]
    )


def test_the_planes_a_kernel_writes_and_the_shape_of_each_slot_are_stated():
    document = describe([osc], targets=("host",))["osc"]
    assert document["planes_written"] == ["terminated", "v", "x"]
    assert document["slot_shapes"] == {
        "dt": [],
        "limit": [],
        "terminated": [],
        "x": [],
        "v": [],
        "finished_count": [],
    }


@hawk.kernel
def shaped(m: Matrix[2, 3], v: Vector[3], x: Scalar, y: Mutable[Vector[2]]):
    y = m @ v + x


def test_each_slot_states_its_shape_a_matrix_as_rows_and_columns_a_vector_as_its_length_a_scalar_as_nothing():
    shapes = describe([shaped], targets=("host",))["shaped"]["slot_shapes"]
    assert shapes == {"m": [2, 3], "v": [3], "x": [], "y": [2]}


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


def test_every_slot_states_its_wire_dtype_uniforms_and_lookup_planes_included():
    document = describe([osc], targets=("host",))["osc"]
    slots = [name for _role, name in document["arg_spec"]]
    assert (
        sorted(document["arg_dtypes"])
        == sorted(slots)
        == sorted(document["slot_shapes"])
    )
    assert document["arg_dtypes"]["dt"] == "float64"  # a uniform


def test_every_stage_of_a_split_kernel_carries_the_document_of_its_own_unit(
    tmp_path, cache_dir
):
    document = describe([share], targets=("host",))["share"]
    bundle = build_bundle(
        [share], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    built = {artifact.name: artifact.sidecar for artifact in bundle.artifacts}
    assert len(document["stages"]) > 1
    for stage in document["stages"]:
        assert stage["sidecar"]["kernel"] == stage["kernel"]
        _same_where_both_have(stage["sidecar"], built[stage["kernel"]])


def test_a_kernel_that_does_not_split_has_no_stage_documents():
    assert "stages" not in describe([square], targets=("host",))["square"]


def test_the_stated_machine_of_a_target_is_the_one_the_sidecar_of_a_build_carries(
    tmp_path, cache_dir
):
    from hawk.targets import machine

    document = machine("host")
    bundle = build_bundle(
        [square], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    assert document and list(sidecar_of(bundle, "square")["stated"].values()) == [
        document
    ]


def test_asking_the_machine_of_a_target_no_library_serves_is_refused_by_name():
    from hawk.targets import machine

    with pytest.raises(HawkError, match="served by no OSPREY"):
        machine("nowhere")


def test_asking_the_machine_with_every_tool_hidden_raises_toolchain_absent(tmp_path):
    """A machine that lacks the tools of a target says so by the library's reason, as ``build`` does."""
    code = textwrap.dedent(
        """
        from hawk.errors import ToolchainAbsent
        from hawk.targets import machine

        try:
            machine("host")
        except ToolchainAbsent as absent:
            print("absent", bool(str(absent)))
        else:
            print("present")
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
    assert done.stdout.split() == ["absent", "True"]


def _published_unit(tmp_path, cache_dir):
    from hawk.artifact import STAMP_NAME

    bundle = build_bundle(
        [square], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    return bundle, json.loads((bundle.directory / STAMP_NAME).read_text())


def test_every_file_of_a_published_unit_digests_to_its_stamped_value(
    tmp_path, cache_dir
):
    bundle, stamp = _published_unit(tmp_path, cache_dir)
    assert stamp["unit"] == bundle.digest
    published = {path.name for path in bundle.files()}
    assert published == set(stamp["files"])
    for name, digest in stamp["files"].items():
        assert digest_bytes((bundle.directory / name).read_bytes()) == digest, name


def test_a_changed_byte_of_a_published_file_no_longer_digests_to_its_stamp(
    tmp_path, cache_dir
):
    bundle, stamp = _published_unit(tmp_path, cache_dir)
    sidecar = bundle.artifacts[0].sidecar_path
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    assert digest_bytes(sidecar.read_bytes()) != stamp["files"][sidecar.name]


def test_every_stage_names_an_artifact_of_the_one_bundle_and_states_that_artifacts_document(
    tmp_path, cache_dir
):
    document = describe([share], targets=("host",))["share"]
    bundle = build_bundle(
        [share], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    names = {artifact.name for artifact in bundle.artifacts}
    assert {stage["kernel"] for stage in document["stages"]} <= names


#: The host profiles the cpu card of the library lists as exact (its `exact_profiles`).
EXACT_PROFILES = {"native", "portable"}


@pytest.mark.parametrize("profile", ["native", "portable", "native-vector-math"])
def test_the_machine_states_whether_the_resolved_profile_is_exact(
    profile, tmp_path, cache_dir
):
    """Oracle: the cpu card's exact list, written out above."""
    from hawk.targets import machine

    stated = machine("host", options={"host.profile": profile})
    assert stated["asked"]["options"]["profile"] == profile
    assert stated["exact"] is (profile in EXACT_PROFILES)


def test_the_stated_machine_of_a_build_carries_the_exact_word_of_its_resolved_profile(
    tmp_path, cache_dir
):
    from hawk.targets import machine

    stated = machine("host")
    bundle = build_bundle(
        [square], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    assert isinstance(stated["exact"], bool)
    assert list(sidecar_of(bundle, "square")["stated"].values()) == [stated]


@hawk.kernel
def contract(
    w: hawk.Table["row":8, "neuron":6],
    f: hawk.Table["row":8, "sample":5],
    y: Mutable[Scalar],
):
    j, n = M.split_index(6)
    acc = 0.0
    for r in range(8):
        acc = acc + w.at(row=r, neuron=n) * f.at(row=r, sample=j)
    y = acc


def test_a_dense_contraction_states_its_route_and_products_before_and_after_a_build(
    tmp_path, cache_dir
):
    """Oracle: the product the library's own contraction test states for the same planes (A 8x5, B 8x6, C 5x6)."""
    document = describe([contract], targets=("host",))["contract"]
    [found] = document["contractions"]
    assert (found["unit"], found["route"]) == ("contract", "gemm")
    [call] = found["products"]["calls"]
    shape = [
        call[k] for k in ("a_rows", "a_cols", "b_rows", "b_cols", "c_rows", "c_cols")
    ]
    assert shape == [8, 5, 8, 6, 5, 6]
    bundle = build_bundle(
        [contract], tmp_path / "out", targets=("host",), cache_dir=cache_dir
    )
    assert sidecar_of(bundle, "contract")["contractions"] == document["contractions"]


def test_a_kernel_that_is_no_contraction_states_none():
    assert "contractions" not in describe([square], targets=("host",))["square"]


@hawk.kernel(steps=1, time="t", time_step="dt")
def falling(
    dt: hawk.Param, terminated: Terminated, y: Mutable[Scalar], t: Mutable[Scalar]
):
    yn = y - dt
    y = yn
    t = t + dt
    terminated = yn < 0.0


def test_a_kernel_with_a_time_step_states_the_planes_its_refine_entry_reads():
    """Oracle: the planes ``hawk.diff.refine_entry`` reads, taken from its own arg list."""
    from hawk import _ir
    from hawk.diff import refine_entry

    stated = describe([falling], targets=("host",))["falling"]["refine_planes"]
    assert stated == {"t0": "t_0", "tau": "dt_tau"}
    reads = {name for _role, name in _ir.arg_spec(refine_entry(falling).walk)}
    assert set(stated.values()) <= reads


def test_a_kernel_without_a_time_step_states_no_refine_planes():
    assert "refine_planes" not in describe([square], targets=("host",))["square"]


def test_a_dropped_ordered_scatter_key_is_refused_by_name_and_a_clean_status_passes():
    team = {"of": "acc", "status": "acc_status"}
    runtime.check_scatter_status("k", team, [0, 0, 0], 4)
    with pytest.raises(
        HawkError,
        match=r"k: hawk.scatter\(acc, contract='ordered'\) got a key outside \[0, 4\)",
    ):
        runtime.check_scatter_status("k", team, [0, 0, 1], 4)


def test_the_name_of_the_per_cell_parameter_plane_is_a_public_name_of_the_derivative_transforms():
    from hawk import _ir
    from hawk.diff import CELL_PLANE

    assert CELL_PLANE == _ir.CELL_PLANE


@hawk.kernel
def square_again(x: Scalar, y: Mutable[Scalar]):
    y = x * x


@hawk.kernel
def cube(x: Scalar, y: Mutable[Scalar]):
    y = x * x * x


def test_the_body_digest_is_the_identity_of_the_traced_body_alone(tmp_path, cache_dir):
    """Same body under two names: equal body digests and different unit digests (the built bundle's, which covers the files'
    names); a changed operation changes the body digest."""
    assert square.body_digest == square_again.body_digest
    assert len(square.body_digest) == 64
    assert cube.body_digest != square.body_digest
    first = build_bundle(
        [square], tmp_path / "a", targets=("host",), cache_dir=cache_dir
    )
    second = build_bundle(
        [square_again], tmp_path / "b", targets=("host",), cache_dir=cache_dir
    )
    assert first.digest != second.digest
    with pytest.raises(AttributeError):
        square.body_digest = "x"


def _plain_body(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x * 2.0


def _same_body_again(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    y = x * 2.0


def test_a_kind_changes_the_lowered_unit_digest_and_not_the_body_digest():
    """Same body under the default kind and an active-set kind: the traced body is the same, the lowered unit also reads the map."""
    from hawk.ext import Guard, Kind

    plain = hawk.kernel(_plain_body)
    active = hawk.kernel(
        _same_body_again,
        kind=Kind("act", guard=Guard(mask="terminated", active_set=True)),
    )
    assert plain.body_digest == active.body_digest
    assert plain.arg_spec == active.arg_spec
    described = describe([plain, active], targets=("host",))
    assert {"active_count", "active_map"} <= set(
        described["_same_body_again"]["slot_shapes"]
    )
    assert not {"active_count", "active_map"} & set(
        described["_plain_body"]["slot_shapes"]
    )
    assert described["_plain_body"]["digest"] != described["_same_body_again"]["digest"]


def test_an_unknown_opt_level_is_refused_by_name_from_the_call_and_from_the_variable(
    monkeypatch,
):
    """Restates the old resolution test: the call's level and ``$HAWK_OPT_LEVEL`` both name an unknown level, the variable by
    its name, and the text lists the levels the library knows."""
    from hawk.targets import machine

    with pytest.raises(
        HawkError, match=r"unknown opt level .*the build's opt_level.*O3"
    ):
        machine("host", opt_level="Ofast")
    monkeypatch.setenv("HAWK_OPT_LEVEL", "Ofast")
    with pytest.raises(HawkError, match=r"unknown opt level .*\$HAWK_OPT_LEVEL.*O3"):
        machine("host")
    with pytest.raises(HawkError, match=r"unknown opt level .*\$HAWK_OPT_LEVEL"):
        describe([square], targets=("host",))
    monkeypatch.setenv("HAWK_OPT_LEVEL", "O0")
    assert machine("host")["asked"]["opt_level"] == "O0"
    assert machine("host", opt_level="O2")["asked"]["opt_level"] == "O2"
