# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Four mechanisms that must not depend on history or on another target: the unit key (import order), a raw block's text in the IR
key, least squares inside a loop body, and each target's own least-squares method."""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Scalar, Vector, kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle

HERE = pathlib.Path(__file__).resolve().parent
LANES = 8

_TWICE = """
import sys
import hawk
from hawk import Mutable, Scalar, Vector
from hawk.artifact import bundle as B, unit_stats
from hawk.math import dot

keys = []
_uk = B._unit_key
def uk(*a, **k):
    keys.append(_uk(*a, **k)); return keys[-1]
B._unit_key = uk

@hawk.kernel
def twice(u: Vector[12], v: Vector[12], y: Mutable[Scalar]):
    y = dot(u, v)

out = sys.argv[1]
B.build_bundle([twice], out + "/a", targets=("host",))
before = unit_stats()["memo_hits"]
B.build_bundle([twice], out + "/a", targets=("host",))
print(keys[0] == keys[1], unit_stats()["memo_hits"] - before)
"""


def test_one_request_has_one_unit_key_and_the_repeat_build_hits_the_memo(tmp_path):
    env = {**os.environ, "HAWK_CACHE_DIR": str(tmp_path / "cache"),
           "PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE), str(HERE.parent / "extensions" / "hawk_c"), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", _TWICE, str(tmp_path / "out")], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split()[-2:] == ["True", "1"]


def _run(bundle, name, **planes):
    art = next(a for a in bundle.artifacts if a.name == name)
    rt.run(rt.load(bundle.directory, name, art.sidecar), **planes)


def test_two_raw_blocks_of_one_name_and_different_text_in_one_kernel_are_two_blocks(tmp_path, cache_dir):
    a = hawk.raw_device("psc_x[i].eval() * 2.0", access="sample_local", returns=Scalar)
    b = hawk.raw_device("psc_x[i].eval() * 3.0", access="sample_local", returns=Scalar)

    @kernel
    def two(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar], w: Mutable[Scalar]):
        w = x                  # reads x, so the plane the blocks name exists
        y = a()
        z = b()

    bundle = build_bundle([two], tmp_path / "two", targets=("host",), cache_dir=cache_dir)
    x, y, z, w = np.arange(1.0, 5.0), np.zeros(4), np.zeros(4), np.zeros(4)
    _run(bundle, "two", x=x, y=y, z=z, w=w)
    np.testing.assert_allclose(y, 2.0 * x)
    np.testing.assert_allclose(z, 3.0 * x)


def test_two_raw_blocks_that_take_operands_in_one_kernel_compile_and_run(tmp_path, cache_dir):
    """Each operand-taking block reads its operand as ``hawk_raw_arg0``; two of them in one kernel must not redefine it."""
    a = hawk.raw_device("hawk_raw_arg0 * 2.0", access="sample_local", returns=Scalar)
    b = hawk.raw_device("hawk_raw_arg0 * 3.0", access="sample_local", returns=Scalar)

    @kernel
    def two(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
        y = a(x)
        z = b(x)

    bundle = build_bundle([two], tmp_path / "two_ops", targets=("host",), cache_dir=cache_dir)
    x, y, z = np.arange(1.0, 5.0), np.zeros(4), np.zeros(4)
    _run(bundle, "two", x=x, y=y, z=z)
    np.testing.assert_allclose(y, 2.0 * x)
    np.testing.assert_allclose(z, 3.0 * x)


def test_one_raw_block_with_an_operand_keeps_its_emitted_text(tmp_path, cache_dir):
    blk = hawk.raw_device("hawk_raw_arg0 * 2.0", access="sample_local", returns=Scalar)

    @kernel
    def one(x: Scalar, y: Mutable[Scalar]):
        y = blk(x)

    bundle = build_bundle([one], tmp_path / "one_op", targets=("host",), cache_dir=cache_dir)
    text = (bundle.directory / "one.cpp").read_text()
    assert "const Real hawk_raw_arg0 = " in text or "hawk_raw_arg0 =" in text
    assert "[&]" not in text                      # the first block is declared in the kernel's scope, as before


def test_raw_blocks_of_one_name_in_other_kernels_are_not_served_each_others_unit(tmp_path, cache_dir):
    x = np.arange(1.0, 5.0)
    for factor in (2, 3, 4, 5):
        blk = hawk.raw_device(f"psc_x[i].eval() * {factor}.0", access="sample_local", returns=Scalar)

        @kernel
        def rk(x: Scalar, y: Mutable[Scalar]):
            y = blk(x)

        bundle = build_bundle([rk], tmp_path / f"k{factor}", targets=("host",), cache_dir=cache_dir)
        y = np.zeros(4)
        _run(bundle, "rk", x=x, y=y)
        np.testing.assert_allclose(y, factor * x)


@kernel
def gn_loop(J: Matrix[4, 3], r: Vector[4], x: Mutable[Vector[3]]):
    acc = 0.0 * m.lstsq(J, r)[0]
    for _k in range(3):
        dx, _ = m.lstsq(J, r - J @ acc)
        acc = acc + 0.5 * dx
    x = acc


@pytest.mark.parametrize("k,rows,cols,trips", [(gn_loop, 4, 3, 3)])
def test_lstsq_in_a_loop_body_runs_against_numpy(tmp_path, cache_dir, k, rows, cols, trips):
    from hawk.passes.lstsq import max_dim_of
    from hawk.targets import build_spec

    assert cols <= max_dim_of(build_spec(("host",)).primary)          # the normal equations: one lane factors it
    assert k.walk.kinds & {"lstsq"}                      # one lstsq node in the traced IR
    rng = np.random.default_rng(5)
    J = rng.standard_normal((LANES, rows, cols)) + 3 * np.eye(rows, cols)
    r = rng.standard_normal((LANES, rows))
    pl = lambda a: np.ascontiguousarray(a.reshape(LANES, -1).T)
    x = np.zeros((cols, LANES))
    bundle = build_bundle([k], tmp_path / k.name, targets=("host",), cache_dir=cache_dir)
    _run(bundle, k.name, J=pl(J), r=pl(r), x=x)
    for lane in range(LANES):
        acc = np.zeros(cols)
        for _ in range(trips):
            acc = acc + 0.5 * np.linalg.lstsq(J[lane], r[lane] - J[lane] @ acc, rcond=None)[0]
        np.testing.assert_allclose(x[:, lane], acc, rtol=1e-9, atol=1e-11)


def test_lstsq_above_max_dim_in_a_loop_body_is_lowered_to_qr_and_the_per_lane_writer_refuses_it_by_name(tmp_path, cache_dir):
    """Above ``linalg.max_dim`` the lowering gives Householder QR inside the body (it is built); one lane cannot hold it, and a team
    stage has no place inside a loop body: that refusal is the writer's, by name, not a missing lowering."""
    from hawk.ir import HawkError

    @kernel
    def wide(J: Matrix[36, 34], r: Vector[36], x: Mutable[Vector[34]]):
        acc = 0.0 * m.lstsq(J, r)[0]
        for _k in range(2):
            dx, _ = m.lstsq(J, r - J @ acc)
            acc = acc + 0.5 * dx
        x = acc

    with pytest.raises(HawkError, match="op 'qr' at n=36 reached the per-lane writer"):
        build_bundle([wide], tmp_path / "wide", targets=("host",), cache_dir=cache_dir)


def test_the_host_text_of_a_device_and_host_build_does_not_depend_on_the_device_max_dim():
    from hawk.artifact import bundle as B
    from hawk.targets import TARGETS, build_spec
    from hawk.variants import derive, load

    @kernel
    def ls(J: Matrix[4, 3], r: Vector[4], x: Mutable[Vector[3]]):
        x, _ = m.lstsq(J, r)

    smode = B.scalar_mode("float64")
    kd = B._member_kind(ls, B._resolve_bundle_kind(["ls"], [ls], None), False)
    options = B._options(device_arch="sm_61")

    def host_text(device):
        spec = build_spec(("cuda", "host"), "sm_61", **device).with_settings(scalar=smode.id)
        low = B.lower(ls, "ls", smode, kd, ("cuda", "host"), spec)
        return TARGETS["host"].emit(low, options).text, spec

    shipped, spec = host_text({})
    planted, spec2 = host_text({"device": derive(load("cuda"), {"linalg": {"max_dim": 2}})})
    assert spec.device["linalg"]["max_dim"] == 32 and spec2.device["linalg"]["max_dim"] == 2
    assert shipped == planted
    host_only = TARGETS["host"].emit(B.lower(ls, "ls", smode, kd, ("host",), build_spec(("host",)).with_settings(scalar=smode.id)), options).text
    assert host_only == shipped


def test_a_variant_without_the_number_is_refused_by_name():
    from hawk.ir import HawkError
    from hawk.passes.lstsq import max_dim_of

    with pytest.raises(HawkError, match="linalg.max_dim"):
        max_dim_of({"form": {}})
