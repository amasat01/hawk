# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Three mechanisms that must not depend on history or on another target: the unit key (import order), least squares inside a loop body, and each target's own least-squares method."""
import pathlib

import numpy as np
import pytest

import hawk.math as m
from hawk import Matrix, Mutable, Vector, _ir, kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle

HERE = pathlib.Path(__file__).resolve().parent
LANES = 8





def _run(bundle, name, **planes):
    art = next(a for a in bundle.artifacts if a.name == name)
    rt.run(rt.load(bundle.directory, name, art.sidecar), **planes)


@kernel
def gn_loop(J: Matrix[4, 3], r: Vector[4], x: Mutable[Vector[3]]):
    acc = 0.0 * m.lstsq(J, r)[0]
    for _k in range(3):
        dx, _ = m.lstsq(J, r - J @ acc)
        acc = acc + 0.5 * dx
    x = acc




def test_lstsq_above_max_dim_in_a_loop_body_is_lowered_to_qr_and_the_per_lane_writer_refuses_it_by_name(tmp_path, cache_dir):
    """Above ``linalg.max_dim`` the lowering gives Householder QR inside the body (it is built); one lane cannot hold it, and a team
    stage has no place inside a loop body: that refusal is the writer's, by name, not a missing lowering."""
    from hawk.errors import HawkError

    @kernel
    def wide(J: Matrix[36, 34], r: Vector[36], x: Mutable[Vector[34]]):
        acc = 0.0 * m.lstsq(J, r)[0]
        for _k in range(2):
            dx, _ = m.lstsq(J, r - J @ acc)
            acc = acc + 0.5 * dx
        x = acc

    with pytest.raises(HawkError, match="op qr of rank > 0"):
        build_bundle([wide], tmp_path / "wide", targets=("host",), cache_dir=cache_dir)


@pytest.mark.parametrize("k,rows,cols,trips", [(gn_loop, 4, 3, 3)])
def test_lstsq_in_a_loop_body_runs_against_numpy(tmp_path, cache_dir, k, rows, cols, trips):

    assert cols <= 32          # the normal equations: one lane factors it
    assert any(n.field_text("kind") == "lstsq" for n in _ir.walk_order(k.walk))                      # one lstsq node in the traced IR
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
