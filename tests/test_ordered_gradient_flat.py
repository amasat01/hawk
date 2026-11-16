# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The reverse of an ordered keyed sum that is not inside a loop.

``hawk.sum(..., contract="ordered")`` adds the samples of each key in sample order, so its value depends on that order
bit for bit. The reverse (``vjp``) of such a loss is one more ordered keyed sum: the gradient of a parameter is the sum, in
sample order, of one contribution per sample. The tests below build that gradient on the host and hold it to a NumPy
reference that adds in sample order in the kernel's own scalar type, bit for bit.

The addends are chosen so that the order shows in the bits. The parameters are zero, so the residual of a sample is minus
its target ``y``; every target is a small multiple of 0.25 except two per key: sample 0 holds ``+big`` and sample 17 holds
``-big/2`` (``big`` is 2**22 for float32, 2**51 for float64, a power of two near where the type's spacing passes 0.25). A
small addend added to a partial sum that already holds ``big`` is rounded away, one added after the two large ones cancel is
kept: the sum changes with the order, so a sum in reverse order differs from one in sample order, and a build that
adds in another order cannot match by luck."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Grid, Kernel, Scalar, Table, kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import vjp
from raptor.errors import RaptorError

KEYS, SAMPLES, PARAMS = 5, 64, 4


@kernel
def flat(lane: Grid["net", "sample":SAMPLES], W: Table["net", "p":PARAMS], x: Scalar, y: Scalar) -> Scalar:
    pred = W.at(net=lane.net, p=0) * x + W.at(net=lane.net, p=1)
    return hawk.sum((pred - y) * (pred - y), over="sample", contract="ordered")


def _inputs(dtype):
    """``(y, x, W)``: targets as described in the module note, unit inputs, zero parameters."""
    big = 2.0 ** (22 if dtype == "float32" else 51)
    y = (np.random.default_rng(5).integers(1, 8, size=(KEYS, SAMPLES)) * 0.25).astype(dtype)
    y[:, 0], y[:, 17] = big, -big / 2
    return y, np.ones((KEYS, SAMPLES), dtype), np.zeros((KEYS, PARAMS), dtype)


def _reference(y, x, dtype, order):
    """The gradient of the loss in ``W`` by the chain rule at ``W = 0``, each key's contributions added one by one in ``order``
    of the samples, every operation in the kernel's scalar type: parameter 0 gets ``2 r x``, parameter 1 gets ``2 r``
    (``r`` is the residual, minus ``y``), the others get 0."""
    t, out = np.dtype(dtype).type, np.zeros((KEYS, PARAMS), dtype)
    for k in range(KEYS):
        for p in (0, 1):
            total = t(0)
            for i in order:
                r = t(0) - y[k, i]
                total = t(total + (t(2) * r * x[k, i] if p == 0 else t(2) * r))
            out[k, p] = total
    return out


@pytest.fixture(scope="module", params=["float32", "float64"])
def built(request, tmp_path_factory):
    """``(dtype, bundle of the gradient kernel, the gradient kernel)`` built for the host."""
    dtype, root = request.param, tmp_path_factory.mktemp(request.param)
    loss = Kernel(f"flat_{dtype}", flat.sinks, flat.planes)
    home = build_bundle([loss], root / "loss", targets=("host",), mode=dtype, host_profile="native")
    grad = Kernel(f"flat_g_{dtype}", vjp(loss, wrt=("W",), primal_unit=home.digest))
    return dtype, build_bundle([grad], root / "grad", targets=("host",), mode=dtype, host_profile="native"), grad


def _launch(built):
    dtype, bundle, grad = built
    y, x, W = _inputs(dtype)
    planes = {"W": W.ravel().copy(), "x": x.ravel().copy(), "y": y.ravel().copy(),
              "bar_out": np.ones(KEYS, dtype), "bar_W": np.zeros(KEYS * PARAMS, dtype)}
    names = {n for _, n in grad.arg_spec}
    rt.run(rt.load(bundle.directory, grad.name, sidecar_of(bundle, grad.name)), **{k: v for k, v in planes.items() if k in names})
    return planes["bar_W"].reshape(KEYS, PARAMS).copy()


def test_the_gradient_adds_in_sample_order_bit_for_bit_and_every_launch_is_the_same(built):
    dtype = built[0]
    y, x, _ = _inputs(dtype)
    runs = [_launch(built) for _ in range(3)]
    want = _reference(y, x, dtype, range(SAMPLES))
    assert runs[0].tobytes() == want.tobytes()
    assert all(r.tobytes() == runs[0].tobytes() for r in runs[1:])
    # not vacuous: the same contributions added in reverse sample order give other bits
    assert _reference(y, x, dtype, reversed(range(SAMPLES))).tobytes() != want.tobytes()


def test_the_gradient_is_one_stage_of_the_kernel_and_one_keyed_fold_stage_over_two_scratch_planes(built):
    """What the sidecar states about the staging: the kernel's own stage, then one team stage that folds per key, and the two
    scratch planes it works on (a value per lane, a cell per key), with the same number of elements per lane."""
    _, bundle, grad = built
    side = sidecar_of(bundle, grad.name)
    first, fold = side["stages"]
    assert first["kernel"] == grad.name and not first.get("team")
    assert fold["kernel"] != grad.name and fold["team"]["grid"] == "keys" and fold["team"]["keyed"]["axes"] == [["net", None]]
    values = {n: s for n, s in side["team_planes"].items() if "keys" not in s}
    cells = {n: s for n, s in side["team_planes"].items() if "keys" in s}
    assert len(values) == len(cells) == 1 == len(side["team_planes"]) - 1
    assert next(iter(values.values()))["per_lane"] == next(iter(cells.values()))["per_lane"]


def test_the_same_reverse_inside_a_loop_is_refused_by_name(tmp_path):
    """Pins a refusal that a later change removes: the same loss with its prediction written as a ``for`` loop over the
    parameters has a reverse that accumulates inside that loop, which is not written yet. When
    that is written, this test is replaced by one that holds the loop's gradient to a reference."""
    @kernel
    def looped(lane: Grid["net", "sample":SAMPLES], W: Table["net", "p":PARAMS], x: Scalar, y: Scalar) -> Scalar:
        pred = 0.0
        for h in range(2):
            pred = pred + W.at(net=lane.net, p=h) * x
        return hawk.sum((pred - y) * (pred - y), over="sample", contract="ordered")

    loss = Kernel("looped", looped.sinks, looped.planes)
    home = build_bundle([loss], tmp_path / "loss", targets=("host",), host_profile="native")
    grad = Kernel("looped_g", vjp(loss, wrt=("W",), primal_unit=home.digest))
    with pytest.raises(RaptorError, match="an ordered accumulate inside a loop is not written yet"):
        build_bundle([grad], tmp_path / "grad", targets=("host",), host_profile="native")
