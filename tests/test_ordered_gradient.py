# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The reverse of an ordered keyed sum, without and with a loop in the loss.

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


def _reference(y, x, dtype, order, loop=False):
    """The gradient of the loss in ``W`` by the chain rule at ``W = 0``, each key's contributions added one by one in ``order``
    of the samples, every operation in the kernel's scalar type: parameter 0 gets ``2 r x``, parameter 1 gets ``2 r``
    (``r`` is the residual, minus ``y``), the others get 0. With ``loop`` every parameter is read by one trip of a loop and
    gets ``2 r x``."""
    t, out = np.dtype(dtype).type, np.zeros((KEYS, PARAMS), dtype)
    for k in range(KEYS):
        for p in range(PARAMS) if loop else (0, 1):
            total = t(0)
            for i in order:
                r = t(0) - y[k, i]
                total = t(total + (t(2) * r * x[k, i] if p == 0 or loop else t(2) * r))
            out[k, p] = total
    return out


@kernel
def looped(lane: Grid["net", "sample":SAMPLES], W: Table["net", "p":PARAMS], x: Scalar, y: Scalar) -> Scalar:
    pred = 0.0
    for h in range(PARAMS):
        pred = pred + W.at(net=lane.net, p=h) * x
    return hawk.sum((pred - y) * (pred - y), over="sample", contract="ordered")


def _build(dtype, root, source):
    """``(dtype, bundle of the gradient kernel, the gradient kernel)`` of the loss ``source``, built for the host."""
    loss = Kernel(f"{source.name}_{dtype}", source.sinks, source.planes)
    home = build_bundle([loss], root / "loss", targets=("host",), mode=dtype, host_profile="native")
    grad = Kernel(f"{source.name}_g_{dtype}", vjp(loss, wrt=("W",), primal_unit=home.digest))
    return dtype, build_bundle([grad], root / "grad", targets=("host",), mode=dtype, host_profile="native"), grad


@pytest.fixture(scope="module", params=["float32", "float64"])
def built(request, tmp_path_factory):
    return _build(request.param, tmp_path_factory.mktemp(request.param), flat)


@pytest.fixture(scope="module", params=["float32", "float64"])
def built_loop(request, tmp_path_factory):
    return _build(request.param, tmp_path_factory.mktemp("loop" + request.param), looped)


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


def test_the_reverse_inside_a_loop_adds_each_cell_in_sample_order_bit_for_bit(built_loop):
    """The loss reads parameter ``h`` at trip ``h`` of a loop, so the reverse accumulates at every trip. The order of one cell
    ``(key, h)``: the reverse writes one scratch slot per trip for each lane (the value plane has ``sites x trips`` slots
    per lane, see the sidecar test below), and the fold stage adds the slot of a cell over the rows of its key in lane
    order, i.e. in sample order. Trips do not mix: a slot is one parameter. The reference adds exactly that."""
    dtype = built_loop[0]
    y, x, _ = _inputs(dtype)
    runs = [_launch(built_loop) for _ in range(3)]
    want = _reference(y, x, dtype, range(SAMPLES), loop=True)
    assert runs[0].tobytes() == want.tobytes()
    assert all(r.tobytes() == runs[0].tobytes() for r in runs[1:])
    assert _reference(y, x, dtype, reversed(range(SAMPLES)), loop=True).tobytes() != want.tobytes()


def test_the_reverse_inside_a_loop_has_one_slot_per_site_and_trip_and_one_keyed_fold_stage(built_loop):
    """One site (the read of ``W``) at ``PARAMS`` trips: both scratch planes hold ``sites x trips`` slots per lane."""
    _, bundle, grad = built_loop
    side = sidecar_of(bundle, grad.name)
    first, fold = side["stages"]
    assert first["kernel"] == grad.name and not first.get("team") and len(side["stages"]) == 2
    assert fold["team"]["grid"] == "keys" and fold["team"]["keyed"]["axes"] == [["net", None]]
    sites, trips = 1, PARAMS
    assert sorted(s["per_lane"] for s in side["team_planes"].values()) == [sites * trips] * 2
