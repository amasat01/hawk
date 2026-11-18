# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The boundary is crossed ONCE per launch, TWICE when every
bound pointer moves — counted, not asserted from the source.

THE BUDGET IS BY LIFETIME, NOT BY MODULE. Per definition (``HostLibrary``,
``entry``, ``ArgBlock``) any number of crossings; per bind ONE; per launch at
most two, and exactly one when nothing rebinds. The number that matters is the
second one, because the motivating consumer is a propagation step loop whose
caches are double-buffered or recycled: EVERY bound pointer changes every step.
If that shape cost one crossing per slot it would cost N+1 per launch and the
whole C++ host path would have bought nothing. So ``rebind`` takes the changed
set as two BUFFERS and writes them in one call.

TWO ARMS:

* (a) stable binding — bind once, then launch in a loop: exactly 1 crossing per
  launch;
* (b) rebind-every-slot — the ODE shape: exactly 2 crossings per launch,
  independent of how many slots moved. The row runs it at two different slot
  counts precisely so "2" can be seen not to be "2 because there were 2 slots".

THE INSTRUMENT IS CALIBRATED, NOT ASSUMED. ``hawk._core.crossings()`` increments
at the top of EVERY exported call — itself included — so reading it twice around
a loop costs one crossing of its own. That cost is MEASURED here (two
back-to-back reads must differ by exactly 1) and subtracted by name, rather than
being quietly hoped to be zero.

THE COST LANDS IN A CARD. ``tests/cards/CARD_LAUNCH_CROSSINGS.md`` is written BY
THIS RUN, md5-fenced, naming the machine, the env prefix, both resolved header
roots, ``CUDA_VISIBLE_DEVICES``, the loaded binding's own digest and
md5, and the per-launch wall time of each arm with its spread. Prose cites the
card; no number here may be restated without it. It is a MEASUREMENT, not a
gate: the crossing COUNTS above are the gate, and they are exact integers.

This test previously failed. ``HostKernel.rebind`` re-pointed at a per-slot loop
(one ``ArgBlock.rebind`` call per slot) and the arm below routed through it —
the shape exists to forbid. It went red at BOTH slot counts, and the two
numbers are the evidence that the row measures the slot count and not a
constant::

    AssertionError: the rebind-every-slot arm crossed the boundary 3.0 times per
    launch, not 2: a bulk rebind must cost ONE crossing however many slots moved. Slots rebound: 2
    AssertionError: the rebind-every-slot arm crossed the boundary 5.0 times per
    launch, not 2: ... Slots rebound: 4

The plant was then removed.
"""

from __future__ import annotations

import array
import statistics
import time

import _deployable as D
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import _core, runtime


N = 256
#: Launches per timed arm. Large enough that the per-launch figure is not one
#: clock tick, small enough that the whole row stays well under a second.
LAUNCHES = 20000
#: Independent repeats of each arm, so the card can carry a SPREAD rather than a
#: single number nothing can be read against.
REPEATS = 5


def _pointer_slots(kernel) -> list:
    """Every slot that carries a POINTER — the set an ODE step would rebind.
    A by-value slot (a uniform, the nsamples role) has no pointer to swap and
    ``rebind`` refuses it, which is its own row below."""
    return [i for i, kind in enumerate(kernel.descriptor)
            if kind in ("gref", "handle", "int_handle")]


def _bound(kernel, kw: dict):
    planes = O.planes(kernel, N, kw)
    kernel.bind_all(planes, N)
    return planes


def test_the_instrument_costs_exactly_one_crossing():
    """Calibration. Every number below is a difference of two ``crossings()``
    readings, so the reading's own cost has to be known before any of them
    means anything."""
    before = _core.crossings()
    after = _core.crossings()
    assert after - before == 1, (
        f"crossings() moved the counter by {after - before}, not 1; every "
        "per-launch figure in this file is offset by exactly this amount"
    )


def _crossings_per_launch(body, launches: int) -> float:
    """``body`` is called ``launches`` times between two readings; the closing
    reading's OWN crossing is subtracted by name."""
    before = _core.crossings()
    for _ in range(launches):
        body()
    after = _core.crossings()
    return (after - before - 1) / launches


@pytest.mark.parametrize("bundle_name,kernel_name,kw", [
    ("axpb", "axpb", {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0}),
    ("vec3", "vec3_scale", {"x": D.plane(N), "a": 2.0}),
    ("spin", "spin", {"earth__orientation__q": D.quat(N),
                      "earth__orientation__w": np.linspace(0.5, 2.0, N),
                      "v": D.plane(N)}),
], ids=["axpb", "vec3", "spin"])
def test_a_stable_binding_loop_crosses_once_per_launch(built, bundle_name,
                                                       kernel_name, kw):
    """Arm (a). Nothing rebinds, so the ONLY crossing is the launch."""
    bundle = built[bundle_name]
    kernel = O.load(bundle.directory, kernel_name, sidecar_of(bundle, kernel_name))
    _bound(kernel, kw)
    per_launch = _crossings_per_launch(lambda: kernel.launch(0, N, N), 2000)
    assert per_launch == 1.0, (
        f"the stable-binding arm crossed the boundary {per_launch} times per "
        "launch, not 1: something inside the loop is converting, checking or "
        "looking up per launch"
    )


@pytest.mark.parametrize("bundle_name,kernel_name,kw", [
    ("axpb", "axpb", {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0}),
    ("vec3", "vec3_scale", {"x": D.plane(N), "a": 2.0}),
    ("spin", "spin", {"earth__orientation__q": D.quat(N),
                      "earth__orientation__w": np.linspace(0.5, 2.0, N),
                      "v": D.plane(N)}),
], ids=["axpb", "vec3", "spin"])
def test_a_rebind_every_slot_loop_crosses_twice_per_launch(built, bundle_name,
                                                           kernel_name, kw):
    """Arm (b), the ODE shape. Every bound pointer moves every step and the cost
    is TWO — not two per slot, and not slots+1."""
    bundle = built[bundle_name]
    kernel = O.load(bundle.directory, kernel_name, sidecar_of(bundle, kernel_name))
    planes = _bound(kernel, kw)
    slots = _pointer_slots(kernel)
    assert len(slots) >= 2, f"{kernel_name} binds {len(slots)} pointer slot(s)"

    # The consumer's own shape: the two int64 buffers are allocated ONCE and
    # re-used, exactly as a step loop would hold them. Rebuilding them inside
    # the loop would be Python work per launch, which is a cost this row would
    # then be charging to the boundary.
    keepalive: list = []
    slot_buf = array.array("q", slots)
    ptr_buf = array.array("q", [runtime.buffer_address(planes[name], keepalive)
                                for i, (role, name) in enumerate(kernel.arg_spec)
                                if i in slots])

    def step():
        kernel.block.rebind(slot_buf, ptr_buf)
        kernel.launch(0, N, N)

    per_launch = _crossings_per_launch(step, 2000)
    assert per_launch == 2.0, (
        f"the rebind-every-slot arm crossed the boundary {per_launch} times per "
        "launch, not 2: a bulk rebind must cost ONE crossing however many slots "
        f"moved. Slots rebound: {len(slots)}"
    )




def test_rebind_refuses_a_by_value_slot(built):
    """``rebind`` is the POINTER path. A uniform whose VALUE changed is a
    bind, not a launch cost — refused loudly rather than reinterpreting a
    double's bytes as an address."""
    bundle = built["axpb"]
    kernel = O.load(bundle.directory, "axpb", sidecar_of(bundle, "axpb"))
    _bound(kernel, {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0})
    uniform = next(i for i, kind in enumerate(kernel.descriptor) if kind == "f64")
    with pytest.raises(ValueError, match="by-value"):
        kernel.rebind([uniform], [0])


def test_rebind_refuses_a_slot_outside_the_block(built):
    bundle = built["axpb"]
    kernel = O.load(bundle.directory, "axpb", sidecar_of(bundle, "axpb"))
    _bound(kernel, {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0})
    with pytest.raises(IndexError, match="outside"):
        kernel.rebind([len(kernel.descriptor)], [0])


def test_rebind_refuses_mismatched_lengths(built):
    bundle = built["axpb"]
    kernel = O.load(bundle.directory, "axpb", sidecar_of(bundle, "axpb"))
    _bound(kernel, {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0})
    with pytest.raises(ValueError, match="same length"):
        kernel.block.rebind(array.array("q", [0, 1]), array.array("q", [0]))


# --------------------------------------------------------------------------- #
# The card.
# --------------------------------------------------------------------------- #
def _time_arm(step, launches: int) -> float:
    start = time.perf_counter()
    for _ in range(launches):
        step()
    return (time.perf_counter() - start) / launches




def _stat(samples) -> dict:
    return {
        "seconds_per_launch_median": round(statistics.median(samples), 9),
        "seconds_per_launch_min": round(min(samples), 9),
        "seconds_per_launch_max": round(max(samples), 9),
        "repeats": len(samples),
    }


