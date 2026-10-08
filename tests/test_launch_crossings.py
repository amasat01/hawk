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
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import _deployable as D
import _oracle as O
import numpy as np
import pytest
from _cards import card_path
from conftest import sidecar_of

from hawk import _core, runtime

CARD = card_path("CARD_LAUNCH_CROSSINGS.md")

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


def test_the_bulk_rebind_cost_is_independent_of_the_slot_count(built):
    """The claim "2, not N+1" is only meaningful across DIFFERENT N. Two kernels
    with different pointer-slot counts must both cost 2."""
    counts = {}
    for bundle_name, kernel_name, kw in (
            ("axpb", "axpb", {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0}),
            ("spin", "spin", {"earth__orientation__q": D.quat(N),
                              "earth__orientation__w": np.linspace(0.5, 2.0, N),
                              "v": D.plane(N)})):
        bundle = built[bundle_name]
        kernel = O.load(bundle.directory, kernel_name,
                        sidecar_of(bundle, kernel_name))
        planes = _bound(kernel, kw)
        slots = _pointer_slots(kernel)
        keepalive: list = []
        slot_buf = array.array("q", slots)
        ptr_buf = array.array("q", [runtime.buffer_address(planes[name], keepalive)
                                    for i, (role, name) in enumerate(kernel.arg_spec)
                                    if i in slots])

        def step(k=kernel, s=slot_buf, p=ptr_buf):
            k.block.rebind(s, p)
            k.launch(0, N, N)

        counts[len(slots)] = _crossings_per_launch(step, 500)
    assert len(counts) >= 2, f"both kernels bind the same number of slots: {counts}"
    assert set(counts.values()) == {2.0}, (
        f"the bulk rebind's cost tracks the slot count: {counts} — it must be 2 "
        "for every slot count"
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


def test_the_launch_crossing_card_is_written(built):
    """Written BY THE RUN, md5-fenced, naming the env axis two readings could
    differ on. The two arms are INTERLEAVED (a, b, a, b, ...) rather than run
    one after the other, because this box is shared and a load change that
    straddled the arms would be read as a difference between them."""
    rows = {}
    for bundle_name, kernel_name, kw in (
            ("axpb", "axpb", {"x": D.plane(N, w=1)[0], "a": 2.0, "b": -1.0}),
            ("vec3", "vec3_scale", {"x": D.plane(N), "a": 2.0}),
            ("spin", "spin", {"earth__orientation__q": D.quat(N),
                              "earth__orientation__w": np.linspace(0.5, 2.0, N),
                              "v": D.plane(N)})):
        bundle = built[bundle_name]
        kernel = O.load(bundle.directory, kernel_name,
                        sidecar_of(bundle, kernel_name))
        planes = _bound(kernel, kw)
        slots = _pointer_slots(kernel)
        keepalive: list = []
        slot_buf = array.array("q", slots)
        ptr_buf = array.array("q", [runtime.buffer_address(planes[name], keepalive)
                                    for i, (role, name) in enumerate(kernel.arg_spec)
                                    if i in slots])

        def stable(k=kernel):
            k.launch(0, N, N)

        def rebound(k=kernel, s=slot_buf, p=ptr_buf):
            k.block.rebind(s, p)
            k.launch(0, N, N)

        stable_s, rebound_s = [], []
        for _ in range(REPEATS):                       # INTERLEAVED, not blocked
            stable_s.append(_time_arm(stable, LAUNCHES))
            rebound_s.append(_time_arm(rebound, LAUNCHES))
        rows[f"{kernel_name}"] = {
            "pointer_slots": len(slots),
            "arg_slots": len(kernel.descriptor),
            "samples": N,
            "launches_per_repeat": LAUNCHES,
            "stable_binding": _stat(stable_s),
            "rebind_every_slot": _stat(rebound_s),
            "crossings_per_launch": {"stable_binding": 1, "rebind_every_slot": 2},
        }

    info = _core.build_info()
    payload = {
        "card": "HAWK launch-boundary crossings and per-launch cost",
        "produced": time.strftime("%Y-%m-%d"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>") or "<empty>",
        # BOTH prefixes: the interpreter that ran the arms is the one whose
        # headers and binding they used, and the ambient $CONDA_PREFIX on a dev
        # box is easily a different env's (which is exactly the axis a card has
        # to be able to name).
        "interpreter": "python " + platform.python_version(),
        "ambient_conda_prefix": os.environ.get("CONDA_PREFIX", "<unset>"),
        "core_path": Path(_core.__file__).name,
        "core_md5": hashlib.md5(Path(_core.__file__).read_bytes()).hexdigest(),
        "core_build_digest": _core.build_digest(),
        "core_compiler": info["compiler_id"],
        "aether_include": Path(info["aether_include"]).name,
        "eagle_include": Path(info["eagle_include"]).name,
        "note": ("the CROSSING COUNTS are the gate and are exact integers; the "
                 "seconds are a MEASUREMENT on a shared box, interleaved arm by "
                 "arm, reported with their spread and gating nothing"),
        "rows": rows,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    fence = hashlib.md5(body.encode()).hexdigest()
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(
        "# CARD_LAUNCH_CROSSINGS — the per-launch boundary cost\n\n"
        "Written BY THE RUN (`tests/test_launch_crossings.py`). Prose cites this "
        "file; no number here may be restated without citing it.\n\n"
        "`crossings_per_launch` is the GATE: 1 for a stable-binding loop, 2 for a "
        "rebind-every-slot loop, whatever the slot count. The seconds "
        "beside them are a measurement on a shared machine — arms interleaved, "
        "spread reported — and gate nothing.\n\n"
        f"md5: `{fence}`\n\n```json\n{body}\n```\n"
    )
    text = CARD.read_text()
    assert f"md5: `{fence}`" in text
    assert hashlib.md5(text.split("```json\n")[1].split("\n```")[0]
                       .encode()).hexdigest() == fence


def _stat(samples) -> dict:
    return {
        "seconds_per_launch_median": round(statistics.median(samples), 9),
        "seconds_per_launch_min": round(min(samples), 9),
        "seconds_per_launch_max": round(max(samples), 9),
        "repeats": len(samples),
    }


def test_the_card_names_the_axis_two_readings_could_differ_on(built):
    """A card that could not say WHICH binding and WHICH headers produced it is
    a number without a subject (the rule, applied to this card)."""
    assert CARD.is_file(), "run test_the_launch_crossing_card_is_written first"
    payload = json.loads(CARD.read_text().split("```json\n")[1].split("\n```")[0])
    for field in ("machine", "interpreter", "aether_include", "eagle_include",
                  "core_md5", "core_build_digest", "cuda_visible_devices"):
        assert payload.get(field), field
    assert payload["core_build_digest"] == _core.build_digest(), (
        "the card was written by a different binding than the one loaded now"
    )
    for name, row in payload["rows"].items():
        assert row["crossings_per_launch"] == {"stable_binding": 1,
                                               "rebind_every_slot": 2}, name
        assert row["stable_binding"]["seconds_per_launch_median"] > 0, name
