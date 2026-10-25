# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The group-map prologue: a kernel declared ``group=G`` under ``Guard(active_set=True)``.

At group 1 the active-set map holds sample indices and the prologue is
unchanged. Above 1 the map holds group indices and the count holds the
launched lanes (live groups times ``G``): launch position ``t`` reads sample
``map[t / G] * G + t % G``, and a last group running past the batch exits
before the body. What is pinned:

* **Text.** Group 1 renders the per-lane map read and no batch-end exit; group
  32 renders the derived read and the exit on both backends, with the host loop
  in its plain (non-SIMD) form. The body is the same at every group.
* **Key and sidecar.** The group is part of the unit key (only above 1), and
  the sidecar records it in ``launch.lane_space.group``.
* **Run.** On a batch that is not a multiple of 32, a group-32 run over a
  group map equals the group-1 run over the matching per-lane map, bit for
  bit (host; the GPU arm runs the device entry).

RED: ignore the group in the prologue and the text rows fail; drop the batch-end
exit and the run reads and writes past the planes.
"""

from __future__ import annotations

import importlib

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.artifact import build_bundle
from hawk.artifact.unit_cache import _unit_key
from hawk.emit import scalar_mode
from hawk.ext import Guard, Kind

B = importlib.import_module("hawk.artifact.bundle")

MAP = Kind("group_rows", guard=Guard(active_set=True))
G = 32
N = 100                      # not a multiple of G: the last group has 4 lanes
DT = 0.25
PER_LANE_READ = "reinterpret_cast<const std::int32_t*>(lut_active_map.data())[hawk_t]"
GROUP_READ = (f"static_cast<long long>(reinterpret_cast<const std::int32_t*>"
              f"(lut_active_map.data())[(hawk_t) / {G}]) * {G} + (hawk_t) % {G}")


def _step(omega: Scalar, dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt * omega


def _kernel(group):
    return hawk.kernel(_step, kind=MAP, group=group)


def _sources(group):
    k = _kernel(group)
    unit = B._active_unit(k, "step", MAP)
    return B._sources(unit, name="step", smode=scalar_mode("float64"), kind=MAP,
                      targets=("cuda", "host"), layout_sizes_override=None)


@pytest.mark.parametrize("target", ["cuda", "host"])
def test_group_one_text_is_the_per_lane_map(target):
    text = _sources(1)[target].text
    assert PER_LANE_READ in text
    assert GROUP_READ not in text and "hawk_i >= nSamples" not in text


@pytest.mark.parametrize("target,exit_line", [("cuda", "if (hawk_i >= nSamples) return;"),
                                              ("host", "if (hawk_i >= nSamples) continue;")])
def test_group_text_carries_the_derived_read_and_the_exit(target, exit_line):
    one, many = _sources(1)[target], _sources(G)[target]
    assert GROUP_READ in many.text and PER_LANE_READ not in many.text
    assert many.text.index(GROUP_READ) < many.text.index(exit_line)
    assert exit_line in many.text
    assert many.body == one.body


def test_the_host_loop_at_a_group_is_not_the_identity_form():
    assert "hawk_identity" not in _sources(G)["host"].text


def test_the_group_is_part_of_the_unit_key_only_above_one():
    def key(group):
        k = _kernel(group)
        return _unit_key(["step"], [k.walk.digest], [""], targets=("host",),
                         smode=scalar_mode("float64"), kind=MAP, defines=(), cache_dir="",
                         device_arch="", derivative_blocks=[{}], layout_sizes_override=None,
                         groups=(group,))

    assert key(32) != key(64) != key(1)
    assert key(1) == _unit_key(["step"], [_kernel(1).walk.digest], [""], targets=("host",),
                               smode=scalar_mode("float64"), kind=MAP, defines=(),
                               cache_dir="", device_arch="", derivative_blocks=[{}],
                               layout_sizes_override=None)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("group_prologue")
    return {g: build_bundle([_kernel(g)], root / f"g{g}", targets=("cuda", "host"),
                            cache_dir=cache_dir, host_profile="native", kind=MAP)
            for g in (1, G)}


def test_the_sidecar_records_the_group(bundles):
    for g, bundle in bundles.items():
        assert sidecar_of(bundle, "_step")["launch"]["lane_space"]["group"] == g


def _run(bundle, target, live_groups):
    """One launch with every lane outside ``live_groups`` (``G``-wide splits of
    ``N`` samples) terminated, through eagle's group-aware ``ActiveSet``."""
    import eagle.exec as eexec
    from eagle import ActiveSet
    from eagle import plan as eplan

    xp = np
    if target == "cuda":
        import cupy as xp
    sidecar = sidecar_of(bundle, "_step")
    group = sidecar["launch"]["lane_space"]["group"]
    dead = np.ones(N, dtype=bool)
    for g in live_groups:
        dead[g * G:(g + 1) * G] = False
    terminated = xp.asarray(dead)
    live = ActiveSet(terminated, group=group)
    live.compact()
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, "_step", sidecar), eexec.HostTeam
    else:
        plugin, structure = (L.device_plugin(bundle.directory, "_step", sidecar),
                             eexec.DeviceKernel)
    x = xp.asarray(np.linspace(0.0, 3.0, N))
    bound = eplan.plan(plugin, structure=structure).bind(
        omega=xp.asarray(np.linspace(-1.0, 1.0, N)), dt=DT, terminated=terminated, x=x,
        active_map=live.map, active_count=live.count)
    bound.launch()
    return np.asarray(x.get() if hasattr(x, "get") else x).copy()


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("live", [[0, 1, 2, 3], [0, 2, 3], [3], [1]])
def test_a_group_run_equals_the_per_lane_run(bundles, target, live):
    want = _run(bundles[1], target, live)
    got = _run(bundles[G], target, live)
    np.testing.assert_array_equal(got, want)
    lanes = [i for g in live for i in range(g * G, min((g + 1) * G, N))]
    start = np.linspace(0.0, 3.0, N)
    assert np.all(got[lanes] != start[lanes]) or lanes == []     # the live lanes stepped
    rest = np.setdiff1d(np.arange(N), lanes)
    np.testing.assert_array_equal(got[rest], start[rest])        # the others did not

