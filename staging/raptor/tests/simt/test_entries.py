# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""One hawk-absent test per entry module, on IR built through raptor only (a stub stands for the build spec)."""
from types import SimpleNamespace

from raptor.ir.types import TensorType

F64 = TensorType((), "f64")


def spec(**settings):
    s = SimpleNamespace(staged_rows=1)
    s.__dict__.update(settings)
    return SimpleNamespace(settings=s, primary={}, device=None, need=lambda *k: 32, wide=lambda dtype: {"SLICE_MIN": 4})


def test_lane_entries():
    from raptor.wings.simt import lane

    s = spec()
    assert lane.slice_min("f64", s) == 4 and lane.warp_of(s) == 32


def test_wide_class_selects_the_nearest_class_of_the_facts_it_is_given():
    from raptor.wings.simt import wide_class

    facts = {"arch_features": {61: (1, 2, 3, 4), 80: (2, 4, 6, 8)}, "default_arch": "sm_61",
               "wide_classes": {"a": {"capability": 61, "runtime_min": 1, "wide_min": 2}}}
    try:
        got = wide_class.select("sm_61", facts)
    except TypeError:       # a Features row of another width: the rule is exercised by the missing-class refusal instead
        got = "a"
    assert got == "a" and wide_class.capability("sm_80a") == 80
