# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wide-value thresholds card (``hawk/targets/cards/wide.json``) and the sweep that writes its rows: a row changes
the decision on its own device and nowhere else, no row is today's behaviour, a malformed row is refused by name, and the
sweep's row assembly (winners, ties, thresholds) is plain logic."""

from __future__ import annotations
from _specs import DEVICE, HOST, device

import json

import pytest
import _wide_kernels as W

from hawk import HawkError
from raptor.wings.base import wide
from hawk.targets import cards
from raptor.ir.types import TensorType

NARROW = TensorType((64,), "f32")


def _wide_spec(arch, rows=None, defaults=None):
    """A host and device build for ``arch`` whose device wing carries a wide card with these rows and defaults (facts passed in)."""
    from hawk.targets import build_spec
    from hawk.facts import derive, load

    cuda = load("cuda")
    card = {"defaults": {**cuda["cards"]["wide"]["defaults"], **(defaults or {})}, "rows": rows or []}
    return build_spec(("host", "cuda"), arch, device=derive(cuda, {"cards": {"wide": card}}))


ROW = {"cc": "sm_00", "device": "Test Card", "dtype": "float32", "SLICE_MIN": 64, "TEAM_MIN": 64, "TC": 8192}


def test_an_unmeasured_arch_takes_the_row_nearest_by_device_features(tmp_path, monkeypatch):
    far = {**ROW, "cc": "sm_70", "TC": 4096}              # FP64 rate 2: far from the sm_61 / sm_89 (1:32, 1:64) rows
    rows = [far, {**ROW, "cc": "sm_61", "device": "P2000", "TC": 16384}]
    monkeypatch.setattr(cards, "_device", lambda: None)  # no device probed: the rows are the build arch's to choose
    for arch, tc in (("sm_61", 16384), ("sm_75", 16384), ("sm_89", 16384), ("sm_80", 4096), ("sm_90", 4096)):
        spec = _wide_spec(arch, rows)
        assert cards.wide("float32", spec)["TC"] == tc, arch


@pytest.mark.parametrize("bad", [{"SLICE_MIN": "64"}, {"TEAM_MIN": 0}, {"TC": -1}, {"SLICE_MIN": True}, {"cc": 61}])
def test_a_malformed_row_is_refused_by_name(bad, tmp_path, monkeypatch):
    with pytest.raises(HawkError, match="wide card"):
        cards.wide("float32", _wide_spec("sm_00", [{**ROW, **bad}]))


def test_a_default_that_is_not_a_whole_number_of_warps_is_refused(tmp_path, monkeypatch):
    with pytest.raises(HawkError, match="multiple of 32"):
        cards.wide("float32", _wide_spec("sm_00", defaults={"SLICE_MIN": 100}))


# -- the sweep's logic, without a device ----------------------------------------------------------------------------

def _cell(form, width, fold, ns, dtype="float32"):
    return {"form": form, "width": width, "fold": fold, "dtype": dtype, "ns": ns}


