# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wide-value thresholds card (``hawk/targets/cards/wide.json``) and the sweep that writes its rows: a row changes
the decision on its own device and nowhere else, no row is today's behaviour, a malformed row is refused by name, and the
sweep's row assembly (winners, ties, thresholds) is plain logic."""

from __future__ import annotations

import json

import pytest
import _wide_kernels as W

from hawk import HawkError
from hawk.artifact.unit_cache import _wide_part
from hawk.ir import wide
from hawk.ir.lane import slice_min, slice_sinks, sliced
from hawk.targets import cards
from hawk.targets.cards import sweep
from hawk.types import TensorType

NARROW = TensorType((64,), "f32")


def _card(tmp_path, monkeypatch, rows=None, defaults=None):
    doc = json.loads(cards.WIDE_FILE.read_text())
    doc["rows"] = rows or []
    doc["defaults"].update(defaults or {})
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(cards, "WIDE_FILE", path)
    monkeypatch.setattr(cards, "_device", lambda: ("sm_00", "Test Card"))


ROW = {"cc": "sm_00", "device": "Test Card", "dtype": "float32", "SLICE_MIN": 64, "TEAM_MIN": 64, "TC": 8192}


def test_the_defaults_are_the_constants_the_warp_form_was_built_with():
    assert cards.wide("float32") == cards.wide("f64") == {"SLICE_MIN": 128, "TEAM_MIN": 128, "TC": 40960}
    assert slice_min("f32") == 128 and not sliced(NARROW)
    assert _wide_part("float32", "sm_61") == ""        # no row: the unit key is the one it always was


def test_a_row_changes_the_decision_for_its_own_device_and_scalar_type_only(tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch, [ROW])
    with cards.build_arch("sm_00"):
        assert cards.wide("float32") == {"SLICE_MIN": 64, "TEAM_MIN": 64, "TC": 8192}
        assert sliced(NARROW) and cards.wide("float64")["SLICE_MIN"] == 128
    assert not sliced(NARROW)                           # a host-only build: no arch, no row
    monkeypatch.setattr(cards, "_device", lambda: ("sm_00", "Another Card"))
    with cards.build_arch("sm_00"):
        assert not sliced(NARROW)                       # the arch matches, the device name does not
    assert _wide_part("float32", "sm_00") == ""         # (and so the unit key keeps its old form)
    monkeypatch.setattr(cards, "_device", lambda: ("sm_00", "Test Card"))
    assert _wide_part("float32", "sm_00") == "wide:64:64:8192"


def test_an_unmeasured_arch_takes_the_row_nearest_by_device_features(tmp_path, monkeypatch):
    far = {**ROW, "cc": "sm_70", "TC": 4096}              # FP64 rate 2: far from the sm_61 / sm_89 (1:32, 1:64) rows
    _card(tmp_path, monkeypatch, [far, {**ROW, "cc": "sm_61", "device": "P2000", "TC": 16384}])
    monkeypatch.setattr(cards, "_device", lambda: None)  # no device probed: the rows are the build arch's to choose
    for arch, tc in (("sm_61", 16384), ("sm_75", 16384), ("sm_89", 16384), ("sm_80", 4096), ("sm_90", 4096)):
        with cards.build_arch(arch):
            assert cards.wide("float32")["TC"] == tc, arch


def test_team_min_decides_auto_and_not_a_kernel_asked_for_the_warp_form(tmp_path, monkeypatch):
    sinks = wide.simplified(W.matvec_in_vjp.sinks)
    assert slice_sinks(sinks, explicit=False) is not None
    _card(tmp_path, monkeypatch, [{**ROW, "dtype": "float64", "SLICE_MIN": 128, "TEAM_MIN": 1024}])
    with cards.build_arch("sm_00"):
        assert slice_sinks(sinks, explicit=False) is None
        assert slice_sinks(sinks) is not None


@pytest.mark.parametrize("bad", [{"SLICE_MIN": "64"}, {"TEAM_MIN": 0}, {"TC": -1}, {"SLICE_MIN": True}, {"cc": 61}])
def test_a_malformed_row_is_refused_by_name(bad, tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch, [{**ROW, **bad}])
    with pytest.raises(HawkError, match=r"wide\.json"):
        cards.wide("float32")


def test_a_default_that_is_not_a_whole_number_of_warps_is_refused(tmp_path, monkeypatch):
    _card(tmp_path, monkeypatch, defaults={"SLICE_MIN": 100})
    with pytest.raises(HawkError, match="multiple of 32"):
        cards.wide("float32")


# -- the sweep's logic, without a device ----------------------------------------------------------------------------

def _cell(form, width, fold, ns, dtype="float32"):
    return {"form": form, "width": width, "fold": fold, "dtype": dtype, "ns": ns}


def test_the_plan_is_the_grid_in_every_form_and_quick_is_smaller():
    full, quick = sweep.plan(("float32", "float64"), False), sweep.plan(("float32",), True)
    assert len(full) == 2 * len(sweep.WIDTHS) * len(sweep.FOLDS) * len(sweep.FORMS) and 0 < len(quick) < len(full)
    assert {c[0] for c in full} == set(sweep.FORMS)


def test_winners_mark_the_forms_within_ten_percent_as_a_tie():
    won = sweep.winners([_cell("thread", 64, 128, 100), _cell("team", 64, 128, 105), _cell("staged", 64, 128, 140)])
    assert won == [{"dtype": "float32", "width": 64, "fold": 128, "form": "thread", "ns": 100, "tied": ["thread", "team"],
                    "tie": True}]
    lone = sweep.winners([_cell("thread", 64, 128, 100), _cell("team", 64, 128, 50)])[0]
    assert lone["form"] == "team" and lone["tied"] == ["team"] and not lone["tie"]


def test_the_row_thresholds_are_where_a_warp_form_first_beats_the_thread_form_clearly():
    cells = []
    for width, fold, thread, team in ((64, 512, 100, 101), (128, 512, 200, 100), (256, 512, 400, 100), (512, 512, 800, 100),
                                      (512, 32, 100, 100), (512, 128, 300, 100)):
        cells += [_cell("thread", width, fold, thread), _cell("team", width, fold, team)]
    row = sweep.wide_row("sm_61", "P2000", "float32", sweep.winners(cells), {"TC": 40960})
    assert (row["SLICE_MIN"], row["TEAM_MIN"], row["TC"]) == (128, 128, 40960) and row["tie"]       # 64 and fold 32 tie
    none = sweep.wide_row("sm_61", "P2000", "float32", sweep.winners([_cell("thread", 64, 128, 10), _cell("team", 64, 128, 90)]),
                          {"TC": 1})
    assert none["SLICE_MIN"] == 128 and none["TEAM_MIN"] == 256       # nothing reached: the form stays out


def test_the_box_row_is_the_fastest_T_and_a_close_second_is_a_tie():
    row = sweep.box_row("sm_61", "P2000", "float64", [{"T": 2, "ns": 5.0}, {"T": 8, "ns": 3.0}, {"T": 16, "ns": 3.2}])
    assert row == {"cc": "sm_61", "device": "P2000", "dtype": "float64", "T": 8, "tie": True}


@pytest.mark.gpu
def test_the_quick_sweep_runs_on_the_device_and_writes_candidate_rows(tmp_path):
    out = tmp_path / "rows.json"
    assert sweep.main(["--quick", "--dtype", "f32", "--out", str(out), "--cache", str(tmp_path / "cache")]) == 0
    got = json.loads(out.read_text())
    assert got["wide"]["rows"][0]["dtype"] == "float32" and got["thread_box"]["rows"] and len(got["cells"]) == 9
    assert all(c["local_B"] is not None for c in got["cells"])
    assert json.loads(cards.WIDE_FILE.read_text())["rows"] == []        # never in place without --write
