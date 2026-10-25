# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thread box (:mod:`hawk.emit.box`): one thread computes ``T`` lanes along
a non-fastest axis, reading what does not move along it once.

On the contraction forward shape the emitted entry carries an ``Item<Real, T>`` and the
same reads per iteration as the box-free body; every lane's value equals the
box-free artifact's bit for bit (launches whose rows fill the boxes, end in a
partial box, or are fewer than ``T``); ``HAWK_BOX=0`` and a kernel without a box
candidate keep today's text; a card row sets ``T``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from conftest import _launch, sidecar_of
from test_ir_lanes import BATCH, L_ALL, N_IN, S_ROW, k_clamp, k_fem, kg_fwd

import hawk
import hawk.math as m
from hawk.artifact import build_bundle
from hawk.artifact.bundle import _sources
from hawk.emit import scalar_mode
from hawk.emit.box import BOX_ENV, Box, thread_box, threads
from hawk.targets import cards

T64 = 16


def _text(kernel, target="cuda", mode="float64"):
    return _sources(kernel, name=kernel.name, smode=scalar_mode(mode), kind=None,
                    targets=(target,), layout_sizes_override=None)[target].text


def _boxed_branch(text):
    return text[text.index("} else {"):]


def _plain_body(text):
    return text[text.index("for (Int hawk_k"):text.index("} else {")]


def _data(rows, mode):
    rng = np.random.default_rng(rows)
    feat = rng.uniform(-1, 1, 8 * L_ALL).astype(mode)
    feat[:L_ALL] = rng.integers(0, 5, L_ALL)
    th = rng.standard_normal(rows * N_IN * S_ROW).astype(mode)
    return feat, th


def _build(tmp_path, monkeypatch, name, mode, targets, box=True):
    if not box:
        monkeypatch.setenv(BOX_ENV, "0")
    bundle = build_bundle([kg_fwd], tmp_path / name, mode=mode, targets=targets,
                          host_profile="native")
    monkeypatch.delenv(BOX_ENV, raising=False)
    return bundle


def _run(bundle, target, rows, mode):
    xp = np if target == "host" else pytest.importorskip("cupy")
    n = rows * BATCH
    feat, th = _data(rows, mode)
    y = xp.full(n, np.nan, dtype=mode)
    _launch(bundle, "kg_fwd", target, {"lane": xp.arange(n, dtype=np.int64),
                                       "feat": xp.asarray(feat), "th": xp.asarray(th),
                                       "y": y}).launch()
    return np.asarray(y.get() if hasattr(y, "get") else y)


# ---- the decision and the text ---------------------------------------------
def test_the_contraction_forward_gets_the_card_default_box():
    assert thread_box(kg_fwd, "float64") == Box(0, T64, BATCH)
    assert thread_box(kg_fwd, "float32") == Box(0, 8, BATCH)


@pytest.mark.parametrize("target", ["cuda", "host"])
def test_the_box_holds_the_lanes_in_an_item_and_reads_feat_once(target):
    text = _text(kg_fwd, target)
    boxed, plain = _boxed_branch(text), _plain_body(text)
    assert "aether::Item<Real, 16> hawk_v" in boxed
    # the reads that do not move along the box axis: as many as one lane makes
    assert boxed.count("lut_feat[") == plain.count("lut_feat[") == 5
    assert boxed.count("lut_th[") == T64 * plain.count("lut_th[")
    # every lane stores only what it owns
    assert boxed.count(">= hawk_box_own) mut_y[") == T64


def test_hawk_box_0_and_a_kernel_without_a_candidate_keep_the_box_free_text(monkeypatch):
    boxed = {k.name: _text(k) for k in (kg_fwd, k_fem, k_clamp)}
    monkeypatch.setenv(BOX_ENV, "0")
    assert thread_box(kg_fwd, "float64") is None
    free = {k.name: _text(k) for k in (kg_fwd, k_fem, k_clamp)}
    assert "hawk_box_own" not in free["kg_fwd"] and "hawk_box_own" in boxed["kg_fwd"]
    monkeypatch.delenv(BOX_ENV)
    for k in (k_fem, k_clamp):
        assert thread_box(k, "float64") is None
        assert hashlib.md5(boxed[k.name].encode()).hexdigest() \
            == hashlib.md5(free[k.name].encode()).hexdigest()


def test_a_card_row_sets_T(tmp_path, monkeypatch):
    card = json.loads(cards.THREAD_BOX_FILE.read_text())
    card["rows"] = [{"cc": "sm_00", "device": "Test Card", "dtype": "float64", "T": 4}]
    path = tmp_path / "thread_box.json"
    path.write_text(json.dumps(card))
    monkeypatch.setattr(cards, "THREAD_BOX_FILE", path)
    monkeypatch.setattr(cards, "_device", lambda: ("sm_00", "Test Card"))
    assert thread_box(kg_fwd, "float64") == Box(0, 4, BATCH)
    assert "aether::Item<Real, 4> hawk_v" in _text(kg_fwd)
    monkeypatch.setattr(cards, "_device", lambda: ("sm_61", "Another Card"))
    assert thread_box(kg_fwd, "float64").T == T64


def test_a_card_T_outside_the_register_budget_is_refused(tmp_path, monkeypatch):
    card = json.loads(cards.THREAD_BOX_FILE.read_text())
    card["defaults"]["float64"] = 32
    path = tmp_path / "thread_box.json"
    path.write_text(json.dumps(card))
    monkeypatch.setattr(cards, "THREAD_BOX_FILE", path)
    with pytest.raises(hawk.HawkError, match="2..16"):
        thread_box(kg_fwd, "float64")


@hawk.kernel
def k_row(lane: hawk.Index, feat: hawk.Table[hawk.Scalar], th: hawk.Table[hawk.Scalar],
          y: hawk.Mutable[hawk.Scalar]):
    j, s = m.split_index(BATCH)
    y = th.at(j) * feat.at(s)


@hawk.kernel(group=2 * BATCH)
def k_row_grouped(lane: hawk.Index, feat: hawk.Table[hawk.Scalar], th: hawk.Table[hawk.Scalar],
                  y: hawk.Mutable[hawk.Scalar]):
    j, s = m.split_index(BATCH)
    y = th.at(j) * feat.at(s)


def test_a_box_that_would_straddle_a_compaction_group_is_not_emitted():
    assert thread_box(k_row, "float64") == Box(0, T64, BATCH)
    assert thread_box(k_row_grouped, "float64") is None


def test_the_sidecar_carries_the_box_and_the_thread_rule(tmp_path, monkeypatch):
    bundle = _build(tmp_path, monkeypatch, "b", "float64", ("host",))
    box = sidecar_of(bundle, "kg_fwd")["launch"]["box"]
    assert box == {"axis": 0, "T": T64, "stride": BATCH}
    assert threads(box, 64 * BATCH) == 4 * BATCH
    assert threads(box, 13 * BATCH) == BATCH
    assert threads(box, 17 * BATCH) == 2 * BATCH
    assert threads(None, 1000) == 1000


# ---- the values ---------------------------------------------------------------
#: rows that fill the boxes, end in a partial (shifted) box, and are fewer than T
ROWS = (32, 21, 13)


@pytest.fixture(scope="module")
def host_pair(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("box_host")
    out = {mode: (_build(root, mp, mode, mode, ("host",)),
                  _build(root, mp, mode + "_free", mode, ("host",), box=False))
           for mode in ("float64", "float32")}
    mp.undo()
    return out


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_host_values_are_the_box_free_values_bit_for_bit(host_pair, mode, rows):
    boxed, free = host_pair[mode]
    assert sidecar_of(free, "kg_fwd")["launch"].get("box") is None
    got, want = _run(boxed, "host", rows, mode), _run(free, "host", rows, mode)
    assert not np.isnan(got).any()
    assert np.array_equal(got, want)


def test_hawk_run_counts_threads(host_pair):
    boxed, free = host_pair["float64"]
    n = 21 * BATCH
    feat, th = _data(21, "float64")
    out = {}
    for tag, bundle in (("boxed", boxed), ("free", free)):
        art = hawk.load(bundle.directory, "kg_fwd")
        out[tag] = np.full(n, np.nan)
        hawk.run(art, lane=np.arange(n, dtype=np.int64), feat=feat, th=th, y=out[tag])
    assert np.array_equal(out["boxed"], out["free"])


@pytest.mark.gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_device_values_are_the_box_free_values_and_near_the_host(tmp_path, monkeypatch,
                                                                 host_pair, mode, rows):
    boxed = _build(tmp_path, monkeypatch, "d", mode, ("cuda",))
    free = _build(tmp_path, monkeypatch, "d_free", mode, ("cuda",), box=False)
    got, want = _run(boxed, "cuda", rows, mode), _run(free, "cuda", rows, mode)
    assert np.array_equal(got, want)
    host = _run(host_pair[mode][0], "host", rows, mode)
    # the device may fuse a multiply into an add the host keeps apart
    np.testing.assert_allclose(got, host, rtol=64 * np.finfo(mode).eps,
                               atol=64 * np.finfo(mode).eps * np.abs(host).max())
