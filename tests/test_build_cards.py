# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A build states other numbers for the lowering to decide on through ``build_spec(cards=...)``.

These tests run the real lowering library: the numbers a build states reach both its facts call and its request, and one changed
number moves a decision the unit text shows."""

from __future__ import annotations

import json

import pytest
import _wide_kernels as W
from _linalg_kernels import chol_kernel
from test_chain import fan_in
from test_team_linalg import ROWS

from aether.wings.base.wrapper import scalar_mode
import hawk
import hawk.math as m
from hawk import Grid, HawkError, Matrix, Vector, _ir, kernel
from hawk.artifact import bundle, request, unit_cache
from hawk.targets import build_spec, shipped_card

pytest.importorskip("_osprey")

CTX = _ir.new_context()  # the context of the shared kernel builders
NAMES = ("build.json", "cuda_build.json", "cuda.json", "arm.json", "x86.json")


def _plan(spec, targets=("host",), kernel=None):
    """``(request, dump, dialects)`` of ``kernel`` (default: the 40 x 40 Cholesky kernel, a team kernel) for ``targets`` under ``spec``."""
    kernel = kernel or chol_kernel(CTX, 40)
    spec = spec.with_settings(scalar="float64", linalg_form="team")
    return bundle._plans([kernel], [kernel.name], [None], smode=scalar_mode("float64"), targets=targets, spec=spec, options={})[0]


def _texts(spec, targets=("host",)):
    """The text of every unit the library writes for the Cholesky kernel, as ``[(target, text)]``."""
    answer = request.lower(*_plan(spec, targets))
    return [(text["target"], text["text"]) for unit in answer["units"] for text in unit["texts"]]


def _team_sizes(spec) -> set:
    """The thread counts of the team records among the stages the library states for the Cholesky kernel."""
    answer = request.lower(*_plan(spec))
    return {stage["team"]["threads"] for unit in answer["units"] for stage in unit["meta"]["stages"] if stage["team"]}


def test_the_shipped_document_of_every_name_builds_the_same_unit_texts_as_a_build_without_cards():
    # Oracle: the build without cards. Only the facts digest member differs between the two answers, and it is no part of a unit text.
    plain = _texts(build_spec(("host", "cuda"), "sm_61"), ("host", "cuda"))
    for name in NAMES:
        stated = build_spec(("host", "cuda"), "sm_61", cards={name: shipped_card(name)})
        assert _texts(stated, ("host", "cuda")) == plain, name


def _smaller_team_card() -> dict:
    """The shipped host build card with the smaller team's matrix order lowered from 128 to 16."""
    card = shipped_card("build.json")
    assert card["lanes"]["team_threads"] == 128
    card["lanes"]["team_threads"] = 16
    return card


def test_one_changed_number_of_the_host_build_card_moves_the_team_size_of_a_cholesky_kernel():
    # Oracle: the library's own test of this rule (tests/own_cards.rs): the smaller team serves a matrix of order up to
    # `team_threads`, the larger team above it. The 40 x 40 matrix is above 16 and below 128, so the shipped build takes the smaller
    # team (128 threads) and the stated one the larger (`team_threads_large`, 256); the other stage of the kernel keeps its 256.
    card = _smaller_team_card()
    assert _team_sizes(build_spec(("host",), "")) == {128, 256}
    assert _team_sizes(build_spec(("host",), "", cards={"build.json": card})) == {card["lanes"]["team_threads_large"]} == {256}


def test_the_unit_key_differs_under_other_cards():
    # The key hashes the request text (the spelling `bundle.build_bundle` uses), which carries the cards.
    keys = []
    for spec in (build_spec(("host",), ""), build_spec(("host",), "", cards={"build.json": _smaller_team_card()})):
        text = json.dumps(_plan(spec)[0], sort_keys=True)
        keys.append(unit_cache._unit_key([text], [""], targets=("host",), defines=(), cache_dir="", device_arch="", derivative_blocks={}))
    assert keys[0] != keys[1]


def test_a_facts_answer_and_a_request_made_under_different_cards_are_refused_with_the_librarys_message():
    # The facts answer inside `plain` was made without cards; the request below states the other build's cards.
    plain, stated = _plan(build_spec(("host",), "")), _plan(build_spec(("host",), "", cards={"build.json": _smaller_team_card()}))
    crossed = {**plain[0], "cards": stated[0]["cards"]}
    with pytest.raises(HawkError, match="made under other cards than the request"):
        request.lower(crossed, plain[1], plain[2])


def test_the_old_entries_are_refused_as_unknown_arguments():
    for old in ("device", "wide_override"):
        with pytest.raises(TypeError, match=old):
            build_spec(("host",), "", **{old: {}})


def test_a_document_given_as_text_is_refused_by_name():
    with pytest.raises(HawkError, match=r"cards\['build.json'\] must be a mapping"):
        build_spec(("host",), "", cards={"build.json": "{}"})


def test_the_library_refuses_an_unknown_document_name_and_an_unknown_key_by_name():
    with pytest.raises(HawkError, match="shipped_card"):
        shipped_card("cpu.json")
    with pytest.raises(Exception, match="build.json"):     # the library's own message reaches the user, whatever its class
        _texts(build_spec(("host",), "", cards={"build.json": {"nonsense": 1}}))


def test_the_shipped_document_is_a_mapping_the_caller_can_edit_and_pass_back():
    card = shipped_card("build.json")
    assert isinstance(card, dict) and card["lanes"]["team_threads"] == 128


def _stages(kernel, spec, targets) -> list:
    """The stage count of each unit the library writes for ``kernel`` (0: the unit is not cut into stages)."""
    answer = request.lower(*_plan(spec, targets, kernel))
    return [len(unit["meta"]["stages"]) for unit in answer["units"]]


def _edited(name: str, edit) -> dict:
    """``{name: the shipped document with ``edit`` applied}``, the ``cards`` of one changed number."""
    document = shipped_card(name)
    edit(document)
    return {name: document}


def _outer_sum(n):
    @kernel
    def outer_sum(lane: Grid["key", "row":ROWS], a: Vector[n], b: Vector[n]) -> Matrix[n, n]:  # noqa: F821  (hawk axis labels)
        return hawk.sum(m.outer(a, b), over="row")

    return outer_sum


@pytest.mark.parametrize("limit,units", [(4096, 3), (64, 3), (32, 1)])
def test_a_keyed_sum_falls_back_above_the_team_dimension_the_host_build_card_states(limit, units):
    # Oracle: the old rule, a keyed sum of a 40 x 40 payload is worked by a team while 40 is at most `linalg_team_max_dim`
    # (the shipped 4096 and 64: team stages and their units), and above it (32) it falls back to the plain form (one unit).
    spec = build_spec(("host",), "", cards=_edited("build.json", lambda card: card.update(linalg_team_max_dim=limit)))
    assert len(_stages(_outer_sum(40), spec.with_settings(linalg_form="team"), ("host",))) == units


def _never_places(card):
    """The device build card with the placement thresholds of every wide class far above any value."""
    for wide_class in card["wide_classes"].values():
        wide_class["runtime_min"] = {pattern: {"4": 10**6, "8": 10**6} for pattern in wide_class["runtime_min"]}
        wide_class["wide_min"] = {"4": 10**6, "8": 10**6}


def test_the_tables_of_the_selected_wide_class_decide_the_placement():
    # Oracle: the old test of this rule: the shipped class places the rotate kernel's wide values (one stage), a class whose
    # thresholds are all 10**6 places none (no stage).
    placed = _stages(W.rotate16, build_spec(("cuda",), "sm_61"), ("cuda",))
    kept = _stages(W.rotate16, build_spec(("cuda",), "sm_61", cards=_edited("cuda_build.json", _never_places)), ("cuda",))
    assert (placed, kept) == ([1], [0])


@pytest.mark.parametrize("arch,stages", [("sm_61", [1]), ("sm_75", [1]), ("sm_80", [0])])
def test_a_known_arch_selects_its_own_wide_class_and_an_unknown_one_the_nearest(arch, stages):
    # Oracle: the old test's feature rows: a class measured on capability 70 (here placing nothing) is nearest sm_80 (FP64 1:2, 2048
    # threads as Volta), while sm_61 is its own class and sm_75 (1:32, 1024 threads) is one step from Pascal.
    def add_class(card):
        never = shipped_card("cuda_build.json")
        _never_places(never)
        card["wide_classes"]["fake70"] = {**never["wide_classes"]["sm61"], "capability": 70}

    spec = build_spec(("cuda",), arch, cards=_edited("cuda_build.json", add_class))
    assert _stages(W.rotate16, spec, ("cuda",)) == stages


def _refused(name, edit, targets, arch, match):
    """Builds the Cholesky kernel under the shipped ``name`` with ``edit`` applied: the library's message ``match`` reaches the user."""
    with pytest.raises(Exception, match=match):     # the class of a library refusal is not pinned: only its message
        _texts(build_spec(targets, arch, cards=_edited(name, edit)), targets)


def test_a_slice_minimum_that_is_not_a_whole_number_of_warps_is_refused_by_the_library():
    _refused("cuda_build.json", lambda card: card["wide"].update(slice_min=100), ("cuda",), "sm_61", "slice_min 100 is not a whole number of warps of 32")


@pytest.mark.parametrize("change,message", [({"slice_min": "64"}, "slice_min is not an integer"), ({"team_min": 0}, "team_min must be positive")])
def test_a_malformed_wide_row_is_refused_by_the_library(change, message):
    row = {"cc": "sm_61", "dtype": "float64", "slice_min": 64, "team_min": 128, "tile_bytes": 40960, **change}
    _refused("cuda_build.json", lambda card: card.update(wide_rows=[row]), ("cuda",), "sm_61", message)


def test_an_unknown_contraction_route_is_refused_by_the_library():
    _refused("cuda_build.json", lambda card: card["thread_box"]["contraction"].update(float32="blas"), ("cuda",), "sm_61", "contraction must be gemm or direct")


def test_a_device_card_without_cells_row_is_refused_by_name():
    _refused("cuda_build.json", lambda card: card.pop("cells_row"), ("cuda",), "sm_61", "cells_row missing")


def test_a_host_card_without_cells_row_is_refused_by_name():
    _refused("build.json", lambda card: card.pop("cells_row"), ("host",), "", "cells_row")


def _fan_in_chain(members, width):
    return hawk.chain(*fan_in(members, width), name="fan_in_chain")


def test_a_host_fusion_row_with_a_small_register_file_cuts_a_chain_the_shipped_row_keeps_whole():
    # Oracle: the library's own test of this rule (chain.rs, own_cards.rs): a row that leaves one register for a lane cuts more.
    # Three members of width 2 fit the shipped register file (one unit, no stages) and are cut into two stages under one register.
    def one_register(card):
        for row in card["fusion_host"].values():
            row["regs_file"] = 1

    chain = _fan_in_chain(3, 2)
    cards = {**_edited("arm.json", one_register), **_edited("x86.json", one_register)}
    assert _stages(chain, build_spec(("host",), ""), ("host",)) == [0]
    assert _stages(chain, build_spec(("host",), "", cards=cards), ("host",)) == [2, 0]


def test_a_device_fusion_row_of_the_build_architecture_and_type_cuts_a_chain_the_shipped_row_keeps_whole():
    # Oracle: the same rule on the device document: a row for (sm_61, each type) with a register file of 64 lanes' registers cuts
    # the six-member chain of width 8 that the shipped file of 255 keeps whole.
    def row_of_64(card):
        card["fusion_rows"] = [{"cc": "sm_61", "dtype": dtype, "regs_file": 64, "local_over_plane": None} for dtype in ("float32", "float64")]

    chain = _fan_in_chain(6, 8)
    assert _stages(chain, build_spec(("cuda",), "sm_61"), ("cuda",)) == [0]
    assert _stages(chain, build_spec(("cuda",), "sm_61", cards=_edited("cuda.json", row_of_64)), ("cuda",)) == [2, 0]
