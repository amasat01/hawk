# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The team stage's body text is written on the emit side, from the lowering component's team description."""


def test_team_emit_writes_a_body_through_the_given_constructor():
    from aether.wings.base import team_emit
    from lowering import team

    text = team_emit.team_body(team.scatter_specs("t", 2, 256, 32)[0], None, False, body=lambda t: t)
    assert isinstance(text, str) and team_emit.TEAM_CALL in text


def test_the_wrapper_helper_texts_are_the_emit_sides():
    from aether.wings.base import wrapper_text as emit_text

    assert "{qualifier}" in emit_text.SAMPLE_SUM and "finish_count" in emit_text.FINISH_COUNT


def test_the_device_wrapper_text_rewrites_the_sample_sum_for_banded_cells():
    from aether.wings.base.wrapper_text import SAMPLE_SUM
    from aether.wings.simt import wrapper_text as device_text

    assert device_text.banded_sample_sum(SAMPLE_SUM) != SAMPLE_SUM


def test_the_lowering_spellings_and_the_box_entry_text_are_the_emit_sides():
    from hawk.targets import _spellings
    from aether.wings.base import aether, box
    from lowering.box import Box

    assert {n for _, n in aether.SPELLINGS} == {"team_store", "team_read", "box_store"}
    assert _spellings(("host",))[("aether", "lowering")] == "aether.wings.base.aether"
    b = Box(0, 4, 1)
    assert box.BOX_OWN in box.owned_lane("long long", "t", b) and "x" in box.frame(b, "plain", "x")
