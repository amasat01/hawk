# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Lane fusion's body path: one body per fused lane group. Each lane goes
through the same renderer every backend consumes, into its own C++ scope
opened by a lane-local prologue hook the composing consumer supplies. A
fused lane's text is byte-identical to what that lane emits alone, and each
lane is a real scope, so two lanes' ``const auto`` names cannot collide.
"""

from __future__ import annotations

import pytest
from _emitted import cases

from hawk.emit import Lane, render_body, render_lane_body
from hawk.ir import HawkError
from hawk.types import LaneMeta


def _lanes(names):
    picked = {n: (s, w) for n, s, w in cases() if n in names}
    return [Lane(n, *picked[n]) for n in names]


def test_a_fused_lane_body_contains_each_lane_verbatim():
    lanes = _lanes(("drag", "scale", "energy"))
    fused = render_lane_body(lanes)
    for lane in lanes:
        alone = render_body(lane.sinks, lane.walk, indent="            ").text
        assert alone in fused.text, (
            f"lane {lane.name!r} was not emitted by the SAME renderer -- a fused "
            "lane's body must be the body that lane emits alone"
        )
        assert f"{{  // lane {lane.name}" in fused.text


def test_each_lane_is_its_own_scope():
    lanes = _lanes(("drag", "chain6"))
    text = render_lane_body(lanes).text
    assert text.count("{  // lane") == 2 and text.count("}  // lane") == 2, (
        "each lane must be a real C++ block, or two lanes' const auto names "
        "collide:\n" + text
    )


def test_the_lane_local_prologue_hook_is_spliced():
    drag = _lanes(("drag",))[0]
    lanes = [Lane("guarded", drag.sinks, drag.walk,
                  prologue="// lane guard goes here")]
    text = render_lane_body(lanes).text
    assert "// lane guard goes here" in text
    assert text.index("// lane guard") < text.index("mut_out[i]"), (
        "the lane-local hook must open the lane's scope, ahead of its body"
    )


def test_lane_metadata_is_published_in_the_cheap_import_tier():
    lanes = _lanes(("drag", "scale"))
    fused = render_lane_body(lanes)
    assert [type(m) for m in fused.lanes] == [LaneMeta, LaneMeta]
    assert [m.name for m in fused.lanes] == ["drag", "scale"]
    assert fused.lanes[0].slots == lanes[0].walk.arg_spec
    assert fused.lanes[0].digest == lanes[0].walk.digest


def test_an_empty_or_ambiguous_lane_group_is_refused():
    with pytest.raises(HawkError, match="empty lane group"):
        render_lane_body([])
    drag = _lanes(("drag",))[0]
    with pytest.raises(HawkError, match="both named"):
        render_lane_body([drag, drag])
