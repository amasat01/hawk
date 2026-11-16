# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The block-shaped keyed sum of the wing: its predicates on the recorded cases, its constructors and its plain forms."""
import json
from pathlib import Path

import pytest
from raptor.errors import RaptorError
from raptor.ir import make
from raptor.ir.nodes import AccumWrite, Const, Leaf, ReduceSamples, SampleSum, extent_of, has_feather
from raptor.ir.types import TensorType

from raptor.wings.base import block, cells

F64, I32 = TensorType((), "f64"), TensorType((), "i32")
CASES = json.loads((Path(__file__).parent / "data" / "block_cases.json").read_text())


def node(s):
    if s[0] == "leaf":
        return Leaf("vocab_read", "per_sample", f"x{abs(hash(json.dumps(s))) % 997}", TensorType(tuple(s[1]), s[2]))
    return make(s[0], tuple(node(c) for c in s[1:]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["want"] + json.dumps(c["payload"])[:40])
def test_the_predicates_on_the_recorded_cases(case):
    payload = node(case["payload"])
    got = f"ok [{block.block_shaped(payload, case['contract'])}, {block.ordered_case(payload, case['contract'])}]"
    assert got == case["want"]


def gram(p=3):
    u = Leaf("vocab_read", "per_sample", "u", TensorType((p,), "f64"))
    return make("outer", (u, u))


def test_an_all_lane_block_is_one_key_of_one_and_expands_to_keyed_accumulates_or_plain_cells():
    blk = block.BlockSum(gram())
    assert block.is_block(blk) and not block.is_block_write(blk) and blk.ttype.shape == (3, 3)
    assert block.key_of(blk).literal == 0 and block.count_of(blk).literal == 1 and block.factors_of(blk)[2] == 1
    write = block.BlockSumWrite("g", blk)
    assert block.is_block_write(write) and block.block_of(write) is blk and write.name == "g"
    keyed = block.expand_stores((write,))
    assert len(keyed) == 9 and all(isinstance(s, AccumWrite) for s in keyed)
    assert all(isinstance(s, SampleSum) for s in cells.stored(block.plain(blk), "g"))


def test_a_keyed_block_keeps_its_key_its_axes_and_its_contract_through_the_plain_form():
    key = Leaf("vocab_read", "per_sample", "key", I32)
    blk = block.BlockSum(gram(), key=key, count=Const(4, I32), contract="ordered", kept=(("key", None),))
    plain = block.plain(blk)
    assert isinstance(plain, ReduceSamples) and plain.key is key and plain.contract == "ordered" and plain.kept == (("key", None),)
    assert block.expand_stores((block.BlockSumWrite("g", blk),))[0].index is key


def test_a_block_refuses_what_a_reduction_over_samples_refuses():
    with pytest.raises(RaptorError, match="contract 'x' is not one of"):
        block.BlockSum(gram(), contract="x")
    with pytest.raises(RaptorError, match="f64/f32 scalar, vector or matrix"):
        block.BlockSum(Leaf("vocab_read", "per_sample", "n", I32))


def test_a_block_reduces_samples_by_feather_and_is_an_all_samples_extent():
    assert has_feather(block.BlockSum(gram()), "reduces_samples") and extent_of(block.BlockSum(gram())) == "all_samples" and extent_of(gram()) == "per_sample"


def test_the_contract_is_part_of_a_block_identity():
    g = gram()
    assert block.BlockSum(g).dedup_extra != block.BlockSum(g, contract="ordered").dedup_extra == ("ordered",)
