# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(mask)[word] |= bits``: a bit mask written by key.

An ``Int32`` / ``UInt32`` plane is the mask (a word per key, 32 bits each; there
is no 64-bit integer plane), and every lane ORs its bits into the word its key
names. The commit is the target's atomic OR (``atomicOr`` on the device, an
``atomic_ref`` under the host team) and the C target's serial ``|=`` at the
clamped row; the result is the exact integer ``np.bitwise_or.at`` gives, for any
number of lanes hitting one word, on every target, bit for bit. A bit mask has
no derivative: ``vjp`` and ``jvp`` refuse by name.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Index, Int32, Mutable, Scalar, UInt32, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp

N = 1000
WORDS = 7


@kernel
def mark(word: Index, bit: UInt32, mask: Mutable[UInt32]):
    hawk.scatter(mask)[word] |= bit


@kernel
def mark_signed(word: Index, bit: Int32, mask: Mutable[Int32]):
    hawk.scatter(mask)[word] |= bit


def _inputs(dtype):
    rng = np.random.default_rng(3)
    # most lanes hit word 0: the contended case; the rest spread over the others
    word = np.where(rng.random(N) < 0.5, 0, rng.integers(0, WORDS, N)).astype(np.int32)
    bit = (np.uint32(1) << (np.arange(N) % 32).astype(np.uint32)).astype(dtype)
    return word, bit


def _oracle(word, bit, dtype):
    want = np.zeros(WORDS, dtype)
    np.bitwise_or.at(want, word, bit)
    return want


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle([mark, mark_signed], tmp_path_factory.mktemp("or"),
                        mode="float64", targets=("cuda", "host"), cache_dir=cache_dir)


@pytest.mark.parametrize("name, dtype", [("mark", np.uint32), ("mark_signed", np.int32)])
def test_many_lanes_into_one_word_equal_bitwise_or_at_on_the_host(bundle, name, dtype):
    word, bit = _inputs(dtype)
    mask = np.zeros(WORDS, dtype)
    _launch(bundle, name, "host", {"word": word, "bit": bit, "mask": mask}).launch()
    assert mask.tobytes() == _oracle(word, bit, dtype).tobytes()
    assert mask[0] == np.uint32(0xFFFFFFFF).astype(dtype)      # word 0 saw every bit


def test_an_or_keeps_the_bits_the_caller_already_set(bundle):
    word, bit = _inputs(np.uint32)
    start = np.arange(WORDS, dtype=np.uint32) << 28
    mask = start.copy()
    _launch(bundle, "mark", "host", {"word": word, "bit": bit, "mask": mask}).launch()
    assert mask.tobytes() == (_oracle(word, bit, np.uint32) | start).tobytes()


def test_a_repeated_launch_is_idempotent(bundle):
    word, bit = _inputs(np.uint32)
    mask = np.zeros(WORDS, np.uint32)
    for _ in range(3):
        _launch(bundle, "mark", "host", {"word": word, "bit": bit, "mask": mask}).launch()
    assert mask.tobytes() == _oracle(word, bit, np.uint32).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("name, dtype", [("mark", np.uint32), ("mark_signed", np.int32)])
def test_the_device_equals_bitwise_or_at(bundle, name, dtype):
    import cupy as cp

    word, bit = _inputs(dtype)
    mask = cp.zeros(WORDS, dtype)
    _launch(bundle, name, "cuda", {"word": cp.asarray(word), "bit": cp.asarray(bit),
                                    "mask": mask}).launch()
    cp.cuda.Device().synchronize()
    assert mask.get().tobytes() == _oracle(word, bit, dtype).tobytes()


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_a_bit_mask_is_refused_by_name(derive):
    with pytest.raises(HawkError, match=r"sets bits by key.*no derivative"):
        derive(mark)


def test_refuses_what_is_not_a_bit_mask_store():
    with pytest.raises(HawkError, match=r"contract='or' is gone: write hawk\.scatter\(m\)\[key\] \|= bits"):
        @kernel
        def k(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m, contract="or")[w] |= b
    with pytest.raises(HawkError, match=r"a bit-or has no sum contract"):
        @kernel
        def k2(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m, contract="ordered")[w] |= b
    with pytest.raises(HawkError, match=r"sets bits of an Int32 or UInt32 plane"):
        @kernel
        def k3(w: Index, x: Scalar, m: Mutable[Scalar]):
            hawk.scatter(m)[w] |= x
    with pytest.raises(HawkError, match=r"only `hawk.scatter\(y\)\[key\] \+= t` accumulates"):
        @kernel
        def k4(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m)[w] &= b


@kernel
def mark_two_axes(i: Index, k: Index, bit: UInt32, mask: Mutable[UInt32, "bin":3, "slot":4]):  # noqa: F821 axis labels
    hawk.scatter(mask)[i, k] |= bit


def test_many_lanes_into_a_plane_with_two_axes_equal_bitwise_or_at_on_the_host(tmp_path, cache_dir):
    rng = np.random.default_rng(11)
    i = rng.integers(0, 3, N).astype(np.int32)
    k = rng.integers(0, 4, N).astype(np.int32)
    bit = (np.uint32(1) << (np.arange(N) % 32).astype(np.uint32)).astype(np.uint32)
    bundle = build_bundle([mark_two_axes], tmp_path, mode="float64", targets=("host",), cache_dir=cache_dir)
    mask = np.zeros((3, 4), np.uint32)
    _launch(bundle, "mark_two_axes", "host", {"i": i, "k": k, "bit": bit, "mask": mask}).launch()
    want = np.zeros((3, 4), np.uint32)
    np.bitwise_or.at(want, (i, k), bit)
    assert mask.tobytes() == want.tobytes()


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_a_bit_mask_with_axes_is_refused_by_name(derive):
    with pytest.raises(HawkError, match=r"sets bits by key.*no derivative"):
        derive(mark_two_axes)
