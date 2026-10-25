# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Sorted[Index]``: a per-lane Index plane whose values never decrease along the
flat lane. A promise the compiler may build on; the bind verifies it only when
``HAWK_CHECK_SORTED`` is set (an O(n) pass), refusing at the first position that
breaks it. The plane is otherwise an ``Index``, and an ``Index`` binds an int32
column as well as an int64 one."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import HawkError, Index, Mutable, Scalar, Sorted
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.ir import Leaf
from hawk.trace.decl import Plane

SEGMENTS = 5


@hawk.kernel
def by_segment(x: Scalar, seg: Sorted[Index], acc: Mutable["segment":SEGMENTS]):
    hawk.scatter(acc)[seg] += x


@hawk.kernel
def by_index(x: Scalar, seg: Index, acc: Mutable["segment":SEGMENTS]):
    hawk.scatter(acc)[seg] += x


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("hawk_sorted")
    out = {}
    for k in (by_segment, by_index):
        bundle = build_bundle([k], pathlib.Path(root) / k.name, targets=("host",),
                              cache_dir=cache_dir, host_profile="native")
        out[k.name] = rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name))
    return out


def _data(n=40):
    seg = np.sort(np.random.default_rng(4).integers(0, SEGMENTS, n)).astype(np.int64)
    x = np.random.default_rng(5).random(n)
    return x, seg, np.bincount(seg, weights=x, minlength=SEGMENTS)


def _leaf(kernel, name):
    stack, seen = list(kernel.sinks), set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Leaf) and node.name == name:
            return node
        stack.extend(node.operands)
    raise AssertionError(name)


def test_the_declaration_is_an_index_plane_with_a_flag():
    assert Sorted[Index] == Plane("in", Index, sorted=True)
    assert Plane("in", Index).sorted is False


def test_only_an_index_can_be_sorted():
    with pytest.raises(HawkError, match=r"Sorted\[\.\.\.\] takes Index \(a per-sample segment id\)"):
        Sorted[Scalar]


def test_the_flag_reaches_the_leaf():
    assert _leaf(by_segment, "seg").sorted is True
    assert _leaf(by_index, "seg").sorted is False


def test_the_sidecar_names_the_sorted_inputs_only_when_declared(loaded):
    assert loaded["by_segment"].sidecar["sorted_inputs"] == ["seg"]
    assert "sorted_inputs" not in loaded["by_index"].sidecar


def test_a_sorted_plane_binds_and_sums(loaded):
    x, seg, want = _data()
    acc = np.zeros(SEGMENTS)
    rt.run(loaded["by_segment"], x=x, seg=seg, acc=acc)
    np.testing.assert_allclose(acc, want)


def test_the_check_is_off_unless_asked(loaded, monkeypatch):
    monkeypatch.delenv("HAWK_CHECK_SORTED", raising=False)
    x, seg, _ = _data()
    seg[7], seg[8] = seg[8] + 1, seg[7]               # break the order
    rt.run(loaded["by_segment"], x=x, seg=seg, acc=np.zeros(SEGMENTS))


def test_the_check_refuses_the_first_offending_position(loaded, monkeypatch):
    monkeypatch.setenv("HAWK_CHECK_SORTED", "1")
    x, seg, want = _data()
    acc = np.zeros(SEGMENTS)
    rt.run(loaded["by_segment"], x=x, seg=seg, acc=acc)         # sorted: passes
    np.testing.assert_allclose(acc, want)
    seg[11], seg[30] = 4, 0                           # two breaks: the first is named
    assert seg[12] < 4
    with pytest.raises(HawkError, match=rf"'seg' is declared Sorted\[Index\] but position 12 "
                                        rf"\({seg[12]}\) is below position 11 \(4\)"):
        rt.run(loaded["by_segment"], x=x, seg=seg, acc=np.zeros(SEGMENTS))


def test_the_check_only_reads_the_launched_lanes(loaded, monkeypatch):
    monkeypatch.setenv("HAWK_CHECK_SORTED", "1")
    x, seg, _ = _data()
    seg[35] = 0                                       # past the 20 lanes launched
    rt.run(loaded["by_segment"], x=x, seg=seg, acc=np.zeros(SEGMENTS), n_samples=20)


def test_an_unsorted_index_is_never_checked(loaded, monkeypatch):
    monkeypatch.setenv("HAWK_CHECK_SORTED", "1")
    x, seg, _ = _data()
    rt.run(loaded["by_index"], x=x, seg=seg[::-1].copy(), acc=np.zeros(SEGMENTS))


@pytest.mark.parametrize("name", ["by_segment", "by_index"])
def test_an_index_binds_an_int32_column(loaded, name):
    x, seg, want = _data()
    acc = np.zeros(SEGMENTS)
    rt.run(loaded[name], x=x, seg=seg.astype(np.int32), acc=acc)
    np.testing.assert_allclose(acc, want)


def test_the_int32_column_is_checked_for_order_too(loaded, monkeypatch):
    monkeypatch.setenv("HAWK_CHECK_SORTED", "1")
    x, seg, _ = _data()
    seg = seg[::-1].astype(np.int32)
    with pytest.raises(HawkError, match=r"Sorted\[Index\] but position"):
        rt.run(loaded["by_segment"], x=x, seg=seg, acc=np.zeros(SEGMENTS))


def test_other_integer_widths_and_floats_are_still_refused(loaded):
    x, seg, _ = _data()
    for bad in (seg.astype(np.int16), seg.astype(np.float64)):
        with pytest.raises(HawkError, match=r"'seg'.*expected dtype int64"):
            rt.run(loaded["by_index"], x=x, seg=bad, acc=np.zeros(SEGMENTS))
