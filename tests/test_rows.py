# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Ragged rows: an offsets plane related to a values plane, and a loop over a row with a stated bound.

``for x in row.bounded(K)`` is the existing run-time bounded loop with the count ``min(length, K)``: a row longer
than ``K`` is truncated at ``K`` and an empty row visits nothing. Values are compared with a NumPy loop over the
same CSR-style data, which is independent of the code under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as hm
from hawk import HawkError, Index, Mutable, Rows, Scalar, runtime
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp

BOUND = 4
#: row lengths 3, 0 (empty), 7 (longer than the bound), 2 and 4 (exactly the bound)
OFFSETS = np.array([0, 3, 3, 10, 12, 16], dtype=np.int32)
ROWS = len(OFFSETS) - 1
RNG = np.random.default_rng(20261122)
VALUES = RNG.uniform(0.5, 2.0, int(OFFSETS[-1]))


@hawk.kernel
def row_sum(r: Rows(Index, Scalar), i: Index, w: Scalar, y: Mutable[Scalar]):
    row = r[i]
    total = 0.0
    for x in row.bounded(4):
        total = total + x * x * w
    y = total + hm.real(row.length)


def _oracle(i, w, values=VALUES):
    """The NumPy loop over the CSR data: at most BOUND entries of row ``i``."""
    start, end = OFFSETS[i], OFFSETS[i + 1]
    kept = values[start : min(end, start + BOUND)]
    return sum(x * x * w for x in kept) + (end - start)


def _env(i, w, values=VALUES):
    return {"r_offsets": OFFSETS, "r_values": values, "i": i, "w": w}


def test_the_planes_the_rows_expand_to_carry_their_relation():
    assert list(row_sum.planes)[:2] == ["r_offsets", "r_values"]
    assert {
        name: plane.group for name, plane in row_sum.planes.items() if plane.group
    } == {
        "r_offsets": ("r", "offsets", "rows"),
        "r_values": ("r", "values", "rows"),
    }


def test_a_row_loop_equals_the_numpy_loop_for_a_short_empty_long_and_exact_row():
    for i in range(ROWS):
        got = evaluate(row_sum.sinks, _env(i, 1.3))["y"]
        np.testing.assert_allclose(got, _oracle(i, 1.3), rtol=1e-13, err_msg=f"row {i}")


def test_a_row_longer_than_the_bound_is_truncated_and_the_length_is_still_reported():
    long_row = 2
    start = OFFSETS[long_row]
    got = evaluate(row_sum.sinks, _env(long_row, 1.0))["y"]
    assert got == pytest.approx(np.sum(VALUES[start : start + BOUND] ** 2) + 7)


def test_the_loop_is_the_existing_bounded_loop_with_a_clamped_run_time_count():
    loops = [n for n in _loop_nodes(row_sum)]
    assert len(loops) == 1
    assert (
        loops[0].field_int("stop") == BOUND and loops[0].field_text("dynamic") == "max"
    )


def _loop_nodes(kernel):
    from hawk import _ir

    seq, _ = _ir.canonical_nodes(kernel.sinks)
    return [n for n in seq if _ir.is_a(n, "loop")]


def test_the_derivatives_through_the_row_loop_equal_the_numpy_formulas():
    w, i = 1.3, 2
    start = OFFSETS[i]
    kept = slice(start, start + BOUND)
    seed = 0.7
    reverse = evaluate(
        vjp(row_sum.sinks),
        {**_env(i, w), "bar_y": seed},
        shapes={"bar_r_values": VALUES.shape},
    )
    want = np.zeros(VALUES.shape)
    want[kept] = 2 * VALUES[kept] * w * seed
    np.testing.assert_allclose(reverse["bar_r_values"], want, rtol=1e-12)
    np.testing.assert_allclose(
        reverse["bar_w"], seed * np.sum(VALUES[kept] ** 2), rtol=1e-12
    )
    dot_values, dot_w = RNG.normal(size=VALUES.shape), 0.4
    forward = evaluate(
        jvp(row_sum.sinks), {**_env(i, w), "dot_r_values": dot_values, "dot_w": dot_w}
    )
    expected = (
        np.sum(2 * VALUES[kept] * dot_values[kept] * w)
        + np.sum(VALUES[kept] ** 2) * dot_w
    )
    np.testing.assert_allclose(forward["dot_y"], expected, rtol=1e-12)


def _bundle(tmp_path, cache_dir):
    return build_bundle(
        [row_sum],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def test_the_sidecar_lists_the_relation(tmp_path, cache_dir):
    meta = sidecar_of(_bundle(tmp_path, cache_dir), "row_sum")
    assert meta["rows"] == {"r": {"offsets": "r_offsets", "values": "r_values"}}


def test_the_rows_run_on_the_host_by_plane_names_a_mapping_or_an_object_alike(
    tmp_path, cache_dir
):
    loaded = runtime.load(_bundle(tmp_path, cache_dir).directory, "row_sum")
    i = np.array([0, 1, 2, 3, 4, 3, 0], dtype=np.int32)
    w = RNG.uniform(0.5, 2, len(i))
    want = [_oracle(int(a), float(b)) for a, b in zip(i, w)]
    for form in (
        {"r_offsets": OFFSETS, "r_values": VALUES},
        {"r": {"offsets": OFFSETS, "values": VALUES}},
        {"r": SimpleNamespace(offsets=OFFSETS, values=VALUES)},
    ):
        y = np.zeros(len(i))
        runtime.run(loaded, **form, i=i, w=w, y=y)
        np.testing.assert_allclose(y, want, rtol=1e-13)


def _trace_refused(body, match):
    """Trace ``body`` as a kernel and expect the refusal ``match``."""
    with pytest.raises(HawkError, match=match):
        hawk.kernel(body)


def test_a_declaration_with_wide_offsets_or_vector_values_is_refused():
    with pytest.raises(HawkError, match="offsets are 32-bit starts"):
        Rows(hawk.Scalar, Scalar)
    with pytest.raises(HawkError, match="values are a rank-0 payload"):
        Rows(Index, hawk.Vector[3])


def test_a_row_loop_with_two_bounds_or_a_pattern_target_is_refused_by_name():
    def two(r: Rows(Index, Scalar), i: Index, y: Mutable[Scalar]):
        total = 0.0
        for x in r[i].bounded(4, 5):
            total = total + x
        y = total

    _trace_refused(two, "ONE compile-time bound K")
