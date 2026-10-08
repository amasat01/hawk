# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.artifact.bundle.COUNTERS`` — the ``builds`` static-graph counter.

``builds`` counts every call to :func:`build_bundle`, an attempt rather than
a success: it is bumped as the first statement in the function body, before
anything can raise, and stays monotonic even when a repeat build of the same
definition is served from the per-process memo. Host-only: publishing needs
a compiler, not a GPU.
"""

from __future__ import annotations

import pytest

import hawk
from hawk.artifact import build_bundle
from hawk.artifact.bundle import COUNTERS, counters_snapshot
from hawk.ir import HawkError

#: Host-only builds: the device arch is never resolved, so no box arch is
#: named here.
DEVICE_ARCH = ""


@hawk.kernel
def _n20i_h1_identity(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """The smallest possible bundle: one Scalar in, one Scalar out, no
    uniform at all."""
    y = x


@hawk.kernel
def _n20i_h1_sample_local(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """``sample_local`` — agrees with ``_n20i_h1_identity``'s axis."""
    y = x


@hawk.kernel
def _n20i_h1_mapreduce(v: hawk.Scalar, total: hawk.Reduce("sum")):
    """``mapreduce`` — DISAGREES with ``_n20i_h1_sample_local``'s axis, so
    bundling the two together trips build_bundle's execution-axis
    refusal (``hawk/artifact/bundle.py`` ~:763-767) — a real, already-
    documented raise, not a fixture-invented one."""
    total.contribute(v)


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


def test_build_bundle_moves_builds_by_exactly_one(tmp_path):
    """One ``build_bundle`` call must move ``builds`` by exactly 1."""
    before = counters_snapshot()
    bundle = build_bundle(
        [_n20i_h1_identity], tmp_path / "ok", targets=("host",),
        device_arch=DEVICE_ARCH,
    )
    after = counters_snapshot()

    assert _delta(before, after) == {"builds": 1}
    assert bundle.digest


def test_a_raising_build_bundle_call_still_moves_builds(tmp_path):
    """A call that raises (disagreeing execution axes) still counts:
    ``builds`` moves by 1 despite the raise, because it counts attempts."""
    before = counters_snapshot()
    with pytest.raises(HawkError, match="ONE execution axis"):
        build_bundle(
            [_n20i_h1_sample_local, _n20i_h1_mapreduce], tmp_path / "bad",
            targets=("host",), device_arch=DEVICE_ARCH,
        )
    after = counters_snapshot()

    assert _delta(before, after) == {"builds": 1}


def test_builds_is_monotonic_never_reset(tmp_path):
    """Two successive builds leave the absolute counter at >= 2 (never reset
    by anything — deliberately NOT wired into
    :func:`~hawk.artifact.bundle.reset_unit_memo`, a different, test-scoped
    mechanism over a different dict)."""
    build_bundle(
        [_n20i_h1_identity], tmp_path / "a", targets=("host",),
        device_arch=DEVICE_ARCH,
    )
    build_bundle(
        [_n20i_h1_identity], tmp_path / "b", targets=("host",),
        device_arch=DEVICE_ARCH,
    )
    assert COUNTERS["builds"] >= 2
