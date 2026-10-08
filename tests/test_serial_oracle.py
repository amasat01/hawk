# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's serial host path IS the oracle.

WHAT THE ROW COMPARES, AND WHICH RULE EACH PAIR IS JUDGED BY. Every fixture
kernel is run three ways and the three are compared under the table — three
DISTINCT comparisons, only one of them bit-exact:

* ``eagle.exec.HostTeam`` (whole, and over k partitions) against
  ``hawk._core``'s serial run — **BIT-identical**. Both arms execute the SAME
  compiled host object on the SAME bytes; the only difference is how many times
  eagle calls it and with which triple. A single bit of difference here is a
  body that read ``count`` where it should have read ``nSamples``, or a
  plane addressed from the tile's base instead of the sample's — never
  "floating point";
* ``eagle.exec.DeviceKernel`` (whole, and over k) against the same serial run —
  **BANDED** at ``S x 2 x eps(dtype)``. A host/device difference is a different
  claim entirely (the third row) and the contract never asks
  for equality there.

THE BAND IS ONE DERIVED ANCHOR, NEVER FITTED. ``S`` is the number of elements
combined, applied relatively against the larger operand — the same single rule
eagle's own rank bed states, restated here rather than imported so the two beds
cannot silently diverge, and NOT widened to whatever the first run happened to
show. A band fitted to an observation is a description nothing maintains.

WHY THE ORACLE IS THE ONE THAT RUNS SERIALLY. ``hawk._core.HostEntry.run``
issues ONE call over ``[0, n)`` with the true ``nSamples``: no tiles, no
threads, no schedule (audited by row). eagle's ``HostTeam`` is the
structure — it tiles — and ``DeviceKernel`` is another. Comparing a structure
against itself certifies nothing, which is why the reference arm is the one that
has no structure at all.

This test previously failed, by two arms of evidence.

(a) STANDING, in the file: ``_planted.count_reading_bundle`` — an artifact whose
body divides by the partition's own ``count`` instead of the ``nsamples`` ROLE
(the failure verbatim) — is compiled by this row and put through the SAME
comparison, which is REQUIRED to detect it. That arm is not a one-off plant; it
runs every time, so the file cannot become vacuous later.

(b) ONE-OFF, on the oracle itself: ``HostEntry::run`` planted to
``fn(params, base, count / 2, n_samples)``, rebuilt — i.e. the reference arm
stopped covering the whole range. Every fixture went red at once::

    AssertionError: axpb under HostTeam(1): NOT bit-identical (max |delta| =
    48.5). Both arms ran the same compiled host object; a difference here is a
    body that read the partition's own extent instead of the sample's... mat_apply... 11678.175000000001... two_outputs... 8281.6875...

The plant was then removed. Note it fired at ``HostTeam(1)`` as well as at k —
which is the point: the oracle is not "the one-partition case of eagle", it is
its own arm, and a defect in it is visible everywhere.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import _oracle as O
import _planted as PL
import numpy as np
import pytest
from conftest import sidecar_of

N = 96

#: How many partitions the "split" arms cut the range into. 3 is deliberate: it
#: does not divide N evenly at every fixture size the file uses, and an even
#: split is the case that hides an off-by-one.
K = 3

_EPS = {"float64": np.finfo(np.float64).eps, "float32": np.finfo(np.float32).eps}


def band(s: int, a, b, scalar_type: str) -> float:
    """The ONE anchor: ``S x 2 x eps(dtype)``, applied relatively.

    ``S`` is the number of elements combined; the relative scale is the larger
    of the two operands (and 1, so a comparison near zero is judged absolutely).
    Derived, never fitted."""
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return s * 2.0 * _EPS[scalar_type] * scale


def _tuple(value):
    # eagle.plan.Plan.run returns a dict keyed by plane name for several
    # outputs; its insertion order follows arg_spec's own order, the same
    # order a plain tuple (the serial oracle's own shape) already carries,
    # so unwrapping it to a tuple of values lines the two zips up correctly.
    if isinstance(value, dict):
        return tuple(value.values())
    return value if isinstance(value, tuple) else (value,)


def _assert_bit_identical(got, want, what: str) -> None:
    for g, w in zip(_tuple(got), _tuple(want)):
        g, w = np.ascontiguousarray(g), np.ascontiguousarray(w)
        assert g.dtype == w.dtype and g.shape == w.shape, (
            f"{what}: shape/dtype disagree — {g.shape}/{g.dtype} vs {w.shape}/{w.dtype}"
        )
        assert g.tobytes() == w.tobytes(), (
            f"{what}: NOT bit-identical (max |delta| = "
            f"{float(np.max(np.abs(g.astype(float) - w.astype(float))))}). Both arms "
            "ran the same compiled host object; a difference here is a body that "
            "read the partition's own extent instead of the sample's"
        )


def _assert_banded(got, want, scalar_type: str, what: str) -> None:
    for g, w in zip(_tuple(got), _tuple(want)):
        g, w = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
        worst = float(np.max(np.abs(g - w))) if g.size else 0.0
        limit = band(g.size, g, w, scalar_type)
        assert worst <= limit, (
            f"{what}: host/device difference {worst} exceeds the ruled band "
            f"S*2*eps*scale = {limit} (S = {g.size})"
        )


def _oracle_run(bundle, kernel, kw):
    """The reference arm: one serial call over the whole range through
    ``hawk._core``."""
    return O.run(bundle.directory, kernel, N, sidecar_of(bundle, kernel), **kw)


def _eagle_host(bundle, kernel, kw, npartitions: int):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, kernel, sidecar_of(bundle, kernel))
    return eplan.plan(plugin, structure=eexec.HostTeam,
                      npartitions=npartitions).run(**kw)


def _eagle_device(bundle, kernel, kw, npartitions: int):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.device_plugin(bundle.directory, kernel, sidecar_of(bundle, kernel))
    return eplan.plan(plugin, structure=eexec.DeviceKernel,
                      npartitions=npartitions).run(**kw)


def _partitions_for(bundle, kernel) -> tuple:
    """1 always; K as well, unless the DECLARED class forbids it (
    a ``cross_sample_write`` body may only be driven single-partition, and
    eagle refuses it at plan time — row is where that refusal is the
    subject)."""
    access = sidecar_of(bundle, kernel)["exec_access"]
    return (1,) if access == "cross_sample_write" else (1, K)


_CASES = D.cases(N)
_IDS = [f"{c[0]}" for c in _CASES]


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES, ids=_IDS)
def test_the_host_structure_is_bit_identical_to_the_serial_oracle(
        built, bundle_name, kernel, kw, _want):
    bundle = built[bundle_name]
    oracle = _oracle_run(bundle, kernel, kw)
    for npartitions in _partitions_for(bundle, kernel):
        got = _eagle_host(bundle, kernel, kw, npartitions)
        _assert_bit_identical(got, oracle,
                              f"{kernel} under HostTeam({npartitions})")


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES, ids=_IDS)
@pytest.mark.gpu
def test_the_device_structure_is_within_the_ruled_band_of_the_oracle(
        built, bundle_name, kernel, kw, _want):
    bundle = built[bundle_name]
    scalar_type = sidecar_of(bundle, kernel)["scalar_type"]
    oracle = _oracle_run(bundle, kernel, kw)
    for npartitions in _partitions_for(bundle, kernel):
        got = _eagle_device(bundle, kernel, kw, npartitions)
        _assert_banded(got, oracle, scalar_type,
                       f"{kernel} under DeviceKernel({npartitions})")


@pytest.mark.parametrize("bundle_name,kernel,kw,want", _CASES, ids=_IDS)
def test_the_oracle_agrees_with_the_numpy_reference(built, bundle_name, kernel,
                                                    kw, want):
    """The oracle is only a reference if it is RIGHT. Compared against the
    fixture's independent numpy expression, band-gated because the two are
    different expressions of the same mathematics (a different toolchain's
    contraction, the fourth row by analogy) — never bit-gated, which the
    contract does not ask for and this file does not invent."""
    bundle = built[bundle_name]
    scalar_type = sidecar_of(bundle, kernel)["scalar_type"]
    got = _oracle_run(bundle, kernel, kw)
    for g, w in zip(_tuple(got), _tuple(want)):
        g, w = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
        limit = band(g.size, g, w, scalar_type)
        worst = float(np.max(np.abs(g - w)))
        assert worst <= limit, (
            f"{kernel}: the serial oracle differs from its numpy reference by "
            f"{worst}, band {limit}"
        )


def test_the_oracle_runs_a_sub_range_when_asked(built):
    """The oracle takes a TRIPLE, not just a length: given ``[base, base+count)``
    it must write exactly that window and leave the rest of the plane untouched.
    Without this, "serial" could mean "always whole" and the rank arm,
    which asks for one rank's window, would be exercising an untested path."""
    bundle = built["axpb"]
    kernel = O.load(bundle.directory, "axpb", sidecar_of(bundle, "axpb"))
    x = D.plane(N, w=1)[0]
    y = np.full(N, -7.5)
    O.runtime.run(kernel, base=10, count=20, n_samples=N, x=x, y=y, a=2.0, b=-1.0)
    want = D.ref_axpb(x, 2.0, -1.0)
    assert y[10:30].tobytes() == want[10:30].tobytes()
    assert np.all(y[:10] == -7.5) and np.all(y[30:] == -7.5), (
        "the serial call wrote outside [base, base+count): it is not honouring "
        "the triple, and every partitioned comparison above is then meaningless"
    )


def test_the_comparison_can_detect_a_body_that_is_not_partition_invariant(
        built, tmp_path, cache_dir):
    """NON-VACUITY (a gate builds its own inputs). The planted artifact divides
    by the partition's own ``count`` instead of the ``nsamples`` role — the
    failure verbatim — and the SAME comparison the rows above run must catch it.

    Whole-view the plant AGREES (one partition's count IS the total), which is
    what makes it the right plant: it is invisible to any check that only ever
    runs the whole view."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(built["fraction"], "fraction")
    directory = PL.with_sidecar(
        PL.count_reading_bundle(D.fraction, tmp_path / "planted",
                                cache_dir=cache_dir), sidecar)
    kw = {"x": D.plane(N, w=1)[0]}
    oracle = O.run(directory, "fraction", N, sidecar, **kw)
    plugin = L.host_plugin(directory, "fraction", sidecar)

    whole = eplan.plan(plugin, structure=eexec.HostTeam, npartitions=1).run(**kw)
    _assert_bit_identical(whole, oracle, "the plant, whole-view")

    split = eplan.plan(plugin, structure=eexec.HostTeam, npartitions=K).run(**kw)
    assert split.tobytes() != oracle.tobytes(), (
        "the planted count-reading body agreed with the oracle under "
        f"{K} partitions; this row cannot detect a body that is not "
        "partition-invariant, and every bit-identity assertion above is vacuous"
    )
