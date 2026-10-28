# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.runtime`` validates every bound argument against the kernel's OWN
declaration before any address is taken (``hawk/_bind_checks.py``'s
``_check_bind``, wired into ``HostKernel.bind_all`` and hence ``run``).

Two REAL defects motivate this row, each one a wrong binding that used to
run silently and return garbage or segfault (an out-of-bounds read) rather
than refuse:

1. an Index plane (``Table[...].at()``'s index, a scatter ``lane``) bound as a
   type other than ``int32`` — an ``Index`` is a 4-byte plane, widened where the body reads
   it; an ``int64`` column is converted when it fits and refused by name when it does not;
2. a ``Terminated`` mask bound as ``int32`` — it must be ``bool``.

Every row below runs the real, compiled, HAWK-emitted artifact (the
``built`` fixture, shared with the rest of the suite) through
``hawk.runtime`` directly — no subprocess: a BAD binding must be refused
before the C call is ever reached, which only a same-process assertion can
show."""

from __future__ import annotations

import _deployable as D
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import runtime as rt
from hawk.ir import HawkError

N = 8


def _gather(built):
    bundle = built["gather"]
    return O.load(bundle.directory, "gather", sidecar_of(bundle, "gather"))


def _energy(built):
    bundle = built["energy"]
    return O.load(bundle.directory, "energy", sidecar_of(bundle, "energy"))


def _scatter(built):
    bundle = built["scatter"]
    return O.load(bundle.directory, "scatter", sidecar_of(bundle, "scatter"))


def _diagnostic(built):
    bundle = built["diag_guarded"]
    return O.load(bundle.directory, "diagnostic", sidecar_of(bundle, "diagnostic"))


# --------------------------------------------------------------------------- #
# 1. the Index plane: Table[...].at()'s index, and a scatter `lane`.
# --------------------------------------------------------------------------- #
def test_table_at_index_bound_as_float_is_refused(built):
    """``gather``'s ``where`` (the ``Table.at()`` index) must be an integer
    column; a float one is refused naming the kernel, the argument, the expected
    and the actual dtype, and the fix — never reaches ``hawk._core``."""
    kernel = _gather(built)
    table = np.arange(N, dtype=float) * 3.0
    where = ((np.arange(N) * 7 + 3) % N).astype(np.float32)
    y = np.zeros(N)
    with pytest.raises(HawkError, match=r"'where'.*int32.*float32"):
        rt.run(kernel, table=table, where=where, y=y, n_samples=N)


def test_table_at_index_bound_as_int32_matches_the_reference(built):
    """An int32 index column binds as it is: the answer is the numpy reference."""
    kernel = _gather(built)
    table = np.arange(N, dtype=float) * 3.0
    where = ((np.arange(N) * 7 + 3) % N).astype(np.int32)
    y = np.zeros(N)
    rt.run(kernel, table=table, where=where, y=y, n_samples=N)
    np.testing.assert_allclose(y, D.ref_gather(table, where))


def test_table_at_index_bound_as_int64_is_converted_when_it_fits(built):
    kernel = _gather(built)
    table = np.arange(N, dtype=float) * 3.0
    where = ((np.arange(N) * 7 + 3) % N).astype(np.int64)
    y = np.zeros(N)
    rt.run(kernel, table=table, where=where, y=y, n_samples=N)
    np.testing.assert_allclose(y, D.ref_gather(table, where))
    where[2] = -(2**31) - 1
    with pytest.raises(HawkError, match=r"'where': values exceed int32"):
        rt.run(kernel, table=table, where=where, y=y, n_samples=N)


def test_scatter_lane_bound_as_int32_matches_the_reference(built):
    """``scatter``'s ``lane`` (the ``hawk.scatter(acc)[lane]`` index) takes an
    int32 column — the same Index rule, a different kernel."""
    kernel = _scatter(built)
    x = np.arange(N, dtype=float) + 1.0
    lane = ((np.arange(N) * 7 + 3) % N).astype(np.int32)
    acc = np.zeros(N)
    rt.run(kernel, x=x, lane=lane, acc=acc, n_samples=N)
    np.testing.assert_allclose(acc, D.ref_scatter(x, lane, N))


def test_scatter_lane_bound_as_int64_is_converted_when_it_fits(built):
    kernel = _scatter(built)
    x = np.arange(N, dtype=float) + 1.0
    lane = ((np.arange(N) * 7 + 3) % N).astype(np.int64)
    acc = np.zeros(N)
    rt.run(kernel, x=x, lane=lane, acc=acc, n_samples=N)
    np.testing.assert_allclose(acc, D.ref_scatter(x, lane, N))


# --------------------------------------------------------------------------- #
# 2. the Terminated mask: must be bool, never an int mask.
# --------------------------------------------------------------------------- #
def test_terminated_mask_bound_as_int32_is_refused(built):
    """``diagnostic``'s ``terminated`` must be ``bool``; an int32 mask is
    refused naming the dtype mismatch."""
    kernel = _diagnostic(built)
    x = np.arange(N, dtype=float)
    terminated = np.zeros(N, dtype=np.int32)
    y = np.zeros(N)
    with pytest.raises(HawkError, match=r"'terminated'.*bool.*int32"):
        rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)


def test_terminated_mask_bound_as_bool_passes(built):
    """Bound bool, ``diagnostic`` writes every sample (the default guard with
    no terminated sample keeps every one live)."""
    kernel = _diagnostic(built)
    x = np.arange(N, dtype=float)
    terminated = np.zeros(N, dtype=bool)
    y = np.zeros(N)
    rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)
    np.testing.assert_allclose(y, x + 1.0)


# --------------------------------------------------------------------------- #
# 3. a sum over samples: one row per 32 samples, plus its compensation plane.
# --------------------------------------------------------------------------- #
def test_a_sum_over_samples_binds_one_row_per_32_samples(built):
    """``energy``'s ``total`` holds ``ceil(n/32)`` rows and ``total_comp``
    its compensation; their sum is the reference total."""
    kernel = _energy(built)
    v = np.arange(3 * N, dtype=float).reshape(3, N) * 0.25 + 1.0
    rows = -(-N // 32)
    total, comp = np.zeros(rows), np.zeros(rows)
    rt.run(kernel, v=v, total=total, total_comp=comp, n_samples=N)
    np.testing.assert_allclose(total.sum() + comp.sum(), D.ref_energy(v).sum())


def test_a_scattered_accum_out_plane_is_not_shape_checked(built):
    """A by-key store target (``scatter``'s ``acc``) is NOT forced to
    ``n_samples``. ``scatter``'s fixture happens to use ``N`` lanes too, so
    this row sizes ``acc`` to HALF that on purpose, to prove the validator
    does not assume an own-column length for every ``accum_out`` plane."""
    kernel = _scatter(built)
    x = np.arange(N, dtype=float) + 1.0
    lane = (np.arange(N) % (N // 2)).astype(np.int32)
    acc = np.zeros(N // 2)
    rt.run(kernel, x=x, lane=lane, acc=acc, n_samples=N)
    np.testing.assert_allclose(acc, D.ref_scatter(x, lane, N // 2))


# --------------------------------------------------------------------------- #
# Structural checks: contiguity, writability, multi-argument reporting.
# --------------------------------------------------------------------------- #
def test_a_non_contiguous_array_is_refused(built):
    """A strided (non-C-contiguous) plane is refused before any address is
    taken."""
    kernel = _diagnostic(built)
    x = np.arange(2 * N, dtype=float)[::2]  # a real stride, not a copy
    terminated = np.zeros(N, dtype=bool)
    y = np.zeros(N)
    with pytest.raises(HawkError, match=r"'x'.*[Cc]ontiguous"):
        rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)


def test_a_read_only_array_for_a_writable_role_is_refused(built):
    """``diagnostic``'s ``y`` is a ``mutable`` (written) role; a read-only
    array is refused rather than silently dropping the kernel's write."""
    kernel = _diagnostic(built)
    x = np.arange(N, dtype=float)
    terminated = np.zeros(N, dtype=bool)
    y = np.zeros(N)
    y.flags.writeable = False
    with pytest.raises(HawkError, match=r"'y'.*writable"):
        rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)


def test_a_read_only_array_for_an_input_role_is_still_refused(built):
    """A read-only INPUT (``x``, a ``per_sample`` leaf) is not flagged by the
    new writable-role check (it is not a sink role) — but it is still
    refused, by the pre-existing ``hawk.runtime.buffer_address`` rule that
    every bound plane (input or output) must be writable, because HAWK binds
    by ADDRESS, never by copy."""
    kernel = _diagnostic(built)
    x = np.arange(N, dtype=float)
    x.flags.writeable = False
    terminated = np.zeros(N, dtype=bool)
    y = np.zeros(N)
    with pytest.raises(HawkError, match=r"read-only"):
        rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)


def test_multiple_bad_arguments_are_reported_together(built):
    """Two independent defects in ONE call are both named in ONE refusal,
    not just the first one found."""
    kernel = _diagnostic(built)
    x = np.arange(N, dtype=float)
    terminated = np.zeros(N, dtype=np.int32)  # defect 1: wrong dtype
    y = np.zeros(N)
    y.flags.writeable = False  # defect 2: read-only mutable
    with pytest.raises(HawkError, match=r"(?s)(?=.*'terminated')(?=.*'y')"):
        rt.run(kernel, x=x, terminated=terminated, y=y, n_samples=N)


def test_an_explicitly_shorter_n_samples_over_a_longer_plane_still_passes(built):
    """Non-vacuity for the minimum-length shape rule: `run`'s own contract
    lets an explicit ``n_samples=`` ask for FEWER samples than the bound
    per-sample planes hold (``test_runtime_n_inference.py``'s own row for
    this) — the validator must not turn that into a false refusal."""
    kernel = _diagnostic(built)
    x = np.arange(2 * N, dtype=float)
    terminated = np.zeros(2 * N, dtype=bool)
    y = np.zeros(2 * N)
    rt.run(kernel, x=x, terminated=terminated, y=y, base=0, count=N, n_samples=N)
    np.testing.assert_allclose(y[:N], x[:N] + 1.0)
