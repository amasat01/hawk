# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`hawk.runtime`'s n_samples inference votes ONLY per-sample roles, never a
`lookup`/`wide_in`/`wide_out`/`accum_out` plane's own runtime length.

**The finding.** `_n_from` used to return the trailing extent of the FIRST
plane present in `arg_spec` order, full stop -- correct for every fixture
whose `arg_spec` leads with a `mutable`/`out` sink (the canonical role
order), silently wrong for a kernel whose FIRST bound plane is a
`lookup`/`wide_in`/`wide_out`/`accum_out` plane, because those four roles'
own length is a runtime quantity UNRELATED to the sample count. This was hit
for real by a KAN fold's reverse pass -- whose `arg_spec` leads with a
SCATTERED `accum_out` plane -- reading that plane's length as n_samples,
over-running the correctly-sized cotangent plane the launch then read, and
returning `6.94e-310` (a pointer's bytes, read back as a double) in the first
two output slots, with no error anywhere.

**The exact real shape, reproduced here without a KAN layer.** `_deployable.gather`'s own VJP (`hawk.diff.vjp(D.gather, wrt=("table"))`) is the SAME
transpose-of-a-gather-is-a-scatter shape: its `arg_spec` is
`[(accum_out, "bar_table"), (per_sample, "bar_y"), (per_sample, "where")]` --
`bar_table` (the scatter target, sized to the ORIGINAL table's row count) is
first, and its length has nothing to do with n_samples. This module builds
that exact kernel once (the `gather_vjp_unit` fixture) and the real
end-to-end row below is measured against it, not against a stand-in shaped
only vaguely like the bug.

**RED, MEASURED** (`git stash` the fix to `hawk/_bind_checks.py`, ran
`test_an_accum_led_kernel_is_silently_wrong_before_the_fix` below, `git stash
pop`): with `bar_table` 5 slots wide and `bar_y = [1..8]`/
`where = [0,1,2,0,1,2,3,4]` (8 samples each), the unfixed `_n_from` inferred
`n_samples=5` (`bar_table`'s own length, being FIRST in `arg_spec`) and the
run silently scattered samples 0..4 only: `bar_table` came back
`[5.0, 7.0, 3.0, 0.0, 0.0]` against the correct (all 8 samples)
`[5.0, 7.0, 9.0, 7.0, 8.0]` -- a wrong ANSWER, not a crash, because this row's
buffers are sized so the mis-inferred, too-SMALL launch stays fully in-bounds
(samples 5..7 of `bar_y`/`where` are simply never read) rather than reading
past one -- which is what makes it safe to run in this test process instead
of merely citing the original pointer-bytes number.

**Why the lookup-led and disagreement rows call `hawk.runtime._n_from`
directly, with a HAND-BUILT `arg_spec`.** The canonical role order
(`ROLE_ORDER` in `hawk/ir/nodes.py`) puts every sink
(`mutable`/`out`/`wide_out`/`accum_out`) BEFORE `lookup` -- no real
single-sink kernel's OWN `arg_spec` can ever put a `lookup` slot first, so
proving the role-based filter is POSITION-independent (not merely "the
second-ranked slot, which happens never to be a table in this corpus") needs
an input this suite's own frontend cannot produce. `_n_from` is a pure
function of `(arg_spec, arrays)`; testing it directly, with an `arg_spec`
this frontend would never emit, is the direct way to prove the filter reads
ROLES and not POSITIONS, rather than assuming it from an example that
happens never to exercise the branch.

**The fix**: `hawk/_bind_checks.py`'s `_n_from` and its `_PER_SAMPLE_ROLES`/
`_NEVER_VOTES` tables (see that function's docstring).
"""

from __future__ import annotations

import array

import _deployable as D
import numpy as np
import pytest

import hawk.runtime as R
from hawk import Kernel
from hawk.artifact import build_bundle
from hawk.diff import vjp
from hawk.ir import HawkError

#: The exact bug shape (this module's docstring): `gather`'s own reverse pass.
#: Built ONCE per test process (host-only -- no row here needs the device leg)
#: and shared by every row that runs it for real.
_WRT = ("table",)


def _mem(values, typecode="q"):
    return array.array(typecode, values)


#: A PLACEHOLDER unit digest (this file never resolves it — that is
#: ``test_artifact_derivative_cross_unit.py``'s subject). `gather`'s own VJP
#: is `cross_sample_write` where `gather` itself is `cross_sample_read`
#: (the transpose), so refuses building the two in one bundle — this
#: file only needs `gather_vjp` runnable ON ITS OWN, and this
#: cross-unit rule (`hawk/artifact/bundle.py`'s `_per_kernel_derivative`) is
#: what makes that possible at all: a `Derived` kernel published without its
#: primal alongside it must name SOME unit its primal lives in, real or not,
#: for THIS module's purpose.
_FAKE_PRIMAL_UNIT = "0" * 64


@pytest.fixture(scope="module")
def gather_vjp_unit(tmp_path_factory, cache_dir):
    """`gather`'s VJP, published host-only, STANDALONE (see
    `_FAKE_PRIMAL_UNIT`). A fresh :class:`Kernel` (cheap) so this fixture
    never shares a mutable IR object with anything else."""
    vjp_kernel = Kernel(
        "gather_vjp", vjp(D.gather, wrt=_WRT, primal_unit=_FAKE_PRIMAL_UNIT), {})
    directory = tmp_path_factory.mktemp("gather_vjp_unit")
    return build_bundle([vjp_kernel], directory, targets=("host",),
                        cache_dir=cache_dir)


# --------------------------------------------------------------------------- #
# Pure `_n_from` rows: role-based filtering, position-independent (see this
# module's docstring for why these are NOT run through a compiled kernel).
# --------------------------------------------------------------------------- #
def test_a_lookup_led_arg_spec_infers_from_the_per_sample_plane_not_the_table():
    """`lookup` first in `arg_spec` (impossible for a real kernel -- see this
    module's docstring). A 20-row table and an 8-sample `where`/`y` pair: the
    RIGHT answer is 8, read off the per-sample roles.

    Without a role check, the trailing extent of the first bound plane in
    `arg_spec` order would be used instead: the table's own row count, 20,
    not the sample count."""
    arg_spec = (("lookup", "table"), ("per_sample", "where"), ("mutable", "y"))
    arrays = {"table": _mem(range(20)), "where": _mem(range(8)),
             "y": _mem([0] * 8)}
    assert R._n_from(arg_spec, arrays) == 8


def test_an_accum_led_arg_spec_infers_from_the_per_sample_plane_not_the_scatter_target():
    """`accum_out` first -- the shape a real kernel CAN produce ( puts
    every sink first, and a kernel whose only sink is `accum_out` has nothing
    ranked before it -- exactly `gather`'s own VJP). The scatter target is 5
    slots wide; the per-sample planes are 8 samples: the right answer is 8.

    Without that, the scatter target's own length (5) would be used instead
    of the sample count -- `test_an_accum_led_kernel_is_silently_wrong_
    before_the_fix` below is this exact shape's real-kernel consequence."""
    arg_spec = (("accum_out", "bar_table"), ("per_sample", "bar_y"),
               ("per_sample", "where"))
    arrays = {"bar_table": _mem(range(5)), "bar_y": _mem(range(8)),
             "where": _mem(range(8))}
    assert R._n_from(arg_spec, arrays) == 8


def test_disagreeing_per_sample_planes_are_refused_naming_both():
    """Two GENUINE per-sample roles of different lengths is a real mismatch --
    resolved by refusing and naming both, never by silently picking one."""
    arg_spec = (("per_sample", "bar_y"), ("per_sample", "where"))
    arrays = {"bar_y": _mem(range(8)), "where": _mem(range(5))}
    with pytest.raises(HawkError,
                       match=r"(?=.*\('bar_y', 8\))(?=.*\('where', 5\))"):
        R._n_from(arg_spec, arrays)


def test_a_kernel_with_no_per_sample_plane_requires_explicit_n_samples():
    """Every declared role is `accum_out`/`lookup`/`uniform` -- nothing
    `_PER_SAMPLE_ROLES` allows a vote from -- so this must refuse rather than
    guess, naming `n_samples` as the way out."""
    arg_spec = (("accum_out", "bar_table"), ("lookup", "table"), ("uniform", "k"))
    arrays = {"bar_table": _mem(range(5)), "table": _mem(range(20)), "k": 2.0}
    with pytest.raises(HawkError, match="n_samples"):
        R._n_from(arg_spec, arrays)


def test_explicit_n_samples_always_wins_over_the_bound_planes(gather_vjp_unit):
    """The positive control every refusal/inference row above is judged
    against: `hawk.runtime.run`'s `n_samples=` bypasses `_n_from` ENTIRELY, so
    it is honoured even where it asks for FEWER samples than the bound
    per-sample planes actually hold -- the caller who knows better is never
    second-guessed by a vote it never asked for."""
    from conftest import sidecar_of

    kernel = R.load(gather_vjp_unit.directory, "gather_vjp",
                    sidecar_of(gather_vjp_unit, "gather_vjp"))
    bar_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])   # 8 samples
    where = np.array([0, 1, 2, 0, 1, 2, 3, 4], dtype=np.int64)
    bar_table = np.zeros(5)
    R.run(kernel, bar_y=bar_y, where=where, bar_table=bar_table, n_samples=3)
    want = np.zeros(5)
    np.add.at(want, where[:3], bar_y[:3])                        # only 3 scattered
    np.testing.assert_allclose(bar_table, want)
    full = np.zeros(5)
    np.add.at(full, where, bar_y)
    assert not np.allclose(bar_table, full), (
        "n_samples=3 must NOT quietly run the whole 8-sample range"
    )


# --------------------------------------------------------------------------- #
# The real, compiled, end-to-end row (this module's docstring's RED number).
# --------------------------------------------------------------------------- #
def test_an_accum_led_kernel_is_silently_wrong_before_the_fix(gather_vjp_unit):
    """`gather`'s own VJP, launched through `hawk.runtime.run` with NO
    `n_samples=` at all -- the exact call `HostCallable` makes when a
    caller does not state `lanes`. `bar_table` (5 slots, the original table's
    row count) is deliberately SHORTER than `bar_y`/`where` (8 samples each);
    `where` only ever addresses `bar_table`'s 5 slots so the run stays fully
    in-bounds under EITHER inference (this row measures a wrong ANSWER, not a
    crash -- see this module's docstring for the measured RED numbers and why
    that is the safe choice for a test process)."""
    from conftest import sidecar_of

    kernel = R.load(gather_vjp_unit.directory, "gather_vjp",
                    sidecar_of(gather_vjp_unit, "gather_vjp"))
    bar_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    where = np.array([0, 1, 2, 0, 1, 2, 3, 4], dtype=np.int64)
    bar_table = np.zeros(5)
    R.run(kernel, bar_y=bar_y, where=where, bar_table=bar_table)  # NO n_samples=
    want = np.zeros(5)
    np.add.at(want, where, bar_y)   # scatter-add over ALL 8 samples
    np.testing.assert_allclose(bar_table, want)
