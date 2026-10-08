# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The compensated-accum and mask-free-guard seams are LOAD-BEARING, not decorative.

Each seam gets the same two observations: the artifact it produces DIFFERS from
the default's, and the DEFAULT's answer is wrong for that kind. A seam whose
only evidence is that a flag exists is decorative — which is exactly what
a hand-patched, anchored-regex approach to generated C++ would have been.

* **The compensated-accum kind.** Same kernel, same declarations; the seam
  decides how the scatter COMMITS. The default (``plain``) is the exemplar's
  read-modify-write, and on a summation with catastrophic cancellation
  (``1e16, 1, 1, …, 1, -1e16`` all into one lane) it returns 0 — every ``1``
  absorbed. ``compensated`` carries the lost bits into the companion plane the
  declaration names, so ``acc + acc_c`` is exact.
* **The mask-free guard kind.** Same kernel again; the seam decides whether
  a mask gates the commit at all. The default gates on the bound ``terminated``
  mask, so a terminated sample keeps its prior value — the right answer for a
  propagator and the WRONG one for a data-only kind that must write every
  sample (``Guard(mask=None)``).

The host arm is run under ONE tile (asserted), because both fixtures write a
shared lane / read a shared mask and a tiled run would be measuring eagle's
schedule rather than the seam.

This test previously failed via four plants, one per observation:

* ``compensated(into)`` degraded to ``SinkPolicy("plain", into)`` --
  ``AssertionError: the seam changed no emitted text -- it is decorative``;
* the Neumaier correction replaced by ``0.0 * hawk_prev0`` (artifact still
  different, arithmetic gutted) -- ``AssertionError: the compensated commit must
  recover the exact sum 14.0; got 0.0 + 0.0``;
* ``_Renderer._guard_slot``'s mask forced to ``None`` --
  ``AssertionError: the seam changed no emitted text -- it is decorative``;
* ``DATA_ONLY = Guard("terminated")`` (the mask-free value made masked) --
  ``Arrays are not equal / Mismatched elements: 8 / 16 (50%) / [8]: 0.0
  (ACTUAL), 9.0 (DESIRED)``.

Every plant was then removed.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Mutable as _Mutable
from hawk import Param as _Param
from hawk import Scalar as _Scalar
from hawk import Terminated as _Terminated
from hawk import Vector as _Vector
from hawk import kernel as _hkernel
from hawk.artifact import build_bundle as _build_bundle
from hawk.ext import Kind as _Kind
from hawk.ext import Output as _Output
from hawk.ext import compensated as _compensated
from hawk.math import norm as _norm

N = 16


def _run_host(bundle, kernel, **kw):
    import eagle.exec as eexec
    from eagle import plan as eplan

    assert eexec.HostTeam.tile_count(eexec.Partition.whole(N), 0) == 1, (
        "this row must run in ONE tile: its fixtures share a lane / a mask, so a "
        "tiled run would observe eagle's schedule, not the seam"
    )
    plugin = L.host_plugin(bundle.directory, kernel, sidecar_of(bundle, kernel))
    return eplan.plan(plugin, structure=eexec.HostTeam).run(**kw)


def _cancelling():
    """A sum whose exact value is ``N - 2`` and whose naive value is 0."""
    x = np.ones(N)
    x[0], x[-1] = 1e16, -1e16
    return x, np.zeros(N, dtype=np.int64)


# --------------------------------------------------------------------------- #
# -- the compensated-accum kind.
# --------------------------------------------------------------------------- #
def test_sink_policy_produces_a_different_artifact_from_the_default(built):
    """Re-owned: the compensated commit used to hoist a Neumaier,
    abs-branching correction (``hawk_sum0``, ``aether::math::abs``) into the
    KERNEL BODY, ending in a plain, non-atomic store -- a race under any
    parallel launch, exactly the one ``hawk_abi::accum_add`` already closed
    for ``plain``/``atomic``. It is now ONE call to
    ``hawk_abi::accum_add_compensated`` -- a target-atomic Knuth two-sum,
    branch-free, defined once in the PRELUDE (``twoSumErr_``) rather than
    hoisted per call site -- so the markers this row checks for moved from
    the body to the prelude, and the algorithm word changed with them."""
    plain = (built["scatter_c_plain"].directory / "scatter_c.cpp").read_text()
    comp = (built["scatter_c_comp"].directory / "scatter_c.cpp").read_text()
    assert plain != comp, "the sink policy changed no emitted text -- it is decorative"
    assert "hawk_abi::accum_add(" in plain and "+=" not in plain
    assert "accum_add_compensated" not in plain and "twoSumErr_" not in plain, (
        "a PLAIN unit's prelude must stay byte-identical: it must carry "
        "neither the compensated helper nor its two-sum residual")
    assert "hawk_abi::accum_add_compensated(" in comp, (
        "the compensated commit must be ONE call to the target-atomic helper:\n"
        + comp)
    assert "twoSumErr_" in comp and "atomicAdd" in comp, (
        "the compensated commit must carry the Knuth two-sum residual, target-"
        "atomic on both lanes:\n" + comp)


def test_sink_policy_the_default_answer_is_wrong_for_the_compensated_kind(built):
    x, lane = _cancelling()
    exact = float(N - 2)

    result = _run_host(built["scatter_c_plain"], "scatter_c", x=x, lane=lane)
    acc, acc_c = result["acc"], result["acc_c"]
    assert acc[0] == 0.0 and acc_c[0] == 0.0, (
        "the DEFAULT commit must lose the small terms for this kind -- if it did "
        f"not, the seam would have nothing to fix (got {acc[0]}, {acc_c[0]})"
    )
    assert acc[0] != exact

    result = _run_host(built["scatter_c_comp"], "scatter_c", x=x, lane=lane)
    acc, acc_c = result["acc"], result["acc_c"]
    assert acc[0] + acc_c[0] == exact, (
        f"the compensated commit must recover the exact sum {exact}; got "
        f"{acc[0]} + {acc_c[0]}"
    )


def test_sink_policy_names_a_slot_it_does_not_build():
    """A seam value is never silently absent from the vocabulary: ``atomic``
    and ``banded`` name who builds them and who asks for them."""
    import pytest

    from hawk.ext import sink_policy
    from hawk.ir import HawkError

    for name in ("banded",):
        with pytest.raises(HawkError) as excinfo:
            sink_policy(name)
        assert "not built" in str(excinfo.value) and "reserved sink policy" in str(excinfo.value)


def test_sink_policy_atomic_is_built_and_names_the_commit_plain_already_makes():
    """``atomic`` is a BUILT value. A scattered accumulate commits
    through the TARGET's own atomic add under ``plain`` as well: a scattered
    read-modify-write has no correct use under any parallel launch (eagle's host
    team is OpenMP over tiles; the device is the device), so the seam offers no
    such commit, and ``atomic`` spells the one it makes out loud."""
    from hawk.emit import render_body
    from hawk.ext import PLAIN, Kind, sink_policy

    atomic = sink_policy("atomic")
    assert atomic.id == "atomic"
    plain = render_body(D.scatter.sinks, D.scatter.walk, kind=Kind("p", sink=PLAIN)).text
    loud = render_body(D.scatter.sinks, D.scatter.walk, kind=Kind("a", sink=atomic)).text
    assert plain == loud
    assert "hawk_abi::accum_add(" in plain and "+=" not in plain


@pytest.mark.gpu
def test_sink_policy_a_multi_writer_scatter_commits_every_term_on_the_device(built):
    """A plain ``+=`` on an ``aether::View`` loses updates under concurrency.
    65536 samples scatter ``1.0`` onto 8 lanes, so every
    lane's exact total is 8192 and a lost update shows as an integer short."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, lanes = 1 << 16, 8
    x = np.ones(n)
    lane = (np.arange(n) % lanes).astype(np.int64)
    bundle = built["scatter"]
    plugin = L.device_plugin(bundle.directory, "scatter", sidecar_of(bundle, "scatter"))
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, lane=lane)
    acc = got[0] if isinstance(got, tuple) else got
    acc = np.asarray(acc.get() if hasattr(acc, "get") else acc)
    np.testing.assert_array_equal(acc[:lanes], np.full(lanes, float(n // lanes)))


@pytest.mark.gpu
def test_sink_policy_a_multi_writer_scatter_commits_every_term_on_the_device_compensated(built):
    """The compensated device row, mirroring the plain/atomic one above: the
    ``compensated`` branch of ``_commit_scattered`` must not end in a plain,
    non-atomic read-modify-write (a const-aliased read of the target, a
    Neumaier correction, a plain store) — the exact race
    ``hawk_abi::accum_add`` closes for ``plain``/``atomic``. Same shape as
    the row above: 65536 samples
    scatter ``1.0`` onto 8 lanes through the COMPENSATED kind (``scatter_c``,
    ``acc``/``acc_c``), so a lane that lost no updates recovers ``acc + acc_c
    == 8192`` exactly -- a lost update under concurrency shows as an integer
    short, same as the plain row, and a target-atomic Knuth two-sum never
    loses a term at this magnitude (every term is ``1.0``, no cancellation)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, lanes = 1 << 16, 8
    x = np.ones(n)
    lane = (np.arange(n) % lanes).astype(np.int64)
    bundle = built["scatter_c_comp"]
    plugin = L.device_plugin(bundle.directory, "scatter_c",
                             sidecar_of(bundle, "scatter_c"))
    result = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, lane=lane)
    acc, acc_c = result["acc"], result["acc_c"]
    acc = np.asarray(acc.get() if hasattr(acc, "get") else acc)
    acc_c = np.asarray(acc_c.get() if hasattr(acc_c, "get") else acc_c)
    total = acc[:lanes] + acc_c[:lanes]
    np.testing.assert_array_equal(total, np.full(lanes, float(n // lanes)), (
        "a lost update under concurrency shows as an integer short of 8192 "
        f"per lane; got acc={acc[:lanes]} acc_c={acc_c[:lanes]} total={total}"))


# --------------------------------------------------------------------------- #
# -- the mask-free (data-only) guard kind.
# --------------------------------------------------------------------------- #
def test_guard_kind_produces_a_different_artifact_from_the_default(built):
    guarded = (built["diag_guarded"].directory / "diagnostic.cpp").read_text()
    free = (built["diag_free"].directory / "diagnostic.cpp").read_text()
    assert guarded != free, "the guard kind changed no emitted text -- it is decorative"
    assert "if (!trm_terminated[i].eval())" in guarded
    assert "if (!trm_terminated" not in free


def test_guard_kind_the_default_answer_is_wrong_for_the_mask_free_kind(built):
    x = np.arange(N, dtype=float)
    mask = np.zeros(N, dtype=bool)
    mask[N // 2:] = True
    want = x + 1.0

    guarded = _run_host(built["diag_guarded"], "diagnostic", x=x, terminated=mask)
    assert np.array_equal(guarded[:N // 2], want[:N // 2])
    assert np.array_equal(guarded[N // 2:], np.zeros(N - N // 2)), (
        "the DEFAULT guard must SKIP the masked samples, or the seam has nothing "
        f"to change: {guarded}"
    )

    free = _run_host(built["diag_free"], "diagnostic", x=x, terminated=mask)
    np.testing.assert_array_equal(free, want)


def test_guard_kind_binds_the_declared_mask_even_where_the_body_never_reads_it(built):
    """The mask is a slot of the kernel's own ``arg_spec`` because the kernel
    DECLARES it — the same "declared => bound, read or not" rule gives
    a compound quantity's unread wires. Without it the seam would have nothing
    to gate on."""
    import _deployable as D

    assert ("terminated", "terminated") in D.diagnostic.walk.slot_of
    assert ["terminated", "terminated"] in sidecar_of(built["diag_free"],
                                                      "diagnostic")["arg_spec"]


# --------------------------------------------------------------------------- #
# -- the multi-mask guard.
# --------------------------------------------------------------------------- #
def test_a_two_mask_guard_emits_one_condition_over_both_masks(built):
    one = (built["diag_one_mask"].directory / "diagnostic_two.cpp").read_text()
    two = (built["diag_two_masks"].directory / "diagnostic_two.cpp").read_text()
    assert "if (!trm_terminated[i].eval())" in one
    assert "trm_rejected[i].eval()" not in one, (
        "the default guard names only `terminated`; it must not gate on `rejected`")
    assert "if (!(trm_terminated[i].eval() || trm_rejected[i].eval()))" in two


def test_a_sample_flagged_by_either_mask_keeps_its_prior_value(built):
    x = np.arange(N, dtype=float)
    terminated = np.zeros(N, dtype=bool)
    rejected = np.zeros(N, dtype=bool)
    terminated[: N // 4] = True
    rejected[N // 4: N // 2] = True
    want = x + 1.0

    two = _run_host(built["diag_two_masks"], "diagnostic_two", x=x,
                    terminated=terminated, rejected=rejected)
    np.testing.assert_array_equal(two[: N // 2], np.zeros(N // 2))
    np.testing.assert_array_equal(two[N // 2:], want[N // 2:])

    one = _run_host(built["diag_one_mask"], "diagnostic_two", x=x,
                    terminated=terminated, rejected=rejected)
    np.testing.assert_array_equal(one[: N // 4], np.zeros(N // 4))
    np.testing.assert_array_equal(one[N // 4:], want[N // 4:])


@pytest.mark.parametrize("bad", [dict(masks="terminated"),
                                 dict(masks=("a", "a")),
                                 dict(masks=("a", "")),
                                 dict(mask="x", masks=("a", "b"))])
def test_a_malformed_guard_is_refused_by_name(bad):
    from hawk.ext import Guard, HawkError

    with pytest.raises(HawkError, match="Guard"):
        Guard(**bad)


def test_guard_names_reads_both_spellings():
    from hawk.ext import DATA_ONLY, DEFAULT_GUARD, Guard

    assert DEFAULT_GUARD.names == ("terminated",)
    assert DATA_ONLY.names == ()
    assert Guard(masks=("terminated", "rejected")).names == ("terminated", "rejected")


# --------------------------------------------------------------------------- #
# -- the own-column compensated sink. Same mechanism as the `compensated` sink policy,
# -- reshaped: a Mutable target (or an Accum target committed with no `at=`)
# -- accumulates into the host's PRIOR value instead of storing, and the
# -- companion is synthesised by the KIND rather than declared by the author.
# --------------------------------------------------------------------------- #
OWN_COLUMN_COMPENSATED = _Kind("own_column_compensated", sink=_compensated(into="c"))
OWN_COLUMN_COMPENSATED_RETURNED = _Kind(
    "own_column_compensated_returned",
    output=_Output.returned(_Scalar, slot="y"), sink=_compensated(into="c"))


@OWN_COLUMN_COMPENSATED
def own_column_compensated_solo(x: _Scalar, terminated: _Terminated, y: _Mutable[_Scalar]):
    """The own-column compensated target: an ordinary elementwise store the
    KIND turns into a Neumaier accumulate into ``y``'s prior value."""
    y = x


@_hkernel
def own_column_plain_twin(x: _Scalar, terminated: _Terminated, y: _Mutable[_Scalar]):
    """The same body under the DEFAULT (plain) kind: a store, not an
    accumulate — what a repeated launch into the same buffer naively loses."""
    y = x


@OWN_COLUMN_COMPENSATED_RETURNED
def own_column_compensated_returns(x: _Scalar):
    """The ``Output.returned`` sugar, paired with the own-column
    compensated sink: the body commits nothing itself, and the synthesised
    ``y`` slot is the compensated target."""
    return x


#: Eight terms whose EXACT sum is 6.0 and whose naive float64 running sum
#: is 0.0 -- 1e16 swallows every `+1.0` (ulp(1e16) >> 1), and the closing
#: `-1e16` then cancels the untouched-looking accumulator to zero.
_COMPENSATED_TERMS = [1e16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1e16]


def _two_sum_err(a, b, s):
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def _neumaier_reference(terms):
    """The SAME twoSumErr_ recurrence the emitted C++ runs, independently in
    Python -- the parity oracle for
    ``test_matches_the_neumaier_reference_bit_for_bit`` below."""
    s = c = 0.0
    for t in terms:
        old = s
        s = old + t
        c += _two_sum_err(old, t, s)
    return s, c


def _sequential_launch(bundle, name, terms, *, terminated=None,
                       companion=True):
    """Bind once, ``rebind``+``launch`` for every later term -- the SAME
    numpy ``y``/``c`` buffers persist across all K launches (``Plan.bind``'s
    own contract: the output planes are the caller's, never reallocated), so
    the own-column accumulate semantics can actually be observed. The TOTAL
    is ``y + c`` -- ``y`` alone is the target's own bytes, which cancellation
    can legitimately drive back through zero (see
    ``test_own_column_compensated_recovers_the_exact_sum``/
    ``test_matches_the_neumaier_reference_bit_for_bit`` below)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
    plan = eplan.plan(plugin, structure=eexec.HostTeam)
    y = np.zeros(N)
    c = np.zeros(N) if companion else None
    mask = terminated if terminated is not None else np.zeros(N, dtype=bool)
    kw = {"x": np.full(N, terms[0]), "y": y}
    if companion:
        kw["c"] = c
    if name != "own_column_compensated_returns":
        kw["terminated"] = mask
    bound = plan.bind(**kw)
    bound.launch()
    for t in terms[1:]:
        bound = bound.rebind(x=np.full(N, t))
        bound.launch()
    return y, c


def test_own_column_compensated_recovers_the_exact_sum(tmp_path, cache_dir):
    bundle = _build_bundle([own_column_compensated_solo], tmp_path / "exact_sum",
                           targets=("host",), cache_dir=cache_dir)
    y, c = _sequential_launch(bundle, "own_column_compensated_solo", _COMPENSATED_TERMS)
    np.testing.assert_array_equal(y + c, np.full(N, 6.0))

    # the PLAIN kind's own-column commit is (and stays) a STORE, never an
    # accumulate -- unlike the SCATTERED form, there is no "naive accumulate"
    # own-column policy to contrast against, so the naive answer here is
    # simply the LAST launch's term, not a lossy running sum.
    plain_bundle = _build_bundle([own_column_plain_twin], tmp_path / "exact_sum_plain",
                                 targets=("host",), cache_dir=cache_dir)
    y_plain, _ = _sequential_launch(plain_bundle, "own_column_plain_twin",
                                    _COMPENSATED_TERMS, companion=False)
    np.testing.assert_array_equal(y_plain, np.full(N, _COMPENSATED_TERMS[-1]), (
        "the PLAIN kind must simply STORE the last term (no accumulate at "
        "all) -- the seam is what turns the store into an accumulate"))


@OWN_COLUMN_COMPENSATED
def own_column_compensated_product(a: _Scalar, b: _Scalar, terminated: _Terminated,
                                   y: _Mutable[_Scalar]):
    """A compensated target whose term is a PRODUCT: the shape FMA
    contraction would fuse into the two-sum."""
    y = a * b


@_hkernel
def contraction_probe(a: _Scalar, b: _Scalar, z: _Mutable[_Scalar]):
    """``a*b - 1``: contracted to one FMA under the fast host profile."""
    z = a * b - 1.0


def test_compensation_stays_exact_under_fma_contraction(tmp_path, cache_dir):
    """The fast host profile contracts ``a*b + c`` into FMAs; the compensated
    commit's two-sum must still see the ROUNDED term.

    Known answer: a = b = 1 + 2^-30, so a*b = 1 + 2^-29 + 2^-60 exactly and
    its rounded value is p = 1 + 2^-29. Accumulated onto 1.0 the sum 1 + p is
    exact, so the companion must stay 0. Fused, the residual would see the
    2^-60 tail (confirmed by removing the AETHER_FP_BARRIER lines from
    hawk/emit/backend.py: c = 2^-60). The probe kernel shows this very build
    does contract (non-vacuity)."""
    import platform

    import eagle.exec as eexec
    from eagle import plan as eplan

    from hawk.compile import host_codegen_flags

    if platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("the fast host profile is x86-64 only")
    assert "-ffp-contract=fast" in host_codegen_flags("native-vector-math")
    bundle, probed = (
        _build_bundle([k], tmp_path / sub, targets=("host",),
                      cache_dir=cache_dir, host_profile="native-vector-math")
        for k, sub in ((own_column_compensated_product, "fma_comp"),
                       (contraction_probe, "fma_probe")))
    a = np.full(N, 1.0 + 2.0 ** -30)

    probe = eplan.plan(L.host_plugin(probed.directory, "contraction_probe",
                                     sidecar_of(probed, "contraction_probe")),
                       structure=eexec.HostTeam)
    z = np.zeros(N)
    probe.bind(a=a, b=a, z=z).launch()
    if z[0] != 2.0 ** -29 + 2.0 ** -60:
        pytest.skip(f"this host build does not contract (a*b - 1 = {z[0]!r}): "
                    "no FMA on this CPU, nothing to guard")

    name = "own_column_compensated_product"
    plan = eplan.plan(L.host_plugin(bundle.directory, name, sidecar_of(bundle, name)),
                      structure=eexec.HostTeam)
    y, c = np.zeros(N), np.zeros(N)
    bound = plan.bind(a=np.ones(N), b=np.ones(N), terminated=np.zeros(N, dtype=bool),
                      y=y, c=c)
    bound.launch()
    bound.rebind(a=a, b=a).launch()
    np.testing.assert_array_equal(y, np.full(N, 2.0 + 2.0 ** -29))
    np.testing.assert_array_equal(c, np.zeros(N), "the term was fused into the two-sum")


def test_store_compensated_appears_once_inside_the_guard(tmp_path, cache_dir):
    bundle = _build_bundle([own_column_compensated_solo], tmp_path / "guard_once",
                           targets=("cuda", "host"), cache_dir=cache_dir)
    for src in ("own_column_compensated_solo.cpp", "own_column_compensated_solo.cu"):
        text = (bundle.directory / src).read_text()
        assert text.count("hawk_abi::store_compensated(") == 1, text
        assert "twoSumErr_" in text
        guard_pos = text.index("if (!trm_terminated[i].eval())")
        commit_pos = text.index("hawk_abi::store_compensated(")
        assert guard_pos < commit_pos, "the commit must sit INSIDE the guard block"

    plain_bundle = _build_bundle([own_column_plain_twin], tmp_path / "guard_once_plain",
                                 targets=("cuda", "host"), cache_dir=cache_dir)
    for src in ("own_column_plain_twin.cpp", "own_column_plain_twin.cu"):
        text = (plain_bundle.directory / src).read_text()
        assert "store_compensated" not in text
        assert "twoSumErr_" not in text


def test_the_companion_is_an_ordinary_synthesised_mutable_slot(tmp_path,
                                                                cache_dir):
    bundle = _build_bundle([own_column_compensated_solo], tmp_path / "companion_slot",
                           targets=("host",), cache_dir=cache_dir)
    sc = sidecar_of(bundle, "own_column_compensated_solo")
    assert ["mutable", "c"] in sc["arg_spec"]
    comp = next(m for m in sc["mutables"] if m["name"] == "c")
    target = next(m for m in sc["mutables"] if m["name"] == "y")
    assert comp["dtype"] == target["dtype"] and comp["width"] == target["width"]


def test_a_terminated_sample_keeps_its_prior_target_and_companion_bytes(
        tmp_path, cache_dir):
    bundle = _build_bundle([own_column_compensated_solo], tmp_path / "terminated_bytes",
                           targets=("host",), cache_dir=cache_dir)
    mask = np.zeros(N, dtype=bool)
    mask[0] = True
    y, c = _sequential_launch(bundle, "own_column_compensated_solo", _COMPENSATED_TERMS,
                              terminated=mask)
    assert y[0] == 0.0 and c[0] == 0.0, (
        "a terminated sample must keep its prior target AND companion bytes "
        f"across every launch: got y={y[0]} c={c[0]}")
    assert (y + c)[1] == 6.0


def test_two_targets_under_one_into_refuses():
    from hawk.ir import HawkError

    K = _Kind("two_targets_one_into", sink=_compensated(into="c"))
    with pytest.raises(HawkError, match="exactly ONE"):
        @K
        def bad(x: _Scalar, y1: _Mutable[_Scalar], y2: _Mutable[_Scalar]):
            y1 = x
            y2 = x


def test_a_companion_name_colliding_with_a_parameter_refuses():
    from hawk.ir import HawkError

    K = _Kind("companion_collides_with_parameter", sink=_compensated(into="x"))
    with pytest.raises(HawkError, match="collides"):
        @K
        def bad(x: _Scalar, y: _Mutable[_Scalar]):
            y = x


def test_a_reduce_target_refuses():
    from hawk import Reduce as _Reduce
    from hawk.ir import HawkError

    K = _Kind("reduce_target_refused", sink=_compensated(into="c"),
             output=_Output.named("total"))
    with pytest.raises(HawkError, match="Reduce"):
        @K
        def bad(x: _Scalar, total: _Reduce("sum")):
            total.contribute(x)


def test_matches_the_neumaier_reference_bit_for_bit(tmp_path, cache_dir):
    bundle = _build_bundle([own_column_compensated_solo], tmp_path / "neumaier_parity",
                           targets=("host",), cache_dir=cache_dir)
    y, c = _sequential_launch(bundle, "own_column_compensated_solo", _COMPENSATED_TERMS)
    ref_s, ref_c = _neumaier_reference(_COMPENSATED_TERMS)
    np.testing.assert_array_equal(y, np.full(N, ref_s))
    np.testing.assert_array_equal(c, np.full(N, ref_c))


def test_returned_output_plus_compensated_accumulates_onto_a_prior_sum(
        tmp_path, cache_dir):
    bundle = _build_bundle([own_column_compensated_returns], tmp_path / "returned_prior_sum",
                           targets=("host",), cache_dir=cache_dir)
    text = (bundle.directory / "own_column_compensated_returns.cpp").read_text()
    assert "hawk_abi::store_compensated(" in text, (
        "the returned value must commit through the compensated sink, not a "
        "plain store:\n" + text)

    import eagle.exec as eexec
    from eagle import plan as eplan

    prior = 42.0
    plugin = L.host_plugin(bundle.directory, "own_column_compensated_returns",
                           sidecar_of(bundle, "own_column_compensated_returns"))
    plan = eplan.plan(plugin, structure=eexec.HostTeam)
    y, c = np.full(N, prior), np.zeros(N)
    bound = plan.bind(x=np.full(N, _COMPENSATED_TERMS[0]), y=y, c=c)
    bound.launch()
    for t in _COMPENSATED_TERMS[1:]:
        bound = bound.rebind(x=np.full(N, t))
        bound.launch()
    ref_s, ref_c = _neumaier_reference([prior, *_COMPENSATED_TERMS])
    np.testing.assert_array_equal(y, np.full(N, ref_s))
    np.testing.assert_array_equal(c, np.full(N, ref_c))
    assert ref_s + ref_c == prior + 6.0


# --------------------------------------------------------------------------- #
# -- the own-column compensated sink: RANK-1 (Vector[3]) coverage, the
# -- downstream consumer's real shape -- an acceleration kind whose
# -- Output is a Vector[3] committed through the compensated sink onto a
# -- host plane that ALREADY holds a prior sum. A Vector plane binds as
# -- ``(width, samples)`` (eagle's own convention).
# --------------------------------------------------------------------------- #
VECTOR_OWN_COLUMN_COMPENSATED = _Kind("vector_own_column_compensated",
                                      sink=_compensated(into="c"))
VECTOR_OWN_COLUMN_COMPENSATED_RETURNED = _Kind(
    "vector_own_column_compensated_returned",
    output=_Output.returned(_Vector[3], slot="y"), sink=_compensated(into="c"))


@VECTOR_OWN_COLUMN_COMPENSATED
def vector_own_column_compensated_solo(x: _Vector[3], terminated: _Terminated,
                                       y: _Mutable[_Vector[3]]):
    """The rank-1 own-column compensated target: the same seam as
    ``own_column_compensated_solo``, one ``Vector[3]`` Mutable instead of a
    Scalar."""
    y = x


@VECTOR_OWN_COLUMN_COMPENSATED_RETURNED
def vector_own_column_compensated_returns(x: _Vector[3]):
    """The ``Output.returned`` sugar over the same rank-1 target."""
    return x


#: Per-component cancelling terms: the SAME scalar sequence whose exact sum
#: is 6.0 and whose naive running sum is 0.0, scaled by (1, 2, 3) so the
#: three components are genuinely distinct while keeping the same
#: catastrophic-cancellation shape (component ``k``'s exact sum is
#: ``6 * scale[k]``).
_VECTOR_SCALE = np.array([1.0, 2.0, 3.0])
_VECTOR_COMPENSATED_TERMS = [_VECTOR_SCALE * t for t in _COMPENSATED_TERMS]


def _vector_sequential_launch(bundle, name, terms, prior, *, mask_param=True):
    """The ``Vector[3]`` twin of ``_sequential_launch``: every sample carries
    the SAME per-component term/prior at each launch, so a mismatch between
    components would show up identically at every sample -- component
    ``k`` is where this row's "per component" claim is actually checked."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
    plan = eplan.plan(plugin, structure=eexec.HostTeam)
    y = np.tile(prior[:, None], (1, N)).astype(float)
    c = np.zeros((3, N))
    kw = {"x": np.tile(terms[0][:, None], (1, N)), "y": y, "c": c}
    if mask_param:
        kw["terminated"] = np.zeros(N, dtype=bool)
    bound = plan.bind(**kw)
    bound.launch()
    for t in terms[1:]:
        bound = bound.rebind(x=np.tile(t[:, None], (1, N)))
        bound.launch()
    return y, c


def test_vector_named_output_compensated_recovers_the_exact_sum_per_component(
        tmp_path, cache_dir):
    bundle = _build_bundle([vector_own_column_compensated_solo],
                           tmp_path / "vector_named", targets=("host",),
                           cache_dir=cache_dir)
    text = (bundle.directory / "vector_own_column_compensated_solo.cpp").read_text()
    assert text.count("hawk_abi::store_compensated(") == 1, (
        "ONE call on the whole Vector[3] view -- the per-component "
        "recurrence is the C++ helper's own loop, not a Python-side "
        "unroll:\n" + text)
    guard_pos = text.index("if (!trm_terminated[i].eval())")
    commit_pos = text.index("hawk_abi::store_compensated(")
    assert guard_pos < commit_pos, "the commit must sit INSIDE the guard block"

    prior = np.array([10.0, -5.0, 100.0])
    y, c = _vector_sequential_launch(bundle, "vector_own_column_compensated_solo",
                                     _VECTOR_COMPENSATED_TERMS, prior)
    for k in range(3):
        ref_s, ref_c = _neumaier_reference(
            [prior[k], *[t[k] for t in _VECTOR_COMPENSATED_TERMS]])
        assert y[k, 0] == ref_s and c[k, 0] == ref_c, (
            k, y[k, 0], c[k, 0], ref_s, ref_c)
        assert ref_s + ref_c == prior[k] + 6.0 * _VECTOR_SCALE[k]
    np.testing.assert_array_equal(y[:, 0], y[:, -1])
    np.testing.assert_array_equal(c[:, 0], c[:, -1])


def test_vector_returned_output_compensated_recovers_the_exact_sum_per_component(
        tmp_path, cache_dir):
    bundle = _build_bundle([vector_own_column_compensated_returns],
                           tmp_path / "vector_returned", targets=("host",),
                           cache_dir=cache_dir)
    text = (bundle.directory / "vector_own_column_compensated_returns.cpp").read_text()
    assert text.count("hawk_abi::store_compensated(") == 1, text

    prior = np.array([10.0, -5.0, 100.0])
    y, c = _vector_sequential_launch(bundle, "vector_own_column_compensated_returns",
                                     _VECTOR_COMPENSATED_TERMS, prior, mask_param=False)
    for k in range(3):
        ref_s, ref_c = _neumaier_reference(
            [prior[k], *[t[k] for t in _VECTOR_COMPENSATED_TERMS]])
        assert y[k, 0] == ref_s and c[k, 0] == ref_c, (
            k, y[k, 0], c[k, 0], ref_s, ref_c)
        assert ref_s + ref_c == prior[k] + 6.0 * _VECTOR_SCALE[k]


@VECTOR_OWN_COLUMN_COMPENSATED_RETURNED
def vector_own_column_compensated_returns_scaled(x: _Vector[3]):
    """The returned rank-1 target as a COMPOUND expression (a scalar times a
    vector), not a bare leaf: the commit must index the term per component
    whatever node the body's arithmetic ends in. ``1.0 * x`` keeps the terms
    exact, so the Neumaier reference is unchanged."""
    return 1.0 * x


@VECTOR_OWN_COLUMN_COMPENSATED_RETURNED
def vector_own_column_compensated_drag(x: _Vector[3], *, k: _Param):
    """A drag-shaped compound return, ``(-k |x|) x``: the acceleration style a
    compensated kind exists for."""
    return (-k * _norm(x)) * x


def test_vector_returned_compound_expression_compensated_recovers_the_exact_sum(
        tmp_path, cache_dir):
    bundle = _build_bundle([vector_own_column_compensated_returns_scaled],
                           tmp_path / "vector_scaled", targets=("host",),
                           cache_dir=cache_dir)
    prior = np.array([10.0, -5.0, 100.0])
    y, c = _vector_sequential_launch(bundle, "vector_own_column_compensated_returns_scaled",
                                     _VECTOR_COMPENSATED_TERMS, prior, mask_param=False)
    for k in range(3):
        ref_s, ref_c = _neumaier_reference(
            [prior[k], *[t[k] for t in _VECTOR_COMPENSATED_TERMS]])
        assert y[k, 0] == ref_s and c[k, 0] == ref_c, (
            k, y[k, 0], c[k, 0], ref_s, ref_c)


def test_a_drag_shaped_compensated_return_compiles_on_host(tmp_path, cache_dir):
    bundle = _build_bundle([vector_own_column_compensated_drag],
                           tmp_path / "vector_drag", targets=("host",),
                           cache_dir=cache_dir)
    text = (bundle.directory / "vector_own_column_compensated_drag.cpp").read_text()
    assert text.count("hawk_abi::store_compensated(") == 1, text


# --------------------------------------------------------------------------- #
# -- the own-column compensated sink: the DERIVATIVE row. ``d(target)/
# -- d(term) = 1``, so vjp/jvp are the PLAIN kernel's -- the companion is
# -- never mirrored as an ADJOINT plane.
# --------------------------------------------------------------------------- #
def test_a_derivative_of_an_own_column_compensated_kernel_commits_plain():
    """The reverse direction binds NO companion slot at all (``bar_c``/``c``
    is nowhere in its ``arg_spec``): the companion's value is a constant, so
    its adjoint contribution to ``bar_x`` is algebraically zero and no
    adjoint plane for it is ever created. The forward direction still
    declares a tangent output for the companion (forward-mode mirrors EVERY
    declared sink, constant-valued ones included), but that tangent commits
    a plain zero store, never the compensated arithmetic -- in both
    directions, the derived kernel never re-runs ``store_compensated``/
    ``accum_add_compensated`` at all."""
    from hawk import Kernel
    from hawk.diff import jvp, vjp
    from hawk.emit import render_body

    vjp_kernel = Kernel("own_column_compensated_solo_vjp",
                        vjp(own_column_compensated_solo, wrt=("x",)), {})
    names = [name for _role, name in vjp_kernel.arg_spec]
    assert "c" not in names, (
        f"the reverse-mode derivative must bind NO companion slot: {names}")
    vjp_text = render_body(vjp_kernel.sinks, vjp_kernel.walk).text
    assert "store_compensated" not in vjp_text
    assert "accum_add_compensated" not in vjp_text

    jvp_kernel = Kernel("own_column_compensated_solo_jvp",
                        jvp(own_column_compensated_solo, wrt=("x",)), {})
    jvp_text = render_body(jvp_kernel.sinks, jvp_kernel.walk).text
    assert "store_compensated" not in jvp_text, (
        "a derivative must never re-run the compensated commit -- "
        "d(target)/d(term) = 1 makes every derived commit a plain store:\n"
        + jvp_text)
    assert "accum_add_compensated" not in jvp_text


def test_a_unit_using_no_compensated_helper_renders_the_plain_prelude():
    """The compensated helpers are spliced only where used, and a unit using none
    must render the prelude exactly as a plain unit always has: one blank line
    before the namespace closes, never one per unused helper. Downstream render
    goldens pin this text byte for byte."""
    from hawk.emit.backend import abi_helpers

    for qualifier in ("__host__ __device__", "inline"):
        plain = abi_helpers(qualifier)
        assert plain.endswith("}\n\n}  // namespace hawk_abi\n"), plain[-80:]
        assert "store_compensated" not in plain and "twoSumErr_" not in plain
