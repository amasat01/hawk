# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A ``KernelKind`` subclass (or :meth:`Kind.extend`, its instance-form twin)
INHERITS its base's vocabulary entries and seam fields, merging rather than
shadowing them, so a vocabulary is extended the ordinary Python way --
subclass it -- and the extended kind works end to end like any other.

Coverage, one row per observation: merge order (most-base first), a seam
field left unset by a subclass keeps the nearest base's value rather than
reverting to ``Kind``'s own bare default, an equal re-declaration of an
inherited entry is fine, a differing one refuses by name, a diamond
(multiple/shared bases) merges once rather than duplicating or falsely
conflicting, the existing guard-mask check sees the MERGED vocabulary (not
just a subclass's own), ``Kind.extend`` reaches the same ``Kind`` the class
form does, and an END TO END row: a kernel on an inherited kind runs on host,
runs on device, and its ``hawk.diff.vjp``/``hawk.diff.jvp`` match a finite
difference -- the inherited vocabulary entry (``scale``) and the kind's own
(``bias``) bind and differentiate exactly as a single-level vocabulary's
would.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

from hawk import Mutable, Scalar, Terminated, Vector
from hawk.artifact import build_bundle as _build_bundle
from hawk.diff import jvp, vjp
from hawk.ext import DATA_ONLY, DEFAULT_GUARD, Guard, Kind, KernelKind
from hawk.ir import HawkError

H = 1e-5


# --------------------------------------------------------------------------- #
# -- merge order: most-base first, every base's vocabulary entries present.
# --------------------------------------------------------------------------- #
class _Base(KernelKind, slug="inherit_base"):
    terminated: Terminated
    scale: Scalar


class _Mid(_Base, slug="inherit_mid"):
    bias: Scalar


class _Leaf(_Mid, slug="inherit_leaf"):
    extra: Scalar


def test_a_subclass_inherits_every_base_vocabulary_entry_most_base_first():
    assert [n for n, _ in _Base.kind.vocabulary] == ["terminated", "scale"]
    assert [n for n, _ in _Mid.kind.vocabulary] == ["terminated", "scale", "bias"]
    assert [n for n, _ in _Leaf.kind.vocabulary] == [
        "terminated", "scale", "bias", "extra"]
    # a base's OWN kind is never mutated by a later subclass's merge.
    assert [n for n, _ in _Base.kind.vocabulary] == ["terminated", "scale"]


# --------------------------------------------------------------------------- #
# -- seam fields: a subclass may override; one that does not keeps the
# -- nearest base's value, never Kind's own bare default.
# --------------------------------------------------------------------------- #
class _GuardBase(KernelKind, slug="inherit_guard_base"):
    terminated: Terminated
    rejected: Terminated
    guard = Guard(masks=("terminated", "rejected"))


class _GuardMid(_GuardBase, slug="inherit_guard_mid"):
    scale: Scalar
    # no guard assignment here -- must keep _GuardBase's two-mask guard.


class _GuardLeaf(_GuardMid, slug="inherit_guard_leaf"):
    bias: Scalar
    guard = DEFAULT_GUARD  # overrides back down to the single "terminated" mask.


def test_a_seam_field_a_subclass_does_not_set_keeps_the_nearest_bases_value():
    assert _GuardBase.kind.guard == Guard(masks=("terminated", "rejected"))
    assert _GuardMid.kind.guard == Guard(masks=("terminated", "rejected")), (
        "a field no subclass's own body sets must inherit the nearest base's "
        "value, not silently revert to Kind's own bare default")
    assert _GuardLeaf.kind.guard == DEFAULT_GUARD


# --------------------------------------------------------------------------- #
# -- re-declaring an inherited entry: equal is fine, different refuses by name.
# --------------------------------------------------------------------------- #
def test_redeclaring_an_inherited_entry_with_an_equal_declaration_is_fine():
    class _Same(_Base, slug="inherit_redeclare_equal"):
        terminated: Terminated  # identical to _Base's own declaration
        extra: Scalar

    names = [n for n, _ in _Same.kind.vocabulary]
    assert names.count("terminated") == 1
    assert dict(_Same.kind.vocabulary)["terminated"] == dict(_Base.kind.vocabulary)["terminated"]


def test_redeclaring_an_inherited_entry_with_a_different_declaration_refuses_by_name():
    with pytest.raises(HawkError, match="scale") as excinfo:
        class _Conflict(_Base, slug="inherit_redeclare_conflict"):
            scale: Vector[3]  # _Base declares `scale: Scalar` -- different shape

    message = str(excinfo.value)
    assert "_Base" in message and "_Conflict" in message, (
        "the refusal must name the base and the conflicting subclass: " + message)


# --------------------------------------------------------------------------- #
# -- diamond / multiple bases: merged once, never duplicated or falsely
# -- conflicting when the SAME ancestor contributes the shared entry twice.
# --------------------------------------------------------------------------- #
class _DiamondA(KernelKind, slug="inherit_diamond_a"):
    a_only: Scalar
    guard = DATA_ONLY  # no `terminated` entry here -- the default guard needs one.


class _DiamondB(KernelKind, slug="inherit_diamond_b"):
    b_only: Scalar
    guard = DATA_ONLY


class _DiamondChild(_DiamondA, _DiamondB, slug="inherit_diamond_child"):
    own: Scalar


def test_multiple_independent_bases_merge_every_vocabulary_entry():
    assert {n for n, _ in _DiamondChild.kind.vocabulary} == {"a_only", "b_only", "own"}


class _DiamondGrandBase(KernelKind, slug="inherit_diamond_grand"):
    terminated: Terminated


class _DiamondLeft(_DiamondGrandBase, slug="inherit_diamond_left"):
    left: Scalar


class _DiamondRight(_DiamondGrandBase, slug="inherit_diamond_right"):
    right: Scalar


class _DiamondJoin(_DiamondLeft, _DiamondRight, slug="inherit_diamond_join"):
    pass


def test_a_shared_grandparent_entry_is_merged_once_not_duplicated_or_conflicting():
    names = [n for n, _ in _DiamondJoin.kind.vocabulary]
    assert names.count("terminated") == 1
    assert set(names) == {"terminated", "left", "right"}


class _ConflictA(KernelKind, slug="inherit_conflict_a"):
    shared: Scalar
    guard = DATA_ONLY


class _ConflictB(KernelKind, slug="inherit_conflict_b"):
    shared: Vector[3]
    guard = DATA_ONLY


def test_two_bases_contributing_a_different_declaration_for_the_same_entry_refuses():
    with pytest.raises(HawkError, match="shared"):
        class _ConflictJoin(_ConflictA, _ConflictB, slug="inherit_conflict_join"):
            pass


# --------------------------------------------------------------------------- #
# -- the existing guard-mask check (`Kind.__post_init__`) must see the
# -- MERGED vocabulary, not just a subclass's own class body.
# --------------------------------------------------------------------------- #
class _GuardMergeBase(KernelKind, slug="inherit_guard_merge_base"):
    rejected: Terminated
    guard = Guard("rejected")


class _GuardMergeChild(_GuardMergeBase, slug="inherit_guard_merge_child"):
    """Declares no mask of its own at all -- `guard`'s `rejected` mask is only
    ever a MERGED-in entry. If the guard check ran against this class's own
    (unmerged) vocabulary, this class statement itself would refuse."""

    extra: Scalar


def test_the_guard_mask_check_runs_against_the_merged_vocabulary():
    assert dict(_GuardMergeChild.kind.vocabulary)["rejected"].form == "terminated"
    assert _GuardMergeChild.kind.guard.names == ("rejected",)


# --------------------------------------------------------------------------- #
# -- Kind.extend: the instance-form twin, same merge + refusal rules, same
# -- resulting Kind for the same inputs as the class form.
# --------------------------------------------------------------------------- #
def test_kind_extend_matches_the_class_form():
    base_kind = Kind("inherit_extend_base",
                     vocabulary={"terminated": Terminated, "scale": Scalar},
                     guard=DATA_ONLY)
    extended = base_kind.extend("inherit_extend_leaf", vocabulary={"bias": Scalar})

    class _ExtendTwinBase(KernelKind, slug="inherit_extend_base"):
        terminated: Terminated
        scale: Scalar
        guard = DATA_ONLY

    class _ExtendTwinLeaf(_ExtendTwinBase, slug="inherit_extend_leaf"):
        bias: Scalar

    assert extended == _ExtendTwinLeaf.kind


def test_kind_extend_allows_an_equal_redeclaration_and_refuses_a_conflicting_one():
    base_kind = Kind("inherit_extend_redeclare_base", vocabulary={"scale": Scalar},
                     guard=DATA_ONLY)

    same = base_kind.extend("inherit_extend_redeclare_same",
                            vocabulary={"scale": Scalar})
    assert dict(same.vocabulary)["scale"] == dict(base_kind.vocabulary)["scale"]

    with pytest.raises(HawkError, match="scale"):
        base_kind.extend("inherit_extend_redeclare_conflict",
                         vocabulary={"scale": Vector[3]})


# --------------------------------------------------------------------------- #
# -- END TO END: a base kind + an inherited kind with one extra vocabulary
# -- entry, a kernel on the inherited kind, run on host, run on device, and
# -- differentiated -- checked against a finite difference.
# --------------------------------------------------------------------------- #
_E2E_N = 8


class _E2EBase(KernelKind, slug="inherit_e2e_base"):
    terminated: Terminated
    scale: Scalar


class _E2EDerived(_E2EBase, slug="inherit_e2e_derived"):
    """Adds ONE vocabulary entry (`bias`) on top of the inherited `terminated`
    and `scale` -- the exact shape the motivating real-world case names
    (`class Perturbed(Orbit, slug="perturbed")`)."""

    bias: Scalar


@_E2EDerived
def inherited_kernel(x: Scalar, terminated, scale, bias, y: Mutable[Scalar]):
    """`scale` is inherited from `_E2EBase`; `bias` is `_E2EDerived`'s own
    addition. Both bind exactly as a single-level vocabulary's would."""
    y = x * scale + bias


def _e2e_inputs():
    x = np.arange(_E2E_N, dtype=float) + 1.0
    scale = np.linspace(1.0, 2.0, _E2E_N)
    bias = np.linspace(-1.0, 1.0, _E2E_N)
    terminated = np.zeros(_E2E_N, dtype=bool)
    terminated[_E2E_N // 2:] = True
    return x, scale, bias, terminated


@pytest.fixture(scope="module")
def _e2e_bundle(tmp_path_factory, cache_dir):
    return _build_bundle([inherited_kernel],
                         tmp_path_factory.mktemp("inherit_e2e"),
                         targets=("host", "cuda"), cache_dir=cache_dir)


def test_an_inherited_kind_kernel_runs_end_to_end_on_host(_e2e_bundle):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(_e2e_bundle.directory, "inherited_kernel",
                           sidecar_of(_e2e_bundle, "inherited_kernel"))
    x, scale, bias, terminated = _e2e_inputs()
    y = np.asarray(eplan.plan(plugin, structure=eexec.HostTeam).run(
        x=x, terminated=terminated, scale=scale, bias=bias))
    want = x * scale + bias
    half = _E2E_N // 2
    np.testing.assert_array_equal(y[:half], want[:half])
    np.testing.assert_array_equal(y[half:], np.zeros(_E2E_N - half), (
        "a terminated sample must keep its prior (zero-initialised) value"))


@pytest.mark.gpu
def test_an_inherited_kind_kernel_runs_end_to_end_on_device(_e2e_bundle):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.device_plugin(_e2e_bundle.directory, "inherited_kernel",
                             sidecar_of(_e2e_bundle, "inherited_kernel"))
    x, scale, bias, terminated = _e2e_inputs()
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(
        x=x, terminated=terminated, scale=scale, bias=bias)
    y = np.asarray(got.get() if hasattr(got, "get") else got)
    want = x * scale + bias
    half = _E2E_N // 2
    np.testing.assert_array_equal(y[:half], want[:half])
    np.testing.assert_array_equal(y[half:], np.zeros(_E2E_N - half))


_E2E_ENV = {"x": 1.7, "terminated": False, "scale": 2.0, "bias": 0.5}


def test_vjp_of_an_inherited_kind_kernel_matches_finite_differences():
    derived = vjp(inherited_kernel, wrt=("x", "scale", "bias"))
    seed = 3.0
    got = evaluate(derived, {**_E2E_ENV, "bar_y": seed})

    for name in ("x", "scale", "bias"):
        h = H * max(1.0, abs(_E2E_ENV[name]))
        plus = evaluate(inherited_kernel.sinks,
                        {**_E2E_ENV, name: _E2E_ENV[name] + h})["y"]
        minus = evaluate(inherited_kernel.sinks,
                         {**_E2E_ENV, name: _E2E_ENV[name] - h})["y"]
        expect = (plus - minus) / (2 * h) * seed
        assert got[f"bar_{name}"] == pytest.approx(expect, rel=1e-6, abs=1e-7), (
            f"VJP of {name!r} disagrees with the central difference: "
            f"{got[f'bar_{name}']} vs {expect}"
        )


def test_jvp_of_an_inherited_kind_kernel_matches_finite_differences():
    derived = jvp(inherited_kernel, wrt=("x", "scale", "bias"))
    tangents = {"x": 0.4, "scale": -0.3, "bias": 1.1}
    got = evaluate(derived, {**_E2E_ENV,
                             **{f"dot_{k}": v for k, v in tangents.items()}})["dot_y"]

    plus_env = {**_E2E_ENV, **{k: _E2E_ENV[k] + H * v for k, v in tangents.items()}}
    minus_env = {**_E2E_ENV, **{k: _E2E_ENV[k] - H * v for k, v in tangents.items()}}
    plus = evaluate(inherited_kernel.sinks, plus_env)["y"]
    minus = evaluate(inherited_kernel.sinks, minus_env)["y"]
    expect = (plus - minus) / (2 * H)
    assert got == pytest.approx(expect, rel=1e-6, abs=1e-7), (
        f"JVP disagrees with the central difference: {got} vs {expect}"
    )
