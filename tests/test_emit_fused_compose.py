# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Fused-lane MEMBERSHIP and the merged ``arg_spec``.

The body path (``tests/test_emit_fused_lanes.py``) ships one body string per
group, each lane through the same renderer, into its own C++ scope. What it
deliberately does not ship is composition, because composition is the
CONSUMER's decision (a downstream package's own group planner) and HAWK must
not invent one. :func:`hawk.emit.compose` is the other half: given the
membership, it produces a deployable kernel — the merged slot order, one
index prologue, one body, one artifact through the ordinary publish path, and
a ``LaneMeta`` record per lane in the sidecar.

THE ROW THAT MATTERS is :func:`test_the_fused_entry_computes_what_the_lanes_compute`
and its device twin: a fused group is only a fused group if it produces the SAME
BITS the lanes produce run separately. Not `allclose` — the same bits, because
each lane's body text is the text that lane emits alone, so any difference at
all would mean the fusion changed the arithmetic rather than merely the
launch. The device arm is there because the two backends wrap the SAME body
string but bind their arguments differently (by value vs a ``params[]``
block), and a merged ``arg_spec`` is exactly the thing those two paths could
disagree about.

Around it sit the three refusals a merged entry cannot survive: one NAME
meaning two things across lanes, two lanes committing to one output plane,
and a mapreduce sink beside anything else (which belongs to ``canonical``,
so the row checks that composition does not route around it).

This test previously failed via two plants, both observed and both removed; recorded per row.
"""

from __future__ import annotations

import _deploy as L
import _oracle as O
import numpy as np
import pytest

import hawk
import hawk.math as hm
from hawk.artifact import build_bundle
from hawk.emit import FusedKernel, Lane, compose
from hawk.ir import HawkError
from hawk.types import LaneMeta

N = 16


# --------------------------------------------------------------------------- #
# The group: three lanes sharing one input plane and one uniform.
# --------------------------------------------------------------------------- #
@hawk.kernel
def lane_scale(x: hawk.Vector[3], k: hawk.Param, scaled: hawk.Mutable[hawk.Vector[3]]):
    scaled = k * x


@hawk.kernel
def lane_energy(x: hawk.Vector[3], sq: hawk.Mutable[hawk.Scalar]):
    sq = hm.dot(x, x)


@hawk.kernel
def lane_speed(x: hawk.Vector[3], k: hawk.Param, spd: hawk.Mutable[hawk.Scalar]):
    spd = hm.norm(x) + k


GROUP = (lane_scale, lane_energy, lane_speed)


@pytest.fixture(scope="module")
def fused():
    return compose(GROUP, "fused_group")


@pytest.fixture(scope="module")
def built_group(tmp_path_factory, fused):
    """The fused unit and the three solo units, one cache, built once."""
    root = tmp_path_factory.mktemp("hawk_fused")
    cache = str(tmp_path_factory.mktemp("hawk_fused_cache"))
    units = {"fused": build_bundle([fused], root / "fused", targets=("cuda", "host"),
                                   cache_dir=cache)}
    for kernel in GROUP:
        units[kernel.name] = build_bundle([kernel], root / kernel.name,
                                          targets=("cuda", "host"), cache_dir=cache)
    return units


def _inputs():
    x = np.arange(3 * N, dtype=float).reshape(3, N) * 0.25 + 1.0
    return {"x": x, "k": 1.75}


# --------------------------------------------------------------------------- #
# The merged arg_spec.
# --------------------------------------------------------------------------- #
def test_a_shared_leaf_is_bound_exactly_once(fused):
    """Needs ``compose``'s merged spec to be the one ``canonical`` walk's
    product, not each lane's own ``arg_spec`` concatenated::

        AssertionError: a slot is bound twice: (('mutable', 'scaled'),
        ('vec_in', 'x'), ('uniform', 'k'), ('mutable', 'sq'), ('vec_in', 'x'),
        ('mutable', 'spd'), ('vec_in', 'x'), ('uniform', 'k'))
        assert 8 == 5

    The merged order is a WALK PRODUCT precisely so it cannot be built
    that way; a second enumeration is what forbids."""
    spec = fused.arg_spec
    assert len(spec) == len(set(spec)), f"a slot is bound twice: {spec}"
    assert spec == (("mutable", "scaled"), ("mutable", "spd"), ("mutable", "sq"),
                    ("vec_in", "x"), ("uniform", "k")), spec
    assert sum(1 for role, _ in spec if role == "vec_in") == 1
    assert sum(1 for role, _ in spec if role == "uniform") == 1


def test_the_lane_metadata_is_published_per_lane(fused):
    assert [type(m) for m in fused.lanes] == [LaneMeta] * 3
    assert [m.name for m in fused.lanes] == [k.name for k in GROUP]
    for meta, kernel in zip(fused.lanes, GROUP):
        assert meta.slots == kernel.arg_spec
        assert meta.digest == kernel.walk.digest


def test_the_composed_object_is_a_kernel_the_artifact_path_accepts(fused,
                                                                  built_group):
    assert isinstance(fused, FusedKernel)
    sidecar = built_group["fused"].artifacts[0].sidecar
    assert [list(pair) for pair in fused.arg_spec] == sidecar["arg_spec"]
    lanes = sidecar["lanes"]
    assert [entry["name"] for entry in lanes] == [k.name for k in GROUP]
    assert [entry["digest"] for entry in lanes] == [k.walk.digest for k in GROUP]
    assert lanes[0]["slots"] == [list(p) for p in lane_scale.arg_spec]


def test_the_emitted_unit_carries_one_index_prologue_and_three_lane_scopes(
        built_group):
    """One entry, one geometry seam, three scopes — a fused group is ONE launch
    and its lanes are C++ blocks, not launches."""
    for target, needle in (("cu", "blockIdx.x"), ("cpp", "const aether::SampleIndex i =")):
        text = (built_group["fused"].directory / f"fused_group.{target}").read_text()
        assert text.count(needle) == 1, f"{target}: {text.count(needle)} prologues"
        assert text.count("{  // lane ") == 3, text
        for kernel in GROUP:
            assert f"{{  // lane {kernel.name}" in text


# --------------------------------------------------------------------------- #
# The row that matters: same bits, both targets.
# --------------------------------------------------------------------------- #
def _solo(built_group, kernel, plugin_of, structure):
    from eagle import plan as eplan

    bundle = built_group[kernel.name]
    sidecar = next(a.sidecar for a in bundle.artifacts if a.name == kernel.name)
    plugin = plugin_of(bundle.directory, kernel.name, sidecar)
    kw = {name: _inputs()[name] for role, name in plugin.arg_spec
          if role != "mutable"}
    return eplan.plan(plugin, structure=structure).run(**kw)


def _fused_run(built_group, plugin_of, structure):
    from eagle import plan as eplan

    bundle = built_group["fused"]
    sidecar = bundle.artifacts[0].sidecar
    plugin = plugin_of(bundle.directory, "fused_group", sidecar)
    kw = {name: _inputs()[name] for role, name in plugin.arg_spec
          if role != "mutable"}
    return eplan.plan(plugin, structure=structure).run(**kw)


@pytest.mark.parametrize("arm", ("host", pytest.param("device", marks=pytest.mark.gpu)))
def test_the_fused_entry_computes_what_the_lanes_compute(built_group, fused, arm):
    """Bit-identical, both targets. A tolerance here would hide the one failure
    fusion can introduce: a lane reading another lane's register copy."""
    import eagle.exec as eexec

    plugin_of, structure = ((L.host_plugin, eexec.HostTeam) if arm == "host"
                            else (L.device_plugin, eexec.DeviceKernel))
    got = _fused_run(built_group, plugin_of, structure)
    names = [name for role, name in fused.arg_spec if role == "mutable"]
    by_name = got if isinstance(got, dict) else {names[0]: got}
    for kernel in GROUP:
        alone = _solo(built_group, kernel, plugin_of, structure)
        solo_names = [name for role, name in kernel.arg_spec if role == "mutable"]
        alone_by_name = alone if isinstance(alone, dict) else {solo_names[0]: alone}
        for name in solo_names:
            np.testing.assert_array_equal(
                np.asarray(by_name[name]), np.asarray(alone_by_name[name]),
                err_msg=f"lane {kernel.name!r} plane {name!r} on the {arm} arm")


def test_the_fused_entry_agrees_with_hawks_own_serial_oracle(built_group, fused):
    """The same claim against HAWK's own host path, which is the arm the
    partitioned/ranked runs are judged against and does not go through eagle."""
    loaded = O.load(built_group["fused"].directory, "fused_group")
    kw = {name: _inputs()[name] for role, name in loaded.arg_spec
          if role in O.INPUT_ROLES or role == "uniform"}
    got = O.run_kernel(loaded, N, **kw)
    x, k = _inputs()["x"], _inputs()["k"]
    want = (k * x, np.linalg.norm(x, axis=0) + k, (x * x).sum(axis=0))
    for mine, theirs in zip(got, want):
        np.testing.assert_allclose(mine, theirs, rtol=1e-13, atol=1e-13)


# --------------------------------------------------------------------------- #
# The refusals.
# --------------------------------------------------------------------------- #
def test_one_name_meaning_two_things_is_refused_naming_both_lanes():
    """Needs ``_check_slots`` to actually run. If it is short-circuited, both
    this row and the double-write one below go::

        Failed: DID NOT RAISE HawkError

    and the group composed: the merged ``arg_spec`` carried ``theta`` once, at
    whichever width came first, and the other lane read three components out of
    a one-component plane. It builds, it runs, and it is wrong."""
    @hawk.kernel
    def wide_theta(theta: hawk.Vector[3], a: hawk.Mutable[hawk.Scalar]):
        a = hm.norm(theta)

    @hawk.kernel
    def scalar_theta(theta: hawk.Scalar, b: hawk.Mutable[hawk.Scalar]):
        b = theta * 2.0

    with pytest.raises(HawkError) as excinfo:
        compose([wide_theta, scalar_theta], "conflict")
    message = str(excinfo.value)
    assert "wide_theta" in message and "scalar_theta" in message, message
    assert "theta" in message and "arg_spec" in message


def test_two_lanes_committing_to_one_plane_are_refused_naming_both():
    @hawk.kernel
    def writes_y_a(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x + 1.0

    @hawk.kernel
    def writes_y_b(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x + 2.0

    with pytest.raises(HawkError) as excinfo:
        compose([writes_y_a, writes_y_b], "double_write")
    message = str(excinfo.value)
    assert "writes_y_a" in message and "writes_y_b" in message, message
    assert "write it twice" in message


def test_a_mapreduce_lane_beside_another_lane_is_refused_by_h86():
    """ is not re-implemented here: the merged sink set goes through
    ``canonical()``, which refuses it for the same reason it refuses a single
    kernel's — ``exec_access`` is ONE scalar per manifest, so eagle would never
    fold the partials."""
    @hawk.kernel
    def reduces(v: hawk.Vector[3], total: hawk.Reduce("sum")):
        total.contribute(hm.dot(v, v))

    with pytest.raises(HawkError, match="may not coexist"):
        compose([reduces, lane_energy], "reduce_plus_mutable")

    @hawk.kernel
    def reduces_again(v: hawk.Vector[3], other: hawk.Reduce("sum")):
        other.contribute(hm.norm(v))

    with pytest.raises(HawkError, match="may not coexist"):
        compose([reduces, reduces_again], "two_reduces")

    lone = compose([reduces], "lone_reduce")
    assert lone.walk.access == ("mapreduce", "sum")
    assert lone.fused_body().reduce_op == "sum"


def test_an_empty_or_ambiguous_group_and_an_unbuilt_scalar_mode_are_refused():
    with pytest.raises(HawkError, match="empty lane group"):
        compose([], "nothing")
    with pytest.raises(HawkError, match="two lanes share a name"):
        compose([lane_energy, lane_energy], "twice")
    with pytest.raises(HawkError, match="neither a Lane nor a traced kernel"):
        compose([object()], "not_a_lane")
    with pytest.raises(HawkError, match="declared but not built yet"):
        compose(GROUP, "banded_group", "banded")


def test_a_lane_carries_its_declared_but_unread_slots_into_the_merged_spec():
    """ "declared => bound, read or not", applied across lanes: a
    lane's ``terminated`` mask is consumed by the seam and never named in its
    body, and a fused entry that dropped it would take a different signature
    from the lane it fused."""
    @hawk.kernel
    def guarded(s: hawk.Scalar, terminated: hawk.Terminated, g: hawk.Mutable[hawk.Scalar]):
        g = s + 1.0

    fused_guarded = compose([guarded, lane_energy], "guarded_group")
    assert ("terminated", "terminated") in fused_guarded.arg_spec
    assert ("terminated", "terminated") in guarded.arg_spec


def test_a_lane_record_may_carry_its_own_prologue_hook():
    """The lane-local hook leaves open survives composition."""
    lane = Lane("hooked", lane_energy.sinks, lane_energy.walk,
                prologue="// lane guard goes here")
    fused_hooked = compose([lane], "hooked_group")
    text = fused_hooked.fused_body().text
    assert "// lane guard goes here" in text
    assert text.index("// lane guard") < text.index("mut_sq[i]")
