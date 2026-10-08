# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A Mutable's launch-start value — a bare read of its own name before its
first store.

The read leaf and the plane's own store merge on ``(role, name)`` exactly
like any other declared slot, so a running statistic (a running max, a
rank-1 recurrence) is one more Mutable body, not a new wire or a new ABI
entry. The host owns the memory: it must initialise a plane read before its
first store before the first launch and preserve it between launches, and a
plane the body only reads before ever storing it — never committed — is an
input, not an output, so it refuses exactly like an undeclared output would.

This test previously failed: reverting ``LEAF_ROLES``/``LEAF_KINDS`` to drop
``mutable``/``prior_read`` makes every kernel below refuse to trace
(``unknown leaf role 'mutable'``); flagging every ``mutables`` sidecar entry
``prior: True`` unconditionally breaks the byte-identity row; disabling the
transform's own refusal lets ``vjp``/``jvp`` silently build the wrong
derivative. Every plant was then removed."""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Accum as _Accum
from hawk import Mutable as _Mutable
from hawk import Scalar as _Scalar
from hawk import Terminated as _Terminated
from hawk import Vector as _Vector
from hawk import kernel as _hkernel
from hawk.artifact import build_bundle as _build_bundle
from hawk.ir import HawkError
from hawk.math import cross as _cross
from hawk.math import maximum as _maximum
from hawk.math import norm as _norm

N = 8


@_hkernel
def running_apogee(position: _Vector[3], terminated: _Terminated,
                   r_max: _Mutable[_Scalar]):
    """a running max — the recurrence a running statistic is written
    through."""
    r_max = _maximum(r_max, _norm(position))


@_hkernel
def spin_prior(w: _Vector[3], terminated: _Terminated, v: _Mutable[_Vector[3]]):
    """the rank-1 twin, whose ``cross(v, w)`` reads EVERY component of the
    prior value before any component of the new one is stored — the
    aliasing hazard the materialisation rule closes."""
    v = _cross(v, w)


@pytest.fixture(scope="module")
def prior_bundle(tmp_path_factory, cache_dir):
    return _build_bundle([running_apogee, spin_prior],
                         tmp_path_factory.mktemp("prior_bundle"),
                         targets=("cuda", "host"), cache_dir=cache_dir)


def _sequential_launch(bundle, name, param, terms, *, mask=None, extra=None):
    """Bind the first term, ``rebind``+``launch`` for every later one — the
    SAME output buffer persists across all K launches, so the recurrence is
    actually observed (:mod:`test_ext_seams`'s own ``_sequential_launch``)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
    plan = eplan.plan(plugin, structure=eexec.HostTeam)
    out_name, out = extra
    kw = {param: terms[0], "terminated": mask, out_name: out}
    bound = plan.bind(**kw)
    bound.launch()
    for t in terms[1:]:
        bound = bound.rebind(**{param: t})
        bound.launch()
    return out


# --------------------------------------------------------------------------- #
# the running max, across launches whose max is NOT the last term.
# --------------------------------------------------------------------------- #
def test_running_max_recovers_the_max_across_launches(prior_bundle):
    magnitudes = [3.0, 7.0, 2.0, 9.0, 4.0, 1.0, 5.0, 6.0]  # max=9.0, not last
    positions = [np.tile(np.array([m, 0.0, 0.0])[:, None], (1, N)) for m in magnitudes]
    mask = np.zeros(N, dtype=bool)
    r_max = np.zeros(N)
    r_max = _sequential_launch(prior_bundle, "running_apogee", "position",
                               positions, mask=mask, extra=("r_max", r_max))
    np.testing.assert_array_equal(r_max, np.full(N, 9.0))


# --------------------------------------------------------------------------- #
# host/cuda BODY byte identity (the renderer is backend-agnostic by
# construction; this row pins that no per-backend branch ever creeps in for
# a `prior_read` leaf).
# --------------------------------------------------------------------------- #
def test_prior_recurrence_renders_byte_identical_host_and_cuda_bodies():
    from hawk.emit.backend import render_source
    from hawk.emit.cuda import CudaBackend
    from hawk.emit.host import HostBackend

    for k in (running_apogee, spin_prior):
        h = render_source(k.name, k.sinks, k.walk, HostBackend(), kind=k.kind)
        c = render_source(k.name, k.sinks, k.walk, CudaBackend(), kind=k.kind)
        assert h.body == c.body, k.name

    spin_body = render_source(spin_prior.name, spin_prior.sinks, spin_prior.walk,
                              HostBackend(), kind=spin_prior.kind).body
    assert "mut_v_i = mut_v[i].get()" in spin_body, (
        "a rank>=1 launch-start read must be materialised BEFORE the store:\n" + spin_body)


# --------------------------------------------------------------------------- #
# the aliasing hazard `cross(v, w)` would hit without the
# materialisation rule — component 1 would read component 0's freshly
# STORED value instead of its prior one.
# --------------------------------------------------------------------------- #
def test_rank1_prior_read_matches_the_numpy_recurrence_bit_for_bit(prior_bundle):
    terms = [np.array(t) for t in
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
             [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0])]
    v_ref = np.zeros(3)
    for t in terms:
        v_ref = np.cross(v_ref, t)

    w_launches = [np.tile(t[:, None], (1, N)) for t in terms]
    mask = np.zeros(N, dtype=bool)
    v = np.zeros((3, N))
    v = _sequential_launch(prior_bundle, "spin_prior", "w", w_launches,
                           mask=mask, extra=("v", v))
    for k in range(3):
        np.testing.assert_array_equal(v[k], np.full(N, v_ref[k]))


# --------------------------------------------------------------------------- #
# a terminated sample keeps its PRIOR bytes across every later launch —
# the guard sits around the store, the read stays unconditional and harmless.
# --------------------------------------------------------------------------- #
def test_a_terminated_sample_keeps_its_prior_value_across_launches(prior_bundle):
    magnitudes = [3.0, 7.0, 2.0, 9.0, 4.0, 1.0, 5.0, 6.0]
    positions = [np.tile(np.array([m, 0.0, 0.0])[:, None], (1, N)) for m in magnitudes]
    mask = np.zeros(N, dtype=bool)
    mask[0] = True
    r_max = np.zeros(N)
    r_max = _sequential_launch(prior_bundle, "running_apogee", "position",
                               positions, mask=mask, extra=("r_max", r_max))
    assert r_max[0] == 0.0, (
        f"a terminated sample must keep its PRIOR bytes across every launch: "
        f"got {r_max[0]}")
    np.testing.assert_array_equal(r_max[1:], np.full(N - 1, 9.0))


# --------------------------------------------------------------------------- #
# the sidecar names a plane read before its first store (host must init + keep
# it), and stays SILENT for an ordinary write-only Mutable — the existing
# corpus's own sidecars are unchanged.
# --------------------------------------------------------------------------- #
def test_sidecar_flags_a_mutable_read_through_prior(prior_bundle):
    sc = sidecar_of(prior_bundle, "running_apogee")
    entry = next(m for m in sc["mutables"] if m["name"] == "r_max")
    assert entry["prior"] is True


def test_sidecar_omits_prior_for_a_write_only_mutable(built):
    sc = sidecar_of(built["vec3"], "vec3_scale")
    entry = next(m for m in sc["mutables"] if m["name"] == "y")
    assert "prior" not in entry


# --------------------------------------------------------------------------- #
# the three refusals.
# --------------------------------------------------------------------------- #
def test_a_mutable_read_through_prior_but_never_committed_refuses():
    with pytest.raises(HawkError, match="declared but never committed"):
        @_hkernel
        def bad(a: _Scalar, r: _Mutable[_Scalar]):  # noqa: F841
            b = a + r  # noqa: F841


def test_prior_on_a_scattered_or_reduced_plane_refuses():
    with pytest.raises(HawkError, match="scattered or reduced"):
        @_hkernel
        def bad(a: _Scalar, acc: _Accum[_Scalar]):  # noqa: F841
            b = a + acc.prior
            acc.add(b)


def test_the_derivative_of_a_prior_recurrence_refuses():
    from hawk.diff import jvp, vjp

    for f in (jvp, vjp):
        with pytest.raises(HawkError, match="is read before it is assigned"):
            f(running_apogee)


# --------------------------------------------------------------------------- #
# host and device compute the SAME running max from the SAME launch-start
# read — a guarantee that holds BY CONSTRUCTION (own
# column, one writer per sample, no atomics), exercised end to end rather
# than through a text comparison alone.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_running_max_matches_on_host_and_device(prior_bundle):
    import eagle.exec as eexec
    from eagle import plan as eplan

    mask = np.zeros(N, dtype=bool)

    def two_launches(plugin, structure):
        plan = eplan.plan(plugin, structure=structure)
        r0 = np.zeros(N)
        got1 = plan.run(position=np.tile(np.array([3.0, 0.0, 0.0])[:, None], (1, N)),
                        terminated=mask, r_max=r0)
        r1 = np.asarray(got1.get() if hasattr(got1, "get") else got1)
        got2 = plan.run(position=np.tile(np.array([9.0, 0.0, 0.0])[:, None], (1, N)),
                        terminated=mask, r_max=r1)
        return np.asarray(got2.get() if hasattr(got2, "get") else got2)

    host_plugin = L.host_plugin(prior_bundle.directory, "running_apogee",
                                sidecar_of(prior_bundle, "running_apogee"))
    dev_plugin = L.device_plugin(prior_bundle.directory, "running_apogee",
                                 sidecar_of(prior_bundle, "running_apogee"))
    r_host = two_launches(host_plugin, eexec.HostTeam)
    r_dev = two_launches(dev_plugin, eexec.DeviceKernel)
    np.testing.assert_array_equal(r_host, r_dev)
    np.testing.assert_array_equal(r_host, np.full(N, 9.0))
