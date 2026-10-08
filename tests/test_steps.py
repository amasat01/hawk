# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.steps(kernel, K)``: K steps of a finishing step kernel per launch.

Rows:

* **bit identity, host AND device** — ``K`` in ``{4, 16}`` run until every
  sample finished leaves every plane, the mask and the counter bit-equal to
  ``K = 1``, in ``ceil(S / K)`` launches (``S`` the longest sample's stop step).
* **sidecar** — ``finish.steps == K``, same slots as the single step.
* **refusals** — a kernel with no finish, an ``out``/``Reduce`` sink and a
  derivative of the derived kernel each refuse naming the kernel.
* **decorator form** — ``@kernel(steps=K)`` is ``hawk.steps(kernel(fn), K)``
  (same name, walk digest and emitted source), ``steps=1`` is the plain kernel
  (same digest and source), and a bad ``steps`` or a kernel ``hawk.steps``
  refuses each refuse at decoration naming the kernel.
* **steps="auto"** — ONE kernel ``<name>_xauto`` reading its trip count from
  the reserved ``fused_steps`` word: sidecar ``finish.steps == "auto"`` with
  ``steps_max == AUTO_K_MAX``, the word a ``lookup`` slot beside
  ``finished_count``, misspellings and a clashing plane refused; compiled, a
  launch with ``fused_steps = k`` takes exactly ``k`` steps — after every
  launch of a hand-driven ``k`` sequence the planes equal the step-by-step
  reference after the same total (host: the ``K = 1`` kernel, bit-identical;
  device: the same auto artifact driven with ``k = 1``, since nvcc's default
  FMA contraction is kept and may differ from ``K = 1`` by an ULP), plain and
  active-set kind (the map compacted by hand between launches). Non-vacuity: a launch of ``k = 37``
  equals 37 one-step launches and differs from 36 and 38, and the fixed
  ``steps=16`` artifact overshoots the same cap to 48.
"""

from __future__ import annotations

import math

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import (
    Accum,
    Index,
    Mutable,
    Param,
    Reduce,
    Scalar,
    Terminated,
    Vector,
    kernel,
)
from hawk.artifact import build_bundle
from hawk.artifact.sidecar import sidecar_meta
from hawk.diff import jvp, vjp
from hawk.ir import HawkError

N = 48
DT = 0.0625
CAP = 400


@kernel(steps=1)
def osc(dt: Param, t_final: Scalar, terminated: Terminated,
        x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def spin3(dt: Param, limit: Scalar, terminated: Terminated,
          r: Mutable[Vector[3]], s: Mutable[Scalar]):
    """A rank-1 carry beside a scalar one, finishing on the carried state."""
    w = hawk.math.vec(0.3, -0.2, 0.7)
    rn = r + hawk.math.cross(w, r) * dt
    r = rn
    sn = s + hawk.math.dot(rn, rn) * dt
    s = sn
    terminated = sn >= limit


KS = (4, 16)


def test_the_derived_kernel_is_named_and_records_its_steps():
    for k in KS:
        derived = hawk.steps(osc, k)
        assert derived.name == f"osc_x{k}"
        assert derived.walk.finish == ("terminated", k)
        assert derived.arg_spec == osc.arg_spec
        meta = sidecar_meta(derived.name, derived.walk, fmt="ptx",
                            scalar_type="float64")
        assert meta["finish"] == {"mask": "terminated",
                                  "counter": "finished_count", "steps": k}


def test_a_kernel_without_a_finish_refuses():
    @kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = y + x
    with pytest.raises(HawkError, match=r"'plain'.*never finishes"):
        hawk.steps(plain, 4)


def test_a_reduce_or_out_sink_refuses():
    """A ``Reduce`` beside a finish already refuses at trace (a mapreduce sink
    is exclusive); an ``Accum`` traces and is refused here."""
    def reducing(x: Scalar, terminated: Terminated, total: Reduce("sum")):
        total.contribute(x)
        terminated = x > 1.0
    with pytest.raises(HawkError, match=r"mapreduce sink 'total'"):
        kernel(reducing)

    @kernel
    def scattering(x: Scalar, lane: Index, terminated: Terminated,
                   acc: Accum[Scalar]):
        acc.add(x, at=lane)
        terminated = x > 1.0
    with pytest.raises(HawkError, match=r"'scattering'.*not a step kernel"):
        hawk.steps(scattering, 4)


def test_a_bad_k_refuses():
    for bad in (0, -1, 2.0, True):
        with pytest.raises(HawkError, match=r"'osc'.*positive int"):
            hawk.steps(osc, bad)


# --------------------------------------------------------------------------- #
# -- the decorator form: @kernel(steps=K) == hawk.steps(kernel(fn), K).
# --------------------------------------------------------------------------- #
def _osc_body(dt: Param, t_final: Scalar, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def _sources(k):
    from hawk.emit import BACKENDS, render_source

    return [render_source(k.name, k.sinks, k.walk, b) for b in BACKENDS.values()]


@pytest.mark.parametrize("k", [4, 16])
def test_the_decorator_form_is_hawk_steps(k):
    decorated = kernel(steps=k)(_osc_body)
    derived = hawk.steps(kernel(_osc_body), k)
    assert decorated.name == derived.name == f"_osc_body_x{k}"
    assert decorated.walk.finish == ("terminated", k)
    assert decorated.walk.digest == derived.walk.digest
    assert decorated.arg_spec == derived.arg_spec
    assert _sources(decorated) == _sources(derived)


def test_the_decorator_form_reaches_a_defining_scope_local():
    t_cap = Scalar

    @kernel(steps=4)
    def local_osc(dt: Param, t_final: t_cap, terminated: Terminated,
                  t: Mutable[Scalar]):
        tn = t + dt
        t = tn
        terminated = tn >= t_final
    assert local_osc.name == "local_osc_x4"
    assert local_osc.walk.finish == ("terminated", 4)


def test_steps_one_is_the_plain_kernel():
    plain = kernel(_osc_body, steps=1)
    for one in (kernel(steps=1)(_osc_body), kernel(_osc_body, steps=1),
                kernel()(_osc_body).step, kernel(_osc_body).step):
        assert one.name == plain.name == "_osc_body"
        assert one.walk.digest == plain.walk.digest
        assert one.walk.finish == ("terminated", 1)
        assert _sources(one) == _sources(plain)


def test_the_decorator_form_keeps_the_kind():
    from hawk.ext import Guard, Kind

    kind = Kind("steps_active", guard=Guard(active_set=True))
    decorated = kernel(kind=kind, steps=4)(_osc_body)
    assert decorated.kind is kind
    assert decorated.walk.digest == hawk.steps(kernel(_osc_body, kind=kind), 4).walk.digest


def test_a_bad_steps_refuses_at_decoration_naming_the_kernel():
    for bad in (0, -1, 2.0, True, "4"):
        with pytest.raises(HawkError, match=r"steps=.*'_osc_body'.*positive int"):
            kernel(steps=bad)(_osc_body)


def test_steps_auto_misspellings_refuse_naming_the_kernel():
    for bad in ("Auto", "fast", "AUTO", " auto", 0):
        with pytest.raises(HawkError, match=r"steps=.*'_osc_body'.*'auto'"):
            kernel(steps=bad)(_osc_body)
        with pytest.raises(HawkError, match=r"hawk.steps.*'osc'.*'auto'"):
            hawk.steps(osc, bad)


def test_the_decorator_form_refuses_what_hawk_steps_refuses():
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = y + x
    with pytest.raises(HawkError, match=r"'plain'.*never finishes"):
        kernel(steps=4)(plain)

    def scattering(x: Scalar, lane: Index, terminated: Terminated,
                   acc: Accum[Scalar]):
        acc.add(x, at=lane)
        terminated = x > 1.0
    with pytest.raises(HawkError, match=r"'scattering'.*not a step kernel"):
        kernel(steps=4)(scattering)


@pytest.mark.parametrize("transform", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_the_derived_kernel_refuses(transform):
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        transform(hawk.steps(osc, 4), wrt=("x",))


# --------------------------------------------------------------------------- #
# -- compiled: K steps per launch == K launches of one step, bit for bit.
# --------------------------------------------------------------------------- #
def _active_kind():
    from hawk.ext import Guard, Kind

    return Kind("steps_auto_active", guard=Guard(active_set=True))


#: The active-set kind's auto kernel (named ``_osc_body_xauto``).
osc_active_auto = hawk.steps(kernel(_osc_body, kind=_active_kind()), "auto")


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("steps")
    kernels = ([osc, spin3] + [hawk.steps(k, n) for k in (osc, spin3) for n in KS]
               + [hawk.steps(osc, "auto"), hawk.steps(spin3, "auto")])
    # Host under the EXACT profile: K fused steps against K = 1 bit for bit
    # (the default fast profile may fuse the two loop shapes differently).
    return build_bundle(kernels, root / "steps", targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native")


@pytest.fixture(scope="module")
def active_bundle(tmp_path_factory, cache_dir):
    """The active-set kind's auto kernel: a bundle has ONE kind."""
    root = tmp_path_factory.mktemp("steps_active")
    return build_bundle([osc_active_auto], root / "steps_active",
                        targets=("cuda", "host"), cache_dir=cache_dir,
                        host_profile="native")


def _planes(which, xp):
    rng = np.random.default_rng(11)
    if which == "osc":
        stop = rng.integers(5, 70, N)
        return {"t_final": xp.array(DT * stop.astype(float)),
                "x": xp.array(rng.uniform(-1, 1, N)),
                "v": xp.array(rng.uniform(-1, 1, N)), "t": xp.zeros(N)}, int(stop.max())
    r = rng.uniform(-1, 1, (3, N))
    return {"limit": xp.array(rng.uniform(0.2, 3.0, N)), "r": xp.array(r),
            "s": xp.zeros(N)}, None


def _run(bundle, name, which, target):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure, xp = (L.host_plugin(bundle.directory, name, sidecar),
                                 eexec.HostTeam, np)
    else:
        import cupy as cp

        plugin, structure, xp = (L.device_plugin(bundle.directory, name, sidecar),
                                 eexec.DeviceKernel, cp)
    planes, stop = _planes(which, xp)
    planes["terminated"] = xp.zeros(N, dtype=bool)
    counter = xp.zeros(1, dtype=xp.uint32)
    bound = eplan.plan(plugin, structure=structure).bind(
        dt=DT, finished_count=counter.view(xp.int32), **planes)
    launches = 0
    while int(counter[0]) < N and launches < CAP:
        bound.launch()
        launches += 1
    got = {k: np.asarray(a.get() if hasattr(a, "get") else a)
           for k, a in planes.items()}
    return got, int(counter[0]), launches, stop


@pytest.mark.parametrize("which", ["osc", "spin3"])
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_k_steps_per_launch_are_bit_identical_to_one(bundles, which, target):
    one, counted, launches, stop = _run(bundles, which, which, target)
    assert counted == N and one["terminated"].all() and launches < CAP
    if stop is not None:
        assert launches == stop
    for k in KS:
        got, counted_k, launches_k, _ = _run(bundles, f"{which}_x{k}", which, target)
        assert counted_k == N, f"{which}_x{k}/{target}: counter {counted_k}"
        assert launches_k == math.ceil(launches / k), (k, launches_k, launches)
        for plane, ref in one.items():
            np.testing.assert_array_equal(got[plane], ref,
                                          err_msg=f"{which}_x{k}/{target}:{plane}")


# --------------------------------------------------------------------------- #
# -- steps="auto": ONE kernel, the trip count a device word.
# --------------------------------------------------------------------------- #
def test_the_auto_kernel_is_named_and_records_auto_steps():
    from hawk.trace.steps import AUTO_K_MAX

    assert AUTO_K_MAX == 64
    for base in (osc, spin3):
        auto = hawk.steps(base, "auto")
        assert auto.name == f"{base.name}_xauto"
        assert auto.walk.finish == ("terminated", "auto")
        spec = set(auto.arg_spec)
        assert {("lookup", "fused_steps"), ("lookup", "finished_count")} <= spec
        assert spec - set(base.arg_spec) == {("lookup", "fused_steps")}
        assert auto.walk.slot_types[("lookup", "fused_steps")].dtype == "i32"
        meta = sidecar_meta(auto.name, auto.walk, fmt="ptx", scalar_type="float64")
        assert meta["finish"] == {"mask": "terminated", "counter": "finished_count",
                                  "steps": "auto", "steps_max": AUTO_K_MAX}
        # the word is the runner's seam, not a cross-sample read of the body
        plain = sidecar_meta(base.name, base.walk, fmt="ptx", scalar_type="float64")
        assert meta["exec_access"] == plain["exec_access"] == "sample_local"
        assert meta["arg_dtypes"]["fused_steps"] == "int64"
    fixed = sidecar_meta("osc_x16", hawk.steps(osc, 16).walk, fmt="ptx",
                         scalar_type="float64")
    assert "steps_max" not in fixed["finish"]


def test_the_auto_decorator_form_is_hawk_steps_auto():
    decorated = kernel(steps="auto")(_osc_body)
    derived = hawk.steps(kernel(_osc_body), "auto")
    assert decorated.name == derived.name == "_osc_body_xauto"
    assert decorated.walk.digest == derived.walk.digest
    assert decorated.arg_spec == derived.arg_spec
    assert _sources(decorated) == _sources(derived)
    assert osc_active_auto.kind.guard.active_set


def test_the_auto_loop_reads_the_word_once_and_the_fixed_loop_does_not():
    """The auto header runs to the word read (and clamped) before the loop; the fixed-K
    header keeps its literal bound and binds no word."""
    import re

    from _interchange import builds

    for source in _sources(hawk.steps(osc, "auto")):
        for shape, text in builds(source.text).items():
            read = re.search(r"const auto (\w+) = lut_fused_steps\[", text)
            assert read and text.count("lut_fused_steps[") == 1, shape
            w = read.group(1)
            clamp = re.search(
                rf"const auto (\w+) = \(\({w} < static_cast<Int>\(0\)\) \? "
                rf"static_cast<Int>\(0\) : {w}\);", text)
            assert clamp, "the word is clamped below at 0 before the loop"
            assert "static_cast<Int>(64)" not in text, "no static bound on the word"
            if shape == "plain" or source.backend != "host":
                assert re.search(rf"hawk_k\d+_step < hawk_clamp_count\({clamp.group(1)}\);", text)
            else:
                # the interchanged loop: the word is each lane's trip count
                assert f"hawk_cnt[hawk_t] = {clamp.group(1)};" in text
                assert "hawk_rcnt[hawk_l] = hawk_cnt[hawk_t];" in text
                assert "hawk_trip + 1 < hawk_rcnt[hawk_l]" in text
    for source in _sources(hawk.steps(osc, 16)):
        for shape, text in builds(source.text).items():
            assert "fused_steps" not in text
            plain = shape == "plain" or source.backend != "host"
            assert re.search(r"hawk_k\d+_step < 16;" if plain
                             else r"hawk_trip < 16;", text), shape


def test_auto_refuses_what_hawk_steps_refuses():
    @kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = y + x
    with pytest.raises(HawkError, match=r"'plain'.*never finishes"):
        hawk.steps(plain, "auto")

    def scattering(x: Scalar, lane: Index, terminated: Terminated,
                   acc: Accum[Scalar]):
        acc.add(x, at=lane)
        terminated = x > 1.0
    with pytest.raises(HawkError, match=r"'scattering'.*not a step kernel"):
        kernel(steps="auto")(scattering)
    with pytest.raises(HawkError, match=r"'osc_xauto'.*already takes"):
        hawk.steps(hawk.steps(osc, "auto"), "auto")


def test_auto_refuses_a_plane_named_like_the_word():
    def clash(fused_steps: Scalar, terminated: Terminated, x: Mutable[Scalar]):
        x_start = x
        x = x_start + fused_steps
        terminated = x_start > 1.0
    with pytest.raises(HawkError, match=r"'clash'.*'fused_steps' is reserved"):
        kernel(steps="auto")(clash)


@pytest.mark.parametrize("transform", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_the_auto_kernel_refuses(transform):
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        transform(hawk.steps(osc, "auto"), wrt=("x",))


#: The hand-driven trip counts: short, long, the bound, odd remainders.
K_SEQ = (1, 3, 64, 7, 2, 16, 5, 33)


def _driver(bundle, name, which, target, *, active=False, never=False):
    """``(bound, planes, counter, word, xp, active_planes)`` of one run."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure, xp = (L.host_plugin(bundle.directory, name, sidecar),
                                 eexec.HostTeam, np)
    else:
        import cupy as cp

        plugin, structure, xp = (L.device_plugin(bundle.directory, name, sidecar),
                                 eexec.DeviceKernel, cp)
    planes, _ = _planes(which, xp)
    if never:
        planes["t_final"] = xp.full(N, np.inf)
    planes["terminated"] = xp.zeros(N, dtype=bool)
    counter = xp.zeros(1, dtype=xp.uint32)
    extra = {}
    if sidecar.get("finish", {}).get("steps") == "auto":
        # the word rides hawk's 64-bit wire for an i32 leaf (arg_dtypes)
        extra["fused_steps"] = xp.zeros(1, dtype=xp.int64)
    if active:
        extra["active_map"] = xp.arange(N, dtype=xp.int32)
        extra["active_count"] = xp.full(1, N, dtype=xp.int32)
    bound = eplan.plan(plugin, structure=structure).bind(
        dt=DT, finished_count=counter.view(xp.int32), **planes, **extra)
    return bound, planes, counter, extra, xp


def _host(planes):
    return {k: np.array(a.get() if hasattr(a, "get") else a, copy=True)
            for k, a in planes.items()}


def _one_step_snapshots(bundle, name, which, target, *, never=False,
                        upto=None, active=False):
    """One step per launch, the planes after every launch: ``snaps[s]`` =
    after ``s`` steps (``snaps[0]`` the initial planes), until all finished or
    ``upto``. ``name`` is the K = 1 kernel, or an auto kernel driven with
    ``fused_steps = 1`` (the device oracle: the same compiled loop body)."""
    bound, planes, counter, extra, xp = _driver(bundle, name, which, target,
                                                never=never, active=active)
    snaps = [_host(planes)]
    while (int(counter[0]) < N if upto is None else len(snaps) <= upto):
        assert len(snaps) < CAP
        if "fused_steps" in extra:
            extra["fused_steps"][0] = 1
        bound.launch()
        snaps.append(_host(planes))
        if active:
            _compact(extra, planes, xp)
    return snaps


def _oracle(bundles, bundle, name, which, target, **kw):
    """The step-by-step reference of an auto kernel: the SAME auto artifact
    one step per launch (``fused_steps = 1``), which runs the single step's
    straight-line body, i.e. the K = 1 kernel. Host results equal it bit for
    bit; on the device nvcc's default FMA contraction is kept, so a fused
    launch may differ from it by an ULP where a multiply-add is contracted
    differently (:func:`_same`)."""
    return _one_step_snapshots(bundle, name, which, target, **kw)


#: The device's tolerance for a fused launch against one step per launch: a
#: few ULPs of contraction difference, accumulated over at most ``CAP`` steps
#: of O(1) planes; a step off (``dt = 0.0625``) is ~1e-2 away.
_DEVICE_RTOL, _DEVICE_ATOL = 1e-12, 1e-14


def _same(got, want, target) -> bool:
    """``got == want`` bit for bit on the host; on the device within
    :data:`_DEVICE_RTOL` for a floating plane (exact for any other)."""
    got, want = np.asarray(got), np.asarray(want)
    if target == "host" or got.dtype.kind != "f":
        return got.shape == want.shape and bool(np.array_equal(got, want))
    return got.shape == want.shape and bool(
        np.allclose(got, want, rtol=_DEVICE_RTOL, atol=_DEVICE_ATOL))


def _assert_same(got, want, target, err_msg=""):
    if target == "host" or np.asarray(got).dtype.kind != "f":
        np.testing.assert_array_equal(got, want, err_msg=err_msg)
    else:
        np.testing.assert_allclose(got, want, rtol=_DEVICE_RTOL, atol=_DEVICE_ATOL,
                                   err_msg=err_msg)


def _compact(extra, planes, xp):
    """The active-set map compacted by hand: the ascending live indices."""
    live = np.flatnonzero(~_host({"m": planes["terminated"]})["m"]).astype(np.int32)
    extra["active_map"][: live.size] = xp.asarray(live)
    extra["active_count"][0] = live.size


@pytest.mark.parametrize("which,name,active", [
    ("osc", "osc_xauto", False), ("spin3", "spin3_xauto", False),
    ("osc", "_osc_body_xauto", True)], ids=["osc", "spin3", "osc-active-set"])
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_auto_takes_exactly_the_word_s_steps_per_launch(bundles, active_bundle,
                                                        which, name, active,
                                                        target):
    """After every launch of the ``K_SEQ`` cycle every plane, the mask and the
    counter equal the step-by-step reference (:func:`_oracle`) after the same
    total number of steps."""
    bundle = active_bundle if active else bundles
    snaps = _oracle(bundles, bundle, name, which, target, active=active)
    bound, planes, counter, extra, xp = _driver(bundle, name, which, target,
                                                active=active)
    total, launches = 0, 0
    while int(counter[0]) < N:
        k = K_SEQ[launches % len(K_SEQ)]
        extra["fused_steps"][0] = k
        bound.launch()
        launches += 1
        total += k
        ref = snaps[min(total, len(snaps) - 1)]
        got = _host(planes)
        for plane, want in ref.items():
            _assert_same(
                got[plane], want, target, err_msg=f"{name}/{target}: {plane} after "
                f"{launches} launches ({total} steps)")
        if active:
            _compact(extra, planes, xp)
        assert launches < CAP
    assert got["terminated"].all() and total >= len(snaps) - 1


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_auto_caps_exactly_where_the_fixed_artifact_overshoots(bundles, target):
    """Samples that never finish: launches of 16, 16, 5 leave every sample at
    exactly step 37 (equal to 37 one-step launches, unlike 36 or 38); the
    fixed ``steps=16`` artifact run to the same cap lands on 48."""
    snaps = _oracle(bundles, bundles, "osc_xauto", "osc", target, never=True,
                    upto=48)
    fixed_snaps = _one_step_snapshots(bundles, "osc", "osc", target, never=True,
                                      upto=48)
    bound, planes, _, extra, _ = _driver(bundles, "osc_xauto", "osc", target,
                                         never=True)
    for k in (16, 16, 5):
        extra["fused_steps"][0] = k
        bound.launch()
    auto = _host(planes)
    fixed_bound, fixed_planes, _, _, _ = _driver(bundles, "osc_x16", "osc",
                                                 target, never=True)
    for _ in range(math.ceil(37 / 16)):
        fixed_bound.launch()
    fixed = _host(fixed_planes)
    for plane in ("x", "v", "t"):
        _assert_same(auto[plane], snaps[37][plane], target)
        _assert_same(fixed[plane], fixed_snaps[48][plane], target)
        for off in (36, 38):
            assert not _same(auto[plane], snaps[off][plane], target), (plane, off)
    assert not auto["terminated"].any()


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_the_word_is_a_runtime_bound_clamped_below_at_zero(bundles, target):
    """Samples that never finish: a word of 100 runs exactly 100 steps (the
    bound is the word, not ``AUTO_K_MAX``: a runner may ask one launch for
    every step left), and a word of 0 or -5 runs none (every plane
    untouched)."""
    snaps = _oracle(bundles, bundles, "osc_xauto", "osc", target, never=True,
                    upto=100)
    for word, want in ((100, 100), (0, 0), (-5, 0)):
        bound, planes, _, extra, _ = _driver(bundles, "osc_xauto", "osc",
                                             target, never=True)
        extra["fused_steps"][0] = word
        bound.launch()
        got = _host(planes)
        for plane in ("x", "v", "t"):
            _assert_same(got[plane], snaps[want][plane], target,
                         err_msg=f"word {word}: {plane}")
        if word == 100:
            assert not _same(got["t"], snaps[64]["t"], target)


@pytest.mark.parametrize("which,name", [("osc", "osc_xauto"), ("spin3", "spin3_xauto")])
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_one_step_launch_is_the_single_step_bit_for_bit(bundles, which, name,
                                                          target):
    """A launch whose word is 1 runs the single step's straight-line body, so
    one step per launch of the auto artifact is the K = 1 kernel bit for bit,
    host AND device, after every launch until every sample finished (the
    device fused loop is held only to the ULP rule)."""
    auto = _one_step_snapshots(bundles, name, which, target)
    one = _one_step_snapshots(bundles, which, which, target)
    assert len(auto) == len(one) > 2
    for step, (got, want) in enumerate(zip(auto, one)):
        for plane in want:
            np.testing.assert_array_equal(got[plane], want[plane],
                                          err_msg=f"{name}/{target}: {plane} "
                                          f"after {step} steps")


def test_the_auto_entry_branches_once_on_the_word_between_the_step_and_the_loop():
    """Built, the auto TU reads the word once at entry and branches on it: the
    word-1 side is the single step's own body text (no loop), the other side
    holds the fused loop, each side opening with its own device marker; a
    fixed-K TU and the single step's TU carry no branch."""
    from hawk.artifact.bundle import _sources as built_sources
    from hawk.emit import FLOAT64

    auto = hawk.steps(osc, "auto")
    texts = built_sources(auto, name=auto.name, smode=FLOAT64, kind=auto.kind,
                          targets=("host", "cuda"), layout_sizes_override=None)
    plain = {s.backend: s for s in _sources(osc)}
    for target, src in texts.items():
        head, _, rest = src.text.partition("== 1) {")
        assert "lut_fused_steps[" in head.splitlines()[-1]
        one, _, many = rest.partition("} else {")
        assert 'asm volatile("// hawk: one step")' in one
        assert 'asm volatile("// hawk: fused steps")' in many
        assert "_step < " not in one and "_step < " in many
        assert plain[target].body in one
    for k in (osc, hawk.steps(osc, 16)):
        assert all("asm volatile" not in s.text for s in _sources(k))


# --------------------------------------------------------------------------- #
# -- the default: @kernel without steps= builds the auto kernel when auto
# -- admits it, under the author's name; everything else stays plain.
# --------------------------------------------------------------------------- #
def _renamed(k, name="k"):
    from hawk.emit import BACKENDS, render_source

    return [render_source(name, k.sinks, k.walk, b).text for b in BACKENDS.values()]


def test_the_default_is_the_auto_kernel_under_the_author_s_name():
    """No ``steps=``: the auto kernel (same IR and emission as
    ``hawk.steps(<plain>, "auto")`` apart from the name), named ``_osc_body``,
    carrying the plain step as ``.step``. Non-vacuity: its emission differs
    from ``steps=1``'s."""
    from hawk.trace.steps import AUTO_K_MAX

    plain = kernel(_osc_body, steps=1)
    for default in (kernel(_osc_body), kernel()(_osc_body)):
        assert default.name == "_osc_body"
        assert default.walk.finish == ("terminated", "auto")
        assert default.step is not None and default.step.walk.digest == plain.walk.digest
        assert set(default.arg_spec) - set(plain.arg_spec) == {("lookup", "fused_steps")}
        explicit = hawk.steps(plain, "auto")
        assert default.walk.digest == explicit.walk.digest
        assert _renamed(default) == _renamed(explicit)
        assert _renamed(default) != _renamed(plain)
        meta = sidecar_meta(default.name, default.walk, fmt="ptx",
                            scalar_type="float64")
        assert meta["kernel"] == "_osc_body"
        assert meta["finish"]["steps"] == "auto"
        assert meta["finish"]["steps_max"] == AUTO_K_MAX
    assert plain.step is None and hawk.steps(plain, 4).step is None
    assert kernel(steps="auto")(_osc_body).step is None


def _not_admitted():
    """Kernels auto does not admit, each beside the reason."""
    def no_finish(x: Scalar, y: Mutable[Scalar]):
        y = y + x

    def scatters(x: Scalar, lane: Index, terminated: Terminated,
                 acc: Accum[Scalar]):
        acc.add(x, at=lane)
        terminated = x > 1.0

    def scatters_and_steps(x: Scalar, lane: Index, terminated: Terminated,
                           acc: Accum[Scalar], y: Mutable[Scalar]):
        y_start = y
        acc.add(x, at=lane)
        y = y_start + x
        terminated = y_start > 1.0

    def word_named(fused_steps: Scalar, terminated: Terminated,
                   x: Mutable[Scalar]):
        x_start = x
        x = x_start + fused_steps
        terminated = x_start > 1.0

    def commits_nothing(x: Scalar, terminated: Terminated):
        terminated = x > 1.0

    return (no_finish, scatters, scatters_and_steps, word_named, commits_nothing)


def test_the_default_leaves_every_kernel_auto_does_not_admit_plain():
    """Byte identity: each non-admitted kernel's default build emits exactly
    ``steps=1``'s source and sidecar, and carries no ``.step``."""
    from hawk.trace.steps import admits_auto

    for fn in _not_admitted():
        default, plain = kernel(fn), kernel(fn, steps=1)
        assert not admits_auto(plain), fn.__name__
        assert default.step is None and default.name == fn.__name__
        assert default.walk.digest == plain.walk.digest
        assert _sources(default) == _sources(plain)
        assert (sidecar_meta(default.name, default.walk, fmt="ptx",
                             scalar_type="float64")
                == sidecar_meta(plain.name, plain.walk, fmt="ptx",
                                scalar_type="float64"))
    derived = hawk.steps(osc, 4)
    assert not admits_auto(derived)
    assert admits_auto(osc) and admits_auto(spin3)


def _finish_on_input(a: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    """A finishing step with no launch-start read: its derivative exists."""
    y = a * a
    terminated = a > 1.0


@pytest.mark.parametrize("transform", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_the_default_kernel_is_its_single_step_s(transform):
    """vjp/jvp of a default auto kernel delegate to ``.step``: the same IR as
    the plain kernel's derivative, tagged with the same primal name. The
    explicit ``steps="auto"`` kernel (``<name>_xauto``) keeps refusing — the
    non-vacuity: without the delegation the default kernel would refuse too."""
    from hawk.ir.walk import canonical

    default = kernel(_finish_on_input)
    plain = kernel(_finish_on_input, steps=1)
    assert default.step is not None
    got, want = transform(default, wrt=("a",)), transform(plain, wrt=("a",))
    assert got.primal == want.primal == "_finish_on_input"
    assert canonical(tuple(got)).digest == canonical(tuple(want)).digest
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        transform(hawk.steps(plain, "auto"), wrt=("a",))
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        transform(kernel(steps="auto")(_finish_on_input), wrt=("a",))
    # a step that reads its state before storing it refuses through its step, exactly as plain
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        transform(kernel(_osc_body), wrt=("x",))


def test_hawk_steps_of_the_default_kernel_derives_from_its_step():
    plain = kernel(_osc_body, steps=1)
    default = kernel(_osc_body)
    for k in (4, "auto"):
        assert (hawk.steps(default, k).walk.digest
                == hawk.steps(plain, k).walk.digest)
        assert hawk.steps(default, k).name == f"_osc_body_x{k}"


def test_the_hawk_host_runtime_takes_one_step_without_the_word(tmp_path_factory,
                                                               cache_dir):
    """``hawk.runtime.run`` of the default kernel with no ``fused_steps``:
    one step per launch (``t`` advances by ``dt`` per launch). Non-vacuity: a
    caller's word of 5 takes five."""
    from hawk import runtime

    default = kernel(_osc_body)
    bundle = build_bundle([default], tmp_path_factory.mktemp("default") / "b",
                          targets=("host",), cache_dir=cache_dir)
    sidecar = sidecar_of(bundle, "_osc_body")
    assert sidecar["finish"]["steps"] == "auto"
    host = runtime.load(bundle.directory, "_osc_body", sidecar)
    count_dtype = np.dtype(sidecar["arg_dtypes"]["finished_count"])

    def launches(count, **word):
        planes, _ = _planes("osc", np)
        planes["t_final"] = np.full(N, np.inf)
        for _ in range(count):
            runtime.run(host, dt=DT, terminated=np.zeros(N, dtype=bool),
                        finished_count=np.zeros(1, dtype=count_dtype),
                        **planes, **word)
        return planes

    np.testing.assert_array_equal(launches(3)["t"], 3 * DT)
    five = launches(1, fused_steps=np.full(1, 5, dtype=np.int64))
    np.testing.assert_array_equal(five["t"], 5 * DT)
