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
  reference after the same total, bit for bit (the reference is the ``K = 1``
  kernel on the host and the same auto artifact driven with ``k = 1`` on the
  device, which compiles the same loop body), plain and
  active-set kind (the map compacted by hand between launches). Non-vacuity: a launch of ``k = 37``
  equals 37 one-step launches and differs from 36 and 38, and the fixed
  ``steps=16`` artifact overshoots the same cap to 48.
"""

from __future__ import annotations

import math

import _deploy as L
import numpy as np
import pytest
from _unit_sidecar import finish_mask, sidecar_of_kernel
from conftest import _launch, _xp, sidecar_of

import hawk
from hawk import (
    Index,
    Mutable,
    Param,
    Scalar,
    Terminated,
    Vector,
    _ir,
    kernel,
)
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.errors import HawkError

N = 48
DT = 0.0625
CAP = 400


@kernel(steps=1)
def osc(
    dt: Param,
    t_final: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
    t: Mutable[Scalar],
):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def spin3(
    dt: Param,
    limit: Scalar,
    terminated: Terminated,
    r: Mutable[Vector[3]],
    s: Mutable[Scalar],
):
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
        assert (finish_mask(derived), derived.steps) == ("terminated", k)
        assert derived.arg_spec == osc.arg_spec
        meta = sidecar_of_kernel(derived,
            steps=derived.steps,
            fmt="ptx",
            scalar_type="float64",
        )
        assert meta["finish"] == {
            "mask": "terminated",
            "counter": "finished_count",
            "steps": k,
        }


def test_the_derived_kernel_keeps_the_compaction_group_of_its_step():
    """``hawk.steps(k, K)`` compacts by the group its single step declares, as the decorator form does."""

    @kernel(steps=1, group=32)
    def grouped(limit: Scalar, terminated: Terminated, s: Mutable[Scalar]):
        sn = s + 1.0
        s = sn
        terminated = sn >= limit

    assert grouped.group == 32
    for k in (*KS, "auto"):
        assert hawk.steps(grouped, k).group == 32


def test_a_kernel_without_a_finish_refuses():
    @kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = y + x

    with pytest.raises(HawkError, match=r"'plain'.*never finishes"):
        hawk.steps(plain, 4)


def test_a_reduce_or_out_sink_refuses():
    """A sum over samples beside a finish and a by-key store both trace and
    are refused here."""

    @kernel
    def reducing(x: Scalar, terminated: Terminated, total: Mutable[Scalar]):
        total = hawk.sum(x)
        terminated = x > 1.0

    with pytest.raises(HawkError, match=r"'reducing'.*not a step kernel"):
        hawk.steps(reducing, 4)

    @kernel
    def scattering(
        x: Scalar, lane: Index, terminated: Terminated, acc: Mutable[Scalar]
    ):
        hawk.scatter(acc)[lane] += x
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
def _osc_body(
    dt: Param,
    t_final: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
    t: Mutable[Scalar],
):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def test_the_decorator_form_reaches_a_defining_scope_local():
    t_cap = Scalar

    @kernel(steps=4)
    def local_osc(
        dt: Param, t_final: t_cap, terminated: Terminated, t: Mutable[Scalar]
    ):
        tn = t + dt
        t = tn
        terminated = tn >= t_final

    assert local_osc.name == "local_osc_x4"
    assert (finish_mask(local_osc), local_osc.steps) == ("terminated", 4)


def test_the_decorator_form_keeps_the_kind():
    from hawk.ext import Guard, Kind

    kind = Kind("steps_active", guard=Guard(active_set=True))
    decorated = kernel(kind=kind, steps=4)(_osc_body)
    assert decorated.kind is kind
    assert (
        _ir.digest(decorated.walk) == _ir.digest(hawk.steps(kernel(_osc_body, kind=kind), 4).walk)
    )


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

    def scattering(
        x: Scalar, lane: Index, terminated: Terminated, acc: Mutable[Scalar]
    ):
        hawk.scatter(acc)[lane] += x
        terminated = x > 1.0

    with pytest.raises(HawkError, match=r"'scattering'.*not a step kernel"):
        kernel(steps=4)(scattering)


def test_a_vjp_of_the_derived_kernel_refuses():
    with pytest.raises(HawkError, match=r"is read before it is assigned"):
        vjp(hawk.steps(osc, 4), wrt=("x",))


def test_a_jvp_of_the_derived_kernel_is_its_fused_step_jvp():
    """The forward mode carries through the loop: ``hawk.steps(step_jvp(osc), 4)``."""
    from hawk.diff import step_jvp

    assert jvp(hawk.steps(osc, 4)).name == hawk.steps(step_jvp(osc), 4).name


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
    kernels = (
        [osc, spin3]
        + [hawk.steps(k, n) for k in (osc, spin3) for n in KS]
        + [hawk.steps(osc, "auto"), hawk.steps(spin3, "auto")]
    )
    # Host under the EXACT profile: K fused steps against K = 1 bit for bit
    # (the default fast profile may fuse the two loop shapes differently).
    return build_bundle(
        kernels,
        root / "steps",
        targets=("cuda", "host"),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def active_bundle(tmp_path_factory, cache_dir):
    """The active-set kind's auto kernel: a bundle has ONE kind."""
    root = tmp_path_factory.mktemp("steps_active")
    return build_bundle(
        [osc_active_auto],
        root / "steps_active",
        targets=("cuda", "host"),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _planes(which, xp):
    rng = np.random.default_rng(11)
    if which == "osc":
        stop = rng.integers(5, 70, N)
        return {
            "t_final": xp.array(DT * stop.astype(float)),
            "x": xp.array(rng.uniform(-1, 1, N)),
            "v": xp.array(rng.uniform(-1, 1, N)),
            "t": xp.zeros(N),
        }, int(stop.max())
    r = rng.uniform(-1, 1, (3, N))
    return {
        "limit": xp.array(rng.uniform(0.2, 3.0, N)),
        "r": xp.array(r),
        "s": xp.zeros(N),
    }, None


def _run(bundle, name, which, target):
    xp = _xp(target)
    planes, stop = _planes(which, xp)
    planes["terminated"] = xp.zeros(N, dtype=bool)
    counter = xp.zeros(1, dtype=xp.uint32)
    bound = _launch(bundle, name, target, {"dt": DT, "finished_count": counter, **planes})
    launches = 0
    while int(counter[0]) < N and launches < CAP:
        bound.launch()
        launches += 1
    got = {
        k: np.asarray(a.get() if hasattr(a, "get") else a) for k, a in planes.items()
    }
    return got, int(counter[0]), launches, stop


@pytest.mark.parametrize("which", ["osc", "spin3"])
@pytest.mark.parametrize(
    "target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
)
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
            _assert_same(got[plane], ref, err_msg=f"{which}_x{k}/{target}:{plane}")


# --------------------------------------------------------------------------- #
# -- steps="auto": ONE kernel, the trip count a device word.
# --------------------------------------------------------------------------- #
def test_the_auto_kernel_is_named_and_records_auto_steps():
    from hawk.trace.steps import AUTO_K_MAX

    assert AUTO_K_MAX == 64
    for base in (osc, spin3):
        auto = hawk.steps(base, "auto")
        assert auto.name == f"{base.name}_xauto"
        assert (finish_mask(auto), auto.steps) == ("terminated", "auto")
        spec = set(auto.arg_spec)
        assert {("lookup", "fused_steps"), ("lookup", "finished_count")} <= spec
        assert spec - set(base.arg_spec) == {("lookup", "fused_steps")}
        assert _ir.slot_type(auto.walk, "lookup", "fused_steps").dtype == "i32"
        meta = sidecar_of_kernel(auto, steps=auto.steps, fmt="ptx", scalar_type="float64"
        )
        assert meta["finish"] == {
            "mask": "terminated",
            "counter": "finished_count",
            "steps": "auto",
            "steps_max": AUTO_K_MAX,
        }
        # the word is the runner's seam, not a cross-sample read of the body
        plain = sidecar_of_kernel(base, steps=base.steps, fmt="ptx", scalar_type="float64"
        )
        assert meta["exec_access"] == plain["exec_access"] == "sample_local"
        assert meta["arg_dtypes"]["fused_steps"] == "int64"
    fixed = sidecar_of_kernel(hawk.steps(osc, 16), steps=16, fmt="ptx", scalar_type="float64"
    )
    assert "steps_max" not in fixed["finish"]


def test_auto_refuses_what_hawk_steps_refuses():
    @kernel
    def plain(x: Scalar, y: Mutable[Scalar]):
        y = y + x

    with pytest.raises(HawkError, match=r"'plain'.*never finishes"):
        hawk.steps(plain, "auto")

    def scattering(
        x: Scalar, lane: Index, terminated: Terminated, acc: Mutable[Scalar]
    ):
        hawk.scatter(acc)[lane] += x
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


def _bind(bundle, name, target, planes, active):
    """The plan bound over ``planes``. The active set of a device row binds through eagle's plan; a host kernel over an active
    set runs under a runner that makes it (:func:`test_a_runner_over_an_active_set_takes_exactly_its_cap_in_steps`)."""
    if not active:
        return _launch(bundle, name, target, planes)
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = (L.host_plugin(bundle.directory, name, sidecar_of(bundle, name)) if target == "host"
              else L.device_plugin(bundle.directory, name, sidecar_of(bundle, name)))
    structure = eexec.HostTeam if target == "host" else eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(**planes)


def _driver(bundle, name, which, target, *, active=False, never=False):
    """``(bound, planes, counter, word, xp, active_planes)`` of one run."""
    sidecar = sidecar_of(bundle, name)
    xp = _xp(target)
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
        extra["active_count"] = xp.full(1, N, dtype=xp.uint32)
    bound = _bind(bundle, name, target, {"dt": DT, "finished_count": counter, **planes, **extra}, active)
    return bound, planes, counter, extra, xp


def _host(planes):
    return {
        k: np.array(a.get() if hasattr(a, "get") else a, copy=True)
        for k, a in planes.items()
    }


def _one_step_snapshots(
    bundle, name, which, target, *, never=False, upto=None, active=False
):
    """One step per launch, the planes after every launch: ``snaps[s]`` =
    after ``s`` steps (``snaps[0]`` the initial planes), until all finished or
    ``upto``. ``name`` is the K = 1 kernel, or an auto kernel driven with
    ``fused_steps = 1`` (the device oracle: the same compiled loop body)."""
    bound, planes, counter, extra, xp = _driver(
        bundle, name, which, target, never=never, active=active
    )
    snaps = [_host(planes)]
    while int(counter[0]) < N if upto is None else len(snaps) <= upto:
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
    straight-line body, i.e. the K = 1 kernel. A fused launch equals it bit
    for bit (:func:`_same`)."""
    return _one_step_snapshots(bundle, name, which, target, **kw)


def _same(got, want) -> bool:
    """``got == want`` bit for bit, shape included."""
    got, want = np.asarray(got), np.asarray(want)
    return got.shape == want.shape and bool(np.array_equal(got, want))


def _assert_same(got, want, err_msg=""):
    np.testing.assert_array_equal(got, want, err_msg=err_msg)


def _compact(extra, planes, xp):
    """The active-set map compacted by hand: the ascending live indices."""
    live = np.flatnonzero(~_host({"m": planes["terminated"]})["m"]).astype(np.int32)
    extra["active_map"][: live.size] = xp.asarray(live)
    extra["active_count"][0] = live.size


def _word_rows():
    """The rows of the word-driven test: each kernel on the host and on the device, except the active-set kernel on the host,
    whose set is the runner's and whose row is the runner test below."""
    for which, name, active, label in (
        ("osc", "osc_xauto", False, "osc"),
        ("spin3", "spin3_xauto", False, "spin3"),
        ("osc", "_osc_body_xauto", True, "osc-active-set"),
    ):
        if not active:
            yield pytest.param(which, name, active, "host", id=f"{label}-host")
        yield pytest.param(which, name, active, "cuda", marks=pytest.mark.gpu, id=f"{label}-cuda")


@pytest.mark.parametrize("which,name,active,target", list(_word_rows()))
def test_auto_takes_exactly_the_word_s_steps_per_launch(
    bundles, active_bundle, which, name, active, target
):
    """After every launch of the ``K_SEQ`` cycle every plane, the mask and the
    counter equal the step-by-step reference (:func:`_oracle`) after the same
    total number of steps."""
    bundle = active_bundle if active else bundles
    snaps = _oracle(bundles, bundle, name, which, target, active=active)
    bound, planes, counter, extra, xp = _driver(
        bundle, name, which, target, active=active
    )
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
                got[plane], want, err_msg=f"{name}/{target}: {plane} after "
                f"{launches} launches ({total} steps)")
        if active:
            _compact(extra, planes, xp)
        assert launches < CAP
    assert got["terminated"].all() and total >= len(snaps) - 1


def _runner_snapshots(plan_of, upto):
    """``snaps[s]`` = the planes after ``s`` steps, from a runner that takes one step per run, until every sample is done or
    ``upto`` steps."""
    planes, _ = _planes("osc", np)
    planes |= {"dt": DT}
    runner = hawk.until_done(plan_of(), max_steps=1, **planes)
    snaps = [_host({k: v for k, v in planes.items() if k != "dt"})]
    while len(snaps) <= upto:
        report = runner.run()
        snaps.append(_host({k: v for k, v in planes.items() if k != "dt"}))
        if report.done:
            break
    runner.close()
    return snaps


def test_a_runner_over_an_active_set_takes_exactly_its_cap_in_steps(active_bundle):
    """The active-set kernel under a runner: after a cap of ``s`` steps (the running totals of the ``K_SEQ`` cycle) every
    plane equals the one-step-per-run reference after ``s`` steps, bit for bit, and a cap below the last step is taken
    exactly; a cap past it leaves the finished state."""
    kernel = hawk.load(active_bundle.directory, "_osc_body_xauto")
    snaps = _runner_snapshots(lambda: hawk.plan(kernel), CAP)
    total = 0
    for k in K_SEQ:
        total += k
        planes, _ = _planes("osc", np)
        report = hawk.until_done(hawk.plan(kernel), max_steps=total, dt=DT, **planes).run()
        got = _host(planes)
        for plane, want in snaps[min(total, len(snaps) - 1)].items():
            _assert_same(got[plane], want, err_msg=f"{plane} after a cap of {total} steps")
        if total < len(snaps) - 1:
            assert report.steps == total and not report.done
        else:
            assert report.done


@pytest.mark.parametrize(
    "target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_auto_caps_exactly_where_the_fixed_artifact_overshoots(bundles, target):
    """Samples that never finish: launches of 16, 16, 5 leave every sample at
    exactly step 37 (equal to 37 one-step launches, unlike 36 or 38); the
    fixed ``steps=16`` artifact run to the same cap lands on 48."""
    snaps = _oracle(bundles, bundles, "osc_xauto", "osc", target, never=True, upto=48)
    fixed_snaps = _one_step_snapshots(
        bundles, "osc", "osc", target, never=True, upto=48
    )
    bound, planes, _, extra, _ = _driver(
        bundles, "osc_xauto", "osc", target, never=True
    )
    for k in (16, 16, 5):
        extra["fused_steps"][0] = k
        bound.launch()
    auto = _host(planes)
    fixed_bound, fixed_planes, _, _, _ = _driver(
        bundles, "osc_x16", "osc", target, never=True
    )
    for _ in range(math.ceil(37 / 16)):
        fixed_bound.launch()
    fixed = _host(fixed_planes)
    for plane in ("x", "v", "t"):
        _assert_same(auto[plane], snaps[37][plane])
        _assert_same(fixed[plane], fixed_snaps[48][plane])
        for off in (36, 38):
            assert not _same(auto[plane], snaps[off][plane]), (plane, off)
    assert not auto["terminated"].any()


@pytest.mark.parametrize(
    "target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_the_word_is_a_runtime_bound_clamped_below_at_zero(bundles, target):
    """Samples that never finish: a word of 100 runs exactly 100 steps (the
    bound is the word, not ``AUTO_K_MAX``: a runner may ask one launch for
    every step left), and a word of 0 or -5 runs none (every plane
    untouched)."""
    snaps = _oracle(bundles, bundles, "osc_xauto", "osc", target, never=True, upto=100)
    for word, want in ((100, 100), (0, 0), (-5, 0)):
        bound, planes, _, extra, _ = _driver(
            bundles, "osc_xauto", "osc", target, never=True
        )
        extra["fused_steps"][0] = word
        bound.launch()
        got = _host(planes)
        for plane in ("x", "v", "t"):
            _assert_same(got[plane], snaps[want][plane],
                         err_msg=f"word {word}: {plane}")
        if word == 100:
            assert not _same(got["t"], snaps[64]["t"])


@pytest.mark.parametrize("which,name", [("osc", "osc_xauto"), ("spin3", "spin3_xauto")])
@pytest.mark.parametrize(
    "target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_a_one_step_launch_is_the_single_step_bit_for_bit(bundles, which, name, target):
    """A launch whose word is 1 runs the single step's straight-line body, so
    one step per launch of the auto artifact is the K = 1 kernel bit for bit,
    after every launch until every sample finished."""
    auto = _one_step_snapshots(bundles, name, which, target)
    one = _one_step_snapshots(bundles, which, which, target)
    assert len(auto) == len(one) > 2
    for step, (got, want) in enumerate(zip(auto, one)):
        for plane in want:
            _assert_same(got[plane], want[plane],
                         err_msg=f"{name}/{target}: {plane} after {step} steps")




# --------------------------------------------------------------------------- #
# -- the default: @kernel without steps= builds the auto kernel when auto
# -- admits it, under the author's name; everything else stays plain.
# --------------------------------------------------------------------------- #
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
    default = kernel(_finish_on_input)
    plain = kernel(_finish_on_input, steps=1)
    assert default.step is not None
    got, want = transform(default, wrt=("a",)), transform(plain, wrt=("a",))
    assert got.primal == want.primal == "_finish_on_input"
    assert got[0].ctx.dump(list(got)) == want[0].ctx.dump(list(want))
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
        assert _ir.digest(hawk.steps(default, k).walk) == _ir.digest(hawk.steps(plain, k).walk)
        assert hawk.steps(default, k).name == f"_osc_body_x{k}"


def test_the_hawk_host_runtime_takes_one_step_without_the_word(
    tmp_path_factory, cache_dir
):
    """``hawk.runtime.run`` of the default kernel with no ``fused_steps``:
    one step per launch (``t`` advances by ``dt`` per launch). Non-vacuity: a
    caller's word of 5 takes five."""
    from hawk import runtime

    default = kernel(_osc_body)
    bundle = build_bundle(
        [default],
        tmp_path_factory.mktemp("default") / "b",
        targets=("host",),
        cache_dir=cache_dir,
    )
    sidecar = sidecar_of(bundle, "_osc_body")
    assert sidecar["finish"]["steps"] == "auto"
    host = runtime.load(bundle.directory, "_osc_body", sidecar)
    count_dtype = np.dtype(sidecar["arg_dtypes"]["finished_count"])

    def launches(count, **word):
        planes, _ = _planes("osc", np)
        planes["t_final"] = np.full(N, np.inf)
        for _ in range(count):
            runtime.run(
                host,
                dt=DT,
                terminated=np.zeros(N, dtype=bool),
                finished_count=np.zeros(1, dtype=count_dtype),
                **planes,
                **word,
            )
        return planes

    np.testing.assert_array_equal(launches(3)["t"], 3 * DT)
    five = launches(1, fused_steps=np.full(1, 5, dtype=np.int64))
    np.testing.assert_array_equal(five["t"], 5 * DT)
