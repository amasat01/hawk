# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host build vectorises; the exact profiles move no bit, the fast one
stays inside a derived error bound.

Four observations, each with a known answer:

* **Bit identity.** A guarded RK4 step (the performance card's oscillator,
  1003 samples so the vector remainder runs, a random ``terminated`` mask)
  built under BOTH exact host profiles (``native``, ``portable``) and stepped
  40 times equals, bit for bit, the same arithmetic done by NumPy (one IEEE
  operation per ufunc, no contraction); terminated samples keep their bytes.
* **The fast profile's bound.** The same step under ``native-vector-math``
  (FMA contraction on) differs from that reference, and by no more than an
  error bound derived from the step's own operation count and the step
  map's propagation (see :func:`_fast_profile_bound`).
* **It is SIMD.** On a CPU with AVX-512 the ``native`` artifact's entry
  carries packed float64 arithmetic (``objdump``); without AVX-512 the
  generated loop is still the vectorisable form (pragma, ``offset_t`` index,
  byte mask), which is what the row then pins.
* **The cache tells the profiles and the CPUs apart.** The same source
  under ``native`` and ``portable`` lands on two keys, and the ``native`` key
  names the target CPU, so a shared cache directory cannot serve one CPU's
  ``-march=native`` binary to another.

This test previously failed via three plants, all then removed:

* ``host_codegen_flags`` with ``-ffp-contract=off`` dropped -- the native
  build contracts to FMA and ``test_the_exact_profiles_are_bit_identical_to_numpy``
  fails on x/v;
* ``HostBackend.unpack``'s mask rebinding skipped -- the loop stays scalar
  (``packed float64 ops: 0``);
* ``_host_key_terms`` without the ``host_target`` term --
  ``test_the_native_key_names_the_target_cpu`` fails.

Three more, for the lane-wide guard and the active-set loops:

* ``HAWK_HOST_LANE_GUARDS`` forced to 0 (the byte guard back in the loop) --
  ``test_the_guard_is_lane_wide_and_nothing_spills`` fails at a 5960-byte
  frame;
* ``lane_guards`` answering ``()`` -- the same row fails on the source;
* the identity scan dropped (only the tile's two ends tested) --
  ``test_an_active_set_build_is_bit_identical_on_every_map`` fails on
  ``native/ends_only``.
"""

from __future__ import annotations

import platform
import re
import subprocess

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.artifact import build_bundle
from hawk.compile import (
    HOST,
    OPT_LEVELS,
    Cache,
    CompileOptions,
    compile_source,
    host_codegen_flags,
    host_profile,
    opt_level,
)
from hawk.compile import toolchain as tc
from hawk.ext import Guard, Kind

N = 1003
STEPS = 40
DT = 0.01


@hawk.kernel
def osc_step(omega: Scalar, zeta: Scalar, dt: Param, terminated: Terminated,
             x: Mutable[Scalar], v: Mutable[Scalar], k: Mutable[Scalar]):
    x0 = x
    v0 = v
    w2 = omega * omega
    c = 2.0 * zeta * omega
    h2 = 0.5 * dt
    h6 = dt * 0.16666666666666666
    a1 = -w2 * x0 - c * v0
    x2 = x0 + h2 * v0
    v2 = v0 + h2 * a1
    a2 = -w2 * x2 - c * v2
    x3 = x0 + h2 * v2
    v3 = v0 + h2 * a2
    a3 = -w2 * x3 - c * v3
    x4 = x0 + dt * v3
    v4 = v0 + dt * a3
    a4 = -w2 * x4 - c * v4
    x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
    v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    k = k + 1.0


def _numpy_step(omega, zeta, dt, x0, v0):
    """The same operations, in the same order, one IEEE op per ufunc."""
    w2 = omega * omega
    c = 2.0 * zeta * omega
    h2 = 0.5 * dt
    h6 = dt * 0.16666666666666666
    a1 = -w2 * x0 - c * v0
    x2 = x0 + h2 * v0
    v2 = v0 + h2 * a1
    a2 = -w2 * x2 - c * v2
    x3 = x0 + h2 * v2
    v3 = v0 + h2 * a2
    a3 = -w2 * x3 - c * v3
    x4 = x0 + dt * v3
    v4 = v0 + dt * a3
    a4 = -w2 * x4 - c * v4
    return (x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4),
            v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4))


def _inputs():
    rng = np.random.default_rng(20261002)
    return dict(omega=rng.uniform(1.0, 10.0, N), zeta=rng.uniform(0.01, 0.2, N),
                x=rng.uniform(-1.0, 1.0, N), v=rng.uniform(-1.0, 1.0, N),
                terminated=rng.random(N) < 0.3)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("simd_cache"))
    return {p: build_bundle([osc_step], tmp_path_factory.mktemp(f"simd_{p}"),
                            targets=("host",), cache_dir=cache, host_profile=p)
            for p in tc.HOST_PROFILES}


def _run(bundle):
    import eagle.exec as eexec
    from eagle import plan as eplan

    inp = _inputs()
    x, v, k = inp["x"].copy(), inp["v"].copy(), np.zeros(N)
    plugin = L.host_plugin(bundle.directory, "osc_step", sidecar_of(bundle, "osc_step"))
    step = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        omega=inp["omega"], zeta=inp["zeta"], dt=DT, terminated=inp["terminated"],
        x=x, v=v, k=k)
    for _ in range(STEPS):
        step.launch()
    return x, v, k


def _packed_double_ops(so, symbol):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)],
                         capture_output=True, text=True, check=True).stdout
    body = re.search(rf"<{re.escape(symbol)}>:\n(.*?)(?:\n\n|\Z)", out, re.S)
    assert body, f"{symbol} not found in {so}"
    return len(re.findall(r"\bv?(?:add|sub|mul|div|sqrt)pd\b", body.group(1)))


def test_the_exact_profiles_are_bit_identical_to_numpy(bundles):
    inp = _inputs()
    live = ~inp["terminated"]
    x, v = inp["x"].copy(), inp["v"].copy()
    for _ in range(STEPS):
        nx, nv = _numpy_step(inp["omega"], inp["zeta"], DT, x, v)
        x, v = np.where(live, nx, x), np.where(live, nv, v)
    for profile in tc.EXACT_HOST_PROFILES:
        hx, hv, hk = _run(bundles[profile])
        assert hx.tobytes() == x.tobytes(), f"{profile}: x is not bit-identical"
        assert hv.tobytes() == v.tobytes(), f"{profile}: v is not bit-identical"
        assert np.array_equal(hk, np.where(live, float(STEPS), 0.0)), profile
        assert hx[~live].tobytes() == inp["x"][~live].tobytes(), profile


#: Unit roundoff of float64.
U64 = 2.0 ** -53
#: The longest chain of dependent IEEE operations from the step's inputs to
#: either output, counted on ``osc_step``: a1 (mul, sub: 2), v2 (mul, add: 4),
#: a2 (6), v3 (8), a3 (10), v4 (12), a4 (14), then the weighted sum
#: (a1 + 2a2: 8, + 2a3: 12, + a4: 15), the ``h6`` product (16) and ``v0 +``
#: (17); ``x`` is shorter.
DEPTH = 17


def _fast_profile_bound(inp, live):
    """Per-sample bound on ``|fast - exact|`` after STEPS steps, derived (not
    fitted): Higham's forward bound gives each output of one step an error of
    at most ``gamma_n * M`` against the exact real-number step, n = DEPTH and
    M = the same expression evaluated on absolute values; that holds for BOTH
    builds (an FMA rounds once, two plain ops twice), so the two builds'
    outputs of one step from the same input differ by at most
    ``delta = 2 gamma_n M``. The step is linear, ``s' = A s`` with A the RK4
    matrix of the damped oscillator, so a difference e_k entering step k
    leaves step STEPS as ``A^(STEPS-k) e_k``: the total is at most
    ``sum_k ||A^(STEPS-k)||_inf delta_k <= STEPS * max_j ||A^j||_inf *
    max_k delta_k``, with M taken over the reference trajectory."""
    gamma = DEPTH * U64 / (1.0 - DEPTH * U64)
    omega, zeta = inp["omega"], inp["zeta"]
    w2, c = omega * omega, 2.0 * zeta * omega
    h2, h6 = 0.5 * DT, DT * 0.16666666666666666
    # The RK4 step matrix, per sample: A = I + dt J + (dt J)^2/2 + ... + (dt J)^4/24.
    J = np.zeros((N, 2, 2))
    J[:, 0, 1] = 1.0
    J[:, 1, 0] = -w2
    J[:, 1, 1] = -c
    hJ = DT * J
    A = np.broadcast_to(np.eye(2), (N, 2, 2)).copy()
    term = A.copy()
    for k in range(1, 5):
        term = term @ hJ / k
        A = A + term
    power, worst_power = np.broadcast_to(np.eye(2), (N, 2, 2)).copy(), np.ones(N)
    for _ in range(STEPS):
        power = power @ A
        worst_power = np.maximum(worst_power, np.abs(power).sum(axis=2).max(axis=1))
    # M: the step evaluated on absolute values, over the reference trajectory.
    x, v = inp["x"].copy(), inp["v"].copy()
    m = np.zeros(N)
    for _ in range(STEPS):
        ax, av = np.abs(x), np.abs(v)
        a1 = w2 * ax + c * av
        v2 = av + h2 * a1
        a2 = w2 * (ax + h2 * av) + c * v2
        v3 = av + h2 * a2
        a3 = w2 * (ax + h2 * v2) + c * v3
        v4 = av + DT * a3
        a4 = w2 * (ax + DT * v3) + c * v4
        mx = ax + h6 * (av + 2.0 * v2 + 2.0 * v3 + v4)
        mv = av + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        m = np.maximum(m, np.maximum(mx, mv))
        nx, nv = _numpy_step(omega, zeta, DT, x, v)
        x, v = np.where(live, nx, x), np.where(live, nv, v)
    return STEPS * worst_power * 2.0 * gamma * m


def test_the_fast_profile_is_within_its_derived_bound(bundles):
    """``native-vector-math`` contracts (FMA): no longer bit-identical, but
    every sample within :func:`_fast_profile_bound`; terminated samples still
    keep their bytes (no arithmetic touches them)."""
    if "native-vector-math" not in bundles or platform.machine().lower() not in (
            "x86_64", "amd64"):
        pytest.skip("the fast profile is x86-64 only")
    inp = _inputs()
    live = ~inp["terminated"]
    x, v = inp["x"].copy(), inp["v"].copy()
    for _ in range(STEPS):
        nx, nv = _numpy_step(inp["omega"], inp["zeta"], DT, x, v)
        x, v = np.where(live, nx, x), np.where(live, nv, v)
    hx, hv, hk = _run(bundles["native-vector-math"])
    bound = _fast_profile_bound(inp, live)
    for name, got, want in (("x", hx, x), ("v", hv, v)):
        err = np.abs(got - want)
        worst = int(np.argmax(err / bound))
        assert np.all(err <= bound), (
            f"{name}: |fast - exact| = {err[worst]!r} > derived bound "
            f"{bound[worst]!r} at sample {worst}")
    assert hx[~live].tobytes() == inp["x"][~live].tobytes()
    assert hv[~live].tobytes() == inp["v"][~live].tobytes()
    assert np.array_equal(hk, np.where(live, float(STEPS), 0.0))
    cpu = open("/proc/cpuinfo").read() if __import__("os").path.exists(
        "/proc/cpuinfo") else ""
    if re.search(r"fma", cpu) and tc.host_compiler().endswith("g++"):
        # Non-vacuity: with hardware FMA the fast build really contracts.
        assert hx.tobytes() != x.tobytes() or hv.tobytes() != v.tobytes(), (
            "the fast profile is bit-identical: contraction did not happen")


def test_the_native_build_is_simd(bundles):
    cpp = (bundles["native"].directory / "osc_step.cpp").read_text()
    assert "#pragma GCC ivdep" in cpp
    assert "for (aether::offset_t hawk_j" in cpp
    assert "hawk_abi::HostMask trm_terminated" in cpp
    flags = open("/proc/cpuinfo").read() if __import__("os").path.exists(
        "/proc/cpuinfo") else ""
    packed = _packed_double_ops(bundles["native"].directory / "osc_step.so",
                                "osc_step_host")
    if re.search(r"\bavx512f\b", flags) and tc.host_compiler().endswith("g++"):
        assert packed > 0, f"packed float64 ops: {packed}"


def _stack_frame(so, symbol):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)],
                         capture_output=True, text=True, check=True).stdout
    body = re.search(rf"<{re.escape(symbol)}>:\n(.*?)(?:\n\n|\Z)", out, re.S)
    assert body, f"{symbol} not found in {so}"
    frame = re.search(r"\bsub\s+\$0x([0-9a-f]+),%rsp", body.group(1))
    return int(frame.group(1), 16) if frame else 0


def test_the_guard_is_lane_wide_and_nothing_spills(bundles):
    """A byte guard made GCC run 64 samples per vector (eight zmm per float64
    value) and spill: a 5960-byte frame on the RK4 step. The
    lane-guarded loop reads one float64 guard per sample instead; its frame is
    the 512-byte guard block plus a few saves."""
    cpp = (bundles["native"].directory / "osc_step.cpp").read_text()
    assert "#if HAWK_HOST_LANE_GUARDS" in cpp
    assert "const hawk_abi::HostLaneMask trm_terminated = hawk_lane0;" in cpp
    flags = open("/proc/cpuinfo").read() if __import__("os").path.exists(
        "/proc/cpuinfo") else ""
    if re.search(r"\bavx512f\b", flags) and tc.host_compiler().endswith("g++"):
        frame = _stack_frame(bundles["native"].directory / "osc_step.so",
                             "osc_step_host")
        assert frame < 1024, f"stack frame {frame} B: the loop spills"


ACTIVE = Kind("active_set", guard=Guard(active_set=True))


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("simd_map_cache"))
    return {p: build_bundle([osc_step], tmp_path_factory.mktemp(f"simd_map_{p}"),
                            targets=("host",), cache_dir=cache, host_profile=p,
                            kind=ACTIVE)
            for p in tc.EXACT_HOST_PROFILES}


def _maps():
    """Three maps over the same live set size: the identity prefix (eagle's
    map right after a physical reorder: the contiguous loop), an ascending
    scattered map (the map loop), and a map equal to the identity at both ends
    of the tile but not in between (must NOT take the contiguous loop)."""
    rng = np.random.default_rng(7)
    live = 700
    scattered = np.sort(rng.choice(N, live, replace=False)).astype(np.int32)
    tricky = np.arange(live, dtype=np.int32)
    tricky[live // 2] = N - 1
    return {"identity": (np.arange(live, dtype=np.int32), live),
            "scattered": (scattered, live), "ends_only": (tricky, live)}


def _run_mapped(bundle, positions, live):
    import eagle.exec as eexec
    from eagle import ActiveSet
    from eagle import plan as eplan

    inp = _inputs()
    x, v, k = inp["x"].copy(), inp["v"].copy(), np.zeros(N)
    terminated = inp["terminated"].copy()
    aset = ActiveSet(terminated)
    aset.map[:] = 0
    aset.map[:live] = positions
    aset.count[0] = live
    plugin = L.host_plugin(bundle.directory, "osc_step", sidecar_of(bundle, "osc_step"))
    step = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        omega=inp["omega"], zeta=inp["zeta"], dt=DT, terminated=terminated,
        x=x, v=v, k=k, **aset.planes())
    for _ in range(STEPS):
        step.launch()
    return x, v, k


def test_an_active_set_build_is_bit_identical_on_every_map(mapped):
    inp = _inputs()
    for label, (positions, live) in _maps().items():
        stepped = np.zeros(N, dtype=bool)
        stepped[positions[:live]] = True
        stepped &= ~inp["terminated"]
        x, v = inp["x"].copy(), inp["v"].copy()
        for _ in range(STEPS):
            nx, nv = _numpy_step(inp["omega"], inp["zeta"], DT, x, v)
            x, v = np.where(stepped, nx, x), np.where(stepped, nv, v)
        for profile in tc.EXACT_HOST_PROFILES:
            hx, hv, hk = _run_mapped(mapped[profile], positions, live)
            where = f"{profile}/{label}"
            assert hx.tobytes() == x.tobytes(), f"{where}: x is not bit-identical"
            assert hv.tobytes() == v.tobytes(), f"{where}: v is not bit-identical"
            assert np.array_equal(hk, np.where(stepped, float(STEPS), 0.0)), where


def test_the_identity_map_takes_the_vector_loop(mapped):
    cpp = (mapped["native"].directory / "osc_step.cpp").read_text()
    assert "auto hawk_sample = [&]" in cpp and "hawk_strays |=" in cpp
    flags = open("/proc/cpuinfo").read() if __import__("os").path.exists(
        "/proc/cpuinfo") else ""
    packed = _packed_double_ops(mapped["native"].directory / "osc_step.so",
                                "osc_step_host")
    if re.search(r"\bavx512f\b", flags) and tc.host_compiler().endswith("g++"):
        assert packed > 0, f"packed float64 ops: {packed}"


def test_a_cross_sample_kernel_keeps_the_plain_loop(tmp_path):
    from _kernels import gather

    bundle = build_bundle([gather], tmp_path / "g", targets=("host",),
                          cache_dir=str(tmp_path / "c"))
    cpp = (bundle.directory / "gather.cpp").read_text()
    assert "ivdep" not in cpp and "for (std::int64_t hawk_i = base;" in cpp


def test_profile_flags_keep_the_bit_identity_guard():
    """The exact profiles keep contraction off; the fast one turns it on;
    none of the three reassociates or enables fast-math."""
    for profile in tc.HOST_PROFILES:
        flags = host_codegen_flags(profile)
        want = ("-ffp-contract=off" if profile in tc.EXACT_HOST_PROFILES
                else "-ffp-contract=fast")
        assert "-O3" in flags and want in flags, flags
        assert not any("fast-math" in f or "Ofast" in f or "reassoc" in f
                       for f in flags), flags
    assert set(tc.EXACT_HOST_PROFILES) == {"native", "portable"}
    assert "-march=native" not in host_codegen_flags("portable")


def test_the_profile_comes_from_the_options_then_the_environment(monkeypatch):
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    assert host_profile() == tc.default_host_profile()
    monkeypatch.setenv("HAWK_HOST_PROFILE", "portable")
    assert host_profile() == "portable"
    assert host_profile("native") == "native"
    monkeypatch.setenv("HAWK_HOST_PROFILE", "fastest")
    with pytest.raises(hawk.HawkError, match="unknown host profile"):
        host_profile()


SOURCE = 'extern "C" int hawk_simd_probe(int a) { return a + 1; }\n'


def test_the_two_profiles_never_share_a_slot(tmp_path):
    keys = {p: compile_source(SOURCE, "probe", CompileOptions(
        backend=HOST, cache_dir=str(tmp_path), host_profile=p)).key
        for p in tc.HOST_PROFILES}
    assert len(set(keys.values())) == len(keys), keys


def test_a_cache_filled_under_the_old_default_is_not_served(tmp_path, monkeypatch):
    """A cache populated before the default became ``native-vector-math``
    (every default-profile entry was then ``native``) must miss under the new
    default, and the explicit ``native`` request must still hit it."""
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    from hawk.compile import _reset_artifact_memo as reset_artifact_memo
    old = compile_source(SOURCE, "probe", CompileOptions(
        backend=HOST, cache_dir=str(tmp_path), host_profile="native"))
    reset_artifact_memo()
    new = compile_source(SOURCE, "probe", CompileOptions(
        backend=HOST, cache_dir=str(tmp_path)))
    if tc.default_host_profile() == "native":
        pytest.skip("the default is native here (architecture or compiler)")
    assert new.key != old.key and not new.hit
    reset_artifact_memo()
    again = compile_source(SOURCE, "probe", CompileOptions(
        backend=HOST, cache_dir=str(tmp_path), host_profile="native"))
    assert again.key == old.key and again.hit


@pytest.mark.parametrize("profile", ["native", "native-vector-math"])
def test_the_native_key_names_the_target_cpu(tmp_path, monkeypatch, profile):
    if profile == "native-vector-math" and platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("native-vector-math is x86-64 only")
    opts = CompileOptions(backend=HOST, cache_dir=str(tmp_path), host_profile=profile)
    first = compile_source(SOURCE, "probe", opts)
    record = Cache(str(tmp_path)).record(first.key)
    assert f"host_target:{tc.host_target_identity()}" in record["key_terms"]
    # Another CPU (a different identity) must miss, not hit.
    monkeypatch.setattr(tc, "host_target_identity", lambda: "another-cpu")
    from hawk.compile import _reset_artifact_memo as reset_artifact_memo
    reset_artifact_memo()
    second = compile_source(SOURCE, "probe", opts)
    assert second.key != first.key and not second.hit


def test_the_vector_math_profile_is_the_x86_default(monkeypatch):
    """`native-vector-math` adds aether's vector math (and lets GCC
    if-convert) and is the default on x86-64; `native` keeps libm's flags and
    stays the default on every other architecture."""
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    x86 = platform.machine().lower() in ("x86_64", "amd64")
    assert host_profile() == ("native-vector-math" if x86 else "native")
    default = tc.host_codegen_flags("native")
    assert "-DAETHER_HOST_VECTOR_MATH" not in default
    assert "-fno-trapping-math" not in default
    if platform.machine().lower() in ("x86_64", "amd64"):
        vm = tc.host_codegen_flags("native-vector-math")
        assert default[:2] == ["-O3", "-ffp-contract=off"]
        assert vm == ["-O3", "-ffp-contract=fast", *default[2:],
                      "-fno-trapping-math", "-DAETHER_HOST_VECTOR_MATH"]
        monkeypatch.setenv("HAWK_HOST_PROFILE", "native-vector-math")
        assert host_profile() == "native-vector-math"


# --------------------------------------------------------------------------- #
# opt_level: a user-controllable optimisation level, shaped exactly like
# host_profile (its own resolution order, its own key term, its own refusal).
# --------------------------------------------------------------------------- #
def test_the_opt_level_comes_from_the_options_then_the_environment(monkeypatch):
    monkeypatch.delenv("HAWK_OPT_LEVEL", raising=False)
    assert opt_level() == tc.DEFAULT_OPT_LEVEL == "O3"
    monkeypatch.setenv("HAWK_OPT_LEVEL", "O0")
    assert opt_level() == "O0"
    assert opt_level("O2") == "O2"            # the compile options win over the env
    monkeypatch.setenv("HAWK_OPT_LEVEL", "Ofast")
    with pytest.raises(hawk.HawkError, match="unknown opt level"):
        opt_level()


def test_host_codegen_flags_carries_the_opt_level_in_place_of_the_hardcoded_O3():
    """The opt level replaces what used to be a hard-coded ``-O3``; nothing
    else in the profile's recipe moves."""
    default = host_codegen_flags("native")
    assert default[0] == "-O3"
    o0 = host_codegen_flags("native", "O0")
    assert o0[0] == "-O0" and o0[1:] == default[1:]
    o1 = host_codegen_flags("native", "O1")
    assert o1[0] == "-O1" and o1[1:] == default[1:]


def test_host_codegen_flags_rejects_an_unknown_opt_level():
    with pytest.raises(hawk.HawkError, match="unknown opt level"):
        host_codegen_flags("native", "Ofast")


def test_the_opt_levels_never_share_a_slot(tmp_path):
    keys = {o: compile_source(SOURCE, "probe", CompileOptions(
        backend=HOST, cache_dir=str(tmp_path), opt_level=o)).key
        for o in OPT_LEVELS}
    assert len(set(keys.values())) == len(keys), keys


def test_an_opt_level_change_is_a_key_change_and_therefore_a_miss(tmp_path):
    from hawk.compile import _reset_artifact_memo as reset_artifact_memo

    opts = dict(backend=HOST, cache_dir=str(tmp_path))
    o3 = compile_source(SOURCE, "probe", CompileOptions(**opts, opt_level="O3"))
    reset_artifact_memo()
    o0 = compile_source(SOURCE, "probe", CompileOptions(**opts, opt_level="O0"))
    assert o0.key != o3.key and not o0.hit
    # the SAME level, a fresh lookup, still hits -- this is a key change, not
    # a cache that forgot how to hit at all.
    reset_artifact_memo()
    again = compile_source(SOURCE, "probe", CompileOptions(**opts, opt_level="O3"))
    assert again.key == o3.key and again.hit


def test_a_host_build_actually_compiles_at_O0_and_at_O3(tmp_path):
    """The gate's own smoke, run for real rather than asserted from flag
    text: one tiny kernel, compiled on the CPU at each end of the ladder."""
    for level in OPT_LEVELS:
        result = compile_source(SOURCE, "probe", CompileOptions(
            backend=HOST, cache_dir=str(tmp_path / level), opt_level=level))
        assert result.artifact.is_file() and not result.hit, level


# --- the default follows the compiler -------------------------------------

import shutil
import threading


def _fresh_family_memo(monkeypatch):
    monkeypatch.setattr(tc, "_GCC_FAMILY_MEMO", {})


def test_gcc_on_x86_defaults_to_vector_math(monkeypatch):
    _fresh_family_memo(monkeypatch)
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    monkeypatch.setattr(tc, "_is_x86_64", lambda: True)
    monkeypatch.setattr(tc, "host_compiler", lambda: "/fake/g++")
    monkeypatch.setattr(tc, "compiler_is_gcc", lambda c: True)
    assert host_profile() == "native-vector-math"
    assert tc.default_host_profile() == "native-vector-math"


def test_a_clang_family_compiler_defaults_to_native(monkeypatch):
    _fresh_family_memo(monkeypatch)
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    clang = shutil.which("clang++")
    if clang:
        monkeypatch.setenv("HAWK_CXX", clang)
        assert not tc.compiler_is_gcc(clang)
    else:
        monkeypatch.setattr(tc, "host_compiler", lambda: "/fake/clang++")
        monkeypatch.setattr(tc, "compiler_is_gcc", lambda c: False)
    monkeypatch.setattr(tc, "_is_x86_64", lambda: True)
    assert tc.default_host_profile() == "native"
    assert host_profile() == "native"


def test_explicit_vector_math_with_clang_raises_before_any_compile(monkeypatch, tmp_path):
    _fresh_family_memo(monkeypatch)
    monkeypatch.setattr(tc, "_is_x86_64", lambda: True)
    monkeypatch.setattr(tc, "host_compiler", lambda: "/fake/clang++")
    monkeypatch.setattr(tc, "compiler_is_gcc", lambda c: False)
    monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
    with pytest.raises(hawk.HawkError, match=r"GCC.*clang\+\+.*native"):
        host_profile("native-vector-math")
    monkeypatch.setenv("HAWK_HOST_PROFILE", "native-vector-math")
    with pytest.raises(hawk.HawkError, match="GCC-only"):
        host_profile()
    clang = shutil.which("clang++")
    if clang:
        monkeypatch.undo()
        monkeypatch.setattr(tc, "_GCC_FAMILY_MEMO", {})
        monkeypatch.setenv("HAWK_CXX", clang)
        monkeypatch.setenv("HAWK_HOST_PROFILE", "native-vector-math")
        with pytest.raises(hawk.HawkError, match="GCC-only"):
            compile_source(SOURCE, "probe", CompileOptions(
                backend=HOST, cache_dir=str(tmp_path)))
        assert not list(tmp_path.glob("**/*.so"))


def test_the_family_memo_agrees_across_threads(monkeypatch):
    _fresh_family_memo(monkeypatch)
    cxx = tc.host_compiler()
    want = tc.compiler_is_gcc(cxx)
    monkeypatch.setattr(tc, "_GCC_FAMILY_MEMO", {})
    barrier = threading.Barrier(8)
    seen = []

    def work():
        barrier.wait()
        seen.append(tc.compiler_is_gcc(cxx))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [want] * 8
    assert tc._GCC_FAMILY_MEMO == {cxx: want}
