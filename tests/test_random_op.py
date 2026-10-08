# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests ``hawk.math.random.{uniform,normal}``: aether's stateless,
counter-based Philox draw, ``(seed, counter)`` -> a rank-0 real, lowered to
``aether::random::detail::{uniform01,standardNormal}<Real>(seed, global,
counter)`` with ``global`` the kernel's own flattened sample index — never a
third IR operand, so every lane is its own stream by construction.

Related coverage lives beside the harness it extends: the NVRTC route in
``test_compile_nvrtc.py``, the static no-include row in
``test_emit_body_identity.py``, and the flat/namespace spellings in
``test_math_namespace.py``.

The device rows bind ``seed``/``counter`` as per-sample i32 planes (bound as
int64 arrays -- `hawk.emit.aether.element_spelling` collapses BOTH i32 and
i64 to the SAME C++ ``Int`` = ``long long``); the by-value ``uniform`` form is
covered separately
(``test_uniform_by_value_integer_seed_crosses_the_launch_intact``). That
row found and pinned an eagle defect: ``eagle/plan.py``'s
``_uniform_box`` packed every uniform-role scalar as a C double, ignoring the
sidecar's ``params`` ``dtype: "int"`` declaration, so an integer uniform
crossed the launch as the double's bit pattern; fixed promptly in eagle.
The plane form stays the other rows' shape because per-lane counters are
what the bootstrap needs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _deploy import device_plugin
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.ir
import hawk.math as m
import hawk.trace.decl as _decl
from hawk.artifact import build, build_bundle
from hawk.diff import jvp, vjp
from hawk.ir import Assign, HawkError, Leaf
from hawk.ir import make as mk
from hawk.ir.ops import OP_KINDS, RANDOM_OPS
from hawk.ir.walk import canonical_nodes
from hawk.types import TensorType

F64 = TensorType((), "f64")
I32 = TensorType((), "i32")
I64 = TensorType((), "i64")
V3 = TensorType((3,), "f64")

_AETHER_ROOT = os.environ.get("HAWK_AETHER_SOURCE", "")


@hawk.kernel
def draw_uniform(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.uniform(seed, counter)


@hawk.kernel
def draw_normal(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.normal(seed, counter)


@hawk.kernel
def draw_exponential(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.exponential(seed, counter, rate=2.0)


@hawk.kernel
def draw_bernoulli(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.bernoulli(seed, counter, p=0.3)


@hawk.kernel
def draw_lognormal(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.lognormal(seed, counter, mu=0.5, sigma=0.25)


@hawk.kernel
def draw_uniform_int(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.uniform_int(seed, counter, -2, 5)


@hawk.kernel
def draw_uniform_range(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.uniform(seed, counter, lo=-1.0, hi=3.0)


@hawk.kernel
def draw_normal_scaled(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.normal(seed, counter, mean=2.0, sd=0.5)


@hawk.kernel
def draw_mvn(seed: hawk.Index, counter: hawk.Index, chol: hawk.Matrix[3, 3],
             y: hawk.Mutable[hawk.Vector[3]]):
    y = m.random.multivariate_normal(seed, counter, m.vec(1.0, -2.0, 0.5), chol)


@hawk.kernel
def draw_poisson1(seed: hawk.Index, counter: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = m.random.poisson1(seed, counter)


#: role=uniform (a by-value scalar), dtype=i32 -- the ONE way to declare
#: an integer "uniform" wire (``hawk.Param`` is fixed to f64 ``Scalar``). One
#: by-value row binds the seed through it; the other kernels bind seed/counter
#: as per-sample PLANES (a bare ``hawk.Index``/``hawk.ir.TensorType((), "i32")``
#: annotation) because per-lane counters are the bootstrap's own shape.
_UniformIndex = _decl.Plane("uniform", hawk.Index)


@hawk.kernel
def readback(seed: _UniformIndex, y: hawk.Mutable[hawk.Scalar]):
    y = seed * 1.0


# --------------------------------------------------------------------------- #
# Closed world: both kinds registered, arity/type refused BY NAME.
# --------------------------------------------------------------------------- #
def test_random_ops_are_in_the_closed_world():
    assert RANDOM_OPS == ("random_uniform", "random_normal")
    assert set(RANDOM_OPS) <= set(OP_KINDS)


def test_an_unknown_random_kind_still_refuses():
    seed = Leaf("vocab_read", "per_sample", "seed", I32)
    counter = Leaf("vocab_read", "per_sample", "counter", I32)
    with pytest.raises(HawkError, match="unknown op kind"):
        mk("random_geometric", (seed, counter))


@pytest.mark.parametrize("kind", RANDOM_OPS)
def test_a_rank1_seed_is_refused_by_name(kind):
    bad = Leaf("vocab_read", "vec_in", "seed", TensorType((3,), "i32"))
    counter = Leaf("vocab_read", "per_sample", "counter", I32)
    with pytest.raises(HawkError, match=f"{kind}: seed"):
        mk(kind, (bad, counter))


@pytest.mark.parametrize("kind", RANDOM_OPS)
def test_a_real_typed_seed_is_refused_by_name(kind):
    bad = Leaf("vocab_read", "per_sample", "seed", F64)
    counter = Leaf("vocab_read", "per_sample", "counter", I32)
    with pytest.raises(HawkError, match=f"{kind}: seed"):
        mk(kind, (bad, counter))


@pytest.mark.parametrize("kind", RANDOM_OPS)
def test_a_real_typed_counter_is_refused_by_name(kind):
    seed = Leaf("vocab_read", "per_sample", "seed", I32)
    bad = Leaf("vocab_read", "per_sample", "counter", F64)
    with pytest.raises(HawkError, match=f"{kind}: counter"):
        mk(kind, (seed, bad))


@pytest.mark.parametrize("kind", RANDOM_OPS)
def test_arity_is_exactly_two(kind):
    seed = Leaf("vocab_read", "per_sample", "seed", I32)
    with pytest.raises(HawkError, match="takes 2 operand"):
        mk(kind, (seed,))


@pytest.mark.parametrize("kind", RANDOM_OPS)
def test_i64_operands_are_accepted_and_the_result_is_rank0_f64(kind):
    seed = Leaf("vocab_read", "per_sample", "seed", I64)
    counter = Leaf("vocab_read", "per_sample", "counter", I64)
    node = mk(kind, (seed, counter))
    assert node.ttype == F64, (
        "a random draw is typed the way a float literal is, so it "
        "composes with the rest of the promotion ladder")


def test_math_random_uniform_refuses_a_literal_float_seed_by_name():
    with pytest.raises(HawkError, match="random_uniform: seed"):
        m.random_uniform(1.5, 0)


# --------------------------------------------------------------------------- #
# helpers shared below: bind seed/counter as per-sample INT64 planes
# (the C++ side is `Int` = `long long` regardless of the IR's i32/i64 dtype,
# `hawk.emit.aether.element_spelling`) and run the compiled artifact.
# --------------------------------------------------------------------------- #
def _plane(n: int, value) -> np.ndarray:
    if np.ndim(value) == 0:
        return np.full(n, value, dtype=np.int64)
    return np.asarray(value, dtype=np.int64)


def _host_run(loaded, n: int, seed, counter) -> np.ndarray:
    import hawk.runtime as RT

    y = np.zeros(n, dtype=np.float64)
    RT.run(loaded, base=0, count=n, n_samples=n, y=y,
          seed=_plane(n, seed), counter=_plane(n, counter))
    return y


def _device_run(bundle, name, sidecar, n: int, seed, counter) -> np.ndarray:
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = device_plugin(bundle.directory, name, sidecar)
    y = cp.empty(n, dtype=np.float64)
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(
        seed=_plane(n, seed), counter=_plane(n, counter), y=y)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(got.get() if hasattr(got, "get") else got)


# --------------------------------------------------------------------------- #
# Replay and host/device twin, n = 65536, on the device (and the host oracle).
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_device_replay_and_host_device_twin(tmp_path, cache_dir):
    import hawk.runtime as RT

    n = 65536
    bundle = build_bundle([draw_uniform], tmp_path / "r2", targets=("cuda", "host"),
                          cache_dir=cache_dir)
    sc = sidecar_of(bundle, "draw_uniform")
    hloaded = RT.load(bundle.directory, "draw_uniform", sc)

    dev1 = _device_run(bundle, "draw_uniform", sc, n, 7, 0)
    dev2 = _device_run(bundle, "draw_uniform", sc, n, 7, 0)
    assert np.array_equal(dev1, dev2), (
        "R2: two device launches of plane(seed=7, counter=0) must be bit-equal")

    host1 = _host_run(hloaded, n, 7, 0)
    assert np.array_equal(dev1, host1), (
        "R2: host and device must be bit-equal (aether's own host==device "
        "contract, exercised through HAWK for the first time here)")

    dev_seed8 = _device_run(bundle, "draw_uniform", sc, n, 8, 0)
    assert not np.array_equal(dev1, dev_seed8), "R2: seed 8 must differ"

    dev_c1 = _device_run(bundle, "draw_uniform", sc, n, 7, 1)
    assert not np.array_equal(dev1, dev_c1), "R2: counter 1 must differ"

    assert not np.all(dev1[1:] == dev1[:-1]), (
        "R2: adjacent lanes must differ -- no constant plane")


# --------------------------------------------------------------------------- #
# Distribution at n = 2^20, bands DERIVED from n. Run on the HOST: the row
# above already proved host==device bit-identical, so host coverage is
# device coverage. RED first: a manufactured constant plane fails the band; only
# then the real compiled plane is checked.
# --------------------------------------------------------------------------- #
def _bands(n: int):
    """(uniform mean, uniform var, uniform KS, normal mean, normal var, KS).

    ``uniform`` mean band ``3/sqrt(12n)`` (3-sigma of the U(0,1) mean's own
    std err ``sqrt(1/(12n))``); var band ``3*sqrt(1/(180n))`` (3-sigma of the
    sample variance's std err, ``Var[s^2] = 2/(n-1) * true_var^2`` collapsed
    with ``true_var=1/12``: ``sqrt(2/n)/12 = sqrt(1/(180n))`` less a factor);
    ``normal`` mean band ``3/sqrt(n)``, var band ``3*sqrt(2/n)`` (the same
    3-sigma rule at ``true_var=1``); both KS bands are the same asymptotic
    Kolmogorov bound ``1.63/sqrt(n)``."""
    ks = 1.63 / np.sqrt(n)
    return (3.0 / np.sqrt(12 * n), 3.0 * np.sqrt(1.0 / (180 * n)), ks,
           3.0 / np.sqrt(n), 3.0 * np.sqrt(2.0 / n), ks)


def _uniform_ok(plane: np.ndarray, n: int):
    from scipy import stats

    mean_b, var_b, ks_b, *_ = _bands(n)
    mean, var = float(plane.mean()), float(plane.var())
    ks = float(stats.kstest(plane, "uniform").statistic)
    return (mean, var, ks,
           abs(mean - 0.5) <= mean_b and abs(var - 1.0 / 12.0) <= var_b and ks <= ks_b)


def _normal_ok(plane: np.ndarray, n: int):
    from scipy import stats

    *_, mean_b, var_b, ks_b = _bands(n)
    mean, var = float(plane.mean()), float(plane.var())
    ks = float(stats.kstest(plane, "norm").statistic)
    return (mean, var, ks,
           abs(mean) <= mean_b and abs(var - 1.0) <= var_b and ks <= ks_b)


def test_uniform_distribution_bands(tmp_path, cache_dir):
    n = 1 << 20
    broken = np.full(n, 0.5)
    *_, broken_ok = _uniform_ok(broken, n)
    assert not broken_ok, "R3 RED: a constant plane (broken RNG) must FAIL the band"

    import _oracle as O

    build(draw_uniform, tmp_path / "r3u", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "r3u", "draw_uniform")
    planes = O.planes(loaded, n, {"seed": _plane(n, 7), "counter": _plane(n, 0)})
    y = np.asarray(O.run_kernel(loaded, n, **planes))
    mean, var, ks, ok = _uniform_ok(y, n)
    mean_b, var_b, ks_b, *_ = _bands(n)
    assert ok, (
        f"R3 uniform: mean={mean} (band {mean_b}), var={var} (band {var_b}), "
        f"ks={ks} (band {ks_b})")


def test_normal_distribution_bands(tmp_path, cache_dir):
    n = 1 << 20
    broken = np.full(n, 0.0)
    *_, broken_ok = _normal_ok(broken, n)
    assert not broken_ok, "R3 RED: a constant plane (broken RNG) must FAIL the band"

    import _oracle as O

    build(draw_normal, tmp_path / "r3n", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "r3n", "draw_normal")
    planes = O.planes(loaded, n, {"seed": _plane(n, 7), "counter": _plane(n, 0)})
    y = np.asarray(O.run_kernel(loaded, n, **planes))
    mean, var, ks, ok = _normal_ok(y, n)
    *_, mean_b, var_b, ks_b = _bands(n)
    assert ok, (
        f"R3 normal: mean={mean} (band {mean_b}), var={var} (band {var_b}), "
        f"ks={ks} (band {ks_b})")


# --------------------------------------------------------------------------- #
# Oracle: aether's own golden header, aether/tests/random/
# golden_random.h (CoreRow rows, all at global=0 so a 1-lane HOST run's own
# sample_index()==0 reproduces the same `global`). uniformDoubleBits is EXACT
# (bit-for-bit); normalDouble is TIER-TOL per the header's own documented
# `kNormalDoubleTol = 1e-12` (aether's Box-Muller routes through
# aether::math, not bit-reproducible against a from-scratch reimplementation
# by construction, and the header says so).
# --------------------------------------------------------------------------- #
#: (seed, counter, uniformDoubleBits, normalDouble) at global=0 -- transcribed
#: from aether/tests/random/golden_random.h's kCoreRows.
_GOLDEN_ROWS = (
    (0x0000000000000000, 0, 0x3FD989FA35800000, -0.31039573600037634),
    (0x0000000000000000, 1, 0x3FEC2D38B1C00000, -0.13595797896327272),
    (0x0000000000000000, 2, 0x3FE78AF589A00000, 0.76338801926953026),
    (0x0000000000000000, 3, 0x3FE3601B7B200000, -0.22271010206952083),
    (0x0000000000000000, 4, 0x3FEF1C9994A00000, 0.3292107012677683),
    (0x0000000000000000, 7, 0x3FA2FDFED0000000, 2.1735188577861648),
    (0x0000000000000000, 9, 0x3FD471CCA9C00000, -1.9252709937181016),
    (-1, 0, 0x3FDCA91DC2800000, 1.6700178282651672),  # seed 0xFFFF...FFFF
)


def _bits_to_double(bits: int) -> float:
    import struct

    return struct.unpack("<d", struct.pack("<Q", bits))[0]


@pytest.mark.skipif(not os.path.isdir(_AETHER_ROOT + "/tests/random"),
                    reason="aether/tests/random/golden_random.h not on this box")
def test_matches_the_aether_golden_oracle(tmp_path, cache_dir):
    import _oracle as O

    build(draw_uniform, tmp_path / "r4u", targets=("host",), cache_dir=cache_dir)
    build(draw_normal, tmp_path / "r4n", targets=("host",), cache_dir=cache_dir)
    u_loaded = O.load(tmp_path / "r4u", "draw_uniform")
    n_loaded = O.load(tmp_path / "r4n", "draw_normal")

    for seed, counter, u_bits, n_val in _GOLDEN_ROWS:
        u_planes = O.planes(u_loaded, 1, {"seed": _plane(1, seed), "counter": _plane(1, counter)})
        u = float(np.asarray(O.run_kernel(u_loaded, 1, **u_planes))[0])
        want_u = _bits_to_double(u_bits)
        assert u == want_u, (
            f"R4 uniform: seed={seed:#x} counter={counter}: got {u!r}, golden "
            f"{want_u!r} (bits {u_bits:#018x})")

        n_planes = O.planes(n_loaded, 1, {"seed": _plane(1, seed), "counter": _plane(1, counter)})
        got_n = float(np.asarray(O.run_kernel(n_loaded, 1, **n_planes))[0])
        assert abs(got_n - n_val) <= 1e-12, (
            f"R4 normal: seed={seed:#x} counter={counter}: got {got_n!r}, golden "
            f"{n_val!r} (TIER-TOL 1e-12)")


# --------------------------------------------------------------------------- #
# Differentiation: f = x * random_normal(seed, c) + y differentiates through the
# FD harness's own pattern; no adjoint term reaches seed/counter; the rule
# card is re-minted by test_diff_rule_card.py's own run (not here).
# --------------------------------------------------------------------------- #
def test_diff_no_adjoint_term_for_seed_or_counter_and_matches_fd():
    seed = Leaf("vocab_read", "per_sample", "seed", I32)
    counter = Leaf("vocab_read", "per_sample", "counter", I32)
    x = Leaf("vocab_read", "per_sample", "x", F64)
    y = Leaf("vocab_read", "per_sample", "y", F64)
    draw = mk("random_normal", (seed, counter))
    f = mk("add", (mk("mul", (x, draw)), y))
    sinks = (Assign("out", f, F64),)

    derived = vjp(sinks)
    assert derived.wrt == ("x", "y"), (
        f"R5: seed/counter must never be selected as gradient inputs (their "
        f"integer dtype keeps them out of the _DIFFERENTIABLE set), got "
        f"wrt={derived.wrt}")

    # structural audit: in the DERIVED (reverse) IR, only the draw node itself
    # may reference seed/counter -- the zero rule must add no OTHER edge onto
    # them (the closed world, read honestly rather than merely absent).
    seq, _ = canonical_nodes(tuple(derived))
    touching = [n for n in seq
               if any(o is seed or o is counter for o in getattr(n, "operands", ()))]
    assert touching == [draw], (
        f"R5: something besides the draw itself depends on seed/counter in "
        f"the derived IR: {touching}")

    jderived = jvp(sinks)
    assert jderived.wrt == ("x", "y")

    H = 1e-5
    env = {"x": 0.6, "y": -0.3, "seed": 11, "counter": 4}

    def f_at(xv, yv):
        return evaluate(sinks, {**env, "x": xv, "y": yv})["out"]

    fd_x = (f_at(env["x"] + H, env["y"]) - f_at(env["x"] - H, env["y"])) / (2 * H)
    fd_y = (f_at(env["x"], env["y"] + H) - f_at(env["x"], env["y"] - H)) / (2 * H)
    got_v = evaluate(derived, {**env, "bar_out": 1.0})
    assert got_v["bar_x"] == pytest.approx(fd_x, rel=1e-6, abs=1e-7)
    assert got_v["bar_y"] == pytest.approx(fd_y, rel=1e-6, abs=1e-7)

    got_j = evaluate(jderived, {**env, "dot_x": 1.0, "dot_y": 0.0})
    assert got_j["dot_out"] == pytest.approx(fd_x, rel=1e-6, abs=1e-7)
    got_j2 = evaluate(jderived, {**env, "dot_x": 0.0, "dot_y": 1.0})
    assert got_j2["dot_out"] == pytest.approx(fd_y, rel=1e-6, abs=1e-7)


# --------------------------------------------------------------------------- #
# Capture (eagle, device): a graph built ONCE, replayed after WRITING
# the seed/counter wire in place -- draws change, no re-capture (asserted on
# the captured-graph handle's identity, GraphPipeline._graph). Counter as a
# per-lane PLANE reproduces the two literal-counter lanes (already proven
# at n=65536 above; here it is re-checked inside the SAME captured graph).
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_capture_replay_write_wire_no_recapture(tmp_path, cache_dir):
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.pipeline import GraphPipeline

    n = 4096
    bundle = build_bundle([draw_uniform], tmp_path / "r6", targets=("cuda",),
                          cache_dir=cache_dir)
    sc = sidecar_of(bundle, "draw_uniform")
    plugin = device_plugin(bundle.directory, "draw_uniform", sc)

    seed_dev = cp.full(n, 7, dtype=cp.int64)
    counter_dev = cp.zeros(n, dtype=cp.int64)
    y_dev = cp.empty(n, dtype=cp.float64)
    bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(
        seed=seed_dev, counter=counter_dev, y=y_dev)

    pipe = GraphPipeline()
    pipe.add(lambda: bound.launch())
    pipe.build()
    graph_id = id(pipe._graph)

    pipe.launch()
    cp.cuda.get_current_stream().synchronize()
    first = cp.asnumpy(y_dev).copy()

    seed_dev.fill(8)
    pipe.launch()
    cp.cuda.get_current_stream().synchronize()
    second = cp.asnumpy(y_dev).copy()
    assert id(pipe._graph) == graph_id, "R6: writing the seed wire must not re-capture"
    assert not np.array_equal(first, second), "R6: draws must change after the write"

    counter_dev[:] = cp.asarray(np.array([0, 1] * (n // 2), dtype=np.int64))
    pipe.launch()
    cp.cuda.get_current_stream().synchronize()
    third = cp.asnumpy(y_dev).copy()
    assert id(pipe._graph) == graph_id, "R6: writing the counter wire must not re-capture"

    seed_dev.fill(8)
    counter_dev.fill(0)
    pipe.launch()
    cp.cuda.get_current_stream().synchronize()
    ref0 = cp.asnumpy(y_dev).copy()
    counter_dev.fill(1)
    pipe.launch()
    cp.cuda.get_current_stream().synchronize()
    ref1 = cp.asnumpy(y_dev).copy()
    assert np.array_equal(third[0::2], ref0[0::2]), (
        "R6: lane-varying counter 0 must reproduce the literal-counter-0 lanes")
    assert np.array_equal(third[1::2], ref1[1::2]), (
        "R6: lane-varying counter 1 must reproduce the literal-counter-1 lanes")


@pytest.mark.gpu
def test_uniform_by_value_integer_seed_crosses_the_launch_intact(tmp_path, cache_dir):
    """The by-value ``uniform`` INTEGER wire -- the "write the seed
    wire, replay" form -- reaches the device intact. The previous version
    of this row found that eagle's ``_uniform_box`` packed every
    uniform as a C double, so ``seed=7`` read back as ``4619567317775286272``
    (the double's bit pattern); fixed in ``eagle/plan.py``
    (``_integer_uniforms`` reads the sidecar's v2 ``params`` ``dtype``, and
    ``hawk.artifact.plan_view`` now carries that block)."""
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = build_bundle([readback], tmp_path / "r6_uniform_int", targets=("cuda",),
                          cache_dir=cache_dir)
    sc = sidecar_of(bundle, "readback")
    plugin = device_plugin(bundle.directory, "readback", sc)
    assert eplan._integer_uniforms(plugin) == {"seed"}
    y_dev = cp.empty(4, dtype=cp.float64)
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(seed=7, y=y_dev)
    cp.cuda.get_current_stream().synchronize()
    acc = cp.asnumpy(got.get() if hasattr(got, "get") else got)
    assert np.all(acc == 7.0), f"integer uniform corrupted across the launch: {acc}"


# --------------------------------------------------------------------------- #
# poisson1: the bootstrap weight. (a) the DEVICE composition equals a numpy
# re-implementation fed the device's OWN sixteen uniform planes (sub-counters
# counter*16 + k), bit-for-bit, and host == device; (b) distribution at
# n = 2^20 on the host, bands derived from n, RED first on a manufactured
# constant plane.
# --------------------------------------------------------------------------- #
def _poisson1_ok(plane: np.ndarray, n: int):
    """Poisson(1): mean 1 (std err ``1/sqrt(n)``), variance 1 (std err of the
    sample variance ``sqrt((mu4 - var^2)/n)`` with ``mu4 = lambda(1+3lambda)
    = 4``, so ``sqrt(3/n)``); 3-sigma bands on both."""
    mean, var = float(plane.mean()), float(plane.var())
    ok = (abs(mean - 1.0) <= 3.0 / np.sqrt(n) and abs(var - 1.0) <= 3.0 * np.sqrt(3.0 / n)
          and np.all(plane == np.round(plane)) and plane.min() == 0.0
          and 4.0 <= plane.max() <= 16.0)
    return mean, var, ok


@pytest.mark.gpu
def test_poisson1_matches_the_reference_from_the_devices_own_uniforms(tmp_path, cache_dir):
    import math

    import hawk.runtime as RT

    n = 65536
    counter = 3
    bundle = build_bundle([draw_uniform, draw_poisson1], tmp_path / "r10",
                          targets=("cuda", "host"), cache_dir=cache_dir)
    sc_u = sidecar_of(bundle, "draw_uniform")
    sc_p = sidecar_of(bundle, "draw_poisson1")
    dev = _device_run(bundle, "draw_poisson1", sc_p, n, 7, counter)
    draws = np.stack([_device_run(bundle, "draw_uniform", sc_u, n, 7, counter * 16 + k)
                      for k in range(16)])
    prefix = np.cumprod(draws, axis=0)  # f64, the device's own left-to-right order
    ref = (prefix > math.exp(-1.0)).sum(axis=0).astype(np.float64)
    assert np.array_equal(dev, ref), (
        "R10: the device poisson1 plane must equal Knuth's count over the device's "
        f"own uniform sub-draws; {int((dev != ref).sum())} lanes differ")
    host = _host_run(RT.load(bundle.directory, "draw_poisson1", sc_p), n, 7, counter)
    assert np.array_equal(dev, host), "R10: host and device poisson1 must be bit-equal"
    assert not np.array_equal(dev, _device_run(bundle, "draw_poisson1", sc_p, n, 7, counter + 1)), (
        "R10: replicate counter+1 must draw fresh weights")


def test_poisson1_distribution_bands(tmp_path, cache_dir):
    n = 1 << 20
    broken = np.ones(n)  # mean 1 exactly, variance 0: the band must FAIL it
    *_, broken_ok = _poisson1_ok(broken, n)
    assert not broken_ok, "R10 RED: a constant plane (broken weights) must FAIL the band"

    import _oracle as O

    build(draw_poisson1, tmp_path / "r10h", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "r10h", "draw_poisson1")
    planes = O.planes(loaded, n, {"seed": _plane(n, 7), "counter": _plane(n, 0)})
    y = np.asarray(O.run_kernel(loaded, n, **planes))
    mean, var, ok = _poisson1_ok(y, n)
    assert ok, (
        f"R10 poisson1: mean={mean} (band 1 +- {3.0 / np.sqrt(n)}), var={var} "
        f"(band 1 +- {3.0 * np.sqrt(3.0 / n)}), min={y.min()}, max={y.max()}")


# --------------------------------------------------------------------------- #
# The distribution FAMILY (compositions over the two primitives): host path
# at n = 2^18, bands derived from n, RED first on a manufactured constant
# plane per family.
# --------------------------------------------------------------------------- #
def _family_ok(name: str, plane: np.ndarray, n: int) -> tuple:
    from scipy import stats

    ks_b = 1.63 / np.sqrt(n)
    mean = float(plane.mean())
    if name == "exponential":  # rate 2: mean 1/2, std 1/2
        ks = float(stats.kstest(plane, "expon", args=(0.0, 0.5)).statistic)
        return mean, ks, (plane.min() >= 0.0 and abs(mean - 0.5) <= 3.0 * 0.5 / np.sqrt(n)
                          and ks <= ks_b)
    if name == "bernoulli":  # p = 0.3
        ok = set(np.unique(plane)) <= {0.0, 1.0} and abs(mean - 0.3) <= 3.0 * np.sqrt(0.21 / n)
        return mean, float("nan"), ok
    if name == "lognormal":  # mu 0.5, sigma 0.25: log(plane) ~ N(0.5, 0.25)
        ks = float(stats.kstest((np.log(plane) - 0.5) / 0.25, "norm").statistic)
        return mean, ks, bool(plane.min() > 0.0 and ks <= ks_b)
    if name == "uniform_int":  # [-2, 5]: 8 values, mean 1.5, var (8^2-1)/12
        ok = (np.all(plane == np.round(plane)) and set(np.unique(plane)) == set(range(-2, 6))
              and abs(mean - 1.5) <= 3.0 * np.sqrt(5.25 / n))
        return mean, float("nan"), ok
    if name == "uniform_range":  # [-1, 3)
        ks = float(stats.kstest(plane, "uniform", args=(-1.0, 4.0)).statistic)
        return mean, ks, bool(plane.min() >= -1.0 and plane.max() < 3.0 and ks <= ks_b)
    if name == "normal_scaled":  # N(2, 0.5^2)
        ks = float(stats.kstest((plane - 2.0) / 0.5, "norm").statistic)
        return mean, ks, bool(ks <= ks_b)
    raise AssertionError(name)


_FAMILY = {
    "exponential": (draw_exponential, 0.5), "bernoulli": (draw_bernoulli, 0.3),
    "lognormal": (draw_lognormal, 1.7), "uniform_int": (draw_uniform_int, 1.5),
    "uniform_range": (draw_uniform_range, 1.0), "normal_scaled": (draw_normal_scaled, 2.0),
}


@pytest.mark.parametrize("name", sorted(_FAMILY))
def test_family_distribution_bands(name, tmp_path, cache_dir):
    import _oracle as O

    n = 1 << 18
    kernel, centre = _FAMILY[name]
    *_, broken_ok = _family_ok(name, np.full(n, centre), n)
    assert not broken_ok, f"R11 RED: a constant plane must FAIL the {name} band"

    build(kernel, tmp_path / name, targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / name, kernel.name)
    planes = O.planes(loaded, n, {"seed": _plane(n, 11), "counter": _plane(n, 0)})
    y = np.asarray(O.run_kernel(loaded, n, **planes), dtype=np.float64)
    mean, ks, ok = _family_ok(name, y, n)
    assert ok, f"R11 {name}: mean={mean} ks={ks} min={y.min()} max={y.max()}"


def test_multivariate_normal_mean_and_covariance(tmp_path, cache_dir):
    import _oracle as O

    n = 1 << 18
    mean = np.array([1.0, -2.0, 0.5])
    chol = np.array([[2.0, 0.0, 0.0], [0.6, 1.5, 0.0], [-0.4, 0.3, 1.0]])
    cov = chol @ chol.T
    build(draw_mvn, tmp_path / "mvn", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "mvn", "draw_mvn")
    planes = O.planes(loaded, n, {"seed": _plane(n, 11), "counter": _plane(n, 0),
                                  "chol": np.repeat(chol.reshape(9, 1), n, axis=1)})
    y = np.asarray(O.run_kernel(loaded, n, **planes), dtype=np.float64)
    y = y.reshape(3, n) if y.shape[0] == 3 else y.reshape(n, 3).T
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(y.mean(axis=1) - mean) <= 3.0 * sd / np.sqrt(n)), y.mean(axis=1)
    S = np.cov(y)
    band = 3.0 * np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov ** 2) / n)
    assert np.all(np.abs(S - cov) <= band), (S, cov, band)
    assert not np.allclose(S, np.diag(np.diag(S))), "the draws must be correlated (L is not diagonal)"

