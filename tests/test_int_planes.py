# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""32-bit integer planes (``Int32``/``UInt32``): every operation against NumPy.

* each operator, on host and device, equals the NumPy ``int32``/``uint32``
  array result over a corpus that enumerates the edges: wraparound at
  ``+-2**31`` and ``2**32 - 1``, the signs of ``//`` and ``%`` (and a zero
  divisor, 0 as NumPy gives), shift counts outside ``[0, 32)``;
* the conversions: float -> integer truncates toward zero, saturates at the
  range and maps NaN to 0 (documented; NumPy leaves those cases undefined),
  integer -> integer wraps (``astype``);
* the refusals: an integer mixed with a float, or with the other integer
  type, or an index, without a cast; a float-only op on an integer; an integer
  plane named in ``wrt=``;
* autodiff: a float computed from an integer through ``real`` matches a central
  difference, and an integer plane gets no adjoint, no tangent and no tape row;
* the sidecar labels every integer plane with its own 4-byte dtype (the
  reserved counters are ``uint32``) and eagle allocates the outputs to match;
* a float-only kernel's emitted text is byte-identical to before integer
  planes existed.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import Index, Int32, Kernel, Mutable, Param, Scalar, Terminated, UInt32
from hawk.diff import jvp, vjp

INT32_EDGES = [0, 1, -1, 2, -2, 3, 7, -7, 31, 32, 33, 12345, -98765,
               2**31 - 1, -2**31]
UINT32_EDGES = [0, 1, 2, 3, 7, 31, 32, 33, 12345, 2**31 - 1, 2**31,
                4_000_000_000, 2**32 - 1]


def _ops(a, b, mn, mx, sel):
    """The operator set, spelled once for the traced body and for NumPy."""
    return {"add": a + b, "sub": a - b, "mul": a * b, "fdiv": a // b, "mod": a % b,
            "band": a & b, "bor": a | b, "bxor": a ^ b, "shl": a << b, "shr": a >> b,
            "inv": ~a, "neg": -a, "absv": abs(a), "mn": mn(a, b), "mx": mx(a, b),
            "lit": a * 3 + 1, "lt": sel(a < b, a, b), "le": sel(a <= b, a, b),
            "gt": sel(a > b, a, b), "ge": sel(a >= b, a, b), "eq": sel(a == b, a, b),
            "ne": sel(a != b, a, b)}


OUTS = tuple(_ops(*([np.int32(1)] * 2), np.minimum, np.maximum, np.where))


@hawk.kernel
def ops_i32(a: Int32, b: Int32, add: Mutable[Int32], sub: Mutable[Int32],
            mul: Mutable[Int32], fdiv: Mutable[Int32], mod: Mutable[Int32],
            band: Mutable[Int32], bor: Mutable[Int32], bxor: Mutable[Int32],
            shl: Mutable[Int32], shr: Mutable[Int32], inv: Mutable[Int32],
            neg: Mutable[Int32], absv: Mutable[Int32], mn: Mutable[Int32],
            mx: Mutable[Int32], lit: Mutable[Int32], lt: Mutable[Int32],
            le: Mutable[Int32], gt: Mutable[Int32], ge: Mutable[Int32],
            eq: Mutable[Int32], ne: Mutable[Int32]):
    add = a + b
    sub = a - b
    mul = a * b
    fdiv = a // b
    mod = a % b
    band = a & b
    bor = a | b
    bxor = a ^ b
    shl = a << b
    shr = a >> b
    inv = ~a
    neg = -a
    absv = abs(a)
    mn = m.minimum(a, b)
    mx = m.maximum(a, b)
    lit = a * 3 + 1
    lt = m.where(a < b, a, b)
    le = m.where(a <= b, a, b)
    gt = m.where(a > b, a, b)
    ge = m.where(a >= b, a, b)
    eq = m.where(a == b, a, b)
    ne = m.where(a != b, a, b)


@hawk.kernel
def ops_u32(a: UInt32, b: UInt32, add: Mutable[UInt32], sub: Mutable[UInt32],
            mul: Mutable[UInt32], fdiv: Mutable[UInt32], mod: Mutable[UInt32],
            band: Mutable[UInt32], bor: Mutable[UInt32], bxor: Mutable[UInt32],
            shl: Mutable[UInt32], shr: Mutable[UInt32], inv: Mutable[UInt32],
            neg: Mutable[UInt32], absv: Mutable[UInt32], mn: Mutable[UInt32],
            mx: Mutable[UInt32], lit: Mutable[UInt32], lt: Mutable[UInt32],
            le: Mutable[UInt32], gt: Mutable[UInt32], ge: Mutable[UInt32],
            eq: Mutable[UInt32], ne: Mutable[UInt32]):
    add = a + b
    sub = a - b
    mul = a * b
    fdiv = a // b
    mod = a % b
    band = a & b
    bor = a | b
    bxor = a ^ b
    shl = a << b
    shr = a >> b
    inv = ~a
    neg = -a
    absv = abs(a)
    mn = m.minimum(a, b)
    mx = m.maximum(a, b)
    lit = a * 3 + 1
    lt = m.where(a < b, a, b)
    le = m.where(a <= b, a, b)
    gt = m.where(a > b, a, b)
    ge = m.where(a >= b, a, b)
    eq = m.where(a == b, a, b)
    ne = m.where(a != b, a, b)


@hawk.kernel
def casts(x: Scalar, n: Int32, u: UInt32, s: Param[Int32], to_i: Mutable[Int32],
          to_u: Mutable[UInt32], back: Mutable[Scalar], wrap_u: Mutable[UInt32],
          wrap_i: Mutable[Int32], shifted: Mutable[Int32]):
    to_i = m.int32(x)
    to_u = m.uint32(x)
    back = m.real(n) * 0.5 + m.real(u)
    wrap_u = m.uint32(n)
    wrap_i = m.int32(u)
    shifted = n + s


def _pairs(edges, dtype):
    a, b = np.meshgrid(np.array(edges, dtype=dtype), np.array(edges, dtype=dtype))
    return a.ravel(), b.ravel()


def _numpy(a, b):
    with np.errstate(all="ignore"):
        return _ops(a, b, np.minimum, np.maximum, np.where)


XS = np.array([0.0, -0.0, 0.5, -0.5, 1.9, -1.9, 2147483647.0, 2147483647.9,
               2147483648.0, -2147483648.0, -2147483648.9, -2147483649.0,
               4294967295.0, 4294967295.5, 4294967296.0, 1e20, -1e20,
               np.inf, -np.inf, np.nan])


def _saturating(x, dtype):
    """The documented float -> integer rule: truncate toward zero, saturate at
    the range, NaN -> 0."""
    info = np.iinfo(dtype)
    t = np.where(np.isnan(x), 0.0, np.trunc(x))
    return np.clip(t, info.min, info.max).astype(dtype)


def _cast_inputs():
    k = XS.size
    n = np.resize(np.array(INT32_EDGES, np.int32), k)
    u = np.resize(np.array(UINT32_EDGES, np.uint32), k)
    return {"x": XS, "n": n, "u": u, "s": 2**31 - 5}


def _cast_reference(planes):
    n, u = planes["n"], planes["u"]
    with np.errstate(all="ignore"):
        return {"to_i": _saturating(planes["x"], np.int32),
                "to_u": _saturating(planes["x"], np.uint32),
                "back": n.astype(np.float64) * 0.5 + u.astype(np.float64),
                "wrap_u": n.astype(np.uint32), "wrap_i": u.astype(np.int32),
                "shifted": n + np.int32(planes["s"])}


def _deploy(cache, target):
    import eagle

    targets = ("host",) if target == "host" else ("host", "cuda")
    return eagle.deploy([ops_i32, ops_u32, casts], cache_dir=str(cache), targets=targets)


@pytest.fixture(scope="module")
def host_plans(tmp_path_factory):
    return _deploy(tmp_path_factory.mktemp("int_planes_host"), "host")


@pytest.fixture(scope="module")
def device_plans(tmp_path_factory):
    return _deploy(tmp_path_factory.mktemp("int_planes_device"), "cuda")


def _run(plan, planes, target):
    if target == "host":
        return plan.run(**planes)
    import cupy as cp

    got = plan.run(**{k: v if np.ndim(v) == 0 else cp.asarray(v) for k, v in planes.items()})
    return {k: cp.asnumpy(v) for k, v in got.items()}


def _plans(request, target):
    return request.getfixturevalue("host_plans" if target == "host" else "device_plans")


TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("which,dtype,edges", [(0, np.int32, INT32_EDGES),
                                               (1, np.uint32, UINT32_EDGES)],
                         ids=["int32", "uint32"])
def test_every_operator_equals_numpy(request, target, which, dtype, edges):
    a, b = _pairs(edges, dtype)
    got = _run(_plans(request, target)[which], {"a": a, "b": b}, target)
    want = _numpy(a, b)
    for name in OUTS:
        assert got[name].dtype == dtype, name
        np.testing.assert_array_equal(got[name], want[name].astype(dtype), err_msg=name)


@pytest.mark.parametrize("target", TARGETS)
def test_conversions_at_the_boundaries(request, target):
    planes = _cast_inputs()
    got = _run(_plans(request, target)[2], planes, target)
    for name, want in _cast_reference(planes).items():
        assert got[name].dtype == want.dtype, name
        np.testing.assert_array_equal(got[name], want, err_msg=name)


def test_the_saturating_rule_is_numpy_wherever_numpy_defines_it():
    inside = np.array([0.0, -0.0, 0.5, -0.5, 1.9, -1.9, 2147483647.9, -2147483648.9])
    np.testing.assert_array_equal(_saturating(inside, np.int32), inside.astype(np.int32))


def test_the_corpus_reaches_the_edges():
    a, b = _pairs(INT32_EDGES, np.int32)
    with np.errstate(all="ignore"):
        assert np.any((a > 0) & (b > 0) & (a + b < 0))           # wraps
        assert np.any((b == 0)) and np.any((a < 0) & (b > 0) & (a % b > 0))
        assert np.any((b >= 32) | (b < 0))                       # out-of-range shift
    assert np.isnan(XS).any() and np.isinf(XS).any()


def test_concurrent_runs_of_one_integer_plan_are_independent(host_plans):
    plan, out, errors = host_plans[0], {}, []

    def work(k):
        try:
            a, b = _pairs(INT32_EDGES, np.int32)
            out[k] = plan.run(a=a + np.int32(k), b=b)["add"]
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    a, b = _pairs(INT32_EDGES, np.int32)
    with np.errstate(all="ignore"):
        for k in range(4):
            np.testing.assert_array_equal(out[k], (a + np.int32(k)) + b)


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #
def _leaf(dtype, name="a"):
    from hawk.ir import Leaf
    from hawk.types import TensorType

    return hawk.Value(Leaf("vocab_read", "per_sample", name, TensorType((), dtype)))


@pytest.mark.parametrize("spell,match", [
    (lambda: _leaf("int32") + _leaf("f64", "x"), r"int32 value does not mix with f64.*hawk\.math\.real"),
    (lambda: _leaf("int32") * 1.5, r"does not mix with f64"),
    (lambda: _leaf("int32") + _leaf("uint32", "u"), r"does not mix with uint32.*hawk\.math\.uint32"),
    (lambda: _leaf("uint32") + _leaf("i32", "k"), r"uint32 value does not mix with i32"),
    (lambda: _leaf("int32") / _leaf("int32", "b"), r"'div' is not defined.*integer division is `//`"),
    (lambda: m.sqrt(_leaf("int32")), r"'sqrt' is not defined on an int32"),
    (lambda: _leaf("int32") ** 2, r"'pow' is not defined"),
    (lambda: _leaf("f64") // _leaf("f64", "b"), r"integer operator.*hawk\.math\.floor"),
    (lambda: _leaf("f64") & _leaf("f64", "b"), r"integer operator"),
    (lambda: _leaf("uint32") + (-1), r"outside the uint32 range"),
    (lambda: _leaf("int32") + 2**31, r"outside the int32 range"),
    (lambda: m.int32(_leaf("bool")), r"a bool picks a value"),
], ids=["float", "float_literal", "other_int", "index", "true_div", "sqrt", "pow",
        "float_floordiv", "float_bitand", "uint_negative_literal", "int_range",
        "bool_cast"])
def test_refusals_name_the_fix(spell, match):
    with pytest.raises(hawk.HawkError, match=match):
        spell()


def test_param_takes_only_a_scalar_or_an_integer():
    assert Param[Int32].ttype == Int32 and Param[UInt32].ttype == UInt32
    with pytest.raises(hawk.HawkError, match="Param\\[...\\] takes Scalar, Int32 or UInt32"):
        Param[Index]


# --------------------------------------------------------------------------- #
# Autodiff.
# --------------------------------------------------------------------------- #
@hawk.kernel
def scaled(n: Int32, x: Scalar, a: Param, y: Mutable[Scalar], c: Mutable[Int32]):
    y = m.real(n) * a * m.sin(x) + m.real(m.int32(x * 3.0))
    c = n * 2


def test_an_integer_plane_takes_no_adjoint_and_no_tangent():
    rev, fwd = vjp(scaled), jvp(scaled)
    assert set(rev.wrt) == set(fwd.wrt) == {"x", "a"}
    rev_names = Kernel("scaled_vjp", rev).walk.arg_spec
    fwd_names = Kernel("scaled_jvp", fwd).walk.arg_spec
    assert not [nm for _, nm in rev_names if nm in ("bar_n", "bar_c")]
    assert not [nm for _, nm in fwd_names if nm in ("dot_n", "dot_c")]
    assert ("per_sample", "n") in rev_names       # read as a primal input, never seeded


def test_an_integer_plane_in_wrt_is_refused_by_name():
    with pytest.raises(hawk.HawkError, match=r"wrt names \['n'\]: an integer plane"):
        vjp(scaled, wrt=["x", "n"])
    with pytest.raises(hawk.HawkError, match=r"wrt names \['n'\]"):
        jvp(scaled, wrt=["n"])


@hawk.kernel(steps=1)
def counted_step(dt: Param, x: Mutable[Scalar], k: Mutable[Int32]):
    x = x + dt * m.sin(x)
    k = k + 1


def test_an_integer_plane_gets_no_tape_column():
    from hawk.diff.tape import taped_planes

    assert [name for name, _ in taped_planes(counted_step)] == ["x"]


@pytest.mark.parametrize("target", TARGETS)
def test_the_gradient_through_a_cast_matches_a_central_difference(tmp_path, target):
    import eagle

    targets = ("host",) if target == "host" else ("host", "cuda")
    primal, rev = eagle.deploy([scaled, Kernel("scaled_vjp", vjp(scaled))],
                               cache_dir=str(tmp_path), targets=targets)
    rng = np.random.default_rng(11)
    n = rng.integers(-5, 6, 16).astype(np.int32)
    x = rng.uniform(0.1, 0.3, 16)          # 3x stays inside one integer step
    a, seed, h = 1.7, rng.uniform(0.5, 1.5, 16), 1e-6
    put = (lambda v: v) if target == "host" else __import__("cupy").asarray
    get = (lambda v: np.asarray(v)) if target == "host" else __import__("cupy").asnumpy

    def f(x_, a_):
        return get(primal.run(n=put(n), x=put(x_), a=a_)["y"])

    grads = rev.run(n=put(n), x=put(x), a=a, bar_y=put(seed))
    dx = (f(x + h, a) - f(x - h, a)) / (2 * h) * seed
    da = float(np.sum((f(x, a + h) - f(x, a - h)) / (2 * h) * seed))
    np.testing.assert_allclose(get(grads["bar_x"]), dx, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(float(np.sum(get(grads["bar_a"]))), da, rtol=1e-7)


# --------------------------------------------------------------------------- #
# Sidecar labels and eagle's allocation.
# --------------------------------------------------------------------------- #
@hawk.kernel(steps=1)
def finishing(dt: Param, t_final: Scalar, terminated: Terminated, t: Mutable[Scalar]):
    t = t + dt
    terminated = t >= t_final


@hawk.kernel(steps=1)
def mixing(k: Index, terminated: Terminated, y: Mutable[Scalar]):
    y = m.dispatch(k, [1.0, 2.0], policy="auto")


def _sidecar(kernel):
    from hawk.artifact.sidecar import sidecar_meta

    return sidecar_meta(kernel.name, kernel.walk, fmt="host", scalar_type="float64")


def test_every_integer_plane_is_labelled_with_its_own_dtype():
    side = _sidecar(casts)
    assert {k: side["arg_dtypes"][k] for k in ("n", "u", "to_i", "to_u", "wrap_u")} == {
        "n": "int32", "u": "uint32", "to_i": "int32", "to_u": "uint32", "wrap_u": "uint32"}
    assert side["arg_dtypes"]["x"] == "float64"
    assert {p["name"]: p["dtype"] for p in side["params"]} == {"s": "int"}


def test_the_reserved_counters_are_uint32():
    assert _sidecar(finishing)["arg_dtypes"]["finished_count"] == "uint32"
    side = _sidecar(mixing)
    mix = next(d["mix"] for d in side["dispatch"])
    assert side["arg_dtypes"][mix] == "uint32"


def test_a_deployed_sidecar_describes_its_integer_mutables():
    from hawk.runtime import descriptor_for_sidecar

    kinds = descriptor_for_sidecar(_sidecar(casts))
    spec = dict(zip((nm for _, nm in casts.walk.arg_spec), kinds))
    assert spec["to_i"] == spec["n"] == "int_handle" and spec["s"] == "int"


def test_eagle_reads_the_label_and_allocates_to_match(host_plans):
    assert host_plans[2].plugin.arg_dtypes["to_u"] == "uint32"
    got = host_plans[2].run(**_cast_inputs())
    assert {k: v.dtype for k, v in got.items()} == {
        "to_i": np.int32, "to_u": np.uint32, "back": np.float64, "wrap_u": np.uint32,
        "wrap_i": np.int32, "shifted": np.int32}


# --------------------------------------------------------------------------- #
# Zero cost for a float-only kernel.
# --------------------------------------------------------------------------- #
#: sha256 over every corpus kernel's emitted translation units (both backends,
#: both scalar modes), taken before integer planes existed.
FLOAT_ONLY_DIGESTS = {
    "as_body": "fabe9da159f0ff3e84534bd5e4f4e31fdfcc22138975fe51f0ff90d5a24e9aa8",
    "chain6": "8cc8a3e4e744156565c17777033e206c007e1df7bd6a0e51aafa08fddba77a77",
    "drag": "daee7a70d11748ebd82c1d03ffe2ab829a2bb507da18f8dceb01f3c256e8fc8c",
    "drag_vjp": "4ab402649410df495823357e4d01d248110fc24ee9b02d5691def5b8c44dcdd7",
    "energy": "425212a8a9a3e45b310229413cdcb73e4203e79b08828cd25e21693651abdc86",
    "gather": "d8c27845d56e55b55e9db4e4bb2bbc5df731d1362aa21116ab0585f3c1ab943e",
    "rotate_quat": "fe1634aba3c69d61177c1b321c588a0972f4052c98667361321a88ac4072e63d",
    "scale": "1e37d343cea724f7480e82378d805e622336414a68250a6d2447db80ce367328",
    "scatter": "aebfe93674f7da66f069d6cd9d939f0180aa2cd774a4940456824894d62a5db7",
    "spin": "1c0ccc4d2025307ab2d5c7a78f54295028d5d2ef277ba7c00816629d32ffce24",
}


def test_a_float_only_kernel_emits_the_same_bytes(monkeypatch):
    import _emitted

    import hawk.emit.host as host_emit

    pinned = host_emit.CacheSizes(32 * 1024, 1024 * 1024, "/sys/devices/system/cpu/cpu0/cache")
    monkeypatch.setattr(host_emit, "machine_caches", lambda: pinned)
    got = {}
    for name, _backend, _mode, src in _emitted.sources():
        got.setdefault(name, hashlib.sha256()).update(src.text.encode())
    assert {k: v.hexdigest() for k, v in got.items()} == FLOAT_ONLY_DIGESTS


def test_an_integer_body_carries_the_integer_helpers_and_a_float_one_does_not():
    from hawk.emit import BACKENDS, render_source

    text = render_source("ops_i32", ops_i32.sinks, ops_i32.walk, BACKENDS["host"]).text
    assert "int_floordiv" in text and "std::int32_t" in text
    from _kernels import CORPUS

    drag = CORPUS["drag"]
    assert "int_traits" not in render_source("drag", drag.sinks, drag.walk,
                                             BACKENDS["host"]).text
