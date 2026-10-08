# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The numpy-parity element-wise set, run: values and derivatives, host and device.

Three claims per function, over a corpus that enumerates the special values
(signed zeros, infinities, NaN, subnormals, the domain edges ``log1p(-1)``,
``acosh(1)``, ``atanh(+-1)``, large ``erf``/``erfc`` arguments, ``fmod`` and
``remainder`` sign cases):

* the host build under the bit-identical ``native`` profile returns EXACTLY the
  scalar reference — glibc called through ``ctypes`` for the libm functions,
  numpy for the ones libm does not carry (``sign``, ``remainder``, ``clip``,
  the class tests);
* the host build under the default profile and the device build agree with that
  reference within :data:`ULP` (libdevice and the host vector packets are not
  glibc), and EXACTLY for the functions that round once by definition;
* every function agrees with numpy within :data:`ULP` (exactly for the exact
  set), ``round`` against C's half-away-from-zero rule rather than ``np.round``.

Derivatives: a forward and a reverse kernel of one body carrying every
function are compared with a central difference of the SAME target's compiled
primal. And each function runs inside a finishing kernel through
``eagle.simulate`` on both targets.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import math

import numpy as np
import pytest
from _eval import c_round

import hawk
import hawk.math as m
from hawk import Kernel, Matrix, Mutable, Scalar, Terminated, Vector
from hawk.diff import jvp, vjp

_LIBM = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")


def _libm(name: str, arity: int):
    fn = getattr(_LIBM, name)
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.c_double] * arity
    return np.vectorize(fn, otypes=[float])


def _clip(x, lo, hi):
    """The scalar clip formula: ``x`` below ``lo`` gives ``lo``, then above ``hi``
    gives ``hi``; a NaN anywhere propagates. Equal to ``np.clip`` by value
    (numpy's ``maximum``/``minimum`` pick a signed zero their own way)."""
    t = np.where((x >= lo) | np.isnan(x), x, lo)
    return np.where((t <= hi) | np.isnan(t), t, hi)


#: ``name -> (scalar reference, numpy twin, exact)``. ``exact`` functions
#: round once (or not at all) by definition, so every target must return the
#: reference bit for bit.
UNARY = {
    "exp2": (_libm("exp2", 1), np.exp2, False),
    "expm1": (_libm("expm1", 1), np.expm1, False),
    "log2": (_libm("log2", 1), np.log2, False),
    "log10": (_libm("log10", 1), np.log10, False),
    "log1p": (_libm("log1p", 1), np.log1p, False),
    "cbrt": (_libm("cbrt", 1), np.cbrt, False),
    "sinh": (_libm("sinh", 1), np.sinh, False),
    "cosh": (_libm("cosh", 1), np.cosh, False),
    "asinh": (_libm("asinh", 1), np.arcsinh, False),
    "acosh": (_libm("acosh", 1), np.arccosh, False),
    "atanh": (_libm("atanh", 1), np.arctanh, False),
    "erf": (_libm("erf", 1), np.vectorize(math.erf, otypes=[float]), False),
    "erfc": (_libm("erfc", 1), np.vectorize(math.erfc, otypes=[float]), False),
    "ceil": (_libm("ceil", 1), np.ceil, True),
    "trunc": (_libm("trunc", 1), np.trunc, True),
    "round": (_libm("round", 1), c_round, True),
    "rint": (_libm("rint", 1), np.rint, True),
    "sign": (np.sign, np.sign, True),
}
PREDICATES = {"isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite}
BINARY = {
    "hypot": (_libm("hypot", 2), np.hypot, False),
    "copysign": (_libm("copysign", 2), np.copysign, True),
    "fmod": (_libm("fmod", 2), np.fmod, True),
    "remainder": (np.remainder, np.remainder, True),
    "fdim": (_libm("fdim", 2), None, True),
}
TERNARY = {
    "fma": (_libm("fma", 3), None, True),
    "clip": (_clip, np.clip, True),
}

#: The ULP budget for an inexact function off the bit-identical host build
#: (the device's libdevice, the default profile's vector packets), against
#: glibc: aether's measured device-vs-host bounds.
ULP = {"erfc": 4, "cbrt": 3}
ULP_DEFAULT = 2

_SUB = 2.2250738585072014e-308 / 4          # a subnormal
SPECIAL = [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, _SUB, -_SUB,
           1e-300, -1e-300, 1e-8, -1e-8, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5,
           0.49999999999999994, -0.49999999999999994, 1.0, -1.0,
           1.0000000000000002, 0.9999999999999999, -0.9999999999999999,
           3.0, -3.0, 6.0, -6.0, 27.0, 30.0, -30.0, 26.5, 100.0, 709.0, 711.0,
           -745.0, 1e15 + 0.5, 4503599627370497.0, 1e300, -1e300]


def _unary_corpus() -> np.ndarray:
    rng = np.random.default_rng(20261004)
    mags = 10.0 ** rng.uniform(-6, 2.5, 160)
    return np.concatenate([SPECIAL, mags, -mags, rng.uniform(-1, 1, 40)])


BIN_VALUES = [0.0, -0.0, 1.0, -1.0, 2.5, -2.5, 3.0, -0.75, 0.3, 7.0, _SUB, 1e300,
              np.inf, -np.inf, np.nan]
TER_VALUES = [0.0, -0.0, 1.0, -2.5, 3.0, 0.1, _SUB, np.inf, -np.inf, np.nan]

XU = _unary_corpus()
AB = np.array(list(itertools.product(BIN_VALUES, repeat=2))).T
ABC = np.array(list(itertools.product(TER_VALUES, repeat=3))).T


@hawk.kernel
def unary_all(x: Scalar, y: Mutable[Vector[21]]):
    """Every unary function and class test of the parity set, one per row."""
    y = m.vec(
        m.exp2(x), m.expm1(x), m.log2(x), m.log10(x), m.log1p(x), m.cbrt(x),
        m.sinh(x), m.cosh(x), m.asinh(x), m.acosh(x), m.atanh(x), m.erf(x),
        m.erfc(x), m.ceil(x), m.trunc(x), m.round(x), m.rint(x), m.sign(x),
        m.select(m.isnan(x), 1.0, 0.0), m.select(m.isinf(x), 1.0, 0.0),
        m.select(m.isfinite(x), 1.0, 0.0))


@hawk.kernel
def binary_all(a: Scalar, b: Scalar, y: Mutable[Vector[5]]):
    """Every binary function of the parity set, one per row."""
    y = m.vec(m.hypot(a, b), m.copysign(a, b), m.fmod(a, b),
                   m.remainder(a, b), m.fdim(a, b))


@hawk.kernel
def ternary_all(a: Scalar, b: Scalar, c: Scalar, y: Mutable[Vector[2]]):
    """``fma`` and ``clip``."""
    y = m.vec(m.fma(a, b, c), m.clip(a, b, c))


@hawk.kernel
def rank1_lifted(v: Vector[3], lo: Scalar, y: Mutable[Vector[3]], z: Mutable[Vector[3]]):
    """A rank-1 operand maps component by component, a rank-0 one broadcasts."""
    y = m.expm1(v) + m.remainder(v, lo)
    z = m.clip(v, lo, 1.0)


def _ordered(x: np.ndarray) -> list:
    bits = np.asarray(x, dtype=np.float64).view(np.int64)
    return [int(b) if b >= 0 else -(int(b) - (-(1 << 63))) for b in bits]


def _ulps(got, want) -> np.ndarray:
    got, want = np.asarray(got, float), np.asarray(want, float)
    out = []
    for g, w, og, ow in zip(got, want, _ordered(got), _ordered(want)):
        if np.isnan(g) or np.isnan(w):
            out.append(0 if np.isnan(g) and np.isnan(w) else np.inf)
        elif np.isinf(g) or np.isinf(w):
            out.append(0 if g == w else np.inf)
        else:
            out.append(abs(og - ow))
    return np.asarray(out, dtype=float)


def _assert_bitwise(name, got, want, inputs):
    got, want = np.asarray(got, float), np.asarray(want, float)
    same = (got.view(np.int64) == want.view(np.int64)) | (np.isnan(got) & np.isnan(want))
    bad = np.flatnonzero(~same)
    assert bad.size == 0, (
        f"{name}: {bad.size} result(s) differ from the reference bit for bit, e.g. "
        f"inputs {[tuple(np.atleast_1d(i)[k] for i in inputs) for k in bad[:4]]}: "
        f"got {got[bad[:4]]!r}, want {want[bad[:4]]!r}")


def _assert_ulps(name, got, want, inputs):
    budget = ULP.get(name, ULP_DEFAULT)
    d = _ulps(got, want)
    bad = np.flatnonzero(d > budget)
    assert bad.size == 0, (
        f"{name}: {bad.size} result(s) beyond {budget} ULP, e.g. inputs "
        f"{[tuple(np.atleast_1d(i)[k] for i in inputs) for k in bad[:4]]}: got "
        f"{np.asarray(got)[bad[:4]]!r}, want {np.asarray(want)[bad[:4]]!r} "
        f"({d[bad[:4]]} ULP)")


def _references():
    """``[(name, inputs, reference, numpy twin, exact)]`` over the three kernels."""
    rows = []
    with np.errstate(all="ignore"):
        for name, (ref, twin, exact) in UNARY.items():
            rows.append((name, (XU,), ref(XU), twin(XU), exact))
        for name, fn in PREDICATES.items():
            rows.append((name, (XU,), fn(XU).astype(float), fn(XU).astype(float), True))
        for name, (ref, twin, exact) in BINARY.items():
            rows.append((name, tuple(AB), ref(*AB), None if twin is None else twin(*AB),
                         exact))
        for name, (ref, twin, exact) in TERNARY.items():
            rows.append((name, tuple(ABC), ref(*ABC), None if twin is None
                         else twin(*ABC), exact))
    return rows


REFS = _references()
NAMES = [r[0] for r in REFS]


def _run_all(plans, xp) -> dict:
    """Run the three kernels through their plans; ``name -> result row``."""
    out = {}
    blocks = (plans[0].run(x=xp.asarray(XU)),
              plans[1].run(a=xp.asarray(AB[0]), b=xp.asarray(AB[1])),
              plans[2].run(a=xp.asarray(ABC[0]), b=xp.asarray(ABC[1]),
                           c=xp.asarray(ABC[2])))
    rows = [np.asarray(b.get() if hasattr(b, "get") else b) for b in blocks]
    names = [list(UNARY) + list(PREDICATES), list(BINARY), list(TERNARY)]
    for block, row_names in zip(rows, names):
        for k, name in enumerate(row_names):
            out[name] = block[k]
    out.update(_run_matrix(plans[3:], xp))
    return out


def _deploy(monkeypatch, cache_dir, profile=None, targets=None):
    import eagle

    if profile is not None:
        monkeypatch.setenv("HAWK_HOST_PROFILE", profile)
    return eagle.deploy([unary_all, binary_all, ternary_all, *MAT_KERNELS],
                        cache_dir=cache_dir, targets=targets)


@pytest.fixture(scope="module")
def host_native(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        plans = _deploy(mp, str(tmp_path_factory.mktemp("hm_native")), "native",
                        ("host",))
        return _run_all(plans, np)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def host_default(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        mp.delenv("HAWK_HOST_PROFILE", raising=False)
        plans = _deploy(mp, str(tmp_path_factory.mktemp("hm_default")), None, ("host",))
        return _run_all(plans, np)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def device(tmp_path_factory):
    import cupy as cp

    mp = pytest.MonkeyPatch()
    try:
        plans = _deploy(mp, str(tmp_path_factory.mktemp("hm_device")), None,
                        ("host", "cuda"))
        return _run_all(plans, cp)
    finally:
        mp.undo()


@pytest.mark.parametrize("row", REFS, ids=NAMES)
def test_host_native_is_the_scalar_reference_bit_for_bit(host_native, row):
    name, inputs, ref, _twin, _exact = row
    _assert_bitwise(name, host_native[name], ref, inputs)


@pytest.mark.parametrize("row", REFS, ids=NAMES)
def test_host_default_profile_within_the_ulp_rule(host_default, row):
    name, inputs, ref, _twin, exact = row
    (_assert_bitwise if exact else _assert_ulps)(name, host_default[name], ref, inputs)


@pytest.mark.gpu
@pytest.mark.parametrize("row", REFS, ids=NAMES)
def test_device_within_the_ulp_rule(device, row):
    name, inputs, ref, _twin, exact = row
    (_assert_bitwise if exact else _assert_ulps)(name, device[name], ref, inputs)


@pytest.mark.parametrize("row", [r for r in REFS if r[3] is not None],
                         ids=[r[0] for r in REFS if r[3] is not None])
def test_the_reference_agrees_with_numpy(row):
    """The scalar reference IS numpy's answer (bitwise for the exact set, within
    the ULP rule otherwise), so the host rows above are numpy parity. ``clip``
    compares by value: numpy's ``maximum(-0.0, 0.0)`` keeps the first zero."""
    name, inputs, ref, twin, exact = row
    if name == "clip":
        same = (ref == twin) | (np.isnan(ref) & np.isnan(twin))
        assert same.all()
        return
    (_assert_bitwise if exact else _assert_ulps)(name, ref, twin, inputs)


def test_round_is_c_round_not_numpy_round():
    """``round`` rounds halves away from zero; numpy's ``np.round`` is ``rint``."""
    halves = np.array([0.5, 1.5, 2.5, -0.5, -2.5])
    np.testing.assert_array_equal(UNARY["round"][0](halves), [1.0, 2.0, 3.0, -1.0, -3.0])
    np.testing.assert_array_equal(np.round(halves), UNARY["rint"][0](halves))


def test_domain_edges_are_in_the_corpus():
    """Including the NaN row ``sign`` needs: aether's ``sign(NaN)`` is 0, so the
    emitted ``sign`` guards it (``isnan(x) ? x : sign(x)``)."""
    assert np.isnan(XU).any()
    for v in (-1.0, 1.0, -0.0, np.inf, 5e-324, 30.0):
        assert np.any((XU == v) & (np.signbit(XU) == np.signbit(v)))


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_rank1_operands_map_component_wise(tmp_path, target):
    import eagle

    rng = np.random.default_rng(3)
    v = rng.uniform(-2, 2, (3, 9))
    lo = rng.uniform(0.3, 0.9, 9)
    plan = eagle.deploy(rank1_lifted, cache_dir=str(tmp_path),
                        targets=("host",) if target == "host" else ("host", "cuda"))
    if target == "host":
        result = plan.run(v=v, lo=lo)
        y, z = result["y"], result["z"]
    else:
        import cupy as cp
        result = plan.run(v=cp.asarray(v), lo=cp.asarray(lo))
        y, z = cp.asnumpy(result["y"]), cp.asnumpy(result["z"])
    np.testing.assert_allclose(y, np.expm1(v) + np.remainder(v, lo), rtol=1e-14)
    np.testing.assert_array_equal(z, np.clip(v, lo, 1.0))


def test_class_tests_on_a_vector_are_a_bool_vector():
    from hawk.ir import Leaf
    from hawk.types import TensorType

    v = hawk.Value(Leaf("vocab_read", "vec_in", "v", TensorType((3,), "f64")))
    assert m.isnan(v).ttype == TensorType((3,), "bool")


# --------------------------------------------------------------------------- #
# Matrix operands: every element-wise function on a Matrix[2, 3], the parity
# set and the functions that predate it, against the same scalar references.
# --------------------------------------------------------------------------- #
#: The element-wise functions that predate the parity set:
#: ``name -> (scalar reference, exact)``.
OLD_UNARY = {
    "exp": (_libm("exp", 1), False), "log": (_libm("log", 1), False),
    "sin": (_libm("sin", 1), False), "cos": (_libm("cos", 1), False),
    "tan": (_libm("tan", 1), False), "asin": (_libm("asin", 1), False),
    "acos": (_libm("acos", 1), False), "atan": (_libm("atan", 1), False),
    "tanh": (_libm("tanh", 1), False), "sqrt": (_libm("sqrt", 1), True),
    "rsqrt": (lambda x: 1.0 / _libm("sqrt", 1)(x), False),
    "floor": (_libm("floor", 1), True), "abs": (np.abs, True),
}
OLD_BINARY = {
    "atan2": (_libm("atan2", 2), False), "power": (_libm("pow", 2), False),
    "minimum": (_libm("fmin", 2), True), "maximum": (_libm("fmax", 2), True),
}


@hawk.kernel
def mat_unary_a(X: Matrix[2, 3], y_exp2: Mutable[Matrix[2, 3]],
                y_expm1: Mutable[Matrix[2, 3]], y_log2: Mutable[Matrix[2, 3]],
                y_log10: Mutable[Matrix[2, 3]], y_log1p: Mutable[Matrix[2, 3]],
                y_cbrt: Mutable[Matrix[2, 3]], y_sinh: Mutable[Matrix[2, 3]],
                y_cosh: Mutable[Matrix[2, 3]], y_asinh: Mutable[Matrix[2, 3]]):
    """The parity set's unaries on a matrix, part one."""
    y_exp2 = m.exp2(X)
    y_expm1 = m.expm1(X)
    y_log2 = m.log2(X)
    y_log10 = m.log10(X)
    y_log1p = m.log1p(X)
    y_cbrt = m.cbrt(X)
    y_sinh = m.sinh(X)
    y_cosh = m.cosh(X)
    y_asinh = m.asinh(X)


@hawk.kernel
def mat_unary_b(X: Matrix[2, 3], y_acosh: Mutable[Matrix[2, 3]],
                y_atanh: Mutable[Matrix[2, 3]], y_erf: Mutable[Matrix[2, 3]],
                y_erfc: Mutable[Matrix[2, 3]], y_ceil: Mutable[Matrix[2, 3]],
                y_trunc: Mutable[Matrix[2, 3]], y_round: Mutable[Matrix[2, 3]],
                y_rint: Mutable[Matrix[2, 3]], y_sign: Mutable[Matrix[2, 3]]):
    """The parity set's unaries on a matrix, part two."""
    y_acosh = m.acosh(X)
    y_atanh = m.atanh(X)
    y_erf = m.erf(X)
    y_erfc = m.erfc(X)
    y_ceil = m.ceil(X)
    y_trunc = m.trunc(X)
    y_round = m.round(X)
    y_rint = m.rint(X)
    y_sign = m.sign(X)


@hawk.kernel
def mat_unary_c(X: Matrix[2, 3], y_isnan: Mutable[Matrix[2, 3]],
                y_isinf: Mutable[Matrix[2, 3]], y_isfinite: Mutable[Matrix[2, 3]],
                y_exp: Mutable[Matrix[2, 3]], y_log: Mutable[Matrix[2, 3]],
                y_sin: Mutable[Matrix[2, 3]], y_cos: Mutable[Matrix[2, 3]],
                y_tan: Mutable[Matrix[2, 3]]):
    """The class tests as a bool-matrix mask with broadcast branches, and the
    older transcendental unaries."""
    y_isnan = m.where(m.isnan(X), 1.0, 0.0)
    y_isinf = m.where(m.isinf(X), 1.0, 0.0)
    y_isfinite = m.where(m.isfinite(X), 1.0, 0.0)
    y_exp = m.exp(X)
    y_log = m.log(X)
    y_sin = m.sin(X)
    y_cos = m.cos(X)
    y_tan = m.tan(X)


@hawk.kernel
def mat_unary_d(X: Matrix[2, 3], y_asin: Mutable[Matrix[2, 3]],
                y_acos: Mutable[Matrix[2, 3]], y_atan: Mutable[Matrix[2, 3]],
                y_tanh: Mutable[Matrix[2, 3]], y_sqrt: Mutable[Matrix[2, 3]],
                y_rsqrt: Mutable[Matrix[2, 3]], y_floor: Mutable[Matrix[2, 3]],
                y_abs: Mutable[Matrix[2, 3]]):
    """The rest of the older unaries."""
    y_asin = m.asin(X)
    y_acos = m.acos(X)
    y_atan = m.atan(X)
    y_tanh = m.tanh(X)
    y_sqrt = m.sqrt(X)
    y_rsqrt = m.rsqrt(X)
    y_floor = m.floor(X)
    y_abs = m.abs(X)


@hawk.kernel
def mat_binary(A: Matrix[2, 3], B: Matrix[2, 3], y_hypot: Mutable[Matrix[2, 3]],
               y_copysign: Mutable[Matrix[2, 3]], y_fmod: Mutable[Matrix[2, 3]],
               y_remainder: Mutable[Matrix[2, 3]], y_fdim: Mutable[Matrix[2, 3]],
               y_atan2: Mutable[Matrix[2, 3]], y_power: Mutable[Matrix[2, 3]],
               y_minimum: Mutable[Matrix[2, 3]], y_maximum: Mutable[Matrix[2, 3]]):
    """Every binary, matrix with matrix."""
    y_hypot = m.hypot(A, B)
    y_copysign = m.copysign(A, B)
    y_fmod = m.fmod(A, B)
    y_remainder = m.remainder(A, B)
    y_fdim = m.fdim(A, B)
    y_atan2 = m.atan2(A, B)
    y_power = m.power(A, B)
    y_minimum = m.minimum(A, B)
    y_maximum = m.maximum(A, B)


@hawk.kernel
def mat_ternary(A: Matrix[2, 3], B: Matrix[2, 3], C: Matrix[2, 3],
                y_fma: Mutable[Matrix[2, 3]], y_clip: Mutable[Matrix[2, 3]]):
    """``fma`` and ``clip``, three matrices."""
    y_fma = m.fma(A, B, C)
    y_clip = m.clip(A, B, C)


@hawk.kernel
def mat_mixed(A: Matrix[2, 3], s: Scalar, y_hypot: Mutable[Matrix[2, 3]],
              y_fmod: Mutable[Matrix[2, 3]], y_atan2: Mutable[Matrix[2, 3]],
              y_power: Mutable[Matrix[2, 3]], y_minimum: Mutable[Matrix[2, 3]],
              y_fma: Mutable[Matrix[2, 3]], y_clip: Mutable[Matrix[2, 3]],
              y_where: Mutable[Matrix[2, 3]]):
    """A rank-0 operand broadcast against a matrix, on either side."""
    y_hypot = m.hypot(A, s)
    y_fmod = m.fmod(s, A)
    y_atan2 = m.atan2(s, A)
    y_power = m.power(A, s)
    y_minimum = m.minimum(s, A)
    y_fma = m.fma(A, s, A)
    y_clip = m.clip(A, s, 1.0)
    y_where = m.where(A > s, A, s)


MAT_KERNELS = (mat_unary_a, mat_unary_b, mat_unary_c, mat_unary_d, mat_binary,
               mat_ternary, mat_mixed)


def _mat_plane(values) -> np.ndarray:
    """``values`` padded with its own head to a multiple of six and laid out as
    a ``Matrix[2, 3]`` plane: ``(6, n)``, one sample per column."""
    values = np.asarray(values, dtype=float)
    pad = (-len(values)) % 6
    return np.concatenate([values, values[:pad]]).reshape(6, -1)


MXU = _mat_plane(XU)
MAB = (_mat_plane(AB[0]), _mat_plane(AB[1]))
MABC = (_mat_plane(ABC[0]), _mat_plane(ABC[1]), _mat_plane(ABC[2]))
MS = np.resize(np.asarray(BIN_VALUES), MAB[0].shape[1])[None, :]


def _mat_inputs() -> list:
    """Per kernel of :data:`MAT_KERNELS`: its input planes by name."""
    unary = {"X": MXU}
    return [unary, unary, unary, unary, {"A": MAB[0], "B": MAB[1]},
            {"A": MABC[0], "B": MABC[1], "C": MABC[2]},
            {"A": MAB[0], "s": MS[0]}]


def _mat_references() -> list:
    """``[(row id, inputs, reference, exact)]``, one per matrix output."""
    rows = []
    unary = {**{k: (r, e) for k, (r, _t, e) in UNARY.items()}, **OLD_UNARY}
    with np.errstate(all="ignore"):
        for name, (ref, exact) in unary.items():
            rows.append((f"mat-{name}", (MXU,), ref(MXU), exact))
        for name, fn in PREDICATES.items():
            rows.append((f"mat-{name}", (MXU,), fn(MXU).astype(float), True))
        binary = {**{k: (r, e) for k, (r, _t, e) in BINARY.items()}, **OLD_BINARY}
        for name, (ref, exact) in binary.items():
            rows.append((f"mat-{name}", MAB, ref(*MAB), exact))
        for name, (ref, _t, exact) in TERNARY.items():
            rows.append((f"mat-{name}", MABC, ref(*MABC), exact))
        a, s = MAB[0], MS
        mixed = {"hypot": (BINARY["hypot"][0](a, s), False),
                 "fmod": (BINARY["fmod"][0](s, a), True),
                 "atan2": (OLD_BINARY["atan2"][0](s, a), False),
                 "power": (OLD_BINARY["power"][0](a, s), False),
                 "minimum": (OLD_BINARY["minimum"][0](s, a), True),
                 "fma": (TERNARY["fma"][0](a, s, a), True),
                 "clip": (_clip(a, s, 1.0), True),
                 "where": (np.where(a > s, a, s), True)}
        for name, (ref, exact) in mixed.items():
            rows.append((f"mixed-{name}", (a, np.broadcast_to(s, a.shape)), ref, exact))
    return rows


MAT_REFS = _mat_references()
MAT_NAMES = [r[0] for r in MAT_REFS]


def _run_matrix(plans, xp) -> dict:
    """Run :data:`MAT_KERNELS`; ``row id -> result plane``."""
    out = {}
    prefixes = ["mat"] * 6 + ["mixed"]
    for plan, inputs, prefix in zip(plans, _mat_inputs(), prefixes):
        got = plan.run(**{k: xp.asarray(v) for k, v in inputs.items()})
        names = [nm for role, nm in plan.plugin.arg_spec if role == "mutable"]
        got = got if isinstance(got, dict) else {names[0]: got}
        for nm in names:
            g = got[nm]
            out[f"{prefix}-{nm[2:]}"] = np.asarray(g.get() if hasattr(g, "get") else g)
    return out


#: ``fmin``/``fmax`` may return either zero of a ``(+0, -0)`` pair (IEEE leaves
#: it open; glibc and aether choose differently), so these compare by value.
SIGNED_ZERO_TIE = ("mat-minimum", "mat-maximum", "mixed-minimum")


def _assert_matrix(name, got, ref, inputs, exact):
    got, ref = np.ravel(got), np.ravel(ref)
    flat = [np.ravel(i) for i in inputs]
    if name in SIGNED_ZERO_TIE:
        same = (got == ref) | (np.isnan(got) & np.isnan(ref))
        assert same.all(), f"{name}: {np.flatnonzero(~same)[:4]} differ by value"
        return
    (_assert_bitwise if exact else _assert_ulps)(name, got, ref, flat)


@pytest.mark.parametrize("row", MAT_REFS, ids=MAT_NAMES)
def test_matrix_host_native_is_the_scalar_reference_bit_for_bit(host_native, row):
    name, inputs, ref, _exact = row
    _assert_matrix(name, host_native[name], ref, inputs, True)


@pytest.mark.parametrize("row", MAT_REFS, ids=MAT_NAMES)
def test_matrix_host_default_profile_within_the_ulp_rule(host_default, row):
    name, inputs, ref, exact = row
    _assert_matrix(name, host_default[name], ref, inputs, exact)


@pytest.mark.gpu
@pytest.mark.parametrize("row", MAT_REFS, ids=MAT_NAMES)
def test_matrix_device_within_the_ulp_rule(device, row):
    name, inputs, ref, exact = row
    _assert_matrix(name, device[name], ref, inputs, exact)


def test_every_elementwise_function_has_a_matrix_row():
    """The matrix rows cover the parity set and the older element-wise names."""
    covered = {r[0].split("-", 1)[1] for r in MAT_REFS if r[0].startswith("mat-")}
    assert covered == (set(UNARY) | set(PREDICATES) | set(BINARY) | set(TERNARY)
                       | set(OLD_UNARY) | set(OLD_BINARY))


def _leaf(shape: tuple, name: str = "a"):
    from hawk.ir import Leaf
    from hawk.types import TensorType

    role = {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(shape)]
    return hawk.Value(Leaf("vocab_read", role, name, TensorType(shape, "f64")))


@pytest.mark.parametrize("spell", [
    lambda M, V, s: m.hypot(M, V),
    lambda M, V, s: m.atan2(V, M),
    lambda M, V, s: m.clip(M, V, s),
    lambda M, V, s: m.fma(s, M, V),
    lambda M, V, s: m.copysign(M, _leaf((3, 2), "T")),
    lambda M, V, s: m.where(m.isnan(V), M, M),
    lambda M, V, s: M > V,
    lambda M, V, s: M + V,
], ids=["hypot", "atan2", "clip", "fma", "transposed", "where-mask", "compare",
        "add"])
def test_mismatched_shapes_refuse_as_the_arithmetic_operators_do(spell):
    """A vector against a matrix (or a 2x3 against a 3x2) is refused in ONE set
    of words, the ``+`` operator's own (the last row)."""
    from hawk.ir import HawkError

    with pytest.raises(HawkError, match=r"do not broadcast — an elementwise op takes "
                       r"equal shapes or one rank-0 operand"):
        spell(_leaf((2, 3), "M"), _leaf((3,), "V"), _leaf((), "s"))


def test_rank0_operands_still_mint_one_op():
    """The entry-by-entry map is for vector and matrix operands only: a rank-0
    call is the op itself, so a rank-0 body's IR (and its digest) is unchanged."""
    a, b = _leaf((), "a"), _leaf((), "b")
    for value, kind in ((m.atan2(a, b), "atan2"), (m.expm1(a), "expm1"),
                        (a < b, "lt"), (m.lnot(a < b), "lnot"),
                        (m.land(a < b, b < a), "land")):
        assert value.node.kind == kind


def test_matrix_class_test_is_a_bool_matrix():
    from hawk.types import TensorType

    M = _leaf((2, 3), "M")
    assert m.isinf(M).ttype == TensorType((2, 3), "bool")
    assert (M >= 0.0).ttype == TensorType((2, 3), "bool")
    assert m.where(m.isinf(M), 0.0, M).ttype == TensorType((2, 3), "f64")


# --------------------------------------------------------------------------- #
# Derivatives on the compiled targets.
# --------------------------------------------------------------------------- #
N_FD = 16
K_FD = 25


@hawk.kernel
def smooth(x: Scalar, a: Scalar, b: Scalar, c: Scalar, y: Mutable[Vector[25]]):
    """Every differentiable function of the set (and the zero-derivative ones),
    on a domain where none sits on a knot."""
    y = m.vec(
        m.exp2(x), m.expm1(x), m.log2(x), m.log10(x), m.log1p(x), m.cbrt(x),
        m.sinh(x), m.cosh(x), m.asinh(x), m.acosh(x + 1.5), m.atanh(x - 0.5),
        m.erf(x), m.erfc(x), m.ceil(x), m.trunc(x), m.round(x), m.rint(x),
        m.sign(a), m.hypot(a, b), m.copysign(a, b), m.fmod(a, b),
        m.remainder(a, b), m.fdim(a, b), m.fma(a, b, c), m.clip(a, b, b + c))


def _fd_point():
    rng = np.random.default_rng(20261005)
    signs_a = np.where(np.arange(N_FD) % 2 == 0, 1.0, -1.0)
    signs_b = np.where(np.arange(N_FD) % 4 < 2, 1.0, -1.0)
    return {"x": rng.uniform(0.3, 0.45, N_FD),
            "a": signs_a * rng.uniform(2.1, 2.3, N_FD),
            "b": signs_b * rng.uniform(0.95, 1.0, N_FD),
            "c": rng.uniform(1.0, 2.0, N_FD)}


def _target_runner(target, tmp_path):
    import eagle

    targets = ("host",) if target == "host" else ("host", "cuda")
    plans = eagle.deploy([smooth, Kernel("smooth_jvp", jvp(smooth)),
                          Kernel("smooth_vjp", vjp(smooth))],
                         cache_dir=str(tmp_path), targets=targets)
    if target == "host":
        return plans, (lambda a: a), (lambda a: np.asarray(a))
    import cupy as cp
    return plans, cp.asarray, cp.asnumpy


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_forward_and_reverse_match_a_central_difference(tmp_path, target):
    (primal, fwd, rev), put, get = _target_runner(target, tmp_path)
    p = _fd_point()
    rng = np.random.default_rng(7)
    h = 1e-6

    def run_primal(env):
        return get(primal.run(**{k: put(v) for k, v in env.items()}))

    # forward: one random tangent direction per sample, every row checked
    t = {k: rng.normal(size=N_FD) for k in p}
    plus = run_primal({k: p[k] + h * t[k] for k in p})
    minus = run_primal({k: p[k] - h * t[k] for k in p})
    fd = (plus - minus) / (2 * h)
    got = get(fwd.run(**{k: put(v) for k, v in p.items()},
                      **{f"dot_{k}": put(v) for k, v in t.items()}))
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-7)
    # the zero-derivative rows are exactly zero, not merely small
    np.testing.assert_array_equal(got[13:18], 0.0)

    # reverse: a random adjoint per row, each input's gradient against its own FD
    bar = rng.normal(size=(K_FD, N_FD))
    result = rev.run(**{k: put(v) for k, v in p.items()}, bar_y=put(bar))
    grads = {nm: get(g) for nm, g in result.items()}
    assert sorted(grads) == ["bar_a", "bar_b", "bar_c", "bar_x"]
    for name in ("x", "a", "b", "c"):
        g = grads[f"bar_{name}"]
        up = run_primal({**p, name: p[name] + h})
        dn = run_primal({**p, name: p[name] - h})
        want = np.sum(bar * (up - dn) / (2 * h), axis=0)
        np.testing.assert_allclose(g, want, rtol=1e-6, atol=1e-6,
                                   err_msg=f"bar_{name} on {target}")


# --------------------------------------------------------------------------- #
# Each function inside a finishing kernel, through eagle.simulate.
# --------------------------------------------------------------------------- #
@hawk.kernel
def finishing_math(x: Scalar, a: Scalar, terminated: Terminated,
                   y: Mutable[Scalar], t: Mutable[Scalar]):
    """One step that uses every function, then finishes the sample."""
    u = x * 0.5
    s = (m.exp2(u) + m.expm1(u) + m.log2(x) + m.log10(x) + m.log1p(u) + m.cbrt(x)
         + m.sinh(u) + m.cosh(u) + m.asinh(x) + m.acosh(x + 1.0) + m.atanh(u * 0.5)
         + m.erf(x) + m.erfc(x) + m.ceil(x) + m.trunc(x) + m.round(x) + m.rint(x)
         + m.sign(a) + m.hypot(x, a) + m.copysign(x, a) + m.fmod(a, x)
         + m.remainder(a, x) + m.fdim(x, a) + m.fma(x, a, u) + m.clip(a, 0.0, x)
         + m.abs(a) + m.pow(x, 1.5))
    y = m.select(m.isfinite(s), s, 0.0) + m.select(m.isnan(a), 1.0, 0.0) \
        + m.select(m.isinf(a), 1.0, 0.0)
    t1 = t + 1.0
    t = t1
    terminated = t1 >= 1.0


def _finishing_reference(x, a):
    u = x * 0.5
    erf = np.vectorize(math.erf)
    erfc = np.vectorize(math.erfc)
    return (np.exp2(u) + np.expm1(u) + np.log2(x) + np.log10(x) + np.log1p(u)
            + np.cbrt(x) + np.sinh(u) + np.cosh(u) + np.arcsinh(x)
            + np.arccosh(x + 1.0) + np.arctanh(u * 0.5) + erf(x) + erfc(x)
            + np.ceil(x) + np.trunc(x) + c_round(x) + np.rint(x) + np.sign(a)
            + np.hypot(x, a) + np.copysign(x, a) + np.fmod(a, x)
            + np.remainder(a, x) + np.maximum(x - a, 0.0) + (x * a + u)
            + np.clip(a, 0.0, x) + np.abs(a) + x ** 1.5)


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_every_function_runs_in_a_finishing_kernel_through_simulate(target):
    import eagle

    rng = np.random.default_rng(11)
    x = rng.uniform(0.2, 1.8, 64)
    a = rng.uniform(-3.0, 3.0, 64)
    if target == "host":
        put, get = (lambda v: v), np.asarray
    else:
        import cupy as cp
        put, get = cp.asarray, cp.asnumpy
    res = eagle.simulate(finishing_math, x=put(x), a=put(a),
                         y=put(np.zeros(64)), t=put(np.zeros(64)), max_steps=4)
    np.testing.assert_allclose(get(res.y), _finishing_reference(x, a), rtol=1e-13)
    np.testing.assert_array_equal(get(res.t), 1.0)


# --------------------------------------------------------------------------- #
# Matrix derivatives and a matrix kernel through eagle.simulate.
# --------------------------------------------------------------------------- #
@hawk.kernel
def smooth_mat(X: Matrix[2, 3], Y: Matrix[2, 3], B: Matrix[2, 3], s: Scalar,
               z1: Mutable[Matrix[2, 3]], z2: Mutable[Matrix[2, 3]],
               z3: Mutable[Matrix[2, 3]]):
    """The parity set (and the older functions) on matrix operands, a rank-0
    one broadcast among them, on a domain where none sits on a knot."""
    z1 = (m.exp2(X) + m.expm1(X) * m.log1p(X) + m.log2(X) + m.log10(X)
               + m.cbrt(X) + m.sinh(X) - m.cosh(X) + m.asinh(X) + m.erf(X)
               + m.erfc(X) * m.acosh(X + 1.5) + m.atanh(X - 0.5))
    z2 = (m.hypot(Y, B) + m.copysign(Y, B) + m.fmod(Y, B) + m.remainder(Y, B)
               + m.fdim(Y, B) + m.atan2(X, B) + m.hypot(X, s))
    z3 = (m.fma(X, Y, B) + m.clip(Y, B, B + s) + m.where(Y > 0.0, X * Y, B)
               + m.floor(X) + m.ceil(Y) + m.sign(Y) + m.round(X)
               + m.sqrt(X) * m.exp(X) + m.maximum(X, B) + m.power(X, s)
               + m.abs(Y) * X + m.minimum(s * 0.3, X))


def _fd_point_mat():
    rng = np.random.default_rng(20261006)
    signs = np.where(rng.uniform(size=(6, N_FD)) < 0.5, 1.0, -1.0)
    return {"X": rng.uniform(0.3, 0.45, (6, N_FD)),
            "Y": signs * rng.uniform(2.1, 2.3, (6, N_FD)),
            "B": signs[::-1] * rng.uniform(0.95, 1.0, (6, N_FD)),
            "s": rng.uniform(1.0, 2.0, N_FD)}


def _outputs(plan, got) -> dict:
    """``Plan.run``'s result as ``{name: plane}``: already a dict for
    several outputs, a bare single output wrapped under its own name."""
    if isinstance(got, dict):
        return dict(got)
    names = [nm for role, nm in plan.plugin.arg_spec if role == "mutable"]
    return {names[0]: got}


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_matrix_forward_and_reverse_match_a_central_difference(tmp_path, target):
    import eagle

    targets = ("host",) if target == "host" else ("host", "cuda")
    primal, fwd, rev = eagle.deploy(
        [smooth_mat, Kernel("smooth_mat_jvp", jvp(smooth_mat)),
         Kernel("smooth_mat_vjp", vjp(smooth_mat))],
        cache_dir=str(tmp_path), targets=targets)
    if target == "host":
        put, get = (lambda a: a), np.asarray
    else:
        import cupy as cp
        put, get = cp.asarray, cp.asnumpy
    p = _fd_point_mat()
    rng = np.random.default_rng(8)
    h = 1e-6
    outs = ("z1", "z2", "z3")

    def run_primal(env):
        res = _outputs(primal, primal.run(**{k: put(v) for k, v in env.items()}))
        return np.stack([get(res[z]) for z in outs])

    # forward: one random tangent per entry and sample, every output entry checked
    t = {k: rng.normal(size=np.shape(v)) for k, v in p.items()}
    fd = (run_primal({k: p[k] + h * t[k] for k in p})
          - run_primal({k: p[k] - h * t[k] for k in p})) / (2 * h)
    got = _outputs(fwd, fwd.run(**{k: put(v) for k, v in p.items()},
                                **{f"dot_{k}": put(v) for k, v in t.items()}))
    got = np.stack([get(got[f"dot_{z}"]) for z in outs])
    np.testing.assert_allclose(got, fd, rtol=1e-6, atol=1e-7)

    # reverse: a random adjoint per output entry; each input ENTRY's gradient
    # against its own central difference
    bar = {f"bar_{z}": rng.normal(size=(6, N_FD)) for z in outs}
    grads = _outputs(rev, rev.run(**{k: put(v) for k, v in p.items()},
                                  **{k: put(v) for k, v in bar.items()}))
    assert sorted(grads) == ["bar_B", "bar_X", "bar_Y", "bar_s"]
    weights = np.stack([bar[f"bar_{z}"] for z in outs])
    for name, value in p.items():
        g = get(grads[f"bar_{name}"])
        assert g.shape == np.shape(value)
        entries = range(6) if value.ndim == 2 else [None]
        for e in entries:
            up, dn = dict(p), dict(p)
            if e is None:
                up[name], dn[name] = value + h, value - h
            else:
                up[name], dn[name] = value.copy(), value.copy()
                up[name][e] += h
                dn[name][e] -= h
            want = np.sum(weights * (run_primal(up) - run_primal(dn)) / (2 * h),
                          axis=(0, 1))
            np.testing.assert_allclose(g if e is None else g[e], want, rtol=1e-6,
                                       atol=1e-6, err_msg=f"bar_{name}[{e}] on {target}")


@hawk.kernel
def finishing_mat(X: Matrix[2, 3], a: Scalar, terminated: Terminated,
                  y: Mutable[Matrix[2, 3]], t: Mutable[Scalar]):
    """One step over a matrix state that mixes mapped and native functions,
    a bool-matrix mask among them, then finishes the sample."""
    s = (m.expm1(X * 0.5) + m.hypot(X, a) + m.clip(X, 0.0, a) + m.atan2(X, a)
         + m.floor(X) + m.remainder(a, X) + m.fma(X, a, X) + m.sign(a) * m.cbrt(X)
         + m.erf(X) + m.exp(X) + m.minimum(X, a))
    y = m.where(m.isfinite(s), s, 0.0) + m.where(X > a, 1.0, 0.0)
    t1 = t + 1.0
    t = t1
    terminated = t1 >= 1.0


def _finishing_mat_reference(X, a):
    erf = np.vectorize(math.erf)
    s = (np.expm1(X * 0.5) + np.hypot(X, a) + np.clip(X, 0.0, a) + np.arctan2(X, a)
         + np.floor(X) + np.remainder(a, X) + (X * a + X) + np.sign(a) * np.cbrt(X)
         + erf(X) + np.exp(X) + np.fmin(X, a))
    return np.where(np.isfinite(s), s, 0.0) + np.where(X > a, 1.0, 0.0)


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_matrix_kernel_runs_through_simulate(target):
    import eagle

    rng = np.random.default_rng(12)
    X = rng.uniform(0.2, 1.8, (6, 64))
    a = rng.uniform(-3.0, 3.0, 64)
    if target == "host":
        put, get = (lambda v: v), np.asarray
    else:
        import cupy as cp
        put, get = cp.asarray, cp.asnumpy
    res = eagle.simulate(finishing_mat, X=put(X), a=put(a),
                         y=put(np.zeros((6, 64))), t=put(np.zeros(64)),
                         max_steps=4)
    np.testing.assert_allclose(get(res.y), _finishing_mat_reference(X, a[None, :]),
                               rtol=1e-13)
    np.testing.assert_array_equal(get(res.t), 1.0)
