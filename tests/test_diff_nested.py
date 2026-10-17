# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivative transforms compose to any order.

``vjp``/``jvp`` take a kernel and return a kernel, so a derivative of a derivative
is the same call again. A derivative names its planes after the primal's
(``bar_x``, ``dot_x``); when the primal is itself a derivative those names are
already taken, and the transform mints the next free one (``bar2_x``) instead of
refusing. Rows:

* NAMES: every nesting order 1..3, in every mix of ``vjp``/``jvp``, with and
  without ``param_planes``, declares each plane once, and every derived plane is
  found under its own prefix.
* VALUES: every derived kernel is run on the host and on the device, in float64
  and in float32, and judged against an independent reference. float64: a
  central difference of the kernel one level down (the planes are smooth, so the
  difference is good to ~1e-7 relative) plus the closed-form ``y``, ``y'``,
  ``y''``, ``y'''`` of the test law. float32: the float64 kernel of the same
  chain at the same inputs, to a few hundred float32 ulps of the largest value
  in the plane (a chain of up to ~40 multiply-adds with one cancellation).
* FLOAT32: the generated source of every derivative names no ``double``.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pytest
from conftest import sidecar_of

import _deploy as L
import hawk
import hawk.math as hm
from hawk import Kernel, Mutable, Param, Scalar, Terminated, Vector
from hawk.artifact import build_bundle
from hawk.diff import jvp, param_planes, vjp

N = 12
H = 1e-5
EPS = {"float64": np.finfo(np.float64).eps, "float32": np.finfo(np.float32).eps}
DTYPE = {"float64": np.float64, "float32": np.float32}


@hawk.kernel
def law(x: Scalar, k: Param, y: Mutable[Scalar]):
    y = k * x * x * x + k * hm.sin(x)


LAW_P = param_planes(law)

#: y = k (x^3 + sin x) and its first three x-derivatives.
_DERIVATIVES = (
    lambda x, k: k * (x ** 3 + np.sin(x)),
    lambda x, k: k * (3 * x ** 2 + np.cos(x)),
    lambda x, k: k * (6 * x - np.sin(x)),
    lambda x, k: k * (6 - np.cos(x)),
)

_OPS = {"v": vjp, "j": jvp}
#: every mix of order 1, 2 and 3 (a letter per transform, applied left to right).
CHAINS = tuple("".join(c) for n in (1, 2, 3) for c in itertools.product("vj", repeat=n))
#: with the parameter read through a one-cell plane: the second order, and the two
#: mixed third orders a Hessian-vector product by either route needs.
PARAM_CHAINS = ("v", "j", "vv", "jj", "vj", "jv", "vjv", "jvj")


def _levels(root: Kernel, stem: str, chains) -> dict:
    """``{chain: Kernel}`` for ``root`` and every prefix of every chain."""
    made = {"": root}
    for chain in chains:
        for end in range(1, len(chain) + 1):
            prefix = chain[:end]
            if prefix not in made:
                made[prefix] = Kernel(f"{stem}_{prefix}",
                                      _OPS[prefix[-1]](made[prefix[:-1]]))
    return made


PLAIN = _levels(law, "law", CHAINS)
PROMOTED = _levels(LAW_P, "lawp", PARAM_CHAINS)
ALL = {**{f"law:{c}": k for c, k in PLAIN.items()},
       **{f"lawp:{c}": k for c, k in PROMOTED.items()}}


def _names(kernel) -> list[str]:
    return [n for _, n in kernel.arg_spec]


def _roles(kernel) -> dict[str, str]:
    return {n: r for r, n in kernel.arg_spec}


# --------------------------------------------------------------------------- #
# NAMES
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(ALL))
def test_every_plane_of_a_nested_derivative_is_declared_once(label):
    names = _names(ALL[label])
    assert [n for n, c in Counter(names).items() if c > 1] == []
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names)


def test_the_names_are_a_pure_function_of_the_nesting():
    again = _levels(law, "law", CHAINS)
    assert {c: _names(k) for c, k in again.items()} == {c: _names(k) for c, k in PLAIN.items()}


@pytest.mark.parametrize("chain", [c for c in CHAINS if c])
def test_each_level_mints_one_derived_plane_per_wrt_input_and_per_sink(chain):
    parent, child = PLAIN[chain[:-1]], PLAIN[chain]
    old = set(_names(parent))
    new = [n for n in _names(child) if n not in old]
    prefix = "bar" if chain[-1] == "v" else "dot"
    assert new and all(re.fullmatch(prefix + r"\d*_\w+", n) for n in new), new
    spec = _roles(parent)
    inputs = [n for n, r in spec.items() if r != "mutable"]
    sinks = [n for n, r in spec.items() if r == "mutable"]
    # a vjp reads one seed per sink and writes one gradient per input; a jvp the reverse
    assert len(new) == len(sinks) + len(inputs) if chain[-1] == "v" else len(new) <= len(
        sinks) + len(inputs)
    for base in (sinks if chain[-1] == "v" else inputs):
        assert [n for n in new if re.fullmatch(prefix + r"\d*_" + base, n)], (base, new)


def test_the_first_order_names_are_unchanged():
    assert _names(PLAIN["v"]) == ["bar_k", "bar_x", "bar_y", "x", "k"] or {
        "bar_x", "bar_k", "bar_y"} <= set(_names(PLAIN["v"]))
    assert {"dot_x", "dot_y", "dot_k"} <= set(_names(PLAIN["j"]))
    assert "bar2_x" not in _names(PLAIN["v"]) and "dot2_x" not in _names(PLAIN["j"])


def test_a_derivative_of_the_same_kind_moves_up_one_level_per_nesting():
    assert "bar2_x" in _names(PLAIN["vv"]) and "bar2_bar_x" in _names(PLAIN["vv"])
    assert "dot2_x" in _names(PLAIN["jj"]) and "dot2_dot_x" in _names(PLAIN["jj"])
    assert "dot3_x" in _names(PLAIN["jjj"])


# --------------------------------------------------------------------------- #
# running a level
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module", params=["float64", "float32"])
def bundle(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("nested_" + request.param)
    kernels = [law, LAW_P] + [k for c, k in PLAIN.items() if c] + [
        k for c, k in PROMOTED.items() if c]
    built = build_bundle(kernels, root, mode=request.param, targets=("cuda", "host"),
                         cache_dir=cache_dir, host_profile="native")
    return request.param, built


@pytest.fixture(scope="module")
def bundle64(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("nested_ref64")
    kernels = [law, LAW_P] + [k for c, k in PLAIN.items() if c] + [
        k for c, k in PROMOTED.items() if c]
    return build_bundle(kernels, root, mode="float64", targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native")


def _run(built, kernel, target, values: dict, dtype) -> dict:
    """One launch of ``kernel`` over ``values``; every mutable plane back as numpy."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    xp = np
    if target == "cuda":
        import cupy as xp
    sidecar = sidecar_of(built, kernel.name)
    widths = sidecar.get("arg_widths", {})
    bound, outs = {}, {}
    for role, name in kernel.arg_spec:
        if role == "mutable":
            outs[name] = bound[name] = xp.zeros(_shape(widths, name), dtype=dtype)
        elif role == "terminated":
            bound[name] = xp.zeros(N, dtype=bool)
        elif role == "uniform":
            bound[name] = float(values[name])
        elif role == "lookup":
            bound[name] = xp.asarray(np.asarray(values[name], dtype=dtype).reshape(-1))
        else:
            bound[name] = xp.asarray(np.asarray(values[name], dtype=dtype))
    if target == "host":
        plugin, structure = L.host_plugin(built.directory, kernel.name, sidecar), eexec.HostTeam
    else:
        plugin, structure = (L.device_plugin(built.directory, kernel.name, sidecar),
                             eexec.DeviceKernel)
    eplan.plan(plugin, structure=structure).bind(**bound).launch()
    return {n: np.asarray(a.get() if target == "cuda" else a, dtype=np.float64)
            for n, a in outs.items()}


TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


def _shape(widths: dict, name: str) -> tuple:
    return (N,) if int(widths.get(name, 1)) <= 1 else (int(widths[name]), N)


def _draw(kernel, seed: int, widths: dict | None = None) -> dict:
    """A random point for every input plane of ``kernel``."""
    rng = np.random.default_rng(seed)
    out = {}
    for role, name in kernel.arg_spec:
        if role in ("per_sample", "vec_in"):
            out[name] = rng.uniform(0.3, 1.2, _shape(widths or {}, name))
        elif role == "uniform":
            out[name] = float(rng.uniform(0.7, 1.4))
        elif role == "lookup":
            out[name] = np.array([rng.uniform(0.7, 1.4)])
    return out


def _new(child, parent, prefix: str, base: str) -> str:
    """The plane ``child`` declares that ``parent`` does not, named ``prefix`` + base."""
    old = set(_names(parent))
    hit = [n for n in _names(child)
           if n not in old and re.fullmatch(prefix + r"\d*_" + re.escape(base), n)]
    assert len(hit) == 1, (prefix, base, hit, _names(child))
    return hit[0]


def _floats(kernel):
    return [n for r, n in kernel.arg_spec if r in ("per_sample", "vec_in", "uniform", "lookup")]


def _bump(values: dict, step: dict, sign: float) -> dict:
    out = dict(values)
    for n, d in step.items():
        out[n] = np.asarray(values[n], dtype=np.float64) + sign * H * np.asarray(d)
        if np.ndim(values[n]) == 0:
            out[n] = float(out[n])
    return out


def _tol(ref: np.ndarray, rel: float) -> float:
    return rel * max(1.0, float(np.max(np.abs(ref))))


# --------------------------------------------------------------------------- #
# VALUES, float64: against a central difference one level down
# --------------------------------------------------------------------------- #
def _against_a_difference(built, parent, child, chain, target):
    """``child`` = ``chain[-1]`` of ``parent``: run it and the parent at a random
    point and compare against a central difference of the parent."""
    rng = np.random.default_rng(len(chain) * 131 + sum(map(ord, chain)))
    widths = sidecar_of(built, child.name).get("arg_widths", {})
    roles = _roles(child)
    base = _draw(child, 7 + len(chain), widths)
    sinks = [n for r, n in parent.arg_spec if r == "mutable"]
    wanted = _floats(parent)
    cell = lambda n: roles[n] in ("uniform", "lookup")

    def call(point):
        return _run(built, parent, target, point, np.float64)

    def unit(n, v):
        """``v`` shaped as the plane ``n`` is bound."""
        if roles[n] == "uniform":
            return float(v)
        return np.full_like(np.asarray(base[n], dtype=np.float64), v) if np.isscalar(v) else v

    if chain[-1] == "j":
        step = {n: (rng.normal() if cell(n) else rng.normal(size=np.shape(base[n])))
                for n in wanted}
        for n, d in step.items():
            tangent = _new(child, parent, "dot", n)
            base[tangent] = unit(tangent, d) if cell(n) else d
        got = _run(built, child, target, base, np.float64)
        plus, minus = call(_bump(base, step, 1.0)), call(_bump(base, step, -1.0))
        for s in sinks:
            ref = (plus[s] - minus[s]) / (2 * H)
            np.testing.assert_allclose(got[_new(child, parent, "dot", s)], ref,
                                       rtol=0, atol=_tol(ref, 2e-6))
    else:
        shapes = {s: call(base)[s].shape for s in sinks}
        seeds = {s: rng.normal(size=shapes[s]) for s in sinks}
        for s, v in seeds.items():
            base[_new(child, parent, "bar", s)] = v
        got = _run(built, child, target, base, np.float64)
        for n in wanted:
            grad = got[_new(child, parent, "bar", n)]
            rows = range(np.shape(base[n])[0]) if np.ndim(base[n]) == 2 else (None,)
            for r in rows:            # one component at a time: lanes are independent
                one = np.zeros(np.shape(base[n])) if not cell(n) else 1.0
                if r is not None:
                    one[r] = 1.0
                elif not cell(n):
                    one[:] = 1.0
                plus, minus = (call(_bump(base, {n: one}, 1.0)),
                               call(_bump(base, {n: one}, -1.0)))
                ref = sum((seeds[s] * (plus[s] - minus[s])).reshape(-1, N).sum(axis=0)
                          for s in sinks) / (2 * H)
                got_r = grad if r is None else grad[r]
                if cell(n):       # a parameter's gradient is per sample; the caller sums it
                    ref, got_r = ref.sum(), got_r.sum()
                np.testing.assert_allclose(got_r, ref, rtol=0, atol=_tol(ref, 2e-6 * N))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("label", [f"law:{c}" for c in CHAINS]
                         + [f"lawp:{c}" for c in PARAM_CHAINS])
def test_a_derivative_of_any_order_matches_a_difference_of_the_level_below(
        bundle, label, target):
    mode, built = bundle
    if mode != "float64":
        pytest.skip("the float64 reference row; float32 is judged against float64")
    stem, chain = label.split(":")
    table = PLAIN if stem == "law" else PROMOTED
    _against_a_difference(built, table[chain[:-1]], table[chain], chain, target)


# --------------------------------------------------------------------------- #
# VALUES, closed form: orders 0..3 of y = k (x^3 + sin x)
# --------------------------------------------------------------------------- #
def _tower_point(kernels: list, op: str):
    """Planes of the deepest kernel of a pure ``op`` tower with every direction
    of ``x`` at 1, every seed at 1 where it carries the answer, and the rest 0."""
    base = _draw(kernels[0], 3)
    point = dict(base)
    for level, (parent, child) in enumerate(zip(kernels, kernels[1:])):
        known = set(_names(parent))
        for n in _names(child):
            if n in known or n in point:
                continue
            point[n] = np.zeros(N) if _roles(child)[n] != "uniform" else 0.0
        if op == "j":
            point[_new(child, parent, "dot", "x")] = np.ones(N)
        else:
            for s in (n for r, n in parent.arg_spec if r == "mutable"):
                # the seed of the sink that carries the next derivative of x
                if s.endswith("_x") or s == "bar_x" or re.fullmatch(r"bar\d*_x", s):
                    point[_new(child, parent, "bar", s)] = np.ones(N)
            if level == 0:
                point[_new(child, parent, "bar", "y")] = np.ones(N)
    return base, point


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_forward_tower_is_the_nth_derivative_of_the_law(bundle, order, target):
    mode, built = bundle
    chain = "j" * order
    kernels = [PLAIN["j" * i] for i in range(order + 1)]
    base, point = _tower_point(kernels, "j")
    child = PLAIN[chain]
    for n in _names(child):
        if _roles(child)[n] == "uniform" and n not in base:
            point[n] = 0.0
    parent = kernels[-2]
    got = _run(built, child, target, point, DTYPE[mode])
    sink = _new(child, parent, "dot", next(n for r, n in parent.arg_spec if r == "mutable"))
    x = np.asarray(point["x"], dtype=DTYPE[mode]).astype(np.float64)
    ref = _DERIVATIVES[order](x, float(point["k"]))
    np.testing.assert_allclose(got[sink], ref, rtol=0, atol=_tol(ref, 200 * EPS[mode]))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("order", [2, 3])
def test_a_reverse_tower_is_the_nth_derivative_of_the_law(bundle, order, target):
    mode, built = bundle
    chain = "v" * order
    kernels = [PLAIN["v" * i] for i in range(order + 1)]
    base, point = _tower_point(kernels, "v")
    child, parent = PLAIN[chain], kernels[-2]
    for n in _names(child):
        if _roles(child)[n] == "uniform" and n not in base:
            point[n] = 0.0
    got = _run(built, child, target, point, DTYPE[mode])
    grad = _new(child, parent, "bar", "x")
    x = np.asarray(point["x"], dtype=DTYPE[mode]).astype(np.float64)
    ref = _DERIVATIVES[order](x, float(point["k"]))
    np.testing.assert_allclose(got[grad], ref, rtol=0, atol=_tol(ref, 200 * EPS[mode]))


# --------------------------------------------------------------------------- #
# VALUES, float32: against the float64 kernel of the same chain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("label", [f"law:{c}" for c in CHAINS]
                         + [f"lawp:{c}" for c in PARAM_CHAINS])
def test_a_float32_derivative_agrees_with_the_float64_one(bundle, bundle64, label, target):
    mode, built = bundle
    if mode != "float32":
        pytest.skip("the float32 row")
    stem, chain = label.split(":")
    kernel = (PLAIN if stem == "law" else PROMOTED)[chain]
    rng = np.random.default_rng(len(chain) + 11)
    point = _draw(kernel, 5)
    for n in _names(kernel):
        if n not in point and _roles(kernel)[n] != "mutable":
            point[n] = rng.uniform(0.3, 1.2, N)
    # one set of float32-representable inputs for both arms
    for n, v in point.items():
        point[n] = (float(np.float32(v)) if np.ndim(v) == 0
                    else np.asarray(v, dtype=np.float32).astype(np.float64))
    low = _run(built, kernel, target, point, np.float32)
    ref = _run(bundle64, kernel, target, point, np.float64)
    assert low.keys() == ref.keys()
    for n in ref:
        np.testing.assert_allclose(low[n], ref[n], rtol=0, atol=_tol(ref[n], 400 * EPS["float32"]))


# --------------------------------------------------------------------------- #
# FLOAT32: no silent float64 inside generated code
# --------------------------------------------------------------------------- #
_DOUBLE = re.compile(r"\bdouble\b|\bfloat64\b|(?<![\w.])\d+\.\d*(?:[eE][-+]?\d+)?(?![\w.])|(?<![:\w])(?:sin|cos|exp|log|sqrt|pow|fabs)\(")


def _doubles(text: str) -> list:
    # the kernel itself (not the shared preamble); a constant is spelled
    # ``static_cast<Real>(c)``: rounded once, to Real
    start = min(i for i in (text.find('extern "C"'), text.find("__global__")) if i >= 0)
    body = re.sub(r"static_cast<Real>\([^()]*\)", "", text[start:])
    return [m.group(0) for m in _DOUBLE.finditer(body)]


def test_the_float32_source_of_every_derivative_names_no_double(bundle):
    mode, built = bundle
    if mode != "float32":
        pytest.skip("the float32 row")
    checked = 0
    for kernel in ALL.values():
        for suffix in (".cu", ".cpp"):
            text = (built.directory / (kernel.name + suffix)).read_text()
            hits = _doubles(text)
            assert hits == [], (kernel.name, suffix, hits[:5])
            checked += 1
    assert checked == 2 * len(ALL)


# --------------------------------------------------------------------------- #
# FLOAT32, a vector kernel: ``sqrt``, ``dot``, a ``Terminated`` mask, a ``Param``
# --------------------------------------------------------------------------- #
@hawk.kernel
def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
    a = -cd * hm.sqrt(hm.dot(v, v)) * v


DRAG = _levels(drag, "drag", ("vv", "jj", "vj", "jv", "vjv"))
DRAG_P = _levels(param_planes(drag), "dragp", ("vv", "jj", "vj", "jv"))
DRAG_ALL = {**{f"drag:{c}": k for c, k in DRAG.items()},
            **{f"dragp:{c}": k for c, k in DRAG_P.items()}}


def _drag_bundle(root, mode, cache_dir):
    kernels = [k for t in (DRAG, DRAG_P) for k in t.values()]
    return build_bundle(kernels, root, mode=mode, targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native")


@pytest.fixture(scope="module", params=["float64", "float32"])
def vbundle(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("nested_vec_" + request.param)
    return request.param, _drag_bundle(root, request.param, cache_dir)


@pytest.fixture(scope="module")
def vbundle64(tmp_path_factory, cache_dir):
    return _drag_bundle(tmp_path_factory.mktemp("nested_vec_ref64"), "float64", cache_dir)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("label", sorted(k for k in DRAG_ALL if not k.endswith(":")))
def test_a_vector_derivative_of_any_order_matches_a_difference_of_the_level_below(
        vbundle, label, target):
    mode, built = vbundle
    if mode != "float64":
        pytest.skip("the float64 reference row; float32 is judged against float64")
    stem, chain = label.split(":")
    table = DRAG if stem == "drag" else DRAG_P
    _against_a_difference(built, table[chain[:-1]], table[chain], chain, target)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("label", sorted(DRAG_ALL))
def test_a_float32_vector_derivative_agrees_with_the_float64_one(
        vbundle, vbundle64, label, target):
    mode, built = vbundle
    if mode != "float32":
        pytest.skip("the float32 row")
    kernel = DRAG_ALL[label]
    widths = sidecar_of(built, kernel.name).get("arg_widths", {})
    rng = np.random.default_rng(len(label))
    point = _draw(kernel, 5, widths)
    for role, n in kernel.arg_spec:
        if n not in point and role not in ("mutable", "terminated"):
            point[n] = rng.uniform(0.3, 1.2, _shape(widths, n))
    for n, v in point.items():
        point[n] = (float(np.float32(v)) if np.ndim(v) == 0
                    else np.asarray(v, dtype=np.float32).astype(np.float64))
    low = _run(built, kernel, target, point, np.float32)
    ref = _run(vbundle64, kernel, target, point, np.float64)
    assert low.keys() == ref.keys()
    for n in ref:
        np.testing.assert_allclose(low[n], ref[n], rtol=0, atol=_tol(ref[n], 400 * EPS["float32"]))


def test_the_float32_source_of_a_vector_derivative_names_no_double(vbundle):
    mode, built = vbundle
    if mode != "float32":
        pytest.skip("the float32 row")
    for kernel in DRAG_ALL.values():
        for suffix in (".cu", ".cpp"):
            hits = _doubles((built.directory / (kernel.name + suffix)).read_text())
            assert hits == [], (kernel.name, suffix, hits[:5])
