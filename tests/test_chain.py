# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.chain``: kernels run in sequence as one kernel.

The chain's values equal the members launched one after another, on the host and on the device; its slots hold no
intermediate; every refusal names both members and the name; the estimate fences the compiled kernel.
"""

from __future__ import annotations

import linecache
import pathlib

import _kernels as K
import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Vector
from conftest import assert_within, device_array, sidecar_of
from hawk.artifact import build_bundle
from hawk.errors import HawkError
from hawk import _ir



@hawk.kernel
def k1(x: Scalar, a: Mutable[Vector[3]]):
    a = m.vec(m.sin(x), m.cos(x), x * 0.5)


@hawk.kernel
def k2(a: Vector[3], x: Scalar, b: Mutable[Scalar]):
    b = m.exp(-m.dot(a, a)) + x


@hawk.kernel
def k3(b: Scalar, acc: Mutable[Scalar]):
    acc = acc + b * b


@hawk.kernel
def k4(a: Vector[3], b: Scalar, y: Mutable[Vector[3]]):
    y = a * b


MEMBERS = (k1, k2, k3, k4)
#: The bound against the long-double oracle, in units of the scalar type's eps times the output's largest magnitude:
#: the chain and its launches each answer within it, on the host and on the device (measured at most 8.3, the
#: launches and the chain alike). x > 0 keeps exp(-a.a) + x away from cancellation.
ORACLE_ULPS = 16
LD = np.longdouble


def _oracle_ulps(got, ref, dtype) -> float:
    return float(np.max(np.abs(got.astype(LD) - ref)) / (np.max(np.abs(ref)) * LD(np.finfo(dtype).eps)))


def _fixture_oracle(dtype: str, n: int = 257) -> dict:
    """``acc`` and ``y`` of :data:`MEMBERS` in long double on :func:`_run`'s inputs."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0.1, 2.0, n).astype(dtype).astype(LD)
    acc = rng.uniform(0.0, 1.0, n).astype(dtype).astype(LD)
    a = np.stack([np.sin(x), np.cos(x), x * LD("0.5")])
    b = np.exp(-(a * a).sum(axis=0)) + x
    return {"acc": acc + b * b, "y": a * b}


def _fan_in_oracle(x, n_members: int, w: int):
    """``out`` of :func:`fan_in` in long double."""
    x, rs = x.astype(LD), []
    for j in range(1, n_members):
        src = x if j == 1 else rs[-1]
        rs.append(np.zeros((w, x.size), LD) + np.sin(src * x + LD(f"{0.25 + 0.01 * j:.2f}"))
                  * np.exp(-x * x * LD(f"{0.01 + 0.001 * j:.3f}")))
    acc = np.zeros((w, x.size), LD)
    for i, r in enumerate(reversed(rs)):
        acc = np.sin(acc + r * LD(f"{0.25 + 0.001 * i:.4f}"))
    return acc


def _generated(text: str, tag: str) -> dict:
    name = f"<{tag}>"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    ns: dict = {}
    exec(compile(text, name, "exec"), ns)          # noqa: S102 - a generated test fixture
    return ns


def fan_in(n: int, w: int = 8, *, pipe: bool = False, guarded: bool = False, group: int | None = None) -> tuple:
    """``n`` members: member j reads ``x`` and ``r{j-1}`` and writes ``r{j}``; the last reads every ``r`` in reverse
    (``pipe``: only ``r{n-1}``, as the others). ``guarded``: every member declares ``terminated``; ``group``: under a
    group map, compacted ``group`` lanes at a time."""
    src = ["import hawk", "from hawk import Scalar, Mutable, Terminated, Vector, math as m",
           "from hawk.ext import Guard, Kind", "MAP = Kind('group_rows', guard=Guard(active_set=True))", ""]
    for j in range(1, n + 1):
        reads = [f"r{i}" for i in range(1, n)] if j == n and not pipe else ([f"r{j - 1}"] if j > 1 else [])
        out = "out" if j == n else f"r{j}"
        params = ", ".join(["x: Scalar", *(["terminated: Terminated"] if guarded else []),
                            *(f"{r}: Vector[{w}]" for r in reads), f"{out}: Mutable[Vector[{w}]]"])
        src += [f"@hawk.kernel(kind=MAP, group={group})" if group else "@hawk.kernel", f"def k{j}({params}):",
                f"    acc = m.vec({', '.join(['x'] * w)}) * 0.0"]
        if j == n and not pipe:
            src += [f"    acc = m.sin(acc + {r} * {0.25 + 0.001 * i:.4f})" for i, r in enumerate(reversed(reads))]
        else:
            src += [f"    acc = acc + m.sin({r} * x + {0.25 + 0.01 * j:.2f}) * m.exp(-x * x * {0.01 + 0.001 * j:.3f})"
                    for r in (reads or ["x"])]
        src += [f"    {out} = acc", ""]
    ns = _generated("\n".join(src), f"chain_fan_in_{n}_{w}_{pipe}_{guarded}_{group}")
    return tuple(ns[f"k{j}"] for j in range(1, n + 1))


def _device() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _to_device(planes: dict) -> dict:
    """The host ``planes`` as the device array library's arrays, copied by hawk's door."""
    cupy = pytest.importorskip("cupy")
    return {k: device_array(v, cupy) for k, v in planes.items()}


def _deploy_run(kernel, where: str, dtype: str, planes: dict):
    """``kernel`` deployed for ``where`` and launched once over the planes it declares, which hold the result in place:
    hawk's verbs on the host (a chain the host cuts into stages included) and on the device, where the arrays live. Returns the
    deployed plan."""
    held = {name: planes[name] for _, name in kernel.arg_spec}
    if where == "host":
        plan = hawk.deploy(kernel, scalar_type=dtype)
        plan.run(**held)
        return plan
    plan = hawk.deploy(kernel, scalar_type=dtype, targets=("host", where))
    plan(**held)
    hawk.synchronize()
    return plan


def _run(kernels, where: str, dtype: str, n: int = 257) -> dict:
    rng = np.random.default_rng(5)
    planes = {"x": rng.uniform(0.1, 2.0, n).astype(dtype), "acc": rng.uniform(0.0, 1.0, n).astype(dtype),
              "a": np.zeros((3, n), dtype), "b": np.zeros(n, dtype), "y": np.zeros((3, n), dtype)}
    if where == "cuda":
        planes = _to_device(planes)
    for k in kernels:
        _deploy_run(k, where, dtype, planes)
    out = {}
    for name in ("acc", "y"):
        held = planes[name]
        if not isinstance(held, np.ndarray):
            held = held.get()
            held = np.asarray(held.to("numpy") if hasattr(held, "to") else held)
        out[name] = held
    return out


def _ulps(got, want, dtype) -> float:
    info = np.finfo(dtype)
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want) * info.eps, info.tiny)))


def _equals_launches(where: str, dtype: str, private_cache) -> dict:
    chained = hawk.chain(*MEMBERS, name="fixture")
    got, want, ref = _run([chained], where, dtype), _run(MEMBERS, where, dtype), _fixture_oracle(dtype)
    for name in got:
        assert _oracle_ulps(got[name], ref[name], dtype) <= ORACLE_ULPS, name
        assert _oracle_ulps(want[name], ref[name], dtype) <= ORACLE_ULPS, name
        print(f"chain vs launches {where} {dtype} {name}: {_ulps(got[name], want[name], dtype):.1f} ulp")
    again = _run([chained], where, dtype)
    for name in got:
        np.testing.assert_array_equal(again[name], got[name])
    return got


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_chain_equals_launches_host(dtype, private_cache):
    _equals_launches("host", dtype, private_cache)


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_chain_equals_launches_cuda_and_host(dtype, private_cache):
    _equals_launches("cuda", dtype, private_cache)     # the host row is the CPU test's: both within the oracle


def test_chain_structure():
    chained = hawk.chain(*MEMBERS, name="fixture")
    assert isinstance(chained, hawk.Kernel)
    assert {name for _, name in chained.arg_spec} == {"x", "acc", "y"}
    assert _ir.digest(hawk.chain(k1, name="one").walk) == _ir.digest(k1.walk)
    nested = hawk.chain(hawk.chain(k1, k2, name="front"), k3, k4, name="fixture")
    assert _ir.digest(nested.walk) == _ir.digest(chained.walk) and nested.members == MEMBERS


def _with(kernel, **attrs):
    twin = hawk.Kernel(kernel.name, kernel.sinks, kernel.planes, kind=kernel.kind)
    for key, value in attrs.items():
        setattr(twin, key, value)
    return twin


def test_chain_refuses_one_name_as_two_types():
    @hawk.kernel
    def flat(a: Scalar, z: Mutable[Scalar]):
        z = a * 2.0

    with pytest.raises(HawkError, match=r"'k1' commits 'a' .* but 'flat' reads it"):
        hawk.chain(k1, flat, name="c")


def test_chain_refuses_a_second_blind_writer():
    @hawk.kernel
    def again(x: Scalar, b: Mutable[Scalar]):
        b = x * 3.0

    with pytest.raises(HawkError, match=r"'k2' and 'again' both commit 'b'"):
        hawk.chain(k1, k2, again, name="c")


def test_chain_refuses_members_that_disagree():
    with pytest.raises(HawkError, match=r"members 'k1' and 'k2' disagree on group"):
        hawk.chain(k1, _with(k2, group=32), name="c")


def test_chain_refuses_a_stepped_member():
    with pytest.raises(HawkError, match=r"member 'k3' is a stepped kernel"):
        hawk.chain(k1, k2, _with(k3, step=k3), name="c")


def test_chain_refuses_a_cross_sample_read():
    @hawk.kernel
    def fill(x: Scalar, table: Mutable[Scalar]):
        table = x * 2.0

    with pytest.raises(HawkError, match=r"'gather' reads 'table', which 'fill' commits, at another sample"):
        hawk.chain(fill, K.gather, name="c")


def _stages(kernel, dtype: str, target: str, tmp_path) -> list:
    """The kernels a bundle of ``kernel`` launches, in order."""
    bundle = build_bundle([kernel], tmp_path / f"{kernel.name}_{target}", targets=(target,), mode=dtype,
                          **({"device_arch": "sm_61"} if target == "cuda" else {}))
    side = sidecar_of(bundle, kernel.name)
    return [st["kernel"] for st in side.get("stages", ())] or [kernel.name]




def _with_isa(machine, isa, vector_isa):
    import dataclasses
    from types import MappingProxyType

    return dataclasses.replace(machine, isa=isa, facts=MappingProxyType({**machine.facts, "vector_isa": vector_isa}))








def _cut_values(target: str, dtype: str, n_members: int = 10, w: int = 16):
    members = fan_in(n_members, w)
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0.1, 2.0, n).astype(dtype)

    def planes(names):
        held = {nm: (x.copy() if nm == "x" else np.zeros((w, n), dtype)) for nm in names}
        return held if target == "host" else _to_device(held)

    def back(a):
        if isinstance(a, np.ndarray):
            return a
        h = a.get()
        return np.asarray(h.to("numpy") if hasattr(h, "to") else h)

    launched = planes({nm for k in members for _, nm in k.arg_spec})
    for k in members:
        _deploy_run(k, target, dtype, launched)
    chained = hawk.chain(*members, name="cut")
    held = planes({nm for _, nm in chained.arg_spec})
    plan = _deploy_run(chained, target, dtype, held)
    got, want, ref = back(held["out"]), back(launched["out"]), _fan_in_oracle(x, n_members, w)
    assert _oracle_ulps(got, ref, dtype) <= ORACLE_ULPS and _oracle_ulps(want, ref, dtype) <= ORACLE_ULPS
    print(f"cut chain vs launches {target} {dtype}: {_ulps(got, want, dtype):.1f} ulp")
    return plan


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_cut_chain_equals_its_launches_on_the_host(dtype, private_cache):
    _cut_values("host", dtype)




def _guarded_run(kernel, target, dead, group_live=None):
    xp = np
    if target == "cuda":
        import cupy as xp
    n = dead.size
    terminated = xp.asarray(dead)
    x = xp.asarray(np.linspace(0.1, 2.0, n))
    planes = {"x": x, "terminated": terminated,
              **{nm: xp.full((8, n), 7.0) for _, nm in kernel.arg_spec if nm not in ("x", "terminated")}}
    if group_live is not None and target == "host":
        # a host whose rule cuts the chain (avx2, sse2) runs it as stages, which ``hawk.load`` loads as one split kernel
        bundle = build_bundle([kernel], pathlib.Path(group_live) / kernel.name, targets=(target,))
        loaded = hawk.load(bundle.directory, kernel.name)
        hawk.plan(loaded).bind(**planes, **hawk.ActiveSet(dead).planes(loaded)).launch()
    elif group_live is not None:
        plan = hawk.deploy(kernel, scalar_type="float64", targets=("host", target))
        made = hawk.ActiveSet(dead).planes(plan)        # the map and count of the live groups, made on the host, copied over
        plan(**planes, **{name: device_array(value, xp) for name, value in made.items()})
        hawk.synchronize()
    else:
        _deploy_run(kernel, target, "float64", planes)
    out = planes["out"]
    return np.asarray(out.get() if hasattr(out, "get") else out).copy()


def _chain_vs_members_bound(n_members: int, live_out) -> float:
    """Bound on ``|chain - member launches|`` of :func:`fan_in` ``(n_members)``, derived (not fitted). Both builds are within
    ``gamma_n * M`` of the exact value (Higham), whatever contraction or library routine the compiler picked for each, so they
    differ by at most ``2 * gamma_n * M``. ``n`` is the longest chain of dependent operations: a member ``r_j`` adds five to
    ``r_{j-1}`` (multiply, add, sine, multiply, add), the last member takes the first read through a multiply, an add and a sine
    and every further read through an add and a sine. ``M`` is the largest magnitude of the output, which every operation
    here keeps (sine and exponential are at most 1 in magnitude, and the sums are of such terms) from growing."""
    n = 5 * (n_members - 1) + 3 + 2 * (n_members - 2)
    u = np.finfo(np.float64).eps / 2
    return 2 * (n * u / (1 - n * u)) * float(np.max(np.abs(live_out)))


def test_chain_terminated_keeps_outputs(private_cache, tmp_path):
    dead = np.zeros(70, dtype=bool)
    dead[::3] = True
    members = fan_in(10, guarded=True)
    chained = hawk.chain(*members, name="guarded")
    assert _stages(chained, "float64", "host", tmp_path) == ["guarded", "guarded_2"]
    got = _guarded_run(chained, "host", dead)
    assert np.all(got[:, dead] == 7.0) and not np.any(got[:, ~dead] == 7.0)
    planes = {"x": np.linspace(0.1, 2.0, dead.size), "terminated": dead,
              **{nm: np.full((8, dead.size), 7.0) for k in members for _, nm in k.arg_spec if nm not in ("x", "terminated")}}
    for k in members:
        _deploy_run(k, "host", "float64", planes)
    assert_within(got, planes["out"], _chain_vs_members_bound(len(members), planes["out"][:, ~dead]))










# ---- derivatives of chains (A5) ------------------------------------------------------------------------------

@hawk.kernel
def d1(x: Scalar, a: Mutable[Vector[3]]):
    a = m.vec(m.sin(x), m.cos(x), x * 0.5)


@hawk.kernel
def d2(a: Vector[3], x: Scalar, b: Mutable[Scalar]):
    b = m.exp(-m.dot(a, a)) + x


@hawk.kernel
def d3(a: Vector[3], b: Scalar, y: Mutable[Vector[3]]):
    y = a * b


def _small_tangent(x):
    """d y / d x of d1, d2, d3 in long double (``y`` is (3, n))."""
    x = x.astype(LD)
    a, da = np.stack([np.sin(x), np.cos(x), x * LD("0.5")]), np.stack([np.cos(x), -np.sin(x), np.full_like(x, LD("0.5"))])
    e = np.exp(-(a * a).sum(axis=0))
    b, db = e + x, e * (-2 * (a * da).sum(axis=0)) + 1
    return da * b + a * db


def _fan_in_tangent(x, n_members: int, w: int):
    """d out / d x of :func:`fan_in` in long double (``out`` is (w, n))."""
    x, rs = x.astype(LD), []
    for j in range(1, n_members):
        src, dsrc = (x, np.ones_like(x)) if j == 1 else rs[-1]
        u, du = src * x + LD(f"{0.25 + 0.01 * j:.2f}"), dsrc * x + src
        e = np.exp(-x * x * LD(f"{0.01 + 0.001 * j:.3f}"))
        de = e * (-2 * x * LD(f"{0.01 + 0.001 * j:.3f}"))
        rs.append((np.sin(u) * e, np.cos(u) * du * e + np.sin(u) * de))
    acc, dacc = np.zeros_like(x), np.zeros_like(x)
    for i, (r, dr) in enumerate(reversed(rs)):
        k = LD(f"{0.25 + 0.001 * i:.4f}")
        acc, dacc = np.sin(acc + r * k), np.cos(acc + r * k) * (dacc + dr * k)
    return np.broadcast_to(dacc, (w, x.size))


def _back(a):
    if isinstance(a, np.ndarray):
        return a
    a = a.get()
    return np.asarray(a.to("numpy") if hasattr(a, "to") else a)


def _call(kernel, planes, where, dtype):
    names = [nm for _, nm in kernel.arg_spec]
    held = {nm: planes[nm].astype(dtype) for nm in names}
    if where == "cuda":
        held = _to_device(held)
    _deploy_run(kernel, where, dtype, held)
    return {nm: _back(a) for nm, a in held.items()}


def _chain_derivative(case, direction, where, dtype):
    """The derivative of a chain at ``x`` along ``x`` (jvp: ``dot_x = 1``; vjp: a random seed), the derivative of its
    composed DAG, the long-double oracle and the central differences of the chain's value."""
    from hawk.diff import jvp, vjp
    from hawk.trace.kernel import Kernel

    members, out, w, tangent = case
    chained = hawk.chain(*members, name="dchain")
    dag = Kernel("ddag", chained.sinks, kind=chained.kind)       # the composed DAG: one kernel, no members
    rng = np.random.default_rng(11)
    n = 300
    x = rng.uniform(0.2, 1.5, n).astype(dtype)
    seed = rng.uniform(0.5, 1.5, (w, n))
    planes = {"x": x, "dot_x": np.ones(n), f"bar_{out}": seed, f"dot_{out}": np.zeros((w, n)), "bar_x": np.zeros(n),
              out: np.zeros((w, n))}
    derive, got_name = (jvp, f"dot_{out}") if direction == "jvp" else (vjp, "bar_x")
    derived = derive(chained, wrt=("x",), primal_unit="0" * 64)
    mine, theirs = Kernel("dchain_d", derived), Kernel("ddag_d", derive(dag, wrt=("x",), primal_unit="0" * 64))
    got, want = _call(mine, planes, where, dtype)[got_name], _call(theirs, planes, where, dtype)[got_name]
    ref = tangent(x) if direction == "jvp" else (seed.astype(LD) * tangent(x)).sum(axis=0)
    def value(shift):
        run = _call(chained, {**planes, "x": x.astype(np.float64) + shift}, "host", "float64")
        return run[out].astype(LD)
    h = 1e-6
    fd = (value(h) - value(-h)) / (2 * LD(h))
    fd = fd if direction == "jvp" else (seed.astype(LD) * fd).sum(axis=0)
    return derived, mine, got, want, ref, fd


SMALL = ((d1, d2, d3), "y", 3, _small_tangent)
ON_DEVICE = pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not _device(), reason="needs a CUDA device")])


def _check_derivative(case, direction, where, dtype):
    derived, mine, got, want, ref, fd = _chain_derivative(case, direction, where, dtype)
    assert [k.name for k in derived.members]
    to_oracle, to_dag = _oracle_ulps(got, ref, dtype), _oracle_ulps(want, ref, dtype)
    print(f"chain {direction} {where} {dtype}: {to_oracle:.1f} ulp vs oracle (dag {to_dag:.1f}), "
          f"{_ulps(got, want, dtype):.1f} ulp vs dag, {float(np.max(np.abs(got - fd)) / np.max(np.abs(fd))):.1e} vs central")
    assert to_oracle <= ORACLE_ULPS and to_dag <= ORACLE_ULPS
    assert float(np.max(np.abs(got.astype(LD) - fd) / np.max(np.abs(fd)))) < (1e-4 if dtype == "float32" else 1e-8)
    return derived, mine


@pytest.mark.parametrize("where", ["host", ON_DEVICE])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_chain_jvp_equals_dag_jvp(where, dtype, private_cache):
    _check_derivative(SMALL, "jvp", where, dtype)


@pytest.mark.parametrize("where", ["host", ON_DEVICE])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_chain_vjp_equals_dag_vjp(where, dtype, private_cache):
    _check_derivative(SMALL, "vjp", where, dtype)


def _cut_case(n_members: int = 6, w: int = 8):
    return fan_in(n_members, w), "out", w, lambda x: _fan_in_tangent(x, n_members, w)


@pytest.mark.parametrize("direction", ["jvp", "vjp"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_cut_chain_derivative_equals_dag_on_the_host(direction, dtype, private_cache, tmp_path):
    case = _cut_case()
    derived, mine = _check_derivative(case, direction, "host", dtype)
    print(f"cut chain {direction} host stages: {_stages(mine, dtype, 'host', tmp_path)} "
          f"(primal {_stages(hawk.chain(*case[0], name='dchain'), dtype, 'host', tmp_path)})")
    assert len(_stages(mine, dtype, "host", tmp_path)) >= 2


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_chain_compaction_twin(target, private_cache, tmp_path):
    dead = np.ones(100, dtype=bool)
    for g in (0, 2):
        dead[g * 32:(g + 1) * 32] = False
    one = _guarded_run(hawk.chain(*fan_in(4, pipe=True, guarded=True, group=1), name="twin"), target, dead,
                       tmp_path / "one")
    grouped = _guarded_run(hawk.chain(*fan_in(4, pipe=True, guarded=True, group=32), name="twin"), target, dead,
                           tmp_path / "grouped")
    np.testing.assert_array_equal(grouped, one)
    assert 0 < np.count_nonzero(grouped[0] != 7.0) < dead.size
