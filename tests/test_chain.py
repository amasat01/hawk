# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.chain``: kernels run in sequence as one kernel.

The chain's values equal the members launched one after another, on the host and on the device; its slots hold no
intermediate; every refusal names both members and the name; the estimate fences the compiled kernel.
"""

from __future__ import annotations

import linecache

import _kernels as K
import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Vector
from hawk.artifact import bundle as B
from hawk.ir import HawkError
from hawk.ir import fusion
from hawk.targets.cards import fusion as fusion_card

eagle = pytest.importorskip("eagle")


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
#: The fixture's bound: the chain against its launches, and the host against the device (sin, cos, exp, a dot;
#: x > 0 keeps exp(-a.a) + x away from cancellation, which would amplify the libraries' last-place differences).
ULPS = 4


def _generated(text: str, tag: str) -> dict:
    name = f"<{tag}>"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    ns: dict = {}
    exec(compile(text, name, "exec"), ns)          # noqa: S102 - a generated test fixture
    return ns


def fan_in(n: int, w: int = 8) -> tuple:
    """``n`` members: member j reads ``x`` and ``r{j-1}`` and writes ``r{j}``; the last reads every ``r`` in reverse."""
    src = ["import hawk", "from hawk import Scalar, Mutable, Vector, math as m", ""]
    for j in range(1, n + 1):
        reads = [f"r{i}" for i in range(1, n)] if j == n else ([f"r{j - 1}"] if j > 1 else [])
        out = "out" if j == n else f"r{j}"
        params = ", ".join(["x: Scalar", *(f"{r}: Vector[{w}]" for r in reads), f"{out}: Mutable[Vector[{w}]]"])
        src += ["@hawk.kernel", f"def k{j}({params}):", f"    acc = m.vec({', '.join(['x'] * w)}) * 0.0"]
        if j == n:
            src += [f"    acc = m.sin(acc + {r} * {0.25 + 0.001 * i:.4f})" for i, r in enumerate(reversed(reads))]
        else:
            src += [f"    acc = acc + m.sin({r} * x + {0.25 + 0.01 * j:.2f}) * m.exp(-x * x * {0.01 + 0.001 * j:.3f})"
                    for r in (reads or ["x"])]
        src += [f"    {out} = acc", ""]
    ns = _generated("\n".join(src), f"chain_fan_in_{n}_{w}")
    return tuple(ns[f"k{j}"] for j in range(1, n + 1))


def _device() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _run(kernels, where: str, dtype: str, n: int = 257) -> dict:
    rng = np.random.default_rng(5)
    planes = {"x": rng.uniform(0.1, 2.0, n).astype(dtype), "acc": rng.uniform(0.0, 1.0, n).astype(dtype),
              "a": np.zeros((3, n), dtype), "b": np.zeros(n, dtype), "y": np.zeros((3, n), dtype)}
    if where == "cuda":
        planes = {k: eagle.arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    for k in kernels:
        plan = eagle.deploy(k, scalar_type=dtype, targets=(where,))
        plan(**{name: planes[name] for _, name in k.arg_spec})
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
    got, want = _run([chained], where, dtype), _run(MEMBERS, where, dtype)
    for name in got:
        assert _ulps(got[name], want[name], dtype) <= ULPS, name
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
    device = _equals_launches("cuda", dtype, private_cache)
    host = _run([hawk.chain(*MEMBERS, name="fixture")], "host", dtype)
    for name in device:
        assert _ulps(device[name], host[name], dtype) <= ULPS, name


def test_chain_structure():
    chained = hawk.chain(*MEMBERS, name="fixture")
    assert isinstance(chained, hawk.Kernel)
    assert {name for _, name in chained.arg_spec} == {"x", "acc", "y"}
    assert hawk.chain(k1, name="one").walk.digest == k1.walk.digest
    nested = hawk.chain(hawk.chain(k1, k2, name="front"), k3, k4, name="fixture")
    assert nested.walk.digest == chained.walk.digest and nested.members == MEMBERS


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


def test_the_rule_fuses_the_device_and_cuts_the_host_at_its_register_line():
    members = fan_in(10)
    for dtype in ("float32", "float64"):
        assert fusion.cuts(members, fusion.row(dtype), dtype, 400_000) == []
    line = fusion_card("float64", "avx512")                             # 32 vector registers
    assert fusion.cuts(members, line, "float64", 100_000) == [9]       # the consumer of every r is its own segment


def _ptxas(kernel, dtype, tmp_path):
    art = B.build(kernel, tmp_path / kernel.name, targets=("cuda",), device_arch="sm_61", mode=dtype)
    return art.sidecar["ptxas"]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_chain_fence(dtype, tmp_path, private_cache, monkeypatch):
    for chained in (hawk.chain(*MEMBERS, name="fixture"), hawk.chain(*fan_in(10), name="fan_in")):
        for row in _ptxas(chained, dtype, tmp_path).values():
            assert row["spill_store_B"] == 0 and not row["spills"], row
            assert row["regs"] + row["stack_B"] / 4 <= 1.6 * row["est_regs"] + 16, row
    monkeypatch.setattr(fusion, "estimate", lambda *a, **k: 0)
    with pytest.raises(HawkError, match=r"chain 'blind' of \['k1', .*'k10'\].* before 'k10'"):
        _ptxas(hawk.chain(*fan_in(10), name="blind"), dtype, tmp_path)
