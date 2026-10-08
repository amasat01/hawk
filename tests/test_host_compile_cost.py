"""A kernel's cold build pays only for what it uses, and its two compiles overlap.

Two observations, each with a known answer:

* **Vector math only where it is called.** Under the default
  ``native-vector-math`` profile the host TU reaches aether's vector-math
  hook, whose symbols (``aether_hvm_<f>`` and its ``_ZGV...`` vector
  variants) are each kept alive by a per-function anchor instantiated only
  by a call of that function. A kernel that calls no math function emits
  none of them; a kernel calling ``sqrt`` and ``pow`` emits exactly those two
  families, every width the CPU has, scalar symbol included. Before the
  anchors every TU carried every function at every width (288 symbols on
  an AVX-512 host), the bulk of a math-free kernel's host compile time.
* **Host and device compile side by side.** ``_compile_targets`` runs one
  kernel's target compiles concurrently (at most ``$HAWK_COMPILE_JOBS``,
  default 2), returns them keyed by target whatever order they finish in,
  and on failure raises the FIRST failing target in ``targets`` order — the
  error a one-after-the-other build reports — only after every started
  compile has finished. ``HAWK_COMPILE_JOBS=1`` is the sequential build.

Plants, each turning a row red, then removed:

* ``AETHER_HVM_ATTR`` given back its blanket ``used`` --
  ``test_a_math_free_kernel_emits_no_vector_math`` fails (288 symbols);
* the anchor odr-use dropped from ``hostvec``'s ``double`` legs --
  ``test_a_math_kernel_emits_only_the_functions_it_calls`` fails (no symbol
  emitted, the loop's vector calls left undefined);
* ``max_workers=1`` forced in ``_compile_targets`` --
  ``test_the_target_compiles_overlap`` times out waiting for the second
  compile to start.
"""

from __future__ import annotations

import platform
import re
import subprocess
import threading

import pytest

import hawk
import hawk.artifact.bundle as bundle_mod
from hawk import Mutable, Param, Scalar, Terminated
from hawk import math as m
from hawk.artifact import build_bundle
from hawk.compile import toolchain as tc
from hawk.ir import HawkError

_X86 = platform.machine().lower() in ("x86_64", "amd64")
needs_x86 = pytest.mark.skipif(not _X86, reason="the vector-math hook is x86-64 only")


@hawk.kernel
def plain_step(omega: Scalar, dt: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar]):
    a = -(omega * omega) * x
    x = x + dt * v
    v = v + dt * a


@hawk.kernel
def root_pow_step(dt: Param, terminated: Terminated, x: Mutable[Scalar],
                  v: Mutable[Scalar]):
    r = m.sqrt(x * x + v * v + 1.0)
    x = x + dt * v / r
    v = v - dt * (r ** 0.75)


def _hvm_symbols(so) -> set:
    out = subprocess.run(["nm", str(so)], capture_output=True, text=True, check=True).stdout
    return {line.split()[-1] for line in out.splitlines() if "aether_hvm_" in line}


def _host_so(kernel, tmp_path):
    bundle = build_bundle([kernel], tmp_path / "b", targets=("host",),
                          cache_dir=str(tmp_path / "c"),
                          host_profile="native-vector-math")
    return bundle.directory / f"{kernel.name}.so"


@needs_x86
def test_a_math_free_kernel_emits_no_vector_math(tmp_path):
    assert _hvm_symbols(_host_so(plain_step, tmp_path)) == set()


@needs_x86
def test_a_math_kernel_emits_only_the_functions_it_calls(tmp_path):
    syms = _hvm_symbols(_host_so(root_pow_step, tmp_path))
    called = {re.sub(r"^.*aether_hvm_", "", s) for s in syms}
    assert called == {"sqrt", "pow"}, sorted(syms)
    # Every family is whole: the scalar symbol and the SSE pair at least,
    # one masked and one unmasked variant per width the TU was built for.
    for fn, arity in (("sqrt", "v"), ("pow", "vv")):
        assert f"aether_hvm_{fn}" in syms
        assert f"_ZGVbN2{arity}_aether_hvm_{fn}" in syms
        assert f"_ZGVbM2{arity}_aether_hvm_{fn}" in syms
        # _ZGV<isa><mask><lanes>: every ISA built for has both mask kinds.
        variants = {s[4:6] for s in syms if s.startswith("_ZGV") and s.endswith(f"_{fn}")}
        isas = {v[0] for v in variants}
        assert variants == {i + k for i in isas for k in "NM"}, sorted(variants)


class _Src:
    def __init__(self, text):
        self.text = text


def test_the_target_compiles_overlap(monkeypatch):
    started = {"host": threading.Event(), "cuda": threading.Event()}

    def fake(text, name, opts):
        started[text].set()
        other = "cuda" if text == "host" else "host"
        # Each compile finishes only once the other one has started.
        assert started[other].wait(timeout=20), "the compiles ran one after the other"
        return f"result:{text}"

    monkeypatch.setattr(bundle_mod, "compile_source", fake)
    monkeypatch.delenv("HAWK_COMPILE_JOBS", raising=False)
    out = bundle_mod._compile_targets({"host": _Src("host"), "cuda": _Src("cuda")}, "k",
                                      {"host": None, "cuda": None})
    assert list(out) == ["host", "cuda"]
    assert out == {"host": "result:host", "cuda": "result:cuda"}


def test_one_job_is_the_sequential_build(monkeypatch):
    order = []

    def fake(text, name, opts):
        order.append((threading.current_thread().name, text))
        return text

    monkeypatch.setattr(bundle_mod, "compile_source", fake)
    monkeypatch.setenv("HAWK_COMPILE_JOBS", "1")
    out = bundle_mod._compile_targets({"host": _Src("host"), "cuda": _Src("cuda")}, "k",
                                      {"host": None, "cuda": None})
    main = threading.current_thread().name
    assert order == [(main, "host"), (main, "cuda")] and out == {"host": "host", "cuda": "cuda"}


@pytest.mark.parametrize("failing", [("host",), ("cuda",), ("host", "cuda")])
def test_the_first_failing_target_is_reported_after_both_finish(monkeypatch, failing):
    finished = []

    def fake(text, name, opts):
        finished.append(text)
        if text in failing:
            raise HawkError(f"{text} compile failed")
        return text

    monkeypatch.setattr(bundle_mod, "compile_source", fake)
    monkeypatch.delenv("HAWK_COMPILE_JOBS", raising=False)
    with pytest.raises(HawkError, match=f"^{failing[0]} compile failed"):
        bundle_mod._compile_targets({"host": _Src("host"), "cuda": _Src("cuda")}, "k",
                                    {"host": None, "cuda": None})
    assert sorted(finished) == ["cuda", "host"]


@pytest.mark.parametrize("raw", ["0", "-1", "two", "1.5"])
def test_an_invalid_job_count_raises(monkeypatch, raw):
    monkeypatch.setenv("HAWK_COMPILE_JOBS", raw)
    with pytest.raises(HawkError, match="HAWK_COMPILE_JOBS"):
        bundle_mod.compile_jobs()


def test_the_job_count_defaults_to_two(monkeypatch):
    monkeypatch.delenv("HAWK_COMPILE_JOBS", raising=False)
    assert bundle_mod.compile_jobs() == 2
    monkeypatch.setenv("HAWK_COMPILE_JOBS", "3")
    assert bundle_mod.compile_jobs() == 3
