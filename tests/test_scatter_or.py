# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(mask, policy="or")[word] |= bits``: a bit mask written by key.

An ``Int32`` / ``UInt32`` plane is the mask (a word per key, 32 bits each; there
is no 64-bit integer plane), and every lane ORs its bits into the word its key
names. The commit is the target's atomic OR (``atomicOr`` on the device, an
``atomic_ref`` under the host team) and the C target's serial ``|=`` at the
clamped row; the result is the exact integer ``np.bitwise_or.at`` gives, for any
number of lanes hitting one word, on every target, bit for bit. A bit mask has
no derivative: ``vjp`` and ``jvp`` refuse by name.
"""

from __future__ import annotations

import _deploy as L
import _embedded as E
import numpy as np
import pytest
from _toolchain import compile_check
from conftest import sidecar_of

import hawk
from hawk import HawkError, Index, Int32, Mutable, Scalar, UInt32, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.embedded import EmbeddedProfile, check, export, render
from hawk.emit import FLOAT64, render_source
from hawk.targets import TARGETS

N = 1000
WORDS = 7


@kernel
def mark(word: Index, bit: UInt32, mask: Mutable[UInt32]):
    hawk.scatter(mask, policy="or")[word] |= bit


@kernel
def mark_signed(word: Index, bit: Int32, mask: Mutable[Int32]):
    hawk.scatter(mask, policy="or")[word] |= bit


def _inputs(dtype):
    rng = np.random.default_rng(3)
    # most lanes hit word 0: the contended case; the rest spread over the others
    word = np.where(rng.random(N) < 0.5, 0, rng.integers(0, WORDS, N)).astype(np.int32)
    bit = (np.uint32(1) << (np.arange(N) % 32).astype(np.uint32)).astype(dtype)
    return word, bit


def _oracle(word, bit, dtype):
    want = np.zeros(WORDS, dtype)
    np.bitwise_or.at(want, word, bit)
    return want


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle([mark, mark_signed], tmp_path_factory.mktemp("or"),
                        mode="float64", targets=("cuda", "host"), cache_dir=cache_dir)


def _launch(bundle, name, target, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    eplan.plan(plugin, structure=structure).bind(**planes).launch()


@pytest.mark.parametrize("name, dtype", [("mark", np.uint32), ("mark_signed", np.int32)])
def test_many_lanes_into_one_word_equal_bitwise_or_at_on_the_host(bundle, name, dtype):
    word, bit = _inputs(dtype)
    mask = np.zeros(WORDS, dtype)
    _launch(bundle, name, "host", {"word": word, "bit": bit, "mask": mask})
    assert mask.tobytes() == _oracle(word, bit, dtype).tobytes()
    assert mask[0] == np.uint32(0xFFFFFFFF).astype(dtype)      # word 0 saw every bit


def test_an_or_keeps_the_bits_the_caller_already_set(bundle):
    word, bit = _inputs(np.uint32)
    start = np.arange(WORDS, dtype=np.uint32) << 28
    mask = start.copy()
    _launch(bundle, "mark", "host", {"word": word, "bit": bit, "mask": mask})
    assert mask.tobytes() == (_oracle(word, bit, np.uint32) | start).tobytes()


def test_a_repeated_launch_is_idempotent(bundle):
    word, bit = _inputs(np.uint32)
    mask = np.zeros(WORDS, np.uint32)
    for _ in range(3):
        _launch(bundle, "mark", "host", {"word": word, "bit": bit, "mask": mask})
    assert mask.tobytes() == _oracle(word, bit, np.uint32).tobytes()


@pytest.mark.parametrize("target", ["cuda", "host"])
@pytest.mark.parametrize("k", [mark, mark_signed], ids=lambda k: k.name)
def test_the_emitted_sources_commit_through_the_atomic_or_and_compile(tmp_path, target, k):
    src = render_source(k.name, k.sinks, k.walk, TARGETS[target], mode=FLOAT64, kind=k.kind)
    assert "hawk_abi::accum_or(" in src.text
    assert ("atomicOr" in src.text) and ("fetch_or" in src.text)
    path = tmp_path / f"{k.name}.{'cu' if target == 'cuda' else 'cpp'}"
    path.write_text(src.text)
    rc, stderr, argv = compile_check(path, target)
    assert rc == 0, f"$ {' '.join(argv)}\n{stderr[:2000]}"


@pytest.mark.gpu
@pytest.mark.parametrize("name, dtype", [("mark", np.uint32), ("mark_signed", np.int32)])
def test_the_device_equals_bitwise_or_at(bundle, name, dtype):
    import cupy as cp

    word, bit = _inputs(dtype)
    mask = cp.zeros(WORDS, dtype)
    _launch(bundle, name, "cuda", {"word": cp.asarray(word), "bit": cp.asarray(bit),
                                    "mask": mask})
    cp.cuda.Device().synchronize()
    assert mask.get().tobytes() == _oracle(word, bit, dtype).tobytes()


def test_the_c_unit_is_misra_clean_and_computes_the_hosts_bits(bundle, tmp_path):
    strict = EmbeddedProfile(require_cppcheck=True)
    checks = check(render(mark, mode="float64"), profile=strict)
    assert all(run["rc"] == 0 for run in checks["cc"])
    assert checks["cppcheck"]["findings"] == []
    art = export(mark, tmp_path, mode="float64")
    word, bit = _inputs(np.uint32)
    c_mask = np.zeros(WORDS, np.uint32)
    E.run_c(E.cc_shared(art.c_path, tmp_path), "mark", art.report, N, mode="float64",
            word=word, bit=bit, mask=c_mask)
    host = np.zeros(WORDS, np.uint32)
    _launch(bundle, "mark", "host", {"word": word, "bit": bit, "mask": host})
    assert c_mask.tobytes() == host.tobytes() == _oracle(word, bit, np.uint32).tobytes()


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_a_bit_mask_is_refused_by_name(derive):
    with pytest.raises(HawkError, match=r"sets bits by key.*no derivative"):
        derive(mark)


def test_refuses_what_is_not_a_bit_mask_store():
    with pytest.raises(HawkError, match=r"\|= sets bits, written hawk\.scatter\(m, policy=\"or\"\)"):
        @kernel
        def k(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m)[w] |= b
    with pytest.raises(HawkError, match=r"policy='or' is for \|= only"):
        @kernel
        def k2(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m, policy="or")[w] += b
    with pytest.raises(HawkError, match=r"sets bits of an Int32 or UInt32 plane"):
        @kernel
        def k3(w: Index, x: Scalar, m: Mutable[Scalar]):
            hawk.scatter(m, policy="or")[w] |= x
    with pytest.raises(HawkError, match=r"only `hawk.scatter\(y\)\[key\] \+= t` accumulates"):
        @kernel
        def k4(w: Index, b: UInt32, m: Mutable[UInt32]):
            hawk.scatter(m, policy="or")[w] &= b
