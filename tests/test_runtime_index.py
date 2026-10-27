# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The gradient of a runtime-indexed read (``add_at``): the emitted text of an
accumulate on the own column, and the compiled HOST kernel against the scratch
interpreter."""

from __future__ import annotations

import re

import _spline_forms as S
import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk import Index, Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp
from hawk.emit import render_body
from hawk.ir import Assign, Leaf, canonical
from hawk.ir import make as mk
from hawk.types import TensorType


@pytest.fixture(autouse=True)
def _card_free_select_weight(monkeypatch):
    """The emitted text must not depend on the device the tests run on."""
    from hawk.emit import rtstore

    monkeypatch.setattr(rtstore, "_select_weight", lambda dtype: 1.0)


def _body(sinks):
    return render_body(tuple(sinks), canonical(tuple(sinks))).text


def test_a_link_of_a_switch_is_a_statement_of_its_case():
    """Each edge's accumulates sit inside the case that owns them: a case that
    does not take the link writes nothing (no ``at(0) += 0`` after the switch)."""
    lines = _body(vjp(S.kan_switch(3), wrt=("x", "theta"))).splitlines()
    depth = lambda ln: len(ln) - len(ln.lstrip())          # noqa: E731
    switches = [n for n, ln in enumerate(lines) if "switch (" in ln]
    links = [n for n, ln in enumerate(lines) if "bar_theta[i].at(" in ln]
    assert len(switches) == 3 and len(links) == 12, (len(switches), len(links))
    for n in links:
        owner = max(m for m in switches if m < n)
        assert depth(lines[n]) > depth(lines[owner]), lines[n]
    assert not any(".at(static_cast<Int>(0)) +=" in ln for ln in lines)


def test_the_links_of_an_edge_precede_the_next_edges_reads():
    """Each link is committed where its index and value are ready, not at the
    end of the body (which keeps every edge's values live to the end)."""
    lines = _body(vjp(S.kan_layer("local", 3), wrt=("x", "theta"))).splitlines()
    links = [n for n, ln in enumerate(lines) if "bar_theta[i].at(" in ln]
    reads = [n for n, ln in enumerate(lines) if "vin_theta[i].at(" in ln]
    assert len(links) == len(reads) == 12, (len(links), len(reads))
    assert links[3] < reads[8] and links[7] < reads[11], (links, reads)


def test_only_a_slot_an_accumulate_may_reach_first_is_zeroed_up_front():
    lines = _body(vjp(S.kan_layer("local", 2), wrt=("x", "theta"))).splitlines()
    zeroed = {ln.split("(")[1].split(",")[0] for ln in lines
              if "mut_bar_theta(" in ln and ln.strip().endswith("= static_cast<Real>(0);")}
    added = {ln.split("(")[1].split(",")[0] for ln in lines
             if "mut_bar_theta(" in ln and "+=" in ln}
    plain = {ln.split("(")[1].split(",")[0] for ln in lines
             if "mut_bar_theta(" in ln and "+=" not in ln and "= static_cast<Real>(0)" not in ln}
    assert added <= zeroed and not (plain & zeroed), (added - zeroed, plain & zeroed)


def test_an_accumulate_on_a_computed_vector_copies_it_once_and_selects():
    """Not an own-column commit of a slot-form base: the vector is copied ONCE
    into a local, and each component is a select on the runtime index."""
    v, k, e = (Leaf("vocab_read", "vec_in", "v", TensorType((3,), "f64")),
               Leaf("vocab_read", "per_sample", "k", TensorType((), "i32")),
               Leaf("vocab_read", "per_sample", "e", TensorType((), "f64")))
    sinks = (Assign("o", mk("add_at", (v, k, e)), TensorType((3,), "f64")),)
    text = _body(sinks)
    assert len(re.findall(r"aether::Item<Real, 3> hawk_sv0 = ", text)) == 1, text
    assert "hawk_sv0(1) = hawk_sv0_0_k == 1 ? hawk_sv0(1) + hawk_sv0_0_v0 : hawk_sv0(1);" in text, text
    assert "mut_o[i] = hawk_sv0;" in text, text


# --------------------------------------------------------------------------
# the compiled host kernel against the interpreter


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("runtime_index")
    cache = {}

    def get(name, primal, sinks):
        if name not in cache:
            d = root / name
            d.mkdir()
            hawk.build([primal, Kernel(name, sinks)], d, targets=("host",),
                       cache_dir=cache_dir)
            cache[name] = hawk.load(d, name)
        return cache[name]

    return get


@hawk.kernel
def _double_hit(theta: Vector[4], j: Index, y: Mutable[Scalar]):
    """One lane read BOTH by constant and by runtime index."""
    y = theta[2] * 3.0 + theta[j]


def test_a_lane_hit_by_a_constant_and_a_runtime_index_sums_both(host):
    """The accumulate ADDS to the slot a plain store wrote (an ``=`` here would
    overwrite the static contribution): compiled host against the interpreter."""
    sinks = vjp(_double_hit, wrt=("theta",))
    n = 6
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(4, n))
    j = np.array([2, 2, 0, 1, 3, 2], dtype=np.int64)
    seed = rng.normal(size=n)
    bar = np.zeros((4, n))
    hawk.run(host("double_hit_vjp", _double_hit, sinks), theta=theta, j=j, bar_y=seed, bar_theta=bar)
    for s in range(n):
        want = evaluate(sinks, {"theta": theta[:, s], "j": int(j[s]), "bar_y": seed[s]})
        assert np.array_equal(bar[:, s], want["bar_theta"]), s


@pytest.mark.parametrize("n_in", [8])
def test_the_local_kan_vjp_host_kernel_equals_the_interpreter(host, n_in):
    """Every lane of the compiled local-spline vjp against the interpreter. The
    host profile contracts multiply-adds (the interpreter does not), so the lanes
    agree to rounding of the largest term of the sample, not bit for bit."""
    cell = S.kan_layer("local", n_in)
    sinks = vjp(cell, wrt=("x", "theta"))
    n = 5
    rng = np.random.default_rng(8)
    x = rng.uniform(-0.9, 0.9, size=(n_in * S.ROWS, n))
    theta = rng.normal(0.0, 0.3, size=(n_in * S.W, n))
    theta[S.W - 2::S.W] = 0.0
    theta[S.W - 1::S.W] = 1.0
    seed = rng.normal(size=(2, n))
    bar_x, bar_theta = np.zeros_like(x), np.zeros_like(theta)
    hawk.run(host(f"kan_local_vjp_{n_in}", cell, sinks), x=x, theta=theta,
             terminated=np.zeros(n, dtype=bool), bar_y=seed, bar_x=bar_x,
             bar_theta=bar_theta)
    for s in range(n):
        want = evaluate(sinks, {"x": x[:, s], "theta": theta[:, s], "terminated": False,
                                "bar_y": seed[:, s]})
        for got, ref in ((bar_theta[:, s], want["bar_theta"]), (bar_x[:, s], want["bar_x"])):
            np.testing.assert_allclose(got, ref, rtol=1e-12,
                                       atol=1e-13 * np.abs(ref).max())
        # the slots no edge reads stay exactly zero
        assert np.array_equal(bar_theta[:, s] == 0.0, np.asarray(want["bar_theta"]) == 0.0)


# --------------------------------------------------------------------------
# a store at a runtime index of a short vector is one select per component


def _stores(width):
    def build(v, j, e, o):
        acc = v
        for c in range(width):
            acc[c] = acc[c] + e
        acc[m.index(j, width)] = e * 2.0
        o = acc
    build.__annotations__ = {"v": Vector[width], "j": Index, "e": Scalar,
                             "o": Mutable[Vector[width]]}
    return hawk.kernel(build)


def _zeros():
    return m.vec(*([0.0] * 15))


def _scatter(trips):
    """The spline scatter: one static slot and four slots at a runtime base
    plus static offsets, in a rolled loop."""
    def build(u, e, o):
        acc = _zeros()
        for t in range(trips):
            i = m.index(u + t, 10)
            w = e * (t + 1.0)
            acc[0] = acc[0] + w
            acc[i + 2] = acc[i + 2] + w
            acc[i + 3] = acc[i + 3] + w * 2.0
            acc[i + 4] = acc[i + 4] + w * 3.0
            acc[i + 5] = acc[i + 5] + w * 4.0
        o = acc
    build.__annotations__ = {"u": Scalar, "e": Scalar, "o": Mutable[Vector[15]]}
    return hawk.kernel(build)


def _zeros_uniform():
    return m.vec(*([0.0] * 15))


@hawk.kernel
def _uniform_scatter(u: hawk.Table[Scalar], o: Mutable[Vector[15]]):
    group, lane_in = m.split_index(64)
    acc = _zeros_uniform()
    for t in range(6):
        i = m.index(u.at(group * 6 + t), 10)
        acc[i + 2] = acc[i + 2] + 1.0
        acc[i + 3] = acc[i + 3] + 2.0
    o = acc


@hawk.kernel
def _one_store4(u: Scalar, e: Scalar, o: Mutable[Vector[4]]):
    acc = m.vec(0.0, 0.0, 0.0, 0.0)
    for t in range(5):
        i = m.index(u + t, 4)
        acc[i] = acc[i] + e
    o = acc


@hawk.kernel
def _one_store5(u: Scalar, e: Scalar, o: Mutable[Vector[5]]):
    acc = m.vec(0.0, 0.0, 0.0, 0.0, 0.0)
    for t in range(5):
        i = m.index(u + t, 5)
        acc[i] = acc[i] + e
    o = acc


def test_one_accumulate_on_a_four_wide_vector_keeps_the_lean_form_for_float():
    """Width <= 4, one store per trip: with a four-byte ``Real`` the vector is
    read back through selects and rebuilt whole; with eight bytes it is selected
    per component."""
    text = render_body(_one_store4.sinks, _one_store4.walk).text
    assert "if constexpr (sizeof(Real) == 8) {" in text and "_old" in text, text
    wide = render_body(_one_store5.sinks, _one_store5.walk).text
    assert "_old" not in wide and "constexpr" not in wide, wide


@hawk.kernel
def _clamped(u: Scalar, e: Scalar, o: Mutable[Vector[15]]):
    """Three accumulates whose indices are each clamped on their own."""
    acc = _zeros()
    for t in range(6):
        p = m.index(u + t, 16)
        acc[2 + m.index(m.real(p - 3), 13)] = acc[2 + m.index(m.real(p - 3), 13)] + e
        acc[2 + m.index(m.real(p - 2), 13)] = acc[2 + m.index(m.real(p - 2), 13)] + e * 2.0
        acc[2 + m.index(m.real(p - 1), 13)] = acc[2 + m.index(m.real(p - 1), 13)] + e * 3.0
    o = acc


def _copies(text, width):
    return re.findall(rf"^\s*aether::Item<Real, {width}> hawk_sv\d+ = ", text, re.M)


@pytest.mark.parametrize("width", [15, 17, 32, 256])
def test_a_runtime_store_never_copies_the_vector_inside_the_loop(width):
    """The carried vector is written where it lives; only the store after the
    loop copies its operand (once)."""
    k = _stores(width)
    text = render_body(k.sinks, k.walk).text
    loop = text.index("for (Int")
    assert "hawk_sv" not in text[loop:text.index("}", loop)] or "hawk_sv0_0" in text, text
    assert len(_copies(text, width)) == 1, text
    assert text.count("hawk_it") == 0, text


def test_a_wide_vector_is_written_in_place_and_a_narrow_one_by_selects():
    scatter, wide = _scatter(6), _stores(256)
    narrow = render_body(scatter.sinks, scatter.walk).text
    wide = render_body(wide.sinks, wide.walk).text
    assert "hawk_sv0_1_k == 0 ?" in narrow and "hawk_sv0_0_k ==" not in wide, (narrow, wide)
    assert "hawk_v2_0_acc(hawk_k2_c) += " in wide, wide


def test_a_key_proven_uniform_over_the_warp_is_written_in_place():
    """The same four accumulates, with the base read at the lane's group of 64
    (``split_index``): every lane of a warp shares the index, so the indexed
    access is one address and the in-place write wins at any width."""
    text = render_body(_uniform_scatter.sinks, _uniform_scatter.walk).text
    assert "== " not in text.split("for (Int")[1] and "+= " in text, text


def test_the_select_weight_follows_the_arch_the_build_resolved(monkeypatch):
    from hawk.targets import cards

    monkeypatch.undo()                      # the card, not the pinned weight
    assert cards.select_weight("float64") == 1.0            # no device arch: the default
    with cards.build_arch("sm_61"):
        assert cards.select_weight("float64") == 4.0
        assert cards.select_weight("float32") == 1.0
    with cards.build_arch("sm_80"):
        assert cards.select_weight("float64") == 1.0


def test_the_select_weight_of_a_card_moves_the_pick(monkeypatch):
    from hawk.emit import rtstore

    k = _scatter(6)
    monkeypatch.setattr(rtstore, "_select_weight", lambda dtype: 100.0)
    assert "hawk_sv0_1_k ==" not in render_body(k.sinks, k.walk).text


def test_stores_at_one_base_fuse_over_the_reachable_components():
    """Four accumulates at ``i + 2 .. i + 5`` with ``i`` clamped to ``[0, 10)``
    are ONE pass: one add per reachable component (slots 2..14), none of slots
    1, and no dynamic read of the vector."""
    k = _scatter(6)
    text = render_body(k.sinks, k.walk).text
    assert not _copies(text, 15) and "hawk_it" not in text, text
    adds = re.findall(r"^\s*hawk_v\d+_0_acc\((\d+)\) \+= ", text, re.M)
    assert adds == ["0", *map(str, range(2, 15))], adds
    assert text.count("hawk_sv0_1_k == 0 ?") == 4, text


def _scatter_inputs(n, trips):
    rng = np.random.default_rng(trips)
    return rng.integers(-2, 12, size=n).astype(float), rng.integers(1, 5, size=n).astype(float)


def test_the_fused_scatter_host_run_equals_the_interpreter_bitwise(host):
    """Integer-valued data: every product and sum is exact, so the fused pass
    and the interpreter agree bit for bit (slot 1 stays the zero it started as)."""
    kernel = _scatter(6)
    n = 9
    u, e = _scatter_inputs(n, 6)
    out = np.full((15, n), 7.0)
    hawk.run(host("scatter_6", kernel, kernel.sinks), u=u, e=e, o=out)
    for s in range(n):
        want = evaluate(kernel.sinks, {"u": u[s], "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s
        assert out[1, s] == 0.0


@pytest.mark.parametrize("width", [15, 17, 256])
def test_a_runtime_store_host_run_equals_the_interpreter(host, width):
    kernel = _stores(width)
    n = 7
    rng = np.random.default_rng(width)
    v, e = rng.normal(size=(width, n)), rng.normal(size=n)
    j = rng.integers(0, width, size=n)
    out = np.zeros((width, n))
    hawk.run(host(f"store_{width}", kernel, kernel.sinks), v=v, j=j, e=e, o=out)
    for s in range(n):
        want = evaluate(kernel.sinks, {"v": v[:, s], "j": int(j[s]), "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s


def _on_device(kernel, cache_dir, **planes):
    """The kernel on the GPU: the arrays go over as given, the output comes back."""
    import cupy as cp
    import eagle
    import eagle.plan

    plan = eagle.plan.auto([kernel], targets=("cuda",), cache_dir=cache_dir,
                           scalar_type="float64")[0]
    arrays = {k: cp.asarray(v) for k, v in planes.items()}
    graph = eagle.Graph()
    graph.add(plan.bind(**arrays), name="n")
    graph.build().launch()
    cp.cuda.runtime.deviceSynchronize()
    return {k: cp.asnumpy(v) for k, v in arrays.items()}


@pytest.mark.gpu
def test_the_fused_scatter_on_the_device_equals_the_host(host, cache_dir):
    kernel, n = _scatter(6), 33
    u, e = _scatter_inputs(n, 6)
    want = np.zeros((15, n))
    hawk.run(host("scatter_6", kernel, kernel.sinks), u=u, e=e, o=want)
    got = _on_device(kernel, cache_dir, u=u, e=e, o=np.zeros((15, n)))["o"]
    assert np.array_equal(got, want)


@pytest.mark.gpu
def test_an_in_place_store_on_the_device_equals_the_host(host, cache_dir):
    kernel, n, width = _stores(256), 33, 256
    rng = np.random.default_rng(3)
    v, e = rng.normal(size=(width, n)), rng.normal(size=n)
    j = rng.integers(0, width, size=n)
    want = np.zeros((width, n))
    hawk.run(host("store_256", kernel, kernel.sinks), v=v, j=j, e=e, o=want)
    got = _on_device(kernel, cache_dir, v=v, j=j, e=e, o=np.zeros((width, n)))["o"]
    assert np.array_equal(got, want)


def test_stores_with_independent_clamped_indices_copy_nothing():
    """Each store clamps its own index, so no base is shared and nothing fuses;
    every store is still written in place or by selects, never through a copy
    of the vector or a dynamic read of a copy."""
    text = render_body(_clamped.sinks, _clamped.walk).text
    assert not _copies(text, 15) and "hawk_it" not in text, text


def test_independent_clamped_stores_host_run_equals_the_interpreter(host):
    n = 9
    rng = np.random.default_rng(4)
    u, e = rng.integers(-2, 14, size=n).astype(float), rng.integers(1, 5, size=n).astype(float)
    out = np.zeros((15, n))
    hawk.run(host("clamped", _clamped, _clamped.sinks), u=u, e=e, o=out)
    for s in range(n):
        want = evaluate(_clamped.sinks, {"u": u[s], "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s
