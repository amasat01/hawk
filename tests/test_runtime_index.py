# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The gradient of a runtime-indexed read (``add_at``): the emitted text of an
accumulate on the own column, and the compiled HOST kernel against the scratch
interpreter."""

from __future__ import annotations

import _spline_forms as S
import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Index, Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp
from hawk.emit import render_body
from hawk.ir import Assign, Const, Leaf, canonical
from hawk.ir import make as mk
from hawk.types import TensorType


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


def test_a_chain_on_a_computed_vector_keeps_the_item_accumulate():
    """Not an own-column commit of a slot-form base: the value is an ``Item``
    copy with the add written into it (the ``set_component_at`` spelling)."""
    v, k, e = (Leaf("vocab_read", "vec_in", "v", TensorType((3,), "f64")),
               Leaf("vocab_read", "per_sample", "k", TensorType((), "i32")),
               Leaf("vocab_read", "per_sample", "e", TensorType((), "f64")))
    sinks = (Assign("o", mk("add_at", (v, k, e)), TensorType((3,), "f64")),)
    text = _body(sinks)
    assert "aether::Item<Real, 3> hawk_sv" in text and ") += " in text, text
    assert "mut_o[i] = hawk_sv" in text, text


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
