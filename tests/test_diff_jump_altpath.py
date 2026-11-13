# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``jump=`` through a branch that decides a future: the alternate path.

A switched value that reaches another comparison, a ``break`` or a nonlinear op
is differentiated by simulating the other side of one drawn switch
(:mod:`hawk.diff.altpath`). Rows, on the host:

* **Truth.** The mean tangent over many lanes sits within four standard errors
  of the exact derivative of a counting game (``gate``, ``mult``, ``chain``), of
  ``exp(5 y)``, and of the likelihood-ratio derivative of a slab transport with
  a ``break`` (same lanes).
* **Seed.** One ``jump_seed`` gives the same bits twice; a lane with fewer than
  two switching instances does not depend on it, and a lane with none is the
  plain piecewise tangent.
* **Off.** A plan whose switched value only reaches the outputs affinely is not
  engaged and takes no seed.
"""

from __future__ import annotations

import numpy as np
from conftest import declared
import pytest

import hawk
import hawk.math as m
from hawk import Index, Kernel, Mutable, Param, Scalar, Terminated, Vector, kernel
from hawk.diff import jvp, step_vjp, vjp
from hawk.ext import DATA_ONLY, KernelKind

H, S, STEPS, LANES, Z = 0.01, 0.3, 6, 400_000, 4.0
W = np.sqrt(5.0) * H


@kernel(jump=H)
def gate(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    alive = 1.0
    c = 0.0
    for i in range(6):
        if alive > 0.5:
            if u[i] < s:
                alive = 0.0
            else:
                c = c + 1.0
    tally = c
    alive_out = alive


@kernel(jump=H)
def mult(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    alive = 1.0
    c = 0.0
    for i in range(6):
        stay = 1.0
        if u[i] < s:
            stay = 0.0
        alive = alive * stay
        c = c + alive
    tally = c
    alive_out = alive


@kernel(jump=H)
def chain(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    y = 0.0
    if u[0] < s:
        y = 1.0
    z = 0.0
    if y > 0.5:
        z = 1.0
    tally = z
    alive_out = y


@kernel(jump=H)
def blow(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    y = 0.0
    if u[0] < s:
        y = 1.0
    tally = m.exp(5.0 * y)
    alive_out = y


@kernel(jump=H)
def affine(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    y = 0.0
    if u[0] < s:
        y = 1.0
    tally = 3.0 * y
    alive_out = y


Q = 1.0 - S
EXACT = (-sum(k * Q ** (k - 1) for k in range(1, STEPS + 1)), -STEPS * Q ** (STEPS - 1))
TRUTH = {"gate": EXACT, "mult": EXACT, "chain": (1.0, 1.0), "blow": (np.e ** 5 - 1.0, 1.0)}
KERNELS = {"gate": gate, "mult": mult, "chain": chain, "blow": blow}


@pytest.fixture(scope="module")
def lanes():
    return np.ascontiguousarray(np.random.default_rng(1).uniform(0.0, 1.0, (STEPS, LANES)))


@pytest.fixture(scope="module")
def tangent(tmp_path_factory, cache_dir, lanes):
    root = tmp_path_factory.mktemp("altpath")
    built: dict = {}

    def run(primal, seed=0):
        if primal.name not in built:
            d = root / primal.name
            d.mkdir()
            hawk.build([primal, Kernel("g", jvp(primal, wrt=("s",)))], d, targets=("host",),
                       cache_dir=cache_dir)
            built[primal.name] = hawk.load(d, "g")
        out = {k: np.zeros(LANES) for k in ("dot_tally", "dot_alive_out")}
        extra = {} if primal is affine or seed is None else {"jump_seed": seed}
        hawk.run(built[primal.name], u=lanes, s=S, dot_s=1.0, **extra, **out, n_samples=LANES)
        return out["dot_tally"], out["dot_alive_out"]

    return run


def _within(values, truth):
    se = values.std() / np.sqrt(values.size)
    assert abs(values.mean() - truth) < Z * se, (values.mean(), truth, se)


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_the_mean_tangent_is_the_derivative_of_the_game(tangent, name):
    tally, alive = tangent(KERNELS[name])
    _within(tally, TRUTH[name][0])
    _within(alive, TRUTH[name][1])


def test_the_seed_fixes_the_bits_and_only_multi_event_lanes_see_it(tangent, lanes):
    a, _ = tangent(KERNELS["gate"], seed=0)
    again, _ = tangent(KERNELS["gate"], seed=0)
    other, _ = tangent(KERNELS["gate"], seed=1)
    assert np.array_equal(a, again)
    events = (np.abs(lanes - S) < W).sum(axis=0)
    assert np.array_equal(a[events < 2], other[events < 2])
    assert (a[events >= 2] != other[events >= 2]).any()


def test_a_lane_with_no_switch_in_reach_is_the_plain_tangent(tangent, lanes, tmp_path, cache_dir):
    d = tmp_path / "plain"
    d.mkdir()

    @kernel
    def gate_off(u: Vector[6], s: Param, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
        alive = 1.0
        c = 0.0
        for i in range(6):
            if alive > 0.5:
                if u[i] < s:
                    alive = 0.0
                else:
                    c = c + 1.0
        tally = c
        alive_out = alive

    hawk.build([gate_off, Kernel("g", jvp(gate_off, wrt=("s",)))], d, targets=("host",), cache_dir=cache_dir)
    out = {k: np.zeros(LANES) for k in ("dot_tally", "dot_alive_out")}
    loaded = hawk.load(d, "g")
    hawk.run(loaded, **declared(loaded, {"u": lanes, "s": S, "dot_s": 1.0}), **out, n_samples=LANES)
    quiet = (np.abs(lanes - S) < W).sum(axis=0) == 0
    got, _ = tangent(KERNELS["gate"])
    assert quiet.sum() > LANES // 2 and np.array_equal(got[quiet], out["dot_tally"][quiet])


def test_the_default_seed_is_zero(tangent):
    bare, _ = tangent(KERNELS["gate"], seed=None)
    zero, _ = tangent(KERNELS["gate"], seed=0)
    assert np.array_equal(bare, zero)


@pytest.mark.parametrize("name", ("gate", "chain"))
def test_the_reverse_is_the_transpose_of_the_forward(tmp_path, cache_dir, lanes, name):
    primal, n = KERNELS[name], 50_000
    u = np.ascontiguousarray(lanes[:, :n])
    hawk.build([primal, Kernel("j", jvp(primal, wrt=("s",))), Kernel("v", vjp(primal, wrt=("s",)))],
               tmp_path, targets=("host",), cache_dir=cache_dir)
    rng = np.random.default_rng(5)
    vt, va = rng.normal(size=n), rng.normal(size=n)
    dt, da, bs = np.zeros(n), np.zeros(n), np.zeros(n)
    hawk.run(hawk.load(tmp_path, "j"), u=u, s=S, dot_s=1.0, jump_seed=3, dot_tally=dt,
             dot_alive_out=da, n_samples=n)
    hawk.run(hawk.load(tmp_path, "v"), u=u, s=S, jump_seed=3, bar_tally=vt, bar_alive_out=va,
             bar_s=bs, n_samples=n)
    want = vt * dt + va * da
    assert (want != 0).sum() > 100
    np.testing.assert_allclose(bs, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_an_affine_switch_is_not_engaged_and_takes_no_seed():
    d = jvp(affine, wrt=("s",))
    assert d.jump["rule"] == "fixed" and "engaged" not in d.jump
    assert jvp(chain, wrt=("s",)).jump["rule"] == "altpath_v1"
    assert jvp(chain, wrt=("s",)).jump["seed_param"] == "jump_seed"


class Slab(KernelKind, slug="slab_alt"):
    seed: Index
    width: Scalar
    c: Param
    sigma_t: Param
    sigma_ref: Param
    guard = DATA_ONLY


def _history(**kw):
    @Slab(**kw)
    def history(seed, width, c, sigma_t, sigma_ref,
                through: Mutable[Scalar], back: Mutable[Scalar], tally: Mutable[Scalar]):
        x = 0.0
        mu = 1.0
        alive = 1.0
        log_w = 0.0
        through = 0.0
        back = 0.0
        fate = 0.0
        s = 0.0
        x_new = 0.0
        dist = 0.0
        for i in range(120):
            if alive < 0.5:
                break
            s = m.random.exponential(seed, 3 * i, sigma_ref)
            x_new = x + mu * s
            if x_new > width:
                dist = (width - x) / mu
                log_w = log_w - (sigma_t - sigma_ref) * dist
                through = m.exp(log_w)
                fate = 1.0
                alive = 0.0
            elif x_new < 0.0:
                dist = x / (-mu)
                log_w = log_w - (sigma_t - sigma_ref) * dist
                back = m.exp(log_w)
                fate = 2.0
                alive = 0.0
            else:
                x = x_new
                log_w = log_w + m.log(sigma_t / sigma_ref) - (sigma_t - sigma_ref) * s
                if m.random.uniform(seed, 3 * i + 1) < c:
                    mu = m.random.uniform(seed, 3 * i + 2, -1.0, 1.0)
                else:
                    alive = 0.0
        hawk.scatter(tally)[m.index(fate, 3)] += through + back
    return history


def test_a_slab_with_a_break_matches_the_likelihood_ratio(tmp_path, cache_dir):
    n, runs = 80_000, 20
    both, plain = _history(jump=H), _history()
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    hawk.build([both, Kernel("alt", jvp(both, wrt=("sigma_t", "sigma_ref")))], tmp_path / "a",
               targets=("host",), cache_dir=cache_dir)
    hawk.build([plain, Kernel("lr", jvp(plain, wrt=("sigma_t",)))], tmp_path / "b",
               targets=("host",), cache_dir=cache_dir)
    alt, lr = hawk.load(tmp_path / "a", "alt"), hawk.load(tmp_path / "b", "lr")
    per_run = {"through": [], "back": [], "bins": []}
    for r in range(runs):
        common = dict(seed=np.arange(n, dtype=np.int32) + r * n, width=np.full(n, 1.0), c=0.7,
                      sigma_t=1.0, sigma_ref=1.0, n_samples=n)
        got = {k: np.zeros(n) for k in ("dot_through", "dot_back")}
        ref = {k: np.zeros(n) for k in ("dot_through", "dot_back")}
        bins_alt, bins_lr = np.zeros(3), np.zeros(3)
        hawk.run(alt, dot_sigma_t=1.0, dot_sigma_ref=1.0, dot_tally=bins_alt,
                 dot_tally_c=np.zeros(3), **got, **common)
        hawk.run(lr, dot_sigma_t=1.0, dot_tally=bins_lr, dot_tally_c=np.zeros(3), **ref, **common)
        per_run["through"].append(got["dot_through"].mean() - ref["dot_through"].mean())
        per_run["back"].append(got["dot_back"].mean() - ref["dot_back"].mean())
        per_run["bins"].append((bins_alt - bins_lr) / n)
    for k in ("through", "back"):
        gap = np.array(per_run[k])
        assert abs(gap.mean()) < Z * gap.std() / np.sqrt(runs), (k, gap.mean())
    gap = np.array(per_run["bins"])             # the keyed tally, per bin
    assert (np.abs(gap.mean(axis=0)) <= Z * gap.std(axis=0) / np.sqrt(runs)).all(), gap.mean(axis=0)


def test_the_slab_reverse_is_the_transpose_of_the_forward(tmp_path, cache_dir):
    n, wrt = 20_000, ("sigma_t", "sigma_ref")
    history = _history(jump=H)
    # the primal scatters and its reverse gathers: two units, the reverse tagged
    # with the unit its primal lives in
    for sub, derived in (("j", jvp(history, wrt=wrt)),
                         ("v", vjp(history, wrt=wrt, primal_unit="0" * 64))):
        (tmp_path / sub).mkdir()
        hawk.build([history, Kernel(sub, derived)] if sub == "j" else [Kernel(sub, derived)],
                   tmp_path / sub, targets=("host",), cache_dir=cache_dir)
    common = dict(seed=np.arange(n, dtype=np.int32), width=np.full(n, 1.0), c=0.7,
                  sigma_t=1.0, sigma_ref=1.0, n_samples=n)
    rng = np.random.default_rng(3)
    vt, vb = rng.normal(size=n), rng.normal(size=n)
    dt, db = np.zeros(n), np.zeros(n)
    hawk.run(hawk.load(tmp_path / "j", "j"), dot_sigma_t=1.0, dot_sigma_ref=1.0, dot_through=dt,
             dot_back=db, dot_tally=np.zeros(3), dot_tally_c=np.zeros(3), **common)
    bar_t, bar_r = np.zeros(n), np.zeros(n)
    hawk.run(hawk.load(tmp_path / "v", "v"), bar_through=vt, bar_back=vb, bar_tally=np.zeros(3), bar_tally_c=np.zeros(3),
             bar_sigma_t=bar_t, bar_sigma_ref=bar_r, **common)
    want = vt * dt + vb * db
    assert np.abs(want).max() > 0
    np.testing.assert_allclose(bar_t + bar_r, want, rtol=1e-9, atol=1e-9 * np.abs(want).max())


def test_the_alternate_path_is_refused_across_launches():
    @kernel(jump=H, steps=1)
    def step(u: Scalar, s: Param, terminated: Terminated, y: Mutable[Scalar]):
        a = 0.0
        if u < s:
            a = 1.0
        y = y + m.exp(a)
        terminated = y > 100.0

    with pytest.raises(hawk.HawkError, match="across launches"):
        step_vjp(step)


#: sha256 of the emitted source of each derivative below, equal to the v0.7.0 tree's, apart from the host loop helpers' text in the k_loop rows
#: (the kernels whose switched value reaches the outputs only affinely keep the plain rule)
# Re-pinned: a host unit carries the host arms only, so every host source moved.
PINNED = {
    "k_lt/jvp": "ebe3ae0ca745b9e46f11f86ad4a6cd087b0cf719c73b3f880b97ad4bd7eb9c9e",
    "k_lt/vjp": "8ef51d7d11c86fa863733fee5c25d2a31e5c921162b77eb6ae2afa9deb4a9895",
    "k_le/jvp": "ebe3ae0ca745b9e46f11f86ad4a6cd087b0cf719c73b3f880b97ad4bd7eb9c9e",
    "k_le/vjp": "8ef51d7d11c86fa863733fee5c25d2a31e5c921162b77eb6ae2afa9deb4a9895",
    "k_gt/jvp": "e23d3e4862f4f1628c67b718fc27009abeeeb9cfd6c6c831d23ecbee45c30714",
    "k_gt/vjp": "6afbee9a8fbcfcc215397ccbc71252bb99d1c6c8ba5dbe059b67ae27c8c40e3e",
    "k_ge/jvp": "e23d3e4862f4f1628c67b718fc27009abeeeb9cfd6c6c831d23ecbee45c30714",
    "k_ge/vjp": "6afbee9a8fbcfcc215397ccbc71252bb99d1c6c8ba5dbe059b67ae27c8c40e3e",
    "k_and/jvp": "c938100fdf47a088942699c8253358df0523fa2cc63aa5f11c940fe0856eab80",
    "k_and/vjp": "f3092e5ec25688348ca033339a8062512d8ed6cedc0b61e1c223be729a406e0f",
    "k_or/jvp": "95ca3157fbe0ddae68ba96731703ed3ea30590ac3d628dca2b1949403c572af2",
    "k_or/vjp": "d2f3cc8615a2aafd9badb90df0ab79898767412703b986a708dfaf0c6fde44a7",
    "k_nested/jvp": "44dcd77bf43881a4ee01f485889470163ccb6627a2d94916c0ac8976dacef838",
    "k_nested/vjp": "03f7b21027f73c02b09ce6d50e8c6fa47dde8c459466efa0f8b142dc8c1b0fe9",
    "k_vec/jvp": "7a5656b77765975e10ece1b9d7804af332e0e99d4d9c458a8be619afdeb792ce",
    "k_vec/vjp": "ab94d3221a41508915ea685bb52ab328545251881fb43438899521587dc84602",
    "k_three/jvp": "b8326fa2a26c3263678d73698b64ac87e0370eff2573af9f1d96b336604fc46b",
    "k_three/vjp": "dd8894a1293bd36a3da8bbe6a257a14d4de8deb44885f06c0c17d1deddf2c569",
    "k_param/jvp": "90abf98a3721d1235fe18ae95380d7e11b89bd073e8f436244a08b9059c17079",
    "k_param/vjp": "81d20dbc5e084ef86928d981421d3f1e0e4d99236e19c10b2d2fd85d7fbf54ef",
    "k_const_cmp/jvp": "39c2fd390002bba57f7d80488f10947c092ad151b9783b50caedf347b730b155",
    "k_const_cmp/vjp": "1e11ad77c17d6b514b7f834714e9ff6d539dbba8e77f0fd6a1fc07f8cbadd127",
    # these two lost the loop's self-move of an unchanged carry (`const n = v; v = n;`): nothing else differs
    "k_loop_carried/jvp": "bda978f19ee1303c9219f553719d1220b9ed0c22619080fe4139be624bc396b8",
    "k_loop_carried/vjp": "6e6292cacca44fef5ca05f14ac07314b909067e4405698ed55044eae6f625d87",
    "k_loop_sum/jvp": "a29cc373d502c37ae729cde8bd5fbf51fbd9fa71d8474682783e3d1d156c9443",
    "k_loop_sum/vjp": "5a37e189898cb3ed07286deb690ac0e11a6b53a74b2acae2035a86256499ce6e",
    "k_loop_param/jvp": "0bdbf2b85ce5a87252ada76056ca94bae348888d8ec25b0f757ec024673235ed",
    "k_loop_param/vjp": "c3218a9f284aece87c82145cafebe119d3524de2637c07cd9fce30cb04f0eef5",
    "step/jvp": "3c2b8686e2318b294cddefeb63211aea88701244952b4d6f05384a6f3eba0410",
    "step/vjp": "59be8778be6eaf41d7a616fc0395f51a604060ba57dad13c57fa276bc7b03bf2",
}


def test_a_switch_that_only_reaches_the_outputs_affinely_keeps_its_source(tmp_path, cache_dir):
    import hashlib

    import test_diff_jump as t

    @kernel(jump=0.05)
    def step(x: Scalar, t: Param, y: Mutable[Scalar]):
        if x < t:
            y = 1.0
        else:
            y = 0.0

    got = {}
    for k in [getattr(t, n) for n in sorted({n.split("/")[0] for n in PINNED}) if n != "step"] + [step]:
        wrt = ("t",) if "t" in k.planes else None
        for tag, derive in (("jvp", jvp), ("vjp", vjp)):
            d = tmp_path / f"{k.name}_{tag}"
            d.mkdir()
            hawk.build([k, Kernel("g", derive(k, wrt=wrt))], d, targets=("host",),
                       cache_dir=cache_dir)
            got[f"{k.name}/{tag}"] = hashlib.sha256((d / "g.cpp").read_bytes()).hexdigest()
    assert got == PINNED


def test_a_switch_that_only_reaches_the_outputs_affinely_keeps_its_rule():
    import test_diff_jump as t

    for k in (t.k_lt, t.k_le, t.k_gt, t.k_ge, t.k_and, t.k_or, t.k_nested, t.k_vec,
              t.k_three, t.k_param, t.k_loop_carried, t.k_loop_sum, t.k_loop_param):
        for derive in (jvp, vjp):
            wrt = ("t",) if "t" in k.planes else None
            side = derive(k, wrt=wrt).jump
            assert side["rule"] != "altpath_v1" and "engaged" not in side, k.name
    assert jvp(t.k_spline, wrt=("g",)).jump["rule"] == "altpath_v1"


@kernel(jump=H)
def keyed(x: Scalar, t: Param, acc: Mutable[Scalar]):
    y = 0.0
    if x < t:
        y = 1.0
    hawk.scatter(acc)[m.index(m.floor(t * 4.0), 4)] += y


@kernel(jump=H)
def nested(u: Vector[6], s: Param, tally: Mutable[Scalar]):
    c = 0.0
    for i in range(3):
        for j in range(2):
            y = 0.0
            if u[i] < s:
                y = 1.0
            c = c + m.exp(y)
    tally = c


@kernel(jump=H)
def middle(u: Vector[6], s: Param, tally: Mutable[Scalar]):
    c = 0.0
    y = 0.0
    for i in range(6):
        c = c + 1.0
        if c > 3.0:
            break
        if u[i] < s:
            y = y + 1.0
    tally = m.exp(y)


@kernel(jump=H)
def readback(x: Scalar, t: Param, y: Mutable[Scalar]):
    a = 0.0
    if x < t:
        a = 1.0
    y = m.exp(a) * hawk.sum(a)


@kernel(jump=H)
def count_below(x: Vector[3], t: Param, tally: Mutable[Scalar]):
    y = m.where(x < t, 1.0, 0.0)
    tally = m.dot(y, y)


@pytest.mark.parametrize("derive", (jvp, vjp))
@pytest.mark.parametrize("primal, wrt, text", (
    (keyed, ("t",), "picks a row or a bin"),
    (nested, ("s",), "nested `for`"),
    (middle, ("s",), "middle of a loop body"),
    (readback, ("t",), "read across samples")))
def test_what_the_alternate_path_cannot_follow_is_refused_by_name(derive, primal, wrt, text):
    with pytest.raises(hawk.HawkError, match=text):
        derive(primal, wrt=wrt)


def test_a_vector_comparison_is_followed_per_component(tmp_path, cache_dir):
    n = LANES
    hawk.build([count_below, Kernel("g", jvp(count_below, wrt=("t",)))], tmp_path,
               targets=("host",), cache_dir=cache_dir)
    x = np.ascontiguousarray(np.random.default_rng(2).uniform(0.0, 1.0, (3, n)))
    out = np.zeros(n)
    hawk.run(hawk.load(tmp_path, "g"), x=x, t=0.4, dot_t=1.0, dot_tally=out, n_samples=n)
    _within(out, 3.0)
