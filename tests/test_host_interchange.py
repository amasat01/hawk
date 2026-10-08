# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The interchanged fused-step host loop moves no bit.

A fused-step kernel's host TU (``steps=K`` or ``"auto"``) runs its fused
side as tiles of samples advanced together, trip by trip, with the carried
values in lane arrays (:func:`hawk.emit.host.interchange`); the SAME TU built
with ``-DHAWK_HOST_INTERCHANGE=0`` runs the plain per-sample fused loop it
replaced. What is pinned:

* **Bit identity.** The RK4 oscillator (automatic and ``steps=16``) and the
  RK7(8) attempt kernel (a ``Vector[6]`` carry beside four scalars, a
  per-sample final time), built both ways and launched with the same words
  over random stop steps, leave every plane, the mask and the counter
  byte-equal after EVERY launch: for N below one tile, N not a multiple of
  the tile, several team tiles, an all-finished batch and a batch that never
  finishes. The words include 1 (the single step's branch), 0, a negative
  and an out-of-range word.
* **The chunks.** The trips run per chunk of one or two vector registers of
  lanes, on the chunk's own arrays: the same identity holds for N around
  the chunk and tile sizes with lanes finishing at different trips inside a
  chunk, and the emitted trip loop touches no lane array of the tile.
* **Non-vacuity.** The default build exports its tile (``<name>_host_tile``
  > 0) and the macro-off build exports 0, so the two runs above really are
  the two shapes; the interchanged TU carries the tile loop.
* **Same source on the device.** The device TU carries none of it.
* **Who keeps today's path, and why.** A kernel that is not
  ``sample_local`` (its iterations are not independent, so its lanes cannot
  advance together), a step whose head holds a second lowered ``for`` (the
  interchange cuts at ONE loop), and an active-set kind (its tile is a
  gathered map, not a contiguous run) emit no interchange.
* **The tile rule.** :func:`hawk.emit.host.host_tile_rule` is a pure function
  of its recorded inputs: L1d share, the L2 fallback, whole lane blocks, the
  ``[64, 1024]`` clamp, ``k``; :func:`hawk.emit.host.machine_caches` reads a
  sysfs tree and falls back when it cannot.

Every build here pins the EXACT ``native`` host profile (FMA contraction
off): the two loop shapes are compared bit for bit, and the default fast
profile contracts ``a*b + c`` into FMAs, which leaves each shape within its
ULP bound but makes the comparison depend on how the compiler happened to
fuse each shape.
"""

from __future__ import annotations

import ctypes
import re

import _deploy as L
import numpy as np
import pytest
from _interchange import builds
from conftest import sidecar_of

import hawk
from hawk import Index, Mutable, Param, Scalar, Table, Terminated, Vector
from hawk import math as m
from hawk.artifact import build_bundle
from hawk.emit import BACKENDS, render_source
from hawk.emit.host import (
    FALLBACK_CACHES,
    TILE_MAX,
    TILE_MIN,
    CacheSizes,
    host_tile_rule,
    interchange,
    machine_caches,
)
from hawk.ext import Guard, Kind

#: The macro-off build: the plain per-sample fused loop of the SAME TU.
PLAIN = ("HAWK_HOST_INTERCHANGE=0",)
#: Pinned on: the default for a long step depends on the host compiler
#: (off below GCC 14), and these rows compare the two shapes themselves.
INTERCHANGED = ("HAWK_HOST_INTERCHANGE=1",)
#: The bit-identical host profile every build here pins (module docstring).
EXACT = "native"
DT = 0.01


# --------------------------------------------------------------------------- #
# The kernels (copied here, never imported from a benchmark)
# --------------------------------------------------------------------------- #
@hawk.kernel
def oscillator(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
               terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar],
               k: Mutable[Scalar]):
    """One RK4 step of a damped oscillator; finishes at its own stop step."""
    x0 = x
    v0 = v
    w2 = -(omega * omega)
    c = 2.0 * zeta * omega
    h2 = 0.5 * dt
    h6 = dt * 0.16666666666666666
    a1 = w2 * x0 - c * v0
    v2 = v0 + h2 * a1
    a2 = w2 * (x0 + h2 * v0) - c * v2
    v3 = v0 + h2 * a2
    a3 = w2 * (x0 + h2 * v2) - c * v3
    v4 = v0 + dt * a3
    a4 = w2 * (x0 + dt * v3) - c * v4
    x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
    v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    kn = k + 1.0
    k = kn
    terminated = kn >= nstop


oscillator_x16 = hawk.steps(oscillator, 16)

# RKF7(8) (Fehlberg's tableau): c, the lower-triangular a, the 8th-order weights b
# and the error weights be.
RK_A = (
    (2. / 27,),
    (1. / 36, 1. / 12),
    (1. / 24, 0, 1. / 8),
    (5. / 12, 0, -25. / 16, 25. / 16),
    (1. / 20, 0, 0, 1. / 4, 1. / 5),
    (-25. / 108, 0, 0, 125. / 108, -65. / 27, 125. / 54),
    (31. / 300, 0, 0, 0, 61. / 225, -2. / 9, 13. / 900),
    (2.0, 0, 0, -53. / 6, 704. / 45, -107. / 9, 67. / 90, 3.0),
    (-91. / 108, 0, 0, 23. / 108, -976. / 135, 311. / 54, -19. / 60, 17. / 6, -1. / 12),
    (2383. / 4100, 0, 0, -341. / 164, 4496. / 1025, -301. / 82, 2133. / 4100,
     45. / 82, 45. / 164, 18. / 41),
    (3. / 205, 0, 0, 0, 0, -6. / 41, -3. / 205, -3. / 41, 3. / 41, 6. / 41, 0),
    (-1777. / 4100, 0, 0, -341. / 164, 4496. / 1025, -289. / 82, 2193. / 4100,
     51. / 82, 33. / 164, 12. / 41, 0, 1.0),
)
RK_B = (0, 0, 0, 0, 0, 34. / 105, 9. / 35, 9. / 35, 9. / 280, 9. / 280, 0,
        41. / 840, 41. / 840)
RK_BE = (41. / 840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41. / 840, -41. / 840, -41. / 840)
TOL, SAFETY, MIN_SCALE, MAX_SCALE = 1e-10, 0.9, 0.2, 5.0
EXP_REJECT, EXP_ACCEPT = -1.0 / 8.0, -1.0 / 9.0


def _sum(terms):
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


def _rhs(s):
    x, y, z = s[0], s[1], s[2]
    r2 = x * x + y * y + z * z
    ir3 = 1.0 / (r2 * m.sqrt(r2))
    return [s[3], s[4], s[5], -x * ir3, -y * ir3, -z * ir3]


def _stages(s0, h):
    k = [_rhs(s0)]
    for row in RK_A:
        k.append(_rhs([s0[d] + h * _sum([a * kj[d] for a, kj in zip(row, k) if a != 0])
                       for d in range(6)]))
    s8 = [s0[d] + h * _sum([b * ks[d] for b, ks in zip(RK_B, k) if b != 0])
          for d in range(6)]
    err = [h * _sum([e * ks[d] for e, ks in zip(RK_BE, k) if e != 0]) for d in range(6)]
    return s8, err


def _error_ratio(s0, s8, err):
    """The mixed max-norm of the error."""
    ratio = None
    for d in range(6):
        r = abs(err[d]) / (TOL + TOL * m.max(abs(s8[d]), abs(s0[d])))
        ratio = r if ratio is None else m.max(ratio, r)
    return ratio


@hawk.kernel
def rkf78_attempt(t_final: Scalar, terminated: Terminated, s: Mutable[Vector[6]],
                  t: Mutable[Scalar], h: Mutable[Scalar], n_acc: Mutable[Scalar],
                  n_rej: Mutable[Scalar]):
    """One adaptive RKF7(8) attempt; finishes at its own final time."""
    s0 = s
    t0 = t
    last = h >= t_final - t0
    step = m.where(last, t_final - t0, h)
    s8, err = _stages(s0, step)
    ratio = _error_ratio(s0, s8, err)
    accept = ratio < 1.0
    expo = m.where(accept, EXP_ACCEPT, EXP_REJECT)
    h = step * m.min(MAX_SCALE, m.max(MIN_SCALE, SAFETY * ratio ** expo))
    s = m.where(accept, m.vec(s8), s0)
    t1 = m.where(accept, m.where(last, t_final, t0 + step), t0)
    t = t1
    n_acc = n_acc + m.where(accept, 1.0, 0.0)
    n_rej = n_rej + m.where(accept, 0.0, 1.0)
    terminated = t1 >= t_final


KERNELS = {"oscillator": oscillator, "oscillator_x16": oscillator_x16,
           "rkf78_attempt": rkf78_attempt}


@pytest.fixture(scope="module")
def shapes(tmp_path_factory):
    """``{"interchanged"|"plain": Bundle}``: the same kernels, the two builds."""
    cache = str(tmp_path_factory.mktemp("interchange_cache"))
    kernels = list(KERNELS.values())
    return {
        "interchanged": build_bundle(kernels, tmp_path_factory.mktemp("ic_on"),
                                     targets=("host",), cache_dir=cache,
                                     defines=INTERCHANGED, host_profile=EXACT),
        "plain": build_bundle(kernels, tmp_path_factory.mktemp("ic_off"),
                              targets=("host",), cache_dir=cache, defines=PLAIN,
                              host_profile=EXACT),
    }


def _exported_tile(bundle, name) -> int:
    lib = ctypes.CDLL(str(bundle.directory / f"{name}.so"))
    return ctypes.c_int64.in_dll(lib, f"{name}_host_tile").value


# --------------------------------------------------------------------------- #
# Bit identity
# --------------------------------------------------------------------------- #
def _planes(name, n, case, seed):
    rng = np.random.default_rng([seed, n])
    term = np.zeros(n, dtype=bool) if case != "all_finished" else np.ones(n, dtype=bool)
    if name.startswith("oscillator"):
        stop = rng.integers(1, 300, n).astype(float)
        if case == "never":
            stop[:] = 1e9
        return {"omega": rng.uniform(1.0, 10.0, n), "zeta": rng.uniform(0.01, 0.2, n),
                "nstop": stop, "x": rng.uniform(-1, 1, n), "v": rng.uniform(-1, 1, n),
                "k": np.zeros(n), "terminated": term}
    e = rng.uniform(0.0, 0.7, n)
    s = np.zeros((6, n))
    s[0] = 1.0 - e
    s[4] = np.sqrt((1.0 + e) / (1.0 - e))
    t_final = rng.uniform(0.05, 2.0, n) if case != "never" else np.full(n, 1e9)
    return {"t_final": t_final, "s": s, "t": np.zeros(n), "h": np.full(n, 0.01),
            "n_acc": np.zeros(n), "n_rej": np.zeros(n), "terminated": term}


def _launches(bundle, name, planes, words):
    """Launch ``name`` with each word of ``words``; the planes, mask and
    counter after every launch."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    plugin = L.host_plugin(bundle.directory, name, sidecar)
    planes = {key: value.copy() for key, value in planes.items()}
    counter = np.zeros(1, dtype=np.uint32)
    counter[0] = np.count_nonzero(planes["terminated"])
    extra = {}
    auto = sidecar["finish"]["steps"] == "auto"
    if auto:
        extra["fused_steps"] = np.zeros(1, dtype=np.int64)
    kw = {"dt": DT} if "dt" in dict((n, r) for r, n in sidecar["arg_spec"]) else {}
    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        finished_count=counter.view(np.int32), **kw, **planes, **extra)
    snaps = []
    for word in words:
        if auto:
            extra["fused_steps"][0] = word
        bound.launch()
        snaps.append({**{key: value.copy() for key, value in planes.items()},
                      "counter": counter.copy()})
    return snaps


#: The words driven: the bound, then a mix (one step, odd remainders, 0, a
#: negative and an out-of-range word, clamped by the kernel), then the bound.
WORDS = (64, 64, 1, 3, 64, 7, 0, -5, 100, 2, 33) + (64,) * 6


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("n,case", [(1, "spread"), (37, "spread"), (200, "spread"),
                                    (1001, "spread"), (300, "all_finished"),
                                    (300, "never")])
def test_interchanged_and_plain_are_byte_identical(shapes, name, n, case):
    planes = _planes(name, n, case, seed=20261004)
    words = WORDS if name != "oscillator_x16" else (1,) * 20
    got = _launches(shapes["interchanged"], name, planes, words)
    want = _launches(shapes["plain"], name, planes, words)
    for launch, (a, b) in enumerate(zip(got, want)):
        for key in b:
            assert a[key].tobytes() == b[key].tobytes(), (
                f"{name} n={n} {case}: {key} differs after launch {launch} "
                f"(word {words[launch]})")
    last = want[-1]
    if case == "spread":
        assert last["terminated"].all(), "the run must reach every sample's stop"
    if case == "all_finished":
        for key, value in planes.items():
            assert last[key].tobytes() == value.tobytes(), key
    if case == "never":
        assert not last["terminated"].any() and int(last["counter"][0]) == 0


def _mixed(name, n, seed):
    """Early termination mixed lane by lane: a quarter of the samples
    finished before the first launch, stops of one and two steps beside
    long and never-reached ones, so a chunk holds live, finishing and
    finished lanes at once and the words' caps cut the rest."""
    rng = np.random.default_rng([seed, n, 7])
    planes = _planes(name, n, "spread", seed)
    planes["terminated"] = rng.random(n) < 0.25
    if name.startswith("oscillator"):
        planes["nstop"] = rng.choice([1.0, 2.0, 5.0, 40.0, 300.0, 1e9], n)
    else:
        planes["t_final"] = rng.choice([1e-3, 0.02, 0.5, 2.0, 1e9], n)
    # whatever the draw: the first sample is capped, the last one finishes
    key, never, soon = (("nstop", 1e9, 2.0) if name.startswith("oscillator")
                        else ("t_final", 1e9, 1e-3))
    planes[key][0], planes[key][-1] = never, soon
    planes["terminated"][0] = planes["terminated"][-1] = False
    return planes


#: Sample counts around the chunk (8 or 16 lanes under AVX-512, 4 or 8 under
#: AVX, 2 or 4 otherwise) and the tile's boundaries: below one chunk, one
#: chunk exactly, one chunk +- 1, several chunks +- 1, a tile +- 1.
CHUNK_COUNTS = (3, 7, 8, 9, 15, 16, 17, 31, 33, 127, 129)


@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("n", CHUNK_COUNTS)
def test_chunk_boundaries_and_mixed_termination_are_byte_identical(shapes, name, n):
    """The chunked trips against the plain fused loop, after every launch:
    N around the chunk and tile sizes, lanes finishing at different trips
    inside one chunk (and some finished before the run), and words that cap
    a launch below a lane's own stop (the per-sample step cap)."""
    planes = _mixed(name, n, seed=20261005)
    words = WORDS if name != "oscillator_x16" else (1,) * 20
    got = _launches(shapes["interchanged"], name, planes, words)
    want = _launches(shapes["plain"], name, planes, words)
    for launch, (a, b) in enumerate(zip(got, want)):
        for key in b:
            assert a[key].tobytes() == b[key].tobytes(), (
                f"{name} n={n}: {key} differs after launch {launch} "
                f"(word {words[launch]})")
    last = want[-1]
    done = last["terminated"]
    assert done.any() and not done.all(), "the batch mixes finished and capped samples"


def _bound(bundle, name, planes):
    """``(bound plan, planes, counter, word)``: ``name`` bound over copies
    of ``planes`` with its own counter and ``fused_steps`` word."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    plugin = L.host_plugin(bundle.directory, name, sidecar)
    planes = {key: value.copy() for key, value in planes.items()}
    counter = np.zeros(1, dtype=np.uint32)
    counter[0] = np.count_nonzero(planes["terminated"])
    word = np.zeros(1, dtype=np.int64)
    kw = {"dt": DT} if "dt" in dict((n, r) for r, n in sidecar["arg_spec"]) else {}
    bound = eplan.plan(plugin, structure=eexec.HostTeam).bind(
        finished_count=counter.view(np.int32), fused_steps=word, **kw, **planes)
    return bound, planes, counter, word


@pytest.mark.parametrize("serial", [True, False])
@pytest.mark.parametrize("name", ["oscillator", "rkf78_attempt"])
@pytest.mark.parametrize("n,left", [(1, 500), (7, 500), (9, 500), (200, 500),
                                    (1001, 150), (1001, 37)])
def test_the_run_entry_is_the_launch_sequence(shapes, name, n, left, serial):
    """``<name>_host_run`` with the steps LEFT in its word, over the whole
    batch in one call (serial or as a team of tiles), against one launch of
    ``<name>_host`` per round with the word ``min(64, left)`` until every
    sample finished or nothing is left: the same planes, mask and counter
    byte for byte, and the rounds it reports are the launches."""
    import eagle.exec as eexec

    lib = ctypes.CDLL(str(shapes["interchanged"].directory / f"{name}.so"))
    run = ctypes.cast(getattr(lib, f"{name}_host_run"), ctypes.c_void_p).value
    rounds = ctypes.c_int64.in_dll(lib, f"{name}_host_run_rounds")
    assert ctypes.c_int64.in_dll(lib, f"{name}_host_run_k").value == 64
    planes = _mixed(name, n, seed=20261006)

    bound, want, counter, word = _bound(shapes["interchanged"], name, planes)
    whole = eexec.Partition(0, n, n)
    budget, launches = left, 0
    while budget > 0 and int(counter[0]) != n:
        word[0] = min(64, budget)
        eexec.HostTeam.run(bound._entry, bound._addrs, whole)
        budget -= int(word[0])
        launches += 1
    want_counter = counter.copy()

    bound, got, counter, word = _bound(shapes["interchanged"], name, planes)
    word[0] = left
    rounds.value = 0
    if serial:
        eexec.HostTeam.run_serial(run, bound._addrs, whole)
    else:
        eexec.HostTeam.run(run, bound._addrs, whole, bytes_per_sample=1 << 20)
    for key in want:
        assert got[key].tobytes() == want[key].tobytes(), (name, n, key)
    assert counter.tobytes() == want_counter.tobytes()
    assert rounds.value == launches, (rounds.value, launches)
    assert int(word[0]) == left, "the caller's word is read, never written"


def test_only_an_automatic_tu_has_a_run_entry(shapes):
    lib = ctypes.CDLL(str(shapes["interchanged"].directory / "oscillator_x16.so"))
    assert not hasattr(lib, "oscillator_x16_host_run")
    text = (shapes["interchanged"].directory / "oscillator.cpp").read_text()
    assert "#pragma omp" not in text, "the TU stays serial: threading is eagle's"


def _trip_loop(text: str) -> str:
    """The interchanged build's trip loop: from its ``for`` to its exit."""
    start = text.index("for (Int hawk_trip = 0;")
    return text[start:text.index("if (hawk_live == 0) break;", start)]


def test_the_trip_loop_runs_on_the_chunk_alone():
    """The hot loop reads and writes only the chunk's own arrays: no lane
    array of the tile (``hawk_lc*``/``hawk_li*``), no tile-strided index, no
    tile live flag; the tile's lane arrays are touched once before the trips
    and once after. A short step's chunk is two vectors, a long one's one."""
    from hawk.emit.host import SHORT_STEP_OPS, chunk_vectors

    assert (chunk_vectors(1), chunk_vectors(SHORT_STEP_OPS),
            chunk_vectors(SHORT_STEP_OPS + 1)) == (2, 2, 1)
    vectors = {"oscillator": 2, "oscillator_x16": 2, "rkf78_attempt": 1}
    for kernel in KERNELS.values():
        host = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["host"],
                             kind=kernel.kind, one_step=getattr(kernel, "one_step", None))
        text = builds(host.text)["interchanged"]
        trips = _trip_loop(text)
        for token in (r"\bhawk_lc\d", r"\bhawk_li\d", r"\bhawk_tile\b", r"\bhawk_on\[",
                      r"\bhawk_cnt\["):
            assert not re.search(token, trips), (kernel.name, token)
        assert re.search(r"hawk_abi::lane_get<.+?, hawk_w>\(hawk_rc0, hawk_l\);", trips)
        assert "hawk_abi::lane_keep<hawk_w>(hawk_rc0, hawk_l, hawk_lane_on," in trips
        assert "for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {" in trips
        assert (f"constexpr aether::offset_t hawk_w = hawk_abi::kHostChunk * "
                f"{vectors[kernel.name]};") in text, kernel.name
        assert text.count("for (Int hawk_trip = 0;") == 1, kernel.name


def test_the_two_builds_are_the_two_shapes(shapes):
    """Non-vacuity of the row above: the default build runs tiles, the
    macro-off build does not."""
    for name in KERNELS:
        tile = _exported_tile(shapes["interchanged"], name)
        assert TILE_MIN <= tile <= TILE_MAX and tile % 64 == 0, (name, tile)
        assert _exported_tile(shapes["plain"], name) == 0, name
        text = (shapes["interchanged"].directory / f"{name}.cpp").read_text()
        assert "#if HAWK_HOST_INTERCHANGE" in text
        assert f"constexpr aether::offset_t hawk_tile = {tile};" in text
        assert "// hawk host tile rule: tile=" in text


def test_the_device_source_carries_no_interchange():
    for kernel in KERNELS.values():
        cuda = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                             kind=kernel.kind,
                             one_step=getattr(kernel, "one_step", None)).text
        for token in ("HAWK_HOST_INTERCHANGE", "hawk_tile", "lane_put", "_host_tile"):
            assert token not in cuda, (kernel.name, token)


def test_both_shapes_wrap_the_one_body_string():
    """The plain build of the host TU is the fused body string verbatim."""
    for kernel in KERNELS.values():
        host = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["host"],
                             kind=kernel.kind, one_step=getattr(kernel, "one_step", None))
        cuda = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["cuda"],
                             kind=kernel.kind, one_step=getattr(kernel, "one_step", None))
        assert host.body == cuda.body
        fused = host.body.split("\n")[-1]
        assert fused in builds(host.text)["plain"]


# --------------------------------------------------------------------------- #
# Who keeps today's path
# --------------------------------------------------------------------------- #
@hawk.kernel
def gathered(table: Table[Scalar], where: Index, nstop: Scalar, terminated: Terminated,
             y: Mutable[Scalar], k: Mutable[Scalar]):
    """Not ``sample_local``: a cross-sample read."""
    y = y + table.at(where)
    kn = k + 1.0
    k = kn
    terminated = kn >= nstop


@hawk.kernel
def head_loop(x: Scalar, g: Param, nstop: Scalar, terminated: Terminated,
              y: Mutable[Scalar], k: Mutable[Scalar]):
    """A second lowered ``for`` beside the fused loop (it reads no carry, so
    it runs once before it)."""
    acc = x
    for j in range(4):
        acc = acc + g * m.sin(acc + j)
    y = y + acc
    kn = k + 1.0
    k = kn
    terminated = kn >= nstop


ACTIVE = Kind("interchange_active", guard=Guard(active_set=True))


def test_what_keeps_the_plain_fused_loop_and_why():
    for kernel, why in ((gathered, "not sample_local: dependent iterations"),
                        (head_loop, "two lowered loops: the cut is at ONE")):
        assert getattr(kernel, "one_step", None) is not None, kernel.name
        assert interchange(kernel.name, kernel.sinks, kernel.walk, kernel.kind) is None, why
        text = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS["host"],
                             kind=kernel.kind, one_step=kernel.one_step).text
        assert "HAWK_HOST_INTERCHANGE" not in text, why
    # an active-set kind: the tile is a gathered map, not a contiguous run
    from hawk.emit.backend import ActiveSpec
    from hawk.emit.spelling import binding_name

    active = ActiveSpec(binding_name("lookup", "active_map"),
                        binding_name("lookup", "active_count"))
    text = render_source(oscillator.name, oscillator.sinks, oscillator.walk,
                         BACKENDS["host"], kind=ACTIVE, active=active,
                         one_step=oscillator.one_step).text
    assert "HAWK_HOST_INTERCHANGE" not in text


# --------------------------------------------------------------------------- #
# The tile rule
# --------------------------------------------------------------------------- #
CACHES = CacheSizes(32 * 1024, 1024 * 1024, "test")


def test_the_tile_rule_fills_half_of_l1d_in_whole_lane_blocks():
    rule = host_tile_rule(96, 30, True, 64, caches=CACHES)
    assert (rule.tile, rule.level, rule.k) == (128, "L1d", 64)   # 16384 // 96 = 170
    assert (rule.lane_bytes, rule.op_count, rule.finishes, rule.stop) == (96, 30, True, 64)
    assert (rule.l1d, rule.l2, rule.cache_source) == (32768, 1048576, "test")
    assert "tile=128 k=64 level=L1d (lane_bytes=96 op_count=30" in rule.comment()


def test_the_tile_rule_clamps_and_falls_back_to_l2():
    assert host_tile_rule(8, 5, True, 16, caches=CACHES).tile == TILE_MAX
    big = host_tile_rule(1000, 900, True, 64, caches=CACHES)   # 64 * 1000 > 16384
    assert (big.level, big.tile) == ("L2", 512)                 # 524288 // 1000 = 524
    assert host_tile_rule(40000, 5, True, 64, caches=CACHES).tile == TILE_MIN
    assert host_tile_rule(96, 30, False, 64, caches=CACHES).k == 1
    # op count is recorded, not (yet) a term
    assert (host_tile_rule(96, 1, True, 64, caches=CACHES).tile
            == host_tile_rule(96, 10_000, True, 64, caches=CACHES).tile)


def test_machine_caches_reads_sysfs_and_falls_back(tmp_path):
    for index, (level, kind, size) in enumerate(
            [(1, "Data", "48K"), (1, "Instruction", "32K"), (2, "Unified", "2048K"),
             (3, "Unified", "16M")]):
        d = tmp_path / f"index{index}"
        d.mkdir()
        (d / "level").write_text(f"{level}\n")
        (d / "type").write_text(f"{kind}\n")
        (d / "size").write_text(f"{size}\n")
    got = machine_caches(str(tmp_path))
    assert (got.l1d, got.l2) == (48 * 1024, 2048 * 1024)
    assert machine_caches(str(tmp_path / "missing")) == FALLBACK_CACHES
    here = machine_caches()
    assert here.l1d > 0 and here.l2 >= here.l1d
