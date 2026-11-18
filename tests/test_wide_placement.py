# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where a per-lane value wider than a thread's registers lives (:mod:`raptor.wings.base.wide`).

Rows: what the pass places and what it leaves (the corpus of :mod:`_wide_kernels`), the sidecar record and
its scratch planes, the build option ``wide_form``, the host text that places nothing, the identity that keeps
a store under a condition one store (``select(c, v, set(v, i, x))``), the 32-bit integers of a lane set,
and, on a device, the corpus values against NumPy.

Run as a script, this file is the wide gate (``make gate-wide``): it builds and runs the corpus here and in the
0.7.2 wheel's environment and prints one PASS or FAIL line per check."""

import json
import os
import pathlib
import sys
import tempfile
import time
import warnings

import _wide_kernels as W
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import HawkError, Mutable, Scalar, Table, Vector
from hawk.artifact import build_bundle
from hawk.ext import IndexSet, Kind
from hawk import _ir

TARGETS_BOTH = ("cuda", "host")

def _unit(bundle, name, suffix=".cu"):
    return (pathlib.Path(bundle.directory) / f"{name}{suffix}").read_text()


#: The arch the committed local-memory figures (``W.LOCAL_OK``, the ``sm61`` class tables) were measured on.
MEASURED_ARCH = "sm_61"


def _a1_arch() -> str:
    """The arch the A1 rows compile for: the one the committed figures were measured on."""
    return MEASURED_ARCH






def _build(entry, tmp_path, cache_dir, **options):
    options.setdefault("device_arch", _a1_arch())
    targets = options.pop("targets", ("cuda",))
    kernel = W.CORPUS[entry].kernel
    kind = {} if W.CORPUS[entry].kind is None else {"kind": W.CORPUS[entry].kind}
    return kernel, build_bundle([kernel], tmp_path, targets=targets, cache_dir=cache_dir, **kind, **options)


# -- what the pass places -----------------------------------------------------------------------

















def test_launch_bounds_are_refused_on_a_warp_per_lane_kernel(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"launch_bounds.*fill_mask.*warp per lane"):
        build_bundle([W.fill_mask], tmp_path, targets=("cuda",), cache_dir=cache_dir, wide_form="team", launch_bounds=(128, 4))




def test_the_team_form_is_refused_by_name(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"wide_form=.team. \(a warp per lane\) is built for device targets only"):
        build_bundle([W.fill_mask], tmp_path, targets=("host",), cache_dir=cache_dir, wide_form="team")


# -- what the build writes -----------------------------------------------------------------------

def test_the_sidecar_names_every_plane_and_the_runtime_makes_the_scratch_ones(tmp_path, cache_dir):
    _, bundle = _build("fill_mask", tmp_path, cache_dir, targets=TARGETS_BOTH, host_profile="native")     # a host target keeps the placed form
    side = sidecar_of(bundle, "fill_mask")
    assert side["launch"]["wide"] == {"hwd0": {"layout": "warp_tiled"}}     # the one fact a run reads: which scratch plane is tiled by warp
    assert side["team_planes"] == {"hwd0": {"dtype": "real", "per_lane": W.W}}
    assert [st["kernel"] for st in side["stages"]] == ["fill_mask"] and ("wide_in", "hwd0") in map(tuple, side["arg_spec"])
    assert side["launch"]["local_bytes"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["matvec_in", "matvec_in_vjp", "dot_norm", "rt_write", "fill_mask_jvp"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
@pytest.mark.parametrize("n", [2048, 2045])         # the second leaves the last block with lanes past the end
def test_the_warp_form_is_exact_and_replays_bit_for_bit_on_the_device(name, mode, n, cache_dir, tmp_path):
    import eagle.plan
    from eagle import arrays
    from eagle import _split
    from eagle._alloc import synchronize
    from eagle.registry import load_manifest

    entry, real = W.CORPUS[name], np.float32 if mode == "float32" else np.float64
    planes = entry.planes(np.random.default_rng(7), real, n, True)
    bound_args = {k: arrays.asarray(np.ascontiguousarray(v), device="cuda:0") for k, v in planes.items()}
    out = arrays.asarray(np.zeros(entry.shape(n), real), device="cuda:0")
    # the warp (team) form, asked for through the build's own wide_form; the plan runs the plugin that build wrote
    bundle = build_bundle([entry.kernel], tmp_path / name, targets=("cuda",), mode=mode, cache_dir=cache_dir, wide_form="team")
    # a kernel with stages runs its stages only when they are linked to its plugin: eagle.deploy (eagle.plan.auto given the kernel)
    # does that, but it builds its own bundle and takes no wide_form, so the build above is linked the way deploy links its own
    by_name = hawk.artifact.plugins(bundle, device_loader=load_manifest)
    _split.link(by_name, bundle)
    bound = eagle.plan.auto(by_name[entry.kernel.name]).bind(**bound_args, **{entry.out: out})
    got = []
    for _ in range(3):
        bound.launch()
        synchronize()
        got.append(np.asarray(out.get()).copy())
    assert np.array_equal(got[0], entry.want(real, planes, n))
    assert all(np.array_equal(got[0], g) for g in got[1:])


@pytest.mark.parametrize("name", ["matvec_in", "matvec_in_vjp", "dot_norm", "rt_write", "fill_mask_jvp"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_the_host_arm_of_the_warp_form_corpus_is_exact(name, mode, tmp_path, cache_dir):
    """Host arm of the device row (a device run needs a GPU): the corpus kernels the warp form is
    checked on read back from a host build against the corpus's own numpy value, bit for bit."""
    import _oracle as O

    entry, real = W.CORPUS[name], np.float32 if mode == "float32" else np.float64
    n = W.LANES_HOST
    planes = entry.planes(np.random.default_rng(7), real, n, True)
    run = {k: v.copy() for k, v in planes.items()}
    run[entry.out] = np.zeros(entry.shape(n), real)
    bundle = build_bundle([entry.kernel], tmp_path / name, mode=mode, targets=("host",), cache_dir=cache_dir,
                          host_profile="native")
    kernel = O.load(bundle.directory, entry.kernel.name, sidecar_of(bundle, entry.kernel.name))
    for _role, nm in kernel.arg_spec:           # a scratch plane of a placement: a host text never touches it
        if nm.startswith("hwd"):
            run[nm] = np.zeros(1)
    got = np.array(O.run_kernel(kernel, n, **run))
    assert np.array_equal(got, entry.want(real, planes, n))


@pytest.mark.parametrize("entry", ["fill_mask", "matvec_in", "matvec_computed", "wide51"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_a_placed_corpus_kernel_holds_no_local_memory(entry, mode, tmp_path, cache_dir):
    # with a host target in the bundle ``auto`` keeps the placed form (the host has no warp)
    _, bundle = _build(entry, tmp_path, cache_dir, mode=mode, targets=TARGETS_BOTH, host_profile="native")
    assert sidecar_of(bundle, W.CORPUS[entry].kernel.name)["launch"]["local_bytes"] == 0


def test_the_device_text_writes_the_plane_in_a_loop_and_reads_it_back_by_the_rolled_sum(tmp_path, cache_dir):
    _, bundle = _build("matvec_in", tmp_path, cache_dir, targets=TARGETS_BOTH, host_profile="native")
    text = _unit(bundle, "matvec_in")
    assert "hawk_abi::tiled<Real, 512>" in text and "aether::rt::sum(win_hwd0_i)" in text
    assert "Item<Real, 512>" not in text and "AETHER_NO_UNROLL()" in text


def test_the_host_text_renders_a_placement_as_its_value(tmp_path, cache_dir):
    kernel = W.fill_mask
    bundle = build_bundle([kernel], tmp_path, targets=TARGETS_BOTH, cache_dir=cache_dir, host_profile="native")
    host = _unit(bundle, "fill_mask", ".cpp")
    assert "hawk_abi::tiled" not in host and "Item<Real, 512>" in host


def test_a_kernel_with_nothing_wide_is_one_kernel_as_it_was(tmp_path, cache_dir):
    _, bundle = _build("masked_sum", tmp_path, cache_dir)
    side = sidecar_of(bundle, "masked_sum")
    assert "stages" not in side and "team_planes" not in side and "wide" not in side["launch"]


# -- the derivatives --------------------------------------------------------------------------





@pytest.mark.parametrize("entry", ["fill_mask_vjp", "fill_mask_jvp", "matvec_in_vjp", "matvec_in_jvp", "steps_acc_jvp_k8"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_a_derivative_corpus_kernel_holds_only_the_local_memory_the_register_form_wins(entry, mode, tmp_path, cache_dir):
    with warnings.catch_warnings():
        if (entry, mode) in W.LOCAL_OK:   # the frame the register form keeps on purpose: the build's resource warning names it
            warnings.filterwarnings("ignore", "hawk: kernel .* local frame", UserWarning)
        _, bundle = _build(entry, tmp_path, cache_dir, mode=mode)
    assert sidecar_of(bundle, W.CORPUS[entry].kernel.name)["launch"]["local_bytes"] == W.LOCAL_OK.get((entry, mode), 0)


@pytest.mark.parametrize("entry", ["fill_mask_seg", "matvec_in_seg"])
def test_an_automatic_kernel_places_the_wide_values_of_its_fused_and_its_single_step_body(entry, tmp_path, cache_dir):
    kernel, bundle = _build(entry, tmp_path, cache_dir, mode="float64", targets=TARGETS_BOTH, host_profile="native")   # a host target keeps the placed form
    side = sidecar_of(bundle, kernel.name)
    assert side["launch"]["local_bytes"] == 0
    planes = {p for p in side["launch"]["wide"] if p != "class"}
    assert {p for p in side["team_planes"]} <= planes and len(side["team_planes"]) >= 2   # one set of scratch planes per body


@pytest.mark.parametrize("entry", ["fill_mask_seg", "matvec_in_seg"])
@pytest.mark.parametrize("form", ["auto", "team"])
def test_the_branch_units_of_a_segmented_kernel_are_placed_for_a_device(entry, form, tmp_path, cache_dir):
    """A branch unit carries no steps record: placing its wide values reads none. The test passes where the device
    compiler is absent too: the placement ran when the build reached the compile."""
    from hawk.errors import ToolchainAbsent

    seg, units = W.segmented_units(W.CORPUS[entry].kernel)
    kind = {} if W.CORPUS[entry].kind is None else {"kind": W.CORPUS[entry].kind}
    try:
        bundle = build_bundle([seg], tmp_path, targets=("cuda",), cache_dir=cache_dir, wide_form=form, **kind)
    except ToolchainAbsent:
        return
    assert all(sidecar_of(bundle, u) for u in units)








def test_a_wide_output_is_staged_through_a_block_that_serves_several_lanes(tmp_path, cache_dir):
    kernel, bundle = _build("fill_mask_jvp", tmp_path, cache_dir, mode="float32")
    team = sidecar_of(bundle, kernel.name)["stages"][0]["team"]
    assert team["lanes"] > 1 and team["threads"] == 32 * team["lanes"]
    kernel, bundle = _build("matvec_in", tmp_path / "narrow", cache_dir, mode="float32")      # 24 elements: a warp is its own block
    assert sidecar_of(bundle, kernel.name)["stages"][0]["team"] == {"threads": 32}


@pytest.mark.parametrize("entry", ["fill_mask_seg", "steps_matvec_k8", "steps_early_k8", "dot_norm"])
def test_a_stepped_or_automatic_kernel_is_a_warp_per_lane_on_a_device_bundle(entry, tmp_path, cache_dir):
    kernel, bundle = _build(entry, tmp_path, cache_dir, mode="float32")
    side = sidecar_of(bundle, kernel.name)
    team = side["stages"][0]["team"]
    assert team["threads"] == 32 * team.get("lanes", 1) and side["launch"]["local_bytes"] == 0
    assert "team_planes" not in side            # nothing is placed in a scratch plane: the value is sliced across the lanes




def test_the_text_of_the_one_measured_class_does_not_depend_on_the_arch(tmp_path, cache_dir):
    texts = {arch: _unit(_build("matvec_in", tmp_path / arch, cache_dir, device_arch=arch)[1], "matvec_in")
             for arch in ("sm_61", "sm_75")}
    assert texts["sm_61"] == texts["sm_75"]


# -- a stepping kernel ---------------------------------------------------------------------------



def _step_loops(sinks):
    """The loops of the sinks that stop on an exit condition (the step loops), each once."""
    return [n for n in _ir.post_order(sinks) if _ir.is_a(n, "loop") and n.field_kind("exit_cond")]






@pytest.mark.parametrize("entry", ["steps_fill_k1", "steps_fill_k8", "steps_early_k8", "steps_matvec_k1", "steps_matvec_k8", "narrow4", "narrow16",
                                   "rotate16"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_a_stepping_or_narrow_corpus_kernel_holds_no_local_memory(entry, mode, tmp_path, cache_dir):
    _, bundle = _build(entry, tmp_path, cache_dir, mode=mode)
    assert sidecar_of(bundle, W.CORPUS[entry].kernel.name)["launch"]["local_bytes"] == W.LOCAL_OK.get((entry, mode), 0)


# -- the identity ------------------------------------------------------------------------------

def _vector_selects(kernel):
    found, nodes = [], list(_ir.canonical_nodes(kernel.sinks)[0])
    while nodes:
        n = nodes.pop()
        if _ir.is_a(n, "loop"):
            nodes.extend(_ir.body_nodes(n))
        elif n.cls == "Select" and _ir.type_of(n).shape:
            found.append(n)
    return found


def test_a_store_under_a_condition_is_one_store_of_one_element_for_a_wide_vector():
    assert not _vector_selects(W.wide51)            # set(v, i, select(c, v[i], x)), not select(c, v, set(v, i, x))


def test_a_narrow_vector_stored_under_a_condition_is_one_store_too():
    @hawk.kernel
    def both(c: Vector[8], y: Mutable[Vector[8]]):
        r = W.zeros(8)
        for j in range(8):
            if c[j] > 0.5:
                r[j] = 1.0
        y = r

    assert not _vector_selects(both)




# -- the integers of a lane set ---------------------------------------------------------------

def _set_text(tmp_path, cache_dir, kind, name):
    def body(lane: hawk.Index, th: Table[hawk.Scalar], y: Mutable[Scalar]):
        y = th.at(lane * 3 + 1) + 1.0

    kernel = hawk.kernel(body, kind=kind)
    return _unit(build_bundle([kernel], tmp_path / name, targets=("cuda",), cache_dir=cache_dir, kind=kind), "body")


def test_a_lane_set_of_32_bit_width_has_a_32_bit_integer(tmp_path, cache_dir):
    assert "using Int = int;" in _set_text(tmp_path, cache_dir, Kind("s32", lanes=IndexSet(width=32)), "a")


def test_a_lane_set_of_64_bit_width_keeps_the_64_bit_integer(tmp_path, cache_dir):
    assert "using Int = long long;" in _set_text(tmp_path, cache_dir, Kind("s64", lanes=IndexSet(width=64)), "b")


def test_a_kernel_on_every_lane_keeps_the_64_bit_integer(tmp_path, cache_dir):
    _, bundle = _build("fill_mask", tmp_path, cache_dir)
    assert "using Int = long long;" in _unit(bundle, "fill_mask")


# -- the corpus on a device -------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("entry,mode", [(e, m_) for e in W.CORPUS for m_ in W.CORPUS[e].modes])
def test_the_corpus_is_exact_on_the_device(entry, mode, cache_dir):
    with warnings.catch_warnings():
        if (entry, mode) in W.LOCAL_OK:   # the frame the register form keeps on purpose (the A1 row pins its size)
            warnings.filterwarnings("ignore", "hawk: kernel .* local frame", UserWarning)
        if W.CORPUS[entry].segmented:     # the rows rewrite their selector on purpose: eagle's advice to stay fused is expected
            warnings.filterwarnings("ignore", "eagle.simulate: dispatch='segmented'")
        got = W.run_device(entry, mode, 4096, cache_dir=cache_dir, repeats=1)
    assert np.array_equal(got["out"], got["want"])


# -- the gate (``make gate-wide``) -----------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
#: The environment of the published 0.7.2 release `make compare-0.7.2` installs from ci/compare_0.7.2.txt (the target sets this).
TWIN = pathlib.Path(os.environ["HAWK_TWIN_VENV"]) if os.environ.get("HAWK_TWIN_VENV") else None




ARCH = _arch() if __name__ == "__main__" else None

_CLOCK = [time.perf_counter()]


def _line(name: str, ok: bool, detail: str) -> bool:
    now = time.perf_counter()
    print(f"{'PASS' if ok else 'FAIL'}  {name:<8} {detail}  [{now - _CLOCK[0]:.0f}s]", flush=True)
    _CLOCK[0] = now
    return ok


def _work(root: str) -> pathlib.Path:
    path = pathlib.Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resources(root: pathlib.Path, cache: pathlib.Path, arch: str = ARCH) -> dict:
    """``{(entry, mode): resources}`` of the corpus built for ``arch``."""
    out = {}
    for entry in W.CORPUS:
        for mode in W.CORPUS[entry].modes:
            bundle = W.build(entry, mode, arch, root / f"{entry}_{mode}_{arch}", cache)
            out[entry, mode] = W.resources(bundle, entry)
            out[entry, mode]["text"] = W.unit_text(bundle, entry)
    return out






def gate(which: str = "gpu") -> int:
    """``gpu``: the correctness checks (A1, C8, A3-numpy, A3-fd, A5); ``compare``: the checks against the published 0.7.2
    release (A3-twin, SPEED), which need ``HAWK_TWIN_VENV``."""
    if which == "compare" and TWIN is None:
        print("FAIL  compare  HAWK_TWIN_VENV is unset: run `make compare-0.7.2`, which installs the release")
        return 1
    with tempfile.TemporaryDirectory(prefix="gate_wide_", dir=os.environ.get("TMPDIR")) as scratch:
        return (_gate if which == "gpu" else _compare)(pathlib.Path(scratch))


#: Compiled kernels persist between runs (a cache is content-addressed: a stale entry cannot be served): a cold gate compiles the
#: 0.7.2 wheel's spilling kernels, which takes minutes, and a warm one takes about two.
CACHE = pathlib.Path(os.environ.get("HAWK_GATE_CACHE", pathlib.Path(tempfile.gettempdir()) / "hawk_gate_wide_cache"))
# CACHE also keeps digests of the NumPy references and of the release's values (under "want" and "twin/results"), keyed by the
# digest of the corpus source: a missing one is recomputed, and no verdict depends on one being there.


def _gate(root: pathlib.Path) -> int:
    """The correctness checks, which need no other release: A1, C8, A3-numpy, A3-fd, A5."""
    ok = True
    # A1 and C8: ptxas, from the build
    mine = _resources(root / "tree", CACHE / "tree")
    bad = {k: v for k, v in mine.items() if (v["spill_store_B"] and k not in W.LOCAL_OK)
           or max(v["stack_B"], v["local_bytes"]) > W.LOCAL_OK.get(k, 0)}
    ok &= _line("A1", not bad, f"{len(mine)} builds at {ARCH}: no spill, stack or local bytes but in the rows the register form wins (W.LOCAL_OK)" if not bad
                else f"nonzero: { {f'{k[0]}/{k[1]}': (v['stack_B'], v['spill_store_B'], v['local_bytes']) for k, v in bad.items()} }")
    regs = {}
    for arch in dict.fromkeys((ARCH, "sm_80")):
        for bits in (32, 64):
            kind = Kind(f"gate_set{bits}", lanes=IndexSet(width=bits))
            kernel = W.lane_set_kernel(kind)
            bundle = build_bundle([kernel], root / f"c8_{bits}_{arch}", mode="float32", targets=("cuda",), device_arch=arch,
                                  cache_dir=CACHE / "tree", kind=kind)
            regs[arch, bits] = json.loads((pathlib.Path(bundle.directory) / f"{kernel.name}.json").read_text())["ptxas"][kernel.name]["regs"]
    ok &= _line("C8", regs[ARCH, 32] <= 124 and regs[ARCH, 32] <= regs[ARCH, 64] and regs["sm_80", 32] <= regs["sm_80", 64],
                f"index set regs {ARCH} {regs[ARCH, 64]} -> {regs[ARCH, 32]} (<= 124), sm_80 {regs['sm_80', 64]} -> {regs['sm_80', 32]} (no increase)")
    # A3: the corpus against NumPy's bits
    values = []
    for entry in W.CORPUS:
        for mode in W.CORPUS[entry].modes:
            got = W.run_device(entry, mode, W.LANES[entry], cache_dir=CACHE / "tree", repeats=1, want_dir=CACHE / "want")
            values.append(W.array_digest(got["out"]) == got["want_sha"])
    ok &= _line("A3-numpy", all(values), f"{len(values)} corpus values bit-identical to NumPy")
    # the derivative rows against central differences of their primals' NumPy references (float64, exact data)
    fd = {}
    for entry in ("fill_mask_jvp", "matvec_in_jvp", "fill_mask_vjp", "matvec_in_vjp"):
        n = W.LANES[entry]
        got = W.run_device(entry, "float64", n, cache_dir=CACHE / "tree", repeats=1)
        planes = W.CORPUS[entry].planes(np.random.default_rng(7), np.float64, n, True)
        fd[entry] = bool(np.allclose(got["out"], W.finite_difference(entry, planes, n), rtol=1e-12, atol=0))
    ok &= _line("A3-fd", all(fd.values()), "derivative rows against central finite differences of their primals: "
                + ", ".join(f"{k} {'ok' if v else 'WRONG'}" for k, v in fd.items()))
    # A5: a fresh process each
    deltas = {}
    for mode in ("float32", "float64"):
        out = _run_memory(sys.executable, mode, None)
        deltas[mode] = float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else float("inf")
    ok &= _line("A5", all(d <= 4.0 for d in deltas.values()),
                f"first launch of a trivial Vector[512] kernel, device MB: f32 {deltas['float32']:.1f}, f64 {deltas['float64']:.1f} (<= 4)")
    return 0 if ok else 1


def _within_rounding(entry: str, mode: str, out) -> bool:
    """Whether ``out``, a corpus row on rounding data, is within the summation bound ``2 W eps sum|terms|`` of the exact result
    (the planes taken in float64; the terms' magnitudes from the absolute planes, which bound a multilinear row)."""
    real = np.float32 if mode == "float32" else np.float64
    n = W.LANES[entry]
    planes = {k: np.asarray(v, np.float64) for k, v in W.CORPUS[entry].planes(np.random.default_rng(7), real, n, False).items()}
    want = W.CORPUS[entry].want(np.float64, planes, n)
    scale = W.CORPUS[entry].want(np.float64, {k: np.abs(v) for k, v in planes.items()}, n)
    return bool(np.all(np.abs(np.asarray(out, np.float64) - want) <= 2 * W.W * np.finfo(real).eps * scale))


def _compare(root: pathlib.Path) -> int:
    """The checks against the published 0.7.2 release (``TWIN``, its environment): A3-twin, SPEED, and the A5 control
    (the release does reserve the memory, or the instrument of A5 is blind)."""
    ok = True
    mine = _resources(root / "tree", CACHE / "tree")
    twin = _work(str(root / "twin"))
    done = _run_twin(HERE / "_wide_kernels.py", twin, "--twin", cache=CACHE / "twin")
    if done.returncode:
        print(done.stderr[-2000:])
        return 1 if not _line("A3", False, "the 0.7.2 environment did not run the corpus") else 1
    twin_times, twin_texts = json.loads((twin / "times.json").read_text()), json.loads((twin / "texts.json").read_text())
    identical, ratios, slower = [], {}, []
    for entry in W.CORPUS:
        for mode in W.CORPUS[entry].modes:
            for exact in (True, False):
                got = W.run_device(entry, mode, W.LANES[entry], cache_dir=CACHE / "tree", exact=exact, repeats=W.REPEATS.get(entry, 7) if exact else 1)
                same = W.array_digest(got["out"]) == (twin / f"{entry}_{mode}_{int(exact)}.sha").read_text()
                if not same and "hawk_wl" in mine[entry, mode]["text"]:        # a warp per lane sums in another order: held to the rounding bound
                    same = _within_rounding(entry, mode, got["out"])
                identical.append(same)
                key = f"{entry}_{mode}_{int(exact)}"
                if exact:
                    ratios[entry, mode] = twin_times[key] / got["ns"]
                    if mine[entry, mode]["text"] != twin_texts[f"{entry}_{mode}"] and ratios[entry, mode] < 0.98:
                        slower.append((entry, mode))
    ok &= _line("A3-twin", all(identical), f"{len(identical)} corpus values (exact and rounding data) bit-identical to the 0.7.2 wheel")
    ok &= _line("SPEED", not slower, "ns/sample 0.7.2 / this tree (a changed kernel must be >= 0.98x, the timing noise): " + ", ".join(
        f"{e}/{m_[-2:]} {r:.2f}x" for (e, m_), r in ratios.items()))
    control = {}
    for mode in ("float32", "float64"):
        out = _run_memory(str(TWIN / "bin" / "python"), mode, "twin")
        control[mode] = float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else float("inf")
    ok &= _line("A5-ctl", all(c > 4.0 for c in control.values()),
                f"the 0.7.2 wheel's first launch of the trivial kernel takes f32 {control['float32']:.1f}, f64 {control['float64']:.1f} MB (> 4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(gate(sys.argv[1] if len(sys.argv) > 1 else "gpu"))
