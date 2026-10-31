# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where a per-lane value wider than a thread's registers lives (:mod:`hawk.ir.wide`).

Rows: what the pass places and what it leaves (the corpus of :mod:`_wide_kernels`), the sidecar record and
its scratch planes, the build option ``wide_form``, the host text that places nothing, the identity that keeps
a store under a condition one store (``select(c, v, set(v, i, x))``), the 32-bit integers of a lane set,
and, on a device, the corpus values against NumPy.

Run as a script, this file is the wide gate (``make gate-wide``): it builds and runs the corpus here and in the
0.7.2 wheel's environment and prints one PASS or FAIL line per check."""

import json
import os
import pathlib
import subprocess
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
from hawk.emit import policy
from hawk.ext import IndexSet, Kind
from hawk.ir import wide
from hawk.ir.nodes import Select

TARGETS_BOTH = ("cuda", "host")


def _unit(bundle, name, suffix=".cu"):
    return (pathlib.Path(bundle.directory) / f"{name}{suffix}").read_text()


#: The arch the committed local-memory figures (``W.LOCAL_OK``, the ``sm61`` class tables) were measured on.
MEASURED_ARCH = "sm_61"


def _a1_arch() -> str:
    """The arch the A1 rows compile for: the card's own when one is reachable (and then only if the committed figures are
    its class's), else the measured one, so a GPU-less run builds a CUBIN of the arch the figures describe (a virtual arch
    gives PTX, whose local sizes are a ceiling before ptxas)."""
    from hawk.compile import toolchain as tc

    probed = tc._probed_device_arch()
    if probed and probed != MEASURED_ARCH:
        pytest.skip(f"the card is {probed} (thresholds class {wide_class_of(probed)}): the committed A1 figures are for {MEASURED_ARCH}")
    return MEASURED_ARCH


def wide_class_of(arch: str) -> str:
    from hawk.ir import wide_class

    return wide_class.select(arch)


def _build(entry, tmp_path, cache_dir, **options):
    options.setdefault("device_arch", _a1_arch())
    targets = options.pop("targets", ("cuda",))
    kernel = W.CORPUS[entry].kernel
    kind = {} if W.CORPUS[entry].kind is None else {"kind": W.CORPUS[entry].kind}
    return kernel, build_bundle([kernel], tmp_path, targets=targets, cache_dir=cache_dir, **kind, **options)


# -- what the pass places -----------------------------------------------------------------------

def test_a_runtime_index_over_a_wide_value_places_it_in_a_scratch_plane_and_the_mask_lands_in_the_output_plane():
    _, planes, record = wide.place(W.fill_mask.sinks, real_bytes=4)
    assert [p.name for p in planes] == ["hwd0"] and planes[0].per_lane == W.W and planes[0].dtype == "real"
    assert record["hwd0"] == {"form": "resident", "plane": "hwd0", "layout": "warp_tiled", "width": W.W}
    assert record["bf"] == {"form": "resident", "plane": "bf", "layout": "plane_major", "width": W.W}   # a Mutable is its own plane


def test_a_row_of_products_read_by_a_rolled_sum_is_placed_and_the_accumulator_is_the_output_plane():
    _, planes, record = wide.place(W.matvec_in.sinks, real_bytes=8)
    assert [p.name for p in planes] == ["hwd0"] and "y" not in record       # the accumulator is written in a loop and read once: registers


def test_a_value_computed_once_is_placed_in_place_over_the_plane_it_was_filled_in():
    _, planes, record = wide.place(W.matvec_computed.sinks, real_bytes=8)
    assert sorted(record) == ["hwd0", "hwd1"]          # the fill and its mask share one plane, the products another


def test_a_dispatched_sum_over_a_vector_input_has_nothing_to_place():
    sinks, planes, record = wide.place(W.masked_sum.sinks, real_bytes=8)
    assert sinks is not None and not planes and not record


def test_a_narrow_vector_read_at_a_runtime_index_is_placed_where_the_resident_form_is_faster():
    for name in ("narrow8", "narrow16", "rotate16"):
        _, planes, record = wide.place(getattr(W, name).sinks, real_bytes=8)
        assert record and all(p.name.startswith("hwd") for p in planes), name
        assert any(v["layout"] == "plane_major" for v in record.values()), name      # the Mutable output is its own plane


def test_the_widths_registers_win_at_stay_in_registers():
    for name, real_bytes in (("narrow4", 8), ("narrow16", 4), ("rotate16", 4)):
        sinks, planes, record = wide.place(getattr(W, name).sinks, real_bytes=real_bytes)
        assert not planes and not record, name


def test_a_vector_whose_every_index_is_a_constant_keeps_its_registers():
    @hawk.kernel
    def narrow(tbl: Table["i":16], y: Mutable[Vector[16]]):
        y = W.mask_vec(tbl, 16) * 2.0

    sinks, planes, record = wide.place(narrow.sinks, real_bytes=8)
    assert sinks == tuple(narrow.sinks) and not planes and not record


def test_the_thread_form_places_nothing(tmp_path, cache_dir):
    with policy.using(wide_form="thread"):
        sinks, planes, record = wide.place(W.fill_mask.sinks, real_bytes=8)
    assert not planes and not record
    with pytest.warns(UserWarning, match="local frame"):     # the thread form spills, and the build says so
        _, bundle = _build("fill_mask", tmp_path, cache_dir, wide_form="thread")
    side = sidecar_of(bundle, "fill_mask")
    assert "wide" not in side["launch"] and "stages" not in side and "hwd" not in _unit(bundle, "fill_mask")


def test_the_team_form_is_refused_by_name(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"wide_form=.team. \(a warp per lane\) is built for device targets only"):
        build_bundle([W.fill_mask], tmp_path, targets=("host",), cache_dir=cache_dir, wide_form="team")


# -- what the build writes -----------------------------------------------------------------------

def test_the_sidecar_names_every_plane_and_the_runtime_makes_the_scratch_ones(tmp_path, cache_dir):
    _, bundle = _build("fill_mask", tmp_path, cache_dir, targets=TARGETS_BOTH, host_profile="native")     # a host target keeps the placed form
    side = sidecar_of(bundle, "fill_mask")
    assert side["launch"]["wide"] == {
        "bf": {"form": "resident", "plane": "bf", "layout": "plane_major", "width": W.W},
        "hwd0": {"form": "resident", "plane": "hwd0", "layout": "warp_tiled", "width": W.W},
        "class": "sm61"}
    assert side["team_planes"] == {"hwd0": {"dtype": "real", "per_lane": W.W}}
    assert [st["kernel"] for st in side["stages"]] == ["fill_mask"] and ("wide_in", "hwd0") in map(tuple, side["arg_spec"])
    assert side["launch"]["local_bytes"] == 0


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

def test_the_derivatives_of_a_wide_kernel_place_what_the_primal_places_and_the_scaled_reverse_reads():
    for name in ("fill_mask_jvp", "fill_mask_vjp", "matvec_in_jvp"):
        _, planes, record = wide.place(W.CORPUS[name].kernel.sinks, real_bytes=8)
        assert planes and record, name


def test_the_tangent_of_a_stepping_accumulation_lives_in_planes_across_the_trips():
    _, planes, record = wide.place(W.steps_acc_jvp_k8.sinks, real_bytes=8)
    assert {"bf", "dot_bf"} <= set(record)


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
    kernel, bundle = _build(entry, tmp_path, cache_dir, mode="float64")
    side = sidecar_of(bundle, kernel.name)
    assert side["launch"]["local_bytes"] == 0
    planes = {p for p in side["launch"]["wide"] if p != "class"}
    assert {p for p in side["team_planes"]} <= planes and len(side["team_planes"]) >= 2   # one set of scratch planes per body


def test_the_warp_form_slices_a_wide_accumulation_across_the_lanes_of_a_warp(tmp_path, cache_dir):
    from hawk.ir.lane import WARP, WarpStore, slice_sinks

    sliced = slice_sinks(wide.simplified(W.matvec_in_vjp.sinks))
    assert [type(s) for s in sliced] == [WarpStore] and sliced[0].value.ttype.shape == (W.W // WARP,)
    assert slice_sinks(W.wide51.sinks) is None               # a write at a run-time index has no rule in the warp form: it keeps its form
    for name in ("matvec_in_vjp", "matvec_in"):         # the second folds the slices with a warp sum and commits from lane 0
        kernel, bundle = _build(name, tmp_path / name, cache_dir, mode="float64", wide_form="team")
        side = sidecar_of(bundle, kernel.name)
        assert side["stages"][0]["team"] == {"threads": WARP}
        assert side["launch"]["local_bytes"] == (0 if name == "matvec_in_vjp" else 192)    # the 24 row sums stay in the lane's local frame


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["matvec_in", "matvec_in_vjp"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_the_warp_form_is_exact_and_replays_bit_for_bit_on_the_device(name, mode, cache_dir):
    import _probe_wide as probe
    from eagle import arrays
    from eagle._alloc import synchronize

    entry, real, n = W.CORPUS[name], np.float32 if mode == "float32" else np.float64, 2048
    planes = entry.planes(np.random.default_rng(7), real, n, True)
    bound_args = {k: arrays.asarray(np.ascontiguousarray(v), device="cuda:0") for k, v in planes.items()}
    out = arrays.asarray(np.zeros(entry.shape(n), real), device="cuda:0")
    bound = probe._auto(entry.kernel, mode, cache_dir, "team").bind(**bound_args, **{entry.out: out})
    got = []
    for _ in range(3):
        bound.launch()
        synchronize()
        got.append(np.asarray(out.get()).copy())
    assert np.array_equal(got[0], entry.want(real, planes, n))
    assert all(np.array_equal(got[0], g) for g in got[1:])


# -- the device feature classes -----------------------------------------------------------------

def test_a_known_arch_selects_its_own_class_and_an_unknown_feature_tuple_the_nearest(monkeypatch):
    from hawk.ir import wide_class as C

    assert C.select("sm_61") == "sm61" and C.select() == "sm61"
    volta = C.Tables(runtime_min={p: {4: 1, 8: 1} for p in C.CLASSES["sm61"][1].runtime_min}, wide_min={4: 64, 8: 32})
    ampere = C.Tables(runtime_min={p: {4: 2, 8: 2} for p in C.CLASSES["sm61"][1].runtime_min}, wide_min={4: 64, 8: 32})
    monkeypatch.setitem(C.CLASSES, "fake70", (C.ARCH_FEATURES[70], volta))
    monkeypatch.setitem(C.CLASSES, "fake90", (C.ARCH_FEATURES[90], ampere))
    assert C.select("sm_61") == "sm61" and C.select("sm_70") == "fake70"
    assert C.select("sm_75") == "sm61"                 # 1:32 and 1024 threads: one step from Pascal, a long way from Volta
    assert C.select("sm_80") == "fake70"               # 1:2 and 2048 threads, as Volta (a tie with Hopper goes to the lower name)
    assert C.select("sm_99") == "fake70"               # an arch nobody listed: the features of the nearest capability (90)
    assert C.select(features=C.Features(2, 65536, 2048, True)) == "fake70"


def test_the_tables_of_the_selected_class_decide_the_placement(monkeypatch):
    from hawk.emit import policy
    from hawk.ir import wide_class as C

    never = C.Tables(runtime_min={p: {4: 10 ** 6, 8: 10 ** 6} for p in C.CLASSES["sm61"][1].runtime_min}, wide_min={4: 10 ** 6, 8: 10 ** 6})
    monkeypatch.setitem(C.CLASSES, "keeps", (C.ARCH_FEATURES[80], never))
    with policy.wide_class_scope("keeps"):
        _, planes, record = wide.place(W.rotate16.sinks, real_bytes=8)
    assert not planes and not record
    _, planes, record = wide.place(W.rotate16.sinks, real_bytes=8)
    assert record


def test_the_sidecar_names_the_class_and_the_unit_key_changes_with_it(tmp_path, cache_dir):
    from hawk.artifact.unit_cache import _unit_key
    from hawk.emit import scalar_mode

    _, bundle = _build("fill_mask", tmp_path, cache_dir, device_arch="sm_75", targets=TARGETS_BOTH, host_profile="native")   # no class of its own: the nearest
    assert sidecar_of(bundle, "fill_mask")["launch"]["wide"]["class"] == "sm61"
    _, host = _build("masked_sum", tmp_path / "plain", cache_dir)
    assert "wide" not in sidecar_of(host, "masked_sum")["launch"]

    def key(cls):
        return _unit_key(["k"], ["d"], ["e"], targets=("cuda",), smode=scalar_mode("float64"), kind=None, defines=(), cache_dir=None,
                         device_arch="sm_61", derivative_blocks=(), layout_sizes_override=None, wide_class=cls)

    assert key("sm61") != key("sm75") and key("sm61") == key("sm61")


def test_the_text_of_the_one_measured_class_does_not_depend_on_the_arch(tmp_path, cache_dir):
    texts = {arch: _unit(_build("matvec_in", tmp_path / arch, cache_dir, device_arch=arch)[1], "matvec_in")
             for arch in ("sm_61", "sm_75")}
    assert texts["sm_61"] == texts["sm_75"]


# -- a stepping kernel ---------------------------------------------------------------------------

def test_a_step_loop_carries_a_wide_mutable_in_its_own_plane_and_places_what_it_reads():
    _, planes, record = wide.place(W.steps_fill_k8.sinks, real_bytes=8)
    assert [p.name for p in planes] == ["hwd0"]
    assert record["bf"] == {"form": "resident", "plane": "bf", "layout": "plane_major", "width": W.W}


def test_a_step_loop_writes_the_carried_plane_once_after_the_loop_from_a_shadow_of_the_last_trip():
    from hawk.ir.loop_nodes import Loop, LoopValue

    sinks, _, _ = wide.place(W.steps_fill_k8.sinks, real_bytes=8)
    step = next(l for l in wide._Graph(sinks).loops if l.exit_cond is not None)
    assert [c.name for c in step.carries].count("c_last") == 1               # the counter's value at the start of a trip
    assert not any(isinstance(n, Loop) or wide.is_resident(n) and n.ttype.shape for n in step.body_nodes())   # nothing wide in the body
    bf = next(s for s in sinks if getattr(s, "name", None) == "bf")
    assert wide.is_resident(bf.operands[0]) and isinstance(bf.operands[0].operands[0].operands[1], LoopValue)


def test_a_step_loop_runs_one_loop_the_finish_follows():
    from hawk.ir.loop_nodes import Loop

    sinks, _, _ = wide.place(W.steps_matvec_k8.sinks, real_bytes=8)
    loops = [n for n in wide._Graph(sinks).loops if isinstance(n, Loop) and n.exit_cond is not None]
    assert len(loops) == 1                          # the finish reads the placed loop's flag, not a second copy of the loop


@pytest.mark.parametrize("entry", ["steps_fill_k1", "steps_fill_k8", "steps_early_k8", "steps_matvec_k1", "steps_matvec_k8", "narrow4", "narrow16",
                                   "rotate16"])
@pytest.mark.parametrize("mode", ["float32", "float64"])
def test_a_stepping_or_narrow_corpus_kernel_holds_no_local_memory(entry, mode, tmp_path, cache_dir):
    _, bundle = _build(entry, tmp_path, cache_dir, mode=mode)
    assert sidecar_of(bundle, W.CORPUS[entry].kernel.name)["launch"]["local_bytes"] == W.LOCAL_OK.get((entry, mode), 0)


# -- the identity ------------------------------------------------------------------------------

def _vector_selects(kernel):
    from hawk.ir.loop_nodes import Loop
    from hawk.ir.walk import canonical_nodes

    found, nodes = [], list(canonical_nodes(kernel.sinks)[0])
    while nodes:
        n = nodes.pop()
        if isinstance(n, Loop):
            nodes.extend(n.body_nodes())
        elif isinstance(n, Select) and n.ttype.shape:
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


def test_the_thread_form_keeps_the_select_over_the_store_at_any_width():
    with policy.using(wide_form="thread"):
        @hawk.kernel
        def pinned(c: Vector[40], y: Mutable[Vector[40]]):
            r = W.zeros(40)
            for j in range(40):
                if c[j] > 0.5:
                    r[j] = 1.0
            y = r

    assert len(_vector_selects(pinned)) == 1


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


def _arch() -> str:
    """The device's own ``sm_<N>`` (the class of the thresholds and of every build the gate makes), never a fixed one."""
    from _toolchain import real_arch

    return real_arch()


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


def _run_memory(python: str, mode: str, twin) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if twin is None or k not in ("HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE", "PYTHONPATH")}
    env["HAWK_DEVICE_COMPILER"] = "nvcc"
    return subprocess.run([python, str(HERE / "_wide_kernels.py"), "--memory", mode], env=env, capture_output=True, text=True)


def _run_twin(script: pathlib.Path, root: pathlib.Path, *args, cache=None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE", "PYTHONPATH")}
    env["HAWK_DEVICE_COMPILER"] = "nvcc"
    env["HAWK_GATE_ARCH"] = ARCH
    return subprocess.run([str(TWIN / "bin" / "python"), str(script), *args, str(root), *([] if cache is None else [str(cache)])],
                          env=env, capture_output=True, text=True)


def gate(which: str = "gpu") -> int:
    """``gpu``: the correctness checks (A1, C8, A3-numpy, A3-fd, A5); ``compare``: the checks against the published 0.7.2
    release (A3-twin, A3-text, SPEED), which need ``HAWK_TWIN_VENV``."""
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
    """The checks against the published 0.7.2 release (``TWIN``, its environment): A3-twin, A3-text, SPEED, and the A5 control
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
    text_ok, compared, changed, bad = _text_diff(root)
    names = sorted({k.rsplit("/", 1)[0] for k in changed})
    ok &= _line("A3-text", text_ok, f"generated text of {compared} suite kernel builds, 0.7.2 vs this tree: {len(changed)} changed "
                f"({len(names)} kernels: {', '.join(names) or 'none'}), every one with a run-time indexed value" if text_ok
                else f"the generated text of a kernel with no run-time indexed value changed: {', '.join(bad)}")
    ok &= _line("SPEED", not slower, "ns/sample 0.7.2 / this tree (a changed kernel must be >= 0.98x, the timing noise): " + ", ".join(
        f"{e}/{m_[-2:]} {r:.2f}x" for (e, m_), r in ratios.items()))
    control = {}
    for mode in ("float32", "float64"):
        out = _run_memory(str(TWIN / "bin" / "python"), mode, "twin")
        control[mode] = float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else float("inf")
    ok &= _line("A5-ctl", all(c > 4.0 for c in control.values()),
                f"the 0.7.2 wheel's first launch of the trivial kernel takes f32 {control['float32']:.1f}, f64 {control['float64']:.1f} MB (> 4)")
    return 0 if ok else 1


def _text_diff(root: pathlib.Path) -> tuple:
    """``(ok, compared, changed, bad)``: the suite kernels whose generated text differs from the 0.7.2 wheel's (``changed``,
    ``kernel/mode``), and which of them have no run-time indexed value (``bad``: the pass may not touch those)."""
    mine = W.suite_texts()
    done = _run_twin(HERE / "_wide_kernels.py", root, "--texts")
    if done.returncode:
        print(done.stderr[-2000:])
        return False, len(mine), [], []
    theirs = json.loads((root / "texts_072.json").read_text())
    changed = sorted(k for k in mine if mine[k] != theirs.get(k))
    kernels = W._suite_kernels()
    bad = [k for k in changed if not W.runtime_indexed(kernels[k.rsplit("/", 1)[0]])]
    return not bad, len(mine), changed, bad


if __name__ == "__main__":
    raise SystemExit(gate(sys.argv[1] if len(sys.argv) > 1 else "gpu"))
