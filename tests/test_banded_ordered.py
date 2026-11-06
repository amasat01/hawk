# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ordered team stages in the ``banded`` scalar mode. Their partial sums live in a scratch plane of 16-byte
``band_partial`` elements (aether's ``BandPartial``), each stage folds with ``BandAccum`` in the same fixed order the
float64 stages add in, and a stored cell is encoded once: the bits do not depend on the run, and the fold is the
compensated one the banded sum promises. A float build keeps its ``real`` partials plane."""

from __future__ import annotations

import pytest
from conftest import sidecar_of

import hawk
from hawk.artifact import build_bundle

TARGETS = [pytest.param("cuda", marks=pytest.mark.gpu)]


@hawk.kernel
def ordered(lane: hawk.Grid["key", "col":64], x: hawk.Scalar) -> hawk.Scalar:
    return hawk.sum(x, over="col", policy="ordered")


@hawk.kernel
def add(key: hawk.Index, x: hawk.Scalar, acc: hawk.Mutable[hawk.Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


@hawk.kernel
def scattered(key: hawk.Index, x: hawk.Scalar, acc: hawk.Mutable[hawk.Scalar]):
    hawk.scatter(acc)[key] += x


@hawk.kernel
def colsum(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x, over="col")


@hawk.kernel
def colsum_compensated(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x, over="col", policy="compensated")


def _partial_dtypes(bundle, name):
    return {p["dtype"] for p in (sidecar_of(bundle, name).get("team_planes") or {}).values() if p["dtype"] != "int32"}


@pytest.mark.parametrize("target", ["cuda"])
@pytest.mark.parametrize("kernel", (ordered, add, scattered, colsum, colsum_compensated), ids=lambda k: k.name)
def test_an_ordered_or_default_policy_reduction_builds_banded_on_the_ordered_stages_with_16_byte_partials(kernel, target, tmp_path):
    banded = build_bundle([kernel], tmp_path / "banded", mode="banded", targets=(target,), host_profile="native")
    assert "band_partial" in _partial_dtypes(banded, kernel.name)
    f64 = build_bundle([kernel], tmp_path / "f64", mode="float64", targets=(target,), host_profile="native")
    assert "band_partial" not in _partial_dtypes(f64, kernel.name)


@hawk.kernel
def colsum_plane(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x, over="col", policy="ordered")


@hawk.kernel
def colsum_squares(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x * x, over="col", policy="ordered")


@pytest.mark.parametrize("target", ["cuda"])
@pytest.mark.parametrize("mode", ("float64", "float32", "banded"))
def test_a_column_sum_of_a_plane_read_has_no_copy_in_team_scratch_and_a_computed_input_keeps_it(mode, target, tmp_path):
    """A reduction input that is a bare read of an input plane is read in place by the block stage: no ``_u`` scratch plane and no
    copy stage. A computed input is stored to its scratch plane first."""
    planes = {}
    for kernel in (colsum_plane, colsum_squares):
        bundle = build_bundle([kernel], tmp_path / f"{mode}_{kernel.name}", mode=mode, targets=(target,), host_profile="native")
        side = sidecar_of(bundle, kernel.name)
        planes[kernel.name] = (sorted(p for p in (side.get("team_planes") or {}) if p.endswith("_u")), len(side["stages"]))
    assert planes["colsum_plane"] == ([], 2)
    assert planes["colsum_squares"][0] and planes["colsum_squares"][1] == 3


def _exact(x):
    from fractions import Fraction

    return sum(Fraction(float(v)) for v in x)


def _cancelling(n):
    import numpy as np

    return np.tile([2.0 ** 30, 1.0, -(2.0 ** 30), 1.0], n // 4 + 1)[:n].astype(np.float32).astype(np.float64)


def _uniform(n):
    import numpy as np

    return np.random.default_rng(n).uniform(-2.0, 2.0, n).astype(np.float32).astype(np.float64)


def _total(policy, n):
    @hawk.kernel
    def total(lane: hawk.Grid["key", "sample":n], x: hawk.Scalar) -> hawk.Scalar:
        return hawk.sum(x, over="sample", policy=policy)
    return total


def _launch_cells(kernel, target, tmp_path, planes, out, runs=3):
    """The cells of plane ``out`` after ``runs`` fresh launches of ``kernel`` through eagle's staged plan."""
    import numpy as np

    import eagle
    import eagle.exec as eexec
    from eagle import arrays
    from eagle import plan as eplan
    from eagle import _split
    from hawk.artifact import plugins

    bundle = build_bundle([kernel], tmp_path / target, mode="banded", targets=(target,), host_profile="native")
    by = plugins(bundle, device_loader=eagle.registry.load_manifest if target == "cuda" else None)
    _split.link(by, bundle)
    plan = eplan.plan(by[kernel.name], structure=eexec.DeviceKernel if target == "cuda" else eexec.HostTeam)
    cells = []
    for _ in range(runs):
        bound = {k: eagle.banded_pack(arrays.asarray(v)) if v.dtype.kind == "f" else arrays.asarray(v)
                 for k, v in {**planes, out: np.zeros_like(planes[out])}.items()}
        if target == "cuda":
            bound = {k: arrays.asarray(v, device="cuda:0") for k, v in bound.items()}
        plan.bind(**bound).launch()
        if target == "cuda":
            from eagle._alloc import synchronize

            synchronize()
        got = bound[out].get() if target == "cuda" else bound[out]
        cells.append(np.asarray(got).view(np.uint64).copy())
        last = got
    return cells, last


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("policy", ("auto", "compensated", "ordered"))
@pytest.mark.parametrize("data", (_cancelling, _uniform), ids=("cancelling", "uniform"))
@pytest.mark.parametrize("n", (1000, 3 * 65536 + 777))
def test_a_banded_column_sum_is_the_same_cell_every_run_and_within_half_an_ulp_of_the_exact_sum(
        n, data, policy, target, tmp_path):
    """The delivered cell (the carrier's figure is pinned in aether) on the cancel-2^-30 and the uniform class."""
    import numpy as np
    from fractions import Fraction

    import eagle
    from eagle import arrays

    x = data(n)
    cells, last = _launch_cells(_total(policy, n), target, tmp_path, {"x": x, "out": np.zeros(1)}, "out")
    assert cells[0].tobytes() == cells[1].tobytes() == cells[2].tobytes()
    got = float(np.asarray(eagle.banded_unpack(last))[0])
    want = _exact(x)
    ulp = abs(Fraction(got) - want) / Fraction(float(np.spacing(abs(float(want)))))
    assert ulp <= 0.5, float(ulp)


@pytest.mark.gpu
def test_a_banded_warp_sum_is_the_same_cells_every_run_and_close_to_the_exact_dot_product(cache_dir):
    """A wide dot product in the warp form sums its lanes' slices with ``warp_sum``: the accumulators fan in through
    eagle's ``shflXor`` and every lane ends with the same bits, run after run."""
    import numpy as np
    from fractions import Fraction

    import _wide_kernels as W
    import eagle
    from eagle import arrays
    from eagle._alloc import synchronize
    from hawk.targets.cards import _probe_wide as probe

    n = 64
    rng = np.random.default_rng(11)
    f32 = lambda a: a.astype(np.float32).astype(np.float64)       # noqa: E731
    x, w = f32(rng.uniform(0.5, 2.0, (W.W, n))), f32(rng.uniform(0.5, 2.0, (W.ROWS, W.W)))
    plan = probe._auto(W.matvec_in, "banded", cache_dir, "team")
    runs = []
    for _ in range(3):
        planes = {k: arrays.asarray(eagle.banded_pack(arrays.asarray(v)), device="cuda:0")
                  for k, v in {"x": x, "w": w, "y": np.zeros((W.ROWS, n))}.items()}
        plan.bind(**planes).launch()
        synchronize()
        held = planes["y"].get()
        runs.append(np.asarray(held).view(np.uint64).copy())
    assert runs[0].tobytes() == runs[1].tobytes() == runs[2].tobytes()
    got = np.asarray(eagle.banded_unpack(held))
    worst = 0.0
    for r in range(W.ROWS):
        for lane in range(0, n, 9):
            want = sum(Fraction(float(x[i, lane])) * Fraction(float(w[r, i])) for i in range(W.W))
            err = float(abs(Fraction(float(got[r, lane])) - want) / Fraction(float(np.spacing(float(want)))))
            if err > worst:
                worst, where = err, (r, lane, float(got[r, lane]), float(want))
    print("warp_sum worst ULP53", round(worst, 4))
    assert worst <= 3.0, (worst, where)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("kernel", (scattered, add), ids=("default", "ordered"))
@pytest.mark.parametrize("data", (_cancelling, _uniform), ids=("cancelling", "uniform"))
@pytest.mark.parametrize("cells", (7, 1000))
def test_a_banded_scatter_is_the_same_cells_every_run_and_within_half_an_ulp_per_cell(cells, data, kernel, target, tmp_path):
    """A keyed accumulate (default policy, routed to the ordered stages, and ``policy="ordered"``) over the
    cancel-2^-30 and the uniform class: each cell within 0.5 ULP53 of its exact sum, three runs the same cells."""
    import numpy as np
    from fractions import Fraction

    import eagle
    from eagle import arrays

    n = 30000
    key = (np.random.default_rng(cells).integers(0, cells, n)).astype(np.int32)
    key[::7] = -1
    x = data(n)
    got_cells, last = _launch_cells(kernel, target, tmp_path, {"key": key, "x": x, "acc": np.zeros(cells)}, "acc")
    assert got_cells[0].tobytes() == got_cells[1].tobytes() == got_cells[2].tobytes()
    got = np.asarray(eagle.banded_unpack(last))
    worst = 0.0
    for c in range(cells):
        want = _exact(x[key == c])
        if want == 0:
            assert got[c] == 0
            continue
        worst = max(worst, float(abs(Fraction(float(got[c])) - want) / Fraction(float(np.spacing(abs(float(want)))))))
    assert worst <= 0.5, worst


@pytest.mark.parametrize("kernel", (colsum, colsum_compensated), ids=lambda k: k.name)
def test_a_banded_default_column_sum_builds_the_block_stages_and_not_the_scatter_log(kernel, tmp_path):
    """A sum over lane axes is a block sum in a banded build, whatever its policy; a keyed scatter keeps the log."""
    steps = lambda k: {st["team"]["grid"] for st in sidecar_of(  # noqa: E731
        build_bundle([k], tmp_path / f"{k.name}", mode="banded", targets=("cuda",), host_profile="native"), k.name)["stages"]
        if st.get("team")}
    assert steps(kernel) == {"keys"}
    assert "chunks" in steps(scattered)


def test_the_banded_fold_only_loops_are_unrolled_by_the_accumulator_period_and_a_float_build_is_not():
    """The block stage's per-lane loop and the ordered fold's row loop (each one ``addTerm`` into one accumulator) carry the
    unroll that makes the renormalise gate a constant; the keyed bin fold does not, and a float rendering has none."""
    from types import SimpleNamespace

    from hawk.emit.team import team_body
    from hawk.ir.split import stages
    from hawk.ir.team import fold_spec

    pragma = "#pragma unroll aether::banded::BandAccum::period"
    part = stages("total", _total("ordered", 100000).sinks, wide=True, real_bytes=8, inplace=True)[0].team
    fold = fold_spec(SimpleNamespace(target="g", width=3), None)
    for spec, loop in ((part, "hawk_s.addTerm"), (fold, "hawk_s.addTerm")):
        text = team_body(spec, None, True).text.splitlines()
        at = [i for i, line in enumerate(text) if pragma in line]
        assert len(at) == 1 and text[at[0] + 1].lstrip().startswith("for (") and loop in text[at[0] + 2]
        assert "#pragma unroll" not in team_body(spec, None, False).text
