# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A value reduced over samples and read later is a launch boundary.

* **two stages** — ``z = x / hawk.sum(x)`` is two kernels; ``hawk.run``
  runs them with the host finish between, matches numpy, and is bitwise the
  same stages run through eagle's ``ordered`` finish (skipped without eagle);
* **three stages** — a chain whose reduction reads an earlier cell;
* **derivative** — the vjp is taken on the unsplit kernel and then split; its
  gradient matches central differences;
* **one stage** — a kernel that reads no reduction is one kernel, as before;
* **cache** — each stage is its own compile: a change after the cut
  recompiles only the later stage;
* **sidecar** — ``stages`` names every stage's cells and recomputed values.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import hawk
import hawk.diff
import hawk.math
from hawk import HawkError, Kernel, Mutable, Param, Scalar, Vector, kernel, runtime
from hawk.artifact import build, build_bundle
from hawk.compile import cache as compile_cache
from hawk.ir.split import stages

SIZES = (1, 31, 33, 1000, 65536)
RUN = 4096


@kernel
def share(x: Scalar, z: Mutable[Scalar]):
    z = x / hawk.sum(x)


@kernel
def standard(x: Scalar, mu: Mutable[Scalar], z: Mutable[Scalar], t: Mutable[Scalar]):
    mu = hawk.mean(x)
    v = hawk.mean((x - mu) * (x - mu))
    z = (x - mu) / hawk.math.sqrt(v)
    t = hawk.sum(z * z)


@kernel
def wave(x: Scalar, k: Param, z: Mutable[Scalar]):
    y = hawk.math.sin(k * x)
    z = y / hawk.sum(y * y)


@kernel
def centred(x: Scalar, z: Mutable[Vector[3]]):
    v = hawk.math.vec(x, x * x, 2.0 * x)
    z = v - hawk.mean(v)


@kernel
def scaled_two(x: Scalar, z: Mutable[Scalar]):
    z = 2.0 * x / hawk.sum(x)


@kernel
def scaled_three(x: Scalar, z: Mutable[Scalar]):
    z = 3.0 * x / hawk.sum(x)


@kernel
def rescale(y: Mutable[Scalar], z: Mutable[Scalar]):
    z = y / hawk.sum(y)
    y = 2.0 * y


@kernel
def stored_vec(x: Scalar, tot: Mutable[Vector[3]], z: Mutable[Vector[3]]):
    v = hawk.math.vec(x, x * x, 2.0 * x)
    tot = hawk.sum(v)
    z = v / hawk.sum(v)


@kernel
def split_keyed(x: Scalar, z: Mutable[Scalar], acc: Mutable[Scalar]):
    j, _ = hawk.math.split_index(RUN)
    hawk.scatter(acc)[j] += x
    z = x / hawk.sum(x)


@kernel
def plain(x: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x * x)


def _bundle(kernels, tmp, mode, cache_dir):
    return build_bundle(kernels, tmp, mode=mode, targets=("host",),
                        cache_dir=cache_dir, host_profile="native")


def _x(n, mode, lo=0.5):
    return np.random.default_rng(n).uniform(lo, 2.0, n).astype(mode)


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    mode = request.param
    out = {k.name: _bundle([k], tmp_path_factory.mktemp(f"split_{k.name}_{mode}"), mode,
                           cache_dir)
           for k in (share, standard, centred)}
    return out, mode


def _tol(mode):
    return 1e-12 if mode == "float64" else 2e-5


# -- two stages -------------------------------------------------------------

def test_a_read_reduction_is_two_stages():
    found = stages("share", share.sinks)
    assert [s.name for s in found] == ["share", "share_2"]
    assert found[0].cells_out == ("cut0",) and found[0].cells_in == ()
    assert found[1].cells_in == ("cut0",) and found[1].cells_out == ()


@pytest.mark.parametrize("n", SIZES)
def test_two_stages_through_hawk_run_match_numpy(built, n):
    bundles, mode = built
    x = _x(n, mode)
    z = np.full(n, np.nan, dtype=mode)
    runtime.run(runtime.load(bundles["share"].directory, "share"), x=x, z=z)
    want = x.astype(np.float64) / x.astype(np.float64).sum()
    np.testing.assert_allclose(z, want, rtol=_tol(mode), atol=0)


def _eagle_stages(bundle, name, mode, **inputs):
    """The stages run one by one through eagle's plan, each cell finished by
    eagle's ``ordered`` tree and bound to the next stage as a one-cell plane."""
    eexec = pytest.importorskip("eagle.exec")
    from eagle import plan as eplan

    import _deploy as L
    from conftest import sidecar_of

    first = sidecar_of(bundle, name)
    cells, outputs = {}, {}
    for record in first["stages"]:
        sidecar = dict(sidecar_of(bundle, record["kernel"]))
        sidecar["cells"] = {c: first["cells"][c] for c in record["cells_out"]}
        sidecar.pop("stages", None)
        if not sidecar["cells"]:
            sidecar.pop("cells")
        plugin = L.host_plugin(bundle.directory, record["kernel"], sidecar)
        plugin._meta = sidecar
        names = {n for _, n in sidecar["arg_spec"]}
        args = {k: v for k, v in inputs.items() if k in names}
        args.update({c: cells[c] for c in record["cells_in"]})
        got = eplan.plan(plugin, structure=eexec.HostTeam).run(**args)
        written = [n for r, n in sidecar["arg_spec"] if r in ("mutable", "out")]
        got = got if isinstance(got, dict) else {(record["cells_out"] or written)[0]: got}
        for c in record["cells_out"]:
            cells[c] = np.asarray(got[c], dtype=mode).reshape(-1)
        outputs.update({k: np.asarray(v) for k, v in got.items()})
    return outputs


@pytest.mark.parametrize("n", SIZES)
def test_two_stages_through_hawk_run_are_eagles_ordered_bits(built, n):
    bundles, mode = built
    x = _x(n, mode)
    z = np.full(n, np.nan, dtype=mode)
    runtime.run(runtime.load(bundles["share"].directory, "share"), x=x, z=z)
    want = _eagle_stages(bundles["share"], "share", mode, x=x)["z"]
    assert z.tobytes() == want.astype(mode).tobytes()


@pytest.mark.parametrize("n", (1, 33, 1000))
def test_a_stored_and_read_vector_sum_reuses_its_cell(tmp_path, cache_dir, n):
    found = stages("stored_vec", stored_vec.sinks)
    assert [c for st in found for c in st.cells_out] == ["tot"]
    bundle = _bundle([stored_vec], tmp_path, "float64", cache_dir)
    side = json.loads((bundle.directory / "stored_vec.json").read_text())
    assert set(side["cells"]) == {"tot"}
    x = _x(n, "float64")
    tot, z = np.zeros(3), np.full((3, n), np.nan)
    runtime.run(runtime.load(bundle.directory, "stored_vec"), x=x, tot=tot, z=z)
    v = np.stack([x, x * x, 2.0 * x])
    np.testing.assert_allclose(tot, v.sum(axis=1), rtol=1e-12, atol=0)
    np.testing.assert_allclose(z, v / v.sum(axis=1, keepdims=True), rtol=1e-12, atol=0)
    eplan = pytest.importorskip("eagle.plan")
    want = eplan.auto(stored_vec).run(x=x)
    assert z.tobytes() == np.asarray(want["z"], dtype=np.float64).reshape(z.shape).tobytes()
    columns = np.concatenate([np.asarray(want[f"tot_{k}"], dtype=np.float64).reshape(-1)
                              for k in range(3)])
    assert tot.tobytes() == columns.tobytes()


# -- three stages -----------------------------------------------------------

@pytest.mark.parametrize("n", (1000, 65536))
def test_a_three_cut_chain_is_three_stages_and_matches_numpy(built, n):
    bundles, mode = built
    sidecar = json.loads((bundles["standard"].directory / "standard.json").read_text())
    assert [s["kernel"] for s in sidecar["stages"]] == ["standard", "standard_2",
                                                        "standard_3"]
    assert sorted(sidecar["cells"]) == ["cut0", "mu", "t"]
    x = _x(n, mode)
    mu, t = np.zeros(1, dtype=mode), np.zeros(1, dtype=mode)
    z = np.full(n, np.nan, dtype=mode)
    runtime.run(runtime.load(bundles["standard"].directory, "standard"), x=x, mu=mu, z=z, t=t)
    xd = x.astype(np.float64)
    tol = 1e-10 if mode == "float64" else 1e-3
    np.testing.assert_allclose(mu[0], xd.mean(), rtol=tol)
    np.testing.assert_allclose(z, (xd - xd.mean()) / xd.std(), rtol=tol, atol=tol)
    np.testing.assert_allclose(t[0], n, rtol=tol)


def test_a_vector_reduction_is_read_back_column_by_column(built):
    bundles, mode = built
    n = 1000
    x = _x(n, mode)
    z = np.full((3, n), np.nan, dtype=mode)
    runtime.run(runtime.load(bundles["centred"].directory, "centred"), x=x, z=z)
    v = np.stack([x, x * x, 2.0 * x]).astype(np.float64)
    np.testing.assert_allclose(z, v - v.mean(axis=1, keepdims=True), rtol=1e-3, atol=_tol(mode) * 10)


def test_a_store_waits_for_the_last_stage_reading_the_planes_earlier_value(
        tmp_path, cache_dir):
    found = stages("rescale", rescale.sinks)
    assert [[getattr(s, "name", None) for s in st.sinks] for st in found] == [
        ["cut0"], ["y", "z"]]
    bundle = _bundle([rescale], tmp_path, "float64", cache_dir)
    y0 = _x(100, "float64")
    y, z = y0.copy(), np.zeros(100)
    runtime.run(runtime.load(bundle.directory, "rescale"), y=y, z=z)
    np.testing.assert_allclose(z, y0 / y0.sum(), rtol=1e-12)
    np.testing.assert_array_equal(y, 2.0 * y0)


def test_a_split_kernel_runs_every_sample(built):
    bundles, mode = built
    kernel_ = runtime.load(bundles["share"].directory, "share")
    with pytest.raises(HawkError, match="every sample"):
        runtime.run(kernel_, x=_x(64, mode), z=np.zeros(64, dtype=mode), count=32)


def test_build_names_build_bundle_for_a_split_kernel(tmp_path, cache_dir):
    with pytest.raises(HawkError, match="build_bundle"):
        build(share, tmp_path, targets=("host",), cache_dir=cache_dir)


def test_a_cut_inside_a_for_is_still_refused():
    with pytest.raises(HawkError, match="a collective.*inside a branch or a `for`"):
        @kernel
        def looped(x: Scalar, z: Mutable[Scalar]):
            acc = 0.0
            for _ in range(2):
                acc = acc + hawk.sum(x)
            z = x / acc


def test_a_max_over_samples_read_in_the_body_has_no_derivative():
    @kernel
    def peak(x: Scalar, z: Mutable[Scalar]):
        z = x / hawk.max(x)
    with pytest.raises(HawkError, match="max over samples"):
        hawk.diff.vjp(peak)


# -- derivative -------------------------------------------------------------

@pytest.mark.parametrize("promoted", [False, True], ids=["param", "param_plane"])
def test_the_vjp_is_split_and_matches_central_differences(tmp_path, cache_dir, promoted):
    primal = hawk.diff.param_planes(wave) if promoted else wave
    fwd = _bundle([wave], tmp_path / "f", "float64", cache_dir)
    grad = Kernel(primal.name + "_vjp", hawk.diff.vjp(primal, primal_unit=fwd.digest))
    assert len(stages(grad.name, grad.sinks)) == 3
    bundle = _bundle([grad], tmp_path / "g", "float64", cache_dir)
    n, k = 1000, 0.7
    x = _x(n, "float64", lo=-2.0)
    w = np.random.default_rng(7).uniform(-1.0, 1.0, n)

    def loss(kv):
        z = np.zeros(n)
        runtime.run(runtime.load(fwd.directory, "wave"), x=x, k=kv, z=z)
        return float(w @ z)

    bar_k = np.zeros(1) if promoted else np.zeros(n)
    runtime.run(runtime.load(bundle.directory, grad.name), x=x, bar_z=w, bar_x=np.zeros(n),
                bar_k=bar_k, k=np.array([k]) if promoted else k)
    h = 1e-6
    fd = (loss(k + h) - loss(k - h)) / (2 * h)
    np.testing.assert_allclose(bar_k.sum(), fd, rtol=1e-6)


# -- one stage --------------------------------------------------------------

def test_a_kernel_reading_no_reduction_is_one_unchanged_kernel(tmp_path, cache_dir):
    assert stages("plain", plain.sinks) is None
    bundle = _bundle([plain], tmp_path, "float64", cache_dir)
    assert [a.name for a in bundle.artifacts] == ["plain"]
    assert "stages" not in bundle.artifacts[0].sidecar


# -- cache ------------------------------------------------------------------

def test_a_change_after_the_cut_recompiles_only_the_later_stage(tmp_path):
    cache = str(tmp_path / "cache")
    _bundle([Kernel("scaled", scaled_two.sinks)], tmp_path / "a", "float64", cache)
    compile_cache.reset_cache_stats()
    bundle = _bundle([Kernel("scaled", scaled_three.sinks)], tmp_path / "b", "float64",
                     cache)
    assert compile_cache.cache_stats() == {"hits": 1, "misses": 1}
    assert [a.name for a in bundle.artifacts] == ["scaled", "scaled_2"]


# -- sidecar ----------------------------------------------------------------

def test_the_first_stage_names_every_stage_and_cell(built):
    bundles, _ = built
    arts = {a.name: a.sidecar for a in bundles["standard"].artifacts}
    assert list(arts) == ["standard", "standard_2", "standard_3"]
    first = arts["standard"]
    assert first["schema_version"] == 3
    assert first["stages"] == [
        {"kernel": "standard", "cells_in": [], "cells_out": ["mu"], "recomputed": []},
        {"kernel": "standard_2", "cells_in": ["mu"], "cells_out": ["cut0"],
         "recomputed": []},
        {"kernel": "standard_3", "cells_in": ["mu", "cut0"], "cells_out": ["t"],
         "recomputed": ["sub"]},
    ]
    for later in ("standard_2", "standard_3"):
        assert "stages" not in arts[later]
        assert ("lookup", "mu") in {tuple(p) for p in arts[later]["arg_spec"]}
    assert arts["standard_2"]["param_planes"] == ["mu"]
    assert arts["standard_3"]["cells"] == {"t": {"op": "sum", "columns": 1}}
    # a cell only a later stage reads is the kernel's own; a stored one is the caller's
    assert first["cells"] == {"mu": {"op": "sum", "columns": 1},
                              "cut0": {"op": "sum", "columns": 1, "internal": True},
                              "t": {"op": "sum", "columns": 1}}


# -- a keyed compensated sum inside a split kernel ---------------------------

@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_a_keyed_compensated_sum_in_a_split_kernel_is_folded(tmp_path, cache_dir, mode,
                                                             monkeypatch):
    """The runtime owns the companion and folds it after the stage that
    writes it: the target alone is within eps * sum|w| (and one ulp) of ``math.fsum``."""
    # the scatter sits in stage 1 only; the bundle's one-axis check is not under test
    from hawk.artifact import bundle as bundles
    manifest = bundles._manifest_of

    def one_axis(drafts):
        for d in drafts:
            d.sidecar["exec_access"] = "cross_sample_write"
        return manifest(drafts)

    monkeypatch.setattr(bundles, "_manifest_of", one_axis)
    bundle = _bundle([split_keyed], tmp_path, mode, cache_dir)
    n = 1 << 16
    rng = np.random.default_rng(7)
    x = (rng.uniform(-1, 1, n) * np.exp(rng.uniform(0, 8, n))).astype(mode)
    acc = np.zeros(n // RUN, dtype=mode)
    z = np.zeros(n, dtype=mode)
    runtime.run(runtime.load(bundle.directory, "split_keyed"), x=x, z=z, acc=acc)
    terms = x.astype(np.float64).reshape(-1, RUN)
    want = np.array([math.fsum(t) for t in terms])
    err = np.abs(acc.astype(np.float64) - want)
    assert np.all(err <= np.finfo(mode).eps * np.abs(terms).sum(axis=1))
    assert np.all(err <= np.abs(np.spacing(want.astype(mode))).astype(np.float64))
