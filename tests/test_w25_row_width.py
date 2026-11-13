"""The row width and the scatter chunk cap are variant facts (``cells.row``, ``scatter.max_chunks``), read where they are used."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

import hawk
from hawk import HawkError, Kernel, Mutable, Scalar, kernel, runtime
from hawk.artifact import build_bundle
from hawk.variants import derive, load

SIZES = (1, 31, 32, 33, 1000)


@kernel
def total(x: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x * x)


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    return build_bundle([Kernel("total", total.sinks)], tmp_path_factory.mktemp("w25_row"), mode="float64", targets=("host",),
                        cache_dir=cache_dir, host_profile="native")


@pytest.mark.parametrize("n", SIZES)
def test_a_host_reduction_equals_numpy_with_the_row_width_from_the_variant(built, n):
    kernel_ = runtime.load(built.directory, "total")
    row = load("cpu")["cells"]["row"]
    assert kernel_.sidecar["cell_row"] == row
    x = np.random.default_rng(n).uniform(-2.0, 2.0, n)
    t = np.full(1, np.nan)
    runtime.run(kernel_, x=x, t=t)
    np.testing.assert_allclose(t[0], (x * x).sum(), rtol=1e-14)
    scratch = kernel_._cell_scratch[n] if hasattr(kernel_, "_cell_scratch") else kernel_.__dict__["_cell_scratches"][n]
    assert len(next(iter(scratch.rows.values()))) == -(-n // row)


def test_the_simt_row_is_one_warp_and_the_cpu_row_a_power_of_two():
    for name in ("simt", "cuda"):
        v = load(name)
        assert v["cells"]["row"] == v["launch"]["warp_size"]
        assert v["scatter"]["max_chunks"] >= 1
    row = load("cpu")["cells"]["row"]
    assert row >= 1 and row & (row - 1) == 0 and load("cpu")["scatter"]["max_chunks"] >= 1


def test_the_text_chunk_cap_is_the_variants_and_a_missing_one_is_refused():
    from aether.wings.base import team_emit
    from lowering import team

    from raptor.errors import RaptorError

    cap = load("cpu")["scatter"]["max_chunks"]
    assert cap == load("simt")["scatter"]["max_chunks"] and not hasattr(team_emit, "_MAX_CHUNKS")
    for n in (cap, 7):
        text = "\n".join(team_emit._chunk_count(team.scatter_specs("t", 2, n, 32)[0].steps[0]))
        assert f"hawk_fit > {n} ? {n} : hawk_fit" in text
    with pytest.raises(RaptorError, match="max_chunks"):
        team_emit._chunk_count(team.scatter_specs("t", 2, 0, 32)[0].steps[0])


def test_a_variant_without_cells_row_is_refused_by_name_at_the_reader():
    from hawk.targets import build_spec

    spec = build_spec(("host",))
    bare = types.SimpleNamespace(**{k: getattr(spec, k) for k in spec.__dataclass_fields__})
    bare = type(spec)(**{**bare.__dict__, "host": {k: v for k, v in load("cpu").items() if k != "cells"}, "device": None, "focus": None})
    with pytest.raises(HawkError, match="no cells.row"):
        bare.need("cells", "row")
    derived = derive(load("cpu"), {"cells": None})
    assert derived["cells"] is None


def test_stages_of_one_run_with_two_row_widths_are_refused_by_name():
    stages = [types.SimpleNamespace(path=types.SimpleNamespace(stem=k), sidecar={"cell_row": w}) for k, w in (("a", 32), ("b", 16))]
    with pytest.raises(HawkError, match="one row width.*a: 32.*b: 16"):
        runtime._one_row_width(stages)


def test_a_sidecar_without_cell_row_is_refused_by_name():
    from hawk import _cells

    with pytest.raises(HawkError, match="no cell_row"):
        _cells.row_of({})


def test_eagle_refuses_two_row_widths_and_a_missing_one_by_name(built):
    _cell_rows = pytest.importorskip("eagle._cell_rows")
    from conftest import sidecar_of

    side = sidecar_of(built, "total")
    one = types.SimpleNamespace(kernel="total", _meta=side, _cells={"t": "sum"})
    other = types.SimpleNamespace(kernel="wide", _meta={**side, "cell_row": side["cell_row"] // 2}, _cells={"t": "sum"})
    assert _cell_rows.one_row_width([one]) == side["cell_row"]
    with pytest.raises(ValueError, match=rf"one row width.*{side['cell_row']}.*{side['cell_row'] // 2}"):
        _cell_rows.one_row_width([one, other])
    with pytest.raises(ValueError, match="no cell_row"):
        _cell_rows.row_of(types.SimpleNamespace(kernel="old", _meta={}))
