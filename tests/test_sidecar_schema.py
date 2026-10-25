# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk sidecar validates against raptor's ``sidecar-v1`` schema, as written.

The schema requires ``vec_widths`` (per-name vector width, so a host can
allocate ``(W, N)`` buffers) for the ``pure`` pattern; it holds the width of
every vector input and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hawk
from hawk import Grid, Scalar, Vector
from hawk.artifact.sidecar import sidecar_meta

jsonschema = pytest.importorskip("jsonschema")
raptor = pytest.importorskip("raptor")

_SCHEMA = Path(raptor.__file__).resolve().parent.parent / "docs/schemas/sidecar-v1.schema.json"
pytestmark = pytest.mark.skipif(not _SCHEMA.is_file(), reason="raptor's schema file is not installed")


@hawk.kernel
def _flat(x: Scalar, v: Vector[3]) -> Scalar:
    return x * v[0] * v[1] + v[2]


@hawk.kernel(group=8)
def _grid(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
    return x * 2.0


def _sidecar(kernel):
    return sidecar_meta(kernel.name, kernel.walk, fmt="host", scalar_type="float64",
                        lane_space=kernel.lane_space, group=kernel.group)


@pytest.mark.parametrize("kernel", [_flat, _grid], ids=["flat", "grid"])
def test_the_sidecar_satisfies_the_schema(kernel):
    schema = json.loads(_SCHEMA.read_text())
    jsonschema.validate(_sidecar(kernel), schema)


def test_vec_widths_names_the_vector_inputs_only():
    meta = _sidecar(_flat)
    assert meta["vec_widths"] == {"v": 3} == {n: meta["arg_widths"][n] for n in meta["vector_inputs"]}
    assert _sidecar(_grid)["vec_widths"] == {}
