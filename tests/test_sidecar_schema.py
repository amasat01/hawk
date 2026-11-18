# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk sidecar keeps its vector-width record and, where the schema file is at hand, validates against it.

The schema is a data file of the consumer (``$HAWK_SIDECAR_SCHEMA`` names it); no Python package is imported for it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _unit_sidecar import sidecar_of_kernel

import hawk
from hawk import Grid, Scalar, Vector

pytest.importorskip("_osprey")


@hawk.kernel
def _flat(x: Scalar, v: Vector[3]) -> Scalar:
    return x * v[0] * v[1] + v[2]


@hawk.kernel(group=8)
def _grid(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
    return x * 2.0


def _sidecar(kernel):
    return sidecar_of_kernel(kernel, fmt="host", scalar_type="float64")


@pytest.mark.parametrize("kernel", [_flat, _grid], ids=["flat", "grid"])
def test_the_sidecar_satisfies_the_schema(kernel):
    jsonschema = pytest.importorskip("jsonschema")
    schema = os.environ.get("HAWK_SIDECAR_SCHEMA", "")
    if not Path(schema).is_file():
        pytest.skip("$HAWK_SIDECAR_SCHEMA does not name the sidecar schema file")
    jsonschema.validate(_sidecar(kernel), json.loads(Path(schema).read_text()))


def test_vec_widths_names_the_vector_inputs_only():
    meta = _sidecar(_flat)
    assert meta["vec_widths"] == {"v": 3} == {n: meta["arg_widths"][n] for n in meta["vector_inputs"]}
    assert _sidecar(_grid)["vec_widths"] == {}
