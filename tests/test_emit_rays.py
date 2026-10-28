# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ray ops through the emitters.

* **aether text**: one ``aether::geometry`` call per node, its four values held once;
  the unit includes the geometry header and no other unit does.
* **C target**: the node is a call to the ``aether_dsc.c_core`` function of the same name;
  the exported unit compiles under c99 and computes the host build's bits (both
  without floating-point contraction), and it passes the strict C checks."""

from __future__ import annotations

import pathlib

import _embedded as E
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of
from test_ir_rays import ball, box, tri

import hawk.embedded as embedded
from hawk.artifact import build_bundle
from hawk.emit import FLOAT64, render_body, render_source
from hawk.ir import canonical
from hawk.targets import TARGETS

KERNELS = {"tri": tri, "box": box, "ball": ball}
CALLS = {"tri": "rayTriangle", "box": "rayAabb", "ball": "raySphere"}
N = 64                                  # the scene helper's size


@pytest.mark.parametrize("name", list(KERNELS))
def test_one_geometry_call_per_node_held_once(name):
    k = KERNELS[name]
    text = render_body(k.sinks, canonical(k.sinks)).text
    assert text.count(f"aether::geometry::{CALLS[name]}<Real>(") == 1
    assert text.count(".v[") == 4


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_unit_includes_the_geometry_header_only_when_it_calls_it(name):
    k = KERNELS[name]
    src = render_source(name, k.sinks, canonical(k.sinks), TARGETS["host"], mode=FLOAT64).text
    assert "#include <aether/geometry/geometry.h>" in src
    from test_dispatch_scoped_hoist import _dk_ladder_lazy as plain
    assert "geometry/geometry.h" not in render_source(
        "plain", plain.sinks, canonical(plain.sinks), TARGETS["host"], mode=FLOAT64).text


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    out = {}
    for name, k in KERNELS.items():
        d = tmp_path_factory.mktemp(f"hawk_rays_c_{name}")
        art = embedded.export(k, d, mode="float64", check=False)
        out[name] = (art, E.cc_shared(art.c_path, d))
    return out


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_c_unit_calls_the_core_twin(units, name):
    art, _so = units[name]
    text = art.c_path.read_text()
    assert f"aether_c_f64_ray_{ {'tri': 'tri', 'box': 'aabb', 'ball': 'sphere'}[name] }(" in text
    assert "aether::" not in text


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_c_unit_computes_the_host_bits(units, name, tmp_path_factory, cache_dir):
    from test_ir_rays import VECTORS, _scene

    k = KERNELS[name]
    rng = np.random.default_rng(23)
    planes = _scene(rng, name)
    given = {nm: v.T.copy() for nm, v in zip(VECTORS[name], planes)}
    given["tmax"] = np.full(N, 6.0)
    if name == "ball":
        given["r"] = rng.uniform(0.5, 2.0, N)
    bundle = build_bundle([k], pathlib.Path(tmp_path_factory.mktemp("hawk_rays_host")) / name,
                          targets=("host",), cache_dir=cache_dir, host_profile="native")
    want = O.run(bundle.directory, name, N, sidecar_of(bundle, name), **given)
    art, so = units[name]
    got = E.run_c(so, name, art.report, N, mode="float64", **given)
    got = np.asarray(got).reshape(want.shape) if np.asarray(got).size == want.size else got
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes(), np.abs(got - want).max()
    assert 0 < want[0].sum() < N


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_c_unit_passes_the_strict_c_checks(name):
    embedded.check(embedded.render(KERNELS[name], mode="float64"))
