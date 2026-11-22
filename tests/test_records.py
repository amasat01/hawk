# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A ``Record`` is a name for a group of planes of one extent: no storage and no node of its own.

A kernel over a record equals, by canonical digest, the same kernel written with the separate planes under the
expanded names; values are compared with NumPy, which is independent of the code under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as hm
from hawk import HawkError, _ir, Mutable, Record, Scalar, Vector, runtime
from hawk.artifact import build_bundle

N = 12
RNG = np.random.default_rng(20261122)
Particle = Record(pos=Vector[3], mass=Scalar)


def _over_a_record():
    @hawk.kernel
    def energy(p: Particle, v: Vector[3], y: Mutable[Scalar]):
        y = 0.5 * p.mass * hm.dot(v, v) + hm.norm(p.pos) * p.mass

    return energy


def _over_planes():
    @hawk.kernel
    def energy(p_pos: Vector[3], p_mass: Scalar, v: Vector[3], y: Mutable[Scalar]):
        y = 0.5 * p_mass * hm.dot(v, v) + hm.norm(p_pos) * p_mass

    return energy


def _digest(kernel):
    return _ir.digest(_ir.canonical(kernel.sinks))


def _data():
    return {
        "p_pos": RNG.normal(size=(3, N)),
        "p_mass": RNG.uniform(0.5, 2, N),
        "v": RNG.normal(size=(3, N)),
    }


def _oracle(d):
    return (
        0.5 * d["p_mass"] * np.sum(d["v"] ** 2, axis=0)
        + np.linalg.norm(d["p_pos"], axis=0) * d["p_mass"]
    )


def test_a_record_parameter_expands_to_one_plane_per_field_named_after_the_parameter():
    kernel = _over_a_record()
    assert list(kernel.planes) == ["p_pos", "p_mass", "v", "y"]
    assert {
        name: plane.group for name, plane in kernel.planes.items() if plane.group
    } == {
        "p_pos": ("p", "pos", "record"),
        "p_mass": ("p", "mass", "record"),
    }


def test_a_kernel_over_a_record_has_the_digest_of_the_same_kernel_over_separate_planes():
    assert _digest(_over_a_record()) == _digest(_over_planes())


def test_the_record_kernel_equals_numpy_by_the_python_evaluation():
    d = _data()
    for lane in range(N):
        env = {
            name: (value[..., lane] if value.ndim == 2 else value[lane])
            for name, value in d.items()
        }
        got = evaluate(_over_a_record().sinks, env)["y"]
        np.testing.assert_allclose(got, _oracle(d)[lane], rtol=1e-13)


def _bundle(tmp_path, cache_dir):
    return build_bundle(
        [_over_a_record()],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def test_the_sidecar_lists_the_grouping(tmp_path, cache_dir):
    meta = sidecar_of(_bundle(tmp_path, cache_dir), "energy")
    assert meta["records"] == {"p": {"pos": "p_pos", "mass": "p_mass"}}


def test_a_call_binds_the_expanded_names_a_mapping_or_an_object_alike_on_the_host(
    tmp_path, cache_dir
):
    bundle = _bundle(tmp_path, cache_dir)
    kernel = runtime.load(bundle.directory, "energy")
    d = _data()
    pos, mass = d.pop("p_pos"), d.pop("p_mass")
    results = []
    for form in (
        {"p_pos": pos, "p_mass": mass},
        {"p": {"pos": pos, "mass": mass}},
        {"p": SimpleNamespace(pos=pos, mass=mass)},
    ):
        y = np.zeros(N)
        runtime.run(kernel, **form, **d, y=y)
        results.append(y)
    want = _oracle({"p_pos": pos, "p_mass": mass, **d})
    for y in results:
        np.testing.assert_allclose(y, want, rtol=1e-13)


def test_a_record_value_missing_a_field_or_clashing_with_an_expanded_name_is_refused(
    tmp_path, cache_dir
):
    kernel = runtime.load(_bundle(tmp_path, cache_dir).directory, "energy")
    d = _data()
    pos, mass = d.pop("p_pos"), d.pop("p_mass")
    with pytest.raises(
        HawkError,
        match=r"record 'p' declares the fields \['mass', 'pos'\]; the value lacks \['mass'\]",
    ):
        runtime.run(kernel, p={"pos": pos}, **d, y=np.zeros(N))
    with pytest.raises(
        HawkError, match="given both by itself and as the field 'mass' of 'p'"
    ):
        runtime.run(
            kernel, p={"pos": pos, "mass": mass}, p_mass=mass, **d, y=np.zeros(N)
        )


def test_a_field_the_record_lacks_is_refused_naming_its_fields():
    with pytest.raises(
        HawkError,
        match=r"record 'p' has no field 'charge'; its fields are \['mass', 'pos'\]",
    ):

        @hawk.kernel
        def wrong(p: Particle, y: Mutable[Scalar]):
            y = p.charge


def test_a_declaration_is_refused_when_empty_called_or_given_a_non_payload_field():
    with pytest.raises(HawkError, match="at least one field"):
        Record()
    with pytest.raises(HawkError, match="field 'pos' takes a payload type"):
        Record(pos=3)
    with pytest.raises(HawkError, match="annotated, not called"):
        Particle()


def test_an_expanded_name_that_another_parameter_already_has_is_refused():
    with pytest.raises(HawkError, match=r"record 'p' expands to \['p_mass'\]"):

        @hawk.kernel
        def clash(p: Particle, p_mass: Scalar, y: Mutable[Scalar]):
            y = p_mass


def _reads_only_the_position():
    @hawk.kernel
    def spread(p: Particle, y: Mutable[Scalar]):
        y = hm.norm(p.pos)

    return spread


def _spread_bundle(tmp_path, cache_dir):
    return build_bundle(
        [_reads_only_the_position()],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def test_a_whole_record_binds_to_a_kernel_that_reads_one_field_and_equals_numpy(
    tmp_path, cache_dir
):
    kernel = runtime.load(_spread_bundle(tmp_path, cache_dir).directory, "spread")
    pos, mass = RNG.normal(size=(3, N)), RNG.uniform(0.5, 2, N)
    for whole in ({"pos": pos, "mass": mass}, SimpleNamespace(pos=pos, mass=mass)):
        y = np.zeros(N)
        runtime.run(kernel, p=whole, y=y)
        np.testing.assert_allclose(y, np.linalg.norm(pos, axis=0), rtol=1e-13)


def test_an_undeclared_field_is_refused_naming_the_record_and_the_field(
    tmp_path, cache_dir
):
    kernel = runtime.load(_spread_bundle(tmp_path, cache_dir).directory, "spread")
    pos, mass = RNG.normal(size=(3, N)), RNG.uniform(0.5, 2, N)
    for whole in (
        {"pos": pos, "mass": mass, "charge": mass},
        SimpleNamespace(pos=pos, mass=mass, charge=mass),
    ):
        with pytest.raises(
            HawkError, match=r"record 'p' declares .*\['charge'\] are not fields"
        ):
            runtime.run(kernel, p=whole, y=np.zeros(N))
