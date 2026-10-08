# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A HAWK artifact loads under eagle and RUNS, on both targets.

The whole path a consumer walks, per fixture: ``eagle.registry.load_manifest``
(the manifest-level schema/exec/ABI gates + each sidecar's validation + the
driver module load), the / self-check read out of the artifact's OWN
exports, then ``eagle.plan`` under ``DeviceKernel`` on GPU-0 and under
``HostTeam`` — compared against a numpy reference.

Fixture set (this is the FULL set, not the scalar-handle
subset scoped it to until then): a ``Vector[3]`` output plane, a
``vec_in``, a ``mat_in``, a ``float32`` arm, a MULTI-output kernel, a
``lookup`` gather (``cross_sample_read``), a mapreduce partial, a scatter
(``cross_sample_write``), the ``nsamples`` role, a compound quantity, the
two-wire compound read and the plain scalar-handle shape. The table itself is
``_deployable.cases`` — one home, because oracle and
partition-soundness rows sweep the SAME set and must not be able to
disagree with this one about what it contains.

The cross-generation half is its own two rows at the bottom: a v1
loader must REFUSE a v2 artifact, and a v1 artifact must be refused by a v2
plan.

This test previously failed. ``manifest_doc`` planted to stamp ``schema_version``
1 beside the v2 keys -- the LOAD itself is refused, which is this row's
subject::

    ValueError: 'manifest.json': schema v1 requires aether_abi='aether-abi/1'
    when present; got 'aether-abi/2'
    (raptor/schema/manifest.py:194, reached through eagle.registry.load_manifest)

and the device entry's int64 TRIPLE narrowed to ``int`` --
``assert ' long long base,\n long long count,\n long long nSamples)'
in "// HAWK-emitted cuda translation unit ..."``. Worth recording WHY the
second plant needs a text assertion: at N=16 the narrowed triple still COMPUTES
the right answer (the low words of base/count/nSamples are the values), which
is precisely the silent 2^31 cap -- an executed arm alone would have
certified it green. Both plants were then removed.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

N = 16


def _cases():
    """``(bundle, kernel, kwargs, reference)`` — the shared fixture table."""
    return D.cases(N)


def _compare(got, want, dtype):
    # eagle.plan.Plan.run returns a dict keyed by plane name for several
    # outputs; dict insertion order follows arg_spec's own order, the same
    # positional order `want` (the hand-written reference tuple) is built
    # in, so unwrapping it to a tuple of values lines the two up correctly.
    if isinstance(got, dict):
        got = tuple(got.values())
    else:
        got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    assert len(got) == len(want), f"{len(got)} output plane(s), expected {len(want)}"
    band = 2 * N * (np.finfo(dtype).eps)
    for g, w in zip(got, want):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w),
                                   rtol=64 * band, atol=64 * band)


@pytest.mark.parametrize("bundle_name,kernel,kw,want", _cases(),
                         ids=[c[0] for c in _cases()])
def test_the_artifact_runs_on_the_host_through_eagle(built, bundle_name, kernel,
                                                     kw, want):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built[bundle_name]
    sc = sidecar_of(bundle, kernel)
    plugin = L.host_plugin(bundle.directory, kernel, sc)
    got = eplan.plan(plugin, structure=eexec.HostTeam).run(**kw)
    _compare(got, want, np.float32 if sc["scalar_type"] == "float32" else np.float64)


@pytest.mark.parametrize("bundle_name,kernel,kw,want", _cases(),
                         ids=[c[0] for c in _cases()])
@pytest.mark.gpu
def test_the_artifact_runs_on_the_device_through_eagle(built, bundle_name, kernel,
                                                       kw, want):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built[bundle_name]
    sc = sidecar_of(bundle, kernel)
    plugin = L.device_plugin(bundle.directory, kernel, sc)
    got = eplan.plan(plugin, structure=eexec.DeviceKernel).run(**kw)
    _compare(got, want, np.float32 if sc["scalar_type"] == "float32" else np.float64)


def test_every_gap_row_of_h51_is_closed(built):
    """The eight rows, on ONE loaded artifact: the ABI tag, the schema
    version, the manifest exec keys, the exported layout self-check, the SERIAL
    host entry, the device entry's TRIPLE, the declared access class, and the
    index width (the triple int64, the ``nsamples`` ROLE its own)."""
    import eagle.exec as eexec

    bundle = built["fraction"]
    sc = sidecar_of(bundle, "fraction")
    cu = (bundle.directory / "fraction.cu").read_text()
    cpp = (bundle.directory / "fraction.cpp").read_text()

    assert sc["aether_abi"] == bundle.manifest["aether_abi"] == eexec.ABI_TAG_V2
    assert sc["schema_version"] == bundle.manifest["schema_version"] == 2
    assert {"exec_targets", "exec_access"} <= set(bundle.manifest)
    assert L.host_selfcheck(bundle.directory / "fraction.so")[1] == eexec.layout_sizes()
    assert ("void fraction_host(void* const* params,\n    std::int64_t base,\n"
            "    std::int64_t count,\n    std::int64_t nSamples)") in cpp
    assert "#pragma omp" not in cpp and "#pragma omp" not in cu
    assert ("    long long base,\n    long long count,\n    long long nSamples)") in cu
    assert sc["exec_access"] in ("sample_local", "cross_sample_read",
                                "cross_sample_write", "mapreduce")
    assert "EAGLE_ABI_INDEX_T p_nsm_n_samples" in cu


def test_a_v1_loader_refuses_a_v2_artifact(built, monkeypatch):
    """The cross-generation ruling's first direction. A v1-era loader's ceiling is
    ``MAX_SCHEMA_VERSION == 1``; against that ceiling HAWK's document is
    refused by VERSION, and even taken as v1 it is refused for carrying the
    execution keys (the legacy bridge is whole-view by the axis's
    ABSENCE, never a silent upgrade)."""
    from raptor.schema import manifest as rmanifest

    doc = built["vec3"].manifest
    monkeypatch.setattr(rmanifest, "MAX_SCHEMA_VERSION", 1)
    with pytest.raises(ValueError, match="upgrade eagle"):
        rmanifest.check_schema_version(doc, name="manifest.json")
    # taken AS v1 it is refused twice over: by the tag it carries, and -- with
    # the tag downgraded -- by the execution keys themselves.
    with pytest.raises(ValueError, match="aether-abi/1"):
        rmanifest.check_execution_axis(doc, 1, name="manifest.json")
    downgraded = {**doc, "aether_abi": "aether-abi/1"}
    with pytest.raises(ValueError, match="execution key"):
        rmanifest.check_execution_axis(downgraded, 1, name="manifest.json")


def test_a_v1_artifact_is_refused_by_a_v2_plan(built):
    """The other direction: a legacy ``aether-abi/1`` plugin carries
    no partition triple, so a v2 plan over more than one partition refuses it
    NAMING the rule (``eagle.plan``'s bridge)."""
    from types import SimpleNamespace

    import eagle.exec as eexec
    from eagle import plan as eplan

    from hawk.artifact import plan_view

    view = plan_view(sidecar_of(built["vec3"], "vec3_scale"))
    view.pop("host_entry")
    legacy = SimpleNamespace(abi_tag=eexec.ABI_TAG_V1, **view)
    with pytest.raises(ValueError, match="legacy"):
        eplan.plan(legacy, structure=eexec.HostTeam, npartitions=2)


def test_the_artifact_is_partition_invariant_on_the_host(built):
    """The partition-invariance ruling, for a ``sample_local`` body: whole vs two partitions must be
    BIT-identical — the property that says the emitted body reads its own
    column through the triple and nothing else."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built["vec3"]
    plugin = L.host_plugin(bundle.directory, "vec3_scale",
                           sidecar_of(bundle, "vec3_scale"))
    x = D.plane(N)
    whole = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=-1.25)
    split = eplan.plan(plugin, structure=eexec.HostTeam,
                       partitions=[(0, 6, N), (6, N - 6, N)]).run(x=x, a=-1.25)
    np.testing.assert_array_equal(whole, split)
