# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A derivative's primal may publish in a different unit, named by that
unit's content digest, when the primal and the derivative disagree on
execution axis and so cannot share one bundle. ``build_bundle`` accepts a
``primal_unit`` override (a ``(primal_name, primal_unit_digest)`` pair, or
the digest tagged directly onto ``vjp``/``jvp``'s returned ``Derived``) so
such a primal can still publish; the sidecar's ``derivative`` block records
the referenced unit's digest, or ``None`` when the primal is in the same
bundle.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Kernel
from hawk.artifact import build_bundle
from hawk.diff import vjp
from hawk.ir import HawkError

#: `table` is the only differentiable leaf of `gather` (`where` is an `Index`,
#: dtype `i32`, excluded from autodiff by `hawk.diff.transform._DIFFERENTIABLE`
#: — see that module), so this is `vjp`'s own default `wrt` made explicit.
_WRT = ("table",)


def _primal_bundle(tmp_path, cache_dir, *, targets=("host",)):
    return build_bundle([D.gather], tmp_path / "primal", targets=targets,
                        cache_dir=cache_dir)


def _vjp_kernel(*, primal_unit):
    return Kernel("gather_vjp", vjp(D.gather, wrt=_WRT, primal_unit=primal_unit))


def test_the_two_axes_cannot_share_one_bundle(tmp_path, cache_dir):
    """Demonstrates directly why a cross-unit reference is necessary: a
    primal and a derivative whose execution axes disagree cannot share one
    bundle, and that rule is not being relaxed here."""
    vjp_kernel = _vjp_kernel(primal_unit=None)
    with pytest.raises(HawkError, match="declares ONE execution axis"):
        build_bundle([D.gather, vjp_kernel], tmp_path, targets=("host",),
                     cache_dir=cache_dir)


def test_a_cross_unit_primal_reference_publishes_named_by_digest(tmp_path, cache_dir):
    """RED today (see this module's docstring for the exact
    refusal measured against 833b3c7): a primal named outside the bundle was
    refused UNCONDITIONALLY, with no digest able to satisfy it. The fix:
    ``vjp(..., primal_unit=primal_bundle.digest)`` — the primal's ALREADY
    published unit — lets the reference publish, and each sidecar is right:
    the primal carries no back-reference of its own ( other half),
    the derivative's ``primal_unit`` names the primal's real digest."""
    primal_bundle = _primal_bundle(tmp_path, cache_dir)
    vjp_kernel = _vjp_kernel(primal_unit=primal_bundle.digest)
    vjp_bundle = build_bundle([vjp_kernel], tmp_path / "vjp", targets=("host",),
                              cache_dir=cache_dir)

    primal_sc = sidecar_of(primal_bundle, "gather")
    vjp_sc = sidecar_of(vjp_bundle, "gather_vjp")

    assert "derivative" not in primal_sc
    assert primal_sc["exec_access"] == "cross_sample_read"
    assert vjp_sc["exec_access"] == "cross_sample_write"
    assert vjp_sc["derivative"] == {
        "kind": "vjp", "wrt": list(_WRT), "primal": "gather",
        "primal_unit": primal_bundle.digest,
    }


def test_a_wrong_primal_unit_digest_is_refused_by_the_consumer_naming_it(
    tmp_path, cache_dir
):
    """``build_bundle`` cannot validate an EXTERNAL digest at publish time —
    the unit it names may not exist yet, or may have been published by a
    different process entirely — so a wrong digest still PUBLISHES; the
    refusal belongs to whoever actually tries to RESOLVE the reference. That
    is ``tests/_deploy.py``'s ``resolve_primal``: given the set of
    units a real consumer has loaded, a digest matching none of them is
    refused NAMING the digest, never silently ignored."""
    primal_bundle = _primal_bundle(tmp_path, cache_dir)
    wrong_digest = "f" * len(primal_bundle.digest)
    assert wrong_digest != primal_bundle.digest
    vjp_kernel = _vjp_kernel(primal_unit=wrong_digest)
    vjp_bundle = build_bundle([vjp_kernel], tmp_path / "vjp", targets=("host",),
                              cache_dir=cache_dir)
    vjp_sc = sidecar_of(vjp_bundle, "gather_vjp")
    assert vjp_sc["derivative"]["primal_unit"] == wrong_digest

    with pytest.raises(ValueError, match=wrong_digest):
        L.resolve_primal(vjp_sc, own_bundle=vjp_bundle,
                         other_bundles={primal_bundle.digest: primal_bundle})


@pytest.mark.gpu
def test_eagle_loads_both_units_and_the_consumer_resolves_the_reference(
    tmp_path, cache_dir
):
    """The end-to-end proof (this file's GPU-touching row, CUDA_VISIBLE_
    DEVICES=0): TWO independent manifests, each self-sufficient (``eagle.
    registry.load_manifest`` reads a sidecar's ``derivative`` in isolation —
    ``eagle.roles.parse_derivative`` never asks whether a bundle's members
    agree on anything about it, this file's docstring's point made
    concrete), and a consumer holding both loaded units can walk the
    back-reference straight to the primal's own sidecar. The primal still
    PLANS and RUNS through ``eagle.plan`` exactly as it would with no
    derivative sibling at all -- the reference is metadata a consumer may
    resolve, never a load-time coupling between the two units."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    primal_bundle = _primal_bundle(tmp_path, cache_dir, targets=("cuda", "host"))
    vjp_kernel = _vjp_kernel(primal_unit=primal_bundle.digest)
    vjp_bundle = build_bundle([vjp_kernel], tmp_path / "vjp",
                              targets=("cuda", "host"), cache_dir=cache_dir)

    reg_primal = L.load_bundle(primal_bundle.directory)
    reg_vjp = L.load_bundle(vjp_bundle.directory)
    assert reg_primal["gather"].derivative is None
    assert reg_vjp["gather_vjp"].derivative["primal_unit"] == primal_bundle.digest

    vjp_sc = sidecar_of(vjp_bundle, "gather_vjp")
    resolved_sc = L.resolve_primal(vjp_sc, own_bundle=vjp_bundle,
                                   other_bundles={primal_bundle.digest: primal_bundle})
    assert resolved_sc["kernel"] == "gather"

    primal_sc = sidecar_of(primal_bundle, "gather")
    plugin = L.host_plugin(primal_bundle.directory, "gather", primal_sc)
    table = np.arange(8, dtype=float) * 3.0
    where = ((np.arange(8) * 7 + 3) % 8).astype(np.int64)
    got = eplan.plan(plugin, structure=eexec.HostTeam).run(table=table, where=where)
    np.testing.assert_allclose(np.asarray(got), D.ref_gather(table, where))
