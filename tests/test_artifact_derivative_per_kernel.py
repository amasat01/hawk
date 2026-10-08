# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``build_bundle(..., derivative=...)`` resolves the ``derivative`` sidecar
block per kernel, not as one value shared across the whole bundle: a
``vjp``/``jvp`` call tags the resulting kernel with its own direction and
primal at the point the transform computes them, so a bundle mixing a primal
with its derivatives stamps the right block on each derivative kernel and
none on the primal itself.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.ir import HawkError

#: The one differentiated input this whole file exercises: ``vec3_scale``'s
#: ``x`` (a ``Vector[3]``) — small enough that a REVERSE gradient (``bar_x``)
#: and a FORWARD tangent (``dot_x``) both come out one Assign deep, so what a
#: row asserts about the shape is not obscured by the arithmetic.
_WRT = ("x",)


def _derivative_kernels():
    """A fresh (vjp, jvp) pair of :data:`_deployable.vec3_scale`, wrapped as
    ordinary :class:`~hawk.trace.Kernel` the way any author who does not go
    through a bundle-composing frontend would: :func:`hawk.diff.vjp`/``jvp``
    return sinks, and a bare sink tuple is not itself buildable (the
    published surface takes ``.name``/``.sinks``/``.walk``) until it is
    wrapped. Fresh per call because :func:`hawk.trace.Kernel` is cheap and a
    shared mutable fixture here would tie every row's cache identity together
    for no reason."""
    vjp_kernel = Kernel("vec3_scale_vjp", vjp(D.vec3_scale, wrt=_WRT))
    jvp_kernel = Kernel("vec3_scale_jvp", jvp(D.vec3_scale, wrt=_WRT))
    return vjp_kernel, jvp_kernel


def _built_unit(tmp_path, cache_dir, *, targets=("host",), **kw):
    """``{vec3_scale, its vjp, its jvp}``, published as ONE bundle — the exact
    shape a producer is asked to publish a forward kernel and its
    derivatives in. Host-only BY DEFAULT: most of this file is about the
    SIDECAR text, which a host-only build produces exactly as a mixed one
    would but faster (no nvcc). The one row that needs the ``cuda`` leg too is
    :func:`test_eagle_loads_and_plans_the_per_kernel_unit_on_the_host_path`,
    which passes ``targets=("cuda", "host")`` explicitly — ``eagle.registry.load_manifest`` only recognizes a manifest ``format`` of ``ptx``/``cubin``/
    ``fatbin`` (a host-only bundle's own ``"host"`` format is not one of
    them, a separate pre-existing gap), so reaching that
    loader at all needs the device leg present, GPU JIT and all — exactly the
    device work every other fixture bundle in this suite already pays for
    the same reason."""
    vjp_kernel, jvp_kernel = _derivative_kernels()
    return build_bundle([D.vec3_scale, vjp_kernel, jvp_kernel], tmp_path,
                        targets=targets, cache_dir=cache_dir, **kw)


def test_the_primal_carries_no_back_reference_and_its_derivatives_do(
    tmp_path, cache_dir
):
    """shape, on ONE published unit, with NO ``derivative=`` override
    at all — the point of reading the direction/primal off the kernel's own
    ``.sinks`` rather than the mapping: the common case needs no restatement."""
    bundle = _built_unit(tmp_path, cache_dir)

    primal_sc = sidecar_of(bundle, "vec3_scale")
    vjp_sc = sidecar_of(bundle, "vec3_scale_vjp")
    jvp_sc = sidecar_of(bundle, "vec3_scale_jvp")

    assert "derivative" not in primal_sc, (
        "R1's other half: a FORWARD kernel sharing a manifest with its own "
        "derivative must carry no back-reference of its own"
    )
    # "primal_unit": None is the canonical "same bundle as this
    # derivative" spelling — see tests/test_artifact_derivative_cross_unit.py
    # for the non-None case, a primal published in a DIFFERENT unit.
    assert vjp_sc["derivative"] == {"kind": "vjp", "wrt": list(_WRT),
                                    "primal": "vec3_scale", "primal_unit": None}
    assert jvp_sc["derivative"] == {"kind": "jvp", "wrt": list(_WRT),
                                    "primal": "vec3_scale", "primal_unit": None}
    # the three blocks are not one shared object re-stamped three times (the
    # RED symptom): the vjp and jvp disagree on `kind`, which a single shared
    # value could never do.
    assert vjp_sc["derivative"] != jvp_sc["derivative"]


def test_an_explicit_primal_override_is_honoured_over_the_auto_detected_one(
    tmp_path, cache_dir
):
    """The mapping's stated job: naming the primal, nothing else. Overriding it
    to the SAME (only valid) name is a narrow row — HAWK has no second primal
    in this fixture to redirect to — but it still proves the override path is
    read at all, and that ``kind``/``wrt`` still come from the kernel's own
    tag even when the mapping fires (the override supplies ``primal`` alone,
    never a substitute for either)."""
    vjp_kernel, jvp_kernel = _derivative_kernels()
    bundle = build_bundle(
        [D.vec3_scale, vjp_kernel, jvp_kernel], tmp_path, targets=("host",),
        cache_dir=cache_dir,
        derivative={"vec3_scale_vjp": "vec3_scale"},
    )
    vjp_sc = sidecar_of(bundle, "vec3_scale_vjp")
    assert vjp_sc["derivative"] == {"kind": "vjp", "wrt": list(_WRT),
                                    "primal": "vec3_scale", "primal_unit": None}


def test_a_mapping_naming_an_unknown_kernel_is_refused(tmp_path, cache_dir):
    """A mapping naming an unknown kernel is refused: a typo in the
    override's KEY must not silently ship a kernel the author believed was
    tagged."""
    vjp_kernel, _jvp_kernel = _derivative_kernels()
    with pytest.raises(HawkError, match="not_a_bundle_member"):
        build_bundle(
            [D.vec3_scale, vjp_kernel], tmp_path, targets=("host",),
            cache_dir=cache_dir,
            derivative={"not_a_bundle_member": "vec3_scale"},
        )


def test_a_primal_name_outside_the_bundle_is_refused(tmp_path, cache_dir):
    """Naming a primal that is not itself published in THIS bundle, and
    naming NO ``primal_unit`` for it either, is refused rather than silently
    accepted. A cross-unit primal IS publishable (see
    ``tests/test_artifact_derivative_cross_unit.py``), but only when the
    caller says WHICH unit it lives in — a bare name with nothing else is
    exactly as unresolvable as it always was, and this row is what proves the
    fix did not turn that refusal into a guess."""
    vjp_kernel, _jvp_kernel = _derivative_kernels()
    with pytest.raises(HawkError, match="elsewhere_entirely"):
        build_bundle(
            [D.vec3_scale, vjp_kernel], tmp_path, targets=("host",),
            cache_dir=cache_dir,
            derivative={"vec3_scale_vjp": "elsewhere_entirely"},
        )


@pytest.mark.gpu
def test_eagle_loads_and_plans_the_per_kernel_unit_on_the_host_path(
    tmp_path, cache_dir
):
    """eagle's loader and plan still load such a unit:
    ``eagle.registry.load_manifest`` reads each of the three sidecars
    independently (``eagle.roles.parse_derivative`` only checks ``kind`` and
    the Phase-A ``residuals`` rule, never that a bundle's members agree on
    anything about ``derivative``), and the forward kernel still PLANS and
    RUNS through ``eagle.plan`` under ``HostTeam`` exactly as it did before it
    had two derivative siblings in its manifest. Built with BOTH targets (see
    :func:`_built_unit`'s docstring for why the ``cuda`` leg has to be there
    too for ``eagle.registry.load_manifest`` to accept this manifest at all) —
    the one row in this file that JITs PTX on GPU-0."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = _built_unit(tmp_path, cache_dir, targets=("cuda", "host"))
    reg = L.load_bundle(bundle.directory)

    assert reg["vec3_scale"].derivative is None
    assert reg["vec3_scale_vjp"].derivative == {"kind": "vjp", "wrt": list(_WRT),
                                                "primal": "vec3_scale",
                                                "primal_unit": None}
    assert reg["vec3_scale_jvp"].derivative == {"kind": "jvp", "wrt": list(_WRT),
                                                "primal": "vec3_scale",
                                                "primal_unit": None}

    sc = sidecar_of(bundle, "vec3_scale")
    plugin = L.host_plugin(bundle.directory, "vec3_scale", sc)
    x = D.plane(8)
    got = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=2.0)
    np.testing.assert_allclose(np.asarray(got), D.ref_vec3_scale(x, 2.0))
