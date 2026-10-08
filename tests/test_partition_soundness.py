# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The DECLARED access class
predicts what happens under ``npartitions=k``.

WHAT A CLASS IS A PROMISE ABOUT. HAWK infers an access class from the DAG's
access FORMS and declares it in the manifest; eagle acts on it at PLAN time,
before anything is packed. So the class makes two checkable promises, and this
file checks both, for every fixture kernel, under BOTH single-node structures:

* PLACEMENT — ``sample_local`` / ``cross_sample_read`` / ``mapreduce`` may be
  split; ``cross_sample_write`` may not, and is refused NAMING the rule,
  because two partitions accumulating into one shared target would need a
  partial-accum combine that is not specified;
* INVARIANCE — where the class permits a split, whole and split must agree:
  BIT-identically for ``sample_local``/``cross_sample_read``, and within
  the one derived band for a ``mapreduce`` FOLD, whose combine order differs
  between a whole run and a partitioned one by construction.

SOUNDNESS IS THE POINT. ranks the classes least-to-most restrictive
and lets the most restrictive win, because over-restriction costs a placement
while UNDER-restriction is a wrong answer. So the row does not merely check that
the declared class is *a* class: it re-derives, from the walk's own nodes, the
FORMS actually present, and requires the declaration to be at least that
restrictive.

NON-VACUITY, TWICE OVER, because neither promise is self-evidencing:

* the placement gate tracks the DECLARATION, not the body — the same plugin,
  with ``exec_access`` overridden in each direction, flips from accepted to
  refused and back. Without that, "cross_sample_write was refused" could just be
  eagle refusing something about the body;
* the invariance comparison can FAIL — proven against
  ``_planted.count_reading_bundle``, an artifact this row compiles whose body
  divides by the partition's own ``count`` instead of the ``nsamples`` role
  (the failure verbatim), and which agrees whole-view and diverges split.

This test previously failed. ``hawk/ir/access.py``'s ``cross_sample_write`` branch
disabled — an indexed accumulate sink no longer folded into the inferred class,
so the scatter fixture declared itself elementwise. Two rows went red, the
soundness one and the coverage one::

    AssertionError: scatter declares 'sample_local' but its walk carries
    ['cross_sample_write'], which is at least 'cross_sample_write' (a
    declaration may never be LESS restrictive than the forms present)
    AssertionError: ['cross_sample_read', 'mapreduce', 'sample_local']
        [test_every_access_class_the_contract_names_is_covered_by_a_fixture]

Worth recording WHY the placement row stayed green under that plant: with the
class mis-declared, eagle correctly PERMITS the split, and on a single process
a scatter over contiguous partitions performs exactly the same additions in
exactly the same order — so the answer does not move. That is precisely why the
soundness arm reads the walk's FORMS instead of trusting the run: F-e is about
partitions that do not share one accumulator, and a single-node run cannot
exhibit it. The plant was then removed.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import _oracle as O
import _planted as PL
import numpy as np
import pytest
from conftest import sidecar_of
from test_serial_oracle import band

from hawk.ir.nodes import AccumWrite, At, MapreducePartial, WideWrite

N = 96
K = 3

#: The access classes that are partition-invariant BIT for bit.
BIT_INVARIANT = ("sample_local", "cross_sample_read")

_CASES = D.cases(N)
_IDS = [c[0] for c in _CASES]


def _forms_present(walk) -> set:
    """The access FORMS the walk actually carries, re-derived from its nodes.

    Deliberately NOT ``hawk.ir.access.infer`` — asking the classifier under test
    what to expect is how a soundness row ends up comparing a function to
    itself. This reads the node kinds defines the forms as."""
    forms = set()
    for node in walk.order:
        if isinstance(node, MapreducePartial):
            forms.add("mapreduce")
        elif isinstance(node, At):
            forms.add("cross_sample_read")
        elif isinstance(node, (WideWrite, AccumWrite)) and node.index is not None:
            forms.add("cross_sample_write")
    return forms


_RANK = {"sample_local": 0, "cross_sample_read": 1, "cross_sample_write": 2}


#: One entry per distinct (bundle, kernel), for the rows that read the KERNEL
#: rather than a run of it.
_KERNELS = sorted({(c[0], c[1]) for c in _CASES})


@pytest.mark.parametrize("name,kernel", _KERNELS,
                         ids=[f"{n}-{k}" for n, k in _KERNELS])
def test_the_declared_class_is_never_less_restrictive_than_the_forms(built, name,
                                                                    kernel):
    """The declared class is never less restrictive than the forms present.
    Over-restriction is legal (it costs a placement); UNDER-restriction
    is a wrong answer under a partitioning the class does not permit."""
    declared = sidecar_of(built[name], kernel)["exec_access"]
    forms = _forms_present(getattr(D, kernel).walk)
    if "mapreduce" in forms:
        assert declared == "mapreduce", (
            f"{kernel} carries a mapreduce partial sink but declares {declared!r}; "
            "makes that sink exclusive and eagle would return un-folded partials"
        )
        return
    floor = max((_RANK[f] for f in forms), default=0)
    assert _RANK[declared] >= floor, (
        f"{kernel} declares {declared!r} but its walk carries "
        f"{sorted(forms)}, which is at least "
        f"{[c for c, r in _RANK.items() if r == floor][0]!r} (a declaration "
        "may never be LESS restrictive than the forms present)"
    )


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES, ids=_IDS)
def test_the_class_predicts_the_placement(built, bundle_name, kernel, kw, _want):
    """Predicts the placement at PLAN time, on both single-node structures."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built[bundle_name]
    sidecar = sidecar_of(bundle, kernel)
    access = sidecar["exec_access"]
    plugin = L.host_plugin(bundle.directory, kernel, sidecar)
    for structure in (eexec.HostTeam, eexec.DeviceKernel):
        if access == "cross_sample_write":
            with pytest.raises(ValueError) as excinfo:
                eplan.plan(plugin, structure=structure, npartitions=K)
            message = str(excinfo.value)
            assert "F-e" in message and "cross_sample_write" in message, (
                f"the refusal must name the rule (I3): {message}"
            )
        else:
            eplan.plan(plugin, structure=structure, npartitions=K)


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES, ids=_IDS)
def test_the_placement_gate_tracks_the_declaration_not_the_body(
        built, bundle_name, kernel, kw, _want):
    """NON-VACUITY for the row above. The SAME plugin, with ``exec_access``
    overridden in each direction, flips from accepted to refused — so what the
    gate reads is the declaration HAWK writes, which is what makes the
    soundness obligation load-bearing rather than advisory."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built[bundle_name]
    plugin = L.host_plugin(bundle.directory, kernel, sidecar_of(bundle, kernel))
    with pytest.raises(ValueError, match="F-e"):
        eplan.plan(plugin, structure=eexec.HostTeam, npartitions=K,
                   _exec_access="cross_sample_write")
    eplan.plan(plugin, structure=eexec.HostTeam, npartitions=K,
               _exec_access="sample_local")


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES, ids=_IDS)
@pytest.mark.gpu
def test_a_split_run_agrees_with_the_whole_run_as_the_class_promises(
        built, bundle_name, kernel, kw, _want):
    """The partition-invariance ruling, on BOTH structures. ``mapreduce`` is judged on its FOLD, which is
    where a partitioned combine order can actually show, and band-gated for
    exactly that reason; every other permitted class is bit-gated."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built[bundle_name]
    sidecar = sidecar_of(bundle, kernel)
    access, scalar_type = sidecar["exec_access"], sidecar["scalar_type"]
    if access == "cross_sample_write":
        # NOT a skip: a skipped row is pinned and certifies nothing. What the
        # class promises for this body is that the split never HAPPENS, so that
        # is what is asserted — the promise, stated positively, on the same
        # subject as every other case in this sweep.
        plugin = L.host_plugin(bundle.directory, kernel, sidecar)
        with pytest.raises(ValueError, match="F-e"):
            eplan.plan(plugin, structure=eexec.HostTeam, npartitions=K)
        whole = eplan.plan(plugin, structure=eexec.HostTeam, npartitions=1).run(**kw)
        oracle = O.run(bundle.directory, kernel, N, sidecar, **kw)
        for g, w in zip(_tuple(whole), _tuple(oracle)):
            assert np.ascontiguousarray(g).tobytes() ==\
                np.ascontiguousarray(w).tobytes(), (
                f"{kernel}: the ONE placement its class permits (whole view) does "
                "not agree with the serial oracle"
            )
        return

    for loader, structure in ((L.host_plugin, eexec.HostTeam),
                              (L.device_plugin, eexec.DeviceKernel)):
        plugin = loader(bundle.directory, kernel, sidecar)
        whole = eplan.plan(plugin, structure=structure, npartitions=1).run(**kw)
        split = eplan.plan(plugin, structure=structure, npartitions=K).run(**kw)
        what = f"{kernel} under {structure.name}: whole vs {K} partitions"
        if access == "mapreduce":
            a = eexec.fold(sidecar["exec_op"], np.asarray(whole).ravel().tolist())
            b = eexec.fold(sidecar["exec_op"], np.asarray(split).ravel().tolist())
            limit = band(np.asarray(whole).size, np.asarray([a]), np.asarray([b]),
                         scalar_type)
            assert abs(a - b) <= limit, (
                f"{what}: the folded partials differ by {abs(a - b)}, band {limit}"
            )
        else:
            for g, w in zip(_tuple(split), _tuple(whole)):
                assert np.ascontiguousarray(g).tobytes() ==\
                    np.ascontiguousarray(w).tobytes(), (
                    f"{what}: a {access!r} body is BIT-invariant under partitioning "
                    "(EC-I2); it is not"
                )


def _tuple(value):
    # eagle.plan.Plan.run returns a dict keyed by plane name for several
    # outputs; its insertion order follows arg_spec's own order, the same
    # order a plain tuple (the serial oracle's own shape) already carries,
    # so unwrapping it to a tuple of values lines the two zips up correctly.
    if isinstance(value, dict):
        return tuple(value.values())
    return value if isinstance(value, tuple) else (value,)


def test_the_invariance_comparison_can_fail(built, tmp_path, cache_dir):
    """NON-VACUITY for the row above (a gate builds its own inputs). The planted
    ``count``-reading artifact declares ``sample_local`` — truthfully, its DAG is
    elementwise — and is therefore permitted to split; its body is nevertheless
    not partition-invariant, and the same comparison must say so."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(built["fraction"], "fraction")
    assert sidecar["exec_access"] == "sample_local"
    directory = PL.with_sidecar(
        PL.count_reading_bundle(D.fraction, tmp_path / "planted",
                                cache_dir=cache_dir), sidecar)
    plugin = L.host_plugin(directory, "fraction", sidecar)
    kw = {"x": D.plane(N, w=1)[0]}
    whole = eplan.plan(plugin, structure=eexec.HostTeam, npartitions=1).run(**kw)
    split = eplan.plan(plugin, structure=eexec.HostTeam, npartitions=K).run(**kw)
    assert whole.tobytes() != split.tobytes(), (
        "the planted body — which divides by the partition's own count — came out "
        f"bit-identical whole and over {K} partitions; the invariance comparison "
        "above cannot detect the defect it exists for"
    )
    # ... and the un-planted artifact, through the identical comparison, agrees:
    real = L.host_plugin(built["fraction"].directory, "fraction", sidecar)
    assert (eplan.plan(real, structure=eexec.HostTeam, npartitions=1).run(**kw)
            .tobytes()
            == eplan.plan(real, structure=eexec.HostTeam, npartitions=K).run(**kw)
            .tobytes())


def test_every_access_class_the_contract_names_is_covered_by_a_fixture(built):
    """A sweep is only as good as the set it covers, and a class with no fixture
    is a class this row silently certifies nothing about."""
    declared = {sidecar_of(built[name], kernel)["exec_access"]
                for name, kernel, _kw, _w in _CASES}
    assert declared == {"sample_local", "cross_sample_read", "cross_sample_write",
                        "mapreduce"}, sorted(declared)


def test_the_serial_oracle_is_the_invariant_arm(built):
    """The oracle is not one of the partitionings: it is what they are compared
    against. Stated here so the partitioned arms and the oracle cannot
    drift apart about which arm is the reference."""
    bundle = built["vec3"]
    sidecar = sidecar_of(bundle, "vec3_scale")
    kw = {"x": D.plane(N), "a": -1.25}
    oracle = O.run(bundle.directory, "vec3_scale", N, sidecar, **kw)

    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, "vec3_scale", sidecar)
    for npartitions in (1, 2, K, N):
        got = eplan.plan(plugin, structure=eexec.HostTeam,
                         npartitions=npartitions).run(**kw)
        assert got.tobytes() == oracle.tobytes(), (
            f"HostTeam({npartitions}) differs from the serial oracle"
        )
