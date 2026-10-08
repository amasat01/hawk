# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The sidecar declares a WIDTH and a WIRE DTYPE for every plane-bound
slot, inputs included — not the output-only ``arg_widths`` shipped with
(this module's subject is ``hawk/artifact/sidecar.py``, read
that module's docstring first).

A v2 consumer previously had ``arg_widths`` for a ``mutable``/``out``/
``wide_out``/``accum_out`` plane and NOTHING for a ``vec_in``/``mat_in``/
``per_sample``/``lookup``/``terminated``/``wide_in`` one — eagle's OWN
``Plan.bind`` docstring says so in as many words ("a v2 sidecar declares
widths for its OUTPUT planes only... that gap belongs to the producer's
sidecar, and is being closed there"), and a downstream package's drift guard
names the same two fields (``input_width``, ``param_width``) it cannot
certify for exactly this reason. This file pins the fix: every fixture in
``tests/_deployable.py``'s ``cases`` gets a complete ``arg_shapes``/
``arg_dtypes`` pair, a lookup table's runtime row count is declared ``None``
rather than guessed (in ``arg_shapes``, never in ``arg_widths``), and a
consumer refuses a wrongly-shaped or wrongly-dtyped plane at bind, naming the
slot — on BOTH the door that needed no eagle change (``eagle.plan.Plan.bind``,
once ``arg_widths`` covers an input role it already reads) and the door HAWK
owns for the one case eagle's own door is not built to reach (an
integer/bool plane's dtype, ``tests/_deploy.py``'s new ``check_declared_planes``).

THE THREE-FIELD SPLIT (``arg_widths`` / ``arg_shapes`` / ``arg_dtypes``, not
just two) exists because of a REAL regression a first attempt landed and this
row set caught by running the full suite, not by reasoning about it in
advance: putting ``None`` straight into ``arg_widths`` for a runtime-length
role broke ``eagle.plan._declared_widths`` (an unconditional ``int(v)``) —
and that function is reached NOT ONLY through ``hawk.artifact.plan_view``
(which could filter it) but ALSO through eagle's own v2 artifact loader
(``eagle.registry.load_manifest``), which reads a deployed sidecar's
``arg_widths`` key directly. Measured via ``tests/mpi/check_hawk_rank_bed.sh``
(eagle's OWN rank bed, driven against a real HAWK artifact) turning RED on
the ``mapreduce`` unit's ``accum_out`` plane::

    TypeError: int() argument must be a string, a bytes-like object or a real
    number, not 'NoneType'
      eagle/plan.py:575, in _declared_widths
        return {str(k): int(v) for k, v in declared.items()}

So ``arg_widths`` stays INT-ONLY and WIDENED (never a runtime-length role's
entry, not even ``None``); ``arg_shapes`` is the SEPARATE, truly COMPLETE
record (``arg_widths`` plus an explicit ``None`` for a runtime-length role)
that nothing existing reads yet, so it is safe to complete. This file pins
BOTH halves: completeness (via ``arg_shapes``) and the int-only safety
invariant (via ``arg_widths``, and directly via the reproduction below).

This test previously failed: running this whole file's ancestor
against an earlier ``hawk/artifact/sidecar.py``/``hawk/artifact/__init__.py``
(``git stash`` the two files, keep the test file, run, ``git stash pop`` — 18
of 19 rows red, only the vacuous "accepts correctly declared planes" arm
passed, because with no fields to check it checks nothing). Representative
failures::

    E KeyError: 'arg_dtypes'
    (every_plane_bound_slot_declares_its_width_and_dtype -- the sidecar
    carried no such block at all, on every one of the 12 parametrized cases)

    E assert sc["arg_widths"]["table"] is None
    E KeyError: 'table'
    (a_lookup_tables_row_count_is_none_not_guessed -- arg_widths covered
    OUTPUT roles only, so a `lookup` slot was absent rather than None)

    E Failed: DID NOT RAISE ValueError
    (eagles_own_bind_door_now_refuses_a_wrong_vec_in_width -- confirming
    eagle.plan.Plan.bind's OWN docstring: "WHAT IS DELIBERATELY NOT CHECKED is
    an input plane's component width... there is nothing to check it against")

    E Failed: DID NOT RAISE ValueError
    (the_hawk_consumer_check_refuses_a_wrong_{width,dtype}_naming_the_slot --
    with arg_widths/arg_dtypes absent, check_declared_planes had nothing to
    compare a bound plane against and raised on neither of its two guards)

A first attempt at the eagle-bind row (``match=r"'x'"``) PASSED even against
the pre-fix code — a false green, not a real one: the call was missing
the kernel's ``uniform`` argument, so ``Plan.bind`` raised earlier for an
unrelated reason ("this plugin's arg_spec declares a 'uniform' named 'a'...
bindable: ['y', 'x', 'a']"), and the loose regex matched the quoted ``'x'``
inside that OTHER message's ``bindable`` list. Fixed by binding the uniform
and tightening the match to ``r"vec_in 'x'"``.

A second RED, this one on the FIX rather than the test: a first version of
the fix put ``None`` straight into ``arg_widths`` — every row in THIS
file passed (``plan_view`` filtered the ``None`` before eagle's ``_check_plane``
ever saw it), but the whole-suite run below caught what this file could not,
because this file never drives eagle's OTHER loader::

    GATE RED: eagle's rank bed came out RED against this HAWK artifact
    (see this module's docstring for the TypeError). The row
    test_declared_widths_never_sees_a_runtime_length_none reproduces it
    directly, on the SAME accum_out shape (``energy``'s ``total``), so a
    future change cannot reintroduce it without this file noticing on its own.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

N = 8

#: Roles whose plane's OWN LENGTH is a runtime quantity — the mirror of
#: ``hawk/artifact/sidecar.py``'s ``_RUNTIME_LENGTH_ROLES``, restated
#: independently here (not imported) so this row cannot pass by testing the
#: implementation against itself.
_RUNTIME_LENGTH_ROLES = ("lookup", "wide_in", "wide_out", "accum_out")
_PLANE_ROLES = ("per_sample", "vec_in", "mat_in", "terminated", "mutable", "out",
                *_RUNTIME_LENGTH_ROLES)


def _expected_width(ttype) -> int:
    w = 1
    for e in ttype.shape:
        w *= int(e)
    return w


def _expected_dtype(ttype, scalar_type: str) -> str:
    """The wire dtype string INDEPENDENTLY re-derived from the role→mirror
    map's own vocabulary (``tests/test_emit_role_mirror_map.py``,
    ``hawk/emit/aether.element_spelling``): an ``f64``/``f32`` leaf always
    resolves through the compiled ``Real`` (so it tracks ``scalar_type``, never
    its own IR literal); an ``i32``/``i64`` leaf always resolves through the
    emitted ``Int`` (``long long``, unconditionally 64-bit); ``bool`` passes
    through."""
    if ttype.dtype in ("f64", "f32"):
        return scalar_type
    if ttype.dtype in ("i32", "i64"):
        return "int64"
    if ttype.dtype == "bool":
        return "bool"
    raise AssertionError(f"no expected wire dtype for IR dtype {ttype.dtype!r}")


def _cases():
    return D.cases(N)


@pytest.mark.parametrize("bundle_name,kernel,kw,want", _cases(),
                         ids=[c[0] for c in _cases()])
def test_every_plane_bound_slot_declares_its_shape_and_dtype(built, bundle_name,
                                                              kernel, kw, want):
    """This test, swept over the WHOLE deployment fixture set: every
    ``arg_spec`` slot that binds a memory plane (every role except ``uniform``/
    ``nsamples``, which are by-value) has an entry in BOTH ``arg_shapes`` and
    ``arg_dtypes`` — inputs included, which is the entire point (as shown
    above: a ``vec_in``/``per_sample``/``lookup`` slot was absent from
    ``arg_widths`` and neither ``arg_shapes`` nor ``arg_dtypes`` existed).

    ``arg_widths`` is checked TOO, but for the opposite property: it must
    agree with ``arg_shapes`` on every STATICALLY-shaped role and carry NO
    entry at all — not even ``None`` — for a runtime-length one (this
    module's docstring's measured ``eagle.plan._declared_widths`` crash)."""
    sc = sidecar_of(built[bundle_name], kernel)
    walk = getattr(D, kernel).walk
    widths, shapes, dtypes = sc["arg_widths"], sc["arg_shapes"], sc["arg_dtypes"]
    for role, name in walk.arg_spec:
        if role in ("uniform", "nsamples"):
            assert name not in widths and name not in shapes and name not in dtypes, (
                f"{kernel}: {role} {name!r} carries no plane; it must not "
                "appear in arg_widths/arg_shapes/arg_dtypes"
            )
            continue
        assert role in _PLANE_ROLES, f"unrecognised plane role {role!r}"
        assert name in shapes, f"{kernel}: slot ({role}, {name!r}) has no arg_shapes entry"
        assert name in dtypes, f"{kernel}: slot ({role}, {name!r}) has no arg_dtypes entry"
        ttype = walk.slot_types[(role, name)]
        if role in _RUNTIME_LENGTH_ROLES:
            assert shapes[name] is None, (
                f"{kernel}: {role} {name!r}'s plane length is a runtime "
                f"quantity (its row/edge count) — declared {shapes[name]!r}, "
                "not None; must never GUESS this number"
            )
            assert name not in widths, (
                f"{kernel}: {role} {name!r} is runtime-length and must be "
                f"ABSENT from arg_widths (never a None value there); got "
                f"{widths[name]!r}"
            )
        else:
            expected = _expected_width(ttype)
            assert shapes[name] == expected, (
                f"{kernel}: {role} {name!r} declares shape-width "
                f"{shapes[name]}, the compiled body's own shape says {expected}"
            )
            assert widths[name] == expected, (
                f"{kernel}: {role} {name!r} declares arg_widths {widths[name]}, "
                f"arg_shapes says {expected} — the two must agree"
            )
        assert dtypes[name] == _expected_dtype(ttype, sc["scalar_type"]), (
            f"{kernel}: {role} {name!r} declares dtype {dtypes[name]!r}, "
            f"expected {_expected_dtype(ttype, sc['scalar_type'])!r}"
        )


def test_a_lookup_tables_row_count_is_none_in_shapes_absent_from_widths(built):
    """The fixture set's one genuinely runtime-length plane: ``gather``'s
    ``table`` (a positional ``Table[Scalar]``, whose length the caller alone
    determines — ``hawk/artifact/sidecar.py``'s own ``buffers`` docstring).
    Declared in ``arg_shapes`` (``None``) but entirely ABSENT from
    ``arg_widths`` — the split this module's docstring explains."""
    sc = sidecar_of(built["gather"], "gather")
    assert sc["arg_shapes"]["table"] is None
    assert "table" not in sc["arg_widths"]
    assert sc["arg_dtypes"]["table"] == sc["scalar_type"]


def test_an_integer_per_sample_plane_is_told_apart_from_a_real_one(built):
    """``gather``'s ``where`` (an ``Index``, IR dtype ``i32``) and ``table``
    (a ``Scalar``, IR dtype ``f64``) share the SAME by-value ABI shape
    (``ScalarHandle`` / ``lookup``+``per_sample`` both resolve to it) but
    must NOT share a declared wire dtype — the exact ambiguity
    ``tests/test_emit_role_mirror_map.py``'s ``_HANDLE_ALIASES`` documents as
    a real, then-unclosed gap in the field set."""
    sc = sidecar_of(built["gather"], "gather")
    assert sc["arg_dtypes"]["where"] == "int64"
    assert sc["arg_dtypes"]["table"] == "float64"


def test_declared_widths_never_sees_a_runtime_length_none(built):
    """THE REGRESSION, reproduced directly (this module's docstring's second
    RED). ``energy``'s ``total`` is an ``accum_out`` plane (``Reduce("sum")``)
    — the SAME role shape as the ``mapreduce`` unit whose ``eagle.plan._declared_widths`` call turned eagle's OWN MPI rank bed red the first
    time this fix tried to put ``None`` in ``arg_widths``. Both the raw
    sidecar's ``arg_widths`` and ``hawk.artifact.plan_view``'s projection of
    it are fed to eagle's REAL ``_declared_widths`` here — not a stand-in —
    so a future regression of the same shape fails HERE, in this repo's own
    suite, rather than only in eagle's rank bed three commands later."""
    from types import SimpleNamespace

    from eagle.plan import _declared_widths

    from hawk.artifact import plan_view

    sc = sidecar_of(built["energy"], "energy")
    assert sc["arg_shapes"]["total"] is None, (
        "this row's premise: 'total' must be the runtime-length accum_out "
        "plane the regression was measured on"
    )
    assert "total" not in sc["arg_widths"]
    for widths in (sc["arg_widths"], plan_view(sc)["arg_widths"]):
        got = _declared_widths(SimpleNamespace(arg_widths=widths))
        assert "total" not in got
        assert got == {"v": 3}, got


def test_eagles_own_bind_door_now_refuses_a_wrong_vec_in_width(built):
    """This fix closes HALF the gap with ZERO eagle-side change:
    ``Plan.bind`` already reads ``arg_widths`` for every declared role's shape
    check (``eagle.plan._check_plane``, called from ``_bind_plan``); once HAWK
    declares a ``vec_in``'s width the SAME door that already protected a
    ``mutable`` plane protects an INPUT one too — exactly the gap
    ``Plan.bind``'s own docstring names ("that gap belongs to the producer's
    sidecar, and is being closed there")."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built["vec3"]
    plugin = L.host_plugin(bundle.directory, "vec3_scale",
                           sidecar_of(bundle, "vec3_scale"))
    p = eplan.plan(plugin, structure=eexec.HostTeam)
    with pytest.raises(ValueError, match=r"vec_in 'x'"):
        p.bind(x=np.zeros((5, N)), y=np.zeros((3, N)), a=2.0)
    # the correctly-shaped plane binds and runs, so the row above is a real
    # refusal and not a door that rejects every call.
    x = np.arange(3 * N, dtype=float).reshape(3, N)
    y = np.zeros((3, N))
    bound = p.bind(x=x, y=y, a=2.0)
    bound.launch()
    np.testing.assert_allclose(y, 2.0 * x)


def test_the_hawk_consumer_check_refuses_a_wrong_width_naming_the_slot(built):
    """The narrower half eagle's own door cannot reach: an INTEGER-typed plane
    bound at the wrong width. ``eagle.plan._check_plane`` still validates
    shape for these (it is not dtype-gated), so this row exercises HAWK's OWN
    re-owned ``vec_widths`` check for symmetry with the dtype row below and to
    pin the message names the slot."""
    sc = sidecar_of(built["vec3"], "vec3_scale")
    with pytest.raises(ValueError, match=r"vec_in 'x'"):
        L.check_declared_planes(
            sc, {"x": np.zeros((5, N)), "y": np.zeros((3, N))}, N)


def test_the_hawk_consumer_check_refuses_a_wrong_dtype_naming_the_slot(built):
    """The gap ``eagle.plan._check_plane`` cannot reach at all: it exempts
    every integer/bool array from its dtype check outright ("an integer or
    bool array is passed at its own dtype"), so a ``lookup``/``per_sample``
    plane bound at the wrong INTEGER width has no eagle-side gate on either
    era's sidecar. HAWK's own re-owned check (``tests/_deploy.py``'s
    ``check_declared_planes``) is where this is caught, using the new
    ``arg_dtypes`` field.

    Needs ``arg_dtypes``: with no dtype field to read, a wrongly-typed plane
    cannot be caught at bind."""
    sc = sidecar_of(built["gather"], "gather")
    arrays = {"y": np.zeros(N, dtype=np.float64),
              "where": np.zeros(N, dtype=np.float64),   # wrong: declared int64
              "table": np.arange(N, dtype=np.float64)}
    with pytest.raises(ValueError, match=r"per_sample 'where'"):
        L.check_declared_planes(sc, arrays, N)


def test_the_hawk_consumer_check_accepts_correctly_declared_planes(built):
    """The known-good arm: nothing above raises because every check is wrong,
    they raise because one declaration is violated — this row proves the same
    kernel's CORRECTLY bound planes pass clean."""
    sc = sidecar_of(built["gather"], "gather")
    arrays = {"y": np.zeros(N, dtype=np.float64),
              "where": (np.arange(N) % N).astype(np.int64),
              "table": np.arange(N, dtype=np.float64)}
    L.check_declared_planes(sc, arrays, N)  # must not raise
