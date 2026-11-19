# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra kinds the library types: one factorisation node, consumers and projections over it, and every refusal by name.

Each row builds nodes through ``hawk._ir`` and reads the type or the refusal the library gives. The oracle of a type is the
shape and dtype the factor states: a factor is one square tagged matrix, a solve has the right-hand side's shape, an
``info`` is one ``int32``, a determinant is one scalar of the working dtype, an ``eigh`` has ``w`` of length ``n`` and ``v``
of shape ``(n, n)``.
"""

from __future__ import annotations

import pytest

import hawk.math as m
from hawk import HawkError, Index, Matrix, Mutable, Vector, _ir
from hawk import kernel as _kernel
from hawk._ir import TensorType

CONTEXT = _ir.new_context()
FACTORISATIONS = ("lu", "chol", "eigh", "chol_from")
FACTOR_TAG = {
    "lu": "lu_factor",
    "chol": "chol_factor",
    "eigh": "eigh_factor",
    "chol_from": "chol_factor",
}


def mat(n, dtype="f64", name="A"):
    return _ir.leaf(CONTEXT, "vocab_read", "mat_in", name, ((n, n), dtype, None))


def vec(n, dtype="f64", name="b"):
    return _ir.leaf(CONTEXT, "vocab_read", "vec_in", name, ((n,), dtype, None))


def rhs(n, k, dtype="f64"):
    return _ir.leaf(CONTEXT, "vocab_read", "mat_in", "B", ((n, k), dtype, None))


def typed(node):
    return _ir.type_of(node)


def test_the_factorisation_kinds_are_in_the_vocabulary():
    assert set(_ir.LINALG_OPS) <= set(_ir.OP_KINDS)
    assert {"lu", "chol", "eigh", "chol_from", "qr"} <= set(_ir.LINALG_OPS)


@pytest.mark.parametrize("kind", FACTORISATIONS)
@pytest.mark.parametrize("dtype", ["f64", "f32"])
def test_a_factorisation_is_one_square_tagged_node(kind, dtype):
    assert typed(_ir.make(kind, (mat(5, dtype),))) == TensorType(
        (5, 5), dtype, FACTOR_TAG[kind]
    )


def test_the_consumers_and_projections_are_typed_off_the_factor():
    lu, chol, eigh = (_ir.make(kind, (mat(4),)) for kind in ("lu", "chol", "eigh"))
    expected = [
        (("lu_solve", (lu, vec(4))), ((4,), "f64")),
        (("lu_solve", (lu, rhs(4, 3))), ((4, 3), "f64")),
        (("cho_solve", (chol, vec(4))), ((4,), "f64")),
        (("cho_solve", (chol, rhs(4, 2))), ((4, 2), "f64")),
        (("lu_info", (lu,)), ((), "int32")),
        (("chol_info", (chol,)), ((), "int32")),
        (("eigh_info", (eigh,)), ((), "int32")),
        (("lu_det", (lu,)), ((), "f64")),
        (("chol_factor", (chol,)), ((4, 4), "f64")),
        (("eigh_w", (eigh,)), ((4,), "f64")),
        (("eigh_v", (eigh,)), ((4, 4), "f64")),
    ]
    for (kind, operands), (shape, dtype) in expected:
        assert typed(_ir.make(kind, operands)) == TensorType(shape, dtype, None), kind
    assert typed(_ir.make("lu_det", (_ir.make("lu", (mat(4, "f32"),)),))).dtype == "f32"


def test_the_solves_a_derivative_needs_are_typed_off_their_factor():
    lu, chol = _ir.make("lu", (mat(4),)), _ir.make("chol", (mat(4),))
    assert typed(_ir.make("lu_solve_t", (lu, rhs(4, 3)))) == TensorType(
        (4, 3), "f64", None
    )
    assert typed(_ir.make("tri_lower", (chol, rhs(4, 4)))) == TensorType(
        (4, 4), "f64", None
    )
    assert typed(_ir.make("tri_upper_t", (chol, vec(4)))) == TensorType(
        (4,), "f64", None
    )
    with pytest.raises(HawkError, match="reads the result of 'chol'"):
        _ir.make("tri_lower", (lu, vec(4)))
    with pytest.raises(HawkError, match="reads the result of 'lu'"):
        _ir.make("lu_solve_t", (chol, vec(4)))


def test_info_is_an_int32_plane_that_compares_and_selects():
    info = _ir.make("lu_info", (_ir.make("lu", (mat(3),)),))
    zero = _ir.const(CONTEXT, 0, ((), "int32", None))
    ok = _ir.make("eq", (info, zero))
    assert typed(ok).dtype == "bool"
    assert typed(_ir.make("select", (ok, info, info))) == typed(info)


@pytest.mark.parametrize(
    "bad",
    [((3,), "f64"), ((), "f64"), ((3, 4), "f64"), ((3, 3), "i32"), ((3, 3), "int32")],
)
def test_only_a_square_float_matrix_is_factored(bad):
    shape, dtype = bad
    leaf = _ir.leaf(CONTEXT, "vocab_read", "mat_in", "A", (shape, dtype, None))
    with pytest.raises(
        HawkError, match="square matrix is factored|f64/f32 matrix is factored"
    ):
        _ir.make("lu", (leaf,))


@pytest.mark.parametrize("n", [1, 32, 33, 4096])
def test_the_dimension_of_a_factor_is_the_static_extent_of_its_matrix(n):
    assert typed(_ir.make("chol", (mat(n),))).shape == (n, n)


def test_a_solve_names_what_it_reads():
    lu, chol = _ir.make("lu", (mat(4),)), _ir.make("chol", (mat(4),))
    for kind, factor, reads in (
        ("lu_solve", chol, "lu"),
        ("cho_solve", lu, "chol"),
        ("eigh_w", lu, "eigh"),
    ):
        operands = (factor,) if kind == "eigh_w" else (factor, vec(4))
        with pytest.raises(HawkError, match=f"reads the result of '{reads}'"):
            _ir.make(kind, operands)
    with pytest.raises(HawkError, match="reads the result of 'lu'"):
        _ir.make("lu_det", (mat(4),))


def test_the_right_hand_side_matches_the_factor():
    lu = _ir.make("lu", (mat(4),))
    scalar = _ir.leaf(CONTEXT, "vocab_read", "uniform", "s", ((), "f64", None))
    for wrong in (vec(3), rhs(3, 2), vec(4, "f32"), scalar):
        with pytest.raises(HawkError, match="right-hand side"):
            _ir.make("lu_solve", (lu, wrong))


def test_a_cholesky_factor_used_as_a_matrix_is_its_lower_projection():
    chol = _ir.make("chol", (mat(3),))
    product = _ir.make("mm", (chol, _ir.make("transpose", (chol,))))
    assert typed(product) == TensorType((3, 3), "f64", None)
    left = product.operands[0]
    assert _ir.is_a(left, "op") and left.operands == (chol,) and typed(left).tag is None
    assert _ir.as_matrix(chol).operands == (chol,)
    plain = mat(3)
    assert _ir.as_matrix(plain) is plain or _ir.as_matrix(plain) == plain


@pytest.mark.parametrize(
    "kind,what",
    [
        ("lu", "an LU factor is not a matrix"),
        ("eigh", "an eigh factor is not a matrix"),
    ],
)
def test_an_lu_or_eigh_factor_used_as_a_matrix_is_refused_by_name(kind, what):
    node = _ir.make(kind, (mat(3),))
    uses = (
        lambda: _ir.make("mm", (node, mat(3))),
        lambda: _ir.make("add", (mat(3), node)),
        lambda: _ir.make("chol", (node,)),
        lambda: _ir.as_matrix(node),
    )
    for use in uses:
        with pytest.raises(HawkError, match=what):
            use()


def test_a_factor_is_read_unprojected_by_its_consumers():
    chol = _ir.make("chol", (mat(3),))
    assert _ir.make("cho_solve", (chol, vec(3))).operands[0] == chol
    assert _ir.make("chol_info", (chol,)).operands == (chol,)


@pytest.mark.parametrize("kind", FACTORISATIONS)
def test_a_select_refuses_factorisation_arms_and_accepts_the_fields_they_read(kind):
    one, other = _ir.make(kind, (mat(3),)), _ir.make(kind, (mat(3, name="C"),))
    flag = _ir.leaf(CONTEXT, "vocab_read", "uniform", "flag", ((), "bool", None))
    with pytest.raises(HawkError, match="factorisations, which are read through"):
        _ir.make_node("Select", typed(one), [flag, one, other])
    field = {
        "lu": "lu_det",
        "chol": "chol_info",
        "eigh": "eigh_info",
        "chol_from": "chol_info",
    }[kind]
    a, b = _ir.make(field, (one,)), _ir.make(field, (other,))
    assert typed(_ir.make("select", (flag, a, b))) == typed(a)


def test_a_dispatch_refuses_factorisation_arms():
    with pytest.raises(HawkError, match="factorisations, which are read through"):

        @_kernel
        def arms(k: Index, A: Matrix[3, 3], B: Matrix[3, 3], y: Mutable[Vector[3]]):
            first, _ = m.lu(A)
            second, _ = m.lu(B)
            y = m.dispatch(k, [first, second])
