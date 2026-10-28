# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra op kinds: one factorisation node, consumers and
projections over it, the type rules, the cost formulas (operations and bytes),
and the general-shape descriptor a sidecar yields.

The formulas are the ones ``aether::linalg`` documents: ``lu`` n^3/3, ``chol``
n^3/6, ``eigh`` S(n) n(n-1)/2 (6n+10), a solve n^2 k.
"""

from __future__ import annotations

import pytest

from hawk import runtime
from hawk.ir import Const, Dispatch, HawkError, Leaf, Select, make
from hawk.ir.cost import linalg_bytes
from hawk.ir.cost import _op_count, _own_cost, _wide
from hawk.ir.linalg import (CONSUMERS, FACTORISATIONS, LINALG_OPS, MAX_DIM, PROJECTIONS, TEAM_MAX_DIM,
                            as_matrix, op_bytes, op_count, sweeps)
from hawk.ir.ops import OP_KINDS
from hawk.types import TensorType


def _mat(n, dtype="f64", name="A"):
    return Leaf("vocab_read", "mat_in", name, TensorType((n, n), dtype))


def _vec(n, dtype="f64", name="b"):
    return Leaf("vocab_read", "vec_in", name, TensorType((n,), dtype))


def _rhs(n, k, dtype="f64"):
    return Leaf("vocab_read", "mat_in", "B", TensorType((n, k), dtype))


def test_the_kinds_are_in_the_vocabulary():
    assert set(LINALG_OPS) <= set(OP_KINDS)
    assert LINALG_OPS == FACTORISATIONS + CONSUMERS + PROJECTIONS
    assert FACTORISATIONS == ("lu", "chol", "eigh", "chol_from", "qr")


@pytest.mark.parametrize("kind,tag", [("lu", "lu_factor"), ("chol", "chol_factor"),
                                      ("eigh", "eigh_factor"), ("chol_from", "chol_factor")])
@pytest.mark.parametrize("dtype", ["f64", "f32"])
def test_a_factorisation_is_one_square_tagged_node(kind, tag, dtype):
    node = make(kind, (_mat(5, dtype),))
    assert node.ttype == TensorType((5, 5), dtype, tag)


def test_the_consumers_and_projections_are_typed_off_the_factor():
    lu, chol, eigh, *_ = (make(k, (_mat(4),)) for k in FACTORISATIONS if k != "qr")
    assert make("lu_solve", (lu, _vec(4))).ttype == TensorType((4,), "f64")
    assert make("lu_solve", (lu, _rhs(4, 3))).ttype == TensorType((4, 3), "f64")
    assert make("cho_solve", (chol, _vec(4))).ttype == TensorType((4,), "f64")
    assert make("cho_solve", (chol, _rhs(4, 2))).ttype == TensorType((4, 2), "f64")
    assert make("lu_info", (lu,)).ttype == TensorType((), "int32")
    assert make("chol_info", (chol,)).ttype == TensorType((), "int32")
    assert make("eigh_info", (eigh,)).ttype == TensorType((), "int32")
    assert make("lu_det", (lu,)).ttype == TensorType((), "f64")
    assert make("chol_factor", (chol,)).ttype == TensorType((4, 4), "f64")
    assert make("eigh_w", (eigh,)).ttype == TensorType((4,), "f64")
    assert make("eigh_v", (eigh,)).ttype == TensorType((4, 4), "f64")
    assert make("lu_det", (make("lu", (_mat(4, "f32"),)),)).ttype == TensorType((), "f32")


def test_the_solves_a_derivative_needs_are_typed_off_their_factor():
    lu, chol = make("lu", (_mat(4),)), make("chol", (_mat(4),))
    assert make("lu_solve_t", (lu, _rhs(4, 3))).ttype == TensorType((4, 3), "f64")
    assert make("tri_lower", (chol, _rhs(4, 4))).ttype == TensorType((4, 4), "f64")
    assert make("tri_upper_t", (chol, _vec(4))).ttype == TensorType((4,), "f64")
    with pytest.raises(HawkError, match="reads the result of 'chol'"):
        make("tri_lower", (lu, _vec(4)))
    with pytest.raises(HawkError, match="reads the result of 'lu'"):
        make("lu_solve_t", (chol, _vec(4)))


def test_info_is_an_int32_plane_that_compares_and_selects():
    info = make("lu_info", (make("lu", (_mat(3),)),))
    ok = make("eq", (info, Const(0, info.ttype)))
    assert ok.ttype == TensorType((), "bool")
    assert make("select", (ok, info, info)).ttype == info.ttype


@pytest.mark.parametrize("bad", [
    TensorType((3,), "f64"), TensorType((), "f64"), TensorType((3, 4), "f64"),
    TensorType((3, 3), "i32"), TensorType((3, 3), "int32")])
def test_only_a_square_float_matrix_is_factored(bad):
    with pytest.raises(HawkError, match="square matrix is factored|f64/f32 matrix is factored"):
        make("lu", (Leaf("vocab_read", "mat_in", "A", bad),))


def test_the_dimension_is_static_and_at_most_the_limit():
    assert MAX_DIM == 32 and TEAM_MAX_DIM == 4096
    assert make("eigh", (_mat(1),)).ttype.shape == (1, 1)
    assert make("eigh", (_mat(32),)).ttype.shape == (32, 32)
    assert make("chol", (_mat(33),)).ttype.shape == (33, 33)         # a team lowers it
    assert make("chol", (_mat(4096),)).ttype.shape == (4096, 4096)
    with pytest.raises(HawkError, match=r"1 <= n <= 4096, got n=4097"):
        make("chol", (_mat(4097),))


def test_a_solve_names_what_it_reads():
    lu, chol = make("lu", (_mat(4),)), make("chol", (_mat(4),))
    with pytest.raises(HawkError, match="reads the result of 'lu'"):
        make("lu_solve", (chol, _vec(4)))
    with pytest.raises(HawkError, match="reads the result of 'chol'"):
        make("cho_solve", (lu, _vec(4)))
    with pytest.raises(HawkError, match="reads the result of 'eigh'"):
        make("eigh_w", (lu,))
    with pytest.raises(HawkError, match="reads the result of 'lu'"):
        make("lu_det", (_mat(4),))


def test_the_right_hand_side_matches_the_factor():
    lu = make("lu", (_mat(4),))
    for rhs in (_vec(3), _rhs(3, 2), _vec(4, "f32"), Leaf("vocab_read", "uniform", "s",
                                                           TensorType((), "f64"))):
        with pytest.raises(HawkError, match="right-hand side"):
            make("lu_solve", (lu, rhs))


def test_a_cholesky_factor_used_as_a_matrix_is_its_lower_projection():
    chol = make("chol", (_mat(3),))
    product = make("mm", (chol, make("transpose", (chol,))))
    assert product.ttype == TensorType((3, 3), "f64")
    left, right = product.operands
    assert left.kind == "chol_factor" and left.operands == (chol,) and left.ttype.tag is None
    assert right.operands[0].kind == "chol_factor" and right.operands[0].operands == (chol,)
    assert make("mv", (chol, _vec(3))).operands[0].kind == "chol_factor"
    assert make("add", (chol, _mat(3))).operands[0].kind == "chol_factor"
    assert as_matrix(chol).kind == "chol_factor"
    assert as_matrix(_mat(3)).kind == "vocab_read"


@pytest.mark.parametrize("kind,what", [("lu", "an LU factor is not a matrix"),
                                       ("eigh", "an eigh factor is not a matrix")])
def test_an_lu_or_eigh_factor_used_as_a_matrix_is_refused_by_name(kind, what):
    node = make(kind, (_mat(3),))
    uses = (lambda: make("mm", (node, _mat(3))), lambda: make("add", (_mat(3), node)),
            lambda: make("chol", (node,)), lambda: as_matrix(node))
    for use in uses:
        with pytest.raises(HawkError, match=what):
            use()


def test_a_factor_is_read_unprojected_by_its_consumers():
    chol = make("chol", (_mat(3),))
    solve = make("cho_solve", (chol, _vec(3)))
    assert solve.operands[0] is chol
    assert make("chol_info", (chol,)).operands == (chol,)


# --- cost ---------------------------------------------------------------------

def _count(node):
    return op_count(node)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 17, 32])
def test_factorisation_operation_counts_are_the_formulas(n):
    assert _count(make("lu", (_mat(n),))) == pytest.approx(n ** 3 / 3)
    assert _count(make("chol", (_mat(n),))) == pytest.approx(n ** 3 / 6)
    assert _count(make("eigh", (_mat(n),))) == sweeps(n) * n * (n - 1) / 2 * (6 * n + 10)


def test_the_sweep_table_is_aethers():
    assert [sweeps(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)] == [
        2, 2, 6, 6, 8, 8, 12, 12, 15, 15]


@pytest.mark.parametrize("n,k", [(1, 1), (4, 1), (4, 3), (32, 5)])
def test_solve_operation_counts_are_n_squared_k(n, k):
    lu, chol = make("lu", (_mat(n),)), make("chol", (_mat(n),))
    rhs = _vec(n) if k == 1 else _rhs(n, k)
    assert _count(make("lu_solve", (lu, rhs))) == n * n * k
    assert _count(make("cho_solve", (chol, rhs))) == n * n * k


@pytest.mark.parametrize("n,k", [(1, 1), (4, 4), (32, 32)])
def test_the_derivative_solves_cost_their_triangular_halves(n, k):
    lu, chol = make("lu", (_mat(n),)), make("chol", (_mat(n),))
    rhs = _rhs(n, k)
    assert _count(make("lu_solve_t", (lu, rhs))) == n * n * k
    assert _count(make("tri_lower", (chol, rhs))) == n * n * k / 2
    assert _count(make("tri_upper_t", (chol, rhs))) == n * n * k / 2


def test_determinant_is_n_multiplies_and_projections_are_free():
    lu, chol, eigh, *_ = (make(k, (_mat(6),)) for k in FACTORISATIONS if k != "qr")
    assert _count(make("lu_det", (lu,))) == 6
    for node in (make("lu_info", (lu,)), make("chol_info", (chol,)), make("chol_factor", (chol,)),
                 make("eigh_w", (eigh,)), make("eigh_v", (eigh,)), make("eigh_info", (eigh,))):
        assert _count(node) == 0
        assert _own_cost(node) == [0.0] * 6      # a free kind adds no class count


def test_cost_lands_in_the_arithmetic_class_of_the_working_width():
    wide, narrow = make("lu", (_mat(4),)), make("lu", (_mat(4, "f32"),))
    assert _op_count(wide) == ("arith", pytest.approx(64 / 3))
    assert _wide(wide) and not _wide(narrow)
    assert _own_cost(wide)[1] == pytest.approx(64 / 3) and _own_cost(wide)[0] == 0
    assert _own_cost(narrow)[0] == pytest.approx(64 / 3) and _own_cost(narrow)[1] == 0


@pytest.mark.parametrize("n,w,dtype", [(1, 8, "f64"), (5, 8, "f64"), (32, 4, "f32"), (7, 4, "f32")])
def test_storage_bytes_per_kind(n, w, dtype):
    lu, chol, eigh, *_ = (make(k, (_mat(n, dtype),)) for k in FACTORISATIONS if k != "qr")
    assert op_bytes(lu) == n * n * w + 4 * n + 4
    assert op_bytes(chol) == n * n * w + 4
    assert op_bytes(eigh) == (2 * n * n + n) * w + 4
    rhs = _rhs(n, 2, dtype)
    assert op_bytes(make("lu_solve", (lu, rhs))) == n * 2 * w
    assert op_bytes(make("cho_solve", (chol, _vec(n, dtype)))) == n * w
    assert op_bytes(make("lu_det", (lu,))) == w
    assert op_bytes(make("lu_info", (lu,))) == 4
    assert op_bytes(make("chol_factor", (chol,))) == n * n * w
    assert op_bytes(make("eigh_w", (eigh,))) == n * w


def test_the_stack_of_a_kernel_is_the_sum_over_its_nodes():
    lu = make("lu", (_mat(32),))
    solve = make("lu_solve", (lu, _vec(32)))
    assert linalg_bytes([lu, solve, _mat(32)]) == 32 * 32 * 8 + 4 * 32 + 4 + 32 * 8
    assert linalg_bytes([]) == 0
    # the n=32 f64 Cholesky alone is above a 4 KB stack
    assert op_bytes(make("chol", (_mat(32),))) > 4096


# --- the sidecar descriptor reads declared shapes ----------------------------------

def _sidecar(**kw):
    base = {"scalar_type": "float64", "mutables": [], "params": [], "arg_dtypes": {}}
    base.update(kw)
    return base


@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (7, 7), (1, 4), (32, 32)])
def test_a_matrix_of_any_extent_gets_its_declared_kind(shape):
    sidecar = _sidecar(arg_spec=[["out", "W"], ["mat_in", "A"], ["vec_in", "v"]],
                       mat_shapes={"A": list(shape)}, vec_widths={"v": 6},
                       mutables=[{"name": "W", "dtype": "matrix", "width": shape[0] * shape[1],
                                  "shape": list(shape), "default": None}])
    assert runtime.descriptor_for_sidecar(sidecar) == ["gref", "gref", "gref"]


def test_a_vector_output_reads_its_width_and_a_missing_extent_is_refused():
    ok = _sidecar(arg_spec=[["out", "y"]],
                  mutables=[{"name": "y", "dtype": "vector", "width": 5, "default": None}])
    assert runtime.descriptor_for_sidecar(ok) == ["gref"]
    for sidecar, what in (
            (_sidecar(arg_spec=[["mat_in", "A"]]), "mat_shapes"),
            (_sidecar(arg_spec=[["vec_in", "v"]]), "vec_widths"),
            (_sidecar(arg_spec=[["out", "W"]],
                      mutables=[{"name": "W", "dtype": "matrix", "width": 6, "default": None}]),
             "mutables.shape")):
        with pytest.raises(HawkError, match=what):
            runtime.descriptor_for_sidecar(sidecar)


# --- a factorisation is never an arm ---------------------------------------------------

@pytest.mark.parametrize("kind", FACTORISATIONS)
def test_select_and_dispatch_refuse_factorisation_arms(kind):
    one, other = make(kind, (_mat(3),)), make(kind, (_mat(3, name="C"),))
    flag = Leaf("vocab_read", "uniform", "flag", TensorType((), "bool"))
    index = Leaf("vocab_read", "uniform", "k", TensorType((), "i32"))
    with pytest.raises(HawkError, match="factorisations, which are read through"):
        Select(flag, one, other, one.ttype)
    with pytest.raises(HawkError, match="factorisations, which are read through"):
        Dispatch(index, (one, other), "switch", one.ttype)
    # the fields they read are values like any other
    field = {"lu": "lu_det", "chol": "chol_info", "eigh": "eigh_info", "chol_from": "chol_info",
             "qr": "qr_info"}[kind]
    a, b = make(field, (one,)), make(field, (other,))
    assert make("select", (flag, a, b)).ttype == a.ttype
