# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plane states its axes and a read or a store takes one index per axis.

The traced kernel holds ONE ``At`` with an index operand per axis and an edge rule, no stride constant and no clamp
operation. The values are checked against NumPy indexing, which is independent of the code under test: ``np.clip``
for ``clamp``, the modulo for ``wrap``, a zero outside for ``zero``, plain indexing for ``unchecked``, and
``np.add.at`` for an accumulate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Index, Mutable, Scalar, Table, Vector, _ir
from hawk.diff import jvp, vjp
from hawk.errors import HawkError

F64 = _ir.TensorType((), "f64")
I32 = _ir.TensorType((), "i32")
CTX = _ir.new_context()
SINK_AXES = (("bin", 3), ("slot", 4))
EDGES = ("clamp", "wrap", "zero", "unchecked")
RNG = np.random.default_rng(20261121)


@hawk.kernel
def _read_two_axes(p: Table["row", "col":4], i: Index, k: Index, y: Mutable[Scalar]):  # noqa: F821 axis labels
    y = p[i, k]


@hawk.kernel
def _read_named(p: Table["row", "col":4], i: Index, k: Index, y: Mutable[Scalar]):  # noqa: F821 axis labels
    y = p.at(row=i, col=k)


@hawk.kernel
def _store_two_axes(i: Index, k: Index, v: Scalar, y: Mutable[Scalar, "bin":3, "slot":4]):  # noqa: F821 axis labels
    hawk.scatter(y)[i, k] += v


def _plane(name, axes):
    return _ir.leaf(CTX, "table_read", "lookup", name, F64, axes=axes)


def _index_leaves(count):
    return [_ir.leaf(CTX, "vocab_read", "plane_in", f"i{n}", I32) for n in range(count)]


def _read(axes, edge):
    """``out = p[i0, i1, ...]`` over a plane with ``axes`` under ``edge``; the plane and the index leaves."""
    plane, indices = _plane("p", axes), _index_leaves(len(axes))
    read = _ir.make_node("At", F64, [plane, *indices], edge=edge)
    return (_ir.make_node("Assign", F64, [read], name="out"),), (plane, indices)


def _expected(table, where, edge):
    """The NumPy oracle of one read of ``table`` at ``where``."""
    extents = table.shape
    if edge == "clamp":
        return table[tuple(np.clip(w, 0, e - 1) for w, e in zip(where, extents))]
    if edge == "wrap":
        return table[tuple(w % e for w, e in zip(where, extents))]
    if edge == "zero" and any(not 0 <= w < e for w, e in zip(where, extents)):
        return 0.0
    return table[tuple(where)]


def _draw(shape, edge):
    """A random index tuple: any integer for an edge rule, in range for ``unchecked``."""
    if edge == "unchecked":
        return [int(RNG.integers(0, e)) for e in shape]
    return [int(RNG.integers(-2 * e, 3 * e)) for e in shape]


def _env(table, where):
    return {"p": table, **{f"i{n}": w for n, w in enumerate(where)}}


def _elements(kernel):
    seq, _ = _ir.canonical_nodes(kernel.sinks)
    return seq


@pytest.mark.parametrize("kernel", [_read_two_axes, _read_named], ids=["subscript", "named"])
def test_a_two_axis_read_is_one_at_with_two_indices_and_no_stride_or_clamp(kernel):
    nodes = _elements(kernel)
    reads = [n for n in nodes if _ir.is_a(n, "at")]
    assert len(reads) == 1 and len(reads[0].operands) == 3
    assert reads[0].field_text("edge") == "clamp"
    assert _ir.field_pairs(reads[0].operands[0], "axes") == (("row", None), ("col", 4))
    assert not [n for n in nodes if _ir.is_a(n, "role_const")], "no stride constant"
    assert not [n for n in nodes if _ir.is_a(n, "op") and n.field_text("kind") in ("clamp_index", "mul")]


def test_the_typing_rule_refuses_a_wrong_number_of_indices_naming_the_plane_and_the_axes():
    plane, (i, k) = _plane("p", (("row", None), ("col", 4))), _index_leaves(2)
    with pytest.raises(HawkError, match=r'plane "p" has 2 axes .*"row".*"col".*got 1'):
        _ir.make_node("At", F64, [plane, i], edge="clamp")
    with pytest.raises(HawkError, match=r'plane "p" has 2 axes .*got 3'):
        _ir.make_node("At", F64, [plane, i, k, k], edge="clamp")


def test_the_typing_rule_refuses_a_real_valued_index_naming_the_plane_and_the_axis():
    plane, (i, _) = _plane("p", (("row", None), ("col", 4))), _index_leaves(2)
    real = _ir.leaf(CTX, "vocab_read", "plane_in", "x", F64)
    with pytest.raises(HawkError, match=r'plane "p" axis "col": the index must be a rank-0 integer'):
        _ir.make_node("At", F64, [plane, i, real], edge="clamp")


def test_the_typing_rule_refuses_an_unknown_edge_rule_and_an_axis_the_plane_lacks():
    plane, (i,) = _plane("q", ()), _index_leaves(1)
    with pytest.raises(HawkError, match="edge rule 'mirror'"):
        _ir.make_node("At", F64, [plane, i], edge="mirror")
    with pytest.raises(HawkError, match='plane "q" has 1 axes, there is no axis 1'):
        _ir.make_node("AxisExtent", I32, [plane], axis=1)


@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("axes", [(("a", None), ("b", 4)), (("a", 3), ("b", None), ("c", 2))],
                         ids=["two_axes", "three_axes"])
def test_a_read_equals_numpy_indexing_under_each_edge_rule(axes, edge):
    sinks, _ = _read(axes, edge)
    table = RNG.normal(size=tuple(3 if e is None else e for _, e in axes))
    for _ in range(40):
        where = _draw(table.shape, edge)
        got = evaluate(sinks, _env(table, where))["out"]
        assert got == _expected(table, where, edge), (where, edge)


def _store(op, edge=None):
    value = _ir.leaf(CTX, "vocab_read", "plane_in", "v", F64)
    i, k = _index_leaves(2)
    fields = {"name": "y", "axes": SINK_AXES, "op": op, **({"reduced": None} if op == "add" else {})}
    if edge is not None:
        fields["edge"] = edge
    return (_ir.make_node("Update", F64, [value, i, k], **fields),)


@pytest.mark.parametrize("edge", [None, "zero"], ids=["unchecked_by_default", "zero"])
def test_an_accumulate_at_two_indices_equals_add_at(edge):
    shape, sinks = (3, 4), _store("add", edge)
    got, expected = np.zeros(shape), np.zeros(shape)
    for _ in range(60):
        where, value = _draw(shape, edge or "unchecked"), RNG.normal()
        got += evaluate(sinks, {"i0": where[0], "i1": where[1], "v": value}, shapes={"y": shape})["y"]
        if all(0 <= w < e for w, e in zip(where, shape)):
            np.add.at(expected, tuple(where), value)
    assert np.allclose(got, expected)


def test_a_store_at_two_indices_equals_assignment():
    shape, sinks = (3, 4), _store("store")
    got, expected = np.zeros(shape), np.zeros(shape)
    for _ in range(60):
        where, value = _draw(shape, "unchecked"), RNG.normal()
        lane = evaluate(sinks, {"i0": where[0], "i1": where[1], "v": value}, shapes={"y": shape})["y"]
        got[tuple(where)] = lane[tuple(where)]
        expected[tuple(where)] = value
    assert np.array_equal(got, expected)


def test_a_traced_keyed_store_is_an_accumulate_with_one_index_per_axis():
    stores = [n for n in _elements(_store_two_axes)
              if _ir.is_accumulate(n) and n.field_text("name") == "y"]
    assert len(stores) == 1 and len(stores[0].operands) == 3
    assert stores[0].field_text("edge") == "unchecked"


@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("axes", [(("a", None), ("b", 4)), (("a", 3), ("b", None), ("c", 2))],
                         ids=["two_axes", "three_axes"])
def test_the_tangent_is_read_and_the_adjoint_scattered_at_the_same_indices(axes, edge):
    sinks, _ = _read(axes, edge)
    table = RNG.normal(size=tuple(3 if e is None else e for _, e in axes))
    tangent = RNG.normal(size=table.shape)
    forward, reverse = jvp(sinks), vjp(sinks)
    for _ in range(20):
        where = _draw(table.shape, edge)
        env = {**_env(table, where), "dot_p": tangent}
        assert evaluate(forward, env)["dot_out"] == _expected(tangent, where, edge)
        seed = float(RNG.normal())
        got = evaluate(reverse, {**_env(table, where), "bar_out": seed}, shapes={"bar_p": table.shape})["bar_p"]
        expected = np.zeros(table.shape)
        outside = any(not 0 <= w < e for w, e in zip(where, table.shape))
        if not (edge == "zero" and outside):
            position = {"clamp": tuple(np.clip(w, 0, e - 1) for w, e in zip(where, table.shape)),
                        "wrap": tuple(w % e for w, e in zip(where, table.shape))}.get(edge, tuple(where))
            np.add.at(expected, position, seed)
        assert np.allclose(got, expected), (where, edge)


def test_the_digest_differs_when_an_index_order_is_swapped_and_not_when_rebuilt():
    plane, (i, k) = _plane("p", (("a", None), ("b", 4))), _index_leaves(2)

    def digest(first, second):
        read = _ir.make_node("At", F64, [plane, first, second], edge="clamp")
        return _ir.digest(_ir.canonical((_ir.make_node("Assign", F64, [read], name="out"),)))

    assert digest(i, k) == digest(i, k)
    assert digest(i, k) != digest(k, i)


def test_the_digest_covers_the_axes_and_the_edge_rule():
    def digest(axes, edge):
        plane, (i, k) = _plane("p", axes), _index_leaves(2)
        read = _ir.make_node("At", F64, [plane, i, k], edge=edge)
        return _ir.digest(_ir.canonical((_ir.make_node("Assign", F64, [read], name="out"),)))

    base = digest((("a", None), ("b", 4)), "clamp")
    assert digest((("a", None), ("b", 5)), "clamp") != base
    assert digest((("a", None), ("b", 4)), "wrap") != base
    assert digest((("a", 2), ("b", 4)), "clamp") != base


def test_every_edge_rule_pairs_with_every_index_order():
    """The digests of all (edge, order) pairs of one read are pairwise distinct."""
    plane, (i, k) = _plane("p", (("a", None), ("b", 4))), _index_leaves(2)
    seen = set()
    for edge, order in itertools.product(EDGES, ((i, k), (k, i))):
        read = _ir.make_node("At", F64, [plane, *order], edge=edge)
        seen.add(_ir.digest(_ir.canonical((_ir.make_node("Assign", F64, [read], name="out"),))))
    assert len(seen) == 8


@hawk.kernel
def _vector_sum(i: Index, k: Index, v: Vector[2], y: Mutable[Vector[2], "bin":3, "slot":4]):  # noqa: F821 axis labels
    hawk.scatter(y)[i, k] += v


def test_a_store_with_a_wrong_index_count_is_refused_naming_the_sink_and_the_axes():
    value, (i, k) = _ir.leaf(CTX, "vocab_read", "plane_in", "v", F64), _index_leaves(2)
    with pytest.raises(HawkError, match=r'sink "y" has 2 axes .*"bin".*"slot".*got 1'):
        _ir.make_node("Update", F64, [value, i], name="y", op="add", reduced=None, axes=SINK_AXES)
    real = _ir.leaf(CTX, "vocab_read", "plane_in", "x", F64)
    with pytest.raises(HawkError, match=r'sink "y" axis "slot": the index must be a rank-0 integer'):
        _ir.make_node("Update", F64, [value, i, real], name="y", op="store", axes=SINK_AXES)


def test_the_reverse_of_a_two_index_store_reads_the_adjoint_by_the_real_axes():
    adjoint = [n for n in vjp(_store("add")) for n in _ir.canonical_nodes((n,))[0] if _ir.is_a(n, "at")]
    assert len(adjoint) == 1
    assert _ir.field_pairs(adjoint[0].operands[0], "axes") == SINK_AXES


def test_a_keyed_vector_sum_into_a_two_axis_sink_equals_add_at():
    shape = (2, 3, 4)
    sinks = _vector_sum.sinks
    stores = [n for n in _elements(_vector_sum) if _ir.is_accumulate(n) and n.field_text("name") == "y"]
    assert len(stores) == 1 and len(stores[0].operands) == 3
    assert _ir.type_of(stores[0]).shape == (2,)
    assert _ir.field_pairs(stores[0], "axes") == (("bin", 3), ("slot", 4))
    got, expected = np.zeros(shape), np.zeros(shape)
    for _ in range(40):
        where, payload = [int(RNG.integers(0, 3)), int(RNG.integers(0, 4))], RNG.normal(size=2)
        got += evaluate(sinks, {"i": where[0], "k": where[1], "v": payload}, shapes={"y": shape, "y_c": shape})["y"]
        np.add.at(expected, (slice(None), where[0], where[1]), payload)
    assert np.allclose(got, expected)


# ---- a plane with axes holding a vector: read at one index per axis under each edge rule ---------------------

VALUE = _ir.TensorType((3,), "f64")
VECTOR_AXES = (("row", 4), ("col", 5))
FIELD = RNG.normal(size=(3, 4, 5))     # bound (*value shape, *axes)


def _vector_read(edge):
    """``y = p[i, k]`` of a plane of vectors with two axes under ``edge``: the sinks."""
    plane = _ir.leaf(CTX, "table_read", "lookup", "p", VALUE, axes=VECTOR_AXES)
    i, k = _index_leaves(2)
    read = _ir.make_node("At", VALUE, [plane, i, k], edge=edge)
    return (_ir.make_node("Assign", VALUE, [read], name="y"),)


def _vector_expected(table, where, edge):
    """The NumPy oracle: the vector at the position the edge rule reaches, zeros outside under ``zero``."""
    extents = table.shape[1:]
    if edge == "clamp":
        where = [np.clip(w, 0, e - 1) for w, e in zip(where, extents)]
    elif edge == "wrap":
        where = [w % e for w, e in zip(where, extents)]
    elif edge == "zero" and any(not 0 <= w < e for w, e in zip(where, extents)):
        return np.zeros(table.shape[0])
    return table[(slice(None), *where)]


@pytest.mark.parametrize("edge", EDGES)
def test_a_vector_read_from_a_plane_with_two_axes_equals_numpy_under_each_edge_rule(edge):
    sinks = _vector_read(edge)
    for _ in range(30):
        where = _draw(FIELD.shape[1:], edge)
        got = evaluate(sinks, _env(FIELD, where))["y"]
        np.testing.assert_array_equal(got, _vector_expected(FIELD, where, edge))


@pytest.mark.parametrize("edge", EDGES)
def test_the_derivatives_of_a_vector_read_from_a_plane_with_two_axes_equal_numpy(edge):
    sinks = _vector_read(edge)
    tangent = RNG.normal(size=FIELD.shape)
    for _ in range(20):
        where = _draw(FIELD.shape[1:], edge)
        env = _env(FIELD, where)
        forward = evaluate(jvp(sinks), {**env, "dot_p": tangent})["dot_y"]
        np.testing.assert_array_equal(forward, _vector_expected(tangent, where, edge))
        seed = RNG.normal(size=3)
        reverse = evaluate(vjp(sinks), {**env, "bar_y": seed}, shapes={"bar_p": FIELD.shape})["bar_p"]
        want = np.zeros(FIELD.shape)
        outside = any(not 0 <= w < e for w, e in zip(where, FIELD.shape[1:]))
        if not (edge == "zero" and outside):
            position = {"clamp": [np.clip(w, 0, e - 1) for w, e in zip(where, FIELD.shape[1:])],
                        "wrap": [w % e for w, e in zip(where, FIELD.shape[1:])]}.get(edge, where)
            want[(slice(None), *position)] += seed
        np.testing.assert_allclose(reverse, want, rtol=1e-14, atol=1e-15)


@pytest.fixture(scope="module")
def host_reads(tmp_path_factory, cache_dir):
    """Each edge rule's read built once for the host."""
    root, loaded = tmp_path_factory.mktemp("vector_reads"), {}
    for edge in EDGES:
        directory = root / edge
        directory.mkdir()
        hawk.build([hawk.Kernel(f"read_{edge}", _vector_read(edge))], directory, targets=("host",),
                   cache_dir=cache_dir)
        loaded[edge] = hawk.load(directory, f"read_{edge}")
    return loaded


@pytest.mark.parametrize("edge", EDGES)
def test_a_vector_read_from_a_plane_with_two_axes_equals_numpy_on_the_host(host_reads, edge):
    lanes = 40
    where = np.array([_draw(FIELD.shape[1:], edge) for _ in range(lanes)], dtype=np.int32)
    out = np.zeros((3, lanes))
    hawk.run(host_reads[edge], p=FIELD, i0=where[:, 0].copy(), i1=where[:, 1].copy(), y=out)
    for lane in range(lanes):
        np.testing.assert_array_equal(out[:, lane], _vector_expected(FIELD, list(where[lane]), edge))


@hawk.kernel
def _declared_vector_read(
    p: Table[Vector[3], "row":4, "col":5], i: Index, k: Index, y: Mutable[Vector[3]]  # noqa: F821 axis labels
):
    y = p[i, k] * 2.0


def test_a_declared_table_of_vectors_is_read_at_two_indices_on_the_host(tmp_path, cache_dir):
    lanes = 30
    where = np.array([_draw(FIELD.shape[1:], "clamp") for _ in range(lanes)], dtype=np.int32)
    bundle = hawk.build([_declared_vector_read], tmp_path, targets=("host",), cache_dir=cache_dir)
    out = np.zeros((3, lanes))
    hawk.run(hawk.load(tmp_path, "_declared_vector_read"), p=FIELD, i=where[:, 0].copy(),
             k=where[:, 1].copy(), y=out)
    del bundle
    for lane in range(lanes):
        np.testing.assert_allclose(out[:, lane], 2.0 * _vector_expected(FIELD, list(where[lane]), "clamp"),
                                   rtol=1e-14)


def test_a_zero_edge_read_of_another_lanes_vector_equals_numpy_on_the_host(tmp_path, cache_dir):
    plane = _ir.leaf(CTX, "vocab_read", "plane_in", "p", VALUE)
    (i,) = _index_leaves(1)
    read = _ir.make_node("At", VALUE, [plane, i], edge="zero")
    sinks = (_ir.make_node("Assign", VALUE, [read], name="y"),)
    table = RNG.normal(size=(3, 13))
    where = np.arange(-3, 10, dtype=np.int32)
    out = np.zeros((3, len(where)))
    hawk.build([hawk.Kernel("zero_flat", sinks)], tmp_path, targets=("host",), cache_dir=cache_dir)
    hawk.run(hawk.load(tmp_path, "zero_flat"), p=table, i0=where, y=out)
    want = np.array([table[:, w] if 0 <= w < 13 else np.zeros(3) for w in where]).T
    np.testing.assert_array_equal(out, want)


# ---- a flat lookup plane holding a vector: the one-runtime-axis case of the read above ----------------------------


@hawk.kernel
def _flat_vector_read(p: Table[Vector[3]], i: Index, y: Mutable[Vector[3]]):
    y = p.at(i) * 2.0


def _flat_expected(table, where, edge):
    """The NumPy oracle over the one runtime axis of ``table`` (bound ``(3, n)``)."""
    return _vector_expected(table, [where], edge)


def test_a_flat_lookup_of_vectors_is_read_at_a_traced_index_on_the_host_under_each_edge_rule(
    tmp_path, cache_dir
):
    table = RNG.normal(size=(3, 9))
    for edge in ("clamp", "zero", "unchecked"):
        plane = _ir.leaf(CTX, "table_read", "lookup", "p", VALUE)
        (i,) = _index_leaves(1)
        read = _ir.make_node("At", VALUE, [plane, i], edge=edge)
        sinks = (_ir.make_node("Assign", VALUE, [read], name="y"),)
        where = np.array([_draw((9,), edge)[0] for _ in range(30)], dtype=np.int32)
        out = np.zeros((3, len(where)))
        directory = tmp_path / edge
        directory.mkdir()
        hawk.build([hawk.Kernel(f"flat_{edge}", sinks)], directory, targets=("host",), cache_dir=cache_dir)
        hawk.run(hawk.load(directory, f"flat_{edge}"), p=table, i0=where, y=out)
        for lane, w in enumerate(where):
            np.testing.assert_array_equal(out[:, lane], _flat_expected(table, int(w), edge))


def test_the_forward_derivative_of_a_flat_lookup_vector_read_equals_numpy():
    plane = _ir.leaf(CTX, "table_read", "lookup", "p", VALUE)
    (i,) = _index_leaves(1)
    sinks = (_ir.make_node("Assign", VALUE, [_ir.make_node("At", VALUE, [plane, i], edge="clamp")], name="y"),)
    table, tangent = RNG.normal(size=(3, 9)), RNG.normal(size=(3, 9))
    for where in (-2, 0, 4, 8, 11):
        got = evaluate(jvp(sinks), {"p": table, "i0": where, "dot_p": tangent})["dot_y"]
        np.testing.assert_array_equal(got, _flat_expected(tangent, where, "clamp"))


def test_the_device_text_of_a_zero_edge_vector_read_names_the_item_view_and_the_zero_item():
    from hawk.artifact import emit_source

    text = emit_source(hawk.Kernel("read_zero", _vector_read("zero")), targets=("cuda",)).sources["cuda"].text
    assert "hawk_abi::view_item<" in text and "aether::constant<aether::extents<" in text


@pytest.mark.gpu
@pytest.mark.parametrize("edge", EDGES)
def test_a_vector_read_from_a_plane_with_two_axes_equals_numpy_on_the_device(tmp_path, cache_dir, edge):
    """Oracle: ``_vector_expected``, the NumPy indexing under the edge rule (zeros outside under ``zero``)."""
    import cupy as cp
    from conftest import _launch

    from hawk.artifact import build_bundle

    lanes = 40
    where = np.array([_draw(FIELD.shape[1:], edge) for _ in range(lanes)], dtype=np.int64)
    bundle = build_bundle([hawk.Kernel(f"read_{edge}", _vector_read(edge))], tmp_path, targets=("cuda", "host"),
                          cache_dir=cache_dir)
    out = cp.zeros((3, lanes))
    planes = {"p": cp.asarray(FIELD), "i0": cp.asarray(where[:, 0].copy()), "i1": cp.asarray(where[:, 1].copy()), "y": out}
    _launch(bundle, f"read_{edge}", "cuda", planes).launch()
    cp.cuda.Device().synchronize()
    for lane in range(lanes):
        np.testing.assert_array_equal(out.get()[:, lane], _vector_expected(FIELD, list(where[lane]), edge))
