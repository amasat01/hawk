# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel whose body ends ``return <expr>`` returns its result: each leaf of the
returned array / tuple / list / dict becomes an ordinary ``Mutable`` slot, so the
program, the digest and the artifact are the ``Mutable`` form's."""

from __future__ import annotations

import linecache
import itertools

import pytest

import hawk
import hawk.math
from hawk import Matrix, Mutable, Param, Scalar, Vector, kernel
from hawk.ir import HawkError

_ids = itertools.count()


class Box:
    def __init__(self, v):
        self.v = v


def _define(body: str, params: str = "x: Scalar, v: Vector[3], k: Param", ret: str = "") -> object:
    """``@kernel`` of a function with this signature and body, its source registered
    with ``linecache`` so the trace can read it."""
    source = f"def f({params}){ret}:\n" + "".join(f"    {ln}\n" for ln in body.splitlines())
    name = f"<return-test-{next(_ids)}>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    scope = {"Scalar": Scalar, "Vector": Vector, "Matrix": Matrix, "Param": Param,
             "Mutable": Mutable, "Terminated": hawk.Terminated, "UInt32": hawk.UInt32,
             "hawk": hawk, "Box": Box}
    exec(compile(source, name, "exec"), scope)  # test fixture
    return kernel(scope["f"])


@kernel
def ret_single(x: Scalar, a: Param, b: Param) -> Scalar:
    y = a * x + b
    return y


@kernel
def mut_single(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@kernel
def ret_tuple(x: Scalar, v: Vector[3]):
    return x * 2.0, v * x


@kernel
def ret_dict(p: Vector[3], v: Vector[3], dt: Param) -> dict[str, Vector[3]]:
    return {"pos": p + dt * v, "vel": v - dt * p}


@kernel
def ret_nested(x: Scalar, v: Vector[3]) -> tuple[Scalar, dict[str, Vector[3]], list[Scalar]]:
    a = x * 2.0
    return a, {"w": v * x}, [x + 1.0, x - 1.0]


@kernel
def mut_nested(x: Scalar, v: Vector[3], a: Mutable[Scalar], w: Mutable[Vector[3]],
               out2: Mutable[Scalar], out3: Mutable[Scalar]):
    a = x * 2.0
    w = v * x
    out2 = x + 1.0
    out3 = x - 1.0


def slots(k):
    return [name for role, name in k.arg_spec if role == "mutable"]


def test_return_single_is_the_mutable_twin():
    assert ret_single.returns == "y"
    assert slots(ret_single) == ["y"]
    assert ret_single.walk.digest == mut_single.walk.digest
    assert mut_single.returns is None


def test_return_tuple_names_unnamed_leaves_out_k():
    assert ret_tuple.returns == ("out0", "out1")
    assert sorted(slots(ret_tuple)) == ["out0", "out1"]


def test_return_dict_keys_are_the_slot_names():
    assert ret_dict.returns == {"pos": "pos", "vel": "vel"}
    assert sorted(slots(ret_dict)) == ["pos", "vel"]


def test_return_nested_keeps_the_tree_and_equals_its_mutable_twin():
    assert ret_nested.returns == ("a", {"w": "w"}, ["out2", "out3"])
    assert ret_nested.walk.digest == mut_nested.walk.digest


def test_return_a_single_unnamed_expression_is_out():
    k = _define("return x * 2.0")
    assert k.returns == "out" and slots(k) == ["out"]


def test_return_mixed_with_mutable():
    k = _define("acc = x + 1.0\nreturn x * 2.0", "x: Scalar, acc: Mutable[Scalar]")
    assert k.returns == "out"
    assert sorted(slots(k)) == ["acc", "out"]


def test_return_annotation_checked_accepts_matching_forms():
    for ann in ("Scalar", "hawk.Scalar"):
        assert _define("return x * 2.0", ret=f" -> {ann}").returns == "out"
    assert _define("return x * 2.0, v * 2.0", ret=" -> tuple[Scalar, Vector[3]]").returns == ("out0", "out1")
    assert _define("return [x * 2.0, x * 3.0]", ret=" -> list[Scalar]").returns == ["out0", "out1"]
    assert _define('return {"a": v * 2.0}', ret=" -> dict[str, Vector[3]]").returns == {"a": "a"}
    assert _define("return x * 2.0, x * 3.0, x * 4.0", ret=" -> tuple[Scalar, ...]").returns == ("out0", "out1", "out2")


@pytest.mark.parametrize("body, ret, needle", [
    ("return x * 2.0", " -> Vector[3]", "`-> Vector[3]`, but the return is Scalar"),
    ("return v * 2.0", " -> Scalar", "but the return is Vector[3]"),
    ("return x * 2.0, v * 2.0", " -> tuple[Scalar, Scalar]", "return[1] is Vector[3]"),
    ("return x * 2.0, v * 2.0", " -> tuple[Scalar]", "declares 1 items, but the return has 2"),
    ("return x * 2.0, v * 2.0", " -> list[Scalar]", "declares a list, but the return is a tuple"),
    ("return [x * 2.0]", " -> tuple[Scalar]", "declares a tuple, but the return is a list"),
    ("return x * 2.0", " -> dict[str, Scalar]", "declares a dict, but the return is Scalar"),
    ('return {"a": x * 2.0}', " -> dict[str, Vector[3]]", "return['a'] is Scalar"),
    ("return x * 2.0", " -> None", "is not a payload type"),
])
def test_return_annotation_mismatch_refused(body, ret, needle):
    with pytest.raises(HawkError) as exc:
        _define(body, ret=ret)
    assert needle in str(exc.value)


def test_return_class_refused():
    with pytest.raises(HawkError, match="class instance is not supported — return a value, a tuple/list, or a dict"):
        _define("return Box(x * 2.0)")


@pytest.mark.parametrize("body, params, needle", [
    ("return {x: v}", "x: Scalar, v: Vector[3]", "string literals"),
    ('d = {"a": x}\nreturn d', "x: Scalar", "string-literal keys"),
    ("return 2.0", "x: Scalar", "must depend on at least one parameter"),
    ("if x > 0.0:\n    return x\nreturn x * 2.0", "x: Scalar", "ends with its one `return <expr>`"),
    ("y = x\nif x > 1.0:\n    y = x * 2.0\n    return\nreturn y", "x: Scalar", "bare `return` in a kernel that returns"),
    ('return {"x": x * 2.0}', "x: Scalar", "after a parameter or another output; use another name (x_next)"),
    ("return x * 2.0, {\"out0\": x * 3.0}", "x: Scalar", "'out0'"),
])
def test_return_refusals_name_the_fix(body, params, needle):
    with pytest.raises(HawkError) as exc:
        _define(body, params)
    assert needle in str(exc.value), str(exc.value)


def test_return_steps_keeps_the_tree():
    step = _define("terminated = x > 1.0\nreturn x * 2.0", "x: Scalar, terminated: Terminated")
    assert hawk.steps(step, 4).returns == step.returns == "out"
    assert hawk.resolve_dispatch(step, {}).returns == "out"


def test_return_vjp_seeds_are_named_after_the_slot():
    k = _define("return x * x", "x: Scalar")
    names = [n for _, n in hawk.Kernel("k_vjp", hawk.diff.vjp(k)).arg_spec]
    assert "bar_out" in names


def test_buffer_is_a_reserved_parameter_name():
    with pytest.raises(HawkError, match="`buffer` is a reserved parameter name"):
        _define("y = x", "x: Scalar, buffer: Mutable[Scalar]")


def test_walk_reduces_sees_a_reduction_over_samples():
    plain = _define("return x * 2.0", "x: Scalar")
    reducing = _define("flag = hawk.math.uint32(hawk.sum(x > 1.0))",
                       "x: Scalar, flag: Mutable[UInt32]")
    assert plain.walk.reduces is False
    assert reducing.walk.reduces is True


def test_return_annotation_mismatch_reads_as_written():
    with pytest.raises(HawkError) as exc:
        _define("return x * 2.0, v * 2.0", ret=" -> tuple[Scalar, Vector[3], Scalar]")
    assert "`-> tuple[Scalar, Vector[3], Scalar]`" in str(exc.value)
    assert "TensorType" not in str(exc.value)
