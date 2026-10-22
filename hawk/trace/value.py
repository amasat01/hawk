# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The traced VALUE — operator overloading from Python expressions to IR nodes.

A :class:`Value` wraps one IR node; every operator mints an
:func:`hawk.ir.ops.make` call, building the DAG as the body runs once. Free
functions below cover what operators can't spell (``norm``, ``dot``,
``cross``, ``select``); three REF objects cover the write side (a mutable
plane, a wide plane) plus a :class:`TableRef` read
with ``t.at(i)``. A :class:`Value` has no truth value — control flow reaches
the IR only through the ``ast`` pass's ``select`` rewrite.
"""

from __future__ import annotations

import math
from typing import Any

from ..ir import At, Const, HawkError, Leaf, Node, Select
from ..ir import make as _make
from ..ir.nodes import (
    AccumWrite,
    Assign,
    Dispatch,
    Finish,
    ReduceSamples,
    RoleConst,
    SampleIndex,
    WideWrite,
)
from ..types import INT_PLANE_DTYPES, TensorType

_F64 = TensorType((), "f64")
_BOOL = TensorType((), "bool")
_I32 = TensorType((), "i32")


def node_of(x: Any) -> Node:
    """The IR node behind a traced value or a Python literal (the constant).

    An object carrying ``_hawk_refuse`` refuses HERE instead of the generic
    message — a loop-local name's sentinel
    (:class:`hawk.trace.astpass._LoopLocal`) reaches this coercion before its
    own ``__radd__`` can run, so it is told what to write instead of "is not
    a traced value"."""
    refuse = getattr(x, "_hawk_refuse", None)
    if refuse is not None:
        refuse()
    if isinstance(x, Value):
        return x.node
    if isinstance(x, Node):
        return x
    if isinstance(x, bool):
        return Const(x, _BOOL)
    if isinstance(x, (int, float)):
        return Const(float(x), _F64)
    raise HawkError(
        f"{x!r} is not a traced value, an IR node or a numeric literal — a kernel "
        "body may only combine declared vocabulary and literals"
    )


def index_node(x: Any) -> Node:
    """The IR node behind an INDEX — a traced value, or an integer literal,
    separate from :func:`node_of`, which types a bare ``int`` as ``f64``. An
    index is an address: as ``f64`` it would cast to ``std::size_t`` and
    promote any stride multiplied with it, so it is typed ``i32`` instead
    (:mod:`hawk.ir.contraction`)."""
    if isinstance(x, bool):
        raise HawkError(f"{x!r} is not an index — a bool is a mask, not an address")
    if isinstance(x, int):
        return Const(int(x), _I32)
    return node_of(x)


def _operands(args: tuple) -> tuple:
    """The operand nodes of one traced op, with literals typed BY COMPANY: a
    bare ``int`` is ``f64`` alone, but when every traced operand is integral
    it types as an integer instead — otherwise an index like
    ``entry * cols + row`` would promote to ``Real`` and lose its stride
    derivation (:mod:`hawk.ir.contraction`)."""
    traced = [a for a in args if isinstance(a, (Value, Node))]
    dtypes = {(a.node if isinstance(a, Value) else a).ttype.dtype for a in traced}
    plane = next((d for d in INT_PLANE_DTYPES if d in dtypes), None)
    if plane is not None:
        return tuple(int_literal(a, plane) if isinstance(a, int) and not isinstance(a, bool)
                     else node_of(a) for a in args)
    integral = bool(dtypes) and dtypes <= {"i32", "i64"}
    return tuple(index_node(a) if integral and isinstance(a, int)
                 and not isinstance(a, bool) else node_of(a) for a in args)


#: The value range of each integer plane dtype.
_INT_RANGE = {"int32": (-2**31, 2**31 - 1), "uint32": (0, 2**32 - 1)}


def int_literal(x: int, dtype: str) -> Const:
    """A Python ``int`` beside an ``int32``/``uint32`` value, typed as that
    value: a literal outside the type's range is refused, never wrapped."""
    lo, hi = _INT_RANGE[dtype]
    if not lo <= x <= hi:
        raise HawkError(
            f"the literal {x} is outside the {dtype} range [{lo}, {hi}]; write a "
            f"value inside it (an {dtype} wraps only through arithmetic)")
    return Const(int(x), TensorType((), dtype))


def _op(kind: str, *args: Any, literal: Any = None) -> Value:
    return Value(_make(kind, _operands(args), literal))


class Value:
    """One traced IR value: every operator on it mints a node."""

    __slots__ = ("node",)

    def __init__(self, node: Node) -> None:
        self.node = node

    @property
    def ttype(self) -> TensorType:
        """The node's tensor type."""
        return self.node.ttype

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Value {self.node.kind} {self.node.ttype}>"

    def __add__(self, o: Any) -> Value:
        return _op("add", self, o)

    def __radd__(self, o: Any) -> Value:
        return _op("add", o, self)

    def __sub__(self, o: Any) -> Value:
        return _op("sub", self, o)

    def __rsub__(self, o: Any) -> Value:
        return _op("sub", o, self)

    def __mul__(self, o: Any) -> Value:
        return _op("mul", self, o)

    def __rmul__(self, o: Any) -> Value:
        return _op("mul", o, self)

    def __truediv__(self, o: Any) -> Value:
        # lever 3: a REAL value divided by a trace-time constant becomes a
        # multiply by the reciprocal (an IEEE division costs a slow-path
        # call; the reciprocal folds into one FFMA). Integral values keep
        # the division — the contraction recognizer reads it as a stride.
        if (isinstance(o, (int, float)) and not isinstance(o, bool)
                and self.ttype.dtype not in ("i32", "i64", "bool", *INT_PLANE_DTYPES)
                and o != 0 and math.isfinite(o) and math.isfinite(1.0 / o)):
            return _op("mul", self, 1.0 / o)
        return _op("div", self, o)

    def __rtruediv__(self, o: Any) -> Value:
        return _op("div", o, self)

    def __pow__(self, o: Any) -> Value:
        return _op("pow", self, o)

    def __neg__(self) -> Value:
        return _op("neg", self)

    def __abs__(self) -> Value:
        return _op("abs", self)

    def __floordiv__(self, o: Any) -> Value:
        return _op("floordiv", self, o)

    def __rfloordiv__(self, o: Any) -> Value:
        return _op("floordiv", o, self)

    def __mod__(self, o: Any) -> Value:
        return _op("mod", self, o)

    def __rmod__(self, o: Any) -> Value:
        return _op("mod", o, self)

    def __and__(self, o: Any) -> Value:
        return _op("band", self, o)

    def __rand__(self, o: Any) -> Value:
        return _op("band", o, self)

    def __or__(self, o: Any) -> Value:
        return _op("bor", self, o)

    def __ror__(self, o: Any) -> Value:
        return _op("bor", o, self)

    def __xor__(self, o: Any) -> Value:
        return _op("bxor", self, o)

    def __rxor__(self, o: Any) -> Value:
        return _op("bxor", o, self)

    def __lshift__(self, o: Any) -> Value:
        return _op("shl", self, o)

    def __rlshift__(self, o: Any) -> Value:
        return _op("shl", o, self)

    def __rshift__(self, o: Any) -> Value:
        return _op("shr", self, o)

    def __rrshift__(self, o: Any) -> Value:
        return _op("shr", o, self)

    def __invert__(self) -> Value:
        return _op("invert", self)

    def __matmul__(self, o: Any) -> Value:
        other = node_of(o)
        return _op("mv" if len(other.ttype.shape) == 1 else "mm", self, other)

    def __lt__(self, o: Any) -> Value:
        return _elementwise("lt", self, o)

    def __le__(self, o: Any) -> Value:
        return _elementwise("le", self, o)

    def __gt__(self, o: Any) -> Value:
        return _elementwise("gt", self, o)

    def __ge__(self, o: Any) -> Value:
        return _elementwise("ge", self, o)

    def __eq__(self, o: Any) -> Value:  # type: ignore[override]
        return _elementwise("eq", self, o)

    def __ne__(self, o: Any) -> Value:  # type: ignore[override]
        return _elementwise("ne", self, o)

    __hash__ = object.__hash__

    def __getitem__(self, k: Any) -> Value:
        """``v[k]`` — one component of a rank>=1 value.

        A STATIC integer is ``component``, checked at trace time; a TRACED
        integer is ``component_at`` — needed inside a lowered ``for``, where
        ``k`` is a symbolic :class:`~hawk.ir.loop_nodes.LoopIndex`, not a
        Python int (it materialises into an ``Item`` first, since aether's
        EXPRESSION only carries compile-time ``eval<Is...>``). A float is
        refused rather than truncated: rounding would silently read the
        wrong component."""
        if isinstance(k, bool):
            raise HawkError(
                f"a traced value is indexed by an integer component, got {k!r} — "
                "a bool is a mask, not an address")
        if isinstance(k, int):
            return _op("component", self, literal=k)
        if not self.node.ttype.shape and isinstance(k, Value):
            raise HawkError(
                "x[key] indexes a component and a rank-0 value has none; samples "
                "are indexed on the view: hawk.all_samples(x)[key]")
        node = k.node if isinstance(k, Value) else k
        if isinstance(node, Node) and node.ttype.shape == ()\
                and node.ttype.dtype in ("i32", "i64"):
            return Value(_make("component_at", (self.node, node)))
        raise HawkError(
            f"a traced value is indexed by a STATIC integer component or by a "
            f"traced INTEGER index, got {k!r}. A lowered `for`'s own index is "
            "the second form (`v[k]` inside the loop body); a rank-0 "
            "floating value is neither — an index is an address, and rounding "
            "one silently reads the wrong component (a cross-sample read of a "
            "whole plane is a Table's at())")

    def __setitem__(self, k: Any, value: Any) -> None:
        """``v[k] = e`` reaches HERE only from a plain-python HELPER whose
        body the ``ast`` pass never parses — inside a kernel body it becomes
        a rebinding (``v = set_component(v, k, e)``) instead, since a HAWK
        value is immutable. A helper gets no such rewrite, so it is told
        where the spelling works."""
        del k, value
        raise HawkError(
            "a traced value is immutable, so `v[k] = …` only works inside a "
            "KERNEL body, where the ast pass rewrites it into a rebinding "
            ". This call is in a plain-python helper, whose body the "
            "pass never sees: return the component values and let the kernel "
            "body write them, or build the whole value at once with vec(...)")

    def sum(self, axis: Any = None) -> Value:
        """The sum over the value's own ``axis`` (None: every entry)."""
        from .reduce import reduce_own
        return reduce_own(self, "sum", axis)

    def max(self, axis: Any = None) -> Value:
        """The maximum over the value's own ``axis`` (None: every entry)."""
        from .reduce import reduce_own
        return reduce_own(self, "max", axis)

    def min(self, axis: Any = None) -> Value:
        """The minimum over the value's own ``axis`` (None: every entry)."""
        from .reduce import reduce_own
        return reduce_own(self, "min", axis)

    def mean(self, axis: Any = None) -> Value:
        """The mean over the value's own ``axis`` (None: every entry)."""
        from .reduce import reduce_own
        return reduce_own(self, "mean", axis)

    @property
    def components(self) -> Any:
        """A vector's components, by name: ``v.components[k]``."""
        from .reduce import view
        return view(self, "components")

    @property
    def rows(self) -> Any:
        """A matrix's rows, by name: ``m.rows[r]`` is the row as a vector."""
        from .reduce import view
        return view(self, "rows")

    @property
    def columns(self) -> Any:
        """A matrix's columns, by name: ``m.columns[c]`` is the column as a vector."""
        from .reduce import view
        return view(self, "columns")

    def __bool__(self) -> bool:
        raise HawkError(
            "a traced value has no truth value: control flow is rewritten into "
            "select nodes by the ast pass. This fires when a traced value "
            "reaches a Python branch, comprehension or `and`/`or` the pass does not "
            "own — use select(cond, a, b), land/lor/lnot"
        )


def _cast(x: Any, dtype: str) -> Value:
    node = node_of(x)
    if node.ttype.dtype == dtype:
        return Value(node)
    return Value(_make("cast", (node,), dtype))


def int32(x: Any) -> Value:
    """``x`` as an ``int32``: a float truncates toward zero, saturates at
    ``[-2**31, 2**31 - 1]`` and maps NaN to 0; an integer wraps modulo 2**32
    (``np.astype``). Zero derivative.

    :param x: a traced rank-0 number.
    :returns: the rank-0 ``int32`` value.
    :raises HawkError: for a bool or a rank>=1 value.

    Example: ``n = hawk.math.int32(x * 10.0)``"""
    return _cast(x, "int32")


def index(x: Any, n: int) -> Value:
    """``x`` as an INDEX into an extent of ``n``: truncated toward zero, then
    clamped to ``[0, n)`` (NaN gives 0). The runtime component form
    ``v[hawk.math.index(x, n)]`` takes it; an extent is a type, so an index outside it
    is not an error the read or the write can report, and this is where it is made
    safe. The clamp can HIDE an index bug (an input far outside the grid reads the
    end element); an author who wants the error computes the index without it.

    :param x: a traced rank-0 number.
    :param n: the extent, a Python integer >= 1.
    :returns: the rank-0 ``i32`` index.
    :raises HawkError: for a bool, a rank>=1 value or ``n < 1``.

    Example: ``c = theta[hawk.math.index(u * 5.0, 13)]``"""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise HawkError(f"index: the extent must be a Python integer >= 1, got {n!r}")
    return _cast(minimum(maximum(real(x), 0.0), float(n - 1)), "i32")


def uint32(x: Any) -> Value:
    """``x`` as a ``uint32``: a float truncates toward zero, saturates at
    ``[0, 2**32 - 1]`` and maps NaN to 0; an integer wraps modulo 2**32
    (``np.astype``). Zero derivative.

    :param x: a traced rank-0 number.
    :returns: the rank-0 ``uint32`` value.
    :raises HawkError: for a bool or a rank>=1 value.

    Example: ``count = hawk.math.uint32(steps)``"""
    return _cast(x, "uint32")


def real(x: Any) -> Value:
    """``x`` as the kernel's floating type (``Real``: float64, or float32 under
    ``mode="float32"``). The derivative with respect to an integer is zero.

    :param x: a traced rank-0 number.
    :returns: the rank-0 floating value.
    :raises HawkError: for a bool or a rank>=1 value.

    Example: ``y = hawk.math.real(n) * x``"""
    return _cast(x, "f64")


def _unary(kind: str):
    def fn(x: Any) -> Value:
        return _op(kind, x)
    fn.__name__ = kind
    fn.__doc__ = f"Elementwise ``{kind}`` of a traced value."
    return fn


sqrt = _unary("sqrt")
rsqrt = _unary("rsqrt")
exp = _unary("exp")
log = _unary("log")
sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
tanh = _unary("tanh")
asin = _unary("asin")
acos = _unary("acos")
atan = _unary("atan")
quat_conj = _unary("quat_conj")
quat_recip = _unary("quat_recip")
as_pure = _unary("as_pure")
as_vec3 = _unary("as_vec3")


def _binary(kind: str):
    def fn(a: Any, b: Any) -> Value:
        return _op(kind, a, b)
    fn.__name__ = kind
    fn.__doc__ = f"``{kind}`` of two traced values."
    return fn


minimum = _binary("min")
maximum = _binary("max")
dot = _binary("dot")
cross = _binary("cross")
#: The outer product ``u v^T``, a FREE function because Python has no infix
#: for it and ``@`` is already the contraction (``mv``/``mm``): the
#: contracting ``dot`` and expanding ``outer`` must be spelled apart.
outer = _binary("outer")
quat_mul = _binary("quat_mul")
quat_rotate = _binary("quat_rotate")
norm = _unary("norm")
vsum = _unary("sum")
transpose = _unary("transpose")


def _entries(shape: tuple) -> list:
    """Every static index of a rank-1 or rank-2 ``shape``, row-major, as the
    ``component`` literal reading that entry."""
    if len(shape) == 1:
        return list(range(shape[0]))
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _elementwise(kind: str, *args: Any) -> Value:
    """``kind`` applied component-wise. Rank-0 operands mint the op
    itself; rank-1 or rank-2 operands are mapped entry by entry (a rank-0
    operand broadcasts) and re-assembled with ``vec``, so the emitter only
    ever spells the rank-0 call. The shapes combine as the arithmetic
    operators' do — equal shapes, or one side rank-0 — and refuse in the
    same words otherwise."""
    nodes = _operands(args)
    shape: tuple = ()
    for n in nodes:
        shape = _broadcast_shape(kind, shape, n.ttype.shape)
    if not shape:
        return Value(_make(kind, nodes))
    if len(shape) > 2:
        raise HawkError(
            f"{kind}: applies component-wise to rank-0, rank-1 or rank-2 values, "
            f"got {[n.ttype for n in nodes]}")
    parts = []
    for k in _entries(shape):
        ops = [n if n.ttype.shape == () else _make("component", (n,), k)
               for n in nodes]
        parts.append(_make(kind, ops))
    return Value(_make("vec", parts, shape if len(shape) == 2 else None))


def _broadcast_shape(kind: str, a: tuple, b: tuple) -> tuple:
    """The shape two elementwise operands combine to — the arithmetic
    operators' rule (:func:`hawk.ir.ops.result_type`), with its refusal."""
    if a == b or b == ():
        return a
    if a == ():
        return b
    raise HawkError(
        f"op {kind!r}: shapes {a} and {b} do not broadcast — an "
        "elementwise op takes equal shapes or one rank-0 operand"
    )


def _mapped(kind: str):
    """``kind`` as a free function that maps rank>=1 operands entry by
    entry (:func:`_elementwise`) — for the kinds aether spells at rank 0 only."""
    def fn(*args: Any) -> Value:
        return _elementwise(kind, *args)
    return fn


def _named(fn, name: str, doc: str):
    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = doc
    return fn


#: The kinds aether spells at rank 0 only, mapped entry by entry on a
#: vector or matrix like the functions below.
atan2 = _named(_mapped("atan2"), "atan2",
               "``atan2(a, b)``, component-wise; rank-0, rank-1 or rank-2 operands.")
land = _named(_mapped("land"), "land",
              "Logical and of two bool values, component-wise.")
lor = _named(_mapped("lor"), "lor", "Logical or of two bool values, component-wise.")
lnot = _named(_mapped("lnot"), "lnot", "Logical not of a bool value, component-wise.")


def _lifted(kind: str, doc: str, params: tuple = ("x",)):
    def fn(*args: Any) -> Value:
        if len(args) != len(params):
            raise HawkError(
                f"{kind}({', '.join(params)}) takes {len(params)} argument(s), "
                f"got {len(args)}")
        return _elementwise(kind, *args)
    fn.__name__ = fn.__qualname__ = kind
    fn.__doc__ = doc
    return fn


_D = "Component-wise; rank-0, rank-1 or rank-2 operands."
_Z = "Zero derivative."
#: Piecewise-constant (lever 2): the largest integer <= ``x``, as an f64
#: value. Its VJP/JVP are the explicit ZERO rule: zero almost everywhere.
floor = _lifted("floor", f"Largest integer <= ``x`` (``np.floor``). {_D} {_Z}")
exp2 = _lifted("exp2", f"``2**x`` (``np.exp2``). {_D}")
expm1 = _lifted("expm1", f"``exp(x) - 1``, accurate near 0 (``np.expm1``). {_D}")
log2 = _lifted("log2", f"Base-2 logarithm (``np.log2``). {_D}")
log10 = _lifted("log10", f"Base-10 logarithm (``np.log10``). {_D}")
log1p = _lifted("log1p", f"``log(1 + x)``, accurate near 0 (``np.log1p``). {_D}")
cbrt = _lifted("cbrt", f"Real cube root, odd in ``x`` (``np.cbrt``). {_D}")
sinh = _lifted("sinh", f"Hyperbolic sine (``np.sinh``). {_D}")
cosh = _lifted("cosh", f"Hyperbolic cosine (``np.cosh``). {_D}")
asinh = _lifted("asinh", f"Inverse hyperbolic sine (``np.arcsinh``). {_D}")
acosh = _lifted("acosh",
                f"Inverse hyperbolic cosine, ``x >= 1`` (``np.arccosh``). {_D}")
atanh = _lifted("atanh",
                f"Inverse hyperbolic tangent, ``|x| <= 1`` (``np.arctanh``). {_D}")
ceil = _lifted("ceil", f"Smallest integer >= ``x`` (``np.ceil``). {_D} {_Z}")
trunc = _lifted("trunc", f"Integer part, toward zero (``np.trunc``). {_D} {_Z}")
round = _lifted(  # noqa: A001 - a namespace module's names are its API (cf. numpy)
    "round", "Nearest integer, halves AWAY from zero (C ``round``; ``np.round`` "
    f"rounds halves to even, which is :func:`rint`). {_D} {_Z}")
rint = _lifted("rint", f"Nearest integer, halves to even (``np.rint``). {_D} {_Z}")
sign = _lifted("sign", "-1, 0 or 1 by the sign of ``x``; NaN stays NaN and both "
               f"zeros give +0 (``np.sign``). {_D} {_Z}")
spacing = _lifted("spacing", "The gap from ``x`` to the next float away from zero, "
                  f"signed like ``x`` (``np.spacing``): one ulp. {_D} {_Z}")
erf = _lifted("erf", f"The error function (``scipy.special.erf``). {_D}")
erfc = _lifted("erfc", "``1 - erf(x)``, accurate for large ``x`` "
               f"(``scipy.special.erfc``). {_D}")
hypot = _lifted("hypot", f"``sqrt(a*a + b*b)`` without overflow (``np.hypot``). {_D}",
                ("a", "b"))
copysign = _lifted("copysign",
                   f"``|a|`` with the sign bit of ``b`` (``np.copysign``). {_D}",
                   ("a", "b"))
fmod = _lifted("fmod", "C remainder of ``a / b``, sign of ``a`` (``np.fmod``). "
               f"{_D}", ("a", "b"))
remainder = _lifted(
    "remainder", "Floor-mod ``a - floor(a / b) * b``, sign of ``b``; a zero result "
    f"carries ``b``'s sign (``np.remainder``, not C's IEEE ``remainder``). {_D}",
    ("a", "b"))
fdim = _lifted("fdim", f"``max(a - b, 0)`` (C ``fdim``). {_D}", ("a", "b"))
fma = _lifted("fma", f"``a * b + c`` with one rounding (C ``fma``). {_D}",
              ("a", "b", "c"))
clip = _lifted("clip", "``minimum(maximum(x, lo), hi)`` with NaN propagated from any "
               f"argument (``np.clip``). {_D}", ("x", "lo", "hi"))
_B = "bool-typed, component-wise on a vector or matrix."
isnan = _lifted("isnan", f"``True`` where ``x`` is NaN (``np.isnan``); {_B}")
isinf = _lifted("isinf", f"``True`` where ``x`` is +-inf (``np.isinf``); {_B}")
isfinite = _lifted("isfinite", "``True`` where ``x`` is neither inf nor NaN "
                   f"(``np.isfinite``); {_B}")
#: ``np.absolute``/``np.power``: the ``abs(x)``/``x ** y`` operators as functions.
absolute = _unary("abs")
absolute.__name__ = absolute.__qualname__ = "absolute"
absolute.__doc__ = "``|x|`` (``np.absolute``; also ``abs(x)``). Any rank."
power = _binary("pow")
power.__name__ = power.__qualname__ = "power"
power.__doc__ = "``a ** b`` (``np.power``; also the ``**`` operator). Any rank."


def vec(*components: Any) -> Value:
    """Build a rank-1 value from static scalar components."""
    return _op("vec", *components)


def select(cond: Any, on_true: Any, on_false: Any) -> Value:
    """``cond ? a: b`` — what the ast pass rewrites control flow into.
    Component-wise on a vector or matrix: the mask is rank-0 or the
    branches' shape, and a rank-0 branch broadcasts."""
    c, a, b = node_of(cond), node_of(on_true), node_of(on_false)
    shape = _broadcast_shape("select", _broadcast_shape(
        "select", c.ttype.shape, a.ttype.shape), b.ttype.shape)
    if shape:
        # a rank-0 branch broadcasts like an arithmetic operand (numpy's
        # ``where``); the mask is rank-0 (one choice per sample) or the
        # branches' own shape (one choice per entry)
        a, b = (_make("splat", (x,), shape) if x.ttype.shape == () else x
                for x in (a, b))
    if a.ttype != b.ttype:
        raise HawkError(f"select: the branches disagree — {a.ttype!r} vs {b.ttype!r}")
    return Value(Select(c, a, b, a.ttype))


def dispatch(kind: Any, branches: Any, policy: str = "switch") -> Value:
    """``branches[clamp(kind, 0)]`` — the finite per-element dispatch.

    ``kind`` is a per-element integer VALUE, never a Python literal — a body
    that knows the branch at TRACE time is writing ordinary control flow, not
    a dispatch. ``policy`` names one of three lowerings
    (``predicated``/``switch``/``segmented``), or ``auto`` — the one value a
    launching runtime's chooser resolves (:func:`hawk.resolve_dispatch`;
    until then it renders as ``switch`` and counts the branch mix); the
    default ``switch`` is what a compiler already picks for a branch-heavy
    chain.
    :class:`~hawk.ir.nodes.Dispatch` does the type/arity checking."""
    bs = [node_of(b) for b in branches]
    ttype = bs[0].ttype if bs else TensorType((), "f64")
    selector = node_of(kind)
    if selector.ttype.dtype in INT_PLANE_DTYPES:
        selector = _make("cast", (selector,), "i32")
    return Value(Dispatch(selector, bs, policy, ttype))


def n_samples() -> Value:
    """The readable sample count: the TRUE ``nSamples``, never ``count``."""
    return Value(Leaf("nsamples", "nsamples", "n_samples", TensorType((), "i32")))


def sample_index() -> Value:
    """This lane's GLOBAL sample index, as a value (the ``own(i)``): it is
    ``base + flat`` under a partition, so an expression built on it is
    partition-invariant — safe for the oracle to compare — and lets a
    2-D-domain kernel recover the coordinates a flattened launch folded
    together."""
    return Value(SampleIndex())


def _random(kind: str, seed: Any, counter: Any) -> Value:
    """The shared body of :func:`random_uniform`/:func:`random_normal`.

    ``seed``/``counter`` go through :func:`index_node`, not :func:`node_of`:
    a bare ``int`` must type as INTEGER (``i32``) here, never the ``f64`` a
    literal means elsewhere, since a random op's operands are stream
    addresses, not real-valued arithmetic."""
    return _op(kind, index_node(seed), index_node(counter))


def random_uniform(seed: Any, counter: Any) -> Value:
    """A reproducible ``U[0, 1)``-shaped draw.

    ``(seed, counter)`` pick the stream: aether's draw is a PURE function of
    ``(seed, lane index, counter)``, so the SAME pair on the SAME lane is
    bit-identical, host and device, across a captured graph's replay — making
    them GPU-graph wires, not mutable RNG state. Scale with ordinary
    arithmetic: ``lo + (hi - lo) * random_uniform(seed, counter)``."""
    return _random("random_uniform", seed, counter)


def random_normal(seed: Any, counter: Any) -> Value:
    """A reproducible standard-normal draw, ``N(0, 1)``.

    Same ``(seed, counter)`` contract as :func:`random_uniform`. Shift and
    scale with ordinary arithmetic: ``mu + sd * random_normal(seed, counter)``."""
    return _random("random_normal", seed, counter)


class TableRef:
    """A ``Table``/``Staged``/``Wide`` parameter: read with ``at()``.

    Two read spellings, one form. ``t.at(expr)`` is the positional read of a
    flat plane; ``t.at(row=…, col=…)`` is the NAMED-AXIS read, folding to
    ``sum(index_d * stride_d)`` over the DECLARED order, each stride minted
    as a :class:`~hawk.ir.nodes.RoleConst` carrying its axis — recoverable as
    the declaration's, unlike the flat form."""

    __slots__ = ("leaf", "name", "dims")

    def __init__(self, name: str, ttype: TensorType, *, role: str = "lookup",
                 dims: Any = ()) -> None:
        kind = "table_read" if role == "lookup" else "vocab_read"
        self.leaf = Leaf(kind, role, name, ttype)
        self.name = name
        self.dims = tuple(dims)

    def at(self, *positional: Any, **named: Any) -> Value:
        """The ``at(expr)``: an absolute-index read of this plane."""
        return Value(At(self.leaf, self._index(positional, named), self.leaf.ttype))

    def __getitem__(self, i: Any) -> Value:
        """``t[i]``: the flat read, the same as ``t.at(i)``."""
        return self.at(i)

    def _index(self, positional: tuple, named: dict) -> Node:
        if named and positional:
            raise HawkError(
                f"{self.name!r}: at() takes a flat index OR every declared axis by "
                "name, never both")
        if not named:
            if len(positional) != 1:
                raise HawkError(
                    f"{self.name!r}: at() takes ONE flat index, got "
                    f"{len(positional)} — a named-axis plane is read "
                    "at(<axis>=…, …)")
            return index_node(positional[0])
        if not self.dims:
            raise HawkError(
                f"{self.name!r} declares no named axes, so it is read at a flat "
                f"index: at(expr). Named axes are declared "
                'Table["row":rows, "col":cols]')
        declared = [label for label, _size, _stride in self.dims]
        missing = [d for d in declared if d not in named]
        unknown = sorted(k for k in named if k not in set(declared))
        if missing or unknown:
            raise HawkError(
                f"{self.name!r}: at() must name every declared axis exactly once "
                f"(declared {declared})"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else "")
                + " — a partial address would silently read the wrong row")
        flat: Node | None = None
        for label, _size, stride in self.dims:
            term = _make("mul", (index_node(named[label]),
                                 RoleConst(stride, f"{self.name}_{label}_stride")))
            flat = term if flat is None else _make("add", (flat, term))
        return flat


def own_column_rewrite_advice(name: str) -> str:
    """The rewrite advice given for every refused ``.prior`` read /
    ``name[...] = e`` write on a Mutable/Terminated/WideOut own-column name:
    the plane's bare name is an ordinary local."""
    return (f"write `{name} = …` / read `{name}` before assigning it; to keep "
            f"the start value after writing, bind `{name}0 = {name}` first")


class FinishRefusal(HawkError):
    """A refusal of a finish statement raised while the body runs, before the
    tracer knows which kernel it is in; :func:`hawk.trace.kernel.kernel`
    re-raises it as a :class:`HawkError` naming the kernel."""


#: The one type a finish condition carries.
_BOOL0 = TensorType((), "bool")


class TerminatedRef(Value):
    """A ``Terminated`` mask parameter.

    Read like any input (it IS the mask's leaf), and FINISHED by ONE
    statement ``mask[...] = cond`` (a traced rank-0 bool): the mask becomes
    ``mask | cond`` (set-only, monotone), rewritten by the ``ast`` pass into
    a name committed once at the end through :meth:`write`."""

    __slots__ = ("sinks",)

    def __init__(self, node: Node) -> None:
        super().__init__(node)
        self.sinks: list[Node] = []

    def unset(self) -> Value:
        """The "does not finish on this path" value an ``if`` merges against."""
        return Value(Const(False, _BOOL0))

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            raise FinishRefusal(
                f"{self.node.name!r}: a mask is finished whole — write "
                f"`{self.node.name}[...] = cond`")
        self.write(value)

    def write(self, value: Any) -> None:
        """Record the finish: the sample's mask becomes ``mask | value``."""
        name = self.node.name
        if type(value).__name__ in ("bool", "bool_"):
            raise FinishRefusal(
                f"{name!r} is finished with the Python literal {value!r}; "
                "`= True` is 'always finish' — spell the condition "
                "(`= t >= t_final`) — and `= False` is a no-op that reads as a bug")
        if self.sinks:
            raise FinishRefusal(
                f"{name!r} is finished twice; a mask takes exactly ONE "
                f"`{name}[...] = cond` statement — combine the conditions with "
                "lor(...)")
        node = node_of(value)
        if node.ttype != _BOOL0:
            raise FinishRefusal(
                f"{name!r} is finished with a value typed {node.ttype!r}; the "
                "condition is a traced rank-0 bool")
        self.sinks = [Finish(name, node)]


_INDEX_DTYPES = ("i32", "i64") + tuple(INT_PLANE_DTYPES)


def _is_stored(value: Any) -> bool:
    from .reduce import _Stored
    return isinstance(value, _Stored)


def _by_key(ref: Any, key: Any, value: Any, accumulate: bool) -> None:
    """One store by key into ``ref``: an accumulate (``AccumWrite``) or a
    store (``WideWrite``), both carrying the index."""
    if any(not isinstance(k, (AccumWrite, WideWrite)) for k in ref.sinks):
        raise HawkError(
            f"{ref.name!r} is written per sample (locked) and by key "
            f"(hawk.all_samples({ref.name})[key]): one plane has one extent")
    index = index_node(key)
    if index.ttype.shape != () or index.ttype.dtype not in _INDEX_DTYPES:
        raise HawkError(f"hawk.all_samples({ref.name})[key]: key is an Index or an "
                        f"integer value, got {index.ttype!r}")
    kind = "accum" if accumulate else "wide_out"
    if ref.ttype.shape:
        raise HawkError(
            f"hawk.all_samples({ref.name})[key]: a by-key commit is the target's "
            "atomic add, which is per component in this release; write one "
            f"Mutable[Scalar] plane per component ({ref.name}0, ...) and "
            f"hawk.all_samples({ref.name}0)[key] += u[0]")
    if ref.by_key_kind is not None and (ref.by_key_kind != kind or not accumulate):
        raise HawkError(
            f"{ref.name!r} is stored by key twice (or both accumulated and stored): "
            "one plane takes one by-key statement, or several `+=`")
    if accumulate:
        ref.sinks.append(AccumWrite(ref.name, ref._typed(value), index, ref.ttype))
    else:
        ref.sinks = [WideWrite(ref.name, ref._typed(value), index, ref.ttype)]
    ref.by_key_kind = kind


def retired_sink_method(name: str, method: str) -> str:
    """The refusal for ``.add`` / ``.write`` / ``.contribute`` on an output."""
    if method == "contribute":
        return (f"{name!r}: write {name} = hawk.all_samples(v).sum() "
                "(or .max()/.min()/.mean())")
    return (f"{name!r}: write hawk.all_samples({name})[key] += v (accumulate by "
            f"key) or hawk.all_samples({name})[key] = v (store by key)")


class _SinkRef:
    """Shared write-side state: the sinks a body commits through one plane."""

    __slots__ = ("name", "ttype", "sinks", "by_key_kind")

    def __init__(self, name: str, ttype: TensorType) -> None:
        self.name = name
        self.ttype = ttype
        self.sinks: list[Node] = []
        self.by_key_kind: str | None = None

    def _typed(self, value: Any) -> Node:
        node = node_of(value)
        if node.ttype != self.ttype:
            raise HawkError(
                f"{self.name!r} is declared {self.ttype!r} but the committed value is "
                f"typed {node.ttype!r}"
            )
        return node

    def add(self, *_: Any, **__: Any) -> None:
        raise HawkError(retired_sink_method(self.name, "add"))

    def write(self, *_: Any, **__: Any) -> None:
        raise HawkError(retired_sink_method(self.name, "write"))

    def contribute(self, *_: Any, **__: Any) -> None:
        raise HawkError(retired_sink_method(self.name, "contribute"))

    @property
    def prior(self) -> Value:
        """``.prior`` reads the plane's value as THIS launch began it — a
        Mutable's own seam (:class:`MutableRef` overrides this). A scattered
        or reduced plane (a by-key or WideOut one) has no single prior value one
        sample owns, so the base form refuses by name.

        A HOST-FACING API: this property is unaffected by the refusal of the
        `.prior`/`[...]=` spelling INSIDE a traced kernel body
        (:mod:`hawk.trace.astpass` refuses those, syntactically, before this
        ever runs); it stays how a host builds a ref's sinks directly (as
        :func:`hawk.trace.kernel.trace_value`'s caller does) and how the ast
        front end itself seeds a bare read before a plane's first store."""
        raise HawkError(
            f"{self.name!r} is used before it is assigned, but a "
            f"{type(self).__name__[:-3]} plane is scattered or reduced across "
            "samples, so it has no start-of-step value a sample owns: assign it "
            "before any read, and assign it after a loop rather than only "
            "inside one"
        )


class MutableRef(_SinkRef):
    """A ``Mutable[...]`` parameter, assigned with ``out[...] = value``."""

    __slots__ = ()

    @property
    def prior(self) -> Value:
        """The value this plane held when THIS launch began, readable
        anywhere an input may be — the recurrence seam a running statistic is
        written through: ``out[...] = maximum(out.prior, term)`` keeps a
        running max across launches. The host must initialise and preserve
        the plane between launches; reading it without ever committing
        refuses.

        A HOST-FACING API (see :attr:`_SinkRef.prior`): in a traced kernel
        body, a bare read before the plane's first store reaches this
        internally, never the `.prior` spelling itself, which the ast front
        end refuses syntactically before this property is ever reached from
        author source."""
        return Value(Leaf("prior_read", "mutable", self.name, self.ttype))

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            raise HawkError(
                f"{self.name!r}: a mutable plane is committed whole — write "
                "`out[...] = value` (a component-wise store is not a HAWK form)"
            )
        self.write(value)

    def by_key(self, key: Any, value: Any, accumulate: bool) -> None:
        """``hawk.all_samples(out)[key] += value`` (``accum_write``) or
        ``= value`` (``wide_write``): the plane becomes shared, free length."""
        _by_key(self, key, value, accumulate)

    def write(self, value: Any) -> None:
        """Commit ``value`` to this plane (the ``assign``); a reduction over
        samples (``hawk.all_samples(x).sum()``) commits a cell."""
        if _is_stored(value):
            return
        if any(isinstance(k, (AccumWrite, WideWrite)) for k in self.sinks):
            raise HawkError(
                f"{self.name!r} is written per sample (locked) and by key "
                f"(hawk.all_samples({self.name})[key]): one plane has one extent")
        node = self._typed(value)
        if isinstance(node, ReduceSamples):
            self.sinks = list(node.stored(self.name, self.ttype))
        else:
            self.sinks = [Assign(self.name, node, self.ttype)]


class WideOutRef(_SinkRef):
    """A ``WideOut[...]`` parameter: ``buf = v`` or
    ``hawk.all_samples(buf)[k] = v`` (the ``wide_write``)."""

    __slots__ = ()

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            raise HawkError(
                f"{self.name!r}: a wide plane is committed whole — write "
                "`buf = value`, or store by key with "
                "`hawk.all_samples(buf)[key] = value`")
        self.write(value)

    def by_key(self, key: Any, value: Any, accumulate: bool) -> None:
        """``hawk.all_samples(out)[key] = value``."""
        _by_key(self, key, value, accumulate)

    def write(self, value: Any) -> None:
        """Commit ``value`` to the own column."""
        if _is_stored(value):
            return
        self.sinks = [WideWrite(self.name, self._typed(value), None, self.ttype)]
