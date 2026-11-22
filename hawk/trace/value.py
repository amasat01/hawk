# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The traced VALUE — operator overloading from Python expressions to IR nodes.

A :class:`Value` wraps one IR node; every operator mints an
:func:`hawk._ir.make` call, building the DAG as the body runs once. Free
functions below cover what operators can't spell (``norm``, ``dot``,
``cross``, ``select``); three REF objects cover the write side (a mutable
plane, a wide plane) plus a :class:`TableRef` read
with ``t.at(i)``. A :class:`Value` has no truth value — control flow reaches
the IR only through the ``ast`` pass's ``select`` rewrite.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from .. import _ir
from ..errors import HawkError

_F64 = _ir.TensorType((), "f64")
_BOOL = _ir.TensorType((), "bool")
_I32 = _ir.TensorType((), "i32")
INDEX = _I32
INT32 = _ir.TensorType((), "int32")
#: the stored type of an ``Index`` plane: the tag marks it for the sidecar
INDEX32 = _ir.TensorType((), "int32", "index")


def _context() -> Any:
    """The context of the body being traced (a local import: ``lane_axes`` imports this module)."""
    from .lane_axes import current_context

    return current_context()


def node_of(x: Any) -> Any:
    """The IR node (a handle) behind a traced value or a Python literal (a constant of the traced body's context).

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
    if isinstance(x, _ir.Node):
        return x
    if isinstance(x, bool):
        return _ir.const(_context(), x, _BOOL)
    if isinstance(x, (int, float)):
        return _ir.const(_context(), float(x), _F64)
    from .reduce import Reduced

    if isinstance(x, Reduced):
        return x.node
    raise HawkError(
        f"{x!r} is not a traced value, an IR node or a numeric literal — a kernel "
        "body may only combine declared vocabulary and literals"
    )


def index_node(x: Any) -> Any:
    """The IR node behind an INDEX — a traced value, or an integer literal,
    separate from :func:`node_of`, which types a bare ``int`` as ``f64``. An
    index is an address: as ``f64`` it would cast to ``std::size_t`` and
    promote any stride multiplied with it, so it is typed ``i32`` instead
    (the library's contraction routing reads the stride)."""
    if isinstance(x, bool):
        raise HawkError(f"{x!r} is not an index — a bool is a mask, not an address")
    if isinstance(x, int):
        return _ir.const(_context(), int(x), _I32)
    return node_of(x)


def element_indices(k: Any) -> tuple:
    """The index operands of ``v[k]`` at run time: one per axis, each a static integer or a traced integer.

    ``k`` is one key or a tuple of keys. A float is refused rather than truncated: an index is an address, and
    rounding one silently reads the wrong component. The count against the value's axes is checked by the IR."""
    nodes = []
    for key in (k if isinstance(k, tuple) else (k,)):
        if isinstance(key, bool):
            raise HawkError(
                f"a traced value is indexed by an integer component, got {key!r} — "
                "a bool is a mask, not an address")
        node = key.node if isinstance(key, Value) else key
        if isinstance(node, int):
            nodes.append(index_node(node))
        elif isinstance(node, _ir.Node) and _ir.type_of(node).shape == ()\
                and _ir.type_of(node).dtype in _INDEX_DTYPES:
            if _ir.type_of(node).dtype not in ("i32", "i64"):     # an int plane value (hm.int32) indexes as i32
                node = _ir.make("cast", (node,), "i32")
            nodes.append(node)
        else:
            raise HawkError(
                f"a traced value is indexed by a STATIC integer component or by a "
                f"traced INTEGER index, got {key!r}. A lowered `for`'s own index is "
                "the second form (`v[k]` inside the loop body); a rank-0 "
                "floating value is neither — an index is an address, and rounding "
                "one silently reads the wrong component (a cross-sample read of a "
                "whole plane is a Table's at())")
    return tuple(nodes)


def _operands(args: tuple) -> tuple:
    """The operand nodes of one traced op, with literals typed BY COMPANY: a
    bare ``int`` is ``f64`` alone, but when every traced operand is integral
    it types as an integer instead — otherwise an index like
    ``entry * cols + row`` would promote to ``Real`` and lose its stride
    derivation (the library's contraction routing)."""
    traced = [a for a in args if isinstance(a, (Value, _ir.Node))]
    dtypes = {_ir.type_of(a.node if isinstance(a, Value) else a).dtype for a in traced}
    plane = next((d for d in _ir.INT_PLANE_DTYPES if d in dtypes), None)
    if plane is not None:
        return tuple(int_literal(a, plane) if isinstance(a, int) and not isinstance(a, bool)
                     else node_of(a) for a in args)
    integral = bool(dtypes) and dtypes <= {"i32", "i64"}
    return tuple(index_node(a) if integral and isinstance(a, int)
                 and not isinstance(a, bool) else node_of(a) for a in args)


#: What a node's ``field_kind`` answers for a field that holds an integer.
_INTEGER_FIELD = 2


def literal_int(node: Any) -> int | None:
    """The integer a constant node holds, ``None`` for any other node (or another literal)."""
    if _ir.is_a(node, "const") and node.field_kind("literal") == _INTEGER_FIELD:
        return node.field_int("literal")
    return None

#: The value range of each integer plane dtype.
_INT_RANGE = {"int32": (-2**31, 2**31 - 1), "uint32": (0, 2**32 - 1)}


def int_literal(x: int, dtype: str) -> Any:
    """A Python ``int`` beside an ``int32``/``uint32`` value, typed as that
    value: a literal outside the type's range is refused, never wrapped."""
    lo, hi = _INT_RANGE[dtype]
    if not lo <= x <= hi:
        raise HawkError(
            f"the literal {x} is outside the {dtype} range [{lo}, {hi}]; write a "
            f"value inside it (an {dtype} wraps only through arithmetic)")
    return _ir.const(_context(), int(x), _ir.TensorType((), dtype))


def _op(kind: str, *args: Any, literal: Any = None) -> Value:
    return Value(_ir.make(kind, _operands(args), literal))


def _is_condition(x: Any) -> bool:
    return isinstance(x, bool) or (isinstance(x, (Value, _ir.Node)) and _ir.type_of(node_of(x)).dtype == "bool")


def _logic(kind: str, a: Any, b: Any) -> Value:
    """``a & b``, ``a | b``, ``a ^ b``: on two conditions the logical ``land``/``lor``/``ne``, whose C++ is a bool; on
    integers the bitwise kind (a bool beside an integer stays refused by it)."""
    if _is_condition(a) and _is_condition(b):
        return _op({"band": "land", "bor": "lor", "bxor": "ne"}[kind], a, b)
    return _op(kind, a, b)


class Value:
    """One traced IR value: every operator on it mints a node."""

    __slots__ = ("node",)

    def __init__(self, node: Any) -> None:
        self.node = node

    @property
    def ttype(self) -> _ir.TensorType:
        """The node's tensor type."""
        return _ir.type_of(self.node)

    def at(self, *positional: Any, edge: str = "clamp", **named: Any) -> Value:
        """``u.at(i)`` / ``u.at(row=…, col=…)``: this per-lane input read at another lane, by flat lane index or by
        every ``Grid`` axis. ``edge`` is what a coordinate outside its axis (a flat index outside the launch) reads:
        ``"clamp"`` (the edge lane), ``"wrap"`` (modulo the extent) or ``"zero"``."""
        from .lane_axes import read_at

        return read_at(self, positional, named, edge)

    def near(self, *, edge: str = "clamp", **offsets: Any) -> Value:
        """``u.near(col=1)``: this per-lane input read at this lane's coordinates moved by the integer offsets on the
        named ``Grid`` axes; an axis not named keeps this lane's coordinate. ``edge`` is as in :meth:`at`."""
        from .lane_axes import near

        return near(self, offsets, edge)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Value {self.node.cls} {self.ttype}>"

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
        # the division — the library's contraction routing reads it as a stride.
        if (isinstance(o, (int, float)) and not isinstance(o, bool)
                and self.ttype.dtype not in ("i32", "i64", "bool", *_ir.INT_PLANE_DTYPES)
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
        return _logic("band", self, o)

    def __rand__(self, o: Any) -> Value:
        return _logic("band", o, self)

    def __or__(self, o: Any) -> Value:
        return _logic("bor", self, o)

    def __ror__(self, o: Any) -> Value:
        return _logic("bor", o, self)

    def __xor__(self, o: Any) -> Value:
        return _logic("bxor", self, o)

    def __rxor__(self, o: Any) -> Value:
        return _logic("bxor", o, self)

    def __lshift__(self, o: Any) -> Value:
        return _op("shl", self, o)

    def __rlshift__(self, o: Any) -> Value:
        return _op("shl", o, self)

    def __rshift__(self, o: Any) -> Value:
        return _op("shr", self, o)

    def __rrshift__(self, o: Any) -> Value:
        return _op("shr", o, self)

    def __invert__(self) -> Value:
        if self.ttype.dtype == "bool":
            return _op("lnot", self)
        return _op("invert", self)

    def __matmul__(self, o: Any) -> Value:
        other = node_of(o)
        return _op("mv" if len(_ir.type_of(other).shape) == 1 else "mm", self, other)

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
        """``v[k]`` — one component of a rank>=1 value; a value of rank two or more takes ``v[i, j, ...]``, one
        index per axis.

        STATIC integers are ``component``, checked at trace time; TRACED integers (mixed with static ones
        for the other axes) are ``component_at`` — needed inside a lowered ``for``, where
        ``k`` is a symbolic loop index, not a
        Python int (it materialises into an ``Item`` first, since aether's
        EXPRESSION only carries compile-time ``eval<Is...>``). A float is
        refused rather than truncated: rounding would silently read the
        wrong component."""
        keys = k if isinstance(k, tuple) else (k,)
        statics = all(isinstance(x, int) and not isinstance(x, bool) for x in keys)
        if statics and (not isinstance(k, tuple) or len(k) == len(self.ttype.shape) >= 2):
            return _op("component", self, literal=k)        # a static entry, `v[i]`, `m[i, j]` or `t[a, b, c]`
        if not self.ttype.shape and isinstance(k, Value):
            raise HawkError(
                "x[key] indexes a component and a rank-0 value has none; another "
                "lane's input is read with x.at(key)")
        return Value(_ir.make("component_at", (self.node, *element_indices(k))))

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
            "reaches a Python branch or comprehension the pass does not own — use "
            "select(cond, a, b). Combine conditions with `and` / `or` / `not` or with & | ~, and parenthesise "
            "each comparison around & and |: `x >= 0 & x <= 3` parses as `x >= (0 & x) <= 3`, "
            "so write `(x >= 0) & (x <= 3)`"
        )


def _cast(x: Any, dtype: str) -> Value:
    node = node_of(x)
    if _ir.type_of(node).dtype == dtype:
        return Value(node)
    return Value(_ir.make("cast", (node,), dtype))


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
    """Every static index of ``shape``, row-major, as the ``component`` literal reading that
    entry: an integer for a vector, one integer per axis for a value of rank two or more."""
    if len(shape) == 1:
        return list(range(shape[0]))
    return list(itertools.product(*(range(extent) for extent in shape)))


def _elementwise(kind: str, *args: Any) -> Value:
    """``kind`` applied component-wise. Rank-0 operands mint the op
    itself; operands of any rank are mapped entry by entry (a rank-0
    operand broadcasts) and re-assembled with ``vec``, so the emitter only
    ever spells the rank-0 call. The shapes combine as the arithmetic
    operators' do — equal shapes, or one side rank-0 — and refuse in the
    same words otherwise."""
    nodes = _operands(args)
    shape: tuple = ()
    for n in nodes:
        shape = _broadcast_shape(kind, shape, _ir.type_of(n).shape)
    if not shape:
        return Value(_ir.make(kind, nodes))
    parts = []
    for k in _entries(shape):
        ops = [n if _ir.type_of(n).shape == () else _ir.make("component", (n,), k)
               for n in nodes]
        parts.append(_ir.make(kind, ops))
    return Value(_ir.make("vec", parts, shape if len(shape) >= 2 else None))


def _broadcast_shape(kind: str, a: tuple, b: tuple) -> tuple:
    """The shape two elementwise operands combine to — the arithmetic
    operators' rule, with its refusal."""
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
               "``atan2(a, b)``, component-wise; operands of any rank.")
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


_D = "Component-wise; operands of any rank."
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
    branches' shape, and a rank-0 branch broadcasts. A bare ``int`` branch takes the type of the other branch
    when that is an integer (as ``x + 1`` does)."""
    c = node_of(cond)
    a, b = _operands((on_true, on_false))
    shape = _broadcast_shape("select", _broadcast_shape(
        "select", _ir.type_of(c).shape, _ir.type_of(a).shape), _ir.type_of(b).shape)
    if shape:
        # a rank-0 branch broadcasts like an arithmetic operand (numpy's
        # ``where``); the mask is rank-0 (one choice per sample) or the
        # branches' own shape (one choice per entry)
        a, b = (_ir.make("splat", (x,), shape) if _ir.type_of(x).shape == () else x
                for x in (a, b))
    ttype = _ir.type_of(a)
    if ttype != _ir.type_of(b):
        raise HawkError(f"select: the branches disagree — {ttype!r} vs {_ir.type_of(b)!r}; "
                        "cast one with m.float64(...) or write the literal in the other's type")
    return Value(_ir.make_node("Select", ttype, [c, a, b]))


def dispatch(kind: Any, branches: Any, strategy: str = "switch") -> Value:
    """``branches[clamp(kind, 0)]`` — the finite per-element dispatch.

    ``kind`` is a per-element integer VALUE, never a Python literal — a body
    that knows the branch at TRACE time is writing ordinary control flow, not
    a dispatch. ``strategy`` names one of three lowerings
    (``predicated``/``switch``/``segmented``), or ``auto`` — the one value a
    launching runtime's chooser resolves (:func:`hawk.resolve_dispatch`;
    until then it renders as ``switch`` and counts the branch mix); the
    default ``switch`` is what a compiler already picks for a branch-heavy
    chain.
    The library checks the type and the arity."""
    bs = [node_of(b) for b in branches]
    ttype = _ir.type_of(bs[0]) if bs else _ir.TensorType((), "f64")
    selector = node_of(kind)
    if _ir.type_of(selector).dtype in _ir.INT_PLANE_DTYPES:
        selector = _ir.make("cast", (selector,), "i32")
    return Value(dispatch_node(selector, bs, strategy, ttype))


#: The strategies a dispatch can be pinned to; ``switch`` is the core kind itself.
DISPATCH_STRATEGIES = _ir.dispatch_strategies()


def dispatch_node(selector: Any, branches: Any, strategy: str, ttype: Any) -> Any:
    """The dispatch of ``selector`` over ``branches`` (each of ``ttype``) pinned to ``strategy``.

    ``switch`` is the core kind; any other strategy is the dialect kind ``<strategy>_dispatch``, which the
    context declares. The library checks the types and the arity."""
    if strategy not in DISPATCH_STRATEGIES:
        raise HawkError(f"dispatch: unknown strategy {strategy!r}; declared are {DISPATCH_STRATEGIES}")
    cls = "Dispatch" if strategy == "switch" else f"{strategy.capitalize()}Dispatch"
    return _ir.make_node(cls, ttype, [selector, *branches])


def dispatch_strategy(node: Any) -> str:
    """The strategy a dispatch node is pinned to: the core class is ``switch``, a pinned one its kind's word."""
    return node.cls.removesuffix("Dispatch").lower() or "switch"


def n_samples() -> Value:
    """The readable sample count: the TRUE ``nSamples``, never ``count``."""
    return Value(_ir.leaf(_context(), "nsamples", "nsamples", "n_samples", _I32))


def sample_index() -> Value:
    """This lane's GLOBAL sample index, as a value (the ``own(i)``): it is
    ``base + flat`` under a partition, so an expression built on it is
    partition-invariant — safe for the oracle to compare — and lets a
    2-D-domain kernel recover the coordinates a flattened launch folded
    together."""
    return Value(_ir.make_node("SampleIndex", _I32, context=_context()))


def sample_position() -> Value:
    """This sample's position ``t`` in the launch, as a value: ``base + flat``.

    Equal to :func:`sample_index` in a plain launch. Under a lane set
    (``Kind(..., lanes=IndexSet())``) :func:`sample_index` is the lane the map
    names, ``map[t]``, and this is ``t`` itself, the slot a compacted output
    ``out[sample_position()]`` is written at."""
    return Value(_ir.make_node("SamplePosition", _I32, context=_context()))


def _random(kind: str, seed: Any, counter: Any) -> Value:
    """The shared body of :func:`random_uniform`/:func:`random_normal`.

    ``seed``/``counter`` go through :func:`index_node`, not :func:`node_of`:
    a bare ``int`` must type as INTEGER (``i32``) here, never the ``f64`` a
    literal means elsewhere, since a random op's operands are stream
    addresses, not real-valued arithmetic."""
    return _op(kind, index_node(seed), index_node(counter))


def random_uniform(seed: Any, counter: Any) -> Value:
    """A reproducible ``U(0, 1)`` draw, open at both ends.

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


def widened(node: Any) -> Any:
    """``node``, an ``int32`` read of an ``Index`` plane, as the 64-bit index the body computes in."""
    return _ir.make("cast", (node,), "i32")


class TableRef:
    """A ``Table``/``Staged``/``Wide`` parameter: read with ``at()``.

    Two read spellings, one element. ``t.at(expr)`` is the read of a plane with
    one axis; ``t.at(row=…, col=…)`` names every declared axis. Either way the
    read is ONE ``At`` over the plane's leaf with one index per declared axis
    and the ``clamp`` edge rule, so a read outside an axis reads that axis's
    edge element."""

    __slots__ = ("leaf", "name", "axes", "widen")

    def __init__(self, name: str, ttype: _ir.TensorType, *, role: str = "lookup",
                 axes: Any = ()) -> None:
        kind = "table_read" if role == "lookup" else "vocab_read"
        #: an ``Index`` payload is stored 32-bit and widened at the read
        self.widen = ttype == INDEX
        self.axes = tuple(axes)
        self.leaf = _ir.leaf(_context(), kind, role, name, INDEX32 if self.widen else ttype,
                             axes=self.axes)
        self.name = name

    def at(self, *positional: Any, **named: Any) -> Value:
        """The read of this plane: ``at(expr)`` on a plane with one axis or, as a position in storage order, on a plane
        with several; ``at(<axis>=…, …)`` naming every axis of a plane declared with several. A real-valued index
        truncates to an integer first."""
        if len(positional) == 1 and not named and len(self.axes) > 1:
            indices, in_range = self._unflattened(positional[0]), True
        else:
            indices = self._indices(positional, named)
            # a literal or compile-time loop index inside a static extent needs no edge rule on that axis
            in_range = all(extent is not None and _inside(index, extent)
                           for index, (_label, extent) in zip(indices, self._declared_axes()))
        read = _ir.make_node("At", _ir.type_of(self.leaf), [self.leaf, *indices],
                             edge="unchecked" if in_range else "clamp")
        return Value(widened(read) if self.widen else read)

    def sample(self, *coordinates: Any, filter: str = "linear", edge: str = "clamp") -> Value:
        """The table read at real coordinates in index units, one per axis (one to three): a whole number is exactly
        the stored element. ``filter`` is ``"linear"`` (the weighted corners of the cell the point is in) or
        ``"nearest"`` (the element at ``floor(x + 0.5)``, a half rounding up); ``edge`` is what a read past an axis
        gives: ``"clamp"`` (the edge element), ``"wrap"`` or ``"zero"``. The derivative with respect to the table is
        the weighted accumulate, with respect to a coordinate the slope between neighbours (``"nearest"`` has none).

        Example: ``h = grid.sample(x, y)`` interpolates the 2-D table ``grid`` linearly at ``(x, y)``."""
        element = _ir.type_of(self.leaf)
        reals = []
        for x in coordinates:
            node = node_of(x)
            reals.append(_ir.make("cast", (node,), element.dtype)
                         if _ir.type_of(node).dtype in ("i32", "i64") else node)
        read = _ir.make_node("Sample", element, [self.leaf, *reals], filter=filter, edge=edge, hardware=False)
        written = _ir.expand_sample(read)
        return Value(widened(written) if self.widen else written)

    def __getitem__(self, i: Any) -> Value:
        """``t[i]``: the read of a plane with one axis; ``t[e, a]``: one
        index per declared axis, in declared order, the same as ``t.at(e=…, a=…)``."""
        if not isinstance(i, tuple):
            return self.at(i)
        labels = [label for label, _extent in self.axes]
        if len(i) != len(labels):
            raise HawkError(
                f"{self.name!r}: t[...] takes one index per declared axis "
                f"{labels or '(none: this plane has one axis, read t[i])'}, got {len(i)}")
        return self.at(**dict(zip(labels, i)))

    def _unflattened(self, flat: Any) -> list:
        """One index per axis for the one-index read ``t[flat]`` of a plane with several axes: ``flat`` is the position in
        storage order (last axis fastest), clamped to ``[0, elements - 1]``, then split by floor division and remainder over
        the trailing extents, which are static by declaration. The first axis may be a runtime one: its extent only enters
        the clamp, as the plane's own extent of that axis."""
        trailing = [extent for _label, extent in self.axes[1:]]
        scalar = _ir.TensorType((), "i32")
        context = self.leaf.ctx
        first = (_ir.const(context, self.axes[0][1], scalar) if self.axes[0][1] is not None
                 else _ir.make_node("AxisExtent", scalar, [self.leaf], axis=0))
        size = first
        for extent in trailing:
            size = _ir.make("mul", (size, _ir.const(context, extent, scalar)))
        last = _ir.make("sub", (size, _ir.const(context, 1, scalar)))
        position = _ir.make("max", (_ir.const(context, 0, scalar),
                                    _ir.make("min", (_integer_index(index_node(flat)), last))))
        indices, stride = [], 1
        for extent in reversed(trailing):
            step = _ir.make("floordiv", (position, _ir.const(context, stride, scalar)))
            indices.append(_ir.make("mod", (step, _ir.const(context, extent, scalar))))
            stride *= extent
        indices.append(_ir.make("floordiv", (position, _ir.const(context, stride, scalar))))
        return indices[::-1]

    def _declared_axes(self) -> tuple:
        """The axes a read addresses: the declared ones, else the one run-time axis of a flat plane."""
        return self.axes or (("", None),)

    def _indices(self, positional: tuple, named: dict) -> list:
        """The integer index nodes of a read, one per axis in declared order."""
        if named and positional:
            raise HawkError(
                f"{self.name!r}: at() takes a flat index OR every declared axis by "
                "name, never both")
        if not named:
            if len(positional) != len(self._declared_axes()):
                raise HawkError(
                    f"{self.name!r}: at() takes ONE index, got {len(positional)} — "
                    + (f"the axes {[label for label, _ in self.axes]} are read "
                       "at(<axis>=…, …)" if self.axes else "this plane has one axis"))
            return [_integer_index(index_node(index)) for index in positional]
        if not self.axes:
            raise HawkError(
                f"{self.name!r} declares no named axes, so it is read at a flat "
                f"index: at(expr). Named axes are declared "
                'Table["row":rows, "col":cols]')
        declared = [label for label, _extent in self.axes]
        missing = [d for d in declared if d not in named]
        unknown = sorted(k for k in named if k not in set(declared))
        if missing or unknown:
            raise HawkError(
                f"{self.name!r}: at() must name every declared axis exactly once "
                f"(declared {declared})"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else "")
                + " — a partial address would silently read the wrong row")
        return [_integer_index(index_node(named[label])) for label in declared]


def _integer_index(index: Any) -> Any:
    """``index`` as an integer node: a real-valued index truncates, as every read always did."""
    if _ir.type_of(index).dtype not in ("i32", "i64"):
        return _ir.make("cast", (index,), "i32")
    return index


def _inside(index: Any, extent: int) -> bool:
    """Whether ``index`` is provably in ``[0, extent)``: a literal, or a compile-time loop's range."""
    lo = hi = literal_int(index)
    if lo is None and _ir.is_a(index, "loop_index") and index.field_kind("bounds"):
        lo, hi = index.field_ints("bounds")
    return lo is not None and lo >= 0 and hi < extent


class SharedValue(Value):
    """A ``Param[Vector[W]]`` / ``Param[Matrix[R,C]]``: ONE value every lane reads.

    The value lives in a ``lookup`` plane of its cells, row-major. ``p[j]`` and
    ``p[i, j]`` (a static or traced integer) are reads of ONE cell from memory,
    so a wide parameter indexed in a loop is never built as a register value. Any
    other use (``p @ x``, ``p + q``, ``m.dot(p, x)``) builds the whole value from its
    cells at that point."""

    __slots__ = ("ref", "shape", "_whole")

    def __init__(self, name: str, shape: tuple, axes: tuple) -> None:  # noqa: D107 - see the class
        self.ref = TableRef(name, _F64, role="lookup", axes=axes)
        self.shape = tuple(shape)
        self._whole: Any = None

    @property  # type: ignore[override]
    def node(self) -> Any:
        """The whole value, built from its cells on first use."""
        if self._whole is None:
            cells = [_ir.make_node("At", _F64, [self.ref.leaf, *(_ir.const(_context(), c, _I32) for c in cell)],
                                   edge="unchecked")
                     for cell in itertools.product(*(range(extent) for extent in self.shape))]
            self._whole = _ir.make("vec", cells, self.shape if len(self.shape) >= 2 else None)
        return self._whole

    def _cell(self, k: Any, extent: int, axis: str) -> Any:
        if isinstance(k, bool):
            raise HawkError(f"{self.ref.name!r}: a bool is a mask, not an index")
        if isinstance(k, int):
            if not 0 <= k < extent:
                raise HawkError(
                    f"{self.ref.name!r}: index {k} is outside the {axis} extent {extent}")
        return k

    def __getitem__(self, k: Any) -> Value:
        """``p[j]`` (a vector) or ``p[i, j, ...]`` (one index per axis): one cell, read from memory."""
        rank = len(self.shape)
        if rank == 1 and not isinstance(k, tuple):
            return self.ref.at(self._cell(k, self.shape[0], "component"))
        if rank >= 2 and isinstance(k, tuple) and len(k) == rank:
            labels = [label for label, _extent in self.ref.axes]
            return self.ref[tuple(self._cell(x, extent, label)
                                  for x, extent, label in zip(k, self.shape, labels))]
        return Value.__getitem__(self, k)


class Row:
    """One row of a ``Rows`` parameter: ``row.length`` entries of the values plane from offset ``row.start``."""

    __slots__ = ("start", "length", "_values")

    def __init__(self, values: TableRef, start: Value, length: Value) -> None:
        """Hold the values plane, the row's first position and its entry count (both traced integers)."""
        self._values, self.start, self.length = values, start, length

    def entry(self, k: Any) -> Value:
        """The ``k``-th entry of the row, read at ``start + k`` from the values plane."""
        return self._values.at(self.start + k)

    def bounded(self, bound: int) -> Any:
        """Refuse a call outside a loop header: the loop ``for x in row.bounded(K):`` visits ``min(length, K)`` entries
        and is rewritten before the body runs, so this is reached only from a helper the rewrite never sees."""
        raise HawkError(
            f"row.bounded({bound!r}) is the header of a `for` inside a kernel body: "
            "`for x in row.bounded(K):` visits at most K entries; in a helper, pass the row's entries instead"
        )


class RowsView:
    """What a kernel body sees for a ``Rows`` parameter: ``rows[i]`` is the row ``i``, a :class:`Row`."""

    __slots__ = ("name", "_offsets", "_values")

    def __init__(self, name: str, *, offsets: TableRef, values: TableRef) -> None:
        """Relate the traced offsets plane to the traced values plane of the parameter ``name``."""
        self.name, self._offsets, self._values = name, offsets, values

    def __getitem__(self, i: Any) -> Row:
        """The row ``i``: its start is ``offsets[i]`` and its length ``offsets[i + 1] - offsets[i]``."""
        start = self._offsets.at(i)
        return Row(self._values, start, self._offsets.at(i + 1) - start)


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
_BOOL0 = _ir.TensorType((), "bool")


class TerminatedRef(Value):
    """A ``Terminated`` mask parameter.

    Read like any input (it IS the mask's leaf), and FINISHED by ONE
    statement ``mask[...] = cond`` (a traced rank-0 bool): the mask becomes
    ``mask | cond`` (set-only, monotone), rewritten by the ``ast`` pass into
    a name committed once at the end through :meth:`write`."""

    __slots__ = ("sinks",)

    def __init__(self, node: Any) -> None:
        super().__init__(node)
        self.sinks: list[Any] = []

    def unset(self) -> Value:
        """The "does not finish on this path" value an ``if`` merges against."""
        return Value(_ir.const(_context(), False, _BOOL0))

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            name = self.node.field_text("name")
            raise FinishRefusal(
                f"{name!r}: a mask is finished whole — write `{name}[...] = cond`")
        self.write(value)

    def write(self, value: Any) -> None:
        """Record the finish: the sample's mask becomes ``mask | value``."""
        name = self.node.field_text("name")
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
        if _ir.type_of(node) != _BOOL0:
            raise FinishRefusal(
                f"{name!r} is finished with a value typed {_ir.type_of(node)!r}; the "
                "condition is a traced rank-0 bool")
        self.sinks = [_ir.make_node("Finish", _BOOL0, [node], name=name)]


_INDEX_DTYPES = ("i32", "i64") + tuple(_ir.INT_PLANE_DTYPES)


def _is_stored(value: Any) -> bool:
    from .reduce import _Stored
    return isinstance(value, _Stored)


#: The operations a by-key update may name, and the ones that fold commutatively (a plane takes several statements of one fold).
_FOLDS = ("add", "min", "max", "and", "or", "xor")


def by_key(
    ref: Any, indices: list, value: Any, op: str, contract: str = "auto", *, conflict: str = "unique",
    expected: Any = None, returns_old: bool = False, edge: str = "unchecked", several: bool = False,
) -> Any:
    """One update by key into ``ref`` at ``indices``, one per axis of the sink: an ``Update`` of the operation ``op``
    (``store``, ``add``, ``min``, ``max``, ``and``, ``or``, ``xor``, ``exchange`` or ``compare_exchange``, which also takes the
    ``expected`` value). Returns the old value the element held, as a value, when ``returns_old``. A vector or matrix payload
    into a sink declared with named axes is summed per key, payload first. ``edge`` is the rule for a key outside the plane
    (``zero`` drops the update); ``several`` states that this is one of many updates of one operation into the plane, which
    the plane then holds together (an append is one of them)."""
    where = f"hawk.scatter({ref.name})[key]"
    if any(not _ir.is_a(k, "update") for k in ref.sinks):
        raise HawkError(
            f"{ref.name!r} is written per lane (locked) and by key ({where}): one "
            "plane has one extent")
    for index in indices:
        key_type = _ir.type_of(index)
        if key_type.shape != () or key_type.dtype not in _INDEX_DTYPES:
            raise HawkError(f"{where}: key is an Index or an integer value, got {key_type!r}")
    if op == "or" and (ref.ttype.shape or ref.ttype.dtype not in _ir.INT_PLANE_DTYPES):
        raise HawkError(
            f"{where}: |= sets bits of an Int32 or UInt32 plane (a word per key), "
            f"{ref.name!r} is declared {ref.ttype!r}")
    kind = "wide_out" if op == "store" else "accum"
    if ref.ttype.shape and not ref.axes:
        raise HawkError(
            f"{where}: a vector payload is written by key into a sink declared with "
            f"named axes (Mutable[Vector[w], \"bin\":n]); a flat sink takes one "
            f"Mutable[Scalar] plane per component ({ref.name}0, ...) and "
            f"hawk.scatter({ref.name}0)[key] += u[0]")
    repeated = several and (ref.by_key_kind, ref.by_key_op) == (kind, op)
    if ref.by_key_kind is not None and not repeated and (
            ref.by_key_kind != kind or op not in _FOLDS or ref.by_key_op != op):
        raise HawkError(
            f"{ref.name!r} is written by key twice with different operations, or twice by an operation that is not a "
            "fold: one plane takes one by-key statement, or several of one of += , .min, .max, .and_, .xor, |=")
    typed = ref._typed(value)
    fields = {"op": op, "conflict": conflict, "returns_old": returns_old}
    if op == "add":
        fields.update(_keyed_fields(contract))
        if returns_old:
            fields["summation"] = "plain"
    elif op in ("min", "max") and contract == "ordered":
        fields["order"] = "lane"
    operands = [typed, *([ref._typed(expected)] if op == "compare_exchange" else []), *indices]
    node = _ir.make_node("Update", ref.ttype, operands, name=ref.name, **fields, reduced=None, edge=edge,
                         axes=ref.axes)
    ref.sinks = [node] if op in ("store", "exchange", "compare_exchange") and not several else [*ref.sinks, node]
    ref.by_key_kind, ref.by_key_op = kind, op
    return Value(node) if returns_old else None


def _keyed_fields(contract: str) -> dict:
    """The fields of the keyed accumulate an author's ``contract`` states: a keyed sum the author left to the library is
    compensated."""
    return _ir.contract_fields("compensated" if contract == "auto" else contract)


def _payload_axes(ref: Any) -> tuple:
    """The leading axes of a sink whose payload is a vector or a matrix: one per payload dimension, labelled so
    that none repeats a declared label."""
    taken = {label for label, _extent in ref.axes}
    axes = []
    for stem, extent in zip(("component",) if len(ref.ttype.shape) == 1 else ("row", "col"), ref.ttype.shape):
        while stem in taken:
            stem += "_"
        taken.add(stem)
        axes.append((stem, extent))
    return tuple(axes)


def retired_sink_method(name: str, method: str) -> str:
    """The refusal for ``.add`` / ``.write`` / ``.contribute`` on an output."""
    if method == "contribute":
        return (f"{name!r}: write {name} = hawk.sum(v) "
                "(or hawk.max/min/mean)")
    return (f"{name!r}: write hawk.scatter({name})[key] += v (accumulate by "
            f"key) or hawk.scatter({name})[key] = v (store by key)")


def _block_write(ref: Any, block: Any) -> Any:
    """The store of a block-shaped keyed sum to the plane of ``ref``: the plane's axes are the payload's, labelled, then the
    declared ones (``(*payload, *kept)``, payload first)."""
    return _ir.make_node("BlockSumWrite", _ir.TensorType((), _ir.type_of(block).dtype), [block], name=ref.name,
                         axes=_payload_axes(ref) + tuple(ref.axes))


class _SinkRef:
    """Shared write-side state: the sinks a body commits through one plane.
    ``axes`` are the sink's declared named axes ``((label, extent), ...)``."""

    __slots__ = ("name", "ttype", "sinks", "by_key_kind", "by_key_op", "axes", "counts", "winner_of")

    def __init__(self, name: str, ttype: _ir.TensorType, axes: tuple = (), counts: str = "", winner_of: str = "") -> None:
        self.name = name
        self.ttype = ttype
        self.axes = axes
        #: The name of the appended plane this one counts (``hawk.append`` makes it), else empty.
        self.counts = counts
        #: The name of the plane this one ranks (a minimum or maximum in lane order makes it), else empty.
        self.winner_of = winner_of
        self.sinks: list[Any] = []
        self.by_key_kind: str | None = None
        self.by_key_op: str | None = None

    def _typed(self, value: Any) -> Any:
        node = _ir.as_matrix(node_of(value))
        if _ir.type_of(node) != self.ttype:
            raise HawkError(
                f"{self.name!r} is declared {self.ttype!r} but the committed value is "
                f"typed {_ir.type_of(node)!r}"
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
        return Value(_ir.leaf(_context(), "prior_read", "mutable", self.name, self.ttype))

    def zero_cover(self, spec: Any) -> Value:
        """The start value of a body that stores elements before it reads them: zeros
        when :func:`hawk.trace.cover.cover_of` proves the body stores EVERY element
        of this plane before reading any (nothing of the launch-start value
        survives, so the kernel differentiates), else the launch-start value (a
        partial cover, a read of an element not yet stored, or a body the analysis
        cannot decide keep it). Reached by the ast front end only."""
        from .cover import cover_of

        shape = self.ttype.shape
        if len(shape) == 1 and cover_of(spec, shape[0]) == frozenset(range(shape[0])):
            return vec(*([0.0] * shape[0]))
        return self.prior

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            raise HawkError(
                f"{self.name!r}: a mutable plane is committed whole — write "
                "`out[...] = value` (a component-wise store is not a HAWK form)"
            )
        self.write(value)

    def write(self, value: Any) -> None:
        """Commit ``value`` to this plane (the ``assign``); a reduction over
        every lane (``hawk.sum(x)``) commits a cell, one over some lane axes
        (``hawk.sum(x, over=...)``) adds into the sink by the kept coordinates."""
        from .reduce import Reduced

        if _is_stored(value):
            return
        if any(_ir.is_a(k, "update") for k in self.sinks):
            raise HawkError(
                f"{self.name!r} is written per lane (locked) and by key "
                f"(hawk.scatter({self.name})[key]): one plane has one extent")
        if isinstance(value, Reduced):
            self._reduced(value)
            return
        node = self._typed(value)
        if _ir.is_a(node, "reduce_samples"):
            self.sinks = list(_ir.cells_stored(node, self.name))
        elif _ir.is_a(node, "block_sum"):
            self.sinks = [_block_write(self, node)]
        else:
            self.sinks = [_ir.make_node("Assign", self.ttype, [node], name=self.name, axes=tuple(self.axes))]

    def _addressed(self, sinks: list, coordinates: tuple, reduction: Any) -> list:
        """The accumulates ``sinks`` of a sum over some lane axes at the lane's coordinate on each kept axis, as the scatter
        ``hawk.scatter(plane)[coordinates]`` states it. A vector or matrix payload keeps one sink per entry, addressed at the
        entry's own coordinates first: the plane's axes are the payload's, then the kept ones. Each entry names the
        ``reduction`` it stores, as a scalar sum does, so the library reads one flat key for the lane's run."""
        if not self.ttype.shape:
            return [_ir.addressed(sink, coordinates, self.axes) if _ir.is_accumulate(sink) else sink for sink in sinks]
        entries = itertools.product(*(range(extent) for extent in self.ttype.shape))
        axes = _payload_axes(self) + tuple(self.axes)
        key_type = _ir.type_of(coordinates[0])
        return [
            _ir.update(
                self.name, sink.operands[0], axes=axes, reduced=reduction, **_ir.operation_fields(sink),
                indices=[*(_ir.const(sink.ctx, k, key_type) for k in entry), *coordinates],
            )
            for sink, entry in zip(sinks, entries)
        ]

    def _reduced(self, value: Any) -> None:
        """``out = hawk.sum(x, over=...)``: ``out`` declares the kept axes."""
        kept = [(label, size) for label, size in self.axes]
        if [label for label, _ in kept] != [label for label, _ in value.kept] or any(
                extent is not None and extent != size
                for (_, size), (_, extent) in zip(kept, value.kept)):
            want = ", ".join(f'"{label}"' if e is None else f'"{label}":{e}'
                             for label, e in value.kept)
            raise HawkError(
                f"{self.name!r} = hawk.sum(x, over=...): the sum keeps the lane axes "
                f"[{want}], so {self.name!r} is declared Mutable[..., {want}] "
                f"(the payload first), got the axes {kept}")
        if value.ttype != self.ttype:
            raise HawkError(f"{self.name!r} is declared {self.ttype!r} but the sum is "
                            f"typed {value.ttype!r}")
        node = value.node
        if self.ttype.shape and all(size is not None for _label, size in kept):
            extent = 1
            for _label, size in kept:
                extent *= size
            if _ir.is_a(node, "block_sum"):
                value, key, _count = node.operands
                node = _ir.rebuild(node, (value, key, _ir.const(node.ctx, extent, _ir.type_of(key))))
            else:
                key_type = _ir.type_of(node.field_nodes("key")[0])
                node = _ir.recount(node, _ir.const(node.ctx, extent, key_type))
        self.sinks = ([_block_write(self, node)] if _ir.is_a(node, "block_sum")
                      else self._addressed(_ir.cells_stored(node, self.name), value.coordinates, node))
        self.by_key_kind = "accum"
        self.by_key_op = "add"


class WideOutRef(_SinkRef):
    """A ``WideOut[...]`` parameter: ``buf = v`` or
    ``hawk.scatter(buf)[k] = v`` (an ``Update`` that stores)."""

    __slots__ = ()

    def __setitem__(self, key: Any, value: Any) -> None:
        if key is not Ellipsis:
            raise HawkError(
                f"{self.name!r}: a wide plane is committed whole — write "
                "`buf = value`, or store by key with "
                "`hawk.scatter(buf)[key] = value`")
        self.write(value)

    def write(self, value: Any) -> None:
        """Commit ``value`` to the own column."""
        if _is_stored(value):
            return
        self.sinks = [_ir.make_node("Update", self.ttype, [self._typed(value)], name=self.name, op="store")]
