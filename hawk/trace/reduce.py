# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reductions and named axes on traced values.

A value is ONE lane's, so ``x.sum(axis=...)`` folds the value's own axes:
the components of a vector, the rows or columns of a matrix. Across lanes there
are two doors: ``hawk.sum/max/min/mean(x, over=, where=, contract=)`` reduces over
lane axes, and ``hawk.scatter(sink)[key] += v`` adds into a sink by key (another
lane's input is read with ``u.at(i)``). The labelled views ``.components``,
``.rows`` and ``.columns`` read a component, a row or a column by name.
"""

from __future__ import annotations

from typing import Any

from .. import _ir
from ..errors import HawkError
from .value import Value, _op, _operands, index_node, n_samples, node_of, real, select, vec


def _block_sum(value: Any, key: Any, count: Any, contract: str, kept: tuple = ()) -> Any:
    """The sum of ``value`` over the lanes of ``key`` (``count`` keys), as the block-shaped kind the library lowers whole."""
    return _ir.make_node("BlockSum", _ir.type_of(value), [value, key, count], contract=contract, kept=kept)


def _axis(value: Value, axis: Any, op: str):
    """``axis`` as the value's own axis number (None for all), or a refusal."""
    rank = len(value.ttype.shape)
    if rank == 0:
        raise HawkError(
            f"x.{op}(): a rank-0 value is this lane's and has no axis to fold; "
            f"the {op} over the lanes is hawk.{op}(x)")
    if axis is None:
        return None
    if axis == "samples":
        raise HawkError(
            f"x.{op}(axis='samples') is gone: the {op} over every lane is "
            f"hawk.{op}(x), over some lane axes hawk.{op}(x, over=...)")
    if axis == "components" and rank == 1:
        return 0
    if axis == "rows" and rank == 2:
        return 0
    if axis == "columns" and rank == 2:
        return 1
    if isinstance(axis, int) and not isinstance(axis, bool) and 0 <= axis < rank:
        return axis
    own = "0 or 'components'" if rank == 1 else "0 / 'rows', 1 / 'columns'"
    raise HawkError(
        f"x.{op}(axis={axis!r}): a rank-{rank} value has the axes {own} "
        "(the lane axes are reduced by hawk.sum(x, over=...))")


def _entry(value: Value, index: tuple) -> Value:
    """The entry of a matrix at the static ``(row, column)``."""
    return Value(_ir.make("component", (value.node,), index))


def _fold(op: str, items: list) -> Value:
    out = items[0]
    for item in items[1:]:
        out = out + item if op == "sum" else (
            Value(_ir.make("max" if op == "max" else "min", (out.node, node_of(item)))))
    return out


def reduce_own(value: Value, op: str, axis: Any = None) -> Value:
    """``value.sum/max/min/mean(axis)`` over the value's own axes."""
    ax = _axis(value, axis, op)
    shape = value.ttype.shape
    base = "sum" if op == "mean" else op
    if ax is None:
        if len(shape) == 1 or base == "sum":
            out = Value(_ir.make("sum", (value.node,))) if base == "sum" else _fold(
                base, [value[k] for k in range(shape[0])])
        else:
            out = _fold(base, [_entry(value, (r, c)) for r in range(shape[0])
                                     for c in range(shape[1])])
        count = _size(shape)
    elif len(shape) == 1:
        out = reduce_own(value, base)
        count = shape[0]
    else:
        keep = 1 - ax
        parts = [_fold(base, [_entry(value, (j, i) if ax == 0 else (i, j))
                              for j in range(shape[ax])]) for i in range(shape[keep])]
        out = vec(*parts)
        count = shape[ax]
    return out / float(count) if op == "mean" else out


def _size(shape: tuple) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


class _View:
    """``v.components`` / ``m.rows`` / ``m.columns``: a component, a row or a
    column read by name; each lowers to the existing ``component`` and ``vec``."""

    __slots__ = ("value", "axis")

    def __init__(self, value: Value, axis: str) -> None:
        self.value = value
        self.axis = axis

    def __getitem__(self, k: Any) -> Value:
        shape = self.value.ttype.shape
        if self.axis == "components":
            if isinstance(k, slice):
                return vec(*[self.value[i] for i in range(*k.indices(shape[0]))])
            return self.value[k]
        if isinstance(k, slice):
            raise HawkError(
                f"a slice of .{self.axis} is not in this release; read one "
                f"{self.axis[:-1]} at a time")
        if isinstance(k, Value):
            return _own_view(self.value, self.axis, k)
        if not isinstance(k, int) or isinstance(k, bool):
            raise HawkError(
                f".{self.axis}[i] takes an integer or a traced integer, got {k!r}")
        extent = shape[0] if self.axis == "rows" else shape[1]
        if not 0 <= k < extent:
            raise HawkError(f".{self.axis}[{k}] is outside the {extent} {self.axis}")
        if self.axis == "rows":
            return vec(*[_entry(self.value, (k, c)) for c in range(shape[1])])
        return vec(*[_entry(self.value, (r, k)) for r in range(shape[0])])


def view(value: Value, axis: str) -> _View:
    """The named view ``axis`` of ``value``, refused where the value lacks it."""
    rank = len(value.ttype.shape)
    want = 1 if axis == "components" else 2
    if rank != want:
        raise HawkError(
            f".{axis} names an axis of a rank-{want} value, got {value.ttype!r}")
    return _View(value, axis)


_MASK_STORE = ("a mask is not a key: a masked store keeps the old value where the "
               "mask is false, y = m.where(mask, v, y)")


def _identity(op: str) -> Value:
    """The identity of ``op``."""
    return real({"sum": 0.0, "max": float("-inf"), "min": float("inf")}[op])


def _own_view(value: Value, axis: str, k: Any) -> Value:
    """``view(value, axis)[k]`` where ``k`` may be a traced index: a row or a
    column picked out of a matrix by a select chain over its static extent
    (a vector's component is the ordinary ``value[k]``)."""
    if axis == "components" or isinstance(k, int):
        return view(value, axis)[k]
    rows, cols = value.ttype.shape
    if axis == "rows":
        pick = [[_entry(value, (r, c)) for r in range(rows)] for c in range(cols)]
    else:
        pick = [[_entry(value, (r, c)) for c in range(cols)] for r in range(rows)]
    out = []
    for line in pick:
        item = line[-1]
        for j in range(len(line) - 2, -1, -1):
            item = select(k == j, line[j], item)
        out.append(item)
    return vec(*out)


def _contract(contract: Any, where: str, allowed: tuple = _ir.CONTRACTS) -> str:
    if contract not in allowed:
        raise HawkError(f"{where}: contract={contract!r} is not one of {allowed}")
    return contract


def _term(value: Any, where: str) -> Value:
    """``value`` as the floating term a reduction or a scatter adds (a bool
    counts 1 / 0)."""
    from .value import _SinkRef

    if isinstance(value, (Reduced, _SinkRef)):
        raise HawkError(f"{where}: the term is a value of this lane, got the output "
                        f"{getattr(value, 'name', 'of a reduction')!r}")
    if not isinstance(value, Value):
        value = Value(node_of(value))
    if value.ttype.dtype == "bool":
        value = select(value, real(1.0), real(0.0))
    if value.ttype.dtype not in ("f64", "f32"):
        raise HawkError(f"{where}: a float or bool value is reduced, got {value.ttype!r}")
    return value


def _reduced_axes(over: Any, axes: tuple, where: str) -> tuple:
    """The positions of the lane axes ``over`` names (``None``: every axis)."""
    n = len(axes)
    if over is None:
        return tuple(range(n))
    labels = [label for label, _ in axes]
    out = []
    for a in over if isinstance(over, tuple) else (over,):
        if isinstance(a, str):
            if a not in labels:
                raise HawkError(
                    f"{where}: over={a!r} names no lane axis; "
                    + (f"this kernel's Grid declares {labels}" if labels[0] is not None
                       else "the kernel declares no Grid[...], so its one axis is over=0"))
            out.append(labels.index(a))
        elif isinstance(a, int) and not isinstance(a, bool) and -n <= a < n:
            out.append(a % n)
        else:
            raise HawkError(f"{where}: over= takes lane axis labels or positions "
                            f"(0 the slowest, -1 the fastest of {n}), got {a!r}")
    if len(set(out)) != len(out):
        raise HawkError(f"{where}: over={over!r} names an axis twice")
    return tuple(sorted(out))


def _extent(axes: tuple, i: int) -> Any:
    """The extent of lane axis ``i``: its declared number, or the runtime one
    read off the lane count."""
    if axes[i][1] is not None:
        return axes[i][1]
    rest = 1
    for _label, extent in axes[1:]:
        rest *= extent
    return n_samples() / rest if rest != 1 else n_samples()


def _product(items: list) -> Any:
    out: Any = 1
    for item in items:
        out = item if isinstance(out, int) and out == 1 else out * item
    return out


def _reduce(op: str, x: Any, over: Any, where_mask: Any, contract: Any) -> Any:
    from .lane_axes import LaneAxes, _trace

    call = f"hawk.{op}(x, over={over!r})" if over is not None else f"hawk.{op}(x)"
    contract = _contract(contract, call)
    value = _term(x, call)
    axes = getattr(_trace, "axes", ((None, None),))
    gone = _reduced_axes(over, axes, call)
    kept = tuple(i for i in range(len(axes)) if i not in gone)
    if kept and contract == "ordered" and gone != tuple(range(len(axes) - len(gone), len(axes))):
        labels = [label if label is not None else i for i, (label, _) in enumerate(axes)]
        raise HawkError(
            f"kernel {getattr(_trace, 'name', None)!r}: {call} with contract=\"ordered\" sums over {[labels[i] for i in gone]}, "
            f"but an ordered sum over some lane axes is built only over the trailing axes of the Grid "
            f"({[labels[i] for i in range(len(axes) - len(gone), len(axes))]} for {len(gone)} summed axis(es); "
            f"over every axis is built too): use contract=\"auto\", or order the Grid so the summed axes come last")
    base = "sum" if op == "mean" else op
    if kept and base != "sum":
        raise HawkError(
            f"{call}: a {op} over some of the lane axes is in the next release; "
            f"hawk.{op}(x) (one value for the run) and hawk.sum/mean(x, over=...) "
            "work now")
    if where_mask is not None:
        mask = where_mask if isinstance(where_mask, Value) else Value(node_of(where_mask))
        if mask.ttype.dtype != "bool" or mask.ttype.shape:
            raise HawkError(f"{call}: where= is this lane's bool mask, got {mask.ttype!r}")
        if op == "mean":
            raise HawkError(
                f"{call}: the mean of the masked lanes needs their count, a second "
                "reduction: hawk.sum(x, where=mask) / hawk.sum(mask)")
        value = select(mask, value, _identity(base))
    if not kept:
        term = value / real(n_samples()) if op == "mean" else value
        if base == "sum" and _ir.block_shaped(node_of(term), "auto"):       # one key, every lane
            return Value(_block_sum(node_of(term), index_node(0), index_node(1), contract))
        return Value(_ir.reduce_samples(base, node_of(term), contract=contract))
    if op == "mean":
        value = value / real(_product([_extent(axes, i) for i in gone]))
    lane = LaneAxes(axes)
    key, stride = None, 1
    for i in reversed(kept):
        term = lane[i] if stride == 1 else lane[i] * stride
        key = term if key is None else term + key
        stride *= axes[i][1] or 1
    count = _product([_extent(axes, i) for i in kept])
    count = Value(index_node(count)) if isinstance(count, int) else count
    kept_axes = tuple(axes[i] for i in kept)
    if _ir.block_shaped(node_of(value), contract):
        reduced = _block_sum(node_of(value), node_of(key), node_of(count), contract, kept_axes)
    else:
        reduced = _ir.reduce_samples(
            "sum", node_of(value), key=node_of(key), count=node_of(count),
            contract=contract, kept=kept_axes,
        )  # fmt: skip
    return Reduced(reduced, kept_axes, tuple(node_of(lane[i]) for i in kept))


def sum(x: Any, over: Any = None, where: Any = None, contract: str = "auto") -> Any:  # noqa: A001
    """The sum of ``x`` over the lane axes ``over`` names (a label, a position
    or a tuple of them; ``None``: every axis, one value for the run). Masked
    lanes (``where=`` false) add nothing; ``contract`` is the sum contract
    (``"auto"``, ``"compensated"``, ``"ordered"``). Over every axis the result
    may be read back later in the body (a stage boundary); over some axes it is
    an output, one value per kept coordinate, bound ``(*payload, *kept)``. Over some axes,
    ``contract="ordered"`` adds each key's lanes in lane order in a stage of its own (no atomics: the
    same bits on every launch, for any lane count), and the summed axes must be the trailing axes of the Grid; the default
    contracts commit atomically."""
    return _reduce("sum", x, over, where, contract)


def max(x: Any, over: Any = None, where: Any = None, contract: str = "auto") -> Any:  # noqa: A001
    """The maximum of ``x`` over every lane (NaN-ignoring, exact)."""
    return _reduce("max", x, over, where, contract)


def min(x: Any, over: Any = None, where: Any = None, contract: str = "auto") -> Any:  # noqa: A001
    """The minimum of ``x`` over every lane (NaN-ignoring, exact)."""
    return _reduce("min", x, over, where, contract)


def mean(x: Any, over: Any = None, where: Any = None, contract: str = "auto") -> Any:
    """The mean of ``x`` over the lane axes ``over`` names: the sum over the
    count of reduced lanes."""
    return _reduce("mean", x, over, where, contract)


class Reduced:
    """``hawk.sum(x, over=...)`` over some lane axes: one value per kept
    coordinate. It is returned or assigned to a ``Mutable`` declared with the
    kept axes, and it reads back in the body as the value of THIS lane's key
    (``x / hawk.sum(x, over="col")``): the kernel runs in stages, the reduction
    committed by the first."""

    __slots__ = ("node", "kept", "coordinates")

    def __init__(self, node: Any, kept: tuple, coordinates: tuple = ()) -> None:
        self.node = node
        self.kept = kept
        #: the lane's coordinate along each kept axis: the indices of the accumulate that stores the sum
        self.coordinates = coordinates

    @property
    def ttype(self) -> Any:
        return _ir.type_of(self.node)

    def __bool__(self) -> bool:
        raise HawkError(
            "hawk.sum(x, over=...) read in the body is a value, not a truth: "
            "compare it, m.where(hawk.sum(x, over=...) > 0, a, b)")


def _read_back(name: str) -> Any:
    def op(self: Reduced, *args: Any) -> Any:
        return getattr(Value(self.node), name)(*args)
    op.__name__ = name
    return op


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
              "__truediv__", "__rtruediv__", "__neg__", "__pow__", "__getitem__",
              "__lt__", "__le__", "__gt__", "__ge__"):
    setattr(Reduced, _name, _read_back(_name))
del _name


class _Stored:
    """What a scatter leaves in the sink's name: the plane is written by key,
    so the name has no value of this lane to read."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def _hawk_refuse(self, *_: Any) -> Any:
        raise HawkError(f"{self.name!r} is written by key (hawk.scatter); it has no "
                        "value of this lane to read in this kernel")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _hawk_refuse
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = __getitem__ = _hawk_refuse


def scatter(sink: Any, contract: str = "auto", conflict: str = "unique") -> Any:
    """``hawk.scatter(sink)[key] += v`` adds ``v`` into ``sink`` at ``key``,
    ``= v`` stores it there. Many lanes may share a key; the adds are combined
    (atomics or a segmented sum), never a race; the order is not fixed unless
    ``contract="ordered"``: then each key's adds are made in lane order within fixed lane chunks and the
    chunks are added in order (no atomics: the same bits on every launch for one lane count and one
    target size, ULP of ``np.add.at`` otherwise). A scalar or vector payload; any number of such
    statements, and in a ``for`` of static trip; a lane that skips it (a masked or index-set lane)
    adds nothing, and a negative key adds nothing (under ``contract="ordered"`` only -1 is the skip key: any
    other key outside the target is reported by name). ``"auto"`` and ``"compensated"`` add with a
    correction kept in a companion plane ``<sink>_c`` that the runtime owns
    (the caller never passes it); ``"plain"`` is the atomic add alone. ``key``
    is an index, one coordinate per axis of a sink declared with named axes, an
    ``Index`` plane or a ``Sorted[Index]`` one.
    ``hawk.scatter(mask)[word] |= bits`` sets bits of an ``Int32``
    or ``UInt32`` plane (a bit mask: the word index is the key): the commit is
    the target's atomic OR, so any number of lanes may hit one word and the
    result is the exact ``np.bitwise_or.at``; the C target ORs serially. A bit
    mask is discrete: ``vjp`` and ``jvp`` of a kernel that writes one refuse by name.

    The same object has one method per other operation, each a statement of the kernel body. The folds combine
    every lane's value with the plane's initial value, in any order (the result is that of ``np.minimum.at``,
    ``np.maximum.at``, ``np.bitwise_and.at``, ``np.bitwise_xor.at``): ``hawk.scatter(y)[k].min(v)``, ``.max(v)``,
    ``.and_(b)`` and ``.xor(b)`` (the integer planes take ``and_`` and ``xor``, the ordered types ``min`` and
    ``max``; a NaN absorbs). The plane keeps the value the caller gave it as the fold's start. ``contract="ordered"`` on
    ``.min`` and ``.max`` of a scalar plane also records the winner of each element, the lowest lane whose value equals the
    finished element (``+0.0`` and ``-0.0`` tie; a NaN element has none), in a plane the run makes beside it and
    :meth:`hawk.runtime.BoundPlan.winner` reads. Two
    methods return the value the element held before this lane's update, which depends on the order the lanes arrive
    in: ``old = hawk.scatter(y)[k].exchange(v)`` and
    ``old = hawk.scatter(y)[k].compare_exchange(expected, v)`` (the lane that finds ``expected`` there installs
    ``v``; exactly one lane wins a cell that starts at ``expected``), and ``old = hawk.scatter(y)[k].add(v, old=True)``
    adds and returns the partial sum before this lane's term (a ticket counter). None of these has a derivative:
    ``vjp`` and ``jvp`` of a kernel that writes one refuse by name.

    ``conflict="unique"`` (the default) promises that no two lanes store at one key; ``conflict="any"`` lets them: some
    lane's value wins, never a torn one, and the kernel has no derivative.
    Written only as the target of a store in a kernel body."""
    raise HawkError(
        "hawk.scatter(y) is the target of a store: write hawk.scatter(y)[key] += v "
        "or hawk.scatter(y)[key] = v as a statement of the kernel body")


def _key_indices(ref: Any, key: Any, where: str) -> list:
    """``key`` as one index node per axis of the sink: a coordinate per declared axis of
    a named-axis sink, or the one index of a plane with one axis."""
    axes = ref.axes or (("", None),)
    if isinstance(key, tuple):
        if len(key) != len(axes):
            raise HawkError(
                f"{where}: a tuple key gives one coordinate per declared axis "
                f"{[label for label, _ in axes] if ref.axes else '(none: this sink has one axis, give one index)'}"
                f", got {len(key)}")
        return [index_node(k) for k in key]
    if isinstance(key, slice) or key is Ellipsis:
        raise HawkError(f"{where}: key is an index, an Index plane or a tuple of "
                        "coordinates")
    if len(axes) != 1:
        raise HawkError(f"{where}: {ref.name!r} has the axes {[label for label, _ in axes]}: "
                        "give one coordinate per axis, a tuple key")
    return [index_node(key)]


#: The operations that return the value the element held.
_RETURNING = ("exchange", "compare_exchange")


def scatter_store(refs: dict, name: str, key: Any, value: Any, op: str, current: Any,
                  contract: str = "auto", conflict: str = "unique", expected: Any = None,
                  old: bool = False, when: Any = True) -> Any:
    """``hawk.scatter(name, contract=...)[key] (op)= value``, as the ast pass
    reaches it: ``+=`` accumulates (an ``Update`` that adds), ``=`` stores
    (an ``Update`` that stores). A method (``op`` an operation name, not an augmented assignment) returns the pair of the
    marker the plane's name takes and the old value its element held (``None`` when the method returns none)."""
    from .value import MutableRef, _SinkRef, by_key

    if op == "append":
        return append_store(refs, name, value, when)
    method = op not in ("=", "+=", "|=")
    where = f"hawk.scatter({name})[key] {op}{'' if method else ' v'}"
    ref = refs.get(name)
    if not isinstance(ref, _SinkRef):
        raise HawkError(f"{where}: {name!r} is not an output plane (declare it "
                        "Mutable[...])")
    if contract == "or":
        raise HawkError(f"{where}: contract='or' is gone: write hawk.scatter({name})[key] |= bits")
    contract = _contract(contract, where, _ir.SCATTER_CONTRACTS)
    if isinstance(key, Value) and key.ttype.dtype == "bool":
        raise HawkError(f"{where}: {_MASK_STORE}")
    if conflict not in ("unique", "any"):
        raise HawkError(f"{where}: conflict is 'unique' (no two lanes store at one key) or 'any' (some lane's value wins)")
    if conflict != "unique" and op != "=":
        raise HawkError(f"{where}: conflict= is for a store, hawk.scatter(y, conflict='any')[key] = v; "
                        "the other operations resolve their conflicts themselves")
    ordered_fold = op in ("min", "max") and contract == "ordered"
    if op not in ("+=", "add") and contract != "auto" and not ordered_fold:
        kind = {"=": "store", "|=": "bit-or"}.get(op, f".{op}")
        raise HawkError(f"{where}: a {kind} has no sum contract; contract= is for += and .add, and 'ordered' for .min and .max")
    if ordered_fold and ref.ttype.shape:
        raise HawkError(f"{where}: contract='ordered' records the winning lane of each element of a scalar plane; "
                        f"{name!r} is declared {ref.ttype!r}")
    if op == "|=":
        contract = "or"
    if contract == "ordered" and op == "+=" and len(ref.ttype.shape) > 1:
        raise HawkError(f"{where}: contract='ordered' adds a scalar or a vector payload")
    if isinstance(ref, MutableRef) and isinstance(current, Value) and not (
            _ir.is_a(current.node, "leaf") and current.node.field_text("kind") == "prior_read"):
        raise HawkError(
            f"{name!r} is written per lane (locked) and by key "
            f"(hawk.scatter({name})[key]): one plane has one extent")
    operation = {"=": "store", "+=": "add", "|=": "or"}.get(op, op)
    returned = by_key(ref, _key_indices(ref, key, where), value, operation, contract, conflict=conflict,
                      expected=expected, returns_old=operation in _RETURNING or old)
    if ordered_fold:
        _winner_ref(refs, ref, where)
    marker = current if isinstance(current, _Stored) else _Stored(name)
    return (marker, returned) if method else marker


def _winner_ref(refs: dict, ranked: Any, where: str) -> None:
    """The plane of winner lanes of the ordered minimum or maximum into ``ranked``, made once: an unsigned integer plane of the
    ranked plane's axes that states the plane it ranks, which the run makes with every element all ones (no winner) when the caller
    leaves it out."""
    from .value import _SinkRef

    name = f"{ranked.name}_winner"
    held = refs.get(name)
    if held is not None and getattr(held, "winner_of", "") != ranked.name:
        raise HawkError(f"{where}: {name!r} is a parameter, the name of the winner plane of {ranked.name!r}; rename the parameter")
    if held is None:
        refs[name] = _SinkRef(name, _ir.TensorType((), "uint32"), ranked.axes, winner_of=ranked.name)


def append(out: Any, value: Any = None, when: Any = True) -> Any:
    """``slot = hawk.append(out, value=v, when=cond)`` adds ``v`` to the list ``out`` in the order the lanes arrive and gives
    this lane's place in it as an integer value (the ticket). ``out`` is a plane declared with ONE axis, the list
    (``Mutable[Scalar, "item":64]``, or a vector value ``Mutable[Vector[2], "item":64]`` stored whole at its place); the extent of
    that axis is its capacity. A lane whose ``when`` is false takes no place and gets ``slot == -1``.

    The run counts every lane that asked in a one-element integer plane that belongs to ``out`` and is made zeroed, like any output
    the caller leaves out: :meth:`hawk.runtime.BoundPlan.count` reads it after the launch. When more lanes ask than ``out``
    holds, the values past its capacity are dropped, the count keeps counting and nothing is raised: the caller compares the
    count with the capacity. A count made with the run starts at zero; a bound count keeps what it holds, so a second launch
    continues the list where the first stopped. The places are a permutation of ``0 .. count - 1`` in an order no launch promises. Two appends to one
    ``out`` in one kernel share its count. A vector value is stored whole. ``slot`` and the count have no derivative; the
    tangent of ``v`` is stored at the place its value takes, and the reverse derivative reads the adjoint at the place a lane took, which
    the kernel states by storing ``slot`` into an integer ``Mutable`` plane that the reverse kernel takes as an input. Written only as a statement of the kernel body."""
    raise HawkError(
        "hawk.append(out, value=v, when=cond) is a statement of the kernel body: write "
        "slot = hawk.append(out, value=v) inside the kernel")


def append_store(refs: dict, name: str, value: Any, when: Any = True) -> Any:
    """``slot = hawk.append(out, value=v, when=cond)``, as the ast pass reaches it: a ticket ``Update`` of the count plane that adds
    one for a lane that asks, and a store of ``v`` at the ticket under the ``zero`` edge rule, which drops a place outside the
    list (a ticket past the capacity, or -1). Returns the pair of the marker ``out`` takes and the slot value."""
    from .value import INT32, _context, _SinkRef, by_key

    where = f"hawk.append({name})"
    ref = refs.get(name)
    if not isinstance(ref, _SinkRef) or ref.counts:
        raise HawkError(f"{where}: {name!r} is not an output plane (declare it Mutable[...])")
    if len(ref.axes) != 1:
        raise HawkError(
            f"{where}: {name!r} is a list, a plane declared with one axis "
            f'(Mutable[Scalar, "item":64]), whose extent is the capacity; it is declared with {len(ref.axes)}')
    if not isinstance(value, Value):
        value = Value(node_of(value))
    always = when is True
    if not always and (not isinstance(when, Value) or when.ttype != _ir.TensorType((), "bool")):
        raise HawkError(f"{where}: when= is this lane's condition (a bool value) or omitted, got {when!r}")
    count_name = f"{name}_count"
    count = refs.get(count_name)
    if count is None:
        if count_name in refs:
            raise HawkError(f"{where}: {count_name!r} is a parameter, the name of the count of {name!r}; rename the parameter")
        count = refs[count_name] = _SinkRef(count_name, INT32, (("count", 1),), counts=name)
    one, zero, none = (Value(_ir.const(_context(), n, INT32)) for n in (1, 0, -1))
    ticket = by_key(count, [index_node(0)], one if always else select(when, one, zero), "add", "plain", returns_old=True,
                    several=True)
    slot = ticket if always else select(when, ticket, none)
    by_key(ref, [_ir.make("cast", (node_of(slot),), "i32")], value, "store", several=True, edge="zero")
    return _Stored(name), slot


def logic(which: str, *args: Any) -> Any:
    """``a and b`` / ``a or b`` / ``not a`` as the ast pass reaches them: with a traced operand, the logical
    ``land`` / ``lor`` / ``lnot`` of conditions (every operand is evaluated: a kernel body is pure, it has no short
    circuit); with none, Python's own result."""
    if not any(isinstance(a, Value) for a in args):
        if which == "not":
            return not args[0]
        result = args[0]
        for a in args[1:]:
            if bool(result) == (which == "or"):
                break
            result = a
        return result
    for a in args:
        if _ir.type_of(node_of(a)).dtype != "bool":
            raise HawkError(f"`{which}` of a traced value needs conditions (bool); compare it first, e.g. `x != 0.0`")
    if which == "not":
        return _op("lnot", args[0])
    out = args[0]
    for a in args[1:]:
        out = _op("land" if which == "and" else "lor", out, a)
    return out


def maxmin(which: str, *args: Any, **kwargs: Any) -> Any:
    """The builtin ``max``/``min`` as the ast pass reaches it: two or more
    traced operands are ``hawk.math.max/min`` (a tie keeps the first
    operand); anything untraced is the Python builtin."""
    import builtins
    traced = [a for a in args if isinstance(a, Value)]
    if len(args) == 1 and not kwargs:
        only = args[0]
        if isinstance(only, Value) or (isinstance(only, (list, tuple))
                                       and any(isinstance(a, Value) for a in only)):
            raise HawkError(
                f"{which}(x) of one traced value or an iterable of them is not a "
                f"HAWK form; compare values two at a time: {which}(a, b) is "
                f"hawk.math.{which}(a, b), and the {which} over the lanes is "
                f"hawk.{which}(x)")
    if not traced:
        return getattr(builtins, which)(*args, **kwargs)
    if kwargs:
        raise HawkError(f"{which}({', '.join(kwargs)}=...) is not a HAWK form")
    out = args[0]
    for a in args[1:]:
        out = Value(_ir.make(which, _operands((out, a))))
    return out
