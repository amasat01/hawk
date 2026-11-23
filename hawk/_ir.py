# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The one module that calls the binding's node surface; hawk reads and builds graphs through it.

A graph lives in one binding context (:func:`new_context`); a node reads its own context
(``node.ctx``), so every function that takes nodes finds the context there, and a builder that has
no operand takes the context as its first argument. Nodes are the binding's handles: equal when
they are the same node of the same context, hashable, never a class of hawk's own. Kind tests go
through the library (:func:`is_a`); the library's constants are read once, when this module loads.
What the library does not own (the walks over operands, the labels of the loop classes, the plain
records, the compound-quantity composition) is written here once. The library refuses by raising
``ValueError`` and the refusal text is the library's.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, NamedTuple, Sequence

import _osprey

from .errors import HawkError

# ---- the library's constants, read once ----

_CONTEXT_OF_CONSTANTS = _osprey.Ctx()
# The constants a one-element answer stands for as a plain string (the library answers every
# constant as a tuple).
_STRING_CONSTANTS = (
    "NAMESPACE_SEP", "PARAM_CELL", "RAW_KIND", "CELL_PLANE", "RAW_LANGUAGE",
    "COMPANION_SUFFIX", "ADJOINT_PREFIX", "REDUCE_KIND", "PRIMITIVE_KIND", "SAMPLE_SUM_COMP_SUFFIX",
)  # fmt: skip
_TUPLE_CONSTANTS = (
    "COMPARE", "LOGICAL", "PREDICATE", "CONTRACTS", "LEAF_ROLES",
    "INT_PLANE_DTYPES", "GATHER_ROLES", "PER_SAMPLE_ROLES", "ACCESS_CLASSES",
    "LINALG_FACTORISATIONS",
)  # fmt: skip

COMPARE, LOGICAL, PREDICATE, CONTRACTS = (
    _CONTEXT_OF_CONSTANTS.const_names(name) for name in _TUPLE_CONSTANTS[:4]
)
(LEAF_ROLES, INT_PLANE_DTYPES, GATHER_ROLES, PER_SAMPLE_ROLES, ACCESS_CLASSES,
 LINALG_FACTORISATIONS) = (
    _CONTEXT_OF_CONSTANTS.const_names(name) for name in _TUPLE_CONSTANTS[4:]
)  # fmt: skip
(NAMESPACE_SEP, PARAM_CELL, RAW_KIND, CELL_PLANE, RAW_LANGUAGE, COMPANION_SUFFIX,
 ADJOINT_PREFIX, REDUCE_KIND, PRIMITIVE_KIND, SAMPLE_SUM_COMP_SUFFIX) = (
    _CONTEXT_OF_CONSTANTS.const_names(name)[0] for name in _STRING_CONSTANTS
)  # fmt: skip
INT_BINARY = _CONTEXT_OF_CONSTANTS.const_names("INT_BINARY")
INT_UNARY = _CONTEXT_OF_CONSTANTS.const_names("INT_UNARY")
MINTED_OPS = _CONTEXT_OF_CONSTANTS.const_names("MINTED_OPS")
RANDOM_OPS = _CONTEXT_OF_CONSTANTS.const_names("RANDOM_OPS")
#: Names the dialects state: the op kinds that are costly to compute, and the two reserved planes the runtime fills.
SPECIAL_KINDS = _CONTEXT_OF_CONSTANTS.wing_names("SPECIAL_KINDS")
FUSED_STEPS_PLANE = _CONTEXT_OF_CONSTANTS.wing_names("FUSED_STEPS_PLANE")[0]
FINISHED_PLANE = _CONTEXT_OF_CONSTANTS.wing_names("FINISHED_PLANE")[0]
OP_KINDS = _CONTEXT_OF_CONSTANTS.const_names("OP_KINDS")
CAST = _CONTEXT_OF_CONSTANTS.const_names("CAST")
LINALG_OPS = _CONTEXT_OF_CONSTANTS.const_names("LINALG_OPS")
ROLE_ORDER = _CONTEXT_OF_CONSTANTS.const_names("ROLE_ORDER")
SINK_ROLES = _CONTEXT_OF_CONSTANTS.const_names("SINK_ROLES")
_ROLE_PAIRS = _CONTEXT_OF_CONSTANTS.const_names("QUANTITY_ROLE_PAIRS")

#: The handle class of a node (an operand check is ``isinstance(x, Node)``).
Node = _osprey.Node

#: The two loop classes: the labels :func:`classify_loop` answers with (they carry no contract
# meaning).
ACCUMULATOR = "accumulator"
RECURRENCE = "recurrence"


def _as_hawk_error(call: Callable) -> Callable:
    """Wrap ``call`` so the library's ``ValueError`` reaches the caller as ``HawkError``.

    Same text. The refusal's ``code`` (a short word the library gives to a refusal a caller may
    act on, or ``None``) is kept as the attribute ``code`` of the ``HawkError``; an error hawk
    raises itself has no ``code`` attribute, so a reader asks ``getattr(error, "code", None)``.
    """

    @functools.wraps(call)
    def refusing(*args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except ValueError as refusal:
            error = HawkError(str(refusal))
            error.code = getattr(refusal, "code", None)
            raise error from refusal

    return refusing


@dataclass(frozen=True, eq=False)
class TensorType:
    """A value's type: its shape, its dtype and its tag, as a node's ``ttype`` reads.

    Not a tuple on purpose: a builtin container subscripted with a tuple (``list[Scalar]``) would read it as
    several type arguments. Equal by value and hashable; iterating gives ``shape, dtype, tag``.
    """

    shape: tuple
    dtype: str
    tag: Any = None

    #: The field names the library reads a record's text by (:func:`_text`).
    _fields = ("shape", "dtype", "tag")

    def __iter__(self) -> Iterator:
        return iter((self.shape, self.dtype, self.tag))

    def __eq__(self, other: object) -> bool:
        """Equal by value to any type record, a subclass (a declaration) included."""
        return isinstance(other, TensorType) and tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))


class Wire(NamedTuple):
    """One wire of a compound quantity: its name, read role, shape, dtype and tag."""

    name: str
    role: str
    shape: tuple = ()
    dtype: str = "f64"
    tag: Any = None


class RawBlock(NamedTuple):
    """The literal of a raw-device op: the library writes its key text from these seven fields."""

    name: str
    access: str
    text: str
    language: str
    reads: tuple
    writes: tuple
    returns: TensorType


class KeyRun(NamedTuple):
    """The run of one keyed sink: its form (``proven``, ``sorted`` or ``tested``), its width and whether it is a block."""

    form: str
    width: int
    block: bool


class QuantityRecord(NamedTuple):
    """The slot names (``wires``) and the slot numbers (``slots``) of one compound quantity a walk reaches."""

    wires: tuple
    slots: tuple


class Slot(NamedTuple):
    """One argument slot of a kernel: its role, its name and its type."""

    role: str
    name: str
    ttype: TensorType


class LoopClass(NamedTuple):
    """Return what :func:`classify_loop` found: the class and an accumulator's per-carry addends."""

    name: str
    addends: tuple = ()


class IndexForm(NamedTuple):
    """How an integer index expression moves with an index variable (see :func:`index_form`)."""

    coefficient: Fraction
    offset: Fraction | None
    labels: tuple = ()
    anonymous: bool = False
    symbol: str = ""


class DispatchGroup(NamedTuple):
    """One dispatch group of a walk: its name, branch count, strategy word and node."""

    name: str
    branches: int
    strategy: str
    node: Any


class GroupPlanes(NamedTuple):
    """The names of the planes a dispatch group owns: its offsets, class and stale planes and its branch-mix plane."""

    offsets: str
    cls: str
    stale: str
    mix: str


class Walk(NamedTuple):
    """Make a canonical walk of the library: its number and the context that holds it."""

    context: Any
    number: int

    @property
    def companions(self) -> dict:
        """The companion plane of each compensated keyed sum the walk holds: ``{companion plane: its target plane}``."""
        return accum_companions(self)

    @property
    def gather_mats(self) -> tuple:
        """The sorted names of the matrix input planes the walk reads only through a gather (their column count is the caller's)."""
        return gather_mats(self)

    @property
    def reduces(self) -> bool:
        """Whether the kernel reduces over samples (a node with the ``reduces_samples`` feather); its access class stays ``sample_local``."""
        return any(has_feather(node, "reduces_samples") for node in walk_order(self))

    @property
    def dispatches(self) -> tuple:
        """The walk's dispatch groups in its order (:func:`dispatch_groups`): each has a ``name`` and a ``strategy``."""
        return dispatch_groups(self)


# ---- value text: how a plain value is handed to the library ----


def _text(value: Any) -> str:
    """Write a plain value (none, bool, int, float, string.

    Or a tuple of them) as the library reads it.
    """
    if hasattr(value, "_fields"):
        # a record: the library reads ``Name(field=value, ...)``
        fields = ", ".join(f"{name}={_text(item)}" for name, item in zip(value._fields, value))
        return f"{type(value).__name__}({fields})"
    if isinstance(value, (tuple, list)):
        return "(" + "".join(_text(item) + ", " for item in value) + ")"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    raise TypeError(f"a node field holds plain values only, got {type(value).__name__}")


def _fields_text(fields: Mapping[str, Any]) -> str:
    """Write the fields of a node as the library's ``((name, value), ...)`` text."""
    return "(" + "".join(f"({name!r}, {_text(value)}), " for name, value in fields.items()) + ")"


def _context_of(nodes: Sequence[Any], what: str) -> Any:
    """Return the context of the first node; refuse, naming the way out, when there is none."""
    if not nodes:
        raise TypeError(f"{what}: no operand to take the context from; pass context=")
    return nodes[0].ctx


def _optional_node(node: Any, name: str) -> Any:
    """Return the node a field names, or ``None`` when the field holds none."""
    return node.field_nodes(name)[0] if node.field_kind(name) else None


def _slot_text(slot: Slot) -> str:
    """Write a slot as the record text the library reads for a declared slot."""
    shape, dtype, tag = slot.ttype
    return (
        f"Slot(role={slot.role!r}, name={slot.name!r}, "
        f"ttype=TensorType(shape={_text(tuple(shape))}, dtype={dtype!r}, tag={tag!r}))"
    )


def _slots_text(slots: Sequence[Slot]) -> str:
    """Write declared slots as the tuple of records the library reads."""
    return "(" + "".join(_slot_text(slot) + ", " for slot in slots) + ")"


# ---- contexts ----


def new_context() -> Any:
    """Make a context: the home of one kernel family's graph (a node keeps its context alive).

    The wings' kinds (dispatch strategies, block sums, ordered writes) are declared in it.
    """
    context = _osprey.Ctx()
    context.declare_kinds()
    return context


@_as_hawk_error
def adopt(context: Any, roots: Sequence[Any]) -> tuple:
    """Return the roots as nodes of ``context``.

    Roots of another context are copied in (dump and load), the rest kept.
    """
    foreign = [root for root in roots if root.ctx is not context]
    if not foreign:
        return tuple(roots)
    # a context dumps only its own nodes, so the foreign roots are copied one context at a time
    copies: dict = {}
    for source in {root.ctx for root in foreign}:
        group = [root for root in foreign if root.ctx is source]
        copies.update(zip(group, context.load(source.dump(group))))
    return tuple(copies.get(root, root) for root in roots)


@_as_hawk_error
def declare_dialect(context: Any, manifest_json: str, first_party: Sequence[str]) -> None:
    """Declare the kinds of a dialect manifest (JSON text) in ``context``."""
    context.declare_dialect(manifest_json, _text(tuple(first_party)))


# ---- builders ----


@_as_hawk_error
def make(kind: str, operands: Sequence[Any], literal: Any = None) -> Any:
    """Make an op of ``kind`` over ``operands`` (at least one), its type inferred.

    ``literal`` is plain data.
    """
    operands = list(operands)
    text = "" if literal is None else _text(literal)
    return _context_of(operands, f"make({kind!r})").make_op(kind, operands, text)


@_as_hawk_error
def make_node(
    cls: str,
    ttype: Sequence,
    operands: Sequence[Any] = (),
    *,
    context: Any = None,
    **fields: Any,
) -> Any:
    """Make a node of the class ``cls`` (a core class or a declared kind's CamelCase name).

    ``ttype`` is ``(shape, dtype, tag)``; ``fields`` are the node's plain fields by name. The
    context is ``context``, else the first operand's.
    """
    operands = list(operands)
    shape, dtype, tag = ttype
    context = context if context is not None else _context_of(operands, f"make_node({cls!r})")
    return context.make(cls, dtype, list(shape), tag, operands, _fields_text(fields))


@_as_hawk_error
def leaf(
    context: Any,
    kind: str,
    role: str,
    name: str,
    ttype: Sequence,
    *,
    quantity: Quantity | None = None,
    wire_index: int = 0,
    sorted: bool = False,
    axes: Sequence[tuple[str, int | None]] = (),
) -> Any:
    """Make a leaf binding the slot ``(role, name)``.

    ``quantity`` is the compound quantity that owns the wire; ``axes`` is the plane's ordered axis
    list ``(label, extent)``, an extent of ``None`` being known at run time (no axes: one runtime axis).
    """
    shape, dtype, tag = ttype
    owner = "None" if quantity is None else quantity.record
    fields = (
        f"(('kind', {kind!r}), ('role', {role!r}), ('name', {name!r}), ('quantity', {owner}), "
        f"('wire_index', {wire_index!r}), ('sorted', {sorted!r}), ('axes', {tuple(axes)!r}), )"
    )
    return context.make("Leaf", dtype, list(shape), tag, [], fields)


@_as_hawk_error
def const(context: Any, value: Any, ttype: Sequence) -> Any:
    """Make a constant of plain ``value`` and type ``ttype``."""
    shape, dtype, tag = ttype
    return context.make("Const", dtype, list(shape), tag, [], _fields_text({"literal": value}))


@_as_hawk_error
def loop(
    index: Any,
    carries: Sequence[Any],
    inits: Sequence[Any],
    nexts: Sequence[Any],
    *,
    start: int,
    stop: int,
    step: int,
    body_sinks: Sequence[Any] = (),
    exit_cond: Any = None,
    exit_values: Sequence[Any] = (),
    count: Any = None,
    count_tail: bool = False,
    dynamic: str | None = None,
) -> Any:
    """Make a lowered ``for`` over ``index`` (its context is the index's).

    The library computes the boundary.
    """
    return index.ctx.new_loop(
        index, list(carries), list(inits), list(nexts), start, stop, step, list(body_sinks),
        exit_cond, list(exit_values), count, int(count_tail), dynamic,
    )  # fmt: skip


#: The fields of an ``Update`` that each contract an author names on an accumulate states: the operation, and the summation and
#: the order where the contract says one. ``auto`` states none, so the lowering is free to choose.
_CONTRACT_FIELDS = MappingProxyType(
    {
        "auto": {"op": "add"},
        "compensated": {"op": "add", "summation": "compensated"},
        "ordered": {"op": "add", "order": "lane"},
        "plain": {"op": "add", "summation": "plain"},
        "or": {"op": "or"},
    }
)
#: The contracts an author may name on a scatter, in the order the refusals list them.
SCATTER_CONTRACTS = tuple(_CONTRACT_FIELDS)


def contract_fields(contract: str) -> dict:
    """The ``Update`` fields the accumulate contract ``contract`` states."""
    return dict(_CONTRACT_FIELDS[contract])


def is_store(sink: Any) -> bool:
    """Whether ``sink`` is an update of the operation ``store``."""
    return is_a(sink, "update") and sink.field_text("op") == "store"


def is_accumulate(sink: Any) -> bool:
    """Whether ``sink`` is an update that accumulates into its plane (any operation but ``store``)."""
    return is_a(sink, "update") and sink.field_text("op") != "store"


def sets_bits(sink: Any) -> bool:
    """Whether ``sink`` is an update of the operation ``or``."""
    return is_a(sink, "update") and sink.field_text("op") == "or"


def operation_fields(sink: Any) -> dict:
    """The operation, summation and order of the update ``sink``, as fields to state on a node that does the same."""
    return {name: sink.field_text(name) for name in ("op", "summation", "order")}


@_as_hawk_error
def update(
    name: str, value: Any, *, indices: Sequence[Any] = (), op: str = "add", summation: str = "auto",
    order: str = "any", edge: str = "unchecked", axes: Sequence[tuple[str, int | None]] = (), reduced: Any = None,
    conflict: str = "unique", returns_old: bool = False, role: str | None = None,
) -> Any:
    """Make the update of ``value`` into the plane ``name`` (at ``indices``, one per axis of ``axes``; none: the lane's cell).

    A ``compare_exchange`` takes the expected value as the first of ``indices``, ahead of the plane's own indices.
    ``reduced`` is the sum over samples this update stores one entry of, which states the flat key of the lane's run.
    ``role`` is the role of the plane written (``scratch`` for a group's scratch); none: the library's default, the output plane."""
    refs = [value, *indices]
    at = f", ('indices', ({''.join(f'#{k + 1}, ' for k in range(len(indices)))}))"
    fields = f"(('name', {name!r}), ('value', #0){at}, ('op', {op!r}), ('summation', {summation!r}), "
    if role is not None:
        fields += f"('role', {role!r}), "
    fields += f"('order', {order!r}), ('conflict', {conflict!r}), ('returns_old', {returns_old!r}), "
    if reduced is not None:
        refs.append(reduced)
        fields += f"('reduced', #{len(refs) - 1}), "
    fields += f"('edge', {edge!r}), ('axes', {tuple(axes)!r}))"
    return value.ctx.new("Update", fields, refs)


@_as_hawk_error
def scratch(context: Any, name: str, ttype: Sequence, axes: Sequence[tuple[str, int | None]]) -> Any:
    """Make the scratch plane ``name`` of a group: its value type ``ttype`` over ``axes`` (an extent of ``None``: not known at build)."""
    shape, dtype, tag = ttype
    record = f"TensorType(shape={_text(tuple(shape))}, dtype={dtype!r}, tag={tag!r})"
    return context.new("Scratch", f"(('name', {name!r}), ('ttype', {record}), ('axes', {_text(tuple(axes))}))", [])


@_as_hawk_error
def group(
    context: Any, membership: str, body: Sequence[Any], scratches: Sequence[Any], *, key: Any = None, count: int | None = None,
    grid: Sequence[tuple[str, int | None]] = (), axes: Sequence[str] = (), max_size: int | None = None,
) -> Any:
    """Make the group of ``membership`` (``count``, ``axes`` or ``key``) declaring ``scratches``, over the sinks ``body`` in order.

    ``count`` is the lanes of a ``count`` group, ``grid`` the kernel's lane axes and ``axes`` the labels among them one
    group spans, ``key`` the per-lane key node and ``max_size`` the largest group of a ``key`` group. The library types the
    group and refuses what it cannot hold."""
    refs = [*body, *scratches, *([] if key is None else [key])]
    body_refs = "".join(f"#{k}, " for k in range(len(body)))
    scratch_refs = "".join(f"#{len(body) + k}, " for k in range(len(scratches)))
    key_ref = "None" if key is None else f"#{len(refs) - 1}"
    fields = (
        f"(('membership', {membership!r}), ('body_sinks', ({body_refs})), ('key', {key_ref}), ('count', {count!r}), "
        f"('grid', {_text(tuple(grid))}), ('axes', {_text(tuple(axes))}), ('max_size', {max_size!r}), "
        f"('scratch', ({scratch_refs})))"
    )
    return context.new("Group", fields, refs)


@_as_hawk_error
def group_element(context: Any, kind: str) -> Any:
    """Make the group element ``kind`` that states nothing but itself: ``GroupRank``, ``GroupSize`` or ``Barrier``."""
    return context.new(kind, "()", [])


def group_body(group: Any) -> tuple:
    """The sinks of the body of ``group`` (a ``Group`` node), in the order the library keeps them."""
    return tuple(group.field_nodes("body_sinks"))


@_as_hawk_error
def group_reduce(op: str, value: Any, order: str) -> Any:
    """Make the group-wide reduce ``op`` (``sum``, ``min``, ``max``, ``any`` or ``all``) of ``value``, added in ``order`` when it is a sum."""
    return value.ctx.new("GroupReduce", f"(('op', {op!r}), ('value', #0), ('order', {order!r}))", [value])


@_as_hawk_error
def loop_value(loop_node: Any, slot: int) -> Any:
    """Make the value carry ``slot`` holds when ``loop_node`` ends (checked by the library)."""
    return loop_node.ctx.new("LoopValue", f"(('loop', #0), ('slot', {int(slot)}))", [loop_node])


def loop_value_of(node: Any) -> tuple:
    """Return ``(loop, slot)`` of a loop value node."""
    return node.operands[0], node.field_int("literal")


@_as_hawk_error
def param_cell(context: Any, name: str, ttype: Sequence, per_cell: bool = False) -> Any:
    """Return the read of parameter ``name``.

    Cell 0 of a one-cell plane, or the cell the sample names.
    """
    shape, dtype, tag = ttype
    return context.param_cell(name, dtype, list(shape), tag, per_cell)


# ---- reads and kind tests ----


def type_of(node: Any) -> TensorType:
    """Return the type of ``node`` as a :class:`TensorType`."""
    shape, dtype, tag = node.ttype
    return TensorType(tuple(shape), dtype, tag)


@_as_hawk_error
def is_a(node: Any, *kinds: str) -> bool:
    """Whether ``node`` is of one of the kinds (the library's names, e.g. ``"leaf"``).

    Or a kind refining it.
    """
    return any(node.ctx.is_a(node, kind) for kind in kinds)


@_as_hawk_error
def has_feather(node: Any, name: str) -> bool:
    """Whether the kind of ``node`` (for an op, its operation) states the feather ``name``."""
    return node.ctx.has_feather(node, name)


@_as_hawk_error
def is_param_cell(node: Any) -> bool:
    """Whether ``node`` is the read of a parameter cell."""
    return node.ctx.is_param_cell(node)


def param_cell_names(nodes: Iterable[Any]) -> tuple:
    """Return the names of the parameter cells read in ``nodes``, in first-seen order."""
    names: dict = {}
    for node in nodes:
        if is_param_cell(node):
            names.setdefault(node.operands[0].field_text("name"), None)
    return tuple(names)


@_as_hawk_error
def compensates(sink: Any) -> bool:
    """Whether the update ``sink`` commits through a companion plane: an add under the compensated summation."""
    return sink.ctx.compensates(sink)


@_as_hawk_error
def as_matrix(node: Any) -> Any:
    """``node`` read as a matrix: a Cholesky factor projected, any other factor refused."""
    return node.ctx.as_matrix(node)


def element_count(ttype: Sequence) -> int:
    """How many scalars one value of ``ttype`` (shape, dtype, tag) holds."""
    count = 1
    for extent in ttype.shape:
        count *= int(extent)
    return count


def planes(cell: str, columns: int) -> tuple:
    """Return the row planes of a cell: ``cell``, or ``cell_0`` .. ``cell_{columns-1}``."""
    return (cell,) if columns == 1 else tuple(f"{cell}_{k}" for k in range(columns))


# ---- walks over operands ----


def post_order(roots: Iterable[Any], stop: Iterable[Any] = ()) -> list:
    """Every node under ``roots``, children before parents, each once.

    A node in ``stop`` is kept, not entered.
    """
    stop = set(stop)
    out: list = []
    seen: set = set()
    stack = [(root, False) for root in reversed(tuple(roots))]
    while stack:
        node, done = stack.pop()
        if node in seen:
            continue
        if done:
            seen.add(node)
            out.append(node)
            continue
        stack.append((node, True))
        if node not in stop:
            stack.extend((child, False) for child in reversed(node.operands) if child not in seen)
    return out


@_as_hawk_error
def rebuild(node: Any, operands: Sequence[Any]) -> Any:
    """Make a node like ``node`` over other operands."""
    return node.ctx.rebuild(node, list(operands))


def replace(node: Any, *, operands: Sequence[Any]) -> Any:
    """:func:`rebuild` with the keyword the callers of the old ``replace`` spell."""
    return rebuild(node, operands)


@_as_hawk_error
def unreduced(sink: Any, operands: Sequence[Any]) -> Any:
    """Return ``sink`` over other operands with its ``reduced`` edge dropped.

    Only a keyed accumulate has that edge; any other sink is rebuilt as :func:`rebuild` does.
    """
    context = sink.ctx
    if not is_accumulate(sink):
        return rebuild(sink, operands)
    refs = list(operands)
    indices = f", ('indices', ({''.join(f'#{k}, ' for k in range(1, len(refs)))}))"
    fields = f"(('name', {sink.field_text('name')!r}), ('value', #0){indices}, "
    fields += f"('op', {sink.field_text('op')!r}), ('summation', {sink.field_text('summation')!r}), "
    fields += f"('order', {sink.field_text('order')!r}), ('edge', {sink.field_text('edge')!r}), ('axes', {field_pairs(sink, 'axes')!r}))"
    return context.new("Update", fields, refs)


@_as_hawk_error
def addressed(sink: Any, indices: Sequence[Any], axes: Sequence[tuple]) -> Any:
    """Return the keyed accumulate ``sink`` over the ``indices`` (one per axis of ``axes``) with its ``reduced`` edge kept."""
    refs = [sink.operands[0], *indices, sink.field_nodes("reduced")[0]]
    named = "".join(f"#{k}, " for k in range(1, len(refs) - 1))
    fields = f"(('name', {sink.field_text('name')!r}), ('value', #0), ('indices', ({named})), "
    fields += f"('op', {sink.field_text('op')!r}), ('summation', {sink.field_text('summation')!r}), "
    fields += f"('order', {sink.field_text('order')!r}), ('reduced', #{len(refs) - 1}), "
    fields += f"('edge', {sink.field_text('edge')!r}), ('axes', {tuple(axes)!r}))"
    return sink.ctx.new("Update", fields, refs)


@_as_hawk_error
def reduce_samples(
    op: str,
    value: Any,
    *,
    key: Any = None,
    count: Any = None,
    contract: str = "auto",
    kept: Sequence[tuple] = (),
) -> Any:
    """Make the reduction of ``value`` over samples; ``key`` and ``count`` are edges or ``None``.

    ``kept`` holds one (label or ``None``, extent or ``None``) pair per lane axis the reduction keeps.
    """
    refs = [value] + [node for node in (key, count) if node is not None]
    edges = "".join(
        f", ('{name}', #{index})"
        for index, name in enumerate(
            [n for n, v in (("key", key), ("count", count)) if v is not None], 1
        )
    )
    head = f"(('op', {op!r}), ('value', #0), ('contract', {contract!r})"
    fields = f"{head}, ('kept', {_text(tuple(kept))}){edges})"
    return value.ctx.new("ReduceSamples", fields, refs)


@_as_hawk_error
def recount(node: Any, count: Any) -> Any:
    """Return the keyed reduction ``node`` over ``count`` kept coordinates.

    The kept axes are carried over as the node states them: (label, extent) pairs.
    """
    kept = field_pairs(node, "kept") if node.field_kind("kept") else ()
    return reduce_samples(
        node.field_text("op"),
        node.operands[0],
        key=_optional_node(node, "key"),
        count=count,
        contract=node.field_text("contract"),
        kept=kept,
    )


@_as_hawk_error
def substitute(roots: Sequence[Any], mapping: Mapping[Any, Any]) -> tuple:
    """Return the roots with every node that is a key of ``mapping`` replaced by its value.

    The originals are kept.
    """
    roots = list(roots)
    return _context_of(roots, "substitute").substitute(roots, list(mapping), list(mapping.values()))


@_as_hawk_error
def with_value(sink: Any, value: Any) -> Any:
    """``sink`` committing ``value``; refuses a sink that commits no value."""
    return sink.ctx.with_value(sink, value)


@_as_hawk_error
def swap(sinks: Sequence[Any], mapping: Mapping[Any, Any]) -> tuple:
    """``sinks`` with the nodes in ``mapping`` replaced, in the value each commits and in the key of an accumulate."""
    roots = substitute([o for sink in sinks for o in sink.operands], mapping)
    out, at = [], 0
    for sink in sinks:
        part = roots[at:at + len(sink.operands)]
        at += len(sink.operands)
        out.append(replace(sink, operands=part) if is_accumulate(sink) else with_value(sink, part[0]))
    return tuple(out)


@_as_hawk_error
def structural_keys(order: Sequence[Any]) -> dict:
    """Return ``{node: key}`` over ``order``: nodes with one key are one value.

    Keys mean nothing across calls.
    """
    order = list(order)
    if not order:
        return {}
    return dict(zip(order, order[0].ctx.structural_keys(order)))


# ---- canonical walks ----


@_as_hawk_error
def canonical(
    sinks: Sequence[Any],
    declared: Sequence[Slot] = (),
    force_access: str | None = None,
    exempt_planes: Sequence[str] = (),
) -> Walk:
    """Return the canonical walk over ``sinks``.

    ``declared`` are the slots a declaration binds whether or not the body reads them (the guard's
    ``terminated`` mask); ``force_access`` floors the access class; ``exempt_planes`` are lookups.
    A dialect's kinds need no argument: they are declared in the context (:func:`declare_dialect`).
    """
    sinks = list(sinks)
    context = _context_of(sinks, "canonical")
    number = context.canonical(
        sinks, _slots_text(declared), force_access, _text(list(exempt_planes))
    )
    return Walk(context, number)


@_as_hawk_error
def cells_walk(
    sinks: Sequence[Any], declared: Sequence[Slot] = (), force_access: str | None = None
) -> Walk:
    """Return the canonical walk of a kernel's sinks, as the kernel takes it.

    Over ``declared`` it adds the compensation slot ``<name>_comp`` of every summing sample sum
    (refused when a sink already holds that plane), the finished-count slot when a sink finishes
    and the ``<group>_mix`` slot of every dispatch group that counts its branch mix; the fused-steps
    plane is exempt from the access class. ``force_access`` floors the access class.
    """
    sinks = list(sinks)
    context = _context_of(sinks, "cells_walk")
    number = context.cells_walk(sinks, _slots_text(declared), force_access)
    return Walk(context, number)


@_as_hawk_error
def slot_type(walk: Walk, role: str, name: str) -> TensorType:
    """Return the type of the slot ``(role, name)`` the walk binds; refuse one it does not."""
    shape, dtype, tag = walk.context.walk_slot_type(walk.number, role, name)
    return TensorType(tuple(shape), dtype, tag)


@_as_hawk_error
def prior_reads(walk: Walk) -> tuple:
    """Return the sorted names of the planes whose launch-start value the walk reads."""
    return tuple(walk.context.walk_prior(walk.number))


@_as_hawk_error
def dispatch_groups(walk: Walk) -> tuple:
    """Return the walk's dispatch groups in its order as :class:`DispatchGroup` records.

    The name is the library's ``dispatch<i>``; the branch count is the operands less the selector;
    the strategy word is the library's (``switch``, ``predicated``, ``segmented``, ``auto``).
    """
    context = walk.context
    nodes = context.dispatch_nodes(walk.number)
    words = context.walk_strategies(walk.number)
    return tuple(
        DispatchGroup(f"dispatch{index}", len(node.operands) - 1, word, node)
        for index, (node, word) in enumerate(zip(nodes, words))
    )


@_as_hawk_error
def mix_slots(walk: Walk) -> dict:
    """Return ``{group name: slot name}`` of the branch-mix plane of every dispatch group that counts its mix."""
    return {
        group: slot for group, slot in walk.context.walk_mix_slots(walk.number) if slot is not None
    }


@_as_hawk_error
def selector_reads(walk: Walk, index: int) -> tuple:
    """Return the ``(role, name)`` slots the selector of dispatch ``index`` reads; refuse a dispatch the walk lacks."""
    return tuple(tuple(slot) for slot in walk.context.walk_selector_reads(walk.number, index))


@_as_hawk_error
def access_class(walk: Walk) -> str:
    """Return the access class of the walk: ``sample_local``, ``cross_sample_read`` or ``cross_sample_write``."""
    return walk.context.walk_access_class(walk.number)


def group_plane_names(group: str) -> GroupPlanes:
    """Return the names of the planes the dispatch group ``group`` (``dispatch<i>``) owns."""
    return GroupPlanes(*_CONTEXT_OF_CONSTANTS.group_plane_names(group))


@_as_hawk_error
def field_pairs(node: Any, name: str) -> tuple:
    """Return the ``(text, text)`` pairs a node's field holds (refused when it holds no list of pairs)."""
    return tuple(tuple(pair) for pair in node.field_pairs(name))


@_as_hawk_error
def static_selector(walk: Walk, index: int) -> bool:
    """Whether the selector of dispatch ``index`` is static: no sink writes a plane it reads."""
    return walk.context.walk_static_selector(walk.number, index)


@_as_hawk_error
def expand_stores(sinks: Sequence[Any]) -> tuple:
    """Return the sinks with every block-sum write replaced by the accumulates of its keyed form."""
    sinks = list(sinks)
    if not sinks:
        return ()
    return tuple(sinks[0].ctx.blocks_expand_stores(sinks))


@_as_hawk_error
def expand_sample(node: Any) -> Any:
    """Return the node that takes the place of a ``Sample``: the reads and weights of its interpolation."""
    return node.ctx.sample_expand(node)


def dispatch_strategies() -> tuple:
    """Return the strategies a dispatch can be pinned to: ``switch``, then one per wing kind."""
    return new_context().dispatch_strategies()


@_as_hawk_error
def dump_numbers(roots: Sequence[Any]) -> tuple:
    """Return the number each root has in the dump of the roots (the ``root|#p`` line), in order."""
    roots = list(roots)
    if not roots:
        return ()
    return tuple(roots[0].ctx.dump_numbers(roots))


@_as_hawk_error
def digest(walk: Walk) -> str:
    """Return the walk's digest (64 hex digits): the identity of the canonical form."""
    return walk.context.digest(walk.number)


def walk_order(walk: Walk) -> tuple:
    """Return the walk's nodes in canonical order."""
    return tuple(walk.context.walk_order(walk.number))


@_as_hawk_error
def dump_text(roots: Sequence[Any]) -> str:
    """Return the text of the graph the roots reach, in the context of the first root.

    Roots of another context are copied in first (:func:`adopt`). The text numbers nodes in the
    context's own order; the library does not answer which number a node has, nor the text of the
    dialects the graph uses.
    """
    roots = list(roots)
    context = _context_of(roots, "dump_text")
    return context.dump(list(adopt(context, roots)))


@_as_hawk_error
def arg_spec(walk: Walk) -> tuple:
    """Return the walk's argument list as ``((role, name), ...)`` in argument order."""
    flat = walk.context.walk_args(walk.number)
    return tuple(zip(flat[::2], flat[1::2]))


def slot_of(walk: Walk) -> dict:
    """Return ``{(role, name): index}`` over the walk's argument list."""
    return {slot: index for index, slot in enumerate(arg_spec(walk))}


@_as_hawk_error
def canonical_nodes(sinks: Sequence[Any]) -> tuple:
    """Return the walk's node order and ``{node: position}`` of every node it reached.

    Nodes inside loop bodies have no position and are left out.
    """
    walk = canonical(sinks)
    context = walk.context
    positions = {
        node: context.walk_position(walk.number, node)
        for node in post_order(sinks)
        if not context.in_loop_body(walk.number, node)
    }
    return context.walk_order(walk.number), positions


@_as_hawk_error
def dispatch_nodes(walk: Walk) -> tuple:
    """Return the dispatch nodes of ``walk``.

    The top-level ones in order, then those inside loop bodies.
    """
    return tuple(walk.context.dispatch_nodes(walk.number))


@_as_hawk_error
def in_loop_body(walk: Walk, node: Any) -> bool:
    """Whether ``node`` lies inside a loop body of ``walk``."""
    return walk.context.in_loop_body(walk.number, node)


@_as_hawk_error
def body_nodes(loop_node: Any) -> tuple:
    """Return the nodes of a loop's body, children first."""
    return tuple(loop_node.ctx.body_nodes(loop_node))


@_as_hawk_error
def body_roots(loop_node: Any) -> tuple:
    """Return what a loop's body evaluates, in emission order."""
    return tuple(loop_node.ctx.body_roots(loop_node))


# ---- loops ----


@_as_hawk_error
def exit_kwargs(loop_node: Any) -> dict:
    """Return the ``break`` and trip count of a loop as :func:`loop` keywords.

    For a rebuild that exits where it did.
    """
    return {
        "exit_cond": _optional_node(loop_node, "exit_cond"),
        "exit_values": loop_node.field_nodes("exit_values"),
        "count": _optional_node(loop_node, "count"),
        "count_tail": loop_node.field_flag("count_tail"),
        "dynamic": loop_node.field_text("dynamic"),
    }


@_as_hawk_error
def addends(node: Any) -> list:
    """Return the terms of an ``add`` chain (a ``sub`` at the top is one term)."""
    return list(node.ctx.add_terms(node))


@_as_hawk_error
def refuse_unsupported_nesting(loop_node: Any) -> None:
    """Refuse, by name, a nested ``for`` that neither direction can carry."""
    loop_node.ctx.refuse_nesting(loop_node)


@_as_hawk_error
def classify_loop(loop_node: Any) -> LoopClass:
    """Return the class of a loop (accumulator with its addends, or recurrence).

    Or a refusal naming why.
    """
    context = loop_node.ctx
    if context.loop_is_accumulator(loop_node):
        return LoopClass(ACCUMULATOR, tuple(context.loop_addends(loop_node)))
    return LoopClass(RECURRENCE)


@_as_hawk_error
def index_form(node: Any, var: Any = None, *, quotients: bool = True) -> IndexForm | None:
    """Classify ``node`` as affine in the index ``var`` (None: the sample index), or ``None``."""
    answer = node.ctx.index_form(node, var, quotients)
    if answer[0] == "none":
        return None
    _, coefficient, offset, anonymous, symbol, *labels = answer
    return IndexForm(
        Fraction(coefficient),
        None if offset == "-" else Fraction(offset),
        tuple(labels),
        anonymous == "1",
        "" if symbol == "-" else symbol,
    )


# ---- compound quantities ----


class Quantity:
    """A compound quantity: one authored name bound to an ordered tuple of wires, each namespaced.

    ``slug`` is the first namespace segment, ``name`` the instance segment (``None``: none),
    ``wires`` objects with ``name``, ``role``, ``shape``, ``dtype`` and ``tag``, ``pack`` how the
    values are handed to the author (``auto`` or ``tuple``). The library refuses a declaration
    it cannot expand.
    """

    __slots__ = ("slug", "name", "wires", "pack")

    @_as_hawk_error
    def __init__(
        self, slug: str, name: str | None = None, *, wires: Sequence = (), pack: str = "auto"
    ) -> None:
        """Check the declaration with the library (it reads no graph.

        So a scratch context serves).
        """
        wires = tuple(wires)
        pairs = tuple((wire.role, wire.name) for wire in wires)
        _osprey.Ctx().quantity_check(slug, pack, _text(pairs))
        self.slug, self.name, self.wires, self.pack = slug, name, wires, pack

    @property
    def prefix(self) -> str:
        """Return the namespace this quantity owns."""
        return self.slug if self.name is None else f"{self.slug}{NAMESPACE_SEP}{self.name}"

    def wire_name(self, wire: Any) -> str:
        """Return the namespaced name of one wire."""
        return f"{self.prefix}{NAMESPACE_SEP}{wire.name}"

    @property
    def record(self) -> str:
        """Return the declaration as the value text a leaf's ``quantity`` field holds."""
        wires = "".join(
            f"Wire(name={w.name!r}, role={w.role!r}, shape={_text(tuple(w.shape))}, "
            f"dtype={w.dtype!r}, tag={w.tag!r}), "
            for w in self.wires
        )
        return (
            f"Quantity(slug={self.slug!r}, name={self.name!r}, wires=({wires}), pack={self.pack!r})"
        )

    def expand(self) -> tuple:
        """Every wire's slot in declaration order (unread wires included)."""
        return tuple(
            Slot(w.role, self.wire_name(w), TensorType(tuple(w.shape), w.dtype, w.tag))
            for w in self.wires
        )

    @_as_hawk_error
    def leaves(self, context: Any) -> tuple:
        """Return the wire leaves in declaration order, each carrying this quantity."""
        kind_of_role = dict(zip(_ROLE_PAIRS[::2], _ROLE_PAIRS[1::2]))
        return tuple(
            leaf(context, kind_of_role[w.role], w.role, self.wire_name(w),
                 (tuple(w.shape), w.dtype, w.tag), quantity=self, wire_index=i)
            for i, w in enumerate(self.wires)
        )  # fmt: skip

    def read_with(self, context: Any, wrap: Callable[[Any], Any]) -> Any:
        """Return what a body reads.

        The wire leaves each passed through ``wrap``, packed as ``pack`` says.
        """
        wires = tuple(wrap(wire) for wire in self.leaves(context))
        return wires if self.pack == "tuple" or len(wires) != 1 else wires[0]

    def read(self, context: Any) -> Any:
        """Return what a body reads: the wire leaves, packed as ``pack`` says."""
        return self.read_with(context, lambda wire: wire)


# ---- derivative rules: the library runs the templates, this wraps the calls ----


def rule_kinds(table: str) -> tuple:
    """Return the kinds the rule table (``unary_d``, ``bin_vjp``, ``core``, ``linalg``.

    ``rays``) has.
    """
    return _CONTEXT_OF_CONSTANTS.rule_kinds(table)


def rule_modes(mode: str) -> tuple:
    """Return the kinds of a rule mode: ``zero`` (no derivative) or ``walker``.

    (left to the transform).
    """
    return _CONTEXT_OF_CONSTANTS.rule_modes(mode)


@_as_hawk_error
def zero_like(context: Any, ttype: Sequence) -> Any:
    """Make a structural zero of ``ttype``: a rank-0 constant, else a splat of it."""
    shape, dtype, _ = ttype
    return context.zero_of(dtype, list(shape))


@_as_hawk_error
def entrywise(kind: str, nodes: Sequence[Any]) -> Any:
    """Apply the op ``kind`` entry by entry when any node is not rank 0.

    Re-assembled as a vector.
    """
    nodes = list(nodes)
    return _context_of(nodes, "entrywise").entrywise(kind, nodes)


@_as_hawk_error
def rule_factor(table: str, kind: str, bound: Mapping[str, Any]) -> tuple:
    """Run the factor template of ``kind`` over the named nodes (``None``: no value).

    Answers one node for a ``unary_d`` factor, the live entries for ``bin_jvp`` (their sum is the
    tangent) and the flattened (child, contribution) pairs for ``bin_vjp``.
    """
    live = {name: node for name, node in bound.items() if node is not None}
    absent = " ".join(name for name, node in bound.items() if node is None)
    nodes = list(live.values())
    context = _context_of(nodes, "rule_factor")
    return tuple(context.rule_factor(table, kind, " ".join(live), nodes, absent))


def _pairs(flat: Sequence[Any]) -> tuple:
    """Group a flat (child, contribution, child, contribution, ...) answer into pairs."""
    return tuple(zip(flat[::2], flat[1::2]))


@_as_hawk_error
def rule_vjp(table: str, kind: str, node: Any, cot: Any) -> tuple:
    """Return the reverse contributions ``((child, adjoint).

    ...)`` of ``node`` under a program table.
    """
    return _pairs(node.ctx.rule_vjp(table, kind, node, cot))


@_as_hawk_error
def rule_jvp(table: str, kind: str, node: Any, tangents: Mapping[Any, Any]) -> Any:
    """Return the tangent of ``node`` under a program table, or ``None``.

    ``tangents`` maps operands to theirs.
    """
    answer = node.ctx.rule_jvp(table, kind, node, list(tangents), list(tangents.values()))
    return answer[0] if answer else None


@_as_hawk_error
def reduce_vjp(node: Any, cot: Any) -> tuple:
    """Return ``((operand, adjoint),)`` of a sum over samples.

    Any other reduction is refused by name.
    """
    return _pairs(node.ctx.reduce_vjp(node, cot))


@_as_hawk_error
def reduce_jvp(node: Any, tangents: Mapping[Any, Any]) -> Any:
    """Return the tangent of a sum over samples, or ``None`` when its operand has none."""
    answer = node.ctx.reduce_jvp(node, list(tangents), list(tangents.values()))
    return answer[0] if answer else None


@_as_hawk_error
def rule_template_check(name: str, roots: Sequence[Any], allowed: Sequence[str]) -> None:
    """Refuse, by name, a graph that is not a closed rule template of primitive ``name``.

    Its free leaves must be among ``allowed``; loop placeholders, edges, plane writes and tape reads
    are checked by the library.
    """
    roots = list(roots)
    _context_of(roots, "rule_template_check").rule_template_check(name, roots, " ".join(allowed))


@_as_hawk_error
def kind_reserved(context: Any, name: str) -> bool:
    """Whether ``name`` is taken as a kind name (primitive, core or declared)."""
    return context.kind_reserved(name)


# ---- the wings' rewrites and predicates (the library runs them) ----


@_as_hawk_error
def exit_form(loop_node: Any) -> str | None:
    """Return where a loop's break sits: ``head``, ``middle`` or ``tail``.

    ``None`` without a break.
    """
    return loop_node.field_text("exit_form")


@_as_hawk_error
def slotwise(roots: Sequence[Any]) -> tuple:
    """Return the roots rewritten slot by slot.

    (a vector op over known slots becomes per-slot ops).
    """
    roots = list(roots)
    if not roots:
        return ()
    return tuple(roots[0].ctx.slotwise(roots))


@_as_hawk_error
def slotwise_chain(node: Any) -> tuple:
    """Return ``(base, links)`` of a chain of per-slot writes over ``base`` (no links.

    Not a chain).
    """
    return node.ctx.slotwise_chain(node)


@_as_hawk_error
def is_structural_zero(node: Any) -> bool:
    """Whether ``node`` is a structural zero (a zero constant or a splat of one)."""
    return node.ctx.slotwise_is_structural_zero(node)


@_as_hawk_error
def cells_stored(reduction: Any, name: str) -> tuple:
    """Return the cell sinks that store a sum over samples to the plane ``name``.

    One sink for a scalar, one per component (planes ``name_0``, ...) for a vector or matrix.
    """
    return tuple(reduction.ctx.cells_stored(reduction, name))


@_as_hawk_error
def block_factors(payload: Any) -> tuple:
    """Return ``(u, v)`` when the payload is a sum over rows of rank-1 products, else ``()``."""
    return tuple(payload.ctx.block_factors(payload))


@_as_hawk_error
def block_factor_rows(payload: Any) -> int:
    """Return how many rows a lane holds for a payload that has factors; refuse one without."""
    return payload.ctx.block_factor_rows(payload)


def _limit(max_dim: int | None) -> list:
    """Write an optional largest extent as the library's list (empty: any length)."""
    return [] if max_dim is None else [max_dim]


@_as_hawk_error
def block_ordered_case(payload: Any, contract: str, max_dim: int | None = None) -> bool:
    """Whether the ordered contract keeps the payload (a float scalar or vector) as one block."""
    return payload.ctx.block_ordered_case(payload, contract, _limit(max_dim))


@_as_hawk_error
def block_shaped(payload: Any, contract: str, max_dim: int | None = None) -> bool:
    """Whether the keyed sum of the payload can be one block, by its structure and type alone."""
    return payload.ctx.block_shaped(payload, contract, _limit(max_dim))


@_as_hawk_error
def logs_ordered(sinks: Sequence[Any]) -> tuple:
    """Return the sinks with each ordered accumulate replaced by the stores of its log.

    Other sinks pass unchanged. The library refuses, by name: a step other than 1, a nested loop,
    and a value that is not a scalar.
    """
    sinks = list(sinks)
    if not sinks:
        return ()
    return tuple(sinks[0].ctx.logs_ordered(sinks))


# ---- the fold threshold, key runs, loop boundaries, quantities and wide placement ----


@_as_hawk_error
def fold_exceeds(count: int, threshold: int | None) -> bool:
    """Whether ``count`` exceeds the fold threshold; the library refuses, by name, a build that names none."""
    return _CONTEXT_OF_CONSTANTS.fold_exceeds(count, threshold)


@_as_hawk_error
def fold_rolled(node: Any, threshold: int | None, form: str) -> bool:
    """Whether the node is spelled as its runtime twin (``form``: ``auto``, ``rolled`` or ``static``)."""
    return node.ctx.fold_rolled(node, threshold, form)


@_as_hawk_error
def key_run(sinks: Sequence[Any], sink: Any, run_max: int, block_run: int) -> KeyRun | None:
    """Return the key run of one keyed sink of the kernel with these sinks, or ``None`` for none.

    ``run_max`` and ``block_run`` are the target's lane numbers (``lanes.run_max``, ``lanes.block_run``).
    """
    sinks = list(sinks)
    found = sinks[0].ctx.lanes_key_run_in_kernel(sinks, sink, run_max, block_run)
    return None if found is None else KeyRun(*found)


@_as_hawk_error
def loop_boundary(roots: Sequence[Any], own: Iterable[Any], gates: Iterable[Any] = ()) -> tuple:
    """Return the maximal loop-invariant values of a loop body's ``roots``, in the library's order.

    ``own`` are the nodes the body changes (its carries and index); a read under a dispatch or select
    arm that gathers stays in the body unless it is in ``gates``.
    """
    roots = list(roots)
    return tuple(roots[0].ctx.loop_boundary(roots, list(own), list(gates)))


@_as_hawk_error
def quantities(walk: Walk) -> dict:
    """Return ``{namespace: QuantityRecord}`` of the compound quantities the walk reaches, in walk order.

    The wires are the slot names of the quantity's wires and the slots their numbers among the walk's arguments.
    """
    context = walk.context
    wires = context.walk_quantity_wires(walk.number)
    slots = context.walk_quantity_slots(walk.number)
    return {
        name: QuantityRecord(tuple(names), tuple(numbers))
        for (name, names), (_, numbers) in zip(wires, slots)
    }


@_as_hawk_error
def wide_place(
    sinks: Sequence[Any],
    *,
    thread_form: bool,
    runtime_min: Sequence[int],
    wide_min: int,
    rolled_min: int | None,
    form: str,
    taken: Iterable[str] = (),
) -> tuple:
    """Return ``(sinks, changed)``: the sinks with each wide value that needs a plane placed in one.

    The target's facts are the caller's: ``runtime_min`` (read, slot_read, rotate, write, accum),
    ``wide_min``, the fold threshold ``rolled_min`` with its ``form``, and the plane names ``taken``.
    """
    sinks = list(sinks)
    placed, changed = sinks[0].ctx.wide_place(
        sinks, int(thread_form), tuple(runtime_min), wide_min, rolled_min, form, "\n".join(taken)
    )
    return tuple(placed), changed


@_as_hawk_error
def wide_record(nodes: Sequence[Any]) -> tuple:
    """Return the placements of the nodes (loop bodies included): ``(plane leaf, layout, width)`` sorted by plane name."""
    nodes = list(nodes)
    return tuple(nodes[0].ctx.wide_record(nodes)) if nodes else ()


@_as_hawk_error
def wide_is_resident(node: Any) -> bool:
    """Whether the node places a value in a plane."""
    return node.ctx.wide_is_resident(node)


@_as_hawk_error
def wide_is_tiled(node: Any) -> bool:
    """Whether the node names a scratch plane the runtime makes, not a mutable's own."""
    return node.ctx.wide_is_tiled(node)


@_as_hawk_error
def accum_companions(walk: Walk) -> dict:
    """Return ``{companion plane: target plane}`` of the compensated keyed sums of the walk, in walk order."""
    return dict(walk.context.walk_accum_companions(walk.number))


@_as_hawk_error
def gather_mats(walk: Walk) -> tuple:
    """Return the sorted names of the matrix input planes the walk reads only through a gather."""
    return tuple(walk.context.walk_gather_mats(walk.number))


@_as_hawk_error
def compensated_target(
    sink_id: str,
    compensation: str | None,
    returned_slot: str | None,
    output_names: Sequence[str],
    sinks: Sequence[Any],
) -> tuple[str, bool] | None:
    """Return ``(target name, own_column)`` of a compensated sink, or ``None`` for another sink contract.

    The target is the one output besides the companion ``compensation``: ``returned_slot`` when the
    kernel returns a typed value, else ``output_names``, else every committed sink. ``own_column`` is
    true when a sink commits into the lane's own column. Refuses when no single target resolves or
    the target is a wide output.
    """
    sinks = list(sinks)
    context = sinks[0].ctx if sinks else _osprey.Ctx()
    return context.split_compensated_target(
        sink_id, compensation or "", returned_slot, "\n".join(output_names), sinks
    )
