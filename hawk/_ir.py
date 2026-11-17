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
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import _osprey

from ._errors import HawkError

# ---- the library's constants, read once ----

_CONTEXT_OF_CONSTANTS = _osprey.Ctx()
# The constants a one-element answer stands for as a plain string (the library answers every
# constant as a tuple).
_STRING_CONSTANTS = (
    "KEYED_AUTO", "NAMESPACE_SEP", "PARAM_CELL", "RAW_KIND", "CELL_PLANE", "RAW_LANGUAGE",
    "COMPANION_SUFFIX", "ADJOINT_PREFIX", "REDUCE_KIND", "PRIMITIVE_KIND",
)  # fmt: skip
_TUPLE_CONSTANTS = (
    "COMPARE", "LOGICAL", "PREDICATE", "CONTRACTS", "SCATTER_CONTRACTS", "LEAF_ROLES",
    "INT_PLANE_DTYPES", "GATHER_ROLES", "PER_SAMPLE_ROLES", "ACCESS_CLASSES",
)  # fmt: skip

COMPARE, LOGICAL, PREDICATE, CONTRACTS, SCATTER_CONTRACTS = (
    _CONTEXT_OF_CONSTANTS.const_names(name) for name in _TUPLE_CONSTANTS[:5]
)
LEAF_ROLES, INT_PLANE_DTYPES, GATHER_ROLES, PER_SAMPLE_ROLES, ACCESS_CLASSES = (
    _CONTEXT_OF_CONSTANTS.const_names(name) for name in _TUPLE_CONSTANTS[5:]
)
(KEYED_AUTO, NAMESPACE_SEP, PARAM_CELL, RAW_KIND, CELL_PLANE, RAW_LANGUAGE, COMPANION_SUFFIX,
 ADJOINT_PREFIX, REDUCE_KIND, PRIMITIVE_KIND) = (
    _CONTEXT_OF_CONSTANTS.const_names(name)[0] for name in _STRING_CONSTANTS
)  # fmt: skip
_ROLE_PAIRS = _CONTEXT_OF_CONSTANTS.const_names("QUANTITY_ROLE_PAIRS")

#: The two loop classes: the labels :func:`classify_loop` answers with (they carry no contract
# meaning).
ACCUMULATOR = "accumulator"
RECURRENCE = "recurrence"


def _as_hawk_error(call: Callable) -> Callable:
    """Wrap ``call`` so the library's ``ValueError`` reaches the caller as ``HawkError``.

    Same text.
    """

    @functools.wraps(call)
    def refusing(*args: Any, **kwargs: Any) -> Any:
        try:
            return call(*args, **kwargs)
        except ValueError as refusal:
            raise HawkError(str(refusal)) from refusal

    return refusing


class TensorType(NamedTuple):
    """Make a value's type: its shape, its dtype and its tag, as a node's ``ttype`` reads."""

    shape: tuple
    dtype: str
    tag: Any = None


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


class Walk(NamedTuple):
    """Make a canonical walk of the library: its number and the context that holds it."""

    context: Any
    number: int


# ---- value text: how a plain value is handed to the library ----


def _text(value: Any) -> str:
    """Write a plain value (none, bool, int, float, string.

    Or a tuple of them) as the library reads it.
    """
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


# ---- contexts ----


def new_context() -> Any:
    """Make a context: the home of one kernel family's graph (a node keeps its context alive)."""
    return _osprey.Ctx()


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
) -> Any:
    """Make a leaf binding the slot ``(role, name)``.

    ``quantity`` is the compound quantity that owns the wire.
    """
    shape, dtype, tag = ttype
    owner = "None" if quantity is None else quantity.record
    fields = (
        f"(('kind', {kind!r}), ('role', {role!r}), ('name', {name!r}), ('quantity', {owner}), "
        f"('wire_index', {wire_index!r}), ('sorted', {sorted!r}), )"
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
def compensates(context: Any, contract: str) -> bool:
    """Whether an accumulate under ``contract`` commits through a companion plane."""
    return context.compensates(contract)


@_as_hawk_error
def as_matrix(node: Any) -> Any:
    """``node`` read as a matrix: a Cholesky factor projected, any other factor refused."""
    return node.ctx.as_matrix(node)


def element_count(ttype: Sequence) -> int:
    """How many scalars one value of ``ttype`` (shape, dtype, tag) holds."""
    count = 1
    for extent in ttype[0]:
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
    if sink.cls != "AccumWrite":
        return rebuild(sink, operands)
    refs = list(operands)
    index = ", ('index', #1)" if len(refs) > 1 else ""
    fields = f"(('name', {sink.field_text('name')!r}), ('value', #0){index}, "
    fields += f"('contract', {sink.field_text('contract')!r}))"
    return context.new("AccumWrite", fields, refs)


@_as_hawk_error
def recount(node: Any, count: Any) -> Any:
    """Return the keyed reduction ``node`` over ``count`` kept coordinates."""
    key = _optional_node(node, "key")
    kept = node.field_names("kept") if node.field_kind("kept") else ()
    refs = [node.operands[0], count] + ([] if key is None else [key])
    fields = (
        f"(('op', {node.field_text('op')!r}), ('value', #0), ('count', #1), "
        f"('contract', {node.field_text('contract')!r}), ('kept', {_text(kept)})"
        + ("" if key is None else ", ('key', #2)")
        + ")"
    )
    return node.ctx.new("ReduceSamples", fields, refs)


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
    """``sinks`` with the nodes in ``mapping`` replaced inside the values they commit."""
    sinks = tuple(sinks)
    values = substitute([sink.operands[0] for sink in sinks], mapping)
    return tuple(with_value(sink, value) for sink, value in zip(sinks, values))


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
    slots = _text(())[:-1] + "".join(_slot_text(slot) + ", " for slot in declared) + ")"
    number = context.canonical(sinks, slots, force_access, _text(list(exempt_planes)))
    return Walk(context, number)


@_as_hawk_error
def canonical_nodes(sinks: Sequence[Any]) -> tuple:
    """Return the walk's node order and ``{node.

    Position}`` of every node it reached (loop-body nodes have none).
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
