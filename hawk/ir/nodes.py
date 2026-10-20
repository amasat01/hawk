# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tensor-typed IR nodes: leaf / op / sink / select.

ONE typed node family, not a class hierarchy per value shape: every node
carries a :class:`~hawk.types.TensorType` ``(shape, dtype, tag)`` as FIELDS.
Nodes are identity objects — structural equality is a KEY the canonical
walk computes (:mod:`hawk.ir.walk`), so a DAG may share a subexpression
freely. Leaves and sinks BIND a slot, keyed on ``(role, name)`` and merged
totally by the walk (:attr:`Node.binding`). Access is structural:
``own(i)`` (default), ``at(expr)`` (:class:`At`), ``scatter(expr)`` (a
wide/accum sink with an index operand).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import Slot, TensorType


class HawkError(Exception):
    """The one HAWK refusal type: every structural refusal named by the spec
    (a total-leaf-merge collision, a mixed sink set, an
    impossible quantity span, an undeclared reduce op) raises it."""


#: Leaf KINDS. ``constant`` is childless but binds no slot, so it is its
#: own class (:class:`Const`), never appearing in ``Walk.leaves``/``arg_spec``.
#: ``prior_read`` is a Mutable's launch-start value, read before its first
#: store (role ``mutable``), merged with that plane's ``Assign`` SINK on
#: ``(role, name)`` by the walk's total merge.
LEAF_KINDS = ("vocab_read", "uniform", "table_read", "terminated", "nsamples",
              "prior_read")

#: Roles a LEAF (read side) may carry. ``mutable`` is carried ONLY by a
#: ``prior_read`` leaf (:meth:`hawk.trace.value.MutableRef.prior`, read
#: internally by the ast front end for a bare read before a plane's first
#: store).
LEAF_ROLES = ("vec_in", "mat_in", "wide_in", "per_sample", "lookup", "terminated",
              "uniform", "nsamples", "mutable")

#: Roles a SINK (write side) may carry.
SINK_ROLES = ("mutable", "out", "wide_out", "accum_out")

#: HAWK's OWN canonical role order for ``Walk.arg_spec``, chosen to match
#: the previous generator's so artifacts stay comparable. The ONE
#: contractual property: ``arg_spec`` order == entry-signature order.
ROLE_ORDER = ("mutable", "out", "wide_out", "accum_out", "vec_in", "mat_in",
              "wide_in", "per_sample", "lookup", "terminated", "uniform", "nsamples")

#: The quaternion op kinds; they render as aether Expression METHODS
#: (``quatMul()``, ``quatConj()``, ``quatReciprocal()``, ``quatRotate()``,
#: ``asPureQuaternion()``, ``asBack3DVector()``) in emitter.
QUAT_OPS = ("quat_mul", "quat_conj", "quat_recip", "quat_rotate", "as_pure", "as_vec3")

#: The DECLARED aether reduction functors (:data:`raptor.schema.manifest.EXEC_OPS`);
#: an undeclared reduction is a refusal, never an inferred ``sum``.
REDUCE_OPS = ("sum", "times", "max", "land")

#: The op KIND an ``@raw_device`` block splices under: opaque author text,
#: not a HAWK op, so it has no type or derivative rule (differentiating it
#: refuses, naming the kind) and its access class must be DECLARED.
RAW_KIND = "raw_device"

#: The op KIND a custom PRIMITIVE records its boundary under, deliberately
#: NOT in ``hawk.ir.ops.OP_KINDS`` like :data:`RAW_KIND`: untyped by any
#: rule of its own (typed by the inlined forward subgraph) and absent from
#: the derivative table (the author supplies that rule). Keeps the
#: closed-world table from carrying a per-primitive row.
PRIMITIVE_KIND = "primitive"

#: The node KIND of the readable LANE INDEX (``own(i)`` as a value): the
#: GLOBAL index — ``base + flat`` under a partition, never partition-local
#: — so an expression built on it is partition-invariant, like
#: ``n_samples``. Lets a 2-D launch domain recover the coordinates the
#: flattening folded together.
SAMPLE_INDEX_KIND = "sample_index"

#: The leaf ROLES an ``at(expr)`` read may target: ``lookup`` (table plane)
#: and ``wide_in`` (pinned to the SAME 32-byte ``ScalarHandle``, same access),
#: and the per-sample input planes, read at another sample through
#: ``hawk.all_samples(x)[i]``.
#: ``GATHER_ROLES`` are the ones read ONLY through ``at()``.
GATHER_ROLES = ("lookup", "wide_in")
AT_ROLES = GATHER_ROLES + ("per_sample", "vec_in", "mat_in")

#: The extents a value can have: one sample's, every sample's (a reduction
#: over samples), or one that does not depend on the sample.
EXTENTS = ("per_sample", "all_samples", "shared")

#: The compound-quantity wire namespace separator: a bound name
#: containing it must be owned by a quantity.
NAMESPACE_SEP = "__"


#: The kind of the transient node :func:`hawk.all_samples` reductions mint;
#: a store turns it into a :class:`SampleSum` sink, any other use is refused.
REDUCE_KIND = "reduce"

#: The reductions over samples a cell can hold.
CELL_OPS = ("sum", "max", "min")


def extent_of(node: Any) -> str:
    """The extent of ``node``: ``all_samples`` for a reduction over samples,
    ``shared`` when nothing under it depends on the sample (constants, uniforms,
    the sample count), else ``per_sample``."""
    if isinstance(node, ReduceSamples):
        return "all_samples"
    stack, seen = [node], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, SampleIndex) or (
                isinstance(n, Leaf) and n.role not in ("uniform", "nsamples", "lookup",
                                                       "wide_in")):
            return "per_sample"
        if isinstance(n, At) and not is_param_cell(n):
            return "per_sample"
        stack.extend(n.operands)
    return "shared"


class Node:
    """Base of every IR node: a :class:`~hawk.types.TensorType` plus ordered
    operands. Identity-hashed on purpose (the canonical walk dedups by a
    computed key)."""

    __slots__ = ("ttype", "operands")

    kind: str = "node"

    def __init__(self, ttype: TensorType, operands: Sequence[Node] = ()) -> None:
        if not isinstance(ttype, TensorType):
            raise HawkError(
                f"{type(self).__name__}: ttype must be a TensorType, got {ttype!r}"
            )
        self.ttype = ttype
        self.operands: tuple[Node, ...] = tuple(operands)
        for o in self.operands:
            if not isinstance(o, Node):
                raise HawkError(f"{type(self).__name__}: operand {o!r} is not a Node")
            if o.kind == REDUCE_KIND and any(
                    getattr(c, "kind", "").startswith("loop") for c in self.operands):
                raise HawkError(
                    "a cut inside a `for` is not in this release (it would be one "
                    "launch per iteration); hoist hawk.all_samples(x).sum() out of the "
                    "loop")
            if o.kind == REDUCE_KIND:
                raise HawkError(
                    "a value reduced over samples is not readable in this release "
                    "of the kernel: store it to a Mutable output "
                    "(total = hawk.all_samples(x).sum()) and read the cell on the host")

    @property
    def binding(self) -> Slot | None:
        """The slot this node binds, or ``None`` if it binds none."""
        return None

    @property
    def dedup_extra(self) -> Any:
        """The trailing component of the node's structural dedup key."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} {self.kind} {self.ttype}>"


class Leaf(Node):
    """A read-side leaf, identified by ``(role, name)`` — the ONLY identity a
    slot is ever keyed on. ``quantity``/``wire_index`` record the
    compound quantity that MINTED this wire; a consumer never sets them
    by hand, :meth:`hawk.ir.compound.Quantity.read` does."""

    __slots__ = ("kind", "role", "name", "quantity", "wire_index")

    def __init__(self, kind: str, role: str, name: str, ttype: TensorType, *,
                 quantity: Any = None, wire_index: int = 0) -> None:
        super().__init__(ttype)
        if kind not in LEAF_KINDS:
            raise HawkError(
                f"leaf {name!r}: unknown leaf kind {kind!r}, expected one of "
                f"{LEAF_KINDS}"
            )
        if role not in LEAF_ROLES:
            raise HawkError(
                f"leaf {name!r}: unknown leaf role {role!r}, expected one of "
                f"{LEAF_ROLES}"
            )
        self.kind = kind
        self.role = role
        self.name = name
        self.quantity = quantity
        self.wire_index = wire_index

    @property
    def binding(self) -> Slot:
        return Slot(self.role, self.name, self.ttype)


class Const(Node):
    """A literal value: childless, but binds NO slot (the ``constant``)."""

    __slots__ = ("literal",)

    kind = "constant"

    def __init__(self, literal: Any, ttype: TensorType) -> None:
        super().__init__(ttype)
        self.literal = literal

    @property
    def dedup_extra(self) -> Any:
        return repr(self.literal)


class RoleConst(Const):
    """A literal a DECLARATION named — an integer STRIDE, carrying its label.

    Same number a bare :class:`Const` would carry; the label adds PROVENANCE
    so a contraction recogniser (:mod:`hawk.ir.contraction`) can tell a
    declared stride from a coincidentally equal literal. It participates in
    the dedup key (two strides of equal value from different axes differ)
    and renders like any other integer literal."""

    __slots__ = ("label",)

    def __init__(self, value: int, label: str,
                 ttype: TensorType | None = None) -> None:
        super().__init__(int(value), ttype or TensorType((), "i32"))
        self.label = label

    @property
    def dedup_extra(self) -> Any:
        return (repr(self.literal), self.label)


class SampleIndex(Node):
    """The body's own lane index, as an IR VALUE (:data:`SAMPLE_INDEX_KIND`).

    Childless and binds NO slot: the index the backend's ONE index prologue
    already binds for every subscript in the body. Exposing it lets a
    2-D-domain kernel recover the coordinates a flattened launch folded
    together, without the author writing the ``/``/``%`` pair by hand.

    A TERMINAL for both transforms: an index is not a differentiable
    quantity, so it carries no derivative rule (like :class:`Const`)."""

    __slots__ = ()

    kind = SAMPLE_INDEX_KIND

    def __init__(self, ttype: TensorType | None = None) -> None:
        super().__init__(ttype or TensorType((), "i32"))


class Op(Node):
    """An interior operation of KIND ``kind`` over its operands."""

    __slots__ = ("kind", "literal")

    def __init__(self, kind: str, operands: Sequence[Node], ttype: TensorType,
                 literal: Any = None) -> None:
        super().__init__(ttype, operands)
        self.kind = kind
        self.literal = literal
        if kind in QUAT_OPS and ttype.shape == (4,) and ttype.tag != "quaternion":
            raise HawkError(
                f"op {kind!r} returns a 4-wide value typed {ttype!r}: a "
                "quaternion-producing op must carry tag='quaternion', because "
                "the tag participates in type identity and therefore in the walk's "
                "merge check"
            )

    @property
    def dedup_extra(self) -> Any:
        return repr(self.literal)


class At(Op):
    """The ``at(expr)``: an ABSOLUTE-index read of a ``lookup`` plane —
    the capability says must survive partitioning, and the one
    form :func:`hawk.ir.access.infer` reads as ``cross_sample_read``."""

    __slots__ = ()

    def __init__(self, plane: Leaf, index: Node, ttype: TensorType) -> None:
        if not isinstance(plane, Leaf) or plane.role not in AT_ROLES:
            raise HawkError(
                f"at(): the indexed plane must be a leaf in one of the roles "
                f"{AT_ROLES}, got {plane!r}"
            )
        super().__init__("at", (plane, index), ttype)

    @property
    def plane(self) -> Leaf:
        return self.operands[0]  # type: ignore[return-value]

    @property
    def index(self) -> Node:
        return self.operands[1]


#: The label of the index a PARAMETER CELL carries: a ``uniform`` promoted to
#: a one-cell ``lookup`` plane (:func:`hawk.diff.param_planes`) is read as
#: ``At(plane, RoleConst(0, PARAM_CELL))``. The label tells the access
#: classifier it is one broadcast value, not a cross-sample read.
PARAM_CELL = "param_cell"


def is_param_cell(node: Any) -> bool:
    """Whether ``node`` is the read of a parameter cell."""
    return (isinstance(node, At) and isinstance(node.index, RoleConst)
            and node.index.label == PARAM_CELL)


def param_cell_names(nodes: Any) -> tuple:
    """The names of the parameter cells read in ``nodes``, in first-seen order."""
    out: dict = {}
    for node in nodes:
        if is_param_cell(node):
            out.setdefault(node.plane.name, None)
    return tuple(out)


class Primitive(Op):
    """A custom primitive's BOUNDARY.

    Operand 0 is the forward subgraph, traced INLINE; the emitter renders it
    by rendering that expression, so the artifact contains no call. Operands
    1.. are the declared INPUTS — the reason the node exists, since
    differentiating THROUGH the subgraph is what the supplied ``rules``
    replaces (pushed to the inputs, never into operand 0, unless another
    part of the body shares it). A direction with no supplied rule refuses
    BY NAME rather than differentiating through silently."""

    __slots__ = ("name", "rules", "input_count")

    def __init__(self, name: str, rules: Any, value: Node,
                 inputs: Sequence[Node] = (), ttype: TensorType | None = None) -> None:
        inputs = tuple(inputs)
        super().__init__(PRIMITIVE_KIND, (value, *inputs),
                         ttype if ttype is not None else value.ttype)
        self.name = name
        self.rules = rules
        self.input_count = len(inputs)

    @property
    def forward(self) -> Node:
        """The inlined forward subgraph this boundary stands in front of."""
        return self.operands[0]

    @property
    def inputs(self) -> tuple:
        """The primitive's declared inputs, in call order."""
        return self.operands[1:]

    @property
    def dedup_extra(self) -> Any:
        # NOT the rules object: its repr carries an address, and the kernel's
        # digest has to be byte-identical across processes. The NAME is the
        # identity, and the registry (hawk.ext) refuses two primitives sharing it.
        return (PRIMITIVE_KIND, self.name)


class Select(Node):
    """``cond ? a: b`` — what the ``ast`` pass rewrites control flow into. A distinct
    family because it is the ONE op with a control shape."""

    __slots__ = ()

    kind = "select"

    def __init__(self, cond: Node, on_true: Node, on_false: Node,
                 ttype: TensorType) -> None:
        super().__init__(ttype, (cond, on_true, on_false))


#: The op KIND of HAWK's finite per-element dispatch, control-shaped like
#: :data:`Select.kind`, but never minted by the ``ast`` pass itself (the
#: author picks a POLICY, a performance call the pass has no basis for).
DISPATCH_KIND = "dispatch"

#: Dispatch's lowerings: ``predicated``/``switch`` are per-node text
#: choices the renderer makes (:mod:`hawk.emit.aether`); ``segmented`` is a
#: per-UNIT split the walk never sees. ``auto`` is the one value a
#: launching runtime's chooser may resolve (to ``switch`` or
#: ``segmented``); a declared policy is never overridden. Until resolved,
#: ``auto`` renders as ``switch`` plus a count of the branch each sample
#: picks into the group's ``<group>_mix`` plane (the chooser's input).
DISPATCH_POLICIES = ("predicated", "switch", "segmented", "auto")

#: The policies the renderer lowers as a ``switch`` statement.
SWITCH_RENDERED = ("switch", "auto")


class Dispatch(Node):
    """``branches[clamp(kind, 0)]`` — HAWK's finite per-element dispatch.

    Operand 0 is the SELECTOR (an i32 value, never compile-time); the rest
    are K >= 2 branches of ONE :class:`~hawk.types.TensorType` (tag
    included), type-checked on the constructor since no body mints this
    through :func:`hawk.ir.ops.make`. ``policy`` changes how the BACKEND
    renders the node, never what it MEANS, so it (with ``K``) is part of
    :attr:`dedup_extra`."""

    __slots__ = ("policy",)

    kind = DISPATCH_KIND

    def __init__(self, selector: Node, branches: Sequence[Node], policy: str,
                 ttype: TensorType) -> None:
        branches = tuple(branches)
        if len(branches) < 2:
            raise HawkError(
                f"dispatch: at least 2 branches are required, got "
                f"{len(branches)}"
            )
        if policy not in DISPATCH_POLICIES:
            raise HawkError(
                f"dispatch: unknown policy {policy!r}; declared are "
                f"{DISPATCH_POLICIES}"
            )
        if selector.ttype.shape != () or selector.ttype.dtype != "i32":
            raise HawkError(
                f"dispatch: kind must be a rank-0 i32 value, got "
                f"{selector.ttype!r} — a compile-time constant is not a "
                "traced VALUE and a policy is an authoring decision, not "
                "something the node infers"
            )
        for b in branches[1:]:
            if b.ttype != branches[0].ttype:
                raise HawkError(
                    f"dispatch: the branches disagree — {branches[0].ttype!r} vs "
                    f"{b.ttype!r} (every branch shares ONE TensorType)"
                )
        super().__init__(ttype, (selector, *branches))
        self.policy = policy

    @property
    def selector(self) -> Node:
        """The ``kind`` — the i32 value that PICKS a branch."""
        return self.operands[0]

    @property
    def branches(self) -> tuple:
        """The K >= 2 candidate values, in declaration order."""
        return self.operands[1:]

    @property
    def K(self) -> int:
        """The branch count — a compile-time fact of the traced call."""
        return len(self.operands) - 1

    @property
    def dedup_extra(self) -> Any:
        return (self.K, self.policy)


class Sink(Node):
    """A write-side node: the walk's ROOT set. Binds a slot on
    ``(role, name)``, exactly like a leaf, and merges on the same key."""

    __slots__ = ("kind", "role", "name")

    def __init__(self, kind: str, role: str, name: str, value: Node,
                 index: Node | None = None, ttype: TensorType | None = None) -> None:
        operands = (value,) if index is None else (value, index)
        super().__init__(ttype if ttype is not None else value.ttype, operands)
        if role not in SINK_ROLES:
            raise HawkError(
                f"sink {name!r}: unknown sink role {role!r}, expected one of "
                f"{SINK_ROLES}"
            )
        self.kind = kind
        self.role = role
        self.name = name

    @property
    def value(self) -> Node:
        return self.operands[0]

    @property
    def index(self) -> Node | None:
        """The ``scatter(expr)`` target lane, or ``None`` for the own column."""
        return self.operands[1] if len(self.operands) > 1 else None

    @property
    def binding(self) -> Slot:
        return Slot(self.role, self.name, self.ttype)

    @property
    def dedup_extra(self) -> Any:
        return (self.role, self.name)


class Assign(Sink):
    """``assign(mutable)``: the own-column write of a mutable plane."""

    __slots__ = ()

    def __init__(self, name: str, value: Node, ttype: TensorType | None = None) -> None:
        super().__init__("assign", "mutable", name, value, None, ttype)


class WideWrite(Sink):
    """``wide_write(wide_out)``; an ``index`` makes it a scatter."""

    __slots__ = ()

    def __init__(self, name: str, value: Node, index: Node | None = None,
                 ttype: TensorType | None = None) -> None:
        super().__init__("wide_write", "wide_out", name, value, index, ttype)


class AccumWrite(Sink):
    """``accum_write(accum_out)`` — the cross-sample accumulate plane;
    an ``index`` other than the own column is a scatter."""

    __slots__ = ()

    def __init__(self, name: str, value: Node, index: Node | None = None,
                 ttype: TensorType | None = None) -> None:
        super().__init__("accum_write", "accum_out", name, value, index, ttype)


class MapreducePartial(Sink):
    """``mapreduce_partial(op)``: the first-stage partials plane eagle folds
    (``eagle.exec.fold``). EXCLUSIVE — it may not coexist with any other sink,
    and ``op`` must be a DECLARED aether functor."""

    __slots__ = ("op",)

    def __init__(self, name: str, value: Node, op: str,
                 ttype: TensorType | None = None) -> None:
        if op not in REDUCE_OPS:
            raise HawkError(
                f"mapreduce_partial {name!r}: reduction op {op!r} is not a declared "
                f"aether functor; declared ops are {REDUCE_OPS}"
            )
        super().__init__("mapreduce_partial", "accum_out", name, value, None, ttype)
        self.op = op

    @property
    def dedup_extra(self) -> Any:
        return (self.role, self.name, self.op)


#: The suffix of a ``sample_sum`` plane's compensation companion.
SAMPLE_SUM_COMP = "_comp"


class ReduceSamples(Node):
    """``reduce(op, samples)``: the sum, max or min of a rank-0 floating
    value over every sample (a vector or matrix: one cell per component,
    row-major). A trace-time node only: storing it to a Mutable makes the
    :class:`SampleSum` sink(s), and nothing else may read it."""

    __slots__ = ("op",)

    kind = REDUCE_KIND

    def __init__(self, op: str, value: Node) -> None:
        if op not in CELL_OPS:
            raise HawkError(f"reduce over samples: op {op!r} is not one of {CELL_OPS}")
        if value.ttype.dtype not in ("f64", "f32") or len(value.ttype.shape) > 2:
            raise HawkError(
                f"reduce over samples: an f64/f32 scalar, vector or matrix is "
                f"reduced, got {value.ttype!r}")
        super().__init__(value.ttype, (value,))
        self.op = op

    @property
    def dedup_extra(self) -> Any:
        return self.op

    def stored(self, name: str, ttype: TensorType | None = None) -> tuple:
        """The cell sinks that store this reduction to the plane ``name``: one
        for a scalar, one per component (planes ``name_0``, ``name_1``, ...)
        for a vector or matrix."""
        value = self.operands[0]
        shape = value.ttype.shape
        if not shape:
            return (SampleSum(name, value, ttype, self.op),)
        scalar = TensorType((), value.ttype.dtype)
        index = [(r,) if len(shape) == 1 else (r, c)
                 for r in range(shape[0]) for c in range(shape[-1] if len(shape) == 2 else 1)]
        width = len(index)
        return tuple(SampleSum(f"{name}_{k}", Op("component", (value,), scalar, ix if len(shape) == 2 else ix[0]),
                               scalar, self.op, cell=name, columns=width)
                     for k, ix in enumerate(index))


class SampleSum(Sink):
    """``sample_sum(accum_out)``: the sum (or max, or min) of a rank-0
    floating ``value`` over each run of 32 consecutive samples (one warp).

    The plane ``name`` holds one result per run (``ceil(n/32)`` of them); a
    sum also writes its compensation term to ``name + "_comp"``. A run's
    result is a fixed pairwise tree over its 32 slots (a slot past ``n``
    holds the identity: 0, -inf, +inf), the same on both targets."""

    __slots__ = ("op", "cell", "columns")

    def __init__(self, name: str, value: Node, ttype: TensorType | None = None,
                 op: str = "sum", cell: str | None = None, columns: int = 1) -> None:
        super().__init__("sample_sum", "accum_out", name, value, None, ttype)
        self.cell = cell or name
        self.columns = columns
        if op not in CELL_OPS:
            raise HawkError(f"sample_sum {name!r}: op {op!r} is not one of {CELL_OPS}")
        self.op = op
        if self.ttype.shape or self.ttype.dtype not in ("f64", "f32"):
            raise HawkError(
                f"sample_sum {name!r}: sums a rank-0 f64/f32 value, got {self.ttype!r}")

    @property
    def dedup_extra(self) -> Any:
        return (self.role, self.name, self.op, self.cell, self.columns)

    @property
    def companion(self) -> Slot:
        """The compensation plane's slot (a sum only)."""
        return Slot(self.role, self.name + SAMPLE_SUM_COMP, self.ttype)

    def rebuilt(self, value: Node) -> SampleSum:
        """The same sink over ``value``."""
        return SampleSum(self.name, value, self.ttype, self.op, self.cell, self.columns)


def sample_sums(sinks: Sequence[Node]) -> tuple:
    """The :class:`SampleSum` sinks of ``sinks``, in order: column ``k`` of
    the kernel's row of terms is the ``k``-th."""
    return tuple(s for s in sinks if isinstance(s, SampleSum))


class TapeWrite(Sink):
    """``tape[row, :] = (v_0, v_1, ...)`` on the OWN column, if ``on``.

    The taping twin's store (:func:`hawk.diff.tape`): one dynamic ROW of a
    per-sample ``(W*w,)`` plane (sample-last, so the memory is ``[W, w, N]``)
    gets the flattened components of ``values`` in order, predicated on the
    rank-0 integer ``on`` (the per-launch ``tape_on`` word). Own column
    only — no other sample's memory is touched, so the access class stays
    ``sample_local`` (:attr:`index` is ``None``). Operands are
    ``(row, on, *values)``; :attr:`value` is the first stored value."""

    __slots__ = ("width",)

    def __init__(self, name: str, row: Node, on: Node, values: Sequence[Node],
                 ttype: TensorType) -> None:
        values = tuple(values)
        Node.__init__(self, ttype, (row, on, *values))
        width = sum(element_count(v.ttype) for v in values)
        if (len(ttype.shape) != 1 or not values or ttype.shape[0] % width
                or row.ttype != TensorType((), "i32")):
            raise HawkError(
                f"tape_write {name!r}: a {ttype!r} plane takes whole rows of the "
                f"{width} stored components at an i32 row")
        self.kind = "tape_write"
        self.role = "out"
        self.name = name
        self.width = width

    @property
    def value(self) -> Node:
        return self.operands[2]

    @property
    def index(self) -> Node | None:
        return None

    @property
    def row(self) -> Node:
        return self.operands[0]

    @property
    def on(self) -> Node:
        return self.operands[1]

    @property
    def values(self) -> tuple:
        return self.operands[2:]

    @property
    def dedup_extra(self) -> Any:
        return (self.role, self.name, self.width)


def element_count(ttype: TensorType) -> int:
    """How many scalars one value of ``ttype`` holds."""
    out = 1
    for extent in ttype.shape:
        out *= int(extent)
    return out


#: The reserved ``lookup`` plane a finishing kernel counts its newly finished
#: samples into: ONE ``uint32`` cell. eagle's ``eagle.FINISHED_PLANE``
#: names the same plane.
FINISHED_PLANE = "finished_count"
FINISHED_TTYPE = TensorType((), "uint32")

#: The reserved ``lookup`` plane a ``steps="auto"`` kernel reads its trip
#: count from (:func:`hawk.steps`): set on the device by the runner, never
#: by the host; eagle's ``eagle.FUSED_STEPS_PLANE`` names the same plane.
FUSED_STEPS_PLANE = "fused_steps"


class Finish(Sink):
    """``terminated = cond``: the kernel FINISHES its own sample.

    Set-only and monotone: the mask becomes ``mask | cond``, and a newly
    finished sample adds one to :data:`FINISHED_PLANE`. It binds the mask's
    ``(terminated, name)`` slot, merged by the walk with the declared leaf,
    rendered LAST. ``steps`` is the launch's step count
    (:func:`hawk.steps`): 1 for authored, ``"auto"`` when read from
    :data:`FUSED_STEPS_PLANE`. Role ``terminated`` is NOT in
    :data:`SINK_ROLES`: this is the guard seam's own mask."""

    __slots__ = ("steps",)

    def __init__(self, name: str, value: Node, steps: int = 1) -> None:
        Node.__init__(self, TensorType((), "bool"), (value,))
        if value.ttype != self.ttype:
            raise HawkError(
                f"finish {name!r}: the condition is typed {value.ttype!r}; a "
                "sample finishes on a traced rank-0 bool")
        self.kind = "finish"
        self.role = "terminated"
        self.name = name
        self.steps = steps if steps == "auto" else int(steps)

    @property
    def dedup_extra(self) -> Any:
        return (self.role, self.name, self.steps)
