# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where a per-lane value wider than a thread's registers lives: the placement pass.

A lane's vector is a value of ``W`` elements. In registers it costs ``W`` of them, and a read
or a write at a runtime index puts all of it in the thread's local memory, a pool the runtime
reserves for every thread of the launch whether it is used or not. The pass runs on the finished
IR, after any derivative is taken (:func:`hawk.ir.split.stages` calls it), and writes one record
per wide value, ``{form, plane, layout}``, which the emitters and the sidecar read. It changes no
type: the type is the declaration; where the value lives is a property of its uses.

Forms (``form``):

* ``thread``: the value stays where it is (registers, local memory): no record is written;
* ``resident``: the value lives in a ``W x lanes`` plane the runtime owns, plane-major so a warp's
  reads and writes coalesce. A use reads it element by element through the plane (a rolled loop or
  a runtime index), and a write at a runtime index is an in-place store. A Mutable output is resident
  in its own plane; any other value in a scratch plane named by the sidecar's ``team_planes``,
  made once per bind.

A use of a wide value is one of: a component read, a runtime component read or store, an
elementwise operation, a reduction of the whole (``sum``, ``dot``, ``norm``) or a store to a
Mutable plane. The pass places a value only when every use is one the resident form spells, and
leaves the rest as they were. A chain of elementwise operations over resident values and vectors
built from parts is written element by element into the plane of its last value: it is never live
whole in one thread.

The IR carries a placement as an ``Op`` of kind ``resident`` over ``(value, plane)``, where the
``plane`` is a leaf read of the plane's name: it is the node that binds the plane in the walk. The
emitter writes the value into the plane where the node stands and reads the plane afterwards; the
host emitter, which has no use for planes, renders the node as its value."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import TensorType
from .loop_nodes import Loop, LoopCarry, LoopIndex, LoopValue
from .nodes import Assign, At, Const, Finish, HawkError, Leaf, Node, Op, Select, Sink
from .slotwise import _resink
from .team import Plane
from .walk import _rebuild, substitute

#: The kind of the ``Op`` that places its first operand in the plane of its second.
RESIDENT_KIND = "resident"

#: The prefix of a scratch plane the pass makes.
PLANE_PREFIX = "hwd"

#: The layouts of a resident plane. A Mutable output is the caller's ``(width, lanes)`` plane: element ``k`` of
#: lane ``i`` is at ``k * lanes + i`` (``plane_major``). A scratch plane is the runtime's own and tiles the lanes by
#: warp: element ``k`` of lane ``i`` is at ``((i // 32) * width + k) * 32 + i % 32``, so a warp's reads and writes of
#: one element are one 128-byte line and an element's address is its lane's base plus ``128 k`` bytes, however
#: wide the value (a team layout, a lane's slice contiguous, is the other: :mod:`hawk.ir.team`).
PLANE_MAJOR = "plane_major"
WARP_TILED = "warp_tiled"
TILE = 32

#: The thresholds are measured data, one table per device feature class (:mod:`hawk.ir.wide_class`): the widest vector kept in
#: registers when accessed at a run-time index, by access pattern and then by the scalar's bytes (``read``: a built value read at a
#: runtime index; ``slot_read``: a loop-written value read by another loop; ``rotate``: a loop-carried value read in its own loop;
#: ``write``: a chain of stores at a runtime index outside a loop), and the widest value with only compile-time indices that
#: stays there (``wide_min``). The faster form wins, a spill-free one only on a tie; a vector only written in a loop never takes
#: local memory in the register form and resident costs 2x, so it stays in registers to ``wide_min``.
def runtime_min(pattern: str, real_bytes: int, tables=None) -> int:
    """The widest vector kept in registers when accessed at a run-time index in ``pattern``, on the current class."""
    table = (tables or _tables()).runtime_min[pattern]
    return table.get(real_bytes, table[8])


def _tables():
    from ..emit.policy import wide_class
    from . import wide_class as classes

    return classes.tables(wide_class() or classes.select())


#: The widest vector that stays in registers when it is read element by element by a rolled site, or read from two
#: places: the emit policy's fold threshold (:data:`hawk.emit.policy.ROLLED_FOLD_MIN`), the boundary between an
#: unrolled and a rolled site.
ROLLED_MIN = 16

#: The operations that act on each element alone: a wide result of these is written one element at
#: a time into its plane, and an operand of them is read one element at a time.
ELEMENTWISE = ("add", "sub", "mul", "div", "neg", "min", "max", "abs", "select")

#: The reductions of a whole vector that read it element by element when the site is rolled.
REDUCTIONS = ("sum", "dot", "norm")


# -- what a node is ----------------------------------------------------------------------------

def is_resident(node: Node) -> bool:
    """Whether ``node`` places a value in a plane (:data:`RESIDENT_KIND`)."""
    return isinstance(node, Op) and node.kind == RESIDENT_KIND


def is_scratch(node: Node) -> bool:
    """Whether ``node`` is the leaf that names a plane a placement writes (operand 1 of one): a Mutable's own plane,
    or a scratch plane the runtime made (:func:`is_tiled`)."""
    return isinstance(node, Leaf) and node.kind == "vocab_read" and (
        node.role == "mutable" or is_tiled(node))


def is_tiled(node: Node) -> bool:
    """Whether ``node`` names a scratch plane (flat, warp-tiled) rather than a Mutable's own."""
    return isinstance(node, Leaf) and node.kind == "vocab_read" and node.role == "wide_in" \
        and node.name.startswith(PLANE_PREFIX)


def plane_leaf(name: str, ttype: TensorType, scratch: bool = False) -> Leaf:
    """The leaf that binds the plane ``name`` of a ``ttype`` value in the walk: the Mutable's own plane, or (``scratch``) a
    flat plane of ``ttype``'s scalar."""
    if scratch:
        return Leaf("vocab_read", "wide_in", name, TensorType((), ttype.dtype))
    return Leaf("vocab_read", "mutable", name, ttype)


def place_node(value: Node, plane: Leaf) -> Op:
    """``value``, placed in ``plane``."""
    return Op(RESIDENT_KIND, (value, plane), value.ttype, plane.name)


def width(node: Node) -> int:
    """The elements of a rank-1 value (0 for any other)."""
    return node.ttype.shape[0] if len(node.ttype.shape) == 1 else 0


def is_prior(node: Node) -> bool:
    """Whether ``node`` is the launch-start value of a vector Mutable: a plane whose element ``k`` is read from the plane
    itself, before the element is written (a segmented unit's guard keeps the old value of a commit this way)."""
    return isinstance(node, Leaf) and node.kind == "prior_read" and node.role == "mutable" and len(node.ttype.shape) == 1


def element_kind(node: Node) -> str | None:
    """How element ``k`` of a rank-1 ``node`` is read without building the whole: ``proxy`` (a
    plane), ``input`` (a read-only vector plane of the kernel), ``parts`` (a ``vec``), ``splat``,
    ``elementwise`` (an operation on elements), else ``None``."""
    if is_resident(node) or is_scratch(node):
        return "proxy"
    if isinstance(node, Leaf):
        return "input" if node.role == "vec_in" and len(node.ttype.shape) == 1 else None
    if isinstance(node, (Op, Select)) and len(node.ttype.shape) == 1:
        if isinstance(node, Select) and node.operands[0].ttype.shape:
            return None                       # a select by a vector of conditions is not one a lane's elements share
        if node.kind == "vec":
            return "parts"
        if node.kind == "splat":
            return "splat"
        if node.kind in ELEMENTWISE and all(len(o.ttype.shape) in (0, 1) for o in node.operands):
            return "elementwise"
    return None


@dataclass(frozen=True)
class Gather:
    """A vector whose element ``k`` is the read ``plane[base + start + k * step]``: ``base`` the node of the
    runtime part of the index (``None`` when there is none), ``start`` and ``step`` integers."""

    plane: Leaf
    base: Node | None
    start: int
    step: int


def const_int(node: Node) -> int | None:
    """The integer ``node`` is made of constants alone (``add``, ``sub``, ``mul``, a cast of them), else ``None``."""
    if isinstance(node, Const):
        return int(node.literal) if node.ttype.dtype in ("i32", "i64") else None
    if isinstance(node, Op) and node.kind in ("add", "sub", "mul") and len(node.operands) == 2:
        a, b = const_int(node.operands[0]), const_int(node.operands[1])
        if a is None or b is None:
            return None
        return a + b if node.kind == "add" else a - b if node.kind == "sub" else a * b
    return None


def _split_index(node: Node) -> tuple:
    """``(base, offset)`` of an index ``base + constants`` (``base`` ``None`` when the index is all constant)."""
    whole = const_int(node)
    if whole is not None:
        return None, whole
    if isinstance(node, Op) and node.kind == "add":
        for a, b in (node.operands, node.operands[::-1]):
            off = const_int(b)
            if off is not None:
                base, more = _split_index(a)
                return base, more + off
    return node, 0


def gather(node: Node) -> Gather | None:
    """The :class:`Gather` a ``vec`` of reads of one plane at an arithmetic run of indices is (a Table read element by
    element, the matrix row of a product), else ``None``: its elements are then one expression of the element index."""
    from ..emit.rtstore import _same

    if not (isinstance(node, Op) and node.kind == "vec" and len(node.ttype.shape) == 1 and len(node.operands) > 1):
        return None
    first = node.operands[0]
    if not (isinstance(first, At) and not first.ttype.shape):
        return None
    base0, start = _split_index(first.index)
    step = None
    for k, part in enumerate(node.operands):
        if not (isinstance(part, At) and not part.ttype.shape
                and (part.plane.role, part.plane.name) == (first.plane.role, first.plane.name)):
            return None
        base, off = _split_index(part.index)
        if (base is None) != (base0 is None) or (base is not None and not _same(base, base0)):
            return None
        if k == 1:
            step = off - start
        if k and off != start + k * step:
            return None
    return Gather(first.plane, base0, start, step)


# -- the analysis ------------------------------------------------------------------------------

@dataclass
class _Use:
    """One use of a node: ``user`` reads it as operand ``k``, or (``user`` a loop) as the initial value
    (``kind == 'init'``) or the end-of-iteration result (``'next'``) of carried slot ``k``."""

    user: Node
    k: int
    kind: str = "operand"


class _Graph:
    """Every node of the kernel with its scope (0 for the body, else the id of the loop whose body
    holds it) and every use of it, the bodies of loops included."""

    def __init__(self, sinks: tuple) -> None:
        self.scope: dict = {}
        self.uses: dict = {}
        self.order: list = []
        self.loops: list = []
        seen: set = set()
        stack = [(s, False, 0) for s in reversed(sinks)]
        while stack:
            node, done, scope = stack.pop()
            if id(node) in seen:
                continue
            if not done:
                stack.append((node, True, scope))
                for child in reversed(node.operands):
                    if id(child) not in seen:
                        stack.append((child, False, scope))
                continue
            seen.add(id(node))
            self.scope[id(node)] = scope
            self.order.append(node)
            for k, child in enumerate(node.operands):
                kind = "operand"
                if isinstance(node, Loop):
                    kind = "init" if k < node.carry_count else "boundary"
                self.uses.setdefault(id(child), []).append(_Use(node, k, kind))
            if isinstance(node, Loop):
                self.loops.append(node)
                self._body(node, seen)

    def _body(self, loop: Loop, seen: set) -> None:
        own = id(loop)
        for node in loop.body_nodes():
            if id(node) in seen:
                continue
            seen.add(id(node))
            self.scope[id(node)] = own
            self.order.append(node)
            for k, child in enumerate(node.operands):
                self.uses.setdefault(id(child), []).append(_Use(node, k))
            if isinstance(node, Loop):
                self.loops.append(node)
                self._body(node, seen)
        for j, nxt in enumerate(loop.nexts):
            self.uses.setdefault(id(nxt), []).append(_Use(loop, j, "next"))
        for root in loop.body_roots()[loop.carry_count:]:
            self.uses.setdefault(id(root), []).append(_Use(loop, -1, "body"))

    def users(self, node: Node) -> list:
        return self.uses.get(id(node), [])


def _carry_chain(loop: Loop, j: int, graph: _Graph) -> bool:
    """Whether carried slot ``j`` is only ever changed by stores at an index into itself, so the
    plane can hold it in place: its end-of-iteration value is a chain of ``set_component_at`` /
    ``add_at`` over the carry (or the carry unchanged), the carry's only users are those stores and
    runtime reads of a component whose uses stay inside the stored values."""
    carry = loop.carries[j]
    node, links = loop.nexts[j], []
    while isinstance(node, Op) and node.kind in ("set_component_at", "add_at"):
        links.append(node)
        node = node.operands[0]
    if node is not carry:
        return False
    if not links:                   # a carry the loop leaves as it is: a vector read at run-time indices
        return all(u.kind == "next" or (u.kind == "operand" and u.k == 0 and isinstance(u.user, Op)
                                        and u.user.kind == "component_at") for u in graph.users(carry))
    chain = {id(link) for link in links}
    inside: set = set()
    stack = [o for link in links for o in link.operands[1:]]
    while stack:
        n = stack.pop()
        if id(n) in inside or n is carry:
            continue
        inside.add(id(n))
        stack.extend(n.operands)
    for use in graph.users(carry):
        user = use.user
        if use.kind == "next" and user is loop:
            continue
        if id(user) in chain and use.k == 0:
            continue
        if isinstance(user, Op) and user.kind == "component_at" and use.k == 0 \
                and all(id(u.user) in inside for u in graph.users(user)):
            continue
        return False
    return True


def _carry_accum(loop: Loop, j: int, graph: _Graph) -> bool:
    """Whether carried slot ``j`` is changed element by element from itself: its end-of-iteration value is an elementwise
    operation whose operands include the carry, which nothing else reads. The plane holds it in place (element ``k`` is
    read, then written, by the same statement), and every element is summed in the order the loop makes: an accumulation
    keeps its order."""
    carry, nxt = loop.carries[j], loop.nexts[j]
    if not (isinstance(nxt, (Op, Select)) and nxt.kind in ELEMENTWISE and graph.users(carry)):
        return False
    seen, stack = set(), [u for u in graph.users(carry)]
    while stack:
        use = stack.pop()
        node = use.user
        if use.kind != "operand" or not isinstance(node, (Op, Select)) or node.kind not in ELEMENTWISE \
                or len(node.ttype.shape) != 1:
            return False
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node is not nxt:
            onward = graph.users(node)
            if not onward:
                return False
            stack.extend(onward)
    return id(nxt) in seen


def _plain_loop(loop: Loop) -> bool:
    """A loop whose carries the plane can hold: no early exit, no committed sinks, no run-time trip count."""
    return loop.exit_cond is None and not loop.body_sinks and loop.count is None


class _Plan:
    """What the pass decides: ``resident`` the ids of the nodes written into a plane, ``fused`` the
    ids of the nodes written element by element into the plane of their one user, ``slots`` the
    carried slots ``(id(loop), j)`` that live in a plane."""

    def __init__(self, graph: _Graph, real_bytes: int) -> None:
        from ..emit.policy import rolled

        self.graph, self.rolled = graph, rolled
        self.tables = _tables()
        self.wide_min = self.tables.wide_min.get(real_bytes, self.tables.wide_min[8])
        self.real_bytes = real_bytes
        closed = {id(loop) for loop in graph.loops if not _plain_loop(loop) and loop.body_sinks}
        self.candidates = {id(n): n for n in graph.order
                           if isinstance(n, (Op, Select)) and element_kind(n) in ("parts", "splat", "elementwise")
                           and graph.scope[id(n)] not in closed}
        self.slots: set = set()
        self.accum: set = set()          # the slots whose next value is an elementwise function of the carry itself
        for loop in graph.loops:
            if not _plain_loop(loop):
                continue
            for j, carry in enumerate(loop.carries):
                pattern = self._slot_pattern(loop, j)
                if element_kind(loop.inits[j]) is not None \
                        and (_carry_chain(loop, j, graph) or (_carry_accum(loop, j, graph) and width(carry) > runtime_min("accum", real_bytes, self.tables))) \
                        and (width(carry) > self.wide_min or (pattern is not None and width(carry) > runtime_min(pattern, real_bytes, self.tables))):
                    self.slots.add((id(loop), j))
                    if not _carry_chain(loop, j, graph):
                        self.accum.add((id(loop), j))
        self.resident = {k for k, n in self.candidates.items() if self._wanted(n)}
        self.carried = self._carried_planes()
        self.resident |= {id(loop.nexts[j]) for (_, j), (loop, _) in self.carried.items()}
        self.resident |= {id(l.nexts[j]) for l in graph.loops for j in range(l.carry_count) if (id(l), j) in self.accum}
        self._follow()
        self.fused: set = set()
        self._settle()

    def _slot_pattern(self, loop: Loop, j: int) -> str | None:
        """How carried slot ``j`` is read at a run-time index: ``rotate`` in its own loop, ``slot_read`` from its final
        value, else ``None``."""
        graph = self.graph
        reads = lambda u: self._indexed(u) and u.user.kind == "component_at"       # noqa: E731 - a store is not a read
        if any(reads(u) for u in graph.users(loop.carries[j])):
            return "rotate"
        if any(isinstance(n, Op) and n.kind == "component_at" and not isinstance(n.operands[1], Const)
               for n in loop.body_nodes()):          # a body with a run-time read is not unrolled: its stores are not constant
            return "slot_read"
        values = [u.user for u in graph.users(loop) if isinstance(u.user, LoopValue) and u.user.slot == j]
        if any(reads(u) for v in values for u in graph.users(v)):
            return "slot_read"
        return None

    def _carried_planes(self) -> dict:
        """``{(id(loop), j): (loop, plane)}``: the slots of a step loop (a tail ``break`` over a device-side trip count)
        that carry a wide Mutable the loop only overwrites: the carry is the Mutable's own plane, each trip writes its
        next value there element by element, and the store after the loop has nothing to copy. A slot qualifies when its
        initial value is the Mutable's launch-start read, the body never reads the carry, its next value is an
        elementwise value and the Mutable is stored whole from the loop's final value."""
        graph, found = self.graph, {}
        self.sunk: dict = {}
        for loop in graph.loops:
            if loop.exit_form != "tail" or loop.body_sinks:
                continue
            for j, carry in enumerate(loop.carries):
                init, nxt = loop.inits[j], loop.nexts[j]
                if width(carry) <= self.wide_min or not (isinstance(init, Leaf) and init.kind == "prior_read"
                                                         and init.role == "mutable"):
                    continue
                if id(nxt) not in self.candidates or nxt.kind not in ELEMENTWISE:
                    continue
                reads = bool(graph.users(carry))
                if reads and not _carry_accum(loop, j, graph):         # read, but only to be added to: in place, every trip
                    continue
                values = [u.user for u in graph.users(loop) if isinstance(u.user, LoopValue) and u.user.slot == j]
                stores = [u.user for v in values for u in graph.users(v)]
                if len(stores) == 1 and isinstance(stores[0], Assign) and stores[0].name == init.name \
                        and stores[0].index is None and len(graph.users(values[0])) == 1:
                    found[id(loop), j] = (loop, init.name)
                    deps = None if reads or loop.count is not None else self._sink_deps(loop, nxt)   # a run-time count may run no trip: the plane is then left as it is
                    if deps is not None:
                        self.sunk[id(loop), j] = deps
                    if reads:
                        self.accum.add((id(loop), j))
        return found

    @staticmethod
    def _sink_deps(loop: Loop, nxt: Node) -> list | None:
        """The scalar carries the next value ``nxt`` of a step loop is computed from, when it can be computed once after the
        loop from their values at the start of the last trip (:meth:`_Plan._carried_planes`: a plane the body never reads
        needs writing only for the last trip); ``None`` when it reads the loop index, a vector carry or an inner loop."""
        body = {id(n) for n in loop.body_nodes()}
        deps, seen, stack = {}, set(), [nxt]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, LoopCarry):
                if node.ttype.shape:
                    return None
                deps[id(node)] = node
            elif isinstance(node, LoopIndex):
                return None
            elif id(node) in body:
                if isinstance(node, Loop) or node.kind in ("tape_read", "loop_count"):
                    return None
                stack.extend(node.operands)
        return list(deps.values())

    def _follow(self) -> None:
        """An elementwise value of any width that reads a placed value or slot is placed too: the plane is read element
        by element, which an elementwise value written into its own plane spells, and a narrow one is no exception."""
        graph, grew = self.graph, True
        while grew:
            grew = False
            for key, node in self.candidates.items():
                if key in self.resident or node.kind not in ELEMENTWISE:
                    continue
                if any(self._placed_operand(o) for o in node.operands if len(o.ttype.shape) == 1):
                    self.resident.add(key)
                    grew = True

    def _placed_operand(self, node: Node) -> bool:
        if isinstance(node, LoopValue):
            return (id(node.loop), node.slot) in self.slots
        return id(node) in self.resident

    # -- what wants a plane -------------------------------------------------------------------

    def _wanted(self, node: Node) -> bool:
        """A value is placed when a use reads it element by element or at a runtime index, or when
        it is a wide result written to a Mutable plane or read from two places (past the width
        a register file holds)."""
        w, uses = width(node), self.graph.users(node)
        if any(self._indexed(u) for u in uses):
            return w > min(runtime_min("write" if u.user.kind == "set_component_at" else "read", self.real_bytes, self.tables)
                           for u in uses if self._indexed(u))
        if w <= ROLLED_MIN:
            return False
        if any(self._reads_elements(u) for u in uses):
            return True
        if any(self._elsewhere(node, u) for u in uses):
            return True                      # an elementwise user in another scope reads the plane
        if node.kind in ELEMENTWISE and w > self.wide_min:
            return any(isinstance(u.user, Assign) for u in uses) or len(uses) >= 2
        return False

    def _elsewhere(self, node: Node, use: _Use) -> bool:
        user = use.user
        return (use.kind == "operand" and isinstance(user, (Op, Select)) and user.kind in ELEMENTWISE
                and self.graph.scope[id(user)] != self.graph.scope[id(node)])

    def _reads_elements(self, use: _Use) -> bool:
        user = use.user
        return (use.kind == "operand" and isinstance(user, Op) and user.kind in REDUCTIONS
                and len(user.operands[0].ttype.shape) == 1 and self.rolled(user))

    @staticmethod
    def _indexed(use: _Use) -> bool:
        return use.kind == "operand" and isinstance(use.user, Op) and use.k == 0 and (
            use.user.kind == "component_at" or (use.user.kind == "set_component_at" and not isinstance(use.user.operands[1], Const)))

    # -- what the emitter can spell -------------------------------------------------------------

    def _readable(self, node: Node, fused=None) -> bool:
        """Whether an elementwise value written into a plane can read ``node`` one element at a time (``fused``: the values
        being decided as written element by element, else those decided)."""
        fused = self.fused if fused is None else fused
        if len(node.ttype.shape) == 0 or is_prior(node):
            return True
        if is_scratch(node) or is_resident(node) or id(node) in self.resident or id(node) in fused:
            return True
        if isinstance(node, LoopValue):
            return (id(node.loop), node.slot) in self.slots
        if isinstance(node, LoopCarry):
            return any(c is node and (id(l), j) in self.accum for l in self.graph.loops for j, c in enumerate(l.carries))
        return element_kind(node) == "input"

    def _supported(self, node: Node, fused=None) -> bool:
        return all(self._readable(o, fused) for o in node.operands if len(o.ttype.shape) == 1)

    def _allowed(self, use: _Use, node: Node) -> bool:
        """Whether the use is one the emitter spells for a value that lives in a plane."""
        user = use.user
        if use.kind == "init":
            return (id(user), use.k) in self.slots
        if use.kind == "next":
            return (id(user), use.k) in self.carried or (id(user), use.k) in self.accum
        if use.kind == "body":              # a tail break's exit values are the next values, which its text does not read
            return user.exit_form == "tail" and any(
                nxt is node for (lid, j), (loop, _) in self.carried.items() if loop is user
                for nxt in (loop.nexts[j],))
        if use.kind == "boundary":
            return True                      # read through the body: those uses decide
        if use.kind != "operand":
            return False
        if isinstance(user, Assign):
            return True
        if not isinstance(user, (Op, Select)):
            return False
        if user.kind in ELEMENTWISE:
            return id(user) in self.resident or id(user) in self.fused
        if user.kind in REDUCTIONS:
            return self._reads_elements(use)
        if user.kind == "set_component_at":      # stored in place: nothing else may read the value it overwrites
            return use.k == 0 and len(self.graph.users(node)) == 1
        return user.kind in ("component_at", "component") and use.k == 0

    def _fuse(self) -> set:
        """The elementwise values (and vectors of parts) written element by element into the plane of the use that places
        them, in their own scope: that use is a placed or fused elementwise value, or the initial value of a placed slot.
        A vector of parts or a splat may serve several such uses (each reads its element again: a gather is cheap)."""
        graph, found = self.graph, set()

        def fusable(use, key):
            user = use.user
            if use.kind == "init":
                return (id(user), use.k) in self.slots
            return (use.kind == "operand" and isinstance(user, (Op, Select)) and user.kind in ELEMENTWISE
                    and graph.scope[id(user)] == graph.scope[key])

        for key, node in self.candidates.items():
            uses = graph.users(node)
            if key in self.resident or not uses or (len(uses) > 1 and element_kind(node) not in ("parts", "splat")):
                continue
            if all(fusable(u, key) for u in uses):
                found.add(key)
        changed = True
        while changed:
            changed = False
            for key in list(found):
                node = self.candidates[key]
                if not (all(u.kind == "init" or id(u.user) in self.resident or id(u.user) in found
                            for u in graph.users(node)) and self._supported(node, found)):
                    found.discard(key)
                    changed = True
        return found

    def _settle(self) -> None:
        """Drop what cannot be spelled until nothing changes."""
        graph = self.graph
        while True:
            self.fused = self._fuse()
            drop = {k for k in self.resident
                    if not self._supported(self.candidates[k])
                    or not all(self._allowed(u, self.candidates[k]) for u in graph.users(self.candidates[k]))}
            slots = {s for s in self.slots if self._slot_ok(s)}
            carried = {k: v for k, v in self.carried.items() if id(v[0].nexts[k[1]]) in self.resident - drop}
            if not drop and slots == self.slots and carried.keys() == self.carried.keys():
                break
            self.resident -= drop
            self.slots = slots
            self.carried = carried
            self.accum = {a for a in self.accum if a in slots or a in carried}

    def _slot_ok(self, slot: tuple) -> bool:
        loop = next(l for l in self.graph.loops if id(l) == slot[0])
        if not self._supported(loop.inits[slot[1]]):
            return False
        for use in self.graph.users(loop):
            user = use.user
            if isinstance(user, LoopValue) and user.slot == slot[1]:
                if not all(self._allowed(u, user) for u in self.graph.users(user)):
                    return False
        return True


# -- the plan, as planes -----------------------------------------------------------------------

def _own_plane(graph: _Graph, node: Node, read: set, owned: set) -> str | None:
    """The Mutable plane ``node`` is stored to whole, when nothing reads that plane's prior value."""
    for use in graph.users(node):
        sink = use.user
        if isinstance(sink, Assign) and use.kind == "operand" and sink.index is None \
                and sink.name not in read and sink.name not in owned and sink.ttype == node.ttype:
            return sink.name
    return None


def _assign_planes(graph: _Graph, plan: _Plan, taken: set) -> tuple:
    """``({id(node): plane}, {(id(loop), j): plane}, scratch)``: the plane of every placed value and slot, a
    Mutable's own when the value ends there, else a scratch plane made for it (``scratch`` is ``{name: width}``).
    An elementwise value takes the plane of an operand that nothing else reads (it is written in place, element
    ``k`` from element ``k``) when both stand in the body of the kernel."""
    read = {n.name for n in graph.order if isinstance(n, Leaf) and n.role == "mutable" and n.kind != "vocab_read"}
    names = {n.name for n in graph.order if isinstance(n, (Leaf, Sink))}
    of, slots, scratch, owned = {}, {}, {}, set()
    count = 0

    def fresh(w: int) -> str:
        nonlocal count
        while True:
            name = f"{PLANE_PREFIX}{count}"
            count += 1
            if name not in names and name not in taken:
                scratch[name] = w
                return name

    for loop in graph.loops:                  # a slot first: its plane is the loop value's own, else scratch
        for j in range(loop.carry_count):
            if (id(loop), j) not in plan.slots:
                continue
            value = next((u.user for u in graph.users(loop) if isinstance(u.user, LoopValue) and u.user.slot == j), None)
            own = _own_plane(graph, value, read, owned) if value is not None and (id(loop), j) not in plan.accum else None
            if own is not None:
                owned.add(own)
            slots[(id(loop), j)] = own or fresh(width(loop.carries[j]))
    sunk_next = {id(plan.carried[k][0].nexts[k[1]]) for k in plan.sunk}
    for (_, j), (loop, plane) in plan.carried.items():          # a stepping loop's Mutable carry: the Mutable's own plane
        slots[id(loop), j] = plane
        if (id(loop), j) not in plan.sunk:                       # a sunk one is written once, after the loop
            of[id(loop.nexts[j])] = plane
        owned.add(plane)
    for loop in graph.loops:                  # an accumulating slot's next value is written in the slot's own plane
        for j in range(loop.carry_count):
            if (id(loop), j) in plan.accum and (id(loop), j) in slots:
                of[id(loop.nexts[j])] = slots[id(loop), j]
    for node in graph.order:
        if id(node) not in plan.resident or id(node) in of or id(node) in sunk_next:
            continue
        own = _own_plane(graph, node, read, owned)
        if own is not None:
            owned.add(own)
            of[id(node)] = own
            continue
        reuse = None
        if graph.scope[id(node)] == 0 and node.kind in ELEMENTWISE:
            for o in node.operands:
                plane = of.get(id(o))
                if plane is None and isinstance(o, LoopValue):
                    plane = slots.get((id(o.loop), o.slot))
                if (plane in scratch and len(graph.users(o)) == 1 and width(o) == width(node)
                        and graph.scope[id(o)] == 0):
                    reuse = plane
                    break
        of[id(node)] = reuse or fresh(width(node))
    return of, slots, scratch


# -- the rewrite -------------------------------------------------------------------------------

def _rewritten(graph: _Graph, plan, of: dict, slots: dict, scratch: dict, sinks: tuple, node_fn=None, loop_fn=None) -> tuple:
    """``sinks`` with every placed value wrapped in its :func:`place_node` and every placed slot's
    initial value too. ``node_fn`` rewrites each rebuilt node and ``loop_fn(old, new)`` each rebuilt loop (:func:`simplified`)."""
    from .loops import build_loop

    memo: dict = {}

    def done(node: Node) -> Node:
        return memo.get(id(node), node)

    after: dict = {}                   # (id(loop), slot) -> the value a sunk slot is written with after its loop

    def rebuild(node: Node) -> None:
        if isinstance(node, LoopValue) and (id(node.loop), node.slot) in after:
            memo[id(node)] = after[id(node.loop), node.slot]
            return
        if isinstance(node, Loop):
            memo[id(node)] = loop_of(node) if loop_fn is None else loop_fn(node, loop_of(node), done)
            return
        ops = [done(o) for o in node.operands]
        new = node if all(a is b for a, b in zip(ops, node.operands)) else _rebuild(node, ops)
        if node_fn is not None:
            new = node_fn(new)
        if id(node) in of:
            new = place_node(new, plane_leaf(of[id(node)], node.ttype, of[id(node)] in scratch))
        memo[id(node)] = new

    def loop_of(loop: Loop) -> Loop:
        for node in loop.body_nodes():
            if id(node) not in memo:
                rebuild(node)
        inits = [done(i) for i in loop.inits]
        for j in range(loop.carry_count):
            if (id(loop), j) in slots:
                plane = plane_leaf(slots[(id(loop), j)], loop.carries[j].ttype, slots[(id(loop), j)] in scratch)
                inits[j] = plane if (id(loop), j) in plan.carried else place_node(inits[j], plane)   # a Mutable's own plane IS its start
        roots = [done(r) for r in loop.body_roots()]
        n = loop.carry_count
        if (all(a is b for a, b in zip(inits, loop.inits)) and all(a is b for a, b in zip(roots, loop.body_roots()))
                and all(a is b for a, b in zip(map(done, loop.free), loop.free))):
            return loop
        exit_cond = exit_values = None
        if loop.exit_cond is not None:
            exit_cond, exit_values = roots[n], tuple(roots[n + 1:])
        carries, nexts = list(loop.carries), roots[:n]
        sunk = [j for j in range(n) if (id(loop), j) in plan.sunk]
        if sunk:
            # a plane the body never reads is written once, from the values the last trip started with: a shadow slot
            # holds each carry's value at the start of a trip, and the break leaves the last trip's
            exit_values, nexts, last = list(exit_values), list(nexts), {j: nexts[j] for j in sunk}
            shadow: dict = {}
            for j in sunk:
                for c in plan.sunk[id(loop), j]:
                    shadow.setdefault(id(c), c)
            for j in sunk:
                nexts[j] = exit_values[j] = loop.carries[j]
            made = {}
            for c in shadow.values():
                made[id(c)] = LoopCarry(c.depth, len(carries), f"{c.name}_last", c.ttype)
                carries.append(made[id(c)])
                inits.append(inits[c.slot])
                nexts.append(c)
                exit_values.append(c)
            new = build_loop(loop.index, carries, inits, nexts, start=loop.start, stop=loop.stop, step=loop.step,
                             exit_cond=exit_cond, exit_values=tuple(exit_values), dynamic=loop.dynamic)
            projected = {id(c): LoopValue(new, made[id(c)].slot) for c in shadow.values()}
            for j in sunk:
                value = substitute([last[j]], projected)[0]
                plane = slots[id(loop), j]
                after[id(loop), j] = place_node(value, plane_leaf(plane, loop.carries[j].ttype, plane in scratch))
            return new
        return build_loop(loop.index, loop.carries, inits, roots[:n], start=loop.start, stop=loop.stop,
                          step=loop.step, exit_cond=exit_cond, exit_values=exit_values or (),
                          count=None if loop.count is None else done(loop.count),
                          count_tail=loop.count_tail, dynamic=loop.dynamic)

    stack = [(s, False) for s in reversed(sinks)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in memo:
            continue
        if not expanded:
            stack.append((node, True))
            stack.extend((o, False) for o in reversed(node.operands) if id(o) not in memo)
            continue
        if isinstance(node, Finish):         # a stepping kernel's finish reads the step loop's flag: it follows the loop
            memo[id(node)] = node if done(node.operands[0]) is node.operands[0] \
                else Finish(node.name, done(node.operands[0]), node.steps)
            continue
        if isinstance(node, Sink):
            memo[id(node)] = _resink(node, {id(c): done(c) for c in node.operands})
            continue
        rebuild(node)
    return tuple(done(s) for s in sinks)


# -- what the pass makes simpler first ---------------------------------------------------------

class _NoPlan:
    """A plan that places nothing (:func:`simplified`)."""

    carried: dict = {}
    sunk: dict = {}
    accum: set = set()
    slots: set = set()


def _scaled_gather(node: Node) -> Node:
    """``vec(s * a_0, s * a_1, ...)``, ``s`` one value and every ``a_k`` a read, as ``splat(s) * vec(a_0, a_1, ...)`` (and
    the factors the other way round): the same product element by element, with the reads a run the emitters spell as one
    loop. The reverse of a matrix-vector product is a vector of such parts."""
    from .ops import make

    if not (isinstance(node, Op) and node.kind == "vec" and len(node.ttype.shape) == 1 and len(node.operands) > 4):
        return node
    parts = node.operands
    if not all(isinstance(p, Op) and p.kind == "mul" and len(p.operands) == 2 and not p.ttype.shape for p in parts):
        return node
    for side in (0, 1):
        scale = parts[0].operands[side]
        if scale.ttype.shape or isinstance(scale, At) or not all(p.operands[side] is scale for p in parts):
            continue
        reads = [p.operands[1 - side] for p in parts]
        if all(isinstance(r, At) and not r.ttype.shape for r in reads) and gather(make("vec", tuple(reads), None)) is not None:
            rows = make("vec", tuple(reads), None)
            spread = make("splat", (scale,), node.ttype.shape)
            return make("mul", (spread, rows) if side == 0 else (rows, spread), None)
    return node


def _dead_take(old: Loop, new: Loop, graph: _Graph) -> Loop:
    """A loop that reads each component of a carried vector once, at its own index, and zeroes it behind itself, with nothing
    reading the vector after the loop: the zero is never read, so the carry stays as it is (the vector is then a value read at
    a run-time index, which the pass places like any other). The reverse of a loop that fills a vector component by component
    is this loop."""
    from .loops import build_loop

    n = new.carry_count
    drop = []
    for j in range(n):
        carry, nxt = new.carries[j], new.nexts[j]
        if any(isinstance(u.user, LoopValue) and u.user.slot == j for u in graph.users(old)):
            continue                 # the vector is read after the loop
        if not (carry.ttype.shape and isinstance(nxt, Op) and nxt.kind == "set_component_at" and nxt.operands[0] is carry
                and nxt.operands[1] is new.index and isinstance(nxt.operands[2], Const)):
            continue
        uses = [o for node in new.body_nodes() for k, o in enumerate(node.operands) if o is carry and not (node is nxt and k == 0)]
        reads = [node for node in new.body_nodes() if isinstance(node, Op) and node.kind == "component_at" and node.operands[0] is carry]
        if len(uses) == len(reads) and all(r.operands[1] is new.index for r in reads) \
                and not any(o is carry for o in new.nexts[:j] + new.nexts[j + 1:]) and new.exit_cond is None and not new.body_sinks:
            drop.append(j)
    if not drop:
        return new
    nexts = [new.carries[j] if j in drop else nxt for j, nxt in enumerate(new.nexts)]
    return build_loop(new.index, new.carries, new.inits, nexts, start=new.start, stop=new.stop, step=new.step,
                      count=new.count, count_tail=new.count_tail, dynamic=new.dynamic)


def simplified(sinks: tuple) -> tuple:
    """``sinks`` with the shapes of a derivative made plain: a vector of scaled reads is a scaled run, and a vector zeroed
    behind its own reads is left as it is."""
    graph = _Graph(sinks)
    if not (any(isinstance(n, Op) and n.kind == "vec" for n in graph.order) or graph.loops):
        return sinks
    return _rewritten(graph, _NoPlan, {}, {}, {}, sinks, node_fn=_scaled_gather,
                      loop_fn=lambda old, new, done: _dead_take(old, new, graph))


# -- the pass ----------------------------------------------------------------------------------

def place(sinks, *, real_bytes: int = 8, taken=()) -> tuple:
    """``(sinks, planes, record)``: ``sinks`` with each wide value that needs a plane placed in one;
    ``planes`` the scratch planes to make (:class:`hawk.ir.team.Plane`, one lane's width each) and ``record``
    ``{plane: {form, plane, layout, width}}`` of every placement, for the sidecar. A kernel the pass
    leaves alone comes back as it went in, with no planes."""
    from ..emit.policy import wide_form

    sinks = tuple(sinks)
    if wide_form() == "thread" or not sinks or not all(isinstance(s, Sink) and hasattr(s, "value") for s in sinks):
        return sinks, (), {}
    taken = set(taken)
    original = sinks
    sinks = simplified(sinks)
    graph = _Graph(sinks)
    plan = _Plan(graph, real_bytes)
    if not plan.resident and not plan.slots:
        return original, (), {}
    of, slots, scratch = _assign_planes(graph, plan, taken)
    try:
        new = _rewritten(graph, plan, of, slots, scratch, sinks)
    except HawkError:               # a node the rewrite has no rule for: the kernel keeps its form
        return original, (), {}
    width_of = {}
    for key, name in of.items():
        width_of[name] = width(plan.candidates[key])
    for (lid, j), name in slots.items():
        loop = next(l for l in graph.loops if id(l) == lid)
        width_of[name] = width(loop.carries[j])
    planes = tuple(Plane(name, "real", w) for name, w in sorted(scratch.items()))
    return new, planes, {name: {"form": "resident", "plane": name,
                                "layout": WARP_TILED if name in scratch else PLANE_MAJOR, "width": w}
                         for name, w in sorted(width_of.items())}


def walk_record(walk) -> dict:
    """:func:`record` of a kernel's walk."""
    return record(walk.order)


def record(order) -> dict:
    """``{plane: {form, plane, layout, width}}`` of every value placed in the walk ``order``, the bodies of its loops
    included (the sidecar's ``launch.wide``)."""
    out, nodes = {}, list(order)
    while nodes:
        node = nodes.pop()
        if isinstance(node, Loop):
            nodes.extend(node.body_nodes())
        elif is_resident(node):
            plane = node.operands[1]
            out[plane.name] = {"form": "resident", "plane": plane.name,
                               "layout": WARP_TILED if is_tiled(plane) else PLANE_MAJOR, "width": width(node)}
    return dict(sorted(out.items()))
