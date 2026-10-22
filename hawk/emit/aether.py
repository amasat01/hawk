# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Aether expression-tree rendering: turns a :class:`~hawk.ir.walk.Walk`
into the kernel body as a STRING.

One renderer builds the body; a backend wraps it with the entry
prologue, so the body never contains a host ``for`` header or a
``params[]`` unpack. A sink lowers to one assignment whose right-hand
side is one aether expression — HAWK never scalarises a chain into
per-element temporaries; it names a ``const auto`` wherever a node's
fan-out is >= 2, leaves included, because an aether ``View`` subscript
is an address computation, not free.

Rank is the dispatch axis: rank-0 spells through ``aether::math``,
rank>=1 through ``Expression``'s methods and ``aether/expr`` operators,
quaternions through aether's method spellings. An op with no aether
spelling at its rank raises :class:`~hawk.ir.HawkError` naming the kind,
the rank and what aether would need — never a silent approximation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..ir import HawkError, Node, Walk
from ..ir.cost import fp64_ops
from ..ir.loop_nodes import Loop, LoopCarry, LoopCount, LoopIndex, LoopValue, TapeRead
from ..ir.nodes import (
    GATHER_ROLES,
    FINISHED_PLANE,
    RAW_KIND,
    SWITCH_RENDERED,
    AccumWrite,
    Assign,
    At,
    Const,
    Dispatch,
    Finish,
    Leaf,
    Op,
    Primitive,
    SampleIndex,
    SampleSum,
    Select,
    Sink,
    TapeWrite,
    WideWrite,
    is_param_cell,
    sample_sums,
)
from ..ir.ops import RANDOM_OPS
from ..ir.walk import canonical_nodes
from ..types import INT_PLANE_DTYPES, TensorType
from .dispatch import _DispatchMixin
from .dispatch import render_dispatch_rank0 as render_dispatch_rank0
from .spelling import (
    _CWISE,
    _CWISE1,
    _INFIX,
    _MATH1,
    _MATH2,
    _MATH3,
    _METHOD0,
    _METHOD1,
    _RANDOM_FN,
    _for_header,
    _ident,
    _no_spelling,
    _ordinal,
    _real_literal,
    _through,
    param_cell_binding,
)
from .spelling import (
    GEOMETRY_TOKENS as GEOMETRY_TOKENS,
)
from .spelling import (
    IDENT_PREFIX as IDENT_PREFIX,
)
from .spelling import (
    MIRROR_OF as MIRROR_OF,
)
from .spelling import (
    SAMPLE_INDEX_IDENT as SAMPLE_INDEX_IDENT,
)
from .spelling import (
    binding_name as binding_name,
)
from .spelling import (
    carry_type as carry_type,
)
from .spelling import (
    element_spelling as element_spelling,
)
from .spelling import (
    finish_binding as finish_binding,
)
from .spelling import (
    geometry_hits as geometry_hits,
)
from .spelling import (
    mirror_of as mirror_of,
)
from .spelling import (
    param_name as param_name,
)
from .spelling import (
    reconstruct_call as reconstruct_call,
)
from .spelling import (
    view_type as view_type,
)


@dataclass(frozen=True)
class Body:
    """A rendered body: the STRING both backends consume, plus what a wrapper needs
    to know about it.

    ``needs_random`` is set the instant the render emits a
    ``random_uniform``/``random_normal`` call, since both backends
    render through this one pass. ``False`` by default so a unit with
    no random op stays byte-identical to before this flag existed."""

    text: str
    needs_random: bool = False


def render_body(sinks: Sequence[Sink], walk: Walk, *, indent: str = "    ",
                kind=None) -> Body:
    """Render one kernel's body STRING from its sink set and its walk.

    The traversal is the walk's own; every slot name comes from
    :attr:`~hawk.ir.walk.Walk.arg_spec`. ``kind`` is the
    :class:`hawk.ext.Kind` the commits render under; omitted, the default applies."""
    return _Renderer(tuple(sinks), walk, indent, kind).run()


@dataclass(frozen=True)
class LaneSplit:
    """A fused-step body cut at its ONE lowered loop, for a backend that
    runs the loop's trips across a tile of samples (the host's
    interchanged loop, :mod:`hawk.emit.host`) instead of one sample at a
    time. Every text is the renderer's own spelling of the same nodes the
    plain body renders, so the arithmetic is the same op for op.

    ``pre`` is the guarded body up to the loop (its values, the carried
    initial values, the trip count), ``step`` one trip (the loop body, then
    ``const`` next values ``carries[j][3]`` and the ``const bool`` break
    flag ``exit``), ``post`` everything after the loop (the commits and the
    finish epilogue). ``guard`` is the body-wide mask condition (``None``
    without one); ``pre``/``post`` exclude it. ``carries`` holds ``(name,
    ttype, init text, next name)`` per carried slot; ``live_ins`` the
    ``(name, ttype)`` of each value ``pre`` names and ``step`` reads;
    ``sample_free`` the declarations, in ``pre``'s order, of those of them
    that read no per-sample value (a ``Param`` and constants only) and of the
    values they read, which a backend may evaluate once instead of carrying.
    ``count`` is the runtime trip-count text (``None`` for a static
    ``stop``), ``index`` the loop index's name. ``speculative`` says one trip
    may run on a finished sample and its result be discarded: the step reads
    no plane at a computed index and divides no integer. ``op_count`` is
    the step's operation count, ``fp64_ops`` its 64-bit floating-point
    operations (:func:`hawk.ir.cost.fp64_ops`)."""

    guard: str | None
    pre: str
    step: str
    post: str
    carries: tuple
    live_ins: tuple
    count: str | None
    stop: int
    index: str
    exit: str
    speculative: bool
    op_count: int
    needs_random: bool = False
    sample_free: tuple = ()
    fp64_ops: int = 0


#: A declaration the renderer emits (``const auto t5 = ...``, ``const
#: aether::Item<...> hz0 = ...``): group 1 is the declared name.
#: INTERIM (hawk 0.6.4), replaced by the 0.6.5 emit policy (template/runtime choice
#: from an IR register-pressure model): the element count above which a read-only
#: leaf read only by static ``component`` is bound as a proxy, not copied whole.
LAZY_LEAF_WIDTH = 32

_DECLARED = re.compile(r"^\s*const [^=;]*?\b(\w+) = ", re.M)
#: A lowered loop's header (``_for_header``): index type, begin, comparison, end.
_HEADER = re.compile(r"^for \((\w+) \w+ = (.*); \w+ ([<>]) (.*); [^;]*\) \{$")
_WORD = re.compile(r"\b\w+\b")


def render_lane_split(sinks: Sequence[Sink], walk: Walk, *, indent: str = "        ",
                      kind=None) -> LaneSplit | None:
    """The :class:`LaneSplit` of a fused-step kernel's body, or ``None`` when
    the body is not that shape.

    The shape is :func:`hawk.steps`' own: ONE top-level lowered loop with a
    tail ``break``, counted from 0 by 1 (a static ``stop`` or a runtime count),
    with no loop, tape or per-iteration commit of its own inside, and nothing
    after it reading a value named before it. Anything else answers
    ``None`` and keeps its plain body: a body with its own inner ``for`` would
    need a second interchange."""
    return _Renderer(tuple(sinks), walk, indent, kind).lane_split()


@dataclass(frozen=True)
class LaneLoop:
    """One loop of a :class:`LaneSegments` body, cut out for the host's
    loop interchange: ``step`` is ONE trip of it for one lane (the body,
    then the ``const`` next values ``carries[j][3]`` and the ``const bool``
    break flag ``exit``, ``"false"`` for a loop without a ``break``), the
    carried values and the iteration index read through their usual names.
    ``carries`` holds ``(name, ttype, init text, next name)`` per carried
    slot; the index runs ``start + trip * by`` for ``trip`` in ``[0,
    trips)``. ``speculative`` as :attr:`LaneSplit.speculative`.

    A loop with a runtime trip count carries ``bounds``: the header's own
    ``(index type, begin text, comparison, end text)``, which the segment
    before it evaluates per sample; a lane then runs while ``begin + trip *
    by`` compares true against its end, as the per-sample header does."""

    step: str
    carries: tuple
    index: str
    start: int
    by: int
    trips: int
    exit: str
    speculative: bool
    op_count: int
    bounds: tuple | None = None


@dataclass(frozen=True)
class LaneSegments:
    """A body cut at each of its top-level loops: ``segments[k]`` is the
    per-sample text before ``loops[k]`` (after ``loops[k - 1]``), the last
    one everything after the last loop (the commits and the finish
    epilogue). ``guard`` is the body-wide mask condition (``None`` without
    one); no segment contains it. ``types`` maps every name a segment may
    declare and a later text read to its IR type. Every text is the
    renderer's own spelling of the same nodes the plain body renders, so
    each sample runs the same operations in the same order."""

    guard: str | None
    segments: tuple
    loops: tuple
    types: dict


def render_lane_segments(sinks: Sequence[Sink], walk: Walk, cut: Sequence[Loop], *,
                         indent: str = "        ", kind=None) -> LaneSegments:
    """The :class:`LaneSegments` of a body whose top-level loops ``cut``
    (:func:`hawk.ir.vectorise.interchanged_loops`) run interchanged."""
    return _Renderer(tuple(sinks), walk, indent, kind).lane_segments(cut)


def _sample_free(node: Node) -> bool:
    """Whether ``node`` reads no per-sample value: built from ``Param`` and
    sample-count leaves and constants by non-random ops only."""
    stack, seen = [node], set()
    while stack:
        n = _through(stack.pop())
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Leaf):
            if n.role not in ("uniform", "nsamples"):
                return False
        elif isinstance(n, Op) and n.kind not in RANDOM_OPS and n.kind != RAW_KIND:
            stack.extend(n.operands)
        elif not isinstance(n, Const):
            return False
    return True


def _declarations(pre: str, names: Sequence[str]) -> tuple:
    """The one-line declarations in ``pre`` of ``names`` and of every value
    declared in ``pre`` they read, in ``pre``'s order; ``()`` when one is not
    a one-line declaration."""
    lines = {}
    for line in pre.splitlines():
        m = _DECLARED.match(line)
        if m:
            lines[m.group(1)] = line
    want, stack = set(), list(names)
    while stack:
        name = stack.pop()
        line = lines.get(name)
        if line is None or not line.rstrip().endswith(";"):
            return ()
        if name not in want:
            want.add(name)
            stack.extend(w for w in _WORD.findall(line.split("=", 1)[1]) if w in lines)
    return tuple(line.strip() for name, line in lines.items() if name in want)


class _Renderer(_DispatchMixin):
    """One rendering pass. Held as a class only to carry the memo tables."""

    def __init__(self, sinks: tuple, walk: Walk, indent: str, kind=None) -> None:
        from ..ext import DEFAULT_KIND

        self.sinks = sinks
        self.walk = walk
        self.indent = indent
        self.kind = kind if kind is not None else DEFAULT_KIND
        self.nodes, self.position = canonical_nodes(sinks)
        self.rendered: dict[int, str] = {}
        self.lines: list[str] = []
        self.hoists = 0
        self.temps = 0
        #: Set the instant a random op renders; see :class:`Body`.
        self.uses_random = False
        #: Set while the body-wide mask guard :meth:`run` opens is open.
        self.guarded = False
        #: id(node) -> its rendering INSIDE a loop scope. A loop body's nodes are not
        #: in the walk's flat order (a loop is a scope, not a run of statements the
        #: emitter may hoist); :meth:`_ref` consults this before the position table.
        self.local: dict[int, str] = {}
        #: id(loop) -> the suffix its emitted identifiers carry (:meth:`_index_loops`).
        self.loop_pos: dict[int, int] = {}
        #: id(node) -> the name of an ``aether::Item`` materialisation of that rank>=1
        #: value, for a runtime component read (``component_at``). Scoped like
        #: :attr:`local`: an Item declared inside a loop dies with that scope.
        self.items: dict[int, str] = {}
        #: The loop-carried C++ locals a lowered loop declares, by loop suffix.
        self.carry_names: dict[int, tuple] = {}
        #: Per-iteration TAPE arrays a forward loop keeps, by loop position ->
        #: {carried slot -> identifier}. Only set for a loop some reverse loop
        #: reads through a TapeRead.
        self.tape_names: dict[int, dict] = {}
        #: Per-sample iteration COUNTERS, by loop position -> identifier. Only set
        #: for a loop whose count a derived :class:`~hawk.ir.loop_nodes.LoopCount`
        #: reads.
        self.count_names: dict[int, str] = {}
        self.fanout: dict[int, int] = {}
        #: Positions a loop boundary forces to a name: a value a loop body reads from
        #: outside is computed once, before the loop, so an invariant ``View``
        #: subscript isn't a strided address computation repeated every iteration.
        self.forced: set = set()
        for node in self.nodes:
            # A Primitive is never rendered: it stands for its forward (operand 0),
            # and its declared inputs exist only for the derivative rule (IR-time).
            # So it's neither a reference nor a referent here — attributing fan-out
            # to it would alias an already-named forward, and the inlined body would
            # stop being byte-identical to the same arithmetic written longhand.
            if isinstance(node, Primitive):
                continue
            for child in node.operands:
                p = self.position[id(_through(child))]
                self.fanout[p] = self.fanout.get(p, 0) + 1
            if isinstance(node, Loop):
                for child in node.operands:
                    child = _through(child)
                    if isinstance(child, Leaf) and child.role in GATHER_ROLES:
                        # A plane read through at() is spelled BY NAME at the
                        # computed index (`_at`), never via a reference to the leaf —
                        # naming it here would emit a dead `plane[i].eval()`.
                        continue
                    self.forced.add(self.position[id(child)])
        self.lazy = self._lazy_leaves()
        self._index_loops()
        #: The loop :meth:`lane_split` cuts the body at, and what
        #: :meth:`_loop_split` recorded of it (``None`` on a plain render).
        self.split_loop: Loop | None = None
        self.split: dict | None = None
        #: The loops :meth:`lane_segments` cuts the body at, and what
        #: :meth:`_loop_cut` recorded of each, in order.
        self.cut_loops: frozenset = frozenset()
        self.cuts: list = []
        #: Positions the flat top-level walk reaches without crossing a Dispatch
        #: branch edge — see :meth:`_outside_reachable`. Computed once, after
        #: :attr:`position` exists and before anything renders.
        self._outside = self._outside_reachable()
        #: position -> the members of its fused ``switch`` group
        #: (:meth:`_fusion_groups`), for every member of a group of two or more.
        self._fused = self._fusion_groups()
        #: Per-component commits (:meth:`_plan_commits`): position -> the stores
        #: due once that node is rendered; the sinks they replace; the ``vec``
        #: positions no longer rendered whole.
        self._due: dict = {}
        self._held: set = set()
        self._skip: set = set()
        #: the slots an accumulate may reach before their own part is stored:
        #: zeroed first (``_plan_commits``)
        self._zero: list = []
        self._late_ready: dict = {}
        #: group position -> the accumulates rendered inside that group's cases
        self._inline_links: dict = {}
        #: How many lowered-loop scopes are open (a body sink renders inside one).
        self._scope_depth = 0

    def _lazy_leaves(self) -> frozenset:
        """Positions of the wide read-only leaves bound as the ``[i]`` proxy.

        INTERIM (hawk 0.6.4), replaced by the 0.6.5 emit policy (a template or
        runtime choice from an IR register-pressure model): a non-mutable
        rank>=1 leaf of more than :data:`LAZY_LEAF_WIDTH` elements whose every
        use is a ``component`` read at a static index is not copied whole
        (``.get()`` is one register per element); each read goes through the
        proxy."""
        used: dict[int, bool] = {}
        runtime: set = set()
        for node in self.nodes:
            if isinstance(node, Leaf):
                continue
            read = isinstance(node, Op) and node.kind in ("component", "component_at")
            for k, child in enumerate(node.operands):
                child = _through(child)
                if isinstance(child, Leaf):
                    p = self.position[id(child)]
                    used[p] = used.get(p, True) and read and k == 0
                    if read and k == 0 and node.kind == "component_at":
                        runtime.add(p)
        out = set()
        for p, ok in used.items():
            leaf = self.nodes[p]
            if (ok and leaf.ttype.shape and leaf.role not in ("mutable", "uniform", "nsamples")
                    and not is_param_cell(leaf)
                    and (math.prod(leaf.ttype.shape) > LAZY_LEAF_WIDTH
                         or (p in runtime and _reads_in_place(leaf)))):
                out.add(p)
        return frozenset(out)

    def _outside_reachable(self) -> frozenset:
        """Positions some path from a root reaches WITHOUT following a
        :class:`~hawk.ir.nodes.Dispatch` node's ``branches`` edges — values at least
        one use of which doesn't depend on which ``switch`` case runs, and so must
        stay computed once, before the switch.

        A Dispatch's own ``selector`` edge IS followed; only ``branches`` is
        excluded, so anything reached ONLY through a branch on EVERY path is absent
        here and :meth:`run` leaves it for :meth:`_dispatch_switches` to render, scoped
        to its case — or, when every non-elided case reads it, once above the
        switch.

        The exclusion is ``switch``-only; ``predicated`` evaluates every branch
        regardless, so it stays reachable like today. A :class:`Loop`'s body is
        never reached either way, since :attr:`Loop.operands` is only its
        ``inits``/``free`` boundary — the body is a separate scope
        :meth:`_render_scope` renders on its own."""
        seen: set = set()
        stack: list = list(self.sinks)
        while stack:
            n = stack.pop()
            pos = self.position[id(n)]
            if pos in seen:
                continue
            seen.add(pos)
            if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
                children = (n.selector,)
            else:
                children = n.operands
            stack.extend(children)
        return frozenset(seen)

    def _fusion_groups(self) -> dict:
        """Switch dispatches the flat walk reaches that share a selector node and
        ``K`` and do not depend on each other, grouped to render as ONE ``switch``
        (:meth:`_dispatch_switches`), emitted at the group's first member. Greedy
        in walk order: a dispatch joins the first group of its key none of whose
        members it depends on (a data dependence through its branches, however
        deep, closes the group to it)."""
        open_groups: dict = {}
        for node in self.nodes:
            if not (isinstance(node, Dispatch) and node.policy in SWITCH_RENDERED
                    and self.position[id(node)] in self._outside):
                continue
            groups = open_groups.setdefault((id(node.selector), node.K), [])
            deps = self._closure(node) if groups else frozenset()
            for g in groups:
                if not any(id(m) in deps for m in g):
                    g.append(node)
                    break
            else:
                groups.append([node])
        return {self.position[id(m)]: tuple(g)
                for groups in open_groups.values() for g in groups
                if len(g) > 1 for m in g}

    def _closure(self, root: Node) -> frozenset:
        """The ids of every node ``root`` reads, through ALL operands (branches
        included)."""
        seen: set = set()
        stack = [root]
        while stack:
            n = stack.pop()
            if id(n) not in seen:
                seen.add(id(n))
                stack.extend(n.operands)
        return frozenset(seen)

    def _index_loops(self) -> None:
        """Give every lowered loop a STABLE identifier suffix, then record which
        ones a reverse loop reads through a
        :class:`~hawk.ir.loop_nodes.TapeRead`.

        A top-level loop is suffixed by its canonical position (keeps a body
        byte-identical between renders); a nested loop has no position of its own
        (its body isn't in the flat order), so it takes the next suffix from a
        counter seeded above every position. The tape scan runs BEFORE anything
        emits, since a forward loop's tape array is declared above its loop."""
        counter = len(self.nodes)
        pending = list(self.nodes)
        while pending:
            node = pending.pop(0)
            if isinstance(node, Loop):
                if id(node) not in self.loop_pos:
                    held = self.position.get(id(node))
                    if held is None:
                        held, counter = counter, counter + 1
                    self.loop_pos[id(node)] = held
                pending.extend(node.body_nodes())
                pending.extend(node.nexts)
        pending = list(self.nodes)
        while pending:
            node = pending.pop()
            if isinstance(node, Loop):
                pending.extend(node.body_nodes())
                pending.extend(node.nexts)
            if isinstance(node, TapeRead):
                pos = self.loop_pos.get(id(node.loop))
                if pos is None:                    # pragma: no cover - structural
                    raise HawkError(
                        "a TapeRead names a forward loop that is not in this "
                        "kernel's walk: the taped loop must be an operand of the "
                        "reverse loop that reads it, so that the ordinary walk "
                        "order emits it first")
                self.tape_names.setdefault(pos, {})[node.slot] = (
                    f"hawk_tp{pos}_{node.slot}")
            if isinstance(node, LoopCount):
                pos = self.loop_pos[id(node.loop)]
                self.count_names[pos] = f"hawk_c{pos}"

    # -- driver ------------------------------------------------------------

    def run(self) -> Body:
        """Render the whole body.

        Under a mask guard the whole body sits inside ``if (!mask) { ... }``,
        opened before the first load, so a terminated sample does no load,
        arithmetic, store or accumulate. The masks are ``terminated``-role planes
        the body only reads, so the mask a sample enters with is the one it
        leaves with, and skipping the computation drops nothing the guarded
        commits would have stored. A kernel whose guard binds no mask renders
        exactly as before."""
        guard = self._guard_slot()
        if guard is not None:
            self._emit(f"if (!{guard}) {{")
            self.indent += "    "
            self.guarded = True
        self._plan_commits()
        for target, at in self._zero:
            self._emit(f"{target}({at}, i.global()) = {self._cast('0', 'f64')};")
        for target, width in self._accumulated_planes():
            # a plane a lowered loop accumulates at runtime indices once per
            # iteration (`bar[i].at(k) += e`, constraint 7): zeroed whole, once,
            # before the walk; the loop's links are the only other writes
            self._emit(f"for (Int hk = 0; hk < {width}; ++hk) "
                       f"{target}[i].at(hk) = {self._cast('0', 'f64')};")
        for node in self.nodes:
            if isinstance(node, Sink):
                continue
            self._emit_outside(node)
        if self._due:                                  # pragma: no cover
            raise HawkError("a per-component commit was planned for a value the "
                            "walk never rendered")
        finish = None
        for sink in self.sinks:
            if not isinstance(sink, Sink) or id(sink) in self._held:
                continue        # a lowered loop ROOT (emitted, scope and all), or a
                                # sink already committed per component
            if isinstance(sink, Finish):
                finish = sink   # the epilogue: after EVERY other commit, below
                continue
            self._sink(sink)
        if finish is not None:
            self._finish(finish)
        if guard is not None:
            self.indent = self.indent[:-4]
            self._emit("}")
        return Body("\n".join(self.lines), self.uses_random)

    def _accumulated_planes(self) -> list:
        """``(target, width)`` of every plane some lowered loop in the walk
        accumulates per iteration (:func:`_accumulate_sink`), each once."""
        out, seen = [], set()
        pending = [n for n in self.nodes if isinstance(n, Loop)]
        while pending:
            loop = pending.pop()
            pending.extend(n for n in loop.body_nodes() if isinstance(n, Loop))
            for sink in loop.body_sinks:
                links = _accumulate_sink(sink)
                if links is None:
                    continue
                target = binding_name(sink.role, sink.name)
                if target not in seen:
                    seen.add(target)
                    out.append((target, math.prod(sink.ttype.shape)))
        return out

    def _plan_commits(self) -> None:
        """Choose the own-column sinks that commit per component.

        An ``Assign``/``WideWrite`` whose value is a ``vec`` stores component
        ``k`` as ``target(k, i.global()) = part_k;`` right after the statement
        that defines the part, so a wide value is never live whole. Only when
        no leaf reads the plane (its prior value would see the early store),
        the sink is not the compensated target, and the body is a plain one
        (a lane split or segment cut moves statements between texts). Anything
        else keeps today's single store at the end."""
        if self.split_loop is not None or self.cut_loops:
            return
        read = {(n.role, n.name) for n in self.nodes if isinstance(n, Leaf)}
        ready = self._ready_times()
        compensated = self.kind.sink.id == "compensated"
        users: dict = {}
        planned = []
        for sink in self.sinks:
            value = getattr(sink, "value", None)
            base, links = _chain(value)
            parts = _parts(base)
            if (not isinstance(sink, (Assign, WideWrite)) or sink.index is not None
                    or parts is None
                    or (sink.role, sink.name) in read
                    or (compensated and (sink.name == self.kind.sink.compensation
                                         or self._is_compensated_own_column(sink)))
                    or any(self.position[id(p)] not in self._outside
                           for p in parts)
                    or any(self.position[id(n)] not in self._outside
                           for k, e in links for n in (k, e))):
                continue
            planned.append((sink, base, links, parts))
        # everything the walk positions after the first planned base is reached
        # only through runtime links or later sinks: ready when its inputs are
        late = min((self.position[id(b)] for _s, b, _l, _p in planned), default=-1)
        for sink, base, links, parts in planned:
            target = binding_name(sink.role, sink.name)
            # a runtime accumulate is due once both its index and its value are;
            # a static part stored BEFORE the first accumulate is a plain store,
            # any other is an accumulate over the zero the prologue wrote
            due_links = [max(self._ready_at(k, ready, late), self._ready_at(e, ready, late))
                         for k, e in links]
            first = min(due_links, default=None)
            for at, part in zip(_flat_indices(sink.ttype.shape), parts):
                when = self._ready_at(part, ready, late)
                plain = first is None or when < first
                if not plain:
                    self._zero.append((target, ", ".join(map(str, at))))
                self._due.setdefault(when, []).append(
                    (target, ", ".join(map(str, at)), part, "=" if plain else "+="))
            for when, (k, e) in zip(due_links, links):
                if not self._link_in_case(target, k, e):
                    self._due.setdefault(when, []).append((target, k, e, "at+="))
            self._held.add(id(sink))
            node = sink.value
            while node is not base:
                users[self.position[id(node)]] = users.get(self.position[id(node)], 0) + 1
                node = node.operands[0]
            pos = self.position[id(base)]
            users[pos] = users.get(pos, 0) + 1
        self._skip = {pos for pos, n in users.items() if self.fanout.get(pos, 0) == n}

    def _link_in_case(self, target: str, index: Node, value: Node) -> bool:
        """Whether an accumulate renders inside the cases of the one fused
        ``switch`` that defines its index and its value, so a case that does not
        take the link writes nothing (and no slot is touched by a no-op add). It
        needs both to be members of that group and read by this link alone."""
        pi, pv = self.position[id(index)], self.position[id(value)]
        group = self._fused.get(pi)
        if (group is None or group != self._fused.get(pv)
                or not isinstance(index, Dispatch) or not isinstance(value, Dispatch)
                or self.fanout.get(pi, 0) != 1 or self.fanout.get(pv, 0) != 1):
            return False
        self._inline_links.setdefault(self.position[id(group[0])], []).append(
            (target, index, value))
        return True

    def _ready_times(self) -> list:
        """Per position: the position whose render makes the node renderable.

        A fused ``switch`` member renders when its group's first member does, a
        loop at its own position, everything else whenever its operands are
        there; so a node is ready at the latest of those over what it reads
        (``-1``: it reads none)."""
        ready: list = []
        for pos, node in enumerate(self.nodes):
            if isinstance(node, Dispatch) and node.policy in SWITCH_RENDERED:
                kids = (node.selector,)
            else:
                kids = node.operands
            at = max((ready[self.position[id(c)]] for c in kids), default=-1)
            group = self._fused.get(pos)
            if group is not None:
                at = max(at, self.position[id(group[0])])
            elif isinstance(node, Loop):
                at = max(at, pos)
            ready.append(at)
        return ready

    def _ready_at(self, part: Node, ready: list, late: int = -1) -> int:
        """The position at whose render ``part`` can be rendered and stored.

        A node the walk reaches only through the sink's runtime links sits
        AFTER the base (position > ``late``) although its inputs are early:
        it is ready when they are (the deepest early input, or the switch
        group it is a member of), and :meth:`_flush` renders it there."""
        pos = self.position[id(part)]
        if pos <= late or pos not in self._outside:
            return pos if ready[pos] < 0 else min(ready[pos], pos)
        memo = self._late_ready
        if pos in memo:
            return memo[pos]
        if isinstance(part, Dispatch) and part.policy in SWITCH_RENDERED:
            kids = (part.selector,)
        else:
            kids = part.operands
        at = max((self._ready_at(c, ready, late) for c in kids), default=0)
        group = self._fused.get(pos)
        if group is not None:
            at = max(at, self.position[id(group[0])])
        memo[pos] = at
        return at

    def _flush(self, pos: int) -> None:
        """The per-component stores due now that ``pos`` is rendered. A part
        whose inputs are all there is rendered here (with whatever of its own it
        still needs, in walk order) rather than at its turn in the walk."""
        for target, at, part, op in self._due.pop(pos, ()):
            need, seen, stack = [], set(), [part] + ([at] if op == "at+=" else [])
            while stack:
                n = stack.pop()
                p = self.position[id(n)]
                if p in self.rendered or p not in self._outside or p in seen:
                    continue
                seen.add(p)
                need.append(n)
                stack.extend((n.selector,) if isinstance(n, Dispatch)
                             and n.policy in SWITCH_RENDERED else n.operands)
            for n in sorted(need, key=lambda n: self.position[id(n)]):
                self._emit_outside(n)
            if op == "at+=":
                self._emit(f"{target}[i].at({self._ref(at)}) += {self._ref(part)};")
            else:
                self._emit(f"{target}({at}, i.global()) {op} {self._ref(part)};")

    def _emit_outside(self, node: Node) -> None:
        """:meth:`_render_outside`, then the commits that were waiting on it."""
        pos = self.position[id(node)]
        if pos in self._skip:
            self.rendered.setdefault(pos, "")   # committed per part, never whole
            return
        self._render_outside(node)
        if self._due:
            group = self._fused.get(pos, ())
            for held in (pos, *(self.position[id(m)] for m in group)):
                self._flush(held)

    def _render_outside(self, node: Node) -> None:
        """Render one node of the flat walk at the top scope, unless it is
        reached only through a Dispatch's branches (rendered scoped to its case(s),
        :meth:`_dispatch_switches`) or already rendered ahead of its turn as an
        operand of a fused group."""
        pos = self.position[id(node)]
        if pos not in self._outside or pos in self.rendered:
            return
        group = self._fused.get(pos)
        if group is not None:
            self._emit_group(group)
            return
        if isinstance(node, Loop):
            # A lowered `for` is a STATEMENT, not a value: it emits its carried
            # locals, header and scope here; its values are read through
            # LoopValue projections.
            if node is self.split_loop:
                self._loop_split(node)
            elif id(node) in self.cut_loops:
                self._loop_cut(node)
            else:
                self._loop(node)
            self.rendered[pos] = ""
            return
        self._bind(node, pos, self._expr(node, materialised=self._materialised(node, pos)))

    def _bind(self, node: Node, pos: int, text: str) -> None:
        """Record ``text`` as ``node``'s rendering, named when :meth:`_materialised`."""
        if self._materialised(node, pos):
            name = self._name(node, pos)
            self._emit(f"const auto {name} = {text};")
            self.rendered[pos] = name
        else:
            self.rendered[pos] = text

    def _emit_group(self, group: tuple) -> None:
        """A fused ``switch`` group, at its first member. Everything a later member's
        branches read that the flat walk has not reached yet is rendered first
        (exact: it is rendered anyway, only earlier, and none of it reads a member)."""
        members = {id(m) for m in group}
        seen: set = set()
        stack = [b for m in group for b in m.branches]
        need: list = []
        while stack:
            n = stack.pop()
            if id(n) in seen or id(n) in members:
                continue
            seen.add(id(n))
            pos = self.position.get(id(n))
            if pos is not None and pos in self.rendered:
                continue
            if pos is not None and pos in self._outside:
                need.append(n)
            stack.extend(n.operands)
        for n in sorted(need, key=lambda n: self.position[id(n)]):
            self._emit_outside(n)
        for node, name in zip(group, self._dispatch_switches(group)):
            if name is None:                   # an accumulate of a case, not a value
                self.rendered.setdefault(self.position[id(node)], "")
            else:
                self._bind(node, self.position[id(node)], name)

    def _finish(self, sink: Finish) -> None:
        """The finish epilogue — the last text of the guarded scope, so the step's
        own commits land before the sample is marked: an unfinished sample whose
        condition holds sets its mask (via :func:`finish_binding`) and adds one to
        the reserved counter (``hawk_abi::finish_count``)."""
        mask = binding_name("terminated", sink.name)
        cond = self._ref(sink.value)
        counter = binding_name("lookup", FINISHED_PLANE)
        self._emit(f"if (!{mask}[i].eval() && {cond}) {{ "
                   f"{finish_binding(sink.name)}[i] = true; "
                   f"hawk_abi::finish_count({counter}); }}")

    def _guard_slot(self) -> str | None:
        """The guard's CONDITION against the walk's own slots: each mask the
        kind's guard names counts only if bound in the ``terminated`` role, and
        the commit is skipped when any bound one is set. ``Guard(mask=None)``
        never fires, nor does a guard none of whose masks the kernel binds."""
        bound = [binding_name("terminated", m) for m in self.kind.guard.names
                 if ("terminated", m) in self.walk.slot_of]
        if not bound:
            return None
        if len(bound) == 1:
            return f"{bound[0]}[i].eval()"
        return "(" + " || ".join(f"{b}[i].eval()" for b in bound) + ")"

    def _emit(self, line: str) -> None:
        self.lines.append(self.indent + line)

    def _materialised(self, node: Node, pos: int) -> bool:
        """A node with fan-out >= 2 is named — leaves and literals included,
        because a ``View`` subscript is a strided address computation. A
        by-value scalar leaf (``uniform``, ``nsamples``) is the exception: its
        rendering is already a bare identifier, so there's no address
        computation to hoist.

        Three renderings are named unconditionally because they produce an
        aether LEAF value (``Item``/``Constant``), which every expression node
        stores BY REFERENCE: an unnamed one would dangle past its
        full-expression.

        A rank>=1 leaf is the fourth, and it's named for CORRECTNESS: aether's
        reductions (``dot``/``norm``/``sum``/``matDot``) are sample-free by
        contract, folding at ``SampleIndex::make(0)`` over an already-materialized
        operand. Handing one a whole-plane ``View`` would silently answer with
        sample 0's reduction for every sample. The subscript is where a
        per-sample value BECOMES one, not an optimisation the fan-out rule may
        trade away."""
        if isinstance(node, Leaf) and node.role in ("uniform", "nsamples"):
            return False
        if is_param_cell(node):
            return False        # a bare identifier, loaded once before the loop
        if isinstance(node, (SampleIndex, LoopIndex, LoopCarry)):
            return False        # already a bare identifier the enclosing scope bound
        if isinstance(node, (Loop, Const, Primitive)):
            return False        # a statement / a literal / a boundary: none is named
        if pos in self.forced:
            return True
        return self.fanout.get(pos, 0) >= 2 or self._leaf_valued(node)

    @staticmethod
    def _leaf_valued(node: Node) -> bool:
        if isinstance(node, (At, Leaf)):
            return bool(node.ttype.shape)
        return isinstance(node, Op) and node.kind in ("splat", "vec")

    def _name(self, node: Node, pos: int) -> str:
        if isinstance(node, Leaf):
            return f"{binding_name(node.role, node.name)}_i"
        return f"t{pos}"

    def _ref(self, child: Node) -> str:
        held = self.local.get(id(child))
        if held is not None:
            return held
        return self.rendered[self.position[id(child)]]

    def _hoist(self, text: str) -> str:
        """Bind ``text`` to a fresh named local ahead of the current line — the
        lifetime-safe home for a value that is an aether LEAF but not a node of
        its own (a broadcast ``Constant``, a raw block's operand)."""
        name = f"hz{self.hoists}"
        self.hoists += 1
        self._emit(f"const auto {name} = {text};")
        return name

    # -- lowered loops -----------------------------------------------

    def _loop(self, loop: Loop) -> None:
        """Emit ONE lowered ``for``: carried locals, an optional tape, the plain
        C++/CUDA header, the body's own scope, and the carried updates.

        Deliberately missing: no ``#pragma unroll``, no trip-count threshold, no
        decision about whether the loop should be a loop at all. Every constant
        the body needs reaches the translation unit as a literal, so
        ``nvcc``/``g++`` can see the trip count and unroll (or not) with register
        and instruction-cache pressure in view — HAWK has neither fact.

        Carried values are declared before the loop and reassigned at the end of
        each iteration through one named temporary each, so a body whose carries
        read one another (``T_next = 2*z*T - T_prev``) updates them
        SIMULTANEOUSLY: assigning in place as each result became available would
        feed one carry's new value into the next carry's expression — a
        different, silently wrong recurrence."""
        pos = self.loop_pos[id(loop)]
        elems = [carry_type(c.ttype) for c in loop.carries]
        names = tuple(f"hawk_v{pos}_{j}_{_ident(c.name)}"
                      for j, c in enumerate(loop.carries))
        self.carry_names[pos] = names
        for name, elem, init in zip(names, elems, loop.inits):
            self._emit(f"{elem} {name} = {self._ref(init)};")
        tapes = self.tape_names.get(pos, {})
        for slot in sorted(tapes):
            self._emit(f"{elems[slot]} {tapes[slot]}[{loop.trip}];")
        counter = self.count_names.get(pos)
        if counter is not None:
            self._emit(f"Int {counter} = 0;")
        index = f"hawk_k{pos}_{_ident(loop.index.name)}"
        count = None if loop.count is None else self._ref(loop.count)
        self._emit(_for_header(index, loop, count))
        outer, self.indent = self.indent, self.indent + "    "
        saved, saved_items = self.local, self.items
        self.local, self.items = dict(saved), dict(saved_items)
        self._scope_depth += 1
        self.local[id(loop.index)] = index
        for carry, name in zip(loop.carries, names):
            self.local[id(carry)] = name
        for slot in sorted(tapes):
            self._emit(f"{tapes[slot]}[{_ordinal(index, loop)}] = {names[slot]};")
        form = loop.exit_form
        if form in ("head", "middle"):
            pre = self._exit_part(loop)
            self._render_scope(loop, only=pre, sinks=False)
            self._emit(f"if ({self._ref(loop.exit_cond)}) {{")
            self.indent += "    "
            moved = [(j, v) for j, v in enumerate(loop.exit_values)
                     if v is not loop.carries[j]]
            for j, value in moved:
                self._emit(f"const {elems[j]} hawk_e{pos}_{j} = {self._ref(value)};")
            for j, _value in moved:
                self._emit(f"{names[j]} = hawk_e{pos}_{j};")
            if counter is not None and form == "middle":
                self._emit(f"++{counter};")   # the exiting iteration ran in part: its row counts
            self._emit("break;")
            self.indent = self.indent[:-4]
            self._emit("}")
            self._render_scope(loop, skip=pre)
        else:
            self._render_scope(loop)
        results = []
        for j, (elem, nxt) in enumerate(zip(elems, loop.nexts)):
            result = f"hawk_n{pos}_{j}"
            self._emit(f"const {elem} {result} = {self._ref(nxt)};")
            results.append(result)
        if form == "tail":
            # The condition reads the iteration's values, possibly inlined over
            # the carried locals, so it's evaluated BEFORE they move.
            self._emit(f"const bool hawk_x{pos} = {self._ref(loop.exit_cond)};")
        for name, result in zip(names, results):
            self._emit(f"{name} = {result};")
        if counter is not None:
            self._emit(f"++{counter};")
        if form == "tail":
            self._emit(f"if (hawk_x{pos}) break;")
        self.local, self.items = saved, saved_items
        self._scope_depth -= 1
        self.indent = outer
        self._emit("}")

    # -- the lane split (render_lane_split) ---------------------------

    def _split_candidate(self) -> Loop | None:
        """The ONE top-level loop :func:`render_lane_split` admits, else ``None``."""
        loops = [n for n in self.nodes if isinstance(n, Loop)]
        if len(loops) != 1 or self.tape_names or self.count_names:
            return None
        loop = loops[0]
        if (self.position[id(loop)] not in self._outside or loop.exit_form != "tail"
                or loop.body_sinks or (loop.start, loop.step) != (0, 1)
                or loop.count_tail):
            return None
        if any(isinstance(n, (Loop, TapeRead, LoopCount)) for n in loop.body_nodes()):
            return None
        return loop

    @staticmethod
    def _speculative(loop: Loop) -> bool:
        """Whether one trip of ``loop`` is safe to run and discard on a
        finished sample (:func:`hawk.ir.vectorise.speculative_step`)."""
        from ..ir.vectorise import speculative_step

        return speculative_step(loop)

    def lane_split(self) -> LaneSplit | None:
        """Render the body with its loop cut out (:class:`LaneSplit`)."""
        loop = self._split_candidate()
        if loop is None:
            return None
        self.split_loop = loop
        guard = self._guard_slot()
        body = self.run()
        info = self.split
        lines = self.lines
        start, end = (1, len(lines) - 1) if guard is not None else (0, len(lines))
        pre = "\n".join(lines[start:info["mark"]])
        post = "\n".join(lines[info["mark"]:end])
        declared = set(_DECLARED.findall(pre))
        if declared & set(_WORD.findall(post)):
            return None             # the tail reads a value the head named
        live_ins = []
        for value in loop.free:
            name = self._ref(value)
            if name in declared and all(n != name for n, _ in live_ins):
                live_ins.append((name, value.ttype))
        step_words = set(_WORD.findall(info["step"]))
        if (declared & step_words) - {n for n, _ in live_ins}:
            return None             # the step reads a head value not on the boundary
        live_ins = [(n, t) for n, t in live_ins if n in step_words]
        ops = sum(isinstance(n, Op) for n in loop.body_nodes())
        free = {self._ref(value) for value in loop.free if _sample_free(value)}
        return LaneSplit(guard, pre, info["step"], post, info["carries"],
                         tuple(live_ins), info["count"], loop.stop, info["index"],
                         info["exit"], self._speculative(loop), ops,
                         body.needs_random,
                         _declarations(pre, [n for n, _ in live_ins if n in free]),
                         fp64_ops(loop.body_nodes()))

    def _loop_split(self, loop: Loop) -> None:
        """:meth:`_loop` for the loop :meth:`lane_split` cuts at: marks where
        the head ends and renders ONE trip into its own text, the carried
        values read through their usual names, which the backend binds."""
        pos = self.loop_pos[id(loop)]
        names = tuple(f"hawk_v{pos}_{j}_{_ident(c.name)}"
                      for j, c in enumerate(loop.carries))
        self.carry_names[pos] = names
        inits = [self._ref(init) for init in loop.inits]
        index = f"hawk_k{pos}_{_ident(loop.index.name)}"
        count = None if loop.count is None else self._ref(loop.count)
        mark = len(self.lines)
        lines, self.lines = self.lines, []
        outer, self.indent = self.indent, self.indent + "    "
        saved, saved_items = self.local, self.items
        self.local, self.items = dict(saved), dict(saved_items)
        self.local[id(loop.index)] = index
        for carry, name in zip(loop.carries, names):
            self.local[id(carry)] = name
        self._render_scope(loop)
        nexts = []
        for j, (carry, nxt) in enumerate(zip(loop.carries, loop.nexts)):
            result = f"hawk_n{pos}_{j}"
            self._emit(f"const {carry_type(carry.ttype)} {result} = {self._ref(nxt)};")
            nexts.append(result)
        flag = f"hawk_x{pos}"
        self._emit(f"const bool {flag} = {self._ref(loop.exit_cond)};")
        step = "\n".join(self.lines)
        self.lines = lines
        self.local, self.items = saved, saved_items
        self.indent = outer
        self.split = {
            "mark": mark, "step": step, "index": index, "count": count, "exit": flag,
            "carries": tuple((name, c.ttype, init, nxt) for name, c, init, nxt
                             in zip(names, loop.carries, inits, nexts)),
        }

    # -- the lane segments (render_lane_segments) ------------------------

    def lane_segments(self, cut: Sequence[Loop]) -> LaneSegments:
        """Render the body with each loop of ``cut`` cut out
        (:class:`LaneSegments`)."""
        self.cut_loops = frozenset(id(loop) for loop in cut)
        guard = self._guard_slot()
        self.run()
        lines = self.lines
        start, end = (1, len(lines) - 1) if guard is not None else (0, len(lines))
        marks = [start, *(c["mark"] for c in self.cuts), end]
        segments = tuple("\n".join(lines[a:b]) for a, b in zip(marks, marks[1:]))
        types: dict = {}
        for node in self.nodes:
            name = self.rendered.get(self.position[id(node)])
            if name and _WORD.fullmatch(name) and getattr(node, "ttype", None) is not None:
                types.setdefault(name, node.ttype)
        for node_id, name in self.items.items():
            node = next((n for n in self.nodes if id(n) == node_id), None)
            if node is not None:
                types.setdefault(name, node.ttype)
        loops = []
        for c in self.cuts:
            for name, ttype, _, _ in c["carries"]:
                types[name] = ttype
            loops.append(LaneLoop(c["step"], c["carries"], c["index"], c["start"],
                                  c["by"], c["trips"], c["exit"], c["speculative"],
                                  c["ops"], c["bounds"]))
        return LaneSegments(guard, segments, tuple(loops), types)

    def _loop_cut(self, loop: Loop) -> None:
        """:meth:`_loop` for a loop :meth:`lane_segments` cuts at: marks where
        the segment before it ends and renders ONE trip into its own text. A
        ``break`` at the head or the middle of the body becomes a select: the
        trip's next values are the values at the break when the condition
        holds (the rest of the body computed and discarded when the trip may
        run speculatively, skipped otherwise), and the lane stops."""
        pos = self.loop_pos[id(loop)]
        names = tuple(f"hawk_v{pos}_{j}_{_ident(c.name)}"
                      for j, c in enumerate(loop.carries))
        self.carry_names[pos] = names
        inits = [self._ref(init) for init in loop.inits]
        index = f"hawk_k{pos}_{_ident(loop.index.name)}"
        bounds = None
        if loop.count is not None:
            header = _HEADER.match(_for_header(index, loop, self._ref(loop.count)))
            if header is None:                       # pragma: no cover - structural
                raise HawkError(f"lane segments: cannot read the header of loop {pos}")
            bounds = header.group(1, 2, 3, 4)
        mark = len(self.lines)
        lines, self.lines = self.lines, []
        outer, self.indent = self.indent, self.indent + "    "
        saved, saved_items = self.local, self.items
        self.local, self.items = dict(saved), dict(saved_items)
        self.local[id(loop.index)] = index
        for carry, name in zip(loop.carries, names):
            self.local[id(carry)] = name
        elems = [carry_type(c.ttype) for c in loop.carries]
        speculative = self._speculative(loop)
        form = loop.exit_form
        flag = f"hawk_x{pos}"
        nexts = [f"hawk_n{pos}_{j}" for j in range(len(names))]
        if form in ("head", "middle"):
            pre = self._exit_part(loop)
            self._render_scope(loop, only=pre, sinks=False)
            self._emit(f"const bool {flag} = {self._ref(loop.exit_cond)};")
            held = [f"hawk_e{pos}_{j}" for j in range(len(names))]
            for elem, name, value in zip(elems, held, loop.exit_values):
                self._emit(f"const {elem} {name} = {self._ref(value)};")
            if speculative:
                self._render_scope(loop, skip=pre)
                for j, (elem, nxt) in enumerate(zip(elems, loop.nexts)):
                    self._emit(f"const {elem} hawk_r{pos}_{j} = {self._ref(nxt)};")
                    self._emit(f"const {elem} {nexts[j]} = "
                               f"{flag} ? {held[j]} : hawk_r{pos}_{j};")
            else:
                for elem, nxt, kept in zip(elems, nexts, held):
                    self._emit(f"{elem} hawk_w{nxt[6:]} = {kept};")
                self._emit(f"if (!{flag}) {{")
                self.indent += "    "
                self._render_scope(loop, skip=pre)
                for nxt, value in zip(nexts, loop.nexts):
                    self._emit(f"hawk_w{nxt[6:]} = {self._ref(value)};")
                self.indent = self.indent[:-4]
                self._emit("}")
                for elem, nxt in zip(elems, nexts):
                    self._emit(f"const {elem} {nxt} = hawk_w{nxt[6:]};")
        else:
            self._render_scope(loop)
            for elem, nxt, value in zip(elems, nexts, loop.nexts):
                self._emit(f"const {elem} {nxt} = {self._ref(value)};")
            if form == "tail":
                self._emit(f"const bool {flag} = {self._ref(loop.exit_cond)};")
        step = "\n".join(self.lines)
        self.lines = lines
        self.local, self.items = saved, saved_items
        self.indent = outer
        self.cuts.append({
            "mark": mark, "step": step, "index": index, "start": loop.start,
            "by": loop.step, "trips": loop.trip,
            "exit": "false" if form is None else flag, "speculative": speculative,
            "ops": sum(isinstance(n, Op) for n in _flat_body(loop)), "bounds": bounds,
            "carries": tuple((name, c.ttype, init, nxt) for name, c, init, nxt
                             in zip(names, loop.carries, inits, nexts)),
        })

    @staticmethod
    def _exit_part(loop: Loop) -> frozenset:
        """The body nodes a ``break`` needs — its condition, the carried values
        at the break, and everything they read. Rendered BEFORE the test; the
        rest of the body after, so an exiting iteration computes nothing it then
        discards."""
        body = {id(n) for n in loop.body_nodes()}
        need: set = set()
        stack = [loop.exit_cond, *loop.exit_values]
        while stack:
            node = stack.pop()
            if id(node) not in body or id(node) in need:
                continue
            need.add(id(node))
            stack.extend(node.operands)
        return frozenset(need)

    def _render_scope(self, loop: Loop, *, only: frozenset | None = None,
                      skip: frozenset = frozenset(), sinks: bool = True) -> None:
        """Render a loop body's own nodes inside the open scope.

        Same expression rules as the flat path (one renderer): fan-out >= 2
        inside the body is named, an aether leaf value is named unconditionally,
        everything else inlines. Fan-out is counted WITHIN the body — that's the
        scope a name would live in; a value read from outside was already named
        by the boundary rule before the loop opened."""
        body = loop.body_nodes()
        # the add_at chain of a per-iteration accumulate is committed link by
        # link by its sink (:func:`_accumulate_sink`), never rendered as a value
        chains: set = set()
        for sink in loop.body_sinks:
            if _accumulate_sink(sink) is not None:
                node = sink.value
                while isinstance(node, Op) and node.kind == "add_at":
                    chains.add(id(node))
                    node = node.operands[0]
        if chains:
            skip = frozenset(skip) | frozenset(chains)
        fanout: dict[int, int] = {}
        exits = ([] if loop.exit_cond is None
                 else [loop.exit_cond, *loop.exit_values])
        for node in list(body) + list(loop.nexts) + exits:
            if isinstance(node, Primitive):
                continue                  # not a reference, not a referent
            for child in node.operands:
                child = _through(child)
                fanout[id(child)] = fanout.get(id(child), 0) + 1
        for node in body:
            if id(node) in skip or (only is not None and id(node) not in only):
                continue
            if isinstance(node, Loop):
                self._loop(node)
                self.local[id(node)] = ""
                continue
            if isinstance(node, Primitive):
                self.local[id(node)] = self._ref(node.forward)
                continue
            named = (fanout.get(id(node), 0) >= 2 or self._leaf_valued(node))
            text = self._expr(node, materialised=named)
            if named:
                name = f"tb{self.temps}"
                self.temps += 1
                self._emit(f"const auto {name} = {text};")
                self.local[id(node)] = name
            else:
                self.local[id(node)] = text
        if not loop.body_sinks or not sinks:
            return
        # Inside the body-wide guard :meth:`run` opened, the loop's commits are
        # already reached only by a live sample: testing the mask again on every
        # iteration would be a load and nothing else.
        guard = None if self.guarded else self._guard_slot()
        if guard is not None:
            self._emit(f"if (!{guard}) {{")
            self.indent += "    "
        for sink in loop.body_sinks:
            self._sink(sink)
        if guard is not None:
            self.indent = self.indent[:-4]
            self._emit("}")

    def _loop_value(self, node: LoopValue) -> str:
        """One carried slot's final value: the C++ local the loop left it in."""
        return self.carry_names[self.loop_pos[id(node.loop)]][node.slot]

    def _tape_read(self, node: TapeRead) -> str:
        """A forward loop's carried value at one iteration, out of its tape."""
        held = self.tape_names.get(self.loop_pos[id(node.loop)], {}).get(node.slot)
        if held is None:                          # pragma: no cover - structural
            raise HawkError(
                f"a TapeRead of slot {node.slot} reached the emitter but its "
                "forward loop declared no tape for that slot")
        return f"{held}[{_ordinal(self._ref(node.index), node.loop)}]"

    # -- sinks ------------------------------------------------------

    def _is_compensated_own_column(self, sink: Sink) -> bool:
        """Whether ``sink`` is the compensated kind's own-column target —
        resolved through :func:`hawk.ext.compensated_target`, the SAME rule the
        tracer used to decide whether to synthesise a companion, so the two can
        never disagree about which sink this is."""
        if self.kind.sink.id != "compensated":
            return False
        from ..ext import compensated_target

        resolved = compensated_target(self.kind, self.sinks)
        return resolved is not None and resolved == (sink.name, True)

    def _sink(self, sink: Sink) -> None:
        if (self.kind.sink.id == "compensated"
                and sink.name == self.kind.sink.compensation):
            # The compensation plane is the seam's to commit (below), not the
            # body's — a second commit here would duplicate its subscript and
            # add to the correction the seam just wrote. The synthesised
            # own-column companion's dummy commit is a rank-matching zero that
            # has no direct ``Const`` spelling at rank>=1 anyway (a broadcast
            # literal renders only through a ``splat``).
            return
        target = binding_name(sink.role, sink.name)
        if isinstance(sink, TapeWrite):
            self._tape_write(sink, target)
            return
        if isinstance(sink, SampleSum):
            self._sample_sum(sink)
            return
        links = _accumulate_sink(sink) if self._scope_depth else None
        if links is not None:
            # a per-iteration runtime accumulate (constraint 7): each link on
            # the own column where it stands, nothing for the zero base
            for k, e in links:
                self._emit(f"{target}[i].at({self._ref(k)}) += {self._ref(e)};")
            return
        value = self._ref(sink.value)
        if not isinstance(sink, (Assign, WideWrite, AccumWrite)):
            raise HawkError(f"no lowering for sink kind {sink.kind!r}")
        if sink.index is None:
            # The own column: one elementwise contribution, one store (eagle
            # folds the partials, the body never reduces) — unless this is the
            # compensated kind's own-column target, which accumulates into the
            # host's prior value instead.
            if self._is_compensated_own_column(sink):
                # `i` is already the body's bound `aether::SampleIndex`, so
                # unlike a computed scatter index it needs no `_sample_index` wrap.
                comp = binding_name("mutable", self.kind.sink.compensation)
                self._emit(
                    f"hawk_abi::store_compensated({target}, {comp}, i, {value});")
                return
            self._emit(f"{target}[i] = {value};")
            return
        index = self._sample_index(self._ref(sink.index))
        if not isinstance(sink, AccumWrite):
            # A scattered wide_write targets a computed ROW: a store, exactly as
            # the reference evaluator reads it.
            self._emit(f"{target}[{index}] = {value};")
            return
        self._commit_scattered(sink, target, index, value)

    def _tape_write(self, sink: TapeWrite, target: str) -> None:
        """One tape ROW on the own column, under ``if (on)``: component
        ``row*w + c`` of the sample's ``(W*w,)`` plane gets the ``c``-th
        flattened component of the stored values, in order. Rendered where the
        sink stands — a twin puts it FIRST, so it reads the pre-step state
        before any commit overwrites it."""
        row = self._ref(sink.row)
        lines = []
        offset = 0
        for value in sink.values:
            text = self._ref(value)
            shape = value.ttype.shape
            comps = ([((), text)] if not shape
                     else [(idx, f"{text}.eval<{', '.join(map(str, idx))}>(i)")
                           for idx in _flat_indices(shape)])
            for _idx, comp in comps:
                lines.append(f"{target}(static_cast<std::size_t>(hawk_tape_row * "
                             f"{sink.width} + {offset}), i.global()) = {comp};")
                offset += 1
        self._emit(f"if ({self._ref(sink.on)} != 0) {{")
        self.indent += "    "
        self._emit(f"const std::int64_t hawk_tape_row = {row};")
        for line in lines:
            self._emit(line)
        self.indent = self.indent[:-4]
        self._emit("}")

    def _sample_sum(self, sink: SampleSum) -> None:
        """The sample's term into its column of the row the wrapper sums
        (:data:`SAMPLE_SUM_TERMS`)."""
        column = sample_sums(self.sinks).index(sink)
        self._emit(f"{SAMPLE_SUM_TERMS}[{column}] = {self._ref(sink.value)};")

    def _commit_scattered(self, sink: Sink, target: str, index: str,
                          value: str) -> None:
        """How a SCATTERED accumulate commits.

        ``plain``/``atomic`` use the target's atomic add (``hawk_abi::accum_add``)
        — a plain ``+=`` here lost 8182 of 8192 terms per lane on the device, and
        the host team is OpenMP over tiles. ``compensated`` is Knuth two-sum
        under that same atomic add (``hawk_abi::accum_add_compensated``),
        closing the same race a prior non-atomic read-modify-write left open.

        The compensated policy applies only to the kind's resolved OUTPUT
        TARGET (:func:`hawk.ext.compensated_target`): a scattered by-key store beside,
        but not itself, that target commits ``plain`` — applying the policy to
        every scattered sink would misfire on an unrelated diagnostic plane."""
        if sink.ttype.shape:
            raise HawkError(
                f"{sink.name!r}: a scattered accumulate of a rank-"
                f"{len(sink.ttype.shape)} value is not built — a scattered commit "
                "is the target's atomic add "
                "(see hawk.ext), which is per component; declare one scalar "
                "accumulate plane per component, or commit the own column"
            )
        if self.kind.sink.id == "compensated":
            from ..ext import compensated_target

            resolved = compensated_target(self.kind, self.sinks)
            if resolved == (sink.name, False):
                comp = self.kind.sink.compensation
                if ("accum_out", comp) not in self.walk.slot_of:
                    raise HawkError(
                        f"sink policy 'compensated' names a companion accum plane "
                        f"{comp!r}, but this kernel binds no accum_out slot of that "
                        "name (bound: "
                        f"{[n for r, n in self.walk.arg_spec if r == 'accum_out']}). "
                        "The kernel declares the plane; the seam only decides the "
                        "arithmetic"
                    )
                self._emit(f"hawk_abi::accum_add_compensated({target}, "
                           f"{binding_name('accum_out', comp)}, {index}, {value});")
                return
        self._emit(f"hawk_abi::accum_add({target}, {index}, {value});")

    @staticmethod
    def _sample_index(expr: str) -> str:
        return f"aether::SampleIndex::make(static_cast<std::size_t>({expr}))"

    # -- expressions ------------------------------------

    def _expr(self, node: Node, *, materialised: bool) -> str:
        if isinstance(node, Leaf):
            return self._leaf(node, materialised)
        if isinstance(node, Const):
            return self._const(node)
        if isinstance(node, SampleIndex):
            return f"static_cast<Int>({SAMPLE_INDEX_IDENT})"
        if isinstance(node, (LoopIndex, LoopCarry)):
            # Bound by the enclosing loop's header/carried local, seeded into
            # `self.local` before the body renders.
            return self.local[id(node)]
        if isinstance(node, LoopValue):
            return self._loop_value(node)
        if isinstance(node, TapeRead):
            return self._tape_read(node)
        if isinstance(node, LoopCount):
            return self.count_names[self.loop_pos[id(node.loop)]]
        if isinstance(node, Primitive):
            # A primitive is INLINED (hawk/ext/primitive.py): the boundary node
            # is a derivative-time record, not a call, so it renders as the
            # forward subgraph it stands in front of.
            return self._ref(node.forward)
        if isinstance(node, At):
            return self._at(node)
        if isinstance(node, Select):
            return self._select(node)
        if isinstance(node, Dispatch):
            return self._dispatch(node)
        if isinstance(node, Op):
            return self._op(node)
        raise HawkError(f"no aether rendering for node {node!r}")

    def _leaf(self, leaf: Leaf, materialised: bool) -> str:
        name = binding_name(leaf.role, leaf.name)
        if leaf.role == "nsamples":
            return f"static_cast<Int>({name})"
        if leaf.role == "uniform":
            if leaf.ttype.dtype in INT_PLANE_DTYPES:
                return f"static_cast<{element_spelling(leaf.ttype.dtype)}>({name})"
            return name
        if leaf.role == "mutable":
            # `.prior` — the only way a `mutable`-role LEAF (as opposed to its
            # SINK) reaches here. The plane's local is a WRITABLE view, so `[i]`
            # calls the non-const `View::operator[]` and returns a `SampleRef`,
            # which offers only `.get()` (no bare `.eval()` — that's
            # `ConstSampleRef`'s overload, reached only through a `const` view).
            # Rank>=1 stops at the materialised `Item`; rank 0 additionally
            # calls its zero-argument `operator()()` to recover the raw scalar
            # `aether::math::fmax`/`+`/... require.
            if leaf.ttype.shape:
                return f"{name}[i].get()"
            return f"{name}[i].get()()"
        if leaf.ttype.shape and self.position.get(id(leaf)) in self.lazy:
            return f"{name}[i]"     # proxy, element reads only (LAZY_LEAF_WIDTH)
        if leaf.ttype.shape:
            # Rank >= 1: always the body's own sample, never the whole plane.
            # `view[i].get()` materialises the row into an `Item`, which is what
            # makes the value sample-bound — a bare view carries no sample, and
            # every aether fold reached through one silently answers at
            # `SampleIndex::make(0)`. `_materialised` names every rank>=1 leaf,
            # so this branch has no unmaterialised arm to take.
            return f"{name}[i].get()"
        return f"{name}[i].eval()"

    def _const(self, node: Const) -> str:
        t = node.ttype
        if t.shape:
            raise HawkError(
                f"a rank-{len(t.shape)} literal typed {t!r} has no aether spelling; "
                "a broadcast literal is a rank-0 constant under a 'splat'"
            )
        if t.dtype == "bool":
            return "true" if node.literal else "false"
        if t.dtype in ("i32", "i64"):
            return f"static_cast<Int>({int(node.literal)})"
        if t.dtype in INT_PLANE_DTYPES:
            return f"static_cast<{element_spelling(t.dtype)}>({int(node.literal)}LL)"
        return f"static_cast<Real>({_real_literal(float(node.literal))})"

    def _at(self, node: At) -> str:
        if is_param_cell(node):
            return param_cell_binding(node.plane.name)
        plane = binding_name(node.plane.role, node.plane.name)
        idx = self._sample_index(self._ref(node.index))
        return f"{plane}[{idx}]" + (".get()" if node.ttype.shape else ".eval()")

    def _select(self, node: Select) -> str:
        """The ``cond ? a : b``. A rank-0 select is a plain C++ ternary; a
        rank>=1 one is ``aether::select``, which reads both branches through
        the shared working-type helper — a raw ternary can't name them, since
        two aether expressions of different C++ types have no common type."""
        c, a, b = (self._ref(o) for o in node.operands)
        if not node.ttype.shape:
            return f"({c} ? {a} : {b})"
        cond_rank = len(node.operands[0].ttype.shape)
        if cond_rank == 0:
            # A per-sample mask: aether::select needs an EXPRESSION, and a
            # rank-0 C++ bool isn't one, so it broadcasts through a rank-0
            # Constant (Select's `Cond::element_extents::Rank == 0` arm).
            c = self._hoist(f"aether::constant<aether::extents<>>({c})")
        return f"aether::select({c}, {a}, {b})"

    def _op(self, node: Op) -> str:
        kind, rank = node.kind, len(node.ttype.shape)
        args = [self._ref(o) for o in node.operands]
        ranks = [len(o.ttype.shape) for o in node.operands]
        if kind == RAW_KIND:
            return self._raw(node, args)
        if rank == 0 and all(r == 0 for r in ranks):
            return self._scalar_op(kind, args, node)
        return self._tensor_op(kind, args, ranks, node)

    def _raw(self, node: Op, args: Sequence[str]) -> str:
        block = node.literal
        for k, arg in enumerate(args):
            self._emit(f"const auto hawk_raw_arg{k} = {arg};")
        return f"({block.text})"

    def _scalar_op(self, kind: str, args: Sequence[str], node: Op) -> str:
        if kind in _INT_CALLS or (kind in _INT_WRAPPING and node.ttype.dtype in INT_PLANE_DTYPES):
            # integer planes (and `//`, `%`, the bit operators on an index):
            # the wrapping, numpy-defined helpers (hawk/emit/backend.py
            # INT_HELPERS), since C++'s own `+ - * / % << >>` are undefined
            # or differently defined at the edges numpy pins.
            name = _INT_CALLS.get(kind) or _INT_WRAPPING[kind]
            return f"hawk_abi::{name}({', '.join(args)})"
        if kind == "cast":
            return self._cast_op(args[0], node)
        if kind in RANDOM_OPS:
            return self._random_draw(kind, args)
        if kind == "spacing":
            return f"hawk_abi::spacing({args[0]})"
        if kind == "sign":
            # aether's sign maps NaN to 0; numpy keeps the NaN.
            x = args[0] if args[0].isidentifier() else self._hoist(args[0])
            return f"(aether::math::isnan({x}) ? {x} : aether::math::sign({x}))"
        if kind in _MATH1:
            return f"aether::math::{_MATH1[kind]}({args[0]})"
        if kind in _MATH2:
            return f"aether::math::{_MATH2[kind]}({args[0]}, {args[1]})"
        if kind in _MATH3:
            return f"aether::math::{_MATH3[kind]}({', '.join(args)})"
        if kind in _INFIX and len(args) == 2:
            return f"({args[0]} {_INFIX[kind]} {args[1]})"
        if kind == "neg":
            return f"(-{args[0]})"
        if kind == "lnot":
            return f"(!{args[0]})"
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(kind, args, node)
        raise HawkError(_no_spelling(kind, 0))

    @staticmethod
    def _cast_op(arg: str, node: Op) -> str:
        """``cast``: float -> integer plane through the saturating helper,
        every other pair a plain ``static_cast`` (modular between integers,
        exact into ``Real`` up to its precision)."""
        target, source = node.ttype.dtype, node.operands[0].ttype.dtype
        elem = element_spelling(target)
        if target in INT_PLANE_DTYPES and source in ("f64", "f32"):
            return f"hawk_abi::int_from_real<{elem}>({arg})"
        return f"static_cast<{elem}>({arg})"

    def _random_draw(self, kind: str, args: Sequence[str]) -> str:
        """HAWK's ``(seed, counter)`` draw: aether's stateless
        ``detail::uniform01``/``detail::standardNormal<Real>(seed, global,
        counter)`` (:data:`_RANDOM_FN`), with ``global`` the kernel's own
        flattened sample index spelled here as the bound :data:`SAMPLE_INDEX_IDENT`
        directly. Every lane is its own stream by construction (an author names
        only ``seed``/``counter``), which is also what lets a captured graph
        re-draw by writing one of those wires rather than re-capturing. Sets
        :attr:`uses_random` so :class:`Body` carries the fact
        :func:`hawk.emit.backend.prelude` needs to add the header."""
        self.uses_random = True
        seed, counter = args
        return (f"aether::random::detail::{_RANDOM_FN[kind]}<Real>("
                f"static_cast<std::uint64_t>({seed}), "
                f"static_cast<aether::offset_t>({SAMPLE_INDEX_IDENT}), "
                f"static_cast<std::uint64_t>({counter}))")

    def _reduction(self, kind: str, args: Sequence[str], node: Op) -> str:
        """The rank>=1 -> rank-0 forms: they READ a tensor and RETURN a scalar."""
        if kind == "component":
            index = node.literal if isinstance(node.literal, tuple) else (node.literal,)
            ks = ', '.join(str(int(k)) for k in index)
            if self.position.get(id(_through(node.operands[0]))) in self.lazy:
                return f"{args[0]}.eval<{ks}>()"
            return f"{args[0]}.eval<{ks}>(i)"
        if kind == "sum" and len(node.operands[0].ttype.shape) == 2:
            return self._matrix_sum(args[0], node.operands[0].ttype)
        if kind in _METHOD1:
            return f"{args[0]}.{_METHOD1[kind]}({args[1]})"
        return f"{args[0]}.{_METHOD0[kind]}()"

    def _runtime_component(self, node: Op, args: Sequence[str]) -> str:
        """``v[k]`` at a RUNTIME index, in aether's own spelling.

        aether answers this through ``Item::operator()``, which casts each
        index to ``std::size_t`` and folds them row-major over the static
        extents — no local array copy is needed. The catch: ``operator()`` is
        ``Item``'s own, not the plain ``Expression`` CRTP's, so the operand must
        first be MATERIALISED into an ``aether::Item`` local. That
        materialisation is named once per value per SCOPE (:attr:`items`), so
        two runtime reads of one vector inside a loop body share one Item, and
        an Item built inside a loop isn't referenced after the scope closes."""
        source = node.operands[0]
        if _reads_in_place(source):
            # a read-only input plane: the element at the runtime index straight
            # from the view (`SampleRef::at`), no `Item` of the whole row
            return f"{binding_name(source.role, source.name)}[i].at({args[1]})"
        held = self.items.get(id(source))
        if held is None:
            t = source.ttype
            extents = ", ".join(str(e) for e in t.shape)
            held = f"hawk_it{self.hoists}"
            self.hoists += 1
            self._emit(f"const aether::Item<{element_spelling(t.dtype)}, "
                       f"{extents}> {held} = {args[0]};")
            self.items[id(source)] = held
        return f"{held}({args[1]})"

    def _runtime_component_store(self, node: Op, args: Sequence[str],
                                 accumulate: bool = False) -> str:
        """``v[k] = e`` at a RUNTIME index, as a VALUE.

        The IR node is functional — ``v`` with one component replaced — and C++
        has no such expression, so this is the one place the emitter turns a
        value into two statements: a non-const ``aether::Item`` copy of the
        incoming vector, then a write through its non-const ``operator()``.
        The copy preserves the IR's functional meaning (a later read of ``v``
        sees the old value); it's register-resident, so a compiler that can
        see the original elides it. Emitting statements mid-expression is safe
        because :meth:`_expr` runs once per node in traversal order, so the
        lines land above their use."""
        t = node.ttype
        name = f"hawk_sv{self.hoists}"
        self.hoists += 1
        self._emit(f"{carry_type(t)} {name} = {args[0]};")
        op = "+=" if accumulate else "="
        self._emit(f"{name}({args[1]}) {op} {self._cast(args[2], t.dtype)};")
        return name

    def _matrix_sum(self, arg: str, ttype: TensorType) -> str:
        """The FULL reduction of a rank-2 expression, in aether's own spelling.

        ``hawk.ir.ops`` admits ``sum`` over a rank-2 operand, and the reverse
        rule mints one for the adjoint of a rank-0 operand broadcast against a
        matrix. aether's ``Expression::sum()`` asserts rank 1, so this uses
        ``Expression::matDot()`` instead — the Frobenius inner product with a
        matrix of ones gives the same total, one aether node, no scalarised
        per-element chain. The ones matrix is HOISTED like any other broadcast
        constant: ``aether::constant`` is a leaf and ``matDot`` binds it by
        reference."""
        ext = ", ".join(str(e) for e in ttype.shape)
        ones = self._hoist(f"aether::constant<aether::extents<{ext}>>("
                           f"{self._cast('1', ttype.dtype)})")
        return f"{arg}.matDot({ones})"

    def _tensor_op(self, kind: str, args: Sequence[str], ranks: Sequence[int],
                   node: Op) -> str:
        rank = len(node.ttype.shape)
        if kind == "component_at":
            return self._runtime_component(node, args)
        if kind == "set_component_at":
            return self._runtime_component_store(node, args)
        if kind == "add_at":
            return self._runtime_component_store(node, args, accumulate=True)
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(kind, args, node)
        if kind == "splat":
            ext = ", ".join(str(e) for e in node.ttype.shape)
            return (f"aether::constant<aether::extents<{ext}>>("
                    f"{self._cast(args[0], node.ttype.dtype)})")
        if kind == "vec":
            if not args:
                raise HawkError("vec: a vector literal needs at least one component")
            elem = element_spelling(node.ttype.dtype)
            # aether added the 1-component `Item<T,1>{v}` ctor beside the
            # N-scalar one, so the arity floor is 1, not 2.
            ext = ", ".join(str(e) for e in node.ttype.shape)
            return f"aether::Item<{elem}, {ext}>{{{', '.join(args)}}}"
        if kind in ("add", "sub"):
            return self._additive(kind, args, ranks, node)
        if kind == "mul":
            return self._multiplicative(args, ranks, "*", "cwiseMul", node)
        if kind == "div" and ranks[1] == 0:
            return self._multiplicative(args, ranks, "/", None, node)
        if kind in _CWISE:
            # min/max/div/pow: one spelling for both operand shapes —
            # `f(expr, bound)` when the right operand is rank-0, `f(expr, expr)`
            # otherwise. Only the LEFT operand must be an expression in every
            # overload, so a rank-0 left one broadcasts through a Constant first.
            left = args[0] if ranks[0] else self._broadcast(args[0], node.ttype)
            return f"{_CWISE[kind]}({left}, {args[1]})"
        if kind == "neg":
            return f"({args[0]} * static_cast<Real>(-1))"
        if kind == "abs":
            return f"aether::cwiseAbs({args[0]})"
        if kind in _CWISE1:
            return f"aether::{_CWISE1[kind]}({args[0]})"
        if kind in ("mv", "mm"):
            return f"({args[0]} * {args[1]})"
        if kind == "outer":
            return f"aether::outer({args[0]}, {args[1]})"
        if kind in _METHOD1:
            return f"{args[0]}.{_METHOD1[kind]}({args[1]})"
        if kind in _METHOD0:
            return f"{args[0]}.{_METHOD0[kind]}()"
        raise HawkError(_no_spelling(kind, rank))

    def _additive(self, kind: str, args: Sequence[str], ranks: Sequence[int],
                  node: Op) -> str:
        sym = "+" if kind == "add" else "-"
        left, right = args
        if ranks[0] == 0:
            left = self._broadcast(left, node.ttype)
        if ranks[1] == 0:
            right = self._broadcast(right, node.ttype)
        return f"({left} {sym} {right})"

    def _multiplicative(self, args: Sequence[str], ranks: Sequence[int], sym: str,
                        cwise: str | None, node: Op) -> str:
        a, b = args
        if ranks[0] == 0 and sym == "*":
            return f"({self._cast(a, node.ttype.dtype)} * {b})"
        if ranks[1] == 0:
            return f"({a} {sym} {self._cast(b, node.ttype.dtype)})"
        if cwise is not None and ranks[0] == ranks[1]:
            return f"{a}.{cwise}({b})"
        kind = "div" if sym == "/" else "mul"
        raise HawkError(
            _no_spelling(kind, len(node.ttype.shape))
            + " for two rank>=1 operands: aether carries a component-wise PRODUCT "
            "(Expression::cwiseMul) and no component-wise quotient"
        )

    def _broadcast(self, scalar: str, ttype: TensorType) -> str:
        """A rank-0 operand added to a tensor becomes an aether broadcast leaf —
        HOISTED, since a ``Constant`` is a leaf and would be stored by reference
        by the enclosing node."""
        ext = ", ".join(str(e) for e in ttype.shape)
        return self._hoist(f"aether::constant<aether::extents<{ext}>>("
                           f"{self._cast(scalar, ttype.dtype)})")

    @staticmethod
    def _cast(text: str, dtype: str) -> str:
        elem = element_spelling(dtype)
        if text.startswith(f"static_cast<{elem}>"):
            return text
        return f"static_cast<{elem}>({text})"


#: Integer-only kinds -> their ``hawk_abi`` helper (any integer type).
_INT_CALLS = {"floordiv": "int_floordiv", "mod": "int_mod", "band": "int_and",
              "bor": "int_or", "bxor": "int_xor", "shl": "int_shl", "shr": "int_shr",
              "invert": "int_invert"}
#: Kinds an integer PLANE value routes through a wrapping helper, where the
#: infix spelling would overflow a signed int (undefined) or promote.
_INT_WRAPPING = {"add": "int_add", "sub": "int_sub", "mul": "int_mul",
                 "neg": "int_neg", "abs": "int_abs", "min": "int_min",
                 "max": "int_max"}


def _flat_body(loop: Loop) -> list:
    """``loop``'s body nodes and, recursively, those of every loop inside it
    (with their results and break values)."""
    out, stack = [], [loop]
    while stack:
        here = stack.pop()
        for node in here.body_nodes():
            out.append(node)
            if isinstance(node, Loop):
                out.extend(node.nexts)
                out.extend(() if node.exit_cond is None
                           else (node.exit_cond, *node.exit_values))
                stack.append(node)
    return out


#: The local row of per-sample terms a ``sample_sum`` body writes; the
#: backend's wrapper declares it and sums it.
SAMPLE_SUM_TERMS = "hawk_ss"


def _chain(node: Node) -> tuple:
    """``(base, [(k, e), ...])`` of an ``add_at`` chain (the runtime form of a
    wide value, :mod:`hawk.ir.slotwise`); a plain value is a chain of none."""
    links = []
    while isinstance(node, Op) and node.kind == "add_at":
        links.append((node.operands[1], node.operands[2]))
        node = node.operands[0]
    return node, links[::-1]


def _accumulate_sink(sink: Node) -> list | None:
    """The links of an own-column ``Assign`` whose value is an ``add_at`` chain
    over a structural zero — a lowered loop's per-iteration accumulate — else
    ``None``."""
    from ..ir.slotwise import is_structural_zero

    if not isinstance(sink, Assign) or sink.index is not None:
        return None
    base, links = _chain(sink.value)
    if not links or not is_structural_zero(base):
        return None
    return links


def _parts(node: Node) -> list | None:
    """The scalar parts of a ``vec`` (or of a ``splat``: its one scalar per
    slot); ``None`` for anything else."""
    if not isinstance(node, Op):
        return None
    if node.kind == "vec":
        return list(node.operands)
    if node.kind == "splat":
        return [node.operands[0]] * math.prod(node.ttype.shape)
    return None


def _reads_in_place(node: Node) -> bool:
    """Whether a runtime component read of ``node`` goes through the plane's own
    view (``name[i].at(k)``): a read-only rank-1 input leaf. A mutable plane,
    a parameter cell and every computed vector keep the ``Item`` copy."""
    return (isinstance(node, Leaf) and node.role in ("vec_in", "per_sample")
            and len(node.ttype.shape) == 1 and not is_param_cell(node))


def _flat_indices(shape: tuple) -> list:
    """Every index tuple of ``shape`` in row-major (C) order."""
    out: list = [()]
    for extent in shape:
        out = [idx + (k,) for idx in out for k in range(int(extent))]
    return out
