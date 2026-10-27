# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Aether expression-tree rendering: turns a :class:`~hawk.ir.walk.Walk`
into the kernel body as a STRING.

One renderer builds the body; a wrapper wraps it with the entry
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
    COMPANION_SUFFIX,
    GATHER_ROLES,
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
    compensates,
    is_param_cell,
    sample_sums,
)
from ..ir.linalg import CONSUMERS, FACTORISATIONS
from ..ir.ops import RANDOM_OPS
from ..ir.lanes import warp_uniform
from ..ir.slotwise import chain, is_structural_zero
from ..ir.walk import canonical_nodes
from ..types import TensorType
from . import rtstore
from .box import BoxStore
from .policy import rolled
from .writer_aether import AETHER
from .dispatch import _DispatchMixin
from .dispatch import render_dispatch_rank0 as render_dispatch_rank0
from .spelling import _flat_indices
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
    _for_header,
    _ident,
    _ordinal,
    _through,
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
    param_cell_binding as param_cell_binding,
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
    """A rendered body: the STRING both wrappers consume, plus what a wrapper needs
    to know about it.

    ``needs_random`` is set the instant the render emits a
    ``random_uniform``/``random_normal`` call, since both wrappers
    render through this one pass. ``False`` by default so a unit with
    no random op stays byte-identical to before this flag existed."""

    text: str
    needs_random: bool = False


def render_body(sinks: Sequence[Sink], walk: Walk, *, indent: str = "    ",
                kind=None, writer=None) -> Body:
    """Render one kernel's body STRING from its sink set and its walk.

    The traversal is the walk's own; every slot name comes from
    :attr:`~hawk.ir.walk.Walk.arg_spec`. ``kind`` is the
    :class:`hawk.ext.Kind` the commits render under; omitted, the default applies.
    ``writer`` is the :class:`hawk.emit.writer.Writer` that spells the text;
    omitted, the aether C++ writer."""
    return _Renderer(tuple(sinks), walk, indent, kind, writer=writer).run()


@dataclass(frozen=True)
class LaneSplit:
    """A fused-step body cut at its ONE lowered loop, for a wrapper that
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
    values they read, which a wrapper may evaluate once instead of carrying.
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
#: INTERIM (hawk 0.6.4), until an emit policy replaces it (template/runtime choice
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

    def __init__(self, sinks: tuple, walk: Walk, indent: str, kind=None, *,
                 writer=None) -> None:
        from ..ext import DEFAULT_KIND

        self.sinks = sinks
        self.walk = walk
        self.indent = indent
        self.kind = kind if kind is not None else DEFAULT_KIND
        #: The spelling seam (:mod:`hawk.emit.writer`): every target text
        #: this traversal emits comes from here.
        self.writer = writer if writer is not None else AETHER
        self.nodes, self.position = canonical_nodes(sinks)
        #: The stores at a runtime index, as runs of one mutable local
        #: (:mod:`hawk.emit.rtstore`), and the carried locals a plain loop lets
        #: such a run write in place.
        self.stores = rtstore.plan(self)
        self._uniform = None
        self.mutable_carries: set = set()
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
        #: The names of the fresh ``Item`` locals the writer declares for a value its
        #: statements fill (:meth:`~hawk.emit.writer_aether.AetherWriter.fresh_value`).
        self.fresh_items: set = set()
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
            if isinstance(node, Primitive) or self.stores.skips(node):
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
        #: The positions no statement renders, because every user reads them through their
        #: own operand (:meth:`~hawk.emit.writer_aether.AetherWriter.unrendered`).
        self.unrendered = self.writer.unrendered(self) if self.writer.rolled_forms else frozenset()
        self.lazy = self._lazy_leaves()
        #: the leaves only an ``At`` reads (by name, at its own index): no local of their own
        self.crossed = self._crossed_leaves()
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
        #: position -> the target a library result is written into directly (:meth:`_direct_outputs`)
        self.direct = self._direct_outputs() if self.writer.rolled_forms else {}
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

    def uniform(self, node: Node) -> bool:
        """Whether ``node`` is proven equal over every warp of lanes."""
        if self._uniform is None:
            self._uniform = warp_uniform(self.sinks)
        return self._uniform(node)

    def _lazy_leaves(self) -> frozenset:
        """Positions of the wide read-only leaves bound as the ``[i]`` proxy.

        INTERIM (hawk 0.6.4), until an emit policy replaces it (a template or
        runtime choice from an IR register-pressure model): a non-mutable
        rank>=1 leaf of more than :data:`LAZY_LEAF_WIDTH` elements whose every
        use is a ``component`` read at a static index is not copied whole
        (``.get()`` is one register per element); each read goes through the
        proxy. A leaf some library call takes as an operand is a proxy at any
        width, when every use of it is a proxy use (:meth:`_proxy_use`)."""
        used: dict[int, bool] = {}
        proxied: set = set()
        runtime: set = set()
        scanned = list(self.nodes)
        for node in self.nodes:
            if isinstance(node, Loop):
                scanned.extend(_flat_body(node))
                scanned.extend(node.body_sinks)
        for node in scanned:
            if isinstance(node, Leaf):
                continue
            read = isinstance(node, Op) and node.kind in ("component", "component_at")
            for k, child in enumerate(self._used(node)):
                child = _through(child)
                if isinstance(child, Leaf):
                    p = self.position[id(child)]
                    proxy = self._proxy_use(node, k)
                    used[p] = used.get(p, True) and ((read and k == 0) or proxy)
                    if proxy:
                        proxied.add(p)
                    if read and k == 0 and node.kind == "component_at":
                        runtime.add(p)
                elif isinstance(child, At) and _plane_read(child):
                    p = self.position[id(child)]
                    proxy = self._proxy_use(node, k)
                    used[p] = used.get(p, True) and proxy
                    if proxy:
                        proxied.add(p)
        out = set()
        for p, ok in used.items():
            leaf = self.nodes[p]
            if isinstance(leaf, At):
                if ok and p in proxied:
                    out.add(p)      # a plane read at another lane, every use a library operand
            elif (ok and leaf.ttype.shape and leaf.role not in ("mutable", "uniform", "nsamples")
                    and not is_param_cell(leaf)
                    and (p in proxied or math.prod(leaf.ttype.shape) > LAZY_LEAF_WIDTH
                         or (p in runtime and _reads_in_place(leaf)))):
                out.add(p)
        return frozenset(out)

    def _crossed_leaves(self) -> frozenset:
        """Positions of the read-only matrix leaves whose every use is an ``At`` of it: the
        read is spelled by name at the computed index, so the leaf's own-lane copy
        (``plane[i].get()``) would be dead."""
        only: dict[int, bool] = {}
        scanned = list(self.nodes)
        for node in self.nodes:
            if isinstance(node, Loop):
                scanned.extend(_flat_body(node))
                scanned.extend(node.body_sinks)
        for node in scanned:
            if isinstance(node, Leaf):
                continue
            for k, child in enumerate(self._used(node)):
                child = _through(child)
                if isinstance(child, Leaf) and child.role == "mat_in":
                    p = self.position[id(child)]
                    only[p] = only.get(p, True) and isinstance(node, At) and k == 0
        return frozenset(p for p, ok in only.items() if ok)

    def _direct_outputs(self) -> dict:
        """The library results an own-column ``Assign`` stores unchanged: the call writes the
        plane through its proxy, with no ``Item`` of the result to copy afterwards.

        Only when the value has no other reader, stands in the flat top-level walk, and the
        kernel never reads that plane's prior value (a read is a snapshot taken at its
        own turn in the walk, and the write would now come earlier); no loop is in the kernel."""
        if any(isinstance(n, Loop) for n in self.nodes):
            return {}
        read = {n.name for n in self.nodes if isinstance(n, Leaf) and n.role == "mutable"}
        out: dict = {}
        for sink in self.sinks:
            if type(sink) is not Assign or sink.index is not None or sink.name in read:
                continue
            node = _through(sink.value)
            pos = self.position.get(id(node))
            if (isinstance(node, Op) and self.writer.direct_kind(node) and pos in self._outside
                    and pos not in self._fused and self.fanout.get(pos, 0) == 1
                    and not self._is_compensated_own_column(sink)
                    and sink.name not in _companions(self.sinks)
                    and self.writer.own_shape(node) == sink.value.ttype.shape):
                out[pos] = f"{binding_name(sink.role, sink.name)}[i]"
        return out

    def _proxy_read(self, node: Node) -> bool:
        """A static component read of a leaf bound as a proxy: a load the use re-issues
        (a local per such read would keep one register each live across the kernel)."""
        return (self.writer.rolled_forms and isinstance(node, Op) and node.kind == "component"
                and self.position.get(id(_through(node.operands[0]))) in self.lazy)

    @staticmethod
    def _used(node: Node) -> Sequence[Node]:
        """The operands of ``node`` it reads itself: a loop reads its initial values
        (a carried local is a copy); what its body reads from outside is found in the body."""
        return node.inits if isinstance(node, Loop) else node.operands

    def _proxy_use(self, node: Node, k: int) -> bool:
        """Whether ``node`` reads its operand ``k`` as a per-sample proxy: a library
        call's operand, or the operand of a value no statement renders."""
        if not self.writer.rolled_forms:
            return False
        if self.position.get(id(node)) in self.unrendered:
            return True
        return isinstance(node, Op) and (
            (node.kind in FACTORISATIONS and k == 0) or (node.kind in CONSUMERS and k == 1) or rolled(node))

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
            self._emit(self.writer.component_commit(
                self, target, at, "=", self.writer.typed_zero(_F64_SCALAR)))
        for target, width in self._accumulated_planes():
            # a plane a lowered loop accumulates at runtime indices once per
            # iteration (`bar[i].at(k) += e`, constraint 7): zeroed whole, once,
            # before the walk; the loop's links are the only other writes
            self._emit(self.writer.zero_row(self, target, width))
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
        if not self.writer.commits_per_component:
            return
        read = {(n.role, n.name) for n in self.nodes if isinstance(n, Leaf)}
        ready = self._ready_times()
        compensated = self.kind.sink.id == "compensated"
        users: dict = {}
        planned = []
        for sink in self.sinks:
            value = getattr(sink, "value", None)
            base, links = chain(value)
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

    def _ready_at(self, part: Node, ready: list, late: int) -> int:
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
                self._emit(self.writer.in_place_accumulate(
                    self, target, self._ref(at), self._ref(part)))
            else:
                self._emit(self.writer.component_commit(
                    self, target, at, op, self._ref(part)))

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
            self._emit(self.writer.declare(name, node.ttype, text))
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
        own commits land before the sample is marked (the writer spells it)."""
        self.writer.finish(self, sink)

    def _guard_slot(self) -> str | None:
        """The guard's CONDITION against the walk's own slots: each mask the
        kind's guard names counts only if bound in the ``terminated`` role, and
        the commit is skipped when any bound one is set. ``Guard(mask=None)``
        never fires, nor does a guard none of whose masks the kernel binds."""
        bound = [binding_name("terminated", m) for m in self.kind.guard.names
                 if ("terminated", m) in self.walk.slot_of]
        return self.writer.guard(bound)

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
        if isinstance(node, Leaf) and node.role in GATHER_ROLES:
            # read BY NAME at a computed index (`_at`), never through a
            # reference to the leaf: naming it would emit a dead (and possibly
            # out-of-bounds) `plane[i].eval()` at the sample index.
            return False
        if is_param_cell(node):
            return False        # a bare identifier, loaded once before the loop
        if pos in self.crossed:
            return False        # only an `At` reads it, by name: a copy would be dead
        if isinstance(node, (SampleIndex, LoopIndex, LoopCarry)):
            return False        # already a bare identifier the enclosing scope bound
        if isinstance(node, (Loop, Const, Primitive)):
            return False        # a statement / a literal / a boundary: none is named
        if isinstance(node, Op) and node.kind in FACTORISATIONS:
            return not self._unnamed(node)      # a struct the solves and field reads share: one named local
        if self.writer.names_every_tensor and node.ttype.shape:
            return True         # a C struct value: always its own statement
        if self._unnamed(node):
            return False        # a store a run absorbs, or its key; a value the writer fills in place: no local
        if pos in self.forced:
            return True
        return self.fanout.get(pos, 0) >= 2 or self._leaf_valued(node)

    def _unnamed(self, node: Node) -> bool:
        """A node that never gets a local of its own: one a store run absorbs
        (:mod:`hawk.emit.rtstore`), one no statement renders, and one the writer
        renders as the name of a fresh ``Item`` already."""
        return (self.stores.unnamed(node) or self._proxy_read(node)
                or self.position.get(id(node)) in self.unrendered
                or (self.writer.rolled_forms and self.writer.fresh_value(node)))

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

    def _hoist(self, text: str, ttype=None) -> str:
        """Bind ``text`` to a fresh named local ahead of the current line (the
        writer's :meth:`~hawk.emit.writer.Writer.hoist`)."""
        return self.writer.hoist(self, text, ttype)

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
        elems = [self.writer.carry_type(c.ttype) for c in loop.carries]
        names = tuple(f"hawk_v{pos}_{j}_{_ident(c.name)}"
                      for j, c in enumerate(loop.carries))
        self.carry_names[pos] = names
        for name, elem, init in zip(names, elems, loop.inits):
            self._emit(f"{elem} {name} = {self._ref(init)};")
        tapes = self.tape_names.get(pos, {})
        for slot in sorted(tapes):
            self._emit(self.writer.tape_decl(elems[slot], tapes[slot], loop.trip))
        counter = self.count_names.get(pos)
        if counter is not None:
            self._emit(self.writer.counter_decl(counter))
        index = f"hawk_k{pos}_{_ident(loop.index.name)}"
        count = None if loop.count is None else self._ref(loop.count)
        self._emit(self.writer.for_header(self, index, loop, count))
        scope = self._push_scope()
        self._scope_depth += 1
        self.local[id(loop.index)] = index
        for carry, name in zip(loop.carries, names):
            self.local[id(carry)] = name
            if loop.exit_cond is None:
                self.mutable_carries.add(id(carry))
        for slot in sorted(tapes):
            self._emit(f"{tapes[slot]}[{_ordinal(index, loop)}] = {names[slot]};")
        form = loop.exit_form
        if form in ("head", "middle"):
            pre = self._exit_part(loop)
            self._render_scope(loop, only=pre, sinks=False)
            self._emit(self.writer.exit_if(self._ref(loop.exit_cond)))
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
            self._emit(self.writer.tail_break(f"hawk_x{pos}"))
        self._pop_scope(scope)
        self._scope_depth -= 1
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
        values read through their usual names, which the wrapper binds."""
        pos = self.loop_pos[id(loop)]
        names = tuple(f"hawk_v{pos}_{j}_{_ident(c.name)}"
                      for j, c in enumerate(loop.carries))
        self.carry_names[pos] = names
        inits = [self._ref(init) for init in loop.inits]
        index = f"hawk_k{pos}_{_ident(loop.index.name)}"
        count = None if loop.count is None else self._ref(loop.count)
        mark = len(self.lines)
        lines, self.lines = self.lines, []
        scope = self._push_scope()
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
        self._pop_scope(scope)
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
        scope = self._push_scope()
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
        self._pop_scope(scope)
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
        by the boundary rule before the loop opened. A node only a ``switch``
        dispatch's branches read is left to its case (:func:`_scope_live`)."""
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
            if isinstance(node, Primitive) or self.stores.skips(node):
                continue                  # not a reference, not a referent
            for child in node.operands:
                child = _through(child)
                fanout[id(child)] = fanout.get(id(child), 0) + 1
        live = _scope_live(loop)
        for node in body:
            if id(node) in skip or (only is not None and id(node) not in only):
                continue
            if id(node) not in live:
                continue                  # its switch case renders it
            if isinstance(node, Loop):
                self._loop(node)
                self.local[id(node)] = ""
                continue
            if isinstance(node, Primitive):
                self.local[id(node)] = self._ref(node.forward)
                continue
            named = ((fanout.get(id(node), 0) >= 2 or self._leaf_valued(node))
                     and not self._unnamed(node))
            self._render_one(node, named)
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
        if sink.name in _companions(self.sinks):
            return      # a ``policy="compensated"`` add's correction plane: same rule
        target = binding_name(sink.role, sink.name)
        if isinstance(sink, TapeWrite):
            self.writer.tape_write(self, sink, target)
            return
        if isinstance(sink, SampleSum):
            self._sample_sum(sink)
            return
        if isinstance(sink, BoxStore):
            self.writer.box_store(self, target, self._ref(sink.index), self._ref(sink.value))
            return
        links = _accumulate_sink(sink) if self._scope_depth else None
        if links is not None:
            # a per-iteration runtime accumulate (constraint 7): each link on
            # the own column where it stands, nothing for the zero base
            for k, e in links:
                self._emit(self.writer.in_place_accumulate(
                    self, target, self._ref(k), self._ref(e)))
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
                comp = binding_name("mutable", self.kind.sink.compensation)
                self.writer.store_compensated(self, sink, target, comp, value)
                return
            self.writer.store(self, sink, target, value)
            return
        index = self._ref(sink.index)
        if not isinstance(sink, AccumWrite):
            # A scattered wide_write targets a computed ROW: a store, exactly as
            # the reference evaluator reads it.
            self.writer.scatter_store(self, sink, target, index, value)
            return
        self._commit_scattered(sink, target, index, value)


    def _sample_sum(self, sink: SampleSum) -> None:
        """The sample's term into its column of the row the wrapper sums
        (:data:`SAMPLE_SUM_TERMS`)."""
        column = sample_sums(self.sinks).index(sink)
        self.writer.sample_sum(self, sink, column)

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
        every scattered sink would misfire on an unrelated diagnostic plane.

        A key the lane facts show moving in runs (:meth:`_key_run`) commits
        once per run through ``aether::accum::run``, the compensated twin for
        the compensated target. The writer spells each commit; this method
        decides which."""
        if sink.ttype.shape:
            raise HawkError(
                f"{sink.name!r}: a scattered accumulate of a rank-"
                f"{len(sink.ttype.shape)} value is not built — a scattered commit "
                "is the target's atomic add "
                "(see hawk.ext), which is per component; declare one scalar "
                "accumulate plane per component, or commit the own column"
            )
        if getattr(sink, "policy", "auto") == "or":
            self.writer.or_scattered(self, sink, target, index, value)
            return
        comp = (binding_name("accum_out", sink.name + COMPANION_SUFFIX)
                if compensates(getattr(sink, "policy", "auto")) else None)
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
                comp = binding_name("accum_out", comp)
        run = self._key_run(sink)
        if run is not None:
            self.writer.accum_run(self, sink, target, comp, index, value, run)
        elif comp is not None:
            self.writer.store_compensated(self, sink, target, comp, value)
        else:
            self.writer.accum_scattered(self, sink, target, index, value)

    def _key_run(self, sink: Sink):
        """The run its key moves in (:func:`hawk.ir.lanes.key_run`: proven,
        tested or sorted), or ``None`` for the per-lane commit. Asked only of
        a writer that spells the run commit, so every other writer's text is
        unchanged."""
        if not hasattr(self.writer, "accum_run"):
            return None
        from ..ir.lanes import key_run

        return key_run(self.sinks, sink)


    # -- expressions (spelled by the writer) -----------------------------

    def _expr(self, node: Node, *, materialised: bool) -> str:
        if isinstance(node, Leaf):
            return self.writer.leaf(self, node, materialised)
        if isinstance(node, Const):
            return self.writer.const(node)
        if isinstance(node, SampleIndex):
            return self.writer.sample_index_value()
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
            return self.writer.at(self, node)
        if isinstance(node, Select):
            c, a, b = (self._ref(o) for o in node.operands)
            return self.writer.select(self, node, c, a, b)
        if isinstance(node, Dispatch):
            return self._dispatch(node)
        if isinstance(node, Op):
            if self.position.get(id(node)) in self.unrendered:
                return ""
            args = [self._ref(o) for o in node.operands]
            ranks = [len(o.ttype.shape) for o in node.operands]
            return self.writer.op(self, node, args, ranks)
        raise HawkError(f"no aether rendering for node {node!r}")


def _companions(sinks) -> frozenset:
    """The correction planes of the ``policy="compensated"`` accumulates."""
    return frozenset(s.name + COMPANION_SUFFIX for s in sinks
                     if isinstance(s, AccumWrite) and compensates(s.policy))


def _scope_live(loop: Loop) -> set:
    """The ids of the body nodes some root of ``loop`` reaches WITHOUT following a
    ``switch`` dispatch's ``branches``: the loop-body twin of
    :meth:`_Renderer._outside_reachable`. What only a branch reads is left to
    that branch's case."""
    seen: set = set()
    stack: list = list(loop.body_roots())
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
            stack.append(n.selector)
        else:
            stack.extend(n.operands)
    return seen


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


#: A rank-0 float64 type: the zero :meth:`_Renderer.run` writes into planes it
#: zero-fills is the writer's typed zero of it.
_F64_SCALAR = TensorType((), "f64")


#: The local row of per-sample terms a ``sample_sum`` body writes; the
#: wrapper declares it and sums it.
SAMPLE_SUM_TERMS = "hawk_ss"


def _accumulate_sink(sink: Node) -> list | None:
    """The links of an own-column ``Assign`` whose value is an ``add_at`` chain
    over a structural zero — a lowered loop's per-iteration accumulate — else
    ``None``."""
    if not isinstance(sink, Assign) or sink.index is not None:
        return None
    base, links = chain(sink.value)
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


def _plane_read(node: At) -> bool:
    """Whether ``node`` is a rank>=1 value of a read-only input plane read at another lane,
    which a proxy ``plane[index]`` can serve element by element."""
    return bool(node.ttype.shape) and node.plane.role not in GATHER_ROLES and not is_param_cell(node)
