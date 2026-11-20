# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recording launches into a graph and replaying it: :class:`GraphPipeline`, :class:`Graph`, the control-flow forms
:func:`skippable` and :func:`repeat_while` over a :class:`SkipGuard`, and :func:`reverse`.

A graph is a tree of plan launches, conditionals and forks recorded once by the host library and replayed as one. What the
library records is what it can replay: launches of plans bound with :meth:`hawk.plans.Plan.bind`, a conditional that tests one
element of an integer cell, and a fork whose branches it orders by the planes they touch. Python is recorded only as a
:func:`hawk.host_step`, which the library calls where a launch would run; any other function is refused by name. The planes are the
caller's memory and a replay sees what they hold at that time.

A graph is written two ways. By hand, ``g.add(bound)`` and ``with g.parallel() as p`` (the exit of the ``with`` is the join; no
edge is ever declared, and two branches that touch one plane with a writer among them are refused by the library, ``wait`` being
the way out). Or by :meth:`Graph.builder`, which runs a function once and records the kernel calls it makes
(:mod:`hawk.graph_builder`).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple

import _osprey

from . import _place, graph_builder
from .errors import HawkError
from .runner import GradientReport, Gradients, RunRecord, check_reverse_options
from .host_steps import BoundHostStep
from .runtime import BoundPlan, PlaneStore, through_door

__all__ = ["Edge", "Graph", "GraphPipeline", "GraphRecord", "RepeatWhile", "SkipGuard", "Skippable", "repeat_while", "reverse",
           "skippable"]

#: The conditional kinds of the library's door.
_IF, _IF_ELSE, _WHILE, _SWITCH = _osprey.EG_COND_IF, _osprey.EG_COND_IF_ELSE, _osprey.EG_COND_WHILE, _osprey.EG_COND_SWITCH


# -- the recorded tree ------------------------------------------------------------------------------------------------------ #
@dataclass
class _Launch:
    """One launch of ``bound``; ``name`` is the label the caller gave it, else the kernel's."""

    bound: BoundPlan
    name: str | None = None


@dataclass
class _HostCall:
    """One call of a host step, which the library runs where a launch would run."""

    bound: BoundHostStep

    @property
    def name(self) -> str:
        """The step's function name, which is also the library's name for the node."""
        return self.bound.step.name


@dataclass
class _Cond:
    """A conditional over one element of ``cell`` (an array) compared with ``baseline[baseline_index]`` when there is one;
    ``arms`` are lists of items: one for IF and WHILE, two for IF_ELSE, one per case for SWITCH."""

    kind: int
    cell: object
    arms: list
    negate: bool = False
    max_iters: int = 0
    index: int = 0
    baseline: object = None
    baseline_index: int = 0
    #: of a WHILE: the one-cell arrays the library counts the iterations in, stops at and is told to stop by
    counter: object = None
    limit: object = None
    brk: object = None


@dataclass
class _Fork:
    """The branches of a fork, each a list of items; the end of the fork joins them."""

    branches: list


@dataclass
class _Wait:
    """In a branch: what follows waits for the earlier sibling branches that write ``arrays``."""

    arrays: tuple = field(default_factory=tuple)


#: The words of the library's node and edge kinds, by their codes.
_NODE_WORDS = MappingProxyType({
    _osprey.EG_GNODE_LAUNCH: "launch", _osprey.EG_GNODE_RUN: "run", _osprey.EG_GNODE_REDUCE: "reduce",
    _osprey.EG_GNODE_COPY: "copy", _osprey.EG_GNODE_FFT: "fft", _osprey.EG_GNODE_GEMM: "gemm", _osprey.EG_GNODE_COND: "cond",
    _osprey.EG_GNODE_FORK: "fork", _osprey.EG_GNODE_BRANCH: "branch", _osprey.EG_GNODE_JOIN: "join",
    _osprey.EG_GNODE_WAIT: "wait"})
_EDGE_WORDS = MappingProxyType({
    _osprey.EG_GEDGE_RAW: "RAW", _osprey.EG_GEDGE_WAR: "WAR", _osprey.EG_GEDGE_WAW: "WAW", _osprey.EG_GEDGE_WAIT: "WAIT",
    _osprey.EG_GEDGE_JOIN: "JOIN", _osprey.EG_GEDGE_ORDER: "ORDER"})
_COND_WORDS = MappingProxyType({_IF: "if", _IF_ELSE: "if / else", _WHILE: "while", _SWITCH: "switch"})
#: The Mermaid arrow and the DOT style of each edge kind: hazards solid or dashed, a wait bold, a recording-order edge dotted.
_MERMAID_ARROWS = MappingProxyType({"RAW": "-->", "WAR": "-.->", "WAW": "-.->", "WAIT": "==>", "JOIN": "-->", "ORDER": "-.->"})
_DOT_STYLES = MappingProxyType({"RAW": "solid", "WAR": "dashed", "WAW": "dashed", "WAIT": "bold", "JOIN": "solid",
                                "ORDER": "dotted"})
_ACCESS_WORDS = MappingProxyType({1: "r", 2: "w", 3: "rw"})


class Edge(NamedTuple):
    """One edge of a described graph: ``src`` must finish before ``dst`` starts, both indices into
    :meth:`GraphPipeline.node_labels`. ``kind`` is ``"RAW"``, ``"WAR"`` or ``"WAW"`` (a hazard through ``plane``, by name),
    ``"WAIT"`` (a writer to a ``wait``), ``"JOIN"`` (a branch to its join) or ``"ORDER"`` (recording order the replay keeps);
    ``plane`` is ``None`` for the last two."""

    src: int
    dst: int
    plane: str | None
    kind: str


# -- guards and the forms over them ----------------------------------------------------------------------------------------- #
class SkipGuard:
    """What a conditional tests: ``count_arr[count_idx] != baseline_arr[baseline_idx]``, or ``!= 0`` with no baseline, read at
    every replay. The arrays are ``int32`` or ``uint32`` and the caller's, so writing a cell between replays changes the
    decision with no re-record.
    """

    def __init__(self, count_arr, count_idx: int = 0, baseline_arr=None, baseline_idx: int = 0, negate: bool = False) -> None:
        """Hold the cell, the index of the element tested and, optionally, the baseline array and index; ``negate`` tests for
        equality instead."""
        self.count_arr, self.count_idx = count_arr, int(count_idx)
        self.baseline_arr, self.baseline_idx = baseline_arr, int(baseline_idx)
        self.negate = bool(negate)

    @classmethod
    def nonzero(cls, n_active) -> SkipGuard:
        """The guard ``n_active[0] != 0``."""
        return cls(n_active, 0)

    @classmethod
    def intent(cls, flags, index: int) -> SkipGuard:
        """The guard ``flags[index] != 0``: an on / off flag the caller owns and writes between replays, one element per
        guarded region."""
        return cls(flags, index)

    def negated(self) -> SkipGuard:
        """The guard that holds exactly when this one does not."""
        return SkipGuard(self.count_arr, self.count_idx, self.baseline_arr, self.baseline_idx, not self.negate)

    def evaluate(self) -> bool:
        """Read the guard now, on the host: for a program that checks what a replay will decide."""
        baseline = 0 if self.baseline_arr is None else int(self.baseline_arr[self.baseline_idx])
        return (int(self.count_arr[self.count_idx]) != baseline) != self.negate


class Skippable:
    """A step that runs only while its guard holds: what :func:`skippable` returns, for :meth:`Graph.add`."""

    def __init__(self, step, guard: SkipGuard) -> None:
        """Hold the step and its guard."""
        self.step, self.guard = step, guard


class RepeatWhile:
    """A step repeated while its guard holds, at most ``max_iters`` times per replay: what :func:`repeat_while` returns.

    ``limit`` and ``brk`` are one-cell ``uint32`` arrays of the caller: the loop runs at most ``limit[0]`` times, read at every
    replay, and ends after the iteration that leaves ``brk[0]`` nonzero. The library counts the iterations in
    :attr:`iteration_index`.
    """

    def __init__(self, step, guard: SkipGuard | None, max_iters: int | None, limit=None, brk=None) -> None:
        """Hold the step, what stops it (a guard, a limit or a break cell: at least one) and the cap (``None``: no cap)."""
        if max_iters is not None and (isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1):
            raise HawkError(f"repeat_while: max_iters is a whole number of 1 or more, or None, got {max_iters!r}")
        if guard is None and limit is None and brk is None:
            raise HawkError("repeat_while: nothing stops the loop; pass a guard, limit= or brk=")
        self.step, self.guard, self.max_iters = step, guard, max_iters
        self.limit, self.brk = limit, brk
        self._counter = None

    @property
    def iteration_index(self):
        """The one-cell ``uint32`` array the library counts in, in the memory of the loop's own cells: the index of the
        iteration while the body runs (0 in the first), which a kernel of the body may read, and the count once the loop
        ends."""
        if self._counter is None:
            like = next(cell for cell in (getattr(self.guard, "count_arr", None), self.limit, self.brk) if cell is not None)
            self._counter = _place.Maker.at(_place.device_of(like), like).zeros((1,), "uint32")
        return self._counter

    def iterations(self) -> int:
        """The iterations the last replay ran; 0 before any. On a device the read waits for the replay."""
        return int(self.iteration_index[0])


def skippable(step, guard: SkipGuard) -> Skippable:
    """``step`` wrapped to run only while ``guard`` holds: one IF in the recorded graph. ``step`` is a bound plan, a list of
    steps run in order, a nested :func:`skippable` or :func:`repeat_while`, or, inside a builder, a function with no arguments
    whose kernel calls are recorded."""
    return Skippable(step, _guard(guard, "skippable"))


def repeat_while(step, guard: SkipGuard | None = None, max_iters: int | None = None, *, limit=None, brk=None) -> RepeatWhile:
    """``step`` (as in :func:`skippable`) repeated while ``guard`` holds, at most ``max_iters`` times per replay: one WHILE in the
    recorded graph. The cap is fixed at record; ``None`` repeats until the guard fails, and a graph kept for :func:`reverse`
    needs it. ``limit`` (a one-cell ``uint32`` array) is a trip count read at every replay; ``brk`` (a one-cell ``uint32``
    array the body writes) ends the loop after the iteration that leaves it nonzero; with either, ``guard`` may be ``None``."""
    return RepeatWhile(step, None if guard is None else _guard(guard, "repeat_while"), max_iters, limit, brk)


def _guard(guard, verb: str) -> SkipGuard:
    """``guard`` when it is a :class:`SkipGuard`, else a refusal by name."""
    if not isinstance(guard, SkipGuard):
        raise HawkError(f"{verb}: the guard is a hawk.SkipGuard, got {type(guard).__name__}")
    return guard


def _cond_of(guard: SkipGuard, kind: int, arms: list, max_iters: int = 0) -> _Cond:
    """The conditional item of ``guard`` over ``arms``."""
    return _Cond(kind, guard.count_arr, arms, guard.negate, max_iters, guard.count_idx, guard.baseline_arr, guard.baseline_idx)


# -- graphs ---------------------------------------------------------------------------------------------------------------- #
def _release(door, handles: list) -> None:
    """Free the graph handles in ``handles``."""
    for handle in handles:
        door.graph_free(handle)
    handles.clear()


class GraphPipeline:
    """A sequence of plan launches recorded once and replayed: ``add`` each :class:`~hawk.runtime.BoundPlan`, ``build``,
    then ``launch`` as often as needed.

    The recorded launches run at every replay in the order they were added; the library decides how a replay is carried
    out. A counter array a plan is bound with receives what each replay counted.
    """

    def __init__(self) -> None:
        """Make an empty pipeline."""
        self._items: list = []
        self._scopes: list = [self._items]
        #: the door and the plane store of the graph: those of the memory its first launch's planes live in
        self._door = None
        self._store: PlaneStore | None = None
        self._launches: list[_Launch] = []
        #: every launch and host step, in recording order
        self._recorded: list = []
        self._counting: list[BoundPlan] = []
        self._params: dict | None = None
        self._handles: list = []
        self._finalizer = None

    def _share_planes(self, other: "GraphPipeline") -> None:
        """Bind this graph's planes through those of ``other``, which has launches: the two then name one plane per region of
        memory, as the graphs of one stepping run must."""
        self._door, self._store = other._door, other._store

    def _store_over(self, door) -> PlaneStore:
        """The plane store of this graph over ``door``, which the first launch makes the graph's: a graph runs on one door, so a
        launch whose planes are in another memory is refused by name."""
        if self._door is None:
            self._door, self._store = door, PlaneStore(door)
        elif door is not self._door:
            raise HawkError(f"{type(self).__name__}: a graph runs on one door, and this launch's planes are in another memory "
                            "than the graph's; a graph over host and device memory is refused: the device library offers no "
                            "host-entry unit")
        return self._store

    def _check_open(self, verb: str) -> None:
        """Refuse ``verb`` once the pipeline is built: what is recorded cannot grow."""
        if self._finalizer is not None:
            raise HawkError(f"{type(self).__name__}.{verb}: the graph is built; make another to record more")

    def _append(self, item) -> None:
        """Put ``item`` at the end of the scope being recorded."""
        self._scopes[-1].append(item)

    def add(self, step) -> GraphPipeline:
        """Append the launch of ``step``, a :class:`~hawk.runtime.BoundPlan`, and return the pipeline.

        A function is refused: the library records launches, not Python, so a function would run once at build time and
        never again at a replay. Write the step as a kernel and add its bound plan; a :class:`hawk.Graph` also takes a bound
        :func:`hawk.host_step`.
        """
        self._check_open("add")
        if not isinstance(step, BoundPlan):
            raise HawkError(f"{type(self).__name__}.add takes a bound plan (hawk.plan(...).bind(...)), got "
                            f"{type(step).__name__}: a graph records launches and never Python; write the step as a kernel")
        step.join(self._store_over(step.door))
        self._append(_Launch(step))
        return self

    def _emit(self, items: list, under_condition: bool = False) -> None:
        """Record ``items`` into the open recording, in order; a launch under a conditional does not settle counters."""
        door = self._door
        for item in items:
            if isinstance(item, _Launch):
                item.bound.enqueue()
                self._launches.append(item)
                self._recorded.append(item)
                if not under_condition:
                    self._counting.append(item.bound)
            elif isinstance(item, _HostCall):
                item.bound.record(door, self._store)
                self._recorded.append(item)
            elif isinstance(item, _Cond):
                self._emit_cond(item)
            elif isinstance(item, _Fork):
                through_door(door.record_fork_begin)
                for position, branch in enumerate(item.branches):
                    if position:
                        through_door(door.record_fork_branch)
                    self._emit(branch, under_condition)
                through_door(door.record_fork_end)
            else:
                through_door(door.record_wait, [self._plane_of(array) for array in item.arrays])

    def _while_of(self, loop: RepeatWhile) -> _Cond:
        """The conditional item of ``loop``: its guard when it has one, its body, and the cells the library counts in, stops at
        and is told to stop by."""
        body, cap = [self._items_of(loop.step)], loop.max_iters or 0
        item = _Cond(_WHILE, None, body, max_iters=cap) if loop.guard is None else _cond_of(loop.guard, _WHILE, body, cap)
        item.counter, item.limit, item.brk = loop.iteration_index, loop.limit, loop.brk
        return item

    def _emit_cond(self, item: _Cond) -> None:
        """Record the conditional ``item`` and its arms."""
        door = self._door
        plane = {name: None if array is None else self._plane_of(array) for name, array in
                 (("cell", item.cell), ("baseline", item.baseline), ("counter", item.counter), ("limit", item.limit),
                  ("brk", item.brk))}
        through_door(door.record_cond_begin, item.kind, plane["cell"], n_arms=len(item.arms) if item.kind == _SWITCH else 0,
                     negate=item.negate, max_iters=item.max_iters, cell_index=item.index, baseline=plane["baseline"],
                     baseline_index=item.baseline_index, counter=plane["counter"], limit=plane["limit"], brk=plane["brk"])
        for position, arm in enumerate(item.arms):
            if position:
                through_door(door.record_cond_arm)
            self._emit(arm, True)
        through_door(door.record_cond_end)

    def _plane_of(self, array):
        """The plane the graph holds over the caller's ``array``: the one a launch bound it with, else one made for it."""
        for launch in self._leaves_everywhere():
            for name, given in launch.bound.planes.items():
                if given is array and name in launch.bound.views:
                    return self._store.plane(launch.bound.views[name])
        return self._store.plane(array)

    def _leaves_everywhere(self, items: list | None = None):
        """Every launch and host step of the tree, in recording order."""
        for item in self._items if items is None else items:
            if isinstance(item, (_Launch, _HostCall)):
                yield item
            elif isinstance(item, _Cond):
                for arm in item.arms:
                    yield from self._leaves_everywhere(arm)
            elif isinstance(item, _Fork):
                for branch in item.branches:
                    yield from self._leaves_everywhere(branch)

    def _described(self) -> tuple:
        """``(description, names)`` of the built graph as the library gives it: its nodes and edges with planes as indices into
        the store's planes, and a name for each plane (the first launch's name for it, else ``cell<i>``)."""
        if self._finalizer is None or not self._finalizer.alive:
            raise HawkError(f"{type(self).__name__}: the graph is not built, or is closed; the library describes a recording, "
                            "so build it first")
        entries = self._store.entries()
        description = through_door(self._door.graph_describe, self._handles[0], [plane for plane, _view in entries])
        first = {}
        for launch in self._recorded:
            for name, view in launch.bound.views.items():
                first.setdefault(PlaneStore.region(view), name)
        for name, array in (self._params or {}).items():
            if hasattr(array, "shape"):
                first.setdefault(PlaneStore.region(array), name)         # a builder parameter no launch binds: a condition cell
        names = [first.get(PlaneStore.region(view), f"cell{i}") for i, (_plane, view) in enumerate(entries)]
        return description, names

    def node_labels(self) -> list:
        """The label of every node of the built graph, in recording order: a launch by its name, the rest by what it is (``fork``,
        ``branch``, ``join``, ``wait y``, ``if flag[0]``, ``while c[0] (at most 4)``, ``else``). The indices of :meth:`edges` and
        :meth:`schedule` are positions in this list, the ordinals the library names in a refusal."""
        description, names = self._described()
        labels = self._labels()
        nodes = description["nodes"]
        return [_node_label(node, names, labels, nodes) for node in nodes]

    def num_nodes(self) -> int:
        """How many nodes the built graph has: launches, conditionals and their arms, forks, branches, joins and waits."""
        return len(self._described()[0]["nodes"])

    def edges(self) -> tuple:
        """The :class:`Edge` of the built graph, in the library's order: every hazard, wait, join and ordering it derived from the
        planes the nodes touch. hawk derives none of them."""
        description, names = self._described()
        return tuple(Edge(edge["from"], edge["to"], None if edge["plane"] in (None, -1) else names[edge["plane"]],
                          _EDGE_WORDS[edge["kind"]]) for edge in description["edges"])

    def schedule(self) -> tuple:
        """The levels of the built graph: a tuple with, for each level, the indices of the nodes that may run at the same time. A
        host replay runs them in recording order, which respects every level."""
        nodes = self._described()[0]["nodes"]
        return tuple(tuple(i for i, node in enumerate(nodes) if node["level"] == level)
                     for level in range(1 + max((node["level"] for node in nodes), default=-1)))

    def to_mermaid(self) -> str:
        """The built graph as Mermaid ``flowchart TD`` text: a node per recorded node, an arrow per edge, drawn by kind."""
        labels = self.node_labels()
        lines = ["flowchart TD"] + [f'    n{i}["{i} {_quoted(label)}"]' for i, label in enumerate(labels)]
        for edge in self.edges():
            note = " ".join(word for word in (edge.plane, edge.kind if edge.kind not in ("RAW", "JOIN", "ORDER") else None) if word)
            arrow = _MERMAID_ARROWS[edge.kind] + (f"|{note}|" if note else "")
            lines.append(f"    n{edge.src} {arrow} n{edge.dst}")
        return "\n".join(lines) + "\n"

    def to_dot(self) -> str:
        """The built graph as Graphviz DOT text: a node per recorded node, an edge per edge, styled by kind."""
        labels = self.node_labels()
        kinds = [node["kind"] for node in self._described()[0]["nodes"]]
        lines = ["digraph hawk_graph {", "    rankdir=TB;"]
        for i, (label, kind) in enumerate(zip(labels, kinds)):
            shape = {_osprey.EG_GNODE_COND: "diamond", _osprey.EG_GNODE_LAUNCH: "box"}.get(kind, "ellipse")
            lines.append(f'    n{i} [label="{i} {_quoted(label)}", shape={shape}];')
        for edge in self.edges():
            note = f', label="{edge.plane or ""}"' if edge.plane else ""
            lines.append(f"    n{edge.src} -> n{edge.dst} [style={_DOT_STYLES[edge.kind]}{note}];")
        return "\n".join(lines + ["}"]) + "\n"

    def __str__(self) -> str:
        """The built graph as text: its nodes in recording order, indented by what encloses them, with their level and, for a
        launch, the planes it reads and writes; then its edges."""
        description, names = self._described()
        nodes, labels = description["nodes"], self._labels()
        lines = [f"hawk.Graph  {len(self._launches)} launches · {len(nodes)} nodes · {description['n_levels']} levels"
                 f"{' · recorded for reverse' if description['flags'] else ''}"]
        depth: dict = {}
        for i, node in enumerate(nodes):
            depth[i] = 0 if node["parent"] < 0 else depth[node["parent"]] + 1
            touched = ", ".join(f"{names[plane]}:{_ACCESS_WORDS[access]}" for plane, access in zip(node["planes"], node["access"])
                                if plane >= 0)
            lines.append(f"{i:>4}  L{node['level']:<3}{'  ' * depth[i]}{_node_label(node, names, labels, nodes)}"
                         f"{f'  ({touched})' if touched else ''}")
        lines.append("edges")
        lines += [f"{edge.src:>4} -> {edge.dst:<4} {edge.kind}{f' {edge.plane}' if edge.plane else ''}" for edge in self.edges()]
        return "\n".join(lines) + "\n"

    def show(self, *, file=None) -> None:
        """Print the built graph as text (:func:`str` returns the same): to ``file``, default standard output."""
        print(str(self), end="", file=file)

    def _end_recording(self):
        """The graph the recording ends in."""
        return through_door(self._door.record_end)

    def _labels(self) -> list:
        """The label of each launch in recording order: the name it was given, else its kernel's, numbered from the second use."""
        seen: dict = {}
        labels = []
        for launch in self._launches:
            base = launch.name or launch.bound.kernel.sidecar["kernel"]
            seen[base] = seen.get(base, 0) + 1
            labels.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
        return labels

    def _abandon(self) -> None:
        """Discard the recording a refusal interrupted, whatever was left open in it, so the next graph of the process can
        record: a refused recording call changes nothing, so the library keeps the recording until it is told to drop it."""
        self._launches.clear()
        self._recorded.clear()
        self._counting.clear()
        through_door(self._door.record_abort)

    def build(self) -> GraphPipeline:
        """Record the added launches into one graph and return the pipeline, ready to :meth:`launch`."""
        self._check_open("build")
        if not self._items:
            raise HawkError(f"{type(self).__name__}.build: nothing has been added")
        through_door(self._door.record_begin)
        try:
            self._emit(self._items)
            self._handles.append(self._end_recording())
        except BaseException:
            self._abandon()
            raise
        self._finalizer = weakref.finalize(self, _release, self._door, self._handles)
        return self

    def launch(self, n: int = 1) -> GraphPipeline:
        """Replay the graph ``n`` times and return the pipeline. A host replay has finished when this returns; a replay over device
        memory is queued, and :meth:`synchronize` (or :func:`hawk.synchronize`) returns when it has run."""
        if self._finalizer is None or not self._finalizer.alive:
            raise HawkError(f"{type(self).__name__}.launch: the graph is not built, or is closed; build it first")
        for _ in range(n):
            through_door(self._door.graph_launch, self._handles[0])
            for bound in self._counting:
                bound.settle_counters()
        return self

    def synchronize(self) -> None:
        """Return when every replay so far has finished: a host replay has, when it returns."""
        if self._door is not None:
            through_door(self._door.sync)

    def close(self) -> None:
        """Free the recorded graph at once; the bound plans stay the caller's. Closing twice is harmless."""
        if self._finalizer is not None:
            self._finalizer()


def _quoted(label: str) -> str:
    """``label`` safe inside the double quotes of a Mermaid or DOT label."""
    return label.replace('"', "'")


def _node_label(node: dict, names: list, labels: list, nodes: list) -> str:
    """The label of one node of a description: a launch by its label, a conditional by its test, an arm by its place in the
    conditional (``then`` and ``else``, ``body`` of a loop, ``case k``), the rest by their kind."""
    kind = node["kind"]
    detail = node["arm"]   # the kind of a conditional, or the place of an arm in its conditional
    if kind == _osprey.EG_GNODE_LAUNCH:
        return labels[node["launch"]]
    if kind == _osprey.EG_GNODE_HOST:
        return node["unit"]
    touched = [names[plane] for plane in node["planes"] if plane >= 0]
    if kind == _osprey.EG_GNODE_COND:
        if detail == _SWITCH:
            return f"switch {touched[0]}[{node['cell_index']}]"
        test = f"{touched[0]}[{node['cell_index']}]"
        if node["has_baseline"]:
            test += f" {'==' if node['cond_flags'] else '!='} {touched[1]}"
        elif node["cond_flags"]:
            test = f"not {test}"
        cap = f" (at most {node['max_iters']})" if node["max_iters"] else ""
        return f"{_COND_WORDS[detail]} {test}{cap}"
    if kind == _osprey.EG_GNODE_ARM:
        owner = nodes[node["parent"]]["arm"]
        return {_WHILE: "body", _SWITCH: f"case {detail}"}.get(owner, ("then", "else")[detail] if detail < 2 else f"case {detail}")
    if kind == _osprey.EG_GNODE_WAIT:
        return f"wait {', '.join(touched)}"
    return _NODE_WORDS[kind]


class GraphRecord:
    """The record of a recorded graph, what :func:`reverse` is asked to differentiate.

    ``graph`` is the :class:`Graph` it is recorded from.
    """

    def __init__(self, graph: Graph) -> None:
        """Hold the graph the record belongs to."""
        self.graph = graph


class _Region:
    """An open fork: its :meth:`branch` es, joined when the ``with`` ends."""

    def __init__(self, graph: Graph) -> None:
        """Open a fork in the scope the graph is recording."""
        self._graph = graph
        self._parent = graph._scopes[-1]
        self._branches: list = []
        self._closed = False

    def __enter__(self) -> _Region:
        """Open the region."""
        return self

    def __exit__(self, error_type, error, trace) -> None:
        """Join: the fork goes into the graph with the branches that hold something."""
        self._closed = True
        kept = [branch for branch in self._branches if branch]
        if error_type is None and kept:
            self._parent.append(_Fork(kept))

    def branch(self) -> _Branch:
        """A branch of the fork: ``with p.branch() as b``."""
        if self._closed:
            raise HawkError("Graph.parallel: the region is closed; open a new one")
        return _Branch(self._graph, self)

    def add(self, step, *, name=None):
        """Refused: a step goes into a branch of the fork, ``p.branch()``."""
        raise HawkError("Graph.parallel: add steps in a branch (`with p.branch() as b: b.add(...)`)")

    def parallel(self):
        """Refused: a fork nests in a branch, ``b.parallel()``."""
        raise HawkError("Graph.parallel: open a nested fork from a branch (`b.parallel()`)")


class _Branch:
    """One branch of a fork: the same vocabulary as the graph (``add``, ``parallel``) and ``wait``."""

    def __init__(self, graph: Graph, fork: _Region) -> None:
        """Make the branch of ``fork``."""
        self._graph = graph
        self._fork = fork
        self._items: list = []

    def add(self, step, *, name: str | None = None) -> _Branch:
        """Append ``step`` (see :meth:`Graph.add`) to this branch and return it."""
        self._graph._check_handle(self, "add")
        self._graph._add_step(step, name)
        return self

    def parallel(self) -> _Region:
        """Open a fork nested in this branch; the branches of the outer fork are joined after it."""
        self._graph._check_handle(self, "parallel")
        return _Region(self._graph)

    def __enter__(self) -> _Branch:
        """Record into this branch."""
        self._fork._branches.append(self._items)
        self._graph._scopes.append(self._items)
        self._graph._branches.append(self)
        return self

    def __exit__(self, error_type, error, trace) -> None:
        """Stop recording into this branch."""
        self._graph._scopes.pop()
        self._graph._branches.pop()

    def wait(self, *arrays) -> _Branch:
        """Order what this branch records next after the earlier sibling branches that write ``arrays``: the way out of a race
        the library refuses at the join."""
        self._graph._check_handle(self, "wait")
        if not arrays:
            raise HawkError("Graph.wait: name the arrays to wait for")
        self._items.append(_Wait(tuple(arrays)))
        return self


class Graph(GraphPipeline):
    """A graph of launches, conditionals and forks, written by hand or by a builder, which may keep a record for
    :func:`reverse`.

    ``record=True`` plans the recording for :func:`reverse`: its plans must carry the vjp of their kernel (a builder's kernel
    calls do; a plan added by hand is made with ``hawk.deploy(..., reverse=True)``).
    """

    def __init__(self, *, record: bool = False) -> None:
        """Make an empty graph; ``record`` as in the class. Made inside a builder, the graph is the one the builder records into."""
        super().__init__()
        self._records = record
        self._branches: list[_Branch] = []
        self._result = None
        graph_builder.attach(self)

    # -- writing the graph --
    def _check_handle(self, owner, verb: str) -> None:
        """Refuse ``verb`` on a handle that is not the innermost open scope: a branch's steps and forks go through its handle."""
        self._check_open(verb)
        innermost = self._branches[-1] if self._branches else None
        if owner is not innermost and not (innermost is None and owner is self):
            raise HawkError(f"Graph.{verb}: inside a branch, use the branch handle (`with p.branch() as b: b.{verb}(...)`)")

    def add(self, step, *, name: str | None = None) -> Graph:
        """Append ``step`` and return the graph. A step is a :class:`~hawk.runtime.BoundPlan`, a built :class:`Graph` (its
        launches run here, as they were recorded), a bound :func:`hawk.host_step`, a :func:`skippable` or :func:`repeat_while`, or a
        list of these in order. ``name`` labels the launch of a bound plan. Any other function is refused: the library records
        launches, never Python."""
        self._check_handle(self, "add")
        self._add_step(step, name)
        return self

    def _add_step(self, step, name: str | None) -> None:
        """Append the items of ``step`` to the scope being recorded."""
        for item in self._items_of(step, name):
            self._append(item)

    def _items_of(self, step, name: str | None = None) -> list:
        """The items ``step`` stands for."""
        if isinstance(step, BoundPlan):
            step.join(self._store_over(step.door))
            return [_Launch(step, name)]
        if isinstance(step, BoundHostStep):
            if name is not None:
                raise HawkError(f"Graph.add: a host step is named by its function ({step.step.name}); name= labels a launch")
            self._store_over(step.door)
            return [_HostCall(step)]
        if isinstance(step, (list, tuple)):
            return [item for part in step for item in self._items_of(part)]
        if isinstance(step, Skippable):
            return [_cond_of(step.guard, _IF, [self._items_of(step.step)])]
        if isinstance(step, RepeatWhile):
            return [self._while_of(step)]
        if isinstance(step, Graph):
            if step._finalizer is None:
                raise HawkError("Graph.add: a graph added as a step is built first (g.build())")
            self._store_over(step._door)
            return list(step._items)
        if callable(step) and graph_builder.current() is not None:
            scope: list = []
            self._scopes.append(scope)
            try:
                step()
            finally:
                self._scopes.pop()
            return scope
        raise HawkError(f"Graph.add takes a bound plan, a built Graph, skippable(...) or repeat_while(...), got "
                        f"{type(step).__name__}: a graph records launches and never Python; write the step as a kernel")

    def parallel(self) -> _Region:
        """Open a fork: ``with g.parallel() as p``, ``with p.branch() as b`` for each branch (``b.add(bound)`` by hand, or kernel
        calls inside a builder); leaving the outer ``with`` joins. No edge is declared: the library orders the branches by the
        planes they touch and refuses two that race, ``b.wait(array)`` being the way out."""
        self._check_handle(self, "parallel")
        return _Region(self)

    # -- building and replaying --
    def _end_recording(self):
        """The graph the recording ends in; a graph kept for :func:`reverse` names the vjp of each launch that has one."""
        if not self._records:
            return super()._end_recording()
        vjps = [(ordinal, unit) for ordinal, launch in enumerate(self._launches)
                if (unit := launch.bound.kernel.graph_vjp(self._door)) is not None]
        return through_door(self._door.record_end, vjps)

    def build(self) -> Graph:
        """Record the graph and return it, ready to :meth:`launch`. A builder builds the graph it makes."""
        if self._branches:
            raise HawkError("Graph.build: a branch is still open; leave the `with` blocks first")
        return super().build()

    @property
    def record(self) -> GraphRecord | None:
        """The :class:`GraphRecord` of a built graph made with ``record=True``, else ``None``."""
        if self._records and self._finalizer is not None:
            return GraphRecord(self)
        return None

    @property
    def result(self):
        """What a builder declared the graph returns: an array, or a tuple, list or dict of arrays, nested; ``None`` when it
        declared nothing. Set it while the graph is being written; read-only once built."""
        return self._result

    @result.setter
    def result(self, tree) -> None:
        """Declare what the graph returns."""
        self._check_open("result")
        _leaves(tree)
        self._result = tree

    def __call__(self, n: int = 1):
        """Replay the graph ``n`` times, wait for it and return :attr:`result`: a graph a builder made is called as the function
        it records, on the arrays it was built over."""
        self.launch(n)
        self.synchronize()
        return self._result

    def bind(self, **planes) -> Graph:
        """Take new arrays for the parameters of the builder that made this graph, with no new record: the launches read them at
        the next replay. Each new array has the shape and dtype of the one it replaces. A parameter that is a Python value was
        decided at build and a parameter that is a condition cell was recorded: both are refused, by name, and the builder is
        called again."""
        if self._params is None:
            raise HawkError("Graph.bind: only a graph made by a builder has parameters; rebind a bound plan with "
                            "BoundPlan.rebind")
        if self._finalizer is None:
            raise HawkError("Graph.bind: the graph is not built")
        swaps = {}
        for name, new in planes.items():
            if name not in self._params:
                raise HawkError(f"Graph.bind: {name!r} is not a parameter of the builder; they are {sorted(self._params)}")
            old = self._params[name]
            if not hasattr(old, "shape"):
                raise HawkError(f"Graph.bind: {name!r} is a Python value, decided at build: call the builder again")
            if not hasattr(new, "shape") or tuple(new.shape) != tuple(old.shape) or new.dtype != old.dtype:
                raise HawkError(f"Graph.bind: {name!r} must keep its shape and dtype ({tuple(old.shape)}, {old.dtype}); "
                                "call the builder again")
            swaps[name] = (old, new)
        for name, (old, _new) in swaps.items():
            if any(old is cell for cell in self._cells()):
                raise HawkError(f"Graph.bind: {name!r} is a condition cell, recorded at build: call the builder again")
            if any(isinstance(leaf, _HostCall) and any(old is array for array in leaf.bound.planes.values())
                   for leaf in self._leaves_everywhere()):
                raise HawkError(f"Graph.bind: {name!r} is an argument of a host step, recorded at build: call the builder again")
        for old, new in swaps.values():
            for launch in self._leaves_everywhere():
                held = {key: new for key, value in launch.bound.planes.items() if value is old}
                if held:
                    launch.bound.rebind(**held)
            self._result = _swapped(self._result, old, new)
        self._params.update({name: new for name, (_old, new) in swaps.items()})
        return self

    def update(self, values: dict | None = None, /, **named) -> Graph:
        """Change values of a built graph's launches between replays: ``g.update(a=0.3)`` sets ``a`` in every launch that binds
        it by value, ``g.update({"scale#2.a": 0.3})`` in one launch (its label, as :meth:`node_labels` shows it, then the name). Nothing is
        recorded again: the library reads the values at the next replay. An array plane is the caller's own memory (write it, or
        :meth:`bind` a new one), a name no launch binds by value is refused, and so is a graph made with ``record=True``: the
        library refuses to replay a recorded graph whose plans were rebound, so build it again with the new value."""
        self._check_built("update")
        if self._records:
            raise HawkError("Graph.update: a graph kept for reverse holds the values it was recorded with (the library refuses "
                            "to replay it once a plan is rebound: a plan of an EG_RECORD_DIFF graph was rebound since "
                            "eg_record_end); build the graph again with the new value")
        given = {**(values or {}), **named}
        found: dict = {}
        by_value = set()
        for label, launch in zip(self._labels(), self._launches):
            for name in launch.bound.planes:
                if name in launch.bound.views:
                    continue
                by_value.add(name)
                found.update({(key, id(launch.bound)): (launch.bound, name) for key in (name, f"{label}.{name}")})
        for key in given:
            if not any(key == hit_key for hit_key, _ in found):
                raise HawkError(f"Graph.update: {key!r} is not a value any launch binds by value; they are "
                                f"{sorted(by_value)}, or <launch label>.<name>; an array plane is written by the caller")
        for (key, _), (bound, name) in found.items():
            if key in given:
                bound.rebind(**{name: given[key]})
        return self

    def _check_built(self, verb: str) -> None:
        """Refuse ``verb`` on a graph that is not built, or is closed."""
        if self._finalizer is None or not self._finalizer.alive:
            raise HawkError(f"Graph.{verb}: the graph is not built, or is closed; build it first")

    def _cells(self, items: list | None = None):
        """The condition cells and baselines of the tree."""
        for item in self._items if items is None else items:
            if isinstance(item, _Cond):
                yield item.cell
                if item.baseline is not None:
                    yield item.baseline
                for arm in item.arms:
                    yield from self._cells(arm)
            elif isinstance(item, _Fork):
                for branch in item.branches:
                    yield from self._cells(branch)

    def __repr__(self) -> str:
        """The launches, by name where one is given."""
        launches = list(self._leaves_everywhere())
        named = ", ".join(launch.name or f"#{i}" for i, launch in enumerate(launches))
        return f"<hawk.Graph {'built' if self._finalizer is not None else 'unbuilt'}: {named or 'empty'}>"

    @staticmethod
    def builder(function):
        """Decorate ``function`` as a builder or a block.

        A function with no :class:`Graph` parameter is a builder: it makes its graph (``g = hawk.Graph()``), calls kernels, which
        are recorded and return their output arrays, and returns ``g``; calling it builds and returns the graph. A function whose
        first parameter is annotated ``hawk.Graph`` is a block: it appends to the open graph it is given. In either, ``if``,
        ``while``, ``for _ in range(cap)`` with ``if not c: break`` and ``match`` on a one-element ``int32`` or ``uint32`` array
        (or a :class:`SkipGuard`) become conditionals of the graph, decided at every replay; any other condition is a Python
        value, decided once at build.
        """
        return graph_builder.make_builder(function)


def _leaves(tree) -> list:
    """The arrays of a result ``tree`` (an array, or nested tuples, lists and dicts of arrays); anything else is refused."""
    if tree is None:
        return []
    if hasattr(tree, "shape"):
        return [tree]
    if isinstance(tree, (tuple, list)):
        return [leaf for part in tree for leaf in _leaves(part)]
    if isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise HawkError("Graph.result: dict keys are strings")
        return [leaf for part in tree.values() for leaf in _leaves(part)]
    raise HawkError(f"Graph.result is an array, or a tuple, list or dict of arrays, nested; got {type(tree).__name__}")


def _swapped(tree, old, new):
    """``tree`` with the array ``old`` replaced by ``new`` wherever it is."""
    if tree is old:
        return new
    if isinstance(tree, dict):
        return {key: _swapped(value, old, new) for key, value in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_swapped(value, old, new) for value in tree)
    return tree


class _Plane(NamedTuple):
    """One plane a launch of a graph binds: the ``view`` the library reads, the array the caller ``given``, and where it is bound."""

    launch: int
    label: str
    name: str
    view: object
    given: object


def _planes_of(graph: Graph) -> tuple:
    """``(planes, values)`` of the graph's launches: every array plane bound (:class:`_Plane`) and every plane bound by value
    (``(label, name)``), in launch order."""
    planes, values = [], []
    launch_labels = iter(graph._labels())
    for ordinal, launch in enumerate(graph._recorded):
        label = launch.name if isinstance(launch, _HostCall) else next(launch_labels)
        bound = launch.bound
        for name, given in bound.planes.items():
            if name in bound.views:
                planes.append(_Plane(ordinal, label, name, bound.views[name], given))
            else:
                values.append((label, name))
    return planes, values


def _resolve(key: str, planes: list, values: list, what: str) -> _Plane:
    """The plane ``key`` names: a plane name, or ``<launch label>.<plane name>`` when two launches bind different arrays under
    one name. A plane bound by value, an unknown name and an ambiguous one are refused by name."""
    qualified = [p for p in planes if f"{p.label}.{p.name}" == key]
    named = [p for p in planes if p.name == key]
    found = qualified or named
    if not found:
        if any(key in (name, f"{label}.{name}") for label, name in values):
            raise HawkError(f"hawk.reverse: {what} {key!r} is bound by value, which has no gradient: pass a one-element array")
        shown = sorted({p.name for p in planes})
        raise HawkError(f"hawk.reverse: {what} {key!r} is not a plane of the graph; its planes are {shown}")
    if len({PlaneStore.region(p.view) for p in found}) > 1:
        options = list(dict.fromkeys(f"{p.label}.{p.name}" for p in found))
        raise HawkError(f"hawk.reverse: {what} {key!r} names different arrays in different launches; use one of {options}")
    return found[0]


def _in_shape(array, plane: _Plane, key: str):
    """The cotangent ``array`` of ``key`` as a C-contiguous array in the shape of the view the library reads."""
    import numpy as np

    seed = np.ascontiguousarray(array, dtype=plane.view.dtype)
    shape = tuple(plane.view.shape)
    if seed.shape == shape:
        return seed
    if seed.T.shape == shape:
        return seed.T
    if seed.size == plane.view.size:
        return seed.reshape(shape)
    raise HawkError(f"hawk.reverse: the cotangent of {key!r} has shape {seed.shape}; the plane is {tuple(plane.given.shape)}")


def _out_shape(grad, plane: _Plane):
    """The gradient ``grad``, in the shape of a view the library reads, as an array shaped like the caller's own."""
    shape = tuple(plane.given.shape)
    if tuple(grad.shape) == shape:
        return grad
    if tuple(grad.T.shape) == shape:
        return grad.T
    return grad.reshape(shape)


def _inputs(graph: Graph, planes: list) -> dict:
    """``{memory key: plane}`` of the planes the graph reads before it writes them and differentiates: the inputs a gradient can
    be asked of, by the first launch that reads each."""
    written, inputs = set(), {}
    for ordinal, launch in enumerate(graph._recorded):
        mine = {p.name: p for p in planes if p.launch == ordinal}
        if isinstance(launch, _HostCall):
            step = launch.bound.step
            # a step with a reverse may read every float argument, and may write the Mutable ones
            reads = [name for name, plane in mine.items() if plane.view.dtype.kind == "f"] if step.reverses else []
            writes = [argument.name for argument in step.arguments if argument.writes]
        else:
            kernel = launch.bound.kernel
            reads = kernel.gradient_names() if kernel.graph_vjp(graph._door) is not None else []
            writes = kernel.output_names()
        for name in reads:
            plane = mine.get(name)
            if plane is not None and PlaneStore.region(plane.view) not in written:
                inputs.setdefault(PlaneStore.region(plane.view), plane)
        written |= {PlaneStore.region(mine[name].view) for name in writes if name in mine}
    return inputs


def _gradient_keys(inputs: dict, planes: list) -> dict:
    """``{memory key: the name its gradient is returned under}``: the plane's name, or ``<launch label>.<name>`` when two of the
    inputs share a name."""
    count: dict = {}
    for plane in inputs.values():
        count[plane.name] = count.get(plane.name, 0) + 1
    return {key: plane.name if count[plane.name] == 1 else f"{plane.label}.{plane.name}" for key, plane in inputs.items()}


def _reverse_graph(graph: Graph, cotangents: dict, wrt, strict: bool) -> Gradients:
    """The gradients of the last replay of ``graph`` for ``cotangents``, by plane name (see :func:`reverse`)."""
    import numpy as np

    if graph._finalizer is None or not graph._finalizer.alive:
        raise HawkError("hawk.reverse: the graph is closed")
    planes, values = _planes_of(graph)
    inputs = _inputs(graph, planes)
    names = _gradient_keys(inputs, planes)
    if wrt is None:
        wanted = dict(inputs)
    else:
        wanted = {}
        for key in ([wrt] if isinstance(wrt, str) else wrt):
            plane = _resolve(key, planes, values, "wrt")
            if plane.view.dtype.kind != "f":
                continue                                    # an integer plane has no gradient and is left out
            memory = PlaneStore.region(plane.view)
            if memory not in inputs:
                raise HawkError(f"hawk.reverse: wrt {key!r} is written by the graph before anything reads it, so it is no "
                                f"input; the inputs are {sorted(names.values())}")
            wanted[memory] = inputs[memory]
    door, store, temporary = graph._door, graph._store, []
    try:
        seeds = []
        for key, array in cotangents.items():
            plane = _resolve(key, planes, values, "cotangent")
            seed = door.plane_wrap(_in_shape(array, plane, key))
            temporary.append(seed)
            seeds.append((store.plane(plane.view), seed))
        gradients, pairs = {}, []
        for memory, plane in wanted.items():
            grad = np.zeros(plane.view.shape, dtype=plane.view.dtype)
            handle = door.plane_wrap(grad)
            temporary.append(handle)
            pairs.append((store.plane(plane.view), handle))
            gradients[names[memory]] = (grad, plane)
        bitwise, passes = through_door(door.graph_reverse, graph._handles[0], seeds, pairs, strict=strict)
    finally:
        for handle in temporary:
            door.plane_free(handle)
    return Gradients({name: _out_shape(grad, plane) for name, (grad, plane) in gradients.items()},
                     GradientReport(dict(zip(gradients, map(bool, bitwise))), max(passes, 1)))


def reverse(record, cotangents: dict, wrt=None, strict: bool = False, policy: str = "compensated") -> Gradients:
    """Reverse a recorded run: the gradients of ``cotangents`` (by plane name) through ``record``.

    ``record`` is the ``record`` of a runner made by :func:`hawk.until_done` or of a result of :func:`hawk.simulate`, with
    ``record=True``: see :meth:`hawk.runner.Runner.reverse`.

    ``record`` may be the ``record`` of a built :class:`Graph` made with ``record=True`` and replayed: ``cotangents`` are the
    arrays ``dJ/d(plane)`` of the planes the graph writes, by name, and the result holds the gradient of every float input (a plane
    the graph reads before it writes it) in the shape the caller bound it. A name is a plane's name, or
    ``<launch label>.<plane>`` where two launches bind different arrays under one name, which an ambiguous bare name is refused
    with. The gradient goes through forks, conditionals and loops as they ran at that replay. ``wrt`` (a name or a sequence of
    names) asks for those inputs only: an unknown name is refused, an integer plane is left out, and a plane bound by a Python
    number is refused by name, since a one-element array has a gradient and a number has none.

    The result is a :class:`~hawk.runner.Gradients` (a read-only mapping by plane name) whose ``report`` says, for each gradient,
    whether it is bit-reproducible (``report.bitwise``: ``False`` for one fed by a cross-sample scatter-add, whose summing order
    varies from run to run) and in how many passes over the memory the library made it (``report.chunks``). ``strict=True``
    refuses such a gradient, with the library's reason, and returns nothing, instead of reporting it. ``policy`` names how the
    sums over samples are made: the library makes them in one deterministic compensated order, which ``"compensated"`` (the
    default) and ``"auto"`` ask for; ``"ordered"`` (a plain sum in sample order) and any other word are refused by name.
    """
    strict = check_reverse_options(strict, policy, "hawk.reverse")
    if isinstance(record, RunRecord):
        if wrt is not None:
            raise HawkError("hawk.reverse: wrt= selects among a graph's inputs; a runner's reverse differentiates the planes of "
                            "its step's vjp")
        return record.runner.reverse(cotangents, strict=strict)
    if isinstance(record, GraphRecord):
        return _reverse_graph(record.graph, cotangents, wrt, strict)
    raise HawkError(f"hawk.reverse takes the record of a run made with record=True, got {type(record).__name__}: "
                    "hawk.until_done(..., record=True).record or hawk.simulate(..., record=True).record, or the record of a "
                    "Graph(record=True)")
