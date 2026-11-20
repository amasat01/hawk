# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recording bound plans into a graph and replaying it: :class:`GraphPipeline`, :class:`Graph` and :func:`reverse`.

A graph is a sequence of plan launches recorded once by the host library and replayed as one. What it records is what the
library can replay: launches of plans bound with :meth:`hawk.plans.Plan.bind`. Python is never recorded, so a step that
is a function is refused by name; the planes are the caller's memory and a replay sees what they hold at that time.

Not offered yet: conditional nodes, child graphs and kernel calls inside a graph.
"""

from __future__ import annotations

import weakref

from . import ospreys
from .errors import HawkError
from .runner import RunRecord
from .runtime import BoundPlan, through_door
from .targets import host_target

__all__ = ["Graph", "GraphPipeline", "GraphRecord", "reverse"]


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
        self._bounds: list[BoundPlan] = []
        self._door = ospreys.serving(host_target().id).runtime_door()
        self._handles: list = []
        self._finalizer = None

    def _check_open(self, verb: str) -> None:
        """Refuse ``verb`` once the pipeline is built: what is recorded cannot grow."""
        if self._finalizer is not None:
            raise HawkError(f"{type(self).__name__}.{verb}: the graph is built; make another to record more")

    def add(self, step) -> GraphPipeline:
        """Append the launch of ``step``, a :class:`~hawk.runtime.BoundPlan`, and return the pipeline.

        A function is refused: the library records launches, not Python, so a function would run once at build time and
        never again at a replay. Write the step as a kernel and add its bound plan.
        """
        self._check_open("add")
        if not isinstance(step, BoundPlan):
            raise HawkError(f"{type(self).__name__}.add takes a bound plan (hawk.plan(...).bind(...)), got "
                            f"{type(step).__name__}: a graph records launches and never Python; write the step as a kernel")
        self._bounds.append(step)
        return self

    def build(self) -> GraphPipeline:
        """Record the added launches into one graph and return the pipeline, ready to :meth:`launch`."""
        self._check_open("build")
        if not self._bounds:
            raise HawkError(f"{type(self).__name__}.build: nothing has been added")
        through_door(self._door.record_begin)
        try:
            for bound in self._bounds:
                bound.enqueue()
        except BaseException:
            try:
                _release(self._door, [self._door.record_end()])  # leave the recording before the refusal goes out
            except Exception:  # the refusal that ended the recording is the one the caller needs
                pass
            raise
        self._handles.append(through_door(self._door.record_end))
        self._finalizer = weakref.finalize(self, _release, self._door, self._handles)
        return self

    def launch(self, n: int = 1) -> GraphPipeline:
        """Replay the graph ``n`` times and return the pipeline. A host replay has finished when this returns."""
        if self._finalizer is None or not self._finalizer.alive:
            raise HawkError(f"{type(self).__name__}.launch: the graph is not built, or is closed; build it first")
        for _ in range(n):
            through_door(self._door.graph_launch, self._handles[0])
            for bound in self._bounds:
                bound.settle_counters()
        return self

    def synchronize(self) -> None:
        """Return when every replay so far has finished: a host replay has, when it returns."""
        through_door(self._door.sync)

    def close(self) -> None:
        """Free the recorded graph at once; the bound plans stay the caller's. Closing twice is harmless."""
        if self._finalizer is not None:
            self._finalizer()


class GraphRecord:
    """The record of a recorded graph, what :func:`reverse` is asked to differentiate.

    ``graph`` is the :class:`Graph` it is recorded from.
    """

    def __init__(self, graph: Graph) -> None:
        """Hold the graph the record belongs to."""
        self.graph = graph


class Graph(GraphPipeline):
    """A :class:`GraphPipeline` whose launches may carry a name, and which can keep a record for :func:`reverse`.

    ``record=True`` asks the graph for a :attr:`record` once it has been launched. Conditional nodes, child graphs and kernel
    calls inside a graph are not offered yet.
    """

    def __init__(self, *, record: bool = False) -> None:
        """Make an empty graph; ``record`` as in the class."""
        super().__init__()
        self._names: list[str | None] = []
        self._records = record

    def add(self, step, *, name: str | None = None) -> Graph:
        """Append the launch of ``step`` (as :meth:`GraphPipeline.add`) under ``name``, and return the graph."""
        super().add(step)
        self._names.append(name)
        return self

    @property
    def record(self) -> GraphRecord | None:
        """The :class:`GraphRecord` of a built graph made with ``record=True``, else ``None``."""
        if self._records and self._finalizer is not None:
            return GraphRecord(self)
        return None

    def __repr__(self) -> str:
        """The launches, by name where one is given."""
        named = ", ".join(name or f"#{i}" for i, name in enumerate(self._names))
        return f"<hawk.Graph {'built' if self._finalizer is not None else 'unbuilt'}: {named or 'empty'}>"


def reverse(record, cotangents: dict) -> dict:
    """Reverse a recorded run: the gradients of ``cotangents`` (by plane name) through ``record``.

    ``record`` is the ``record`` of a runner made by :func:`hawk.until_done` or of a result of :func:`hawk.simulate`, with
    ``record=True``: see :meth:`hawk.runner.Runner.reverse`. A recorded graph (``graph.record``) is refused by name: the
    library reverses a runner's record and has no reverse over a graph's yet.
    """
    if isinstance(record, RunRecord):
        return record.runner.reverse(cotangents)
    if isinstance(record, GraphRecord):
        raise HawkError("hawk.reverse over a recorded graph: the library reverses a runner's record and has no reverse over "
                        "a graph's yet; it lands with a later library addition")
    raise HawkError(f"hawk.reverse takes the record of a run made with record=True, got {type(record).__name__}: "
                    "hawk.until_done(..., record=True).record or hawk.simulate(..., record=True).record")
