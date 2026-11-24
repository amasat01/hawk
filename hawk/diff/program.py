# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivatives of a recorded program: ``hawk.diff.jvp(builder)`` and ``hawk.diff.vjp(builder)``.

A builder (:meth:`hawk.Graph.builder`) makes a graph of launches, whole-plane nodes, conditionals and forks. The forward
derivative re-records that graph with tangent planes: a launch becomes the launch of the kernel that computes its outputs
and their tangents together, a whole-plane node gains its tangent node, and conditionals and forks keep their structure and
their tests. The reverse derivative is the door's reverse of the recorded run, the mechanism of :func:`hawk.reverse`.

Tangent arrays: the caller gives the tangent of a builder's array parameter by the parameter's name in ``tangents=``; every
other float array of the program gets a zero tangent made beside it. :meth:`~hawk.Graph.tangent_of` answers the tangent of
any array.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .. import graph_builder
from ..errors import HawkError

#: The element types that carry a tangent.
_FLOATS = ("float64", "float32")


def is_builder(primal: Any) -> bool:
    """Whether ``primal`` is something other than a kernel or a sink set, which a derivative takes only as a builder: a
    callable that is not a kernel is refused by name unless it is one."""
    return callable(primal) and not hasattr(primal, "sinks")


def _refuse_unless_builder(verb: str, step: Any) -> None:
    """Refuse by name a callable that is not a builder."""
    if getattr(step, "makes_graph", False):
        return
    what = getattr(step, "__name__", type(step).__name__)
    raise HawkError(
        f"hawk.diff.{verb}: {what!r} is neither a kernel nor a builder; a builder is a function decorated with "
        "@hawk.Graph.builder that makes and returns its graph"
    )


class _Tangents:
    """The tangent arrays of one program: the ones the caller gave, and the zeros made beside every other float array."""

    def __init__(self, given: dict) -> None:
        from ..runtime import PlaneStore

        self._region = PlaneStore.region
        self._held: dict = {self._region(array): tangent for array, tangent in given}

    def of(self, array: Any) -> Any:
        """The tangent of ``array``: none for anything but a float array, the given one, else zeros made on first ask."""
        if getattr(array, "dtype", None) is None or str(array.dtype) not in _FLOATS:
            return None
        key = self._region(array)
        if key not in self._held:
            import numpy

            if not isinstance(array, numpy.ndarray):
                raise HawkError(
                    f"hawk.diff.jvp: the tangent of a {type(array).__name__} is not made here; "
                    "pass one in tangents= or program over NumPy arrays"
                )
            self._held[key] = numpy.zeros_like(array)
        return self._held[key]


class _Capture:
    """The recorder a whole-plane node is re-called under: the nodes it makes are kept, not added to a graph."""

    def __init__(self, store_over: Callable) -> None:
        self.nodes: list = []
        self._store_over = store_over
        self.graph = self
        self._branches: list = []
        self.kind, self.name, self.block_depth = "builder", "hawk.diff.jvp", 0

    def _graph(self) -> _Capture:
        return self

    def _append(self, item: Any) -> None:
        self.nodes.append(item)


def _settings_of(items: list) -> dict:
    """The settings the first deployed kernel launched in ``items`` was built with; none when there is none."""
    from ..graph import _Cond, _Fork, _Launch

    for item in items:
        if isinstance(item, _Launch) and item.bound.settings is not None:
            return item.bound.settings
        nested = (
            item.arms
            if isinstance(item, _Cond)
            else item.branches
            if isinstance(item, _Fork)
            else ()
        )
        for part in nested:
            if found := _settings_of(part):
                return found
    return {}


def _tangent_launch(launch: Any, tangents: _Tangents, cache: dict) -> Any:
    """The launch of the kernel that computes ``launch``'s outputs and their tangents: its planes, then a tangent plane for each
    float plane it names."""
    from ..graph import _Launch
    from ..plans import deploy
    from .transform import with_derivative

    bound = launch.bound
    traced = bound.traced
    if traced is None or bound.settings is None:
        raise HawkError(
            f"hawk.diff.jvp: the launch of {launch.name or bound.kernel.path.name!r} was made from a loaded artifact, "
            "which carries no kernel to differentiate; deploy the kernel itself"
        )
    written = [
        bound.planes[n] for n in bound.kernel.output_names() if n in bound.planes
    ]
    if not any(tangents.of(a) is not None for a in written):
        return launch  # it writes no float plane (a counter, an index): nothing to differentiate
    # the tangent kernel is built as the primal was; a graph kept for hawk.reverse had its kernels built with their vjp, which the
    # tangent graph does not need
    settings = {**bound.settings, "reverse": False}
    key = (traced.name, tuple(sorted((k, repr(v)) for k, v in settings.items())))
    if key not in cache:
        derived = with_derivative(traced)
        cache[key] = (derived, deploy(derived, **settings))
    derived, plan = cache[key]
    planes = dict(bound.planes)
    for dot, primal in derived.tangents.items():
        given = bound.planes.get(primal)
        planes[dot] = (
            0.0 if given is None or not hasattr(given, "dtype") else tangents.of(given)
        )
    names = {name for _role, name in plan.kernel.arg_spec}
    planes = {name: value for name, value in planes.items() if name in names}
    return _Launch(plan.bind(n_samples=bound.n, **planes), launch.name)


def _winner_tangent(
    node: Any, tangents: _Tangents, capture: _Capture, cache: dict
) -> list:
    """The nodes that give the tangent of a one-column minimum or maximum: the same fold over the plane again, which says where
    the winner is (the lowest lane among equals), then a gather of the plane's tangent at that lane."""
    from ..collectives import reduce

    plane, result = node.arrays[:2]
    if plane.ndim != 1:
        raise HawkError(
            f"hawk.diff.jvp: {node.label}: the tangent of a minimum or maximum is built for a plane of one axis, "
            f"this one has shape {tuple(plane.shape)}"
        )
    dots = tangents.of(plane), tangents.of(result)
    if any(d is None for d in dots):
        return []
    token = graph_builder._ACTIVE.set(capture)
    try:
        _, where = reduce(plane, node.op, with_index=True)
    finally:
        graph_builder._ACTIVE.reset(token)
    return _gathered(plane, where, dots, capture, cache)


def _running_winner_tangent(
    node: Any, tangents: _Tangents, capture: _Capture, cache: dict
) -> list:
    """The nodes that give the tangent of a running minimum or maximum: the same scan over the plane again, writing the winner of
    every prefix (the lowest lane among equals), then a gather of the plane's tangent at those lanes. A prefix with no winner
    (the first lane of an exclusive scan) gets a zero tangent."""
    from ..collectives import _scan_with_winners

    plane, result = node.arrays[:2]
    dots = tangents.of(plane), tangents.of(result)
    if any(d is None for d in dots):
        return []
    token = graph_builder._ACTIVE.set(capture)
    try:
        _, where = _scan_with_winners(plane, node.op, node.exclusive)
    finally:
        graph_builder._ACTIVE.reset(token)
    return _gathered(plane, where, dots, capture, cache)


def _gathered(
    plane: Any, where: Any, dots: tuple, capture: _Capture, cache: dict
) -> list:
    """The nodes recorded so far in ``capture`` (which made ``where``) and the launch that writes ``dots[1]`` as the tangent
    ``dots[0]`` of ``plane`` read at the lanes ``where`` names."""
    from ..graph import _Launch
    from ..plans import deploy

    scalar = str(plane.dtype)
    if ("gather", scalar) not in cache:
        # built as the program's own kernels were, when it has any, for the plane's type
        settings = {**cache["settings"], "scalar_type": scalar, "reverse": False}
        cache[("gather", scalar)] = deploy(_gather_kernel(), **settings)
    made, capture.nodes = capture.nodes, []
    launch = _Launch(
        cache[("gather", scalar)].bind(
            table=dots[0], where=where, count=float(plane.shape[0]), y=dots[1]
        ),
        "winner tangent",
    )
    return [*made, launch]


def _gather_kernel() -> Any:
    """The kernel that reads the plane ``table`` at the lane ``where`` names; a lane at or beyond ``count`` (no winner) gives zero."""
    import hawk
    from .. import Index, Mutable, Param, Scalar, Table, kernel

    @kernel
    def winner_tangent(
        table: Table[Scalar], where: Index, count: Param, y: Mutable[Scalar]
    ):
        y = hawk.math.where(where < count, table.at(where), 0.0)

    return winner_tangent


def _tangent_nodes(
    node: Any, tangents: _Tangents, capture: _Capture, cache: dict
) -> list:
    """The whole-plane nodes that carry ``node``'s tangent, in order, after ``node``.

    A node is called again over the tangent arrays of its own: the operation is linear in them (a fold of a sum, a scan of a sum,
    a move, a copy, a transform), and a product is the sum of two products."""
    kind, arrays = node.kind, node.arrays
    if kind in ("compact", "sort") or node.op in ("all", "any"):
        return []  # integer or mask planes only: no tangent
    if node.kind == "scan" and node.op in ("min", "max"):
        return _running_winner_tangent(node, tangents, capture, cache)
    if node.op in ("min", "max"):
        return _winner_tangent(node, tangents, capture, cache)
    dots = [tangents.of(a) for a in arrays]
    token = graph_builder._ACTIVE.set(capture)
    try:
        if kind == "reorder":
            if any(d is not None for d in dots[:-1]):
                node.redo([*(d for d in dots[:-1] if d is not None), arrays[-1]])
        elif kind == "matmul":
            if dots[2] is not None:
                node.redo([dots[0], arrays[1], dots[2]])
                node.redo([arrays[0], dots[1], dots[2]], accumulate=True)
        elif dots[0] is not None and dots[1] is not None:
            node.redo(dots[:2])
    finally:
        graph_builder._ACTIVE.reset(token)
    made, capture.nodes = capture.nodes, []
    return made


def _items(items: list, tangents: _Tangents, cache: dict, capture: Any) -> list:
    """``items`` re-recorded with tangents: the same structure, each launch the tangent-carrying launch, each node followed by
    its tangent nodes."""
    from ..collectives import Collective
    from ..graph import _Cond, _Fork, _HostCall, _Launch, _Wait

    out: list = []
    for item in items:
        if isinstance(item, _Launch):
            out.append(_tangent_launch(item, tangents, cache))
        elif isinstance(item, Collective):
            out.append(item)
            out.extend(_tangent_nodes(item, tangents, capture, cache))
        elif isinstance(item, _Cond):
            out.append(
                dataclasses.replace(
                    item,
                    arms=[_items(arm, tangents, cache, capture) for arm in item.arms],
                )
            )
        elif isinstance(item, _Fork):
            out.append(
                _Fork(
                    [
                        _items(branch, tangents, cache, capture)
                        for branch in item.branches
                    ]
                )
            )
        elif isinstance(item, _Wait):
            waited = [t for a in item.arrays if (t := tangents.of(a)) is not None]
            out.append(_Wait((*item.arrays, *waited)))
        elif isinstance(item, _HostCall):
            raise HawkError(
                f"hawk.diff.jvp: the host step {item.name!r} has no tangent; write it as a kernel"
            )
        else:
            out.append(item)
    return out


def jvp_builder(step: Callable, wrt: Any) -> Callable:
    """The builder of the forward derivative of the builder ``step``: called as ``step`` is, with the tangent of any array
    parameter in ``tangents={name: array}``, it returns the built graph that replays the program with tangents."""
    _refuse_unless_builder("jvp", step)
    if wrt is not None:
        raise HawkError(
            "hawk.diff.jvp: a builder is differentiated along the arrays given in tangents=, not wrt="
        )

    def tangent_step(*args: Any, tangents: dict | None = None, **kwargs: Any) -> Any:
        from ..graph import Graph

        graph = step(*args, **kwargs)
        params = graph._params or {}
        given = tangents or {}
        unknown = sorted(set(given) - set(params))
        if unknown:
            raise HawkError(
                f"hawk.diff.jvp: tangents= names {unknown}, which are not parameters of {step.__name__!r}; "
                f"its parameters are {sorted(params)}"
            )
        known = _Tangents([(params[name], array) for name, array in given.items()])
        capture = _Capture(graph._store_over)
        result = Graph()
        result._share_planes(graph)
        result._items.extend(
            _items(
                graph._items, known, {"settings": _settings_of(graph._items)}, capture
            )
        )
        result._params = params
        result.build()
        result.tangent_of = known.of
        result.tangents = {
            name: t
            for name, value in params.items()
            if (t := known.of(value)) is not None
        }
        return result

    return tangent_step


def vjp_builder(step: Callable, wrt: Any) -> Callable:
    """The step of the reverse derivative of the builder ``step``: called as ``step`` is, plus ``cotangents={parameter: array}``,
    it replays the program once and returns the gradients :func:`hawk.reverse` gives for the recorded run.

    ``cotangents`` and ``wrt`` name the builder's array parameters; the gradients come back by plane name, as from
    :func:`hawk.reverse`."""
    _refuse_unless_builder("vjp", step)

    def gradient_step(
        *args: Any, cotangents: dict, strict: bool = False, **kwargs: Any
    ) -> Any:
        from ..graph import _planes_of, _reverse_graph

        graph = step(*args, **kwargs)
        if graph.record is None:
            raise HawkError(
                f"hawk.diff.vjp: {step.__name__!r} makes a graph that keeps no record; make it with hawk.Graph(record=True)"
            )
        params = graph._params or {}
        planes, _ = _planes_of(graph)

        def plane_of(name: str, what: str) -> Any:
            if name not in params:
                raise HawkError(
                    f"hawk.diff.vjp: {what} names {name!r}, which is not a parameter of {step.__name__!r}; "
                    f"its parameters are {sorted(params)}"
                )
            return next((p for p in planes if p.given is params[name]), None)

        named, loose = {}, []
        for name, cotangent in cotangents.items():
            plane = plane_of(name, "cotangents=")
            if plane is None:
                loose.append((params[name], cotangent))
            else:
                named[f"{plane.label}.{plane.name}"] = cotangent
        wanted, asked = None, []
        if wrt is not None:
            wanted = []
            for name in [wrt] if isinstance(wrt, str) else wrt:
                plane = plane_of(name, "wrt=")
                if plane is None:
                    asked.append((name, params[name]))
                else:
                    wanted.append(f"{plane.label}.{plane.name}")
        graph()
        return _reverse_graph(graph, named, wanted, strict, tuple(loose), tuple(asked))

    return gradient_step
