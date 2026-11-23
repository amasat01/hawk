# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Groups of lanes: ``with hawk.group(...) as g:`` and what the body of the scope holds.

A group is a set of lanes. The scope states WHAT the group is (which lanes belong to it, the scratch it declares, where its
barriers are); the library computes everything else, the phases between barriers included. The scope collects the sinks the
body makes, in the order the author wrote them, and closing it builds ONE ``Group`` node over them: a body statement that
commits an output plane or stores into a scratch plane becomes a sink of the group, a statement outside the scope stays an
ordinary sink beside it. Typing and every refusal about the group itself are the library's; a refusal reaches the author
unchanged, with the kernel's name in front.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .. import _ir
from ..errors import HawkError
from .lane_axes import EDGES, _trace, current_context
from .value import Value, index_node, node_of

_REDUCE_ORDERS = ("tree", "lane", "any")


def open_scope() -> GroupScope | None:
    """The group scope the traced body is inside of, or ``None`` (read by the output planes, which commit into it)."""
    return getattr(_trace, "group", None)


def _kernel_name() -> str | None:
    """The kernel being traced."""
    return getattr(_trace, "name", None)


def _named(refusal: HawkError) -> HawkError:
    """``refusal`` with the traced kernel's name in front, the way the other elements state theirs."""
    name = _kernel_name()
    return refusal if name is None else HawkError(f"kernel {name!r}: {refusal}")


@contextmanager
def _library() -> Iterator[None]:
    """Let the library's refusal of what is built inside reach the author with the kernel's name in front."""
    try:
        yield
    except HawkError as refusal:
        raise _named(refusal) from refusal


def group(
    *,
    lanes: int | None = None,
    over: Any = None,
    key: Any = None,
    max_size: int | None = None,
) -> GroupScope:
    """The scope of a group of lanes: ``with hawk.group(lanes=8) as g: ...``.

    Exactly one of three says which lanes make one group. ``lanes=N``: N consecutive lanes. ``over=("col",)``: the lanes
    along the named axes of the kernel's ``Grid`` (a label, or a tuple of labels); the other axes tell the groups apart.
    ``key=k``: the lanes whose integer value ``k`` is equal; ``max_size=N`` bounds the lanes of one such group and is needed
    when the body declares scratch. Anything else is refused by name here; whether the facts make a group (a lane count of
    zero, an axis the kernel does not have, a scratch larger than the group) is the library's refusal, raised when the
    scope closes.

    Inside the scope: ``g.rank`` (this lane's place in its group) and ``g.size`` (the lanes of its group) are values;
    ``g.scratch(name, type)`` declares a scratch plane; ``g.barrier()`` separates two phases; ``g.sum / min / max / any /
    all`` combine one value per lane over the group. Outputs assigned in the scope are sinks of the group. A kernel holds
    one group; a group sits at the top level of the body (not inside an ``if`` or a ``for``) and holds no branch or loop."""
    given = {"lanes": lanes, "over": over, "key": key}
    stated = [name for name, value in given.items() if value is not None]
    if len(stated) != 1:
        raise _named(
            HawkError(
                "hawk.group takes exactly one of lanes=, over= or key= (which lanes make a group); "
                + (f"got {' and '.join(stated)}" if stated else "got none")
            )
        )
    return GroupScope(lanes, over, key, max_size)


class GroupScope:
    """The open scope of one group: the handle ``g`` of ``with hawk.group(...) as g``. Made by :func:`group`."""

    def __init__(self, lanes: Any, over: Any, key: Any, max_size: Any) -> None:
        self._membership = (
            "count" if lanes is not None else "axes" if over is not None else "key"
        )
        self._count = self._whole("lanes", lanes)
        self._axes = self._labels(over)
        self._key = None if key is None else node_of(key)
        self._max_size = self._whole("max_size", max_size)
        self._body: list = []
        self._scratch: list = []
        self._written: list = []
        self._state = "new"

    @staticmethod
    def _whole(what: str, value: Any) -> int | None:
        """``value`` (a lane count or a bound) as a whole number, or ``None`` when not stated."""
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            return value
        raise _named(
            HawkError(
                f"hawk.group({what}=...) is a whole number of lanes, got {value!r}"
            )
        )

    @staticmethod
    def _labels(over: Any) -> tuple:
        """The axis labels ``over=`` names: one label or a tuple of labels."""
        if over is None:
            return ()
        labels = (
            (over,)
            if isinstance(over, str)
            else tuple(over)
            if isinstance(over, (tuple, list))
            else None
        )
        if labels is None or not all(isinstance(label, str) for label in labels):
            raise _named(
                HawkError(
                    f"hawk.group(over=...) names Grid axes by label, a label or a tuple of them, got {over!r}"
                )
            )
        return labels

    # ---- the scope ----

    def __enter__(self) -> GroupScope:
        self._context = current_context()
        self._outer = open_scope()
        _trace.group = self
        self._state = "open"
        return self

    def __exit__(self, kind: Any, error: Any, trace: Any) -> bool:
        _trace.group = self._outer
        self._state = "closed"
        if error is not None:
            return False
        group = self._build()
        if self._outer is not None:
            self._outer._body.append(group)
            self._outer._written.extend(self._written)
        else:
            held = set(self._body)
            for ref in self._written:
                ref.sinks = [*(k for k in ref.sinks if k not in held), group]
        return False

    def _build(self) -> Any:
        """The ``Group`` node of the scope, typed by the library."""
        axes = self._lane_axes()
        grid = (
            tuple(axes) if self._membership == "axes" and axes[0][0] is not None else ()
        )
        with _library():
            return _ir.group(
                self._context,
                self._membership,
                self._body,
                self._scratch,
                key=self._key,
                count=self._count,
                grid=grid,
                axes=self._axes,
                max_size=self._max_size,
            )

    @staticmethod
    def _lane_axes() -> tuple:
        """The lane axes of the traced kernel, ``((label|None, extent|None), ...)``."""
        return getattr(_trace, "axes", ((None, None),))

    def _live(self, what: str) -> None:
        """Refuse ``what`` unless the scope is open."""
        if self._state != "open":
            raise _named(
                HawkError(
                    f"{what}: this group's scope has "
                    + ("not started" if self._state == "new" else "ended")
                    + "; a group's rank, size, scratch and reductions are used inside its `with hawk.group(...)` block"
                )
            )

    def _record(self, sink: Any) -> Any:
        """``sink`` as the next sink of the group's body, in the author's order."""
        self._body.append(sink)
        return sink

    def adopt(self, ref: Any, sink: Any) -> None:
        """Take ``sink``, an update of the plane ``ref`` the body just wrote by key, as the next sink of the body: it
        runs in the group, in the author's order."""
        self._record(sink)
        if ref not in self._written:
            self._written.append(ref)

    def commit(self, ref: Any, value: Any) -> Any:
        """The commit of ``value`` to the lane's own element of the output plane ``ref``, as the next sink of the body: the sink an
        output has outside a scope."""
        if ref.axes:
            raise _named(
                HawkError(
                    f"{ref.name!r}: a group commits the lane's own element of an output plane, so the plane declares no named axes"
                )
            )
        sink = self._record(self._ordinary_sink(ref, value))
        self._written.append(ref)
        return sink

    @staticmethod
    def _ordinary_sink(ref: Any, value: Any) -> Any:
        """The sink the kernel builds for ``ref = value`` outside a scope: the plane's own ``write`` builds it, with this scope
        closed for the moment so that the write is not handed back to the scope."""
        outer = _trace.group
        _trace.group = None
        try:
            with _library():
                ref.write(value)
        finally:
            _trace.group = outer
        (sink,) = ref.sinks
        return sink

    # ---- what the body uses ----

    @property
    def rank(self) -> Value:
        """This lane's place in its group (an index from 0)."""
        self._live("g.rank")
        return Value(_ir.group_element(self._context, "GroupRank"))

    @property
    def size(self) -> Value:
        """The number of lanes of this lane's group."""
        self._live("g.size")
        return Value(_ir.group_element(self._context, "GroupSize"))

    def scratch(self, name: str, ttype: Any) -> ScratchPlane:
        """Declare the scratch plane ``name`` of this group: one element of type ``ttype`` (``Scalar``, ``Vector[3]``, ...)
        for each place of the group, read and written by place (``t[g.rank] = v``, ``t[g.rank]``, ``t.at(...)``). The group
        holds it only while it runs; the number of places is the group's lane count (``lanes=``, the extent of the ``over=``
        axes, or ``max_size=``)."""
        self._live("g.scratch")
        if not isinstance(name, str) or not isinstance(ttype, _ir.TensorType):
            raise _named(
                HawkError(
                    f"g.scratch(name, type) takes a name and a type (Scalar, Vector[W], ...), got {name!r}, {ttype!r}"
                )
            )
        plane = ScratchPlane(self, name, ttype, self._places())
        self._scratch.append(plane.leaf)
        return plane

    def _places(self) -> int | None:
        """The number of places a scratch plane of this group holds: what the membership states, ``None`` when it states none."""
        if self._membership == "count":
            return self._count
        if self._membership == "key":
            return self._max_size
        extents = [extent for label, extent in self._lane_axes() if label in self._axes]
        return None if not extents or None in extents else _product(extents)

    def barrier(self) -> None:
        """End the phase here: every lane of the group reaches it before any goes on, so what a lane stored into scratch
        before it is visible to every lane after it."""
        self._live("g.barrier")
        self._record(_ir.group_element(self._context, "Barrier"))

    def _reduce(self, op: str, value: Any, order: str) -> Value:
        self._live(f"g.{op}")
        with _library():
            return Value(_ir.group_reduce(op, node_of(value), order))

    def sum(self, value: Any, order: str = "tree") -> Value:
        """The sum of ``value`` over the lanes of the group, the same value on every lane, added in ``order``: ``"tree"``
        (a fixed tree, the same bits on every run), ``"lane"`` (one by one in rank order) or ``"any"`` (the library's choice).
        The result is available after the end of the phase that evaluates ``value``."""
        if order not in _REDUCE_ORDERS:
            raise _named(
                HawkError(
                    f"g.sum(order={order!r}): the order is one of {_REDUCE_ORDERS}"
                )
            )
        return self._reduce("sum", value, order)

    def min(self, value: Any) -> Value:
        """The minimum of ``value`` over the lanes of the group (exact; no order to state)."""
        return self._reduce("min", value, "tree")

    def max(self, value: Any) -> Value:
        """The maximum of ``value`` over the lanes of the group (exact; no order to state)."""
        return self._reduce("max", value, "tree")

    def any(self, flag: Any) -> Value:
        """Whether ``flag`` (a bool) holds on any lane of the group."""
        return self._reduce("any", flag, "tree")

    def all(self, flag: Any) -> Value:
        """Whether ``flag`` (a bool) holds on every lane of the group."""
        return self._reduce("all", flag, "tree")


def _product(items: list) -> int:
    out = 1
    for item in items:
        out *= item
    return out


class ScratchPlane:
    """A scratch plane of a group, made by ``g.scratch``: one element per place of the group, indexed by place.

    ``t[g.rank] = v`` stores, ``t[place]`` reads (an index outside the plane is clamped to it), ``t.at(place, edge=...)``
    reads under an edge rule. Another lane's element is read only after a ``g.barrier()`` that follows the store."""

    __slots__ = ("_scope", "name", "ttype", "axes", "leaf")

    def __init__(
        self, scope: GroupScope, name: str, ttype: Any, places: int | None
    ) -> None:
        self._scope = scope
        self.name = name
        self.ttype = ttype
        self.axes = (("rank", places),)
        with _library():
            self.leaf = _ir.scratch(scope._context, name, ttype, self.axes)

    def _live(self, what: str) -> None:
        if self._scope._state != "open":
            raise _named(
                HawkError(
                    f"scratch {self.name!r} {what}: its group's scope has ended; a scratch plane is used inside its own "
                    "`with hawk.group(...)` block"
                )
            )

    def __setitem__(self, place: Any, value: Any) -> None:
        """Store ``value`` into the element at ``place`` (usually ``g.rank``)."""
        self._live("is stored")
        typed = Value(node_of(value))
        with _library():
            sink = _ir.update(
                self.name,
                typed.node,
                indices=[index_node(place)],
                op="store",
                axes=self.axes,
                role="scratch",
            )
        self._scope._record(sink)

    def __getitem__(self, place: Any) -> Value:
        """The element at ``place``."""
        return self.at(place)

    def at(self, place: Any, edge: str = "clamp") -> Value:
        """The element at ``place`` under the edge rule ``edge`` for a place outside the plane: ``"clamp"`` (the nearest
        element), ``"wrap"`` (around) or ``"zero"`` (zero)."""
        self._live("is read")
        if edge not in EDGES:
            raise _named(
                HawkError(
                    f"{self.name!r}.at(edge={edge!r}): the edge rule is one of {EDGES}"
                )
            )
        with _library():
            return Value(
                _ir.make_node(
                    "At", self.ttype, [self.leaf, index_node(place)], edge=edge
                )
            )
