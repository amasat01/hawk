# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Your own Python step in a graph: :func:`host_step` wraps a function so a graph records it beside kernel launches, and
:meth:`HostStep.reverse` gives it a gradient.

The library runs the function where a launch would run and hands it the memory of the arrays the step was bound to; hawk only
wraps the function so that its arguments arrive by name, as arrays of the library the caller's own arrays come from, and checks
what the function returns. Where and when the step runs, and how a gradient is accumulated, are the library's.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import NamedTuple

import _osprey

from . import _place, graph_builder
from .errors import HawkError
from .runtime import through_door
from .trace.decl import Mutable, Plane

__all__ = ["host_step"]

_READ = _osprey.EG_ACCESS_READ
_READ_WRITE = _osprey.EG_ACCESS_READ | _osprey.EG_ACCESS_WRITE


class _Argument(NamedTuple):
    """One argument of a step: its name and whether the step may write it (``Mutable``) or only read it."""

    name: str
    writes: bool


def _arguments_of(function: Callable) -> tuple:
    """The :class:`_Argument` of every parameter of ``function``, marked by its annotation: ``hawk.Mutable``, bare or
    subscripted, may be written and read; no mark is read-only. Parameters that cannot be bound by name are refused."""
    name = function.__name__
    try:
        marks = typing.get_type_hints(function)
    except Exception as failure:
        raise HawkError(
            f"host_step {name}: an annotation cannot be read ({failure}); annotate with hawk.Mutable or leave the "
            "argument unmarked"
        ) from failure
    arguments = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind not in (
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY,
        ):
            raise HawkError(
                f"host_step {name}: argument {parameter.name!r} cannot be bound by name; a step takes plain "
                "named arguments"
            )
        mark = marks.get(parameter.name)
        writes = mark is Mutable or (isinstance(mark, Plane) and mark.form == "mutable")
        arguments.append(_Argument(parameter.name, writes))
    return tuple(arguments)


class _Arrays:
    """The arrays of one role (``value``, ``cotangent`` or ``gradient``) of a reverse step, by argument name. A name the role
    does not hold is refused by name; an in-place update (``gradient.x += ...``) is accepted, a replacement is not."""

    __slots__ = ("__dict__", "_where")

    def __init__(self, where: str, arrays: dict) -> None:
        """Hold ``arrays``; ``where`` names the step and the role for refusals."""
        object.__setattr__(self, "_where", where)
        self.__dict__.update(arrays)

    def __getattr__(self, name: str):
        """Refuse a name that is not an argument of this role."""
        raise HawkError(
            f"{self._where}.{name} does not exist; this role holds {sorted(self.__dict__)}"
        )

    def __setattr__(self, name: str, value) -> None:
        """Accept ``name = value`` only when ``value`` is the array already there, which is what ``+=`` on an array does."""
        if self.__dict__.get(name) is not value:
            raise HawkError(
                f"{self._where}.{name}: write into the array (`{name}[...] = ...` or `+=`); the array itself is "
                "the library's and cannot be replaced"
            )


class HostStep:
    """A Python function a graph records as one step; what :func:`host_step` returns.

    Call it with the arrays to bind it, as keywords or positions; every argument is given.
    """

    def __init__(self, function: Callable) -> None:
        """Read the marks of ``function``'s arguments."""
        self.function = function
        self.name = function.__name__
        self.arguments = _arguments_of(function)
        self.reverse_function: Callable | None = None

    @property
    def reverses(self) -> bool:
        """Whether a gradient was given to this step."""
        return self.reverse_function is not None

    def reverse(self, function: Callable) -> Callable:
        """Give the step its gradient: ``function(value, cotangent, gradient)`` adds to ``gradient.<argument>`` for every
        argument the step may read (``value.<argument>`` holds the arrays as the step saw them, ``cotangent.<argument>`` the
        adjoint of each ``Mutable`` argument). The library hands zeroed ``gradient`` arrays and accumulates them itself.
        Returns ``function``."""
        if self.reverse_function is not None:
            raise HawkError(
                f"host_step {self.name}: it has a reverse already; give one"
            )
        self.reverse_function = function
        return function

    def __call__(self, *positional, **named):
        """Bind the step to arrays: a :class:`BoundHostStep` for :meth:`hawk.Graph.add`, or, inside a graph builder, record the
        step and return the arrays it writes (one array, a tuple of them, or ``None`` when it has no ``Mutable`` argument)."""
        names = [argument.name for argument in self.arguments]
        if len(positional) > len(names):
            raise HawkError(
                f"host_step {self.name}: takes {len(names)} arguments {names}, got {len(positional)} positional"
            )
        given = dict(zip(names, positional))
        for name, value in named.items():
            if name not in names:
                raise HawkError(
                    f"host_step {self.name}: {name!r} is not an argument; they are {names}"
                )
            if name in given:
                raise HawkError(
                    f"host_step {self.name}: argument {name!r} is given twice"
                )
            given[name] = value
        missing = [name for name in names if name not in given]
        if missing:
            raise HawkError(
                f"host_step {self.name}: give every argument; missing {missing}"
            )
        bound = BoundHostStep(self, {name: given[name] for name in names})
        recorder = graph_builder.current()
        if recorder is None:
            return bound
        recorder.record_step(bound)
        written = [
            bound.planes[argument.name]
            for argument in self.arguments
            if argument.writes
        ]
        return written[0] if len(written) == 1 else tuple(written) or None


class BoundHostStep:
    """A :class:`HostStep` bound to arrays: what :meth:`hawk.Graph.add` takes. ``planes`` holds the arrays by argument name."""

    def __init__(self, step: HostStep, arrays: dict) -> None:
        """Check that every argument is an array of one place, and keep them."""
        for name, array in arrays.items():
            if _place.device_of(array) is None:
                raise HawkError(
                    f"host_step {step.name}: argument {name!r} is a {type(array).__name__}, not an array; bind "
                    "arrays (anything that exports DLPack)"
                )
        found = _place.place_of(arrays, f"host_step {step.name}")
        if found is None:
            raise HawkError(
                f"host_step {step.name}: it has no argument to bind; a step reads or writes at least one array"
            )
        device, like = found
        self.step = step
        self.planes = arrays
        self.door = _place.door_of(device)
        # the array library the caller's arrays come from reads the library's views
        self._from_dlpack = _place.reader(like)

    @property
    def views(self) -> dict:
        """The arrays the library reads, by argument name: the bound arrays themselves."""
        return self.planes

    def record(self, door, store) -> None:
        """Record the step into the recording ``door`` has open, over the planes of ``store``. The recording keeps the node
        alive, so the handle is freed at once."""
        step = self.step
        node = through_door(
            door.host_node,
            step.name,
            self._run,
            self._reverse if step.reverses else None,
        )
        try:
            through_door(
                door.host_call,
                node,
                [store.plane(self.planes[a.name]) for a in step.arguments],
                [_READ_WRITE if a.writes else _READ for a in step.arguments],
            )
        finally:
            door.host_node_free(node)

    def _arrays(self, views: list, arguments) -> dict:
        """``{argument: array}`` of the library's ``views`` (one per argument, ``None`` where it gives none)."""
        return {
            argument.name: self._from_dlpack(view)
            for argument, view in zip(arguments, views)
            if view is not None
        }

    def _run(self, views: list) -> None:
        """What the library calls where the step runs: call the function with its arrays and copy what it returns into the
        ``Mutable`` arguments."""
        step = self.step
        arrays = self._arrays(views, step.arguments)
        self._take(step.function(**arrays), arrays)

    def _take(self, returned, arrays: dict) -> None:
        """Copy the values ``returned`` into the ``Mutable`` arrays in their order, refusing a wrong count, shape or dtype."""
        if returned is None:
            return
        step = self.step
        targets = [argument.name for argument in step.arguments if argument.writes]
        if not targets:
            raise HawkError(
                f"host_step {step.name}: it returned a value but has no Mutable argument to receive it"
            )
        values = list(returned) if isinstance(returned, (tuple, list)) else [returned]
        if len(values) != len(targets):
            raise HawkError(
                f"host_step {step.name}: it returned {len(values)} values for the Mutable arguments {targets}; "
                "return one array each, in order"
            )
        for name, value in zip(targets, values):
            held = arrays[name]
            if (
                tuple(getattr(value, "shape", ())) != tuple(held.shape)
                or getattr(value, "dtype", None) != held.dtype
            ):
                raise HawkError(
                    f"host_step {step.name}: the value returned for {name!r} has shape "
                    f"{tuple(getattr(value, 'shape', ()))} and dtype {getattr(value, 'dtype', None)}; the argument "
                    f"is {tuple(held.shape)} {held.dtype}"
                )
            held[...] = value

    def _reverse(self, primals: list, adjoints_in: list, adjoints_out: list) -> None:
        """What the library calls to reverse the step: map its three lists onto ``value``, ``cotangent`` and ``gradient`` by
        argument name and call the step's reverse."""
        step = self.step
        written = {argument.name for argument in step.arguments if argument.writes}
        where = f"host_step {step.name} reverse"
        adjoints = {
            k: v
            for k, v in self._arrays(adjoints_in, step.arguments).items()
            if k in written
        }
        step.reverse_function(
            _Arrays(f"{where}: value", self._arrays(primals, step.arguments)),
            _Arrays(f"{where}: cotangent", adjoints),
            _Arrays(f"{where}: gradient", self._arrays(adjoints_out, step.arguments)),
        )


def host_step(function: Callable) -> HostStep:
    """Make ``function`` a step a graph records: ``@hawk.host_step``.

    An argument annotated ``hawk.Mutable`` (bare or subscripted) may be written and read first; any other argument is read-only
    and arrives as a read-only array. The step may return the values of its ``Mutable`` arguments, in order, instead of writing
    them. Bind it by calling it with arrays, ``double(x=a, y=b)``, and add the result to a :class:`hawk.Graph`, or call it inside
    a graph builder. ``@double.reverse`` gives it a gradient.
    """
    return HostStep(function)
