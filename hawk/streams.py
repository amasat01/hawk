# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.Stream``: an ordered queue of one device and the scope that puts launches on it.

A stream is either made by the library (``hawk.Stream(device="cuda:0")``: the library owns the queue, hawk holds its opaque
handle and frees it) or adopted (``hawk.Stream(theirs)``: the caller's raw stream, which stays the caller's). ``with s:`` puts
every launch made inside it on ``s``: a kernel call, a whole-plane operation, a graph replay. Which device a stream belongs to
is the runtime's to find; hawk carries the handle and the scope and decides nothing about a target.
"""

from __future__ import annotations

import contextvars
import threading
import weakref
from typing import Any

from . import _place, graph_builder
from .errors import HawkError
from .runtime import through_door

__all__ = ["Stream", "apply", "door_in_scope"]

#: The scope this context is in: ``(stream, outer scope, open)``, or ``None`` outside every scope. ``open`` is set while the
#: thread that entered the scope is inside it; a copy of the context made for another thread shares it, so that leaving the
#: scope is seen wherever the scope was inherited.
_SCOPE: contextvars.ContextVar = contextvars.ContextVar(
    "hawk_stream_scope", default=None
)


def _current() -> tuple | None:
    """The innermost scope still open, else ``None``. A scope the calling thread inherited and whose starter has left is not in
    scope any more: the doors this thread stated for it are put back here, as leaving would have."""
    scope = _SCOPE.get()
    left = []
    while scope is not None and not scope[2].is_set():
        left.append(scope[0])
        scope = scope[1]
    for stream in left:
        stream._put_back(scope)
    return scope


def apply(door: Any) -> None:
    """State the stream of the scope to ``door`` on the calling thread, at call time, before every launch a door queues. The
    scope is read on the calling thread and the door keeps its stream in use per thread, so a launch made on another thread is
    placed by that thread's own scope. Outside every scope there is nothing to state: leaving a scope has already put back the
    doors its thread used."""
    scope = _current()
    if scope is None:
        return
    through_door(door.stream_use, scope[0]._handle)
    scope[0]._doors.add(door)
    scope[0]._stated_here().add(door)


def door_in_scope() -> Any:
    """The door the stream in scope launches on, else the host's door (where a call with no array and no stream runs).

    A stream knows its door once it has launched through it, and a stream the library made knows it from the start; a stream
    adopted and not yet launched on, or one launched on over several doors, has no one door and is refused by name."""
    scope = _current()
    if scope is None:
        return _place.door_of(_place.HOST)
    doors = scope[0]._doors
    if len(doors) != 1:
        raise HawkError(
            "hawk.Stream: the stream in scope is on "
            + (
                "no door yet (launch on it once, or make it with device=)"
                if not doors
                else "several doors"
            )
            + ", so the door of this call is not known"
        )
    return next(iter(doors))


def _free(door: Any, handle: int) -> None:
    """Release the library's queue ``handle`` on ``door``; the door puts its own stream back in scope if it was the one."""
    through_door(door.stream_free, handle)


def _door_of_device(device: str) -> Any:
    """The door of the device ``device`` names (``"<target>:<ordinal>"``, as :func:`hawk.asarray` reads it); the text itself goes
    on to the door unparsed."""
    # plans imports the runtime, which calls apply: the import is made here to break that cycle.
    from . import plans

    served, ordinal = plans._device_target(device, "hawk.Stream")
    return through_door(_place.open_door, served.id, ordinal)


def _raw_handle(theirs: Any) -> int:
    """The raw stream handle of ``theirs``: an integer, or the handle an object states by ``__cuda_stream__``."""
    if isinstance(theirs, int) and not isinstance(theirs, bool):
        return theirs
    if hasattr(theirs, "__cuda_stream__"):
        return int(theirs.__cuda_stream__()[1])
    raise HawkError(
        f"hawk.Stream: {type(theirs).__name__} is neither a raw stream handle (an int) nor an object with __cuda_stream__"
    )


class Stream:
    """An ordered queue of one device. ``Stream(device="cuda:0")`` makes one the library owns, freed by :meth:`close` or when the
    object is collected; ``Stream(theirs)`` adopts a raw handle or an object with ``__cuda_stream__``, which the caller keeps
    alive and frees. ``with stream:`` puts every launch made inside on it; scopes nest per thread and the outer one is back on
    exit, an exception included. A scope cannot be opened inside a graph builder or block: there, order is ``parallel()`` and
    ``wait(planes)``."""

    def __init__(self, theirs: Any = None, *, device: str | None = None) -> None:
        """Adopt ``theirs``, or make a stream the library owns on ``device``; exactly one is given. The library's reason for a
        refusal (a host-only context has one queue) is raised as given."""
        if (theirs is None) == (device is None):
            raise HawkError(
                "hawk.Stream: give theirs (a stream to adopt) or device= (a stream to make), not both and not neither"
            )
        self._doors: set = set()
        self._stated = threading.local()
        self._finalizer = None
        if theirs is not None:
            self._handle = _raw_handle(theirs)
            return
        door = _door_of_device(device)
        self._handle = through_door(door.stream_new, device)
        self._doors.add(door)
        self._finalizer = weakref.finalize(self, _free, door, self._handle)

    def __enter__(self) -> Stream:
        """Put the launches made inside the ``with`` on this stream."""
        if graph_builder.current() is not None:
            raise HawkError(
                "hawk.Stream: a stream scope cannot be opened inside a recording; there the queues are derived from "
                "parallel() branches and wait(planes)"
            )
        opened = threading.Event()
        opened.set()
        _SCOPE.set((self, _SCOPE.get(), opened))
        return self

    def _stated_here(self) -> set:
        """The doors this stream was stated to on the calling thread: only those are put back when that thread leaves the scope,
        because a door keeps its stream in use per thread."""
        return self._stated.__dict__.setdefault("doors", set())

    def __exit__(self, *exc: object) -> None:
        """Leave the scope: the outer stream is in scope again on every door this thread stated this stream to, also after an
        exception."""
        _, outer, opened = _SCOPE.get()
        opened.clear()
        _SCOPE.set(outer)
        self._put_back(_current())

    def _put_back(self, outer: tuple | None) -> None:
        """State ``outer``'s stream (the library's own when there is none) on the doors this stream was stated to on the calling
        thread."""
        back = 0 if outer is None else outer[0]._handle
        for door in self._stated_here():
            through_door(door.stream_use, back)
            if outer is not None:
                outer[0]._doors.add(door)
                outer[0]._stated_here().add(door)
        self._stated_here().clear()

    def synchronize(self) -> None:
        """Return when everything queued on this stream has run. A stream nothing was launched on has nothing to wait for."""
        for door in self._doors:
            through_door(door.stream_sync, self._handle)

    def close(self) -> None:
        """Free a stream the library made at once, after the work queued on it; an adopted stream is the caller's and stays.
        Closing twice is harmless."""
        if self._finalizer is not None:
            self._finalizer()
