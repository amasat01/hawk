# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.Event``: a mark on a stream that another stream waits for and that gives the time between two marks.

``e.record()`` marks the stream in scope, ``e.wait()`` makes the stream in scope wait for the mark, and
``stop.elapsed_ms(since=start)`` is the time between two recorded marks. An event is a mark of the eager level: inside a graph
builder or block order is ``wait(planes)`` and time is taken around the replay. The library makes and answers every event; hawk
holds the opaque handle and the door it was made on.
"""

from __future__ import annotations

import weakref
from typing import Any

from . import graph_builder, streams
from .errors import HawkError
from .runtime import through_door

__all__ = ["Event"]


def _refuse_in_recording(what: str) -> None:
    """Refuse ``what`` by name while a graph builder or block is recording."""
    if graph_builder.current() is not None:
        raise HawkError(
            f"hawk.Event.{what}: an event cannot be used inside a recording; there the order is wait(planes) and the time is "
            "taken around the replay"
        )


class Event:
    """A mark on a stream. It is made on the door of the stream in scope at its first :meth:`record` (the host's door with no
    stream in scope) and stays on it; it is freed by :meth:`close` or when the object is collected."""

    def __init__(self) -> None:
        """Make an event that has not been recorded."""
        self._door: Any = None
        self._handle: Any = None
        self._finalizer = None

    def record(self) -> None:
        """Mark the stream in scope: the mark is reached when that stream has run what was queued on it before this call."""
        _refuse_in_recording("record")
        if self._finalizer is not None and not self._finalizer.alive:
            raise HawkError("hawk.Event.record: the event is closed; make a new one")
        door = streams.door_in_scope()
        if self._door is None:
            self._door = door
            self._handle = through_door(door.event_new)
            self._finalizer = weakref.finalize(self, door.event_free, self._handle)
        elif self._door is not door:
            raise HawkError(
                "hawk.Event.record: the event was recorded on another device; use one event per device"
            )
        streams.apply(door)
        through_door(door.event_record, self._handle)

    def wait(self) -> None:
        """Make the stream in scope wait, at this point of its queue, for the mark; the calling thread does not wait."""
        _refuse_in_recording("wait")
        self._require_recorded("wait", "the event")
        door = streams.door_in_scope()
        if self._door is not door:
            raise HawkError(
                "hawk.Event.wait: the stream in scope is on another device than the event was recorded on"
            )
        streams.apply(door)
        through_door(door.event_wait, self._handle)

    def elapsed_ms(self, since: Event) -> float:
        """The milliseconds from the mark of ``since`` to this event's mark, waiting for both to be reached first."""
        _refuse_in_recording("elapsed_ms")
        self._require_recorded("elapsed_ms", "the stop event")
        since._require_recorded("elapsed_ms", "the start event (since=)")
        if self._door is not since._door:
            raise HawkError(
                "hawk.Event.elapsed_ms: the two events were recorded on different devices"
            )
        return through_door(self._door.event_elapsed, since._handle, self._handle)

    def close(self) -> None:
        """Free the event at once; closing twice is harmless."""
        if self._finalizer is not None:
            self._finalizer()

    def _require_recorded(self, call: str, which: str) -> None:
        """Refuse by name unless ``which`` (this event) has been recorded and not closed."""
        if self._finalizer is None or not self._finalizer.alive:
            raise HawkError(
                f"hawk.Event.{call}: {which} was never recorded, or is closed; record it first"
            )
