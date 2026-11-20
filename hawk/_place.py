# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where an array lives, which door serves that place, and how arrays cross it: the one home of hawk's device handling.

Arrays cross by the DLPack protocol and nothing else. hawk reads an array's device tuple (``__dlpack_device__``), asks the
OSPREYs for the door of that place (:func:`door_of`), wraps the array as a plane through it and hands arrays back by the
protocol too: a plane the door makes is read by the array library the caller's own arrays come from (NumPy on the host). hawk
imports no array library, holds no stream and decides nothing about a device: a device context, its stream and its memory are
the door's, and whether a library runs here is its own machine answer (:func:`hawk.targets.runs_here`).
"""

from __future__ import annotations

import re
import sys
from types import MappingProxyType

import _osprey

from . import ospreys
from ._plane_layout import MARK_ATTR
from .errors import HawkError
from .targets import TARGETS, host_target, runs_here

#: The DLPack device tuple of ordinary host memory, and the types whose memory the host reads: host and pinned host memory.
HOST = (1, 0)
_HOST_TYPES = (1, 3)

_KINDS = MappingProxyType({"int": 0, "uint": 1, "float": 2})


def dlpack_dtype(name: str) -> tuple:
    """``(DLPack type code, bits)`` of the dtype spelled ``name`` as the sidecar spells it (``float64``, ``uint32``, ``bool``)."""
    if name == "bool":
        return 6, 8
    parsed = re.fullmatch(r"(u?int|float)(\d+)", name)
    if parsed is None:
        raise HawkError(f"the dtype {name!r} has no DLPack form")
    return _KINDS[parsed[1]], int(parsed[2])


class _Exported:
    """A plane the door made, as an object that exports DLPack: an array library reads it with its ``from_dlpack``, once."""

    def __init__(self, capsule, device: tuple) -> None:
        """Hold the door's ``capsule`` over the plane's memory, which lives on ``device`` (a DLPack ``(type, id)``)."""
        self._capsule = capsule
        self._device = device

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """The capsule, handed over once; the consumer's stream is the door's to order against, not this object's."""
        if self._capsule is None:
            raise BufferError("this DLPack export was already consumed; ask for the array again")
        capsule, self._capsule = self._capsule, None
        return capsule

    def __dlpack_device__(self) -> tuple:
        """The DLPack ``(type, id)`` of the memory."""
        return self._device


def _unmarked(value):
    """The array a layout marker (``hawk.samples_last``) holds, else ``value``."""
    return value.array if getattr(value, MARK_ATTR, None) is not None and hasattr(value, "array") else value


def device_of(array) -> tuple | None:
    """The DLPack ``(type, id)`` of ``array``, a host array being :data:`HOST`; ``None`` for an object that exports no DLPack."""
    array = _unmarked(array)
    if not hasattr(array, "__dlpack_device__"):
        return None
    kind, ordinal = (int(n) for n in array.__dlpack_device__())
    return HOST if kind in _HOST_TYPES else (kind, ordinal)


def place_of(arrays: dict, where: str) -> tuple | None:
    """``(device, first array there)`` of the arrays of one launch, which all live in one place; ``None`` when none is an array.

    Two places are refused by name: a launch runs on one door, so its planes are in one memory.
    """
    found: dict = {}
    for name, value in arrays.items():
        device = device_of(value)
        if device is not None:
            found.setdefault(device, (name, _unmarked(value)))
    if len(found) > 1:
        (first, (a, _)), (second, (b, _)) = list(found.items())[:2]
        raise HawkError(f"{where}: the planes of one launch live in one place; {a!r} is on the DLPack device {first} and "
                        f"{b!r} is on {second}")
    return next(iter(found.items())) if found else None


def open_door(target: str, ordinal: int):
    """The door of ``target``'s OSPREY for device ``ordinal``, or the library's reason why it does not run here."""
    osprey = ospreys.serving(target)
    if ordinal in osprey.open_doors():      # a door made before needs no second machine answer
        return osprey.runtime_door(ordinal)
    runs, why = runs_here(target)
    if not runs:
        raise HawkError(f"the {osprey.name} library does not run here: {why}")
    return osprey.runtime_door(ordinal)


def door_of(device: tuple):
    """The runtime door of the place ``device`` (a DLPack ``(type, id)``): the host OSPREY's for host memory, else the door of
    the device OSPREY that runs here and whose device ``id`` is of that type, made on first use and kept. No such OSPREY is
    refused by name, each one asked with its reason."""
    if device == HOST:
        return ospreys.serving(host_target().id).runtime_door()
    kind, ordinal = device
    asked = []
    for target in TARGETS.values():
        if not target.device:
            continue
        try:
            door = open_door(target.id, ordinal)
        except HawkError as refusal:
            asked.append(f"{target.osprey}: {refusal}")
            continue
        if door.ctx_info()["device_type"] == kind:
            return door
        asked.append(f"{target.osprey}: its device {ordinal} is of another type")
    raise HawkError(f"no installed OSPREY runs on the DLPack device {device}: " + ("; ".join(asked) or "none serves a device"))


def stream_of(door) -> int | None:
    """The DLPack stream code a producer orders its pending work against before the door reads its array; ``None`` for a host
    door, which has no stream."""
    stream = door.ctx_info()["stream"]
    return stream if stream >= 0 else None


def wrap(door, array):
    """The plane over ``array``, the caller's memory, made by ``door`` and ordered behind the producer's pending work."""
    stream = stream_of(door)
    return door.plane_wrap(array) if stream is None else door.plane_wrap(array, stream=stream)


def _reader(like):
    """The ``from_dlpack`` of the array library ``like`` comes from."""
    library = sys.modules.get(type(like).__module__.partition(".")[0])
    reader = getattr(library, "from_dlpack", None)
    if reader is None:
        raise HawkError(f"{type(like).__name__} comes from a library with no from_dlpack, so an output left out cannot be made "
                        "for it; pass the output")
    return reader


class Maker:
    """Makes arrays in the memory of one place by the place's door: zeroed, in a shape and dtype the caller states.

    On the host an array is NumPy's; on a device it is of the library the caller's own arrays (``like``) come from.
    """

    def __init__(self, door, device: tuple, like=None) -> None:
        """Make arrays through ``door``, whose memory is ``device``; ``like`` is an array of that place, which names the library
        a device array is read by."""
        self.device = device
        self.door = door
        self._like = like

    @classmethod
    def at(cls, device: tuple, like=None) -> Maker:
        """The maker of the place ``device`` (:func:`door_of`)."""
        return cls(door_of(device), device, like)

    def copy_of(self, array):
        """The NumPy array ``array`` in this place's memory, copied through the door; on the host the array itself."""
        if self.device == HOST:
            return array
        layout = self.door.array_layout(array)
        source = wrap(self.door, array)
        try:
            made = self.door.plane_new(list(layout["shape"]), *layout["dtype"][:2], kind=_osprey.EG_MEM_DEVICE)
            try:
                self.door.copy(source, made)
                self.door.sync()
                return read(self.door, made, self.device, self._like)
            finally:
                self.door.plane_free(made)
        finally:
            self.door.plane_free(source)

    def zeros(self, shape, dtype: str):
        """A zeroed array of ``shape`` and ``dtype`` (a sidecar word) in this place's memory."""
        return self._made(shape, dlpack_dtype(dtype))

    def like(self, array):
        """A zeroed array of the shape and dtype of ``array``, read from the layout its DLPack states."""
        layout = self.door.array_layout(array)
        return self._made(layout["shape"], layout["dtype"][:2])

    def _made(self, shape, dtype: tuple):
        """The zeroed plane of ``shape`` and DLPack ``dtype`` the door makes, read by the array library of this place and freed
        as a handle at once: the array keeps the memory."""
        kind = _osprey.EG_MEM_HOST if self.device == HOST else _osprey.EG_MEM_DEVICE
        plane = self.door.plane_new([int(n) for n in shape], *dtype, kind=kind, flags=_osprey.EG_PLANE_ZERO)
        try:
            return read(self.door, plane, self.device, self._like)
        finally:
            self.door.plane_free(plane)


def read(door, plane, device: tuple, like=None):
    """The memory of ``plane`` as an array: NumPy's on the host, else of the library ``like`` comes from; with no ``like`` on a
    device, the object that exports it."""
    exported = _Exported(door.plane_dlpack(plane), device)
    if device == HOST:
        import numpy

        return numpy.from_dlpack(exported)
    return exported if like is None else _reader(like)(exported)


def to_host(array):
    """``array``, device memory, as a NumPy array: copied by the door of its place into host memory it makes."""
    device = device_of(array)
    door = door_of(device)
    layout = door.array_layout(array)
    code, bits, _lanes = layout["dtype"]
    source = wrap(door, array)
    try:
        target = door.plane_new(list(layout["shape"]), code, bits, kind=_osprey.EG_MEM_HOST)
        try:
            door.copy(source, target)
            door.sync()
            return read(door, target, HOST)
        finally:
            door.plane_free(target)
    finally:
        door.plane_free(source)


def to_device(array, target: str, ordinal: int):
    """The NumPy array ``array`` copied into the memory of device ``ordinal`` of the OSPREY serving ``target``, as the object that
    exports it (an array library reads it with its ``from_dlpack``)."""
    door = open_door(target, ordinal)
    info = door.ctx_info()
    return Maker(door, (info["device_type"], info["device"])).copy_of(array)
