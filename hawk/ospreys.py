# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Finding and opening the OSPREY libraries a build calls.

An OSPREY is a folder ``osprey_<name>/`` holding ``osprey.json``, its facts and one compiled library
``libosprey_<name>``. The manifest states ``name``, ``version``, ``door`` (the version of the door
the library speaks) and ``targets``: a list of rows ``{"request": what users write (the target's id: host, cuda), "target": the library's own name for
it (the name its requests and answers spell it by), "about": a phrase, "device": whether it builds for a
device, "entry": "host" or "device" (how a runtime enters it), "format": the manifest's word for
its object}``. Which manifests are valid and how a target id is spelled are decided by the binding; hawk passes it folder paths
and raw names and takes its checked answers and refusals as they are.

hawk looks in the folders that installed distributions declare under the entry-point group
``hawk.ospreys`` (the value is a data path, nothing is imported) and in the folders of the
environment variable ``HAWK_OSPREY_PATH``. It asks a library for nothing but the door calls of
:class:`Osprey`: the machine, the lowering, one unit's build, the link, the store's state, and the runtime door that
loads and launches the built units.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
from importlib import metadata
from pathlib import Path

from .errors import HawkError, ToolchainAbsent

ENTRY_POINT_GROUP = "hawk.ospreys"
SEARCH_ENV = "HAWK_OSPREY_PATH"


def _ask(call: str, *args) -> str:
    """Return the text answer of the binding's manifest call ``backend_<call>``; its refusal is raised as ``HawkError``.

    The binding decides what a valid manifest is and what a target id is; its refusal sentence names the folder or the id and the
    rule, and is passed on as it is.
    """
    import _osprey

    try:
        return getattr(_osprey.Ctx(), "backend_" + call)(*args)
    except (ValueError, _osprey.Fault) as refusal:
        raise HawkError(str(refusal)) from refusal


def target_id(name: str) -> str:
    """The one spelling of a target id, as the binding states it: lowercase, with whitespace, dashes and underscores removed
    (``" Cu-Da "`` is ``cuda``). A name that is no target id is refused by name.
    """
    if not isinstance(name, str):
        raise HawkError(
            f"the target id {name!r} is not a target id: a string is required"
        )
    return _ask("target_id", name)


class Osprey:
    """One found OSPREY: its manifest as data and, once a call needs it, its opened library.

    Every method takes and returns plain data (dicts parsed from the library's JSON). A refusal
    of the library is raised as :class:`~hawk.errors.HawkError` with the library's ``code``.
    """

    def __init__(self, folder: Path, manifest: dict) -> None:
        """Hold the OSPREY of ``folder`` described by its checked ``manifest``; open nothing yet."""
        self.folder = folder
        self.name = manifest["name"]
        self.version = manifest["version"]
        self.door = manifest["door"]
        self.targets = {row["request"]: row for row in manifest["targets"]}
        self._lock = threading.Lock()
        self._open = None
        self._doors: dict = {}

    def _opened(self) -> tuple:
        """``(binding module, context, library)``, opened on the first call."""
        with self._lock:
            if self._open is None:
                import _osprey

                suffix = "dylib" if sys.platform == "darwin" else "so"
                library = self.folder / f"libosprey_{self.name}.{suffix}"
                if not library.is_file():
                    raise HawkError(
                        f"the OSPREY {self.name!r} at {self.folder} holds no {library.name}"
                    )
                context = _osprey.Ctx()
                try:
                    self._open = (_osprey, context, context.backend_open(str(library)))
                except (ValueError, _osprey.Fault) as refusal:
                    raise HawkError(
                        f"the OSPREY {self.name!r} at {self.folder} cannot be opened: {refusal}"
                    ) from refusal
            return self._open

    def runtime_door(self, device: int | None = None):
        """Return the runtime door of this library: one :class:`_osprey.Door` for the host (``device=None``) or one for each
        device ordinal, each made on first use and kept with the opened library, which stays open as long as a door lives.

        Units, planes and plans are made through it; its context admits one calling thread at a time, and a device door has
        one stream. A library that cannot make the door refuses by its reason.
        """
        module, _context, library = self._opened()
        with self._lock:
            if device not in self._doors:
                try:
                    self._doors[device] = module.Door(library) if device is None else module.Door(library, device=device)
                except (ValueError, module.Fault) as refusal:
                    place = "" if device is None else f" for device {device}"
                    raise HawkError(
                        f"the OSPREY {self.name!r} at {self.folder} has no runtime door{place}: {refusal}"
                    ) from refusal
            return self._doors[device]

    def open_doors(self) -> dict:
        """Return the runtime doors made so far, ``{device ordinal or None for the host: door}``; none is made by asking."""
        with self._lock:
            return dict(self._doors)

    def _refused(self, what: str, refusal: Exception) -> HawkError:
        """Return the ``HawkError`` for the library's ``refusal`` of ``what``, keeping its code."""
        code = getattr(refusal, "code", None)
        if code == "tool_absent":
            return ToolchainAbsent(f"{self.name}: {refusal}", toolchain=self.name)
        error = HawkError(f"the OSPREY {self.name!r} refuses {what}: {refusal}")
        error.code = code
        return error

    def _call(self, door: str, *args) -> dict:
        """Call door function ``door`` and return its JSON answer parsed."""
        module, context, library = self._opened()
        try:
            return json.loads(getattr(context, "backend_" + door)(library, *args))
        except (ValueError, module.Fault) as refusal:
            raise self._refused(door, refusal) from refusal

    def machine(self, ask: dict) -> dict:
        """Return the machine the library builds on for ``ask``, kept as given.

        The answer is ``{"key": ..., "stated": ..., "tools_absent": reason or None}``; a machine with
        no tools still states itself, so source can be emitted. :meth:`require_tools` is what a build asks.
        """
        return self._call("machine", json.dumps(ask))

    def require_tools(self, machine: dict) -> None:
        """Raise :class:`~hawk.errors.ToolchainAbsent` with the library's reason when ``machine`` says its tools are absent."""
        if machine.get("tools_absent"):
            raise ToolchainAbsent(
                f"{self.name}: {machine['tools_absent']}", toolchain=self.name
            )

    def lower(self, request: dict, dump: str, dialects: str) -> dict:
        """Return the library's answer to ``request`` over the nodes of ``dump``.

        The answer holds the texts, the units' records and the working set.
        """
        module, context, library = self._opened()
        try:
            return json.loads(
                context.backend_lower(library, dump, dialects, json.dumps(request))
            )
        except (ValueError, module.Fault) as refusal:
            raise self._refused(repr(request["name"]), refusal) from refusal

    def build(self, unit: dict, root: Path) -> dict:
        """Build one unit under the store folder ``root``; return the library's answer.

        The answer names the files, the format, the figures and the report.
        """
        return self._call("build", json.dumps(unit), str(root))

    def merge(self, answers: list) -> dict:
        """Return the one answer this OSPREY makes of the lowering ``answers`` of every OSPREY of a build, as given."""
        return self._call("merge", json.dumps(answers))

    def link(self, parts: list, set_text: str, name: str, root: Path) -> dict:
        """Link the units' ``parts`` into the set library ``name``; return ``{"library": path}``."""
        return self._call("link", json.dumps(parts), set_text, name, str(root))

    def set_text(self, units: list, exports=None) -> str:
        """Return the text of a set of units built into one library, from their records."""
        module, context, library = self._opened()
        asked = {"v": 1, "units": units}
        if exports is not None:
            asked["exports"] = [int(n) for n in exports]
        try:
            return context.backend_set_text(library, json.dumps(asked))
        except (ValueError, module.Fault) as refusal:
            names = [u["name"] for u in units]
            raise self._refused(f"the set of units {names}", refusal) from refusal

    def state(self, op: str, root: Path) -> dict:
        """Return the state of the store under ``root``; ``op`` is ``"stats"`` or ``"reset"``."""
        return self._call("state", op, str(root))


def _declared_folders() -> list:
    """Return the folders distributions declare under the entry-point group, then those of the variable."""
    folders = [
        Path(point.dist.locate_file(point.value))
        for point in metadata.entry_points(group=ENTRY_POINT_GROUP)
    ]
    folders += [Path(p) for p in os.environ.get(SEARCH_ENV, "").split(os.pathsep) if p]
    return folders


@functools.lru_cache(maxsize=None)
def _roster() -> tuple:
    """Return ``(the binding's roster text, the same parsed)`` of the folders the search list holds.

    Two OSPREYs that serve one target are refused by name; a folder whose manifest is refused is listed with its reason and the
    others are kept.
    """
    text = _ask("roster", json.dumps([str(folder) for folder in _declared_folders()]))
    return text, json.loads(text)


@functools.lru_cache(maxsize=None)
def found() -> dict:
    """Return ``{OSPREY name: Osprey}`` of every OSPREY the search list holds, each manifest as the binding checked it."""
    return {m["name"]: Osprey(Path(m["folder"]), m) for m in _roster()[1]["ospreys"]}


def refused() -> dict:
    """Return ``{folder: why}`` of every folder of the search list that cannot be used, each with its refusal sentence."""
    return dict(_roster()[1]["refused"])


def forget() -> None:
    """Search again on the next call (the environment or the installed distributions changed)."""
    _roster.cache_clear()
    found.cache_clear()


def targets() -> dict:
    """Return ``{target id: its manifest row}`` of every target some OSPREY serves."""
    return {t: row for osprey in found().values() for t, row in osprey.targets.items()}


def serving(target: str) -> Osprey:
    """Return the OSPREY that serves ``target``, in any spelling of its id; a target nobody serves is refused by name.

    The binding states the folders that were searched; where those folders come from is what this module decides, so the
    refusal ends with it: a reader with no OSPREY installed learns how one is found.
    """
    try:
        answer = _ask("serving", _roster()[0], target)
    except HawkError as refusal:
        places = f"the {ENTRY_POINT_GROUP} entry points and ${SEARCH_ENV}"
        raise HawkError(f"{refusal}; OSPREY folders are found through {places}") from refusal
    return found()[json.loads(answer)["osprey"]]
