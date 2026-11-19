# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Finding and opening the OSPREY libraries a build calls.

An OSPREY is a folder ``osprey_<name>/`` holding ``osprey.json``, its facts and one compiled library
``libosprey_<name>``. The manifest states ``name``, ``version``, ``door`` (the version of the door
the library speaks) and ``targets``: a list of rows ``{"request": what users write (the target's id: host, cuda), "target": the library's own name for
it (the name its requests and answers spell it by), "about": a phrase, "device": whether it builds for a
device, "entry": "host" or "device" (how a runtime enters it), "format": the manifest's word for
its object}``.

hawk looks in the folders that installed distributions declare under the entry-point group
``hawk.ospreys`` (the value is a data path, nothing is imported) and in the folders of the
environment variable ``HAWK_OSPREY_PATH``. It asks a library for nothing but the door calls of
:class:`Osprey`: the machine, the lowering, one unit's build, the link and the store's state.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
from importlib import metadata
from pathlib import Path

from .errors import HawkError, ToolchainAbsent

ENTRY_POINT_GROUP = "hawk.ospreys"
SEARCH_ENV = "HAWK_OSPREY_PATH"
_MANIFEST = "osprey.json"
_MEMBERS = ("name", "version", "door", "targets")
_TARGET_MEMBERS = ("request", "target", "about", "device", "entry", "format")
#: The grammar of a target id once spelled canonically: a lowercase letter, then lowercase letters and digits.
_ID = re.compile(r"[a-z][a-z0-9]*\Z")
#: What the canonical spelling of a target id drops: whitespace, dashes and underscores.
_SEPARATORS = re.compile(r"[\s_-]+")
#: ``{folder: why it was refused}`` of the folders the last search could not use; kept so that a caller asking for a target only
#: such a folder would serve is told which folder and why (:func:`refused`, :func:`serving`).
_REFUSED: dict = {}


def target_id(name: str) -> str:
    """The one spelling of a target id: lowercase, with whitespace, dashes and underscores removed (``" Cu-Da "`` is ``cuda``).

    Every target id a manifest states and every target name a caller passes goes through here. A name that is not a string, or
    whose canonical form is not a lowercase letter followed by lowercase letters and digits, is refused by name.
    """
    spelled = _SEPARATORS.sub("", name).lower() if isinstance(name, str) else ""
    if not _ID.match(spelled):
        raise HawkError(
            f"the target id {name!r} is not a target id: lowercased, with whitespace, dashes and underscores removed, "
            "it must be a letter followed by letters and digits"
        )
    return spelled


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


def _read(folder: Path) -> Osprey:
    """Return the OSPREY in ``folder``, its manifest checked: a missing or unknown member, a target id outside the grammar or a
    Python wing is refused."""
    path = folder / _MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as why:
        raise HawkError(f"{path}: no readable {_MANIFEST} ({why})") from why
    if missing := [m for m in _MEMBERS if m not in manifest]:
        raise HawkError(f"{path}: the manifest lacks {', '.join(missing)}")
    if "wings" in manifest:
        raise HawkError(
            f"{path}: an OSPREY keeps its wings inside its library; a Python wing "
            "is refused"
        )
    for key in sorted(set(manifest) - set(_MEMBERS)):
        raise HawkError(
            f"{path}: unknown manifest key {key!r} (the keys are {list(_MEMBERS)}); a misspelt key is refused, never ignored"
        )
    rows = []
    for row in manifest["targets"]:
        if missing := [m for m in _TARGET_MEMBERS if m not in row]:
            raise HawkError(
                f"{path}: target {row.get('request')!r} lacks {', '.join(missing)}"
            )
        try:
            rows.append({**row, "request": target_id(row["request"])})
        except HawkError as why:
            raise HawkError(f"{path}: the target request: {why}") from why
    ids = [row["request"] for row in rows]
    if twice := sorted({i for i in ids if ids.count(i) > 1}):
        raise HawkError(f"{path}: the manifest states the target id {twice[0]!r} in more than one row (ids are compared in their one spelling)")
    return Osprey(folder, {**manifest, "targets": rows})


@functools.lru_cache(maxsize=None)
def found() -> dict:
    """Return ``{OSPREY name: Osprey}`` of every OSPREY the search list holds.

    Two OSPREYs that serve one target (one canonical id) are refused by name. A folder that cannot be used (a manifest refused, none readable) is
    isolated: the others load, and the folder with its reason is kept in :func:`refused`.
    """
    out, owner = {}, {}
    _REFUSED.clear()
    for folder in _declared_folders():
        try:
            osprey = _read(folder)
        except HawkError as why:
            _REFUSED[str(folder)] = str(why)
            continue
        if osprey.name in out:
            continue
        for target in osprey.targets:
            if target in owner:
                raise HawkError(
                    f"target {target!r} is served by both {owner[target]!r} and "
                    f"{osprey.name!r}; keep one"
                )
            owner[target] = osprey.name
        out[osprey.name] = osprey
    return out


def refused() -> dict:
    """Return ``{folder: why}`` of every folder of the search list that cannot be used, each with its refusal sentence."""
    found()
    return dict(_REFUSED)


def forget() -> None:
    """Search again on the next call (the environment or the installed distributions changed)."""
    found.cache_clear()


def targets() -> dict:
    """Return ``{target id: its manifest row}`` of every target some OSPREY serves."""
    return {t: row for osprey in found().values() for t, row in osprey.targets.items()}


def serving(target: str) -> Osprey:
    """Return the OSPREY that serves ``target``; a target nobody serves is refused by name."""
    target = target_id(target)
    for osprey in found().values():
        if target in osprey.targets:
            return osprey
    unusable = "".join(f"; refused: {why}" for why in refused().values())
    raise HawkError(
        f"target {target!r} is served by no OSPREY found; the served targets are "
        f"{tuple(targets()) or 'none'} (searched: the {ENTRY_POINT_GROUP} entry "
        f"points and ${SEARCH_ENV}){unusable}"
    )
