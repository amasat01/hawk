# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The targets table: a read-only view of the targets the found OSPREYs serve (:mod:`hawk.ospreys`).

A row is what the manifest of the serving OSPREY states about one target, as plain data. Nothing here knows how a target is
built: that is the library's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import ospreys
from ..errors import HawkError

#: How eagle enters a built kernel (the manifest's ``exec_targets`` words).
ENTRY_KINDS = ("device", "host")


@dataclass(frozen=True)
class Target:
    """One served target: ``id``, ``osprey`` (the name of the OSPREY that serves it), ``request`` (how the lowering request
    spells it), ``about`` (what it builds, in words), ``device`` (built for a device) and ``entry`` (``"host"`` or ``"device"``:
    how eagle enters it), ``format`` (the manifest's word for its compiled object)."""

    id: str
    osprey: str
    request: str
    target: str
    about: str
    device: bool
    entry: str
    format: str

    @property
    def host_entry(self) -> bool:
        """Whether eagle enters a host symbol of it."""
        return self.entry == "host"


class _Targets(Mapping):
    """The served targets by id, in the order the OSPREYs were found."""

    def __getitem__(self, key) -> Target:
        try:
            key = ospreys.target_id(key)
        except HawkError:
            raise KeyError(key) from None
        for osprey in ospreys.found().values():
            if key in osprey.targets:
                return Target(key, osprey.name, **{k: osprey.targets[key][k] for k in ("request", "target", "about", "device", "entry", "format")})
        raise KeyError(key)

    def __iter__(self):
        return iter(ospreys.targets())

    def __len__(self):
        return len(ospreys.targets())


class _Default(Sequence):
    """What ``build`` builds when ``targets`` is not given: every served target."""

    def __getitem__(self, i):
        return tuple(ospreys.targets())[i]

    def __len__(self):
        return len(ospreys.targets())

    def __repr__(self):
        return repr(tuple(ospreys.targets()))


TARGETS = _Targets()
DEFAULT_TARGETS = _Default()


def host_target() -> Target:
    """The target whose OSPREY links the library of a build: the first served target that eagle enters as a host symbol."""
    for target in TARGETS.values():
        if target.host_entry:
            return target
    raise HawkError("a build links its units into one library with a host OSPREY: none is found")


def buildable(target: str) -> bool:
    """Whether the OSPREY serving ``target`` says its tools are present on this machine.

    A target no OSPREY serves is refused by name.
    """
    from ..artifact import default_cache_dir    # the artifact package imports this one: a module-level import would be a cycle

    target = ospreys.target_id(target)
    ask = {"target": target, "options": {}, "opt_level": "", "root": str(default_cache_dir())}
    return ospreys.serving(target).machine(ask)["tools_absent"] is None


def runs_here(target: str) -> tuple:
    """``(whether the OSPREY serving ``target`` runs on this machine, its reason)``, as the library's machine answer states them.

    A library whose answer does not state ``runs_here`` is not taken to run here: the reason says it does not say.
    """
    from ..artifact import default_cache_dir    # the artifact package imports this one: a module-level import would be a cycle

    target = ospreys.target_id(target)
    ask = {"target": target, "options": {}, "opt_level": "", "root": str(default_cache_dir())}
    answer = ospreys.serving(target).machine(ask)
    if "runs_here" not in answer:
        return False, f"the {ospreys.serving(target).name} library does not say whether it runs here"
    return bool(answer["runs_here"]), answer.get("why", "")


def machine(target: str, *, options=None, opt_level: str = "") -> dict:
    """What the library says about this machine for ``target``: the ``stated`` record of its machine answer, as given (compiler
    path and identity, profile, flags, device architecture). The keys are the library's; hawk reads none of them.

    ``options`` and ``opt_level`` are those of :func:`hawk.artifact.build`. Raises :class:`~hawk.errors.ToolchainAbsent` with the
    library's reason when the machine lacks the tools the target is built with, and :class:`~hawk.errors.HawkError` for a target
    no library serves.
    """
    from ..artifact.bundle import _machines, _user_options, default_cache_dir    # the artifact package imports this one: a cycle

    target = ospreys.target_id(target)
    machine = _machines((target,), _user_options(options, False, None, "", ""), opt_level, default_cache_dir())[target]
    ospreys.serving(target).require_tools(machine)
    return machine["stated"]
