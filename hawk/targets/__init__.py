# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The targets table: a read-only view of the registered backends.

Each row is a :class:`hawk.backends.Backend`: a writer paired with a wrapper,
plus the data and traits the build and the package step read. Nothing here
imports a backend: the registry (:mod:`hawk.registry`) finds them through the
OSPREY manifests and imports one on its first lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from .. import registry
from ..backends import need
from .._errors import HawkError

#: How eagle enters a built kernel (the manifest's ``exec_targets`` words).
ENTRY_KINDS = ("device", "host")


class _Targets(Mapping):
    """The backends by id, in (priority, id) order; a row is imported when read, never when listed.

    LISTING (iteration, ``len``, ``items``) gives the default set: the built-in and first-party backends and the
    OSPREYs named in ``HAWK_OSPREYS``. An installed OSPREY that was not named is reached only by its
    id (``TARGETS["x"]``, ``"x" in TARGETS``): a caller that walks every row (a runtime asking which rows it can
    enter, on every deploy) must neither read the installed distributions nor import a backend nobody named."""

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return need(key)

    def __contains__(self, key):
        return registry.has("backends", key)       # the installed OSPREYs are read only for an id that is not built in

    def __iter__(self):
        return iter(registry.backend_ids(default=True))

    def __len__(self):
        return len(registry.backend_ids(default=True))


class _Default(Sequence):
    def _ids(self):
        registry.check(default=True)
        return registry.loadable_backend_ids(t for t in registry.backend_ids(default=True) if registry.traits(t).get("builds", True))

    def __getitem__(self, i):
        return self._ids()[i]

    def __len__(self):
        return len(self._ids())

    def __repr__(self):
        return repr(self._ids())


TARGETS = _Targets()


def entry_ids(entry: str, loadable: bool = False) -> tuple:
    """The ids of the default set's rows that eagle enters as ``entry``, as their manifests state it: LISTING reads manifest data
    only and imports nothing. ``loadable``: only the rows whose backend imports, for the moment a row is about to be built (an
    implicit call skips a row that fails to import, with one warning that names it and the cause)."""
    ids = tuple(t for t in registry.backend_ids(default=True) if registry.traits(t).get("entry") == entry)
    return registry.loadable_backend_ids(ids) if loadable else ids


#: What ``build`` builds when ``targets`` is not given: every default-set row with a compiler.
DEFAULT_TARGETS = _Default()


@dataclass(frozen=True)
class Machine:
    """The host and its device as plain data: the INPUT of a build, read once (:func:`detect_machine`) from the facts the rows of
    the build supply. ``isa`` is ``"x86_64"`` or ``"arm64"``; ``arch`` the device arch the build resolved (``""``: a host-only
    build); ``reserved_names`` the names a kernel of the build may not take, each with the reason its row gives (a mapping name -> reason);
    ``facts`` every other fact the rows supply, merged by name and read by name (empty where no row knows one)."""

    isa: str = ""
    arch: str = ""
    reserved_names: Mapping = field(default_factory=lambda: MappingProxyType({}))
    facts: Mapping = field(default_factory=lambda: MappingProxyType({}))


def detect_machine(arch: str = "", targets=()) -> Machine:
    """The machine of a build of ``targets``, with the device ``arch`` it resolved: the facts each row supplies
    (:meth:`hawk.backends.Backend.machine_facts`); hawk reads no host itself. A set-valued fact is the union of the rows' sets, a mapping-valued one the union of the rows' mappings (the first row's value for a key two rows give);
    two rows that disagree on any other fact are refused."""
    facts: dict = {}
    for target in targets:
        row = TARGETS.get(target)
        for name, value in ({} if row is None else row.machine_facts()).items():
            known = facts.setdefault(name, value)
            if isinstance(value, frozenset) and isinstance(known, frozenset):
                facts[name] = known | value
            elif isinstance(value, Mapping) and isinstance(known, Mapping):
                facts[name] = {**value, **known}
            elif known != value:
                raise HawkError(f"the backends of this build disagree on the machine's {name}: {facts[name]!r}, and "
                                f"{value!r} from {target!r}")
    return Machine(isa=facts.pop("isa", ""), arch=arch, reserved_names=facts.pop("reserved_names", MappingProxyType({})),
                   facts=MappingProxyType(facts))


@dataclass(frozen=True)
class BuildSpec:
    """The one object that carries the variant of the target being built through a build: the merged variant data of the
    host row and of the device row (``None``: no such target; a backend whose manifest names no variant has none: its own text
    gets an EMPTY variant, never another target's), the machine as data, the class the device's thresholds come from, the
    thresholds a kit forces. Holds no callable, no backend and no
    per-build setting of its own: those are :attr:`settings`, a frozen :class:`lowering.settings.Settings` set by the entry."""

    host: dict | None
    device: dict | None
    machine: Machine
    wide_class: str
    wide_override: dict
    by_target: dict
    focus: dict | None = None
    settings: "Settings" = None
    #: ``{(writer, dialect): module name}`` from the manifests: where a writer finds its spellings of a dialect (data; the module loads when used)
    spellings: dict = field(default_factory=dict)
    #: The variant keys the backend rows of the build state under ``reads`` (``None``: a spec made by hand, nothing is checked): :meth:`need`
    #: refuses a key no row states.
    reads: frozenset | None = None

    def __post_init__(self):
        if self.settings is None:
            from ..build_settings import default_settings

            object.__setattr__(self, "settings", default_settings())

    def with_settings(self, **changes) -> "BuildSpec":
        """A new spec with these per-build settings changed (:class:`lowering.settings.Settings`); this one is unchanged."""
        from dataclasses import replace

        return replace(self, settings=replace(self.settings, **changes))

    @property
    def primary(self) -> dict:
        """The variant the code in hand reads: that of the text being rendered (:meth:`for_target`; empty when its backend names
        none), else, for what is common to every target of the build, the device's when there is one, else the host's."""
        if self.focus is not None:
            return self.focus
        data = self.device if self.device is not None else self.host
        return {} if data is None else data

    def need(self, section: str, key: str | None = None):
        """``primary[section][key]`` (the section when ``key`` is ``None``), refused by name when the variant has no such data."""
        dotted = section if key is None else f"{section}.{key}"
        if self.reads is not None and dotted not in self.reads and section not in self.reads:
            raise HawkError(f"the build reads the variant key {dotted!r}, which no backend row of the build states under `reads` in its manifest")
        data = self.primary.get(section)
        if data is None or (key is not None and key not in data):
            raise HawkError(f"the variant of the target being built has no {section}{'.' + key if key else ''}: a backend whose manifest "
                            "names no variant cannot use a pass that reads it")
        return data if key is None else data[key]

    @property
    def uniform_run(self) -> int | None:
        """The lanes an aligned run a uniform access is proven over spans in the text being rendered: the device's warp width, else the
        host's own run (``None``: the variant names none, so nothing is proven uniform)."""
        data = self.primary
        return data.get("launch", {}).get("warp_size") or data.get("uniform_run")

    @property
    def placement_class(self) -> str:
        """The name of the wide-value class the placement reads: the device class of the build, else the host variant's own."""
        if self.wide_class:
            return self.wide_class
        from .. import wings as _wings
        wide_class = _wings.device_view("wide_class")

        return wide_class.select("", self.device) if self.device is not None else self.need("host_wide")["name"]

    def wide_tables(self) -> dict:
        """The wide-value thresholds the placement reads, ``{"runtime_min", "wide_min"}``: the host variant's own table, else the table of
        the device class of the build."""
        if self.device is None:
            return {k: v for k, v in self.need("host_wide").items() if k != "name"}
        from .. import wings as _wings

        found = _wings.device_view("wide_class").tables(self.placement_class, self.device)
        return {"runtime_min": found.runtime_min, "wide_min": found.wide_min}

    def device_fp64_ratio(self) -> int:
        """FP32 lanes per FP64 lane of the device the build runs on (a build with a device variant only)."""
        from .. import wings as _wings

        arch = self.machine.arch or self.device["default_arch"]
        return _wings.device_view("wide_class").features_of(arch, self.device).fp64_ratio

    def for_target(self, row) -> "BuildSpec":
        """This spec as seen by the text of backend ``row``: its own variant (empty when it names none) is the one :attr:`primary` answers."""
        from dataclasses import replace

        return replace(self, focus=self.by_target.get(row.id) or {}, reads=frozenset(registry.traits(row.id).get("reads", ())))

    def wide(self, scalar_type: str) -> dict:
        """The wide-value thresholds card for ``scalar_type`` on this spec's device arch."""
        from . import cards

        return cards.wide(scalar_type, self)

    def fusion(self, scalar_type: str, host: bool = False) -> dict:
        """The fusion constants for ``scalar_type`` (of this host's vector ISA with ``host``)."""
        from . import cards

        return cards.fusion(scalar_type, self, host)


def build_spec(targets, arch: str = "", *, machine: Machine | None = None, device: dict | None = None,
               wide_override: dict | None = None, settings=None, build_settings=None) -> BuildSpec:
    """The spec of a build of ``targets`` (ids) for device ``arch``: each row's manifest names its wings (the raptor leaf, a map from the host's
    isa for a host row, gives its facts). ``device`` replaces the device variant's data (a test or a kit passes other numbers), ``wide_override``
    forces card thresholds. ``settings`` are the per-build settings (default: all unset); ``build_settings`` (:class:`hawk.build_settings.BuildSettings`) is laid over them after its keys
    are checked against the targets' manifests."""
    from .. import wings as _wings
    from ..variants import load, named

    machine = machine or detect_machine(arch, targets)
    host = dev = None
    by_target = {}
    for target in targets:
        if target not in TARGETS:
            continue                    # lowering refuses it by name
        row = TARGETS[target]
        _wings.require(_wings.row_wings(row.wings, machine.isa), target)
        if row.variant is None:
            continue
        data = load(named(row.variant, machine.isa))
        by_target[target] = device if (row.device and device is not None) else data
        if row.device:
            dev = dev or data
        else:
            host = host or data
    dev = device if device is not None else dev
    return BuildSpec(host, dev, machine, _wings.device_view("wide_class").select(arch, dev) if arch and dev else "", dict(wide_override or {}), by_target,
                     settings=_laid(build_settings, settings, targets), spellings=_spellings(targets, machine.isa),
                     reads=frozenset(k for t in targets if t in TARGETS for k in registry.traits(t).get("reads", ())))


def _spellings(targets, isa: str) -> dict:
    """``{(writer, dialect): module name}`` the build names: the spellings arms of its backends' OSPREYs and of the wings
    those backends (and the wings in turn) name."""
    from .. import wings as _wings

    found = registry.spelling_refs(tuple(targets))
    for key in _wings.chain([k for t in targets if t in TARGETS for k in _wings.row_wings(TARGETS[t].wings, isa)]):
        manifest = _wings.wing(*key).MANIFEST
        found.update({(arm["writer"], arm["dialect"]): arm["ref"] for arm in manifest.get("spellings", ())})
    return found


def _laid(build_settings, settings, targets):
    """``settings`` (default: all unset) with ``build_settings`` laid over it."""
    from ..build_settings import default_settings, refuse_undeclared, settings_of

    if build_settings is not None:
        refuse_undeclared(build_settings, targets)
    return settings_of(build_settings, default_settings() if settings is None else settings)
