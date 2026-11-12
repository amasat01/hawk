# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The targets table: a read-only view of the registered backends.

Each row is a :class:`hawk.backends.Backend`: a writer paired with a wrapper,
plus the data and traits the build and the package step read. Nothing here
imports a backend: the registry (:mod:`hawk.registry`) finds them through the
extension manifests and imports one on its first lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import registry
from ..backends import need
from .._errors import HawkError

#: How eagle enters a built kernel (the manifest's ``exec_targets`` words).
ENTRY_KINDS = ("device", "host")


class _Targets(Mapping):
    """The backends by id, in (priority, id) order; a row is imported when read, never when listed.

    LISTING (iteration, ``len``, ``items``) gives the default set: the built-in and first-party backends and the
    extensions named in ``HAWK_EXTENSIONS``. An installed extension that was not named is reached only by its
    id (``TARGETS["x"]``, ``"x" in TARGETS``): a caller that walks every row (a runtime asking which rows it can
    enter, on every deploy) must neither read the installed distributions nor import a backend nobody named."""

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return need(key)

    def __contains__(self, key):
        return registry.has("backends", key)       # the installed extensions are read only for an id that is not built in

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
    """The host and its device as plain data: the INPUT of a build, read once (:func:`detect_machine`). ``isa`` is ``"x86_64"`` or
    ``"arm64"``; ``host_profile`` the host profile in effect when none is asked and ``env_profile`` the one ``$HAWK_HOST_PROFILE``
    asks; ``compiler`` / ``gcc`` the host compiler and whether it is GCC; ``vector_isa`` the key of the fusion card's host row;
    ``arch`` the device arch the build resolved (``""``: a host-only build)."""

    isa: str
    arm64_gcc: bool
    exact_host_profiles: tuple
    host_profile: str
    env_profile: str
    compiler: str
    gcc: bool
    vector_isa: str
    arch: str

    def profile(self, requested: str | None = None) -> str:
        """The host profile ``requested`` (else the environment's, else this machine's), refused by name when unknown, and refused when
        an explicit vector-math profile meets a non-GCC host compiler on x86-64 (its vector function ABI is GCC-only)."""
        from ..compile.toolchain import HOST_PROFILES, VECTOR_MATH_PROFILES

        explicit = requested or self.env_profile
        name = explicit or self.host_profile
        if name not in HOST_PROFILES:
            raise HawkError(f"unknown host profile {name!r} (from {'the compile options' if requested else '$HAWK_HOST_PROFILE'}); "
                            f"the profiles are {HOST_PROFILES}")
        if explicit and name in VECTOR_MATH_PROFILES and self.isa == "x86_64" and not self.gcc:
            raise HawkError(f"host profile {name!r} needs a GCC host compiler (its vector function ABI is GCC-only), but "
                            f"{self.compiler!r} is not GCC; use the 'native' profile, or point $HAWK_CXX at a GCC g++")
        return name


def _vector_isa() -> str:
    """The vector ISA of this host, the key of the fusion card's host row."""
    import platform
    from pathlib import Path

    if platform.machine().lower() in ("arm64", "aarch64"):
        return "neon"
    try:
        flags = next((line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")), "")
    except OSError:
        flags = ""
    words = set(flags.split())
    return "avx512" if "avx512f" in words else "avx2" if "avx2" in words else "sse2"


def detect_machine(arch: str = "") -> Machine:
    """The machine this process runs on, with the device ``arch`` the build resolved. The only reader of the host."""
    import os

    from ..compile.toolchain import EXACT_HOST_PROFILES, _arm64_gcc, _is_x86_64, compiler_is_gcc, default_host_profile, host_compiler

    try:
        compiler = host_compiler()
        gcc = bool(compiler_is_gcc(compiler))
    except Exception:                    # no compiler: a compile refuses it by name, later
        compiler, gcc = "", False
    return Machine("x86_64" if _is_x86_64() else "arm64", bool(_arm64_gcc()), tuple(EXACT_HOST_PROFILES), default_host_profile(),
                   os.environ.get("HAWK_HOST_PROFILE") or "", str(compiler), gcc, _vector_isa(), arch)


@dataclass(frozen=True)
class BuildSpec:
    """The one object that carries the variant of the target being built through a build: the merged variant data of the
    host row and of the device row (``None``: no such target; a backend whose manifest names no variant has none: its own text
    gets an EMPTY variant, never another target's), the machine as data, the class the device's thresholds come from, the
    thresholds a kit forces. Holds no callable, no backend and no
    per-build setting of its own: those are :attr:`settings`, a frozen :class:`hawk.passes.policy.Settings` set by the entry."""

    host: dict | None
    device: dict | None
    machine: Machine
    wide_class: str
    wide_override: dict
    by_target: dict
    focus: dict | None = None
    settings: "Settings" = None

    def __post_init__(self):
        if self.settings is None:
            from ..passes.policy import Settings

            object.__setattr__(self, "settings", Settings())

    def with_settings(self, **changes) -> "BuildSpec":
        """A new spec with these per-build settings changed (:class:`hawk.passes.policy.Settings`); this one is unchanged."""
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
        from .. import components as _components
        wide_class = _components.device_view("wide_class")

        return wide_class.select("", self.device) if self.device is not None else self.need("host_wide")["name"]

    def wide_tables(self) -> dict:
        """The wide-value thresholds the placement reads, ``{"runtime_min", "wide_min"}``: the host variant's own table, else the table of
        the device class of the build."""
        if self.device is None:
            return {k: v for k, v in self.need("host_wide").items() if k != "name"}
        from .. import components as _components

        found = _components.device_view("wide_class").tables(self.placement_class, self.device)
        return {"runtime_min": found.runtime_min, "wide_min": found.wide_min}

    def device_fp64_ratio(self) -> int:
        """FP32 lanes per FP64 lane of the device the build runs on (a build with a device variant only)."""
        from .. import components as _components

        arch = self.machine.arch or self.device["default_arch"]
        return _components.device_view("wide_class").features_of(arch, self.device).fp64_ratio

    def for_target(self, row) -> "BuildSpec":
        """This spec as seen by the text of backend ``row``: its own variant (empty when it names none) is the one :attr:`primary` answers."""
        from dataclasses import replace

        return replace(self, focus=self.by_target.get(row.id) or {})

    def wide(self, scalar_type: str) -> dict:
        """The wide-value thresholds card for ``scalar_type`` on this spec's device arch."""
        from . import cards

        return cards.wide(scalar_type, self)

    def fusion(self, scalar_type: str, host: bool = False) -> dict:
        """The fusion constants for ``scalar_type`` (of this host's vector ISA with ``host``)."""
        from . import cards

        return cards.fusion(scalar_type, self, host)


def build_spec(targets, arch: str = "", *, machine: Machine | None = None, device: dict | None = None,
               wide_override: dict | None = None, settings=None, build_policy=None) -> BuildSpec:
    """The spec of a build of ``targets`` (ids) for device ``arch``: each row's manifest names its variant (a map from the host's
    isa for a host row). ``device`` replaces the device variant's data (a test or a kit passes other numbers), ``wide_override``
    forces card thresholds. ``settings`` are the per-build settings (default: all unset); ``build_policy`` (:class:`hawk.build_policy.BuildPolicy`) is laid over them after its keys
    are checked against the targets' manifests."""
    from .. import components as _components
    from ..variants import load, named

    machine = machine or detect_machine(arch)
    host = dev = None
    by_target = {}
    for target in targets:
        if target not in TARGETS:
            continue                    # lowering refuses it by name
        row = TARGETS[target]
        for need in row.components:
            _components.require(need, target)
        if row.variant is None:
            continue
        data = load(named(row.variant, machine.isa))
        by_target[target] = device if (row.device and device is not None) else data
        if row.device:
            dev = dev or data
        else:
            host = host or data
    dev = device if device is not None else dev
    return BuildSpec(host, dev, machine, _components.device_view("wide_class").select(arch, dev) if arch and dev else "", dict(wide_override or {}), by_target,
                     settings=_laid(build_policy, settings, targets))


def _laid(build_policy, settings, targets):
    """``settings`` (default: all unset) with ``build_policy`` laid over it."""
    from ..build_policy import refuse_undeclared, settings_of
    from ..passes.policy import Settings

    if build_policy is not None:
        refuse_undeclared(build_policy, targets)
    return settings_of(build_policy, Settings() if settings is None else settings)
