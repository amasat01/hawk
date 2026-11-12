# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Backends: one object per thing ``hawk.build(targets=...)`` can build.

A backend carries its row DATA (id, entry, format, suffixes) and the
TRAITS the build asks about it. It is found, never imported by name: the
registry (:mod:`hawk.registry`) reads each extension's manifest and imports a
backend on its first :func:`need`, so adding a backend is shipping an extension that declares one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..registry import backend_ids, load, missing_device, traits


@dataclass(frozen=True)
class Lowered:
    """What hawk's lowering works out ONCE per kernel, for every backend: the kernel with its name, scalar mode and kind,
    the shared fused ``body`` (``None`` without a hook), the ``segment`` spec, the ``one_step`` kernel, the thread ``box``,
    the ``active`` spec and the ``rows`` a block serves. Build inputs a backend alone reads travel in ``options``."""

    kernel: object
    name: str
    smode: object
    kind: object
    body: object
    segment: object
    one_step: object
    box: object
    active: object
    rows: int


@dataclass(frozen=True)
class BuiltLowered(Lowered):
    """:class:`Lowered` with the build's :class:`hawk.targets.BuildSpec`: the variant data every pass and writer of the build reads."""

    spec: object


@dataclass(frozen=True)
class Part:
    """One backend's share of a draft, as :meth:`Backend.build` returns it: ``entry`` is ``(result, artifact name, source
    name, format)``, ``files`` the files it contributes, ``report`` its report (``None`` for a compiled backend),
    ``report_file`` the name of that report's file (``None`` for a compiled one) and ``facts`` the sidecar parts only it
    knows (:meth:`Backend.facts`)."""

    entry: tuple
    files: list
    report: object
    report_file: str | None
    facts: dict


class Backend:
    """One backend. Subclass, set the data and traits, instantiate once.

    Data: ``id``, ``about`` (the words that say what it builds), ``writer``
    and ``wrapper`` (``wrapper`` is ``None`` when the writer produces the
    whole file), ``entry`` (how eagle enters it: ``"device"``, ``"host"`` or
    ``None``), ``format`` (the manifest format), ``artifact`` and ``source``
    (file suffixes)."""

    id: str
    about: str
    writer: object = None
    wrapper: object = None
    entry: str | None = None
    format: str
    artifact: str
    source: str
    #: compiled with a toolchain (``False``: exported whole, never built)
    builds: bool = True
    #: built for a device: resolves a device arch, runs team forms and banded modes
    device: bool = False
    #: its compile reads the host profile
    host_compiled: bool = False
    #: eagle enters a host symbol of it
    host_entry: bool = False
    #: for a unit with no entry eagle can plan: what it is, said in a refusal to load it
    unit_note: str = ""
    #: the module of this backend's own user API, which ``hawk.<door>`` hands out (``None``: it has none)
    api: object = None
    #: its loop over samples may be vectorised
    vectorise: bool = False
    #: its variant as its manifest names it: a name, or a map from the host's isa to a name (:func:`hawk.variants.named`)
    variant: object = None
    #: the components its builds need installed (manifest ``components``); a build without one is refused by name (:func:`hawk.components.require`)
    components: tuple = ()

    def own_options(self, options) -> dict:
        """The part of ``options`` under this backend's ``<id>.`` prefix, prefix dropped (the driver refused any key its manifest does not declare)."""
        prefix = self.id + "."
        return {k[len(prefix):]: v for k, v in options.items() if k.startswith(prefix)}

    def emit(self, lowered: Lowered, options) -> object:
        """The unit's source (:class:`hawk.emit.Source`) for ``lowered``: pure, no compiler, no disk."""
        return self.render(lowered, options)

    def build(self, lowered: Lowered, source, options) -> Part:
        """Compile ``source`` through the cache (a backend with ``builds``) and package it: the :class:`Part` of this backend.
        Never asked on a memo hit."""
        from .. import variants
        from ..compile import CompileOptions, compile_source

        own = self.own_options(options)
        arch = options.get("device_arch", "")
        result = None if not self.builds else compile_source(source.text, lowered.name, lowered.kind.apply(CompileOptions(
            target=self.id, mode=lowered.smode.id, arch=arch, defines=tuple(options.get("defines", ())),
            extra=variants.include_flags(lowered.spec.for_target(self).primary),
            cache_dir=options.get("cache_dir"), host_profile=options.get("host_profile") or "",
            opt_level=options.get("opt_level") or "")))
        entry, files, report = self.package(source, result, name=lowered.name, smode=lowered.smode, profile=own.get("profile"))
        facts = self.facts(lowered.kernel, name=lowered.name, smode=lowered.smode, block=options.get("block"), spec=lowered.spec.for_target(self),
                           built=entry, arch=arch)
        return Part(entry, files, report, self.report_file(lowered.name), facts)

    def render(self, lowered: Lowered, options):
        """The emitted unit of ``lowered.kernel``: the writer and wrapper around the shared body, under this backend's policy scopes."""
        from ..artifact import layout
        from ..emit import render_source
        kernel = lowered.kernel
        team = getattr(kernel, "team", None)
        # the device text keeps wide values in their planes; a host row renders float selects as blends
        spec = lowered.spec.for_target(self).with_settings(placed=self.device, blend=not self.device, staged_rows=lowered.rows,
                                                           warps=getattr(team, "warps", 1))
        source = render_source(
            lowered.name, kernel.sinks, kernel.walk, self, mode=lowered.smode, kind=lowered.kind,
            body=lowered.body, segment=lowered.segment, one_step=lowered.one_step, box=lowered.box,
            active=None if getattr(team, "grid", None) in ("keys", "chunks", "targets") else lowered.active,
            team=None if team is None else team.threads, rows=lowered.rows,
            exports=layout.exports(self.id, layout_sizes_override=options.get("layout_sizes_override")),
            spec=spec)
        return self.finish(source, lowered, options)

    def compile(self, source: str, name: str, opts):
        """Compile ``source`` into this backend's artifact, through the content-closure cache
        (:class:`hawk.compile.CompileResult`). Only a backend with ``builds`` is asked."""
        raise NotImplementedError

    def object_format(self, path) -> tuple[str, str]:
        """``(manifest format, file suffix)`` of the compiled object at ``path``; the default is the row's own."""
        return self.format, self.artifact

    def package(self, source, result, *, name: str, smode, profile):
        """This backend's share of a draft: ``((result, artifact name, source name, format), files, report)``.
        ``result`` is its compile (:class:`hawk.compile.CompileResult`); ``report`` is ``None`` for a compiled backend."""
        from ..artifact.bundle import _File, _blob, _digest_bytes
        from ..compile import digest_file
        from .._errors import HawkError

        fmt, art_ext = self.object_format(result.artifact)
        art_name, src_name = f"{name}{art_ext}", f"{name}{self.source}"
        compiled = digest_file(result.artifact)
        if compiled is None:                                 # pragma: no cover
            raise HawkError(
                f"the content-closure cache says {name!r}'s {self.id} object is at "
                f"{result.artifact} but it cannot be read; the unit cannot be "
                "published from a store that lost its own artifact")
        self.check(name, smode, result.artifact, fmt)
        blob = _blob(source.text)
        files = [_File(art_name, compiled, source_path=result.artifact),
                 _File(src_name, _digest_bytes(blob), blob=blob)]
        return (result, art_name, src_name, fmt), files, None

    def key_terms(self, source_digests, own) -> tuple:
        """Strings this backend adds to a unit key after its options: ``source_digests`` are the kernels'
        ``source_digest`` in order, ``own`` its options (:meth:`own_options`)."""
        return ()

    def report_file(self, name: str) -> str | None:
        """The report file this backend writes for kernel ``name`` (and which stands in for the sidecar when
        no compiled target builds), or ``None``."""
        return None

    def refuse_kernels(self, names, kernels) -> None:
        """Refuse, before anything is emitted, kernels this backend cannot build; the default accepts every one."""

    def check(self, name: str, smode, artifact, fmt: str) -> None:
        """Refuse a compiled object this backend cannot ship in ``smode``; the default accepts every one."""

    def facts(self, kernel, *, name: str, smode, block, spec, built=None, arch: str = "") -> dict:
        """The sidecar parts only this backend knows, for the unit it leads: ``block`` (launch block), ``ptxas``
        (resource figures) and ``local`` (local memory). ``built`` is the lead's draft entry
        ``(result, artifact name, source name, format)`` once compiled, ``None`` when only the source was emitted."""
        return {"block": None, "ptxas": None, "local": None}

    def finish(self, source, lowered: Lowered, options):
        """The unit after the render's scopes close; the default leaves it as rendered."""
        return source

    def process_env(self, env: dict) -> dict:
        """``env`` as a compiler subprocess of any backend should see it, with what this backend's toolchain needs added."""
        return env

    def resolve_arch(self, device_arch: str = "") -> str:
        """The machine arch this backend builds for (``device_arch`` asked, else its own probe and fallback); ``""`` has none."""
        return ""

    def probe_device(self):
        """``(compute capability, name)`` of the device present, or raise ``ImportError``/``HawkError`` with none."""
        raise ImportError(self.id)


def need(backend_id: str) -> Backend:
    """The backend registered as ``backend_id``, imported on first use; a HawkError says what to install or why it is refused."""
    return load("backends", backend_id)


def default_backends(trait: str | None = None) -> list:
    """The loaded backends of the default set that have ``trait`` (all of them when ``None``); one that fails to load is skipped."""
    from .._errors import HawkError

    out = []
    for backend_id in backend_ids(default=True):
        if trait is None or traits(backend_id).get(trait):
            try:
                out.append(need(backend_id))
            except HawkError:
                pass
    return out


def need_device() -> Backend:
    """The default backend built for a device, or a refusal saying which extension to install."""
    for backend in default_backends("device"):
        return backend
    from .._errors import HawkError

    raise HawkError(missing_device())
