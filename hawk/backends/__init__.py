# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Backends: one object per thing ``hawk.build(targets=...)`` can build.

A backend carries its row DATA (id, entry, format, suffixes) and the
TRAITS the build asks about it. It is found, never imported by name: the
registry (:mod:`hawk.registry`) reads each OSPREY's manifest and imports a
backend on its first :func:`need`, so adding a backend is shipping an OSPREY that declares one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..registry import backend_ids, load, missing_device, traits


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
    entry: str | None = None
    format: str
    artifact: str
    source: str
    #: compiled with a toolchain (``False``: exported whole, never built)
    builds: bool = True
    #: built for a device: resolves a device arch, runs team forms and banded modes
    device: bool = False
    #: its compile reads the host profile, and its units are objects linked into one library for the build
    host_compiled: bool = False
    #: eagle enters a host symbol of it
    host_entry: bool = False
    #: for a unit with no entry eagle can plan: what it is, said in a refusal to load it
    unit_note: str = ""
    #: the module of this backend's own user API, which ``hawk.<door>`` hands out (``None``: it has none)
    api: object = None
    #: its loop over samples may be vectorised
    vectorise: bool = False
    #: the facts it reads, as its manifest's raptor wing names them: a name, or a map from the host's isa to a name (:func:`hawk.facts.named`)
    facts_name: object = None
    #: the wing each family contributes (manifest ``wings``); a build without one installed is refused by name (:func:`hawk.wings.require`)
    wings: object = None

    def own_options(self, options) -> dict:
        """The part of ``options`` under this backend's ``<id>.`` prefix, prefix dropped (the driver refused any key its manifest does not declare)."""
        prefix = self.id + "."
        return {k[len(prefix):]: v for k, v in options.items() if k.startswith(prefix)}

    def build(self, lowered, source, options) -> Part:
        """Compile ``source`` through the cache (a backend with ``builds``) and package it: the :class:`Part` of this backend.
        Never asked on a memo hit."""
        from ..compile import compile_source

        own = self.own_options(options)
        arch = options.get("device_arch", "")
        opts = self._compile_options(lowered, options)
        if not self.builds:
            result = None
        elif self.host_compiled:
            # a host unit is compiled to an object by the library's instructions; the bundle links the objects of its units
            from ..compile import host

            result = host.compile_object(source.text, lowered.name, opts, self._host_build(lowered, options, opts))
        else:
            result = compile_source(source.text, lowered.name, opts)
        entry, files, report = self.package(source, result, name=lowered.name, smode=lowered.smode, profile=own.get("profile"))
        facts = self.facts(lowered.kernel, name=lowered.name, smode=lowered.smode, block=options.get("block"), spec=lowered.spec.for_target(self),
                           built=entry, arch=arch)
        return Part(entry, files, report, self.report_file(lowered.name), facts)

    def _compile_options(self, lowered, options):
        """The :class:`~hawk.compile.CompileOptions` of this backend's compile in a build, as the kind of the build adjusts them."""
        from ..facts import include_flags
        from ..compile import CompileOptions

        return lowered.kind.apply(CompileOptions(
            target=self.id, mode=lowered.smode.id, arch=options.get("device_arch", ""), defines=tuple(options.get("defines", ())),
            extra=() if self.host_compiled else include_flags(lowered.spec.for_target(self).primary),
            cache_dir=options.get("cache_dir"), host_profile=options.get("host_profile") or "",
            opt_level=options.get("opt_level") or "", settings=lowered.spec.for_target(self).settings))

    def _host_build(self, lowered, options, opts):
        """The :class:`~hawk.compile.host.HostBuild` of a build: the library's argv templates (with the witness's verdict, when the
        compiler needs one), this machine's compiler and the terms its keys hold. The caller's own flags (``opts``) go with the
        question: a macro they set for the masked call wins over the witness (:func:`hawk.artifact.request.host_build`)."""
        from ..artifact.request import compiling_host, host_build
        from ..compile import host
        from ..compile import toolchain as tc
        from ..facts import include_dir, include_flags

        primary = lowered.spec.for_target(self).primary
        compiler = self.compiler()
        identity = tc.compiler_identity(compiler)
        facts, line = compiling_host()
        profile = options.get("host_profile") or facts["host_profile"]
        build = host.HostBuild({}, compiler, identity, profile, tc.opt_level(options.get("opt_level") or None), self.target_identity(),
                               include_dir(primary), next((f.key_term for f in include_flags(primary)), ""))
        def witness(template):
            return host.masked_call_verdict(build, self.masked_call_witness_source, template)

        caller_flags = (*(f"-D{d}" for d in opts.defines), *map(str, opts.extra))
        return replace(build, instructions=host_build(facts, line, options, witness, caller_flags))

    def link(self, lowered, options, units: list, library: str, recipe: dict | None = None):
        """Link the objects of ``units`` (``(host object, device image, unit record)`` triples, in
        order; either of the first two may be ``None``) with the text of their set into the shared
        library ``library`` (a file name) by the library's instructions: the
        :class:`~hawk.compile.CompileResult` of the link, whose ``artifact`` is the library in the
        cache. A device image is assembled into an object first, by ``recipe`` (the ``embed`` record
        of the device build instructions, :meth:`embed_recipe`). ``lowered`` carries the build's
        scalar mode, kind and spec."""
        from ..artifact.request import set_text
        from ..compile import host

        opts = self._compile_options(lowered, options)
        build = self._host_build(lowered, options, opts)
        objects = []
        for unit_object, image, meta in units:
            if unit_object is not None:
                objects.append(unit_object)
            if image is not None:
                objects.append(host.embed_image(image, meta, recipe, opts, build))
        text = set_text([meta for _, _, meta in units], options.get("layout_sizes_override"))
        set_object = host.compile_object(text, "set", opts, build, of_set=True)
        return host.link(objects, set_object, text, library, opts, build)

    def embed_recipe(self, lowered, options) -> dict:
        """The recipe the library states for assembling this backend's image file into the set
        library: the ``embed`` record of its compile plan (a device backend)."""
        from ..compile import drivers

        return drivers.plan_for(self._compile_options(lowered, options))["embed"]

    #: The program the masked-call witness builds and runs (``None``: the backend has none).
    masked_call_witness_source = None

    def compiler(self) -> str:
        """The compiler a host build runs (discovery stays with the backend); a backend with no host build has none."""
        raise NotImplementedError(self.id)

    def target_identity(self) -> str:
        """A digest naming the CPU a native profile of this backend targets."""
        raise NotImplementedError(self.id)

    def machine_facts(self) -> dict:
        """The facts this backend knows of the machine of a build it is in, as plain data (:class:`hawk.targets.Machine`
        field names: its compiler and profiles, the isa); the default knows none."""
        return {}

    def compile_plan(self, opts, inputs):
        """The DATA of one compile (``None`` for a backend that does not build): a dict with ``compiler``,
        ``identity`` (the key's compiler term), ``key_terms`` (the key's terms after the source, target, mode and identity),
        ``payload`` (the sealed-payload digest the entry is valid by, else ``None``), ``deps`` (the hook writes a dependency
        file), ``serve`` (hawk serves the payload for the compile), ``flags``, ``hit_argv`` (what a hit reports), ``tail``
        (said after "failed" in a refusal) and ``meta`` (fields of the record). ``inputs`` is what hawk knows of the header
        roots: ``sealed``, ``payload``, ``include_flags()``."""
        return None

    def compile_hook(self, plan, job, run) -> dict:
        """Compile ``job.src`` into ``job.out`` (and ``job.dep`` when ``plan["deps"]``), running every command through
        ``run(argv)``: ``{"argv": what ran, "figures": the resource figures the compiler reported}``."""
        raise NotImplementedError

    def object_format(self, path) -> tuple[str, str]:
        """``(manifest format, file suffix)`` of the compiled object at ``path``; the default is the row's own."""
        return self.format, self.artifact

    def package(self, source, result, *, name: str, smode, profile):
        """This backend's share of a draft: ``((result, artifact name, source name, format), files, report)``.
        ``result`` is its compile (:class:`hawk.compile.CompileResult`); ``report`` is ``None`` for a compiled backend."""
        from ..artifact.bundle import UnitFile, text_blob, digest_bytes
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
        blob = text_blob(source.text)
        files = [UnitFile(src_name, digest_bytes(blob), blob=blob)]
        if not self.host_compiled:                           # a host unit's object is linked, with its bundle's, into one library
            files.insert(0, UnitFile(art_name, compiled, source_path=result.artifact))
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
    """The default backend built for a device, or a refusal saying which OSPREY to install."""
    for backend in default_backends("device"):
        return backend
    from .._errors import HawkError

    raise HawkError(missing_device())
