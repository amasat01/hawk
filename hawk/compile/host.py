# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host build as the lowering library states it: one object for each unit text, one object for the text of the set, and
one link of all of them into a shared library.

The library's facts answer gives the three commands as argv templates (its ``host_build`` record: ``compile_unit``,
``compile_set`` and ``link``) with places in them for what only the caller knows: the compiler, the files, and where the
headers are. This module fills the places and runs the commands. It adds, drops and reorders no element of a template; the
caller's own ``-D`` defines are appended after it, where a compiler reads them wherever they stand.

Both compiles go through the content-closure cache (:func:`~hawk.compile.drivers.drive`, the key of an object is the unit's
text, its target and mode, the compiler's identity, the argv as given, the optimisation level and profile in effect, the C
library and, for a profile that targets this CPU, the CPU's identity). The link has a key of its own: the keys of the
objects in order, the text of the set and the link argv. Its record keeps the objects' recorded header closures, so a
header that changed since the library was linked makes the library stale as it makes its objects stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .._errors import HawkError
from . import drivers
from . import payload as _payload
from . import toolchain as tc
from .cache import Cache, default_cache_dir, digest_file, publish_atomic, tmp_sibling
from .drivers import CompileOptions, CompileResult

#: A place in an argv template: ``{compiler}``, ``{object}``, ``{include:eagle}`` ...
_PLACE = re.compile(r"\{[a-z_]+(?::[a-z_]+)?\}")


@dataclass(frozen=True)
class HostBuild:
    """What a host compile of a build knows beyond the library's argv templates: the compiler and its identity, the profile and
    the optimisation level in effect, the identity of the CPU a native profile targets, the folder of the facts header and the
    term its header contributes to a key."""

    instructions: dict
    compiler: str
    identity: str
    profile: str
    opt_level: str
    target_identity: str
    facts_dir: object
    facts_term: str


def fill_argv(template, places: dict, lists: dict | None = None) -> list:
    """``template`` with each place replaced by its text (``places`` maps a place, braces included, to a function that gives the
    text), and each element that is a place of ``lists`` (``{objects}``, ``{defines}``) by one element for each of its items, none
    when it has none. A place nobody fills is refused by name."""
    out = []
    for element in template:
        if element in (lists or {}):
            out.extend(str(item) for item in lists[element])
            continue

        def fill(found):
            if found.group(0) not in places:
                raise HawkError(f"the host build instructions name the place {found.group(0)}, which this build does not fill")
            return str(places[found.group(0)]())

        out.append(_PLACE.sub(fill, element))
    return out


class _ObjectRow:
    """What :func:`~hawk.compile.drivers.drive` asks of a target row, for the compile of one text to an object: the suffixes of
    the files in its slot and the command to run."""

    artifact = ".o"
    source = ".cpp"

    @staticmethod
    def compile_hook(plan: dict, job, run) -> dict:
        """Run the compile template of ``plan`` for ``job``, then the caller's extra flags."""
        places = {**plan["places"], "{compiler}": lambda: plan["compiler"], "{source}": lambda: job.src, "{object}": lambda: job.out,
                  "{depfile}": lambda: job.dep}
        if job.served is not None:
            places["{include:served}"] = lambda: job.served
        argv = [*fill_argv(plan["template"], places, {"{defines}": plan["defines"]}), *plan["extra"]]
        run(argv)
        return {"argv": tuple(argv), "figures": {}}


def _places(build: HostBuild, *, door: bool) -> dict:
    """The places of a compile that do not depend on the file: the compiler's include roots, the folder of the facts header
    and, for the text of a set, the root of the door headers. Each is found when a template asks for it."""
    places = {"{include:aether}": lambda: tc.include_roots()[0], "{include:eagle}": lambda: tc.include_roots()[1],
              "{include:facts}": lambda: build.facts_dir}
    if door:
        places["{include:door}"] = tc.door_include
    return places


def _key_terms(build: HostBuild, template, served: bool) -> tuple:
    """The key terms of a compile beyond its text, target, mode and the compiler's identity: the argv as given, where the headers
    are (the roots, or the digest of the sealed payload) and the header of the facts, then the profile, the optimisation level
    and the C library (a library links against the libc it was built on); for a profile that targets this CPU also the CPU, since
    its flag reads the same on every machine and a shared cache directory would otherwise serve one CPU's binary to another's."""
    aether, eagle = ("", "") if served else tc.include_roots()
    headers = (f"payload:{_payload.current_payload().digest}",) if served else ("-I", aether, "-I", eagle)
    libc = "-".join(part for part in platform.libc_ver() if part) or "unknown"
    terms = ("host_build", *template, *headers, build.facts_term, f"host_profile:{build.profile}", f"opt_level:{build.opt_level}",
             f"libc:{libc}")
    return terms if build.profile == "portable" else (*terms, f"host_target:{build.target_identity}")


def compile_object(text: str, name: str, opts: CompileOptions, build: HostBuild, *, of_set: bool = False) -> CompileResult:
    """Compile the text of a unit (or, with ``of_set``, the text of the set) to an object with the template of ``build`` (the
    library's ``host_build`` record), through the cache. ``opts`` carries the target, the scalar mode, the cache directory and the
    caller's defines, which the template of a unit and of a set both place (the set text states the layout sizes a define can change),
    and extra flags, which follow the template."""
    template = build.instructions["compile_set" if of_set else "compile_unit"]
    served = drivers.header_shape() == "served"
    defines = [f"-D{d}" for d in opts.defines]
    return drivers.drive(text, name, opts, _ObjectRow, {
        "compiler": build.compiler, "identity": build.identity, "payload": _payload.current_payload().digest if served else None,
        "serve": served, "tail": " against the sealed payload" if served else "", "deps": True,
        "template": template, "defines": defines, "extra": [*opts.extra], "places": _places(build, door=of_set),
        "key_terms": _key_terms(build, [*template, *defines, *opts.extra], served),
        "hit_argv": tuple(template), "meta": {"flags": list(template)}})


class _EmbedRow:
    """What :func:`~hawk.compile.drivers.drive` asks of a target row, for the assembly of the text
    that embeds an image: the suffixes of the files in its slot and the command to run."""

    artifact = ".o"
    source = ".s"

    @staticmethod
    def compile_hook(plan: dict, job, run) -> dict:
        """Run the assemble template of ``plan`` for ``job``."""
        places = {"{host_compiler}": lambda: plan["compiler"], "{source}": lambda: job.src,
                  "{object}": lambda: job.out}
        argv = fill_argv(plan["template"], places)
        run(argv)
        return {"argv": tuple(argv), "figures": {}}


def embed_image(image: CompileResult, meta: dict, recipe: dict, opts: CompileOptions,
                build: HostBuild) -> CompileResult:
    """Assemble the object that holds the image file of a device unit under the two symbols its
    record names (``meta["device_image"]``, ``meta["device_image_size"]``), through the cache.
    ``recipe`` is the ``embed`` record of the library's ``device_build`` answer: the text for this
    operating system and the command that assembles it with the host compiler of ``build``. The key
    is the text, the command, the compiler and the CONTENT of the image, which the text reaches only
    by its path."""
    digest = digest_file(image.artifact)
    if digest is None:
        raise HawkError(f"the image of {meta['name']!r} is at {image.artifact} "
                        "but it cannot be read")
    text = (recipe["source"][_system()].replace("{size_symbol}", meta["device_image_size"])
            .replace("{symbol}", meta["device_image"]).replace("{image_path}", str(image.artifact)))
    template = recipe["compile"]
    return drivers.drive(text, f"{meta['name']}__image", opts, _EmbedRow, {
        "compiler": build.compiler, "identity": build.identity, "payload": None, "serve": False,
        "tail": "", "deps": False, "template": template, "key_terms": ("embed", *template, f"image:{digest}"),
        "hit_argv": tuple(template), "meta": {"flags": list(template)}})


def _system() -> str:
    """The key of the link template for this operating system."""
    return "macos" if sys.platform == "darwin" else "linux"


def link(objects: list, set_object: CompileResult | None, set_text: str | None, library: str, opts: CompileOptions,
         build: HostBuild) -> CompileResult:
    """Link the objects of the units, in order, and the object of the set text into the shared library ``library`` (a file name),
    through the cache. The result's ``artifact`` is the library in its cache slot, ``source`` the set text, ``key`` the key of
    the library: the keys of the objects in order, the text of the set and the link argv.

    A library of plain objects has no set: ``set_object`` and ``set_text`` are both ``None``, ``source`` is the last object's text
    and the key names the compiler's identity in place of the set text."""
    template = build.instructions["link"][_system()]
    cache = Cache(opts.cache_dir)
    parts = [r.key for r in objects] + ([] if set_object is None else [set_object.key])
    h = hashlib.sha256(b"hawk.compile.link/1\n")
    for part in (*parts, build.identity if set_object is None else set_text, library, *template):
        h.update(part.encode("utf-8") + b"\0")
    key = h.hexdigest()
    slot = cache.slot(key)
    digest = _payload.current_payload().digest if drivers.header_shape() == "served" else None
    done = [*objects] if set_object is None else [*objects, set_object]
    last = done[-1]
    closure = tuple(set_object.closure) if set_object else tuple(dict.fromkeys(pair for r in objects for pair in r.closure))

    def lookup():
        if not cache.check(key, payload_digest=digest):
            return None
        cache.note(True)
        return CompileResult(slot / library, last.source, key, True, tuple(template), closure=closure)

    def build_library():
        cache.note(False)
        slot.mkdir(parents=True, exist_ok=True)
        tmp = tmp_sibling(slot / library)
        placed = {"{compiler}": lambda: build.compiler, "{output}": lambda: tmp}
        argv = fill_argv(template, placed, {"{objects}": [r.artifact for r in done]})
        label = "host link of " + repr(library)
        start = time.perf_counter()
        try:
            finished = drivers.run_compiler(argv, label, opts.target)
            if finished.returncode != 0:
                drivers.refuse(finished, argv, label, "", cache.root)
            publish_atomic(tmp, slot / library)
        finally:
            tmp.unlink(missing_ok=True)
        seconds = time.perf_counter() - start
        # the library is as valid as the objects it was linked from: it keeps the header closure each of them recorded
        recorded = []
        for r in done:
            recorded += [tuple(pair) for pair in cache.record(r.key).get("closure", ())]
        meta = {"target": opts.target, "mode": opts.mode, "compiler": build.identity, "key_terms": list(parts), "seconds": seconds,
                "name": library, "flags": list(template)}
        if digest is not None:
            meta["payload_digest"] = digest
        cache.store(key, artifact=slot / library, closure=tuple(dict.fromkeys(recorded)), meta=meta)
        return CompileResult(slot / library, last.source, key, False, tuple(argv), seconds, closure=closure)

    return drivers.single_flight(key, lookup, build_library)


#: The verdicts of the masked-call witness already known to this process, by key.
_WITNESS_MEMO: dict = {}


def masked_call_verdict(build: HostBuild, source: Path, template: list) -> str:
    """What the masked-call witness says of this compiler under the flags of ``template`` (a unit's compile argv of the library
    for the verdict ``unrun``): ``fixed`` (it was built and ran correctly), ``miscompile`` (it ran and gave wrong results) or ``unrun``
    (it could not be built or run). The witness ``source`` is compiled with the template's own flags (no header is read: every
    include place is an empty folder), linked and run. The verdict is kept in the process and in the compile cache directory, by the
    compiler's identity, the CPU and the template."""
    key = hashlib.sha256("\n".join([build.identity, build.target_identity, *template]).encode()).hexdigest()[:16]
    if key in _WITNESS_MEMO:
        return _WITNESS_MEMO[key]
    stored = default_cache_dir() / "toolchain" / f"masked_call_{key}.json"
    try:
        verdict = json.loads(stored.read_text())["verdict"]
    except (OSError, ValueError, KeyError):
        verdict, why = _run_witness(build, source, template)
        if verdict == "unrun":
            logging.getLogger("hawk").warning("hawk: the masked-call witness did not run (%s): aether's workaround stays", why)
        else:   # an unrun witness is not stored: the cause may be passing
            try:
                stored.parent.mkdir(parents=True, exist_ok=True)
                tmp = stored.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps({"verdict": verdict}))
                os.replace(tmp, stored)
            except OSError:
                pass
    return _WITNESS_MEMO.setdefault(key, verdict)


def _run_witness(build: HostBuild, source: Path, template: list) -> tuple:
    """``(verdict, why)`` of one run of the witness (:func:`masked_call_verdict`)."""
    with tempfile.TemporaryDirectory(prefix="hawk-witness-") as tmp:
        exe, obj = os.path.join(tmp, "witness"), os.path.join(tmp, "witness.o")
        names = ("aether", "eagle", "served", "facts", "door")
        places = {f"{{include:{n}}}": (lambda: tmp) for n in names}
        places.update({"{compiler}": lambda: build.compiler, "{source}": lambda: source, "{object}": lambda: obj,
                       "{depfile}": lambda: os.path.join(tmp, "witness.d")})
        try:
            built = subprocess.run(fill_argv(template, places, {"{defines}": []}), capture_output=True, text=True, timeout=120)
            if built.returncode == 0:
                built = subprocess.run([build.compiler, obj, "-o", exe], capture_output=True, text=True, timeout=120)
            if built.returncode != 0:
                return "unrun", "the witness does not compile with the host flags"
            ran = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return "unrun", f"{type(exc).__name__}: cannot build or run the witness"
    return {0: ("fixed", ""), 1: ("miscompile", "")}.get(ran.returncode, ("unrun", f"the witness exited {ran.returncode}"))
