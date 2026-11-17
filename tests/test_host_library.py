# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host half of a build call is ONE shared library: the host units of every kernel of the call are compiled to objects, linked
with the text of their set, and published as one file with a link named for each unit beside it (the runtime still opens a unit by
its name).

What these rows state: the layout of the published files, that every unit's entry resolves from the one library, that a set of
units of two scalar modes links and runs both, that two units of one name are refused before anything is compiled, and that a
second build of the same set runs no compiler.

A device unit's image is assembled into an object and linked into the same library, under the
symbols its record names. No device compiler is here: a stand-in for the image compile alone writes
a file that depends on the source text; everything after it (the embedding text, the assembler,
the set text, the link) is real."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import stat
from types import SimpleNamespace

import _deployable as D
import numpy as np
import pytest
from ospreys.cuda import nvrtc as device_nvrtc
from ospreys.cuda import toolchain as device_toolchain
from raptor.errors import RaptorError

import hawk.runtime as R
from hawk.artifact import build, build_bundle, request
from hawk import facts
from hawk.compile import host as host_driver
from hawk.compile import toolchain as host_toolchain
from hawk.artifact.bundle import _emitted, _prepared
from hawk.compile import drivers
from hawk.artifact.bundle import reset_unit_memo
from hawk._errors import HawkError
from hawk.targets import TARGETS

KERNELS = (D.axpb, D.vec3_scale, D.fraction)


@pytest.fixture(scope="module")
def three(tmp_path_factory, cache_dir):
    return build_bundle(list(KERNELS), tmp_path_factory.mktemp("three"), targets=("host",), cache_dir=cache_dir)


def test_a_bundle_of_three_kernels_has_one_library_and_a_link_for_each_unit(three):
    directory = three.directory
    library = directory / "units.so"
    assert library.is_file() and not library.is_symlink()
    links = sorted(p.name for p in directory.glob("*.so") if p.is_symlink())
    assert links == sorted(f"{k.name}.so" for k in KERNELS)
    assert sorted(p.name for p in directory.glob("*.so")) == sorted(["units.so", *links])
    for link in links:
        assert (directory / link).resolve() == library.resolve()
    # every kernel's entry names the library; the files of the bundle list the library and the links
    assert {a.entries["host"].artifact for a in three.artifacts} == {library}
    assert {name for p in three.files() for name in [p.name] if p.suffix == ".so"} == {"units.so", *links}
    for plugin in three.manifest["plugins"]:
        assert plugin["library"] == "units.so" and plugin["artifact"] == f"{plugin['id']}.so"


def test_every_unit_entry_resolves_from_the_one_library(three):
    library = ctypes.CDLL(str(three.directory / "units.so"))
    for kernel in KERNELS:
        assert ctypes.cast(getattr(library, f"{kernel.name}_host"), ctypes.c_void_p).value
    # opened through a link, a unit is the same library
    through = ctypes.CDLL(str(three.directory / "fraction.so"))
    assert ctypes.cast(through.axpb_host, ctypes.c_void_p).value == ctypes.cast(library.axpb_host, ctypes.c_void_p).value


def test_a_name_given_to_the_bundle_names_its_library(tmp_path, cache_dir):
    bundle = build_bundle([D.axpb, D.fraction], tmp_path, targets=("host",), cache_dir=cache_dir)
    assert (tmp_path / "units.so").is_file()
    named = build_bundle([D.axpb], tmp_path / "n", targets=("host",), cache_dir=cache_dir, name="law")
    # the one unit is named like the bundle: the library is that file, and no link is needed
    assert (tmp_path / "n" / "law.so").is_file() and not (tmp_path / "n" / "law.so").is_symlink()
    assert named.artifacts[0].links == () and bundle.artifacts[0].links == (tmp_path / "axpb.so",)


def test_a_single_kernel_build_is_a_library_named_for_its_unit(tmp_path, cache_dir):
    artifact = build(D.axpb, tmp_path, targets=("host",), cache_dir=cache_dir)
    assert artifact.entries["host"].artifact == tmp_path / "axpb.so"
    assert not (tmp_path / "axpb.so").is_symlink() and not (tmp_path / "units.so").exists()


def _unit(kernel, mode, cache_dir):
    """``(part the host backend built, record of the unit, lowered)`` of one kernel in one scalar mode, as the build makes them."""
    lowered, answer, unit, options = _emitted(kernel, kernel.name, mode, None, ("host",), "", "test", "build_bundle", None, exports_block=False,
                                              cache_dir=cache_dir)
    prepared = _prepared(unit, lowered, kernel, ("host",))
    part = TARGETS["host"].build(prepared[0], prepared[1]["host"], options)
    return part, unit["meta"], lowered, options


def test_a_set_of_float32_and_float64_units_links_and_runs_both(tmp_path, cache_dir):
    # the units of one build call share a scalar mode, so a set of both is made from two calls' units: the case in which the objects of
    # one library must agree on every entity their headers define with external linkage
    (p64, m64, lowered, options), (p32, m32, *_) = _unit(D.axpb, "float64", cache_dir), _unit(D.fraction, "float32", cache_dir)
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for kernel, mode in ((D.axpb, "float64"), (D.fraction, "float32")):
        side = build_bundle([kernel], tmp_path / mode, mode=mode, targets=("host",), cache_dir=cache_dir)
        shutil.copy(side.artifacts[0].sidecar_path, mixed)
    result = TARGETS["host"].link(lowered, options, [(p64.entry[0], None, m64), (p32.entry[0], None, m32)], "mixed.so")
    shutil.copy(result.artifact, mixed / "mixed.so")
    for unit in ("axpb", "fraction"):
        os.symlink("mixed.so", mixed / f"{unit}.so")
    x = np.arange(1.0, 6.0)
    y = np.zeros(5)
    R.run(R.load(mixed, "axpb"), x=x, a=2.0, b=1.0, y=y)
    assert y.tolist() == (2.0 * x + 1.0).tolist()
    x32, y32 = x.astype(np.float32), np.zeros(5, dtype=np.float32)
    R.run(R.load(mixed, "fraction"), x=x32, y=y32)
    assert y32.tolist() == (x32 / np.float32(5)).tolist()


def test_two_units_of_one_name_are_refused_naming_both_kernels(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"'axpb' and 'fraction' both write a unit named 'same'"):
        build_bundle([D.axpb, D.fraction], tmp_path, targets=("host",), cache_dir=cache_dir, name="same")
    assert not list(tmp_path.iterdir())


def test_a_second_build_of_the_same_set_runs_no_compiler(tmp_path, cache_dir, monkeypatch):
    first = build_bundle(list(KERNELS), tmp_path / "a", targets=("host",), cache_dir=cache_dir)
    reset_unit_memo()                       # the process remembers nothing: the answer comes from the cache
    calls = []
    real = drivers.run_compiler
    monkeypatch.setattr(drivers, "run_compiler", lambda argv, what, *target: calls.append(what) or real(argv, what, *target))
    second = build_bundle(list(KERNELS), tmp_path / "b", targets=("host",), cache_dir=cache_dir)
    assert calls == []
    assert all(e.hit for a in second.artifacts for e in a.entries.values())
    assert (tmp_path / "b" / "units.so").read_bytes() == (tmp_path / "a" / "units.so").read_bytes()
    assert first.digest == second.digest


WIDE_INDEX = ("AETHER_INDEX_T=std::uint64_t",)
INDEX_FIELD = 3                             # the slot of ``eagle_layout_sizes`` that holds ``sizeof(EAGLE_ABI_INDEX_T)``


def _exported_index_width(library) -> int:
    """The index width, in bytes, that a built library states in its exported layout sizes."""
    sizes = (ctypes.c_uint64 * 5).in_dll(ctypes.CDLL(str(library)), "eagle_layout_sizes")
    return sizes[INDEX_FIELD]


def test_the_index_width_a_host_library_exports_is_the_width_its_units_were_compiled_with(tmp_path, cache_dir):
    narrow = build_bundle([D.fraction], tmp_path / "narrow", targets=("host",), cache_dir=cache_dir)
    wide = build_bundle([D.fraction], tmp_path / "wide", targets=("host",), cache_dir=cache_dir, defines=WIDE_INDEX)
    assert _exported_index_width(narrow.directory / "units.so") == 4
    assert _exported_index_width(wide.directory / "units.so") == 8


def test_changing_only_the_defines_rebuilds_the_set_text_object(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")                 # its own cache: a shared one may hold the wide build already
    build_bundle([D.fraction], tmp_path / "a", targets=("host",), cache_dir=cache)
    reset_unit_memo()
    labels = _compiles(monkeypatch)
    build_bundle([D.fraction], tmp_path / "b", targets=("host",), cache_dir=cache, defines=WIDE_INDEX)
    assert "host compile of 'set'" in labels


def _x86_gcc():
    """``(facts, line)`` of a machine and a GCC the masked-call table lists, as a compile would state them: the library answers for them
    with no compiler present."""
    return {"isa": "x86_64", "vector_isa": "avx2", "host_profile": "native-vector-math"}, "g++ || g++ (GCC) 13.2.0 || 0"


@pytest.mark.parametrize("verdict", ["fixed", "miscompile", "unrun"])
def test_the_masked_call_flag_follows_the_verdict_of_the_witness(verdict):
    (facts, line), asked = _x86_gcc(), []
    build = request.host_build(facts, line, {}, witness=lambda template: asked.append(template) or verdict)
    # the witness is asked once, for the template of a compiler not proven, and the question then states its verdict
    assert len(asked) == 1 and "-DAETHER_HVM_MASKED=1" not in asked[0]
    stated = {**request._stated("x86_64", facts), "opt_level": "O3", "masked_call_witness": verdict}
    expected = json.loads(request._facts("host", line, json.dumps(stated)))["host_build"]
    assert build == expected
    assert ("-DAETHER_HVM_MASKED=1" in build["compile_unit"]) == (verdict == "fixed")
    # the argv the driver runs is that template with its places filled and the caller's defines after it
    ran = []
    # the include roots are this environment's (a sealed payload or a source checkout)
    aether, eagle = host_toolchain.include_roots()
    driver = host_driver.HostBuild(build, "c++", "c++ || x || 0", "native-vector-math", "O3", "cpu", "/facts", "facts:x")
    plan = {"places": host_driver._places(driver, door=False), "template": build["compile_unit"], "compiler": "c++", "defines": ["-DX"],
            "extra": []}
    job = SimpleNamespace(src="/s.cpp", out="/s.o", dep="/s.d", served="/served")
    host_driver._ObjectRow.compile_hook(plan, job, ran.append)
    assert ran[0] == host_driver.fill_argv(build["compile_unit"], {"{compiler}": lambda: "c++", "{source}": lambda: "/s.cpp", "{object}": lambda: "/s.o",
                                                              "{depfile}": lambda: "/s.d", "{include:served}": lambda: "/served",
                                                              "{include:facts}": lambda: "/facts", "{include:aether}": lambda: aether,
                                                              "{include:eagle}": lambda: eagle}, {"{defines}": ["-DX"]})
    assert ("-DAETHER_HVM_MASKED=1" in ran[0]) == (verdict == "fixed")


def test_a_host_build_does_not_call_the_old_command_builder(tmp_path, monkeypatch):
    import ospreys.cpu.toolchain as old_tools

    def called(*args, **kwargs):
        raise AssertionError("the old host command builder was called")

    backend = TARGETS["host"]
    for owner, name in ((type(backend), "compile_plan"), (type(backend), "compile_hook"), (old_tools, "host_flags"), (old_tools, "host_codegen_flags")):
        monkeypatch.setattr(owner, name, called)
    reset_unit_memo()
    bundle = build_bundle([D.axpb, D.fraction], tmp_path, targets=("host",), cache_dir=str(tmp_path / "cache"))
    assert (tmp_path / "units.so").is_file() and not any(e.hit for a in bundle.artifacts for e in a.entries.values())


@pytest.mark.parametrize("macro", ["AETHER_HVM_MASKED=0", "AETHER_HVM_MASKED=1", "AETHER_HVM_MASKED"])
def test_a_masked_call_macro_of_the_caller_wins_without_a_probe(macro):
    """The caller's own macro for the masked call is in the argv once, from the template's ``{defines}``, and no flag of the template's
    own follows: the witness is not run and the question states no verdict."""
    build = request.host_build(*_x86_gcc(), {})          # no witness: no verdict is stated
    assert "-DAETHER_HVM_MASKED=1" not in build["compile_unit"]
    driver = host_driver.HostBuild(build, "c++", "c++ || x || 0", "native-vector-math", "O3", "cpu", "/facts", "")
    plan = {"places": host_driver._places(driver, door=False), "template": build["compile_unit"], "compiler": "c++", "defines": [f"-D{macro}"],
            "extra": []}
    ran = []
    host_driver._ObjectRow.compile_hook(plan, SimpleNamespace(src="/s.cpp", out="/s.o", dep="/s.d", served="/served"), ran.append)
    assert [a for a in ran[0] if a.startswith("-DAETHER_HVM_MASKED")] == [f"-D{macro}"]


@pytest.mark.parametrize("defines,probed", [((), True), (("WITNESSED=0",), False), (("WITNESSED",), False), (("OTHER",), True)])
def test_the_witness_is_asked_for_only_when_the_caller_set_no_macro_of_the_fixed_template(monkeypatch, defines, probed):
    """Which macro the witness decides is read from the library's two templates (here a stand-in library whose fixed template
    adds ``-DWITNESSED=1``): a caller that sets that macro itself is not probed and gets no flag from the template."""
    import json

    def facts(target, line, stated):
        fixed = json.loads(stated).get("masked_call_witness") == "fixed"
        return json.dumps({"host_build": {"compile_unit": ["{compiler}", *(["-DWITNESSED=1"] if fixed else [])]}})

    monkeypatch.setattr(request, "_facts", facts)
    asked = []
    build = request.host_build(*request.compiling_host(), {}, witness=lambda template: asked.append(template) or "fixed",
                               caller_flags=[f"-D{d}" for d in defines])
    assert bool(asked) == probed
    assert ("-DWITNESSED=1" in build["compile_unit"]) == probed


def test_the_witness_runs_for_real_under_the_template_of_this_machine(tmp_path, monkeypatch):
    """The witness of this machine's compiler is built from the library's own template and run: this compiler is not a flagged GCC, and
    a program that is correct says so."""
    from hawk.compile import toolchain as tc

    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path))
    backend = TARGETS["host"]
    template = request.host_build(*request.compiling_host(), {})["compile_unit"]
    driver = host_driver.HostBuild({}, backend.compiler(), tc.compiler_identity(backend.compiler()), "native", "O3", "cpu", "/facts", "")
    assert host_driver._run_witness(driver, backend.masked_call_witness_source, template) == ("fixed", "")


def test_the_build_template_is_the_library_answer_for_this_machine_whatever_machine_the_spec_names():
    """A spec may name another machine than the one that compiles (its texts are written for that one). The build template is the
    library's answer for THIS machine and compiler: a flag of another machine does not compile here."""
    import dataclasses
    from types import MappingProxyType

    from hawk.compile import toolchain as tc
    from hawk.targets import build_spec

    real = build_spec(("host",), "")
    other = "x86_64" if real.machine.isa == "arm64" else "arm64"
    machine = dataclasses.replace(real.machine, isa=other, facts=MappingProxyType({**real.machine.facts, "vector_isa": "avx512"}))
    spec = dataclasses.replace(real, machine=machine)
    opts = SimpleNamespace(defines=(), extra=())
    got = TARGETS["host"]._host_build(SimpleNamespace(spec=spec), {}, opts).instructions
    stated = {**request._stated(real.machine.isa, real.machine.facts), "opt_level": "O3"}
    line = request._identity(tc.compiler_identity(TARGETS["host"].compiler()))
    # a compiler the library asks the masked-call witness about gets the answer for its verdict
    answers = [json.loads(request._facts("host", line, json.dumps({**stated, **verdict})))["host_build"]
               for verdict in ({}, *({"masked_call_witness": v} for v in ("fixed", "miscompile", "unrun")))]
    assert got in answers


# -- device images in the set library ------------------------------------------------------------

#: The stand-in for the device compiler: it reports a release, logs each argv (one argument to a
#: line, then a mark) and writes an image that starts with the ELF magic and holds the source text,
#: so two sources never give one image.
_STAND_IN = r"""#!/bin/sh
if [ "$1" = "--version" ]; then echo "Cuda compilation tools, release 12.6, V12.6.77"; exit 0; fi
printf '%s\n' "$@" >> "$STAND_IN_LOG"
echo -- >> "$STAND_IN_LOG"
while [ $# -gt 0 ]; do
  case "$1" in -o) out=$2; shift ;; -MF) dep=$2; shift ;; *.cu) src=$1 ;; esac
  shift
done
{ printf '\177ELF'; cat "$src"; } > "$out"
echo "$out: $src" > "$dep"
"""
DEVICE_BUILD = {"targets": ("cuda", "host"), "options": {"cuda.arch": "sm_80"}}


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """The stand-in device compiler, found as ``$HAWK_NVCC``; its log file's path."""
    script, log = tmp_path / "nvcc", tmp_path / "nvcc.log"
    script.write_text(_STAND_IN)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HAWK_NVCC", str(script))
    monkeypatch.setenv("STAND_IN_LOG", str(log))
    monkeypatch.delenv("HAWK_DEVICE_COMPILER", raising=False)
    monkeypatch.setattr(device_toolchain, "toolkit_archs",
                        lambda kind=None: ("stand-in toolkit", (50, 90)))

    def no_driver():
        raise RaptorError("the stand-in machine has no device driver")

    # the image form follows the driver's release: these rows state a machine with none
    monkeypatch.setattr(device_nvrtc, "driver_cuda_version", no_driver)
    return log


def _image_held_by(library, unit: str) -> tuple:
    """``(size word, bytes)`` of the image a library holds under the symbols of ``unit``'s record."""
    size = ctypes.c_uint64.in_dll(library, f"{unit}__image_size").value
    start = ctypes.addressof(ctypes.c_char.in_dll(library, f"{unit}__image"))
    return size, ctypes.string_at(start, size)


def _compiles(monkeypatch) -> list:
    """The labels of every compiler run from now on."""
    labels, real = [], drivers.run_compiler
    monkeypatch.setattr(drivers, "run_compiler",
                        lambda argv, what, *target: labels.append(what) or real(argv, what, *target))
    return labels


def test_the_image_symbols_of_every_device_unit_resolve_in_the_set_library_and_hold_the_image_file(
        tmp_path, stand_in):
    cache = str(tmp_path / "cache")
    bundle = build_bundle(list(KERNELS), tmp_path / "a", cache_dir=cache, **DEVICE_BUILD)
    library = ctypes.CDLL(str(bundle.directory / "units.so"))
    for artifact in bundle.artifacts:
        image = artifact.entries["cuda"].artifact.read_bytes()
        assert _image_held_by(library, artifact.name) == (len(image), image)
        assert ctypes.cast(getattr(library, f"{artifact.name}_host"), ctypes.c_void_p).value


def test_a_second_build_of_a_set_with_device_units_runs_no_compiler_and_no_assembler(
        tmp_path, stand_in, monkeypatch):
    cache = str(tmp_path / "cache")
    build_bundle(list(KERNELS), tmp_path / "a", cache_dir=cache, **DEVICE_BUILD)
    reset_unit_memo()
    labels = _compiles(monkeypatch)
    second = build_bundle(list(KERNELS), tmp_path / "b", cache_dir=cache, **DEVICE_BUILD)
    assert labels == []
    assert all(e.hit for a in second.artifacts for e in a.entries.values())


def test_a_change_to_one_unit_rebuilds_its_image_object_the_set_text_and_the_link_and_nothing_else(
        tmp_path, stand_in, monkeypatch):
    cache = str(tmp_path / "cache")
    first = [D.axpb, D.vec3_scale, D.fraction]
    build_bundle(first, tmp_path / "a", cache_dir=cache, **DEVICE_BUILD)
    reset_unit_memo()
    labels = _compiles(monkeypatch)
    build_bundle([D.axpb, D.vec3_scale, D.speed], tmp_path / "b", cache_dir=cache, **DEVICE_BUILD)
    assert sorted(labels) == sorted(["cuda compile of 'speed'", "host compile of 'speed'",
                                     "host compile of 'speed__image'",
                                     "host compile of 'set'", "host link of 'units.so'"])


def test_a_set_of_device_units_alone_has_a_library_the_manifest_names_and_no_link_for_a_unit(
        tmp_path, stand_in):
    bundle = build_bundle(list(KERNELS), tmp_path / "a", targets=("cuda",),
                          options={"cuda.arch": "sm_80"}, cache_dir=str(tmp_path / "cache"))
    library = bundle.directory / "units.so"
    assert library.is_file() and not library.is_symlink()
    assert not [p for p in bundle.directory.iterdir() if p.is_symlink()]
    assert library in bundle.files() and {a.library for a in bundle.artifacts} == {library}
    for plugin in bundle.manifest["plugins"]:
        assert plugin["library"] == "units.so" and plugin["artifact"] == f"{plugin['id']}.cubin"
    held = ctypes.CDLL(str(library))
    for artifact in bundle.artifacts:
        image = artifact.entries["cuda"].artifact.read_bytes()
        assert _image_held_by(held, artifact.name) == (len(image), image)


def test_the_image_argv_a_build_runs_is_the_librarys_template_with_its_places_filled(
        tmp_path, stand_in):
    build_bundle([D.axpb], tmp_path / "a", cache_dir=str(tmp_path / "cache"), **DEVICE_BUILD)
    ran = stand_in.read_text().split("--\n")[0].splitlines()
    identity = host_toolchain.compiler_identity(os.environ["HAWK_NVCC"])
    template = request.device_build(identity, arch="sm_80", release=None, driver=None,
                                    level="O3")["compile"]
    aether, eagle = host_toolchain.include_roots()
    filled = {"{include:aether}": aether, "{include:eagle}": eagle,
              "-I{include:facts}": f"-I{facts.include_dir({'name': 'cuda'})}"}
    unplaced = ("{depfile}", "{source}", "{output}")    # the files of the compile: only that each is given
    expected = [filled.get(element, element) for element in template[1:] if element != "{defines}"]
    assert len(ran) == len(expected)
    assert [(got if want not in unplaced else want) for got, want in zip(ran, expected)] == expected
