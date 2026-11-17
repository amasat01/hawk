# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host half of a build call is ONE shared library: the host units of every kernel of the call are compiled to objects, linked
with the text of their set, and published as one file with a link named for each unit beside it (the runtime still opens a unit by
its name).

What these rows state: the layout of the published files, that every unit's entry resolves from the one library, that a set of
units of two scalar modes links and runs both, that two units of one name are refused before anything is compiled, and that a
second build of the same set runs no compiler."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
from types import SimpleNamespace

import _deployable as D
import numpy as np
import pytest

import hawk.runtime as R
from hawk.artifact import build, build_bundle, request
from hawk.compile import host as host_driver
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
    result = TARGETS["host"].link(lowered, options, [(p64.entry[0], m64), (p32.entry[0], m32)], "mixed.so")
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
    driver = host_driver.HostBuild(build, "c++", "c++ || x || 0", "native-vector-math", "O3", "cpu", "/facts", "facts:x")
    plan = {"places": host_driver._places(driver, door=False), "template": build["compile_unit"], "compiler": "c++", "defines": ["-DX"],
            "extra": []}
    job = SimpleNamespace(src="/s.cpp", out="/s.o", dep="/s.d", served="/served")
    host_driver._ObjectRow.compile_hook(plan, job, ran.append)
    assert ran[0] == host_driver._argv(build["compile_unit"], {"{compiler}": lambda: "c++", "{source}": lambda: "/s.cpp", "{object}": lambda: "/s.o",
                                                              "{depfile}": lambda: "/s.d", "{include:served}": lambda: "/served",
                                                              "{include:facts}": lambda: "/facts", "{include:aether}": lambda: "",
                                                              "{include:eagle}": lambda: ""}, {"{defines}": ["-DX"]})
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
    assert got == json.loads(request._facts("host", line, json.dumps(stated)))["host_build"]
