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

import _deployable as D
import pytest
from conftest import sidecar_of

from hawk.artifact import build, build_bundle, describe, emit_source
from hawk import _ir
from hawk.errors import HawkError

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


@pytest.mark.parametrize("first_is_the_bundle", [True, False])
def test_a_library_is_named_by_the_build_that_asks_whatever_build_filled_the_cache(tmp_path, first_is_the_bundle):
    """Oracle: a set of one unit and a single-kernel build link the same objects with the same text, so the cache serves both one
    library; each published directory still names it as its own call does (the bundle's name, the unit's name)."""
    cache = str(tmp_path / "cache")
    calls = {"bundle": lambda: build_bundle([D.axpb], tmp_path / "bundle", targets=("host",), cache_dir=cache),
             "single": lambda: build(D.axpb, tmp_path / "single", targets=("host",), cache_dir=cache)}
    for call in ("bundle", "single") if first_is_the_bundle else ("single", "bundle"):
        calls[call]()
    assert (tmp_path / "bundle" / "units.so").is_file() and (tmp_path / "bundle" / "axpb.so").is_symlink()
    assert (tmp_path / "single" / "axpb.so").is_file() and not (tmp_path / "single" / "axpb.so").is_symlink()
    assert not (tmp_path / "single" / "units.so").exists()








class _UnitMeta(ctypes.Structure):
    """The head of ``struct eg_unit_meta`` (eagle/door/eg_door.h) up to its stages."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("name", ctypes.c_char_p),
                ("digest", ctypes.c_uint8 * 32), ("pattern", ctypes.c_int32), ("scalar_type", ctypes.c_int32),
                ("host_entry", ctypes.c_char_p), ("device_image", ctypes.c_char_p), ("device_function", ctypes.c_char_p),
                ("device_image_bytes", ctypes.c_uint64), ("device_image_format", ctypes.c_int32), ("n_stages", ctypes.c_int32),
                ("stages", ctypes.c_void_p)]


class _Stage(ctypes.Structure):
    """The head of ``struct eg_unit_stage``: the reference to the unit a stage runs."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("n_cells_in", ctypes.c_int32), ("n_cells_out", ctypes.c_int32),
                ("reserved0", ctypes.c_int32), ("unit", ctypes.POINTER(ctypes.c_void_p))]


class _LibraryMeta(ctypes.Structure):
    """``struct eg_library_meta``: the tree a unit library returns from ``eg_unit_library_meta``."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("schema_major", ctypes.c_uint32), ("schema_minor", ctypes.c_uint32),
                ("n_units", ctypes.c_uint32), ("units", ctypes.POINTER(ctypes.POINTER(_UnitMeta)))]


def _units_of(library) -> dict:
    """``{unit name: [names of the units its stages run]}`` of a library, read off the meta it exports."""
    ask = library.eg_unit_library_meta
    ask.restype = ctypes.POINTER(_LibraryMeta)
    meta = ask().contents
    found = {}
    for at in range(meta.n_units):
        unit = meta.units[at].contents
        stages = ctypes.cast(unit.stages, ctypes.POINTER(ctypes.POINTER(_Stage)))
        # a stage's unit reference starts with its size and a reserved word, then the name
        found[unit.name.decode()] = [ctypes.c_char_p.from_address(ctypes.addressof(stages[i].contents.unit.contents) + 8).value.decode()
                                     for i in range(unit.n_stages)]
    return found


def test_the_set_library_of_a_team_kernel_lists_its_source_unit_and_the_stage_units_it_names(tmp_path, cache_dir):
    """Oracle: the bundle's own artifacts are the units that have a text (the stages); the source unit is the one of the lowering
    library's answer that has none (checked there: ``<kernel>_source``, which names the stages in order). The library lists the
    source unit with its stages and each stage unit with none."""
    from _linalg_kernels import chol_kernel

    kernel = chol_kernel(_ir.new_context(), 40)
    bundle = build_bundle([kernel], tmp_path, targets=("host",), cache_dir=cache_dir, linalg_form="auto")
    listed = _units_of(ctypes.CDLL(str(bundle.directory / "units.so")))
    stage_units = [artifact.name for artifact in bundle.artifacts]
    source = f"{kernel.name}_source"
    assert listed == {source: stage_units, **{name: [] for name in stage_units}}


def test_a_team_kernel_is_refused_by_emit_source_and_build_as_stages_with_the_way_out(tmp_path):
    """Oracle: the answer's own records (the stages of ``<kernel>_source`` carry a team member), not the name."""
    from _linalg_kernels import chol_kernel

    with pytest.raises(HawkError, match=r"worked by a team per lane.*several stages.*call build_bundle"):
        emit_source(chol_kernel(_ir.new_context(), 40), targets=("host",))
    with pytest.raises(HawkError, match=r"worked by a team per lane.*call build_bundle"):
        build(chol_kernel(_ir.new_context(), 40), tmp_path, targets=("host",))


def test_a_value_reduced_over_samples_is_refused_by_emit_source_and_build_as_several_launches_with_the_way_out(tmp_path):
    """Oracle: the answer's stages state no team, so the split is the reduction's."""
    from test_split_stages import standard

    with pytest.raises(HawkError, match=r"reduced over samples.*call build_bundle"):
        emit_source(standard, targets=("host",))
    with pytest.raises(HawkError, match=r"reduced over samples.*call build_bundle"):
        build(standard, tmp_path, targets=("host",))










def test_two_units_of_one_name_are_refused_naming_both_kernels(tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"'axpb' and 'fraction' both write a unit named 'same'"):
        build_bundle([D.axpb, D.fraction], tmp_path, targets=("host",), cache_dir=cache_dir, name="same")
    assert not list(tmp_path.iterdir())




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
    # the sidecar states the same width, by the library, so no reader opens the binary
    assert sidecar_of(narrow, "fraction")["index_bytes"] == 4
    assert sidecar_of(wide, "fraction")["index_bytes"] == 8
    assert describe([D.fraction], targets=("host",))["fraction"]["index_bytes"] == 4
    # a device-only description states it too: the device library states it from its own embedded header, with every tool hidden
    assert describe([D.fraction], targets=("cuda",))["fraction"]["index_bytes"] == 4




def _x86_gcc():
    """``(facts, line)`` of a machine and a GCC the masked-call table lists, as a compile would state them: the library answers for them
    with no compiler present."""
    return {"isa": "x86_64", "vector_isa": "avx2", "host_profile": "native-vector-math"}, "g++ || g++ (GCC) 13.2.0 || 0"




















#: Compilers of the lanes that run this: the verdict the witness gives each (GCC 13.3 miscompiles; the 14.2 backport is fixed).
REAL_WITNESSED = [("/home/mas/mambaforge/bin/x86_64-conda-linux-gnu-g++", "miscompile"),
                  ("/opt/rh/gcc-toolset-14/root/usr/bin/g++", "fixed")]








# -- device images in the set library ------------------------------------------------------------

DEVICE_BUILD = {"targets": ("cuda", "host"), "options": {"cuda.arch": "sm_80"}}


def _image_held_by(library, unit: str) -> tuple:
    """``(size word, bytes)`` of the image a library holds under the symbols of ``unit``'s record."""
    size = ctypes.c_uint64.in_dll(library, f"{unit}__image_size").value
    start = ctypes.addressof(ctypes.c_char.in_dll(library, f"{unit}__image"))
    return size, ctypes.string_at(start, size)












