# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Publishing a deployment unit is content-addressed and idempotent.

A repeat build of an unchanged unit is served by the OSPREY's store (no tool runs) and gives the same unit digest; into
another directory it lays down the same bytes; into the directory that already holds it, it leaves the files alone; a directory whose file was altered is laid down again.
"""

from __future__ import annotations

import logging

import hawk
from hawk import Scalar
from hawk.artifact import build_bundle


@hawk.kernel
def scaled(x: Scalar) -> Scalar:
    return x * 2.0


def build_into(tmp_path, name):
    """Build ``scaled`` for the host into ``tmp_path / name`` over one store shared by every call of the test."""
    return build_bundle(
        [scaled], tmp_path / name, targets=("host",), cache_dir=str(tmp_path / "store")
    )


def bytes_of(directory):
    """The files of ``directory`` as ``{name: bytes}``."""
    return {p.name: p.read_bytes() for p in directory.iterdir()}


def test_a_repeat_build_into_another_directory_holds_the_same_bytes_and_digest_and_runs_no_tool(
    tmp_path, caplog
):
    """The unit is named by its content: the second directory is a copy of the first and the second build is a store hit."""
    first = build_into(tmp_path, "first")
    with caplog.at_level(logging.INFO, logger="hawk"):
        second = build_into(tmp_path, "second")
    assert first.digest == second.digest
    assert bytes_of(first.directory) == bytes_of(second.directory)
    assert [r for r in caplog.records if "building" in r.getMessage()] == []


def test_a_repeat_build_into_the_directory_that_already_holds_the_unit_leaves_every_file_untouched(
    tmp_path,
):
    """The files keep their modification times: a directory holding the same bytes is recognised, not rewritten."""
    build_into(tmp_path, "unit")
    before = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "unit").iterdir()}
    build_into(tmp_path, "unit")
    assert {
        p.name: p.stat().st_mtime_ns for p in (tmp_path / "unit").iterdir()
    } == before


def test_a_directory_whose_file_was_altered_is_rewritten_to_the_unit(tmp_path):
    """A file whose bytes differ from the unit's is not the unit: the publish lays every file down again."""
    unit = build_into(tmp_path, "unit").directory
    whole = bytes_of(unit)
    victim = next(p for p in unit.iterdir() if p.name.endswith(".json") and not p.name.startswith("."))
    victim.write_bytes(b"altered")
    build_into(tmp_path, "unit")
    assert bytes_of(unit) == whole


def test_a_reader_of_a_unit_file_that_is_being_rewritten_sees_the_old_bytes_or_the_whole_new_ones(
    tmp_path,
):
    """Files are written whole to a private sibling and renamed into place: no read finds a partial file."""
    import threading

    from hawk.artifact.bundle import UnitFile

    old, new = b"o" * 2_000_000, b"n" * 2_000_000
    (tmp_path / "unit.bin").write_bytes(old)
    stop = threading.Event()
    seen = set()

    def read():
        while not stop.is_set():
            seen.add((tmp_path / "unit.bin").read_bytes())

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for round_ in range(200):
            UnitFile("unit.bin", "d", blob=new if round_ % 2 == 0 else old).write(
                tmp_path
            )
    finally:
        stop.set()
        reader.join()
    assert seen <= {old, new} and list(tmp_path.glob(".*.tmp")) == []
