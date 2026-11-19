# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dead ends of a first run name the next step: no host compiler, a compile that fails, a build that says where it works."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import pytest

import hawk
from hawk import HawkError, Scalar, ToolchainAbsent
from hawk.artifact import build_bundle


@hawk.kernel
def scaled(x: Scalar) -> Scalar:
    return x * 2.0


def build_on_host(tmp_path):
    """Build ``scaled`` for the host under a cache of its own, so no earlier build answers for it."""
    return build_bundle(
        [scaled],
        tmp_path / "bundle",
        targets=("host",),
        cache_dir=str(tmp_path / "cache"),
    )


def write_compiler(path, body):
    """Write an executable shell script at ``path``."""
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_a_host_compiler_named_by_hawk_cxx_that_does_not_exist_is_refused_naming_the_variable_and_the_way_out(
    monkeypatch, tmp_path
):
    """The refusal quotes the value of ``HAWK_CXX`` it could not find and says to set it to a C++23 compiler."""
    missing = tmp_path / "no-such-c++"
    monkeypatch.setenv("HAWK_CXX", str(missing))
    with pytest.raises(
        ToolchainAbsent, match=r"\$HAWK_CXX=.*no-such-c\+\+.* not found on PATH"
    ) as refused:
        build_on_host(tmp_path)
    assert "set HAWK_CXX to a C++23 compiler, or CXX, or put g++ on PATH" in str(
        refused.value
    )


def test_a_machine_with_no_host_compiler_is_refused_listing_each_place_looked_in_order(
    monkeypatch, tmp_path
):
    """With both variables unset and an empty ``PATH`` the refusal lists ``HAWK_CXX``, ``CXX`` and ``g++`` in that order."""
    monkeypatch.delenv("HAWK_CXX", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(
        ToolchainAbsent,
        match=r"no compiler found\. Tried, in order: \$HAWK_CXX \(unset\)",
    ) as refused:
        build_on_host(tmp_path)
    assert "$CXX (unset), g++ on PATH (absent)" in str(refused.value)


def test_a_failed_tool_run_keeps_its_command_and_complaint_and_ends_with_its_folder_and_the_way_out(
    monkeypatch, tmp_path
):
    """A compiler that answers ``--version`` and fails every run gives a refusal with the command and the error text, ending in
    the folder the run was in and the tool table's own way out (set ``HAWK_CXX``, ``CXX`` or put ``g++`` on ``PATH``)."""
    failing = tmp_path / "failing-c++"
    write_compiler(
        failing,
        'if [ "$1" = "--version" ]; then echo "g++ 13.0"; exit 0; fi\necho "error: boom" >&2\nexit 1',
    )
    monkeypatch.setenv("HAWK_CXX", str(failing))
    with pytest.raises(HawkError) as refused:
        build_on_host(tmp_path)
    message = str(refused.value)
    assert f"$ {failing} " in message and "error: boom" in message
    folder, _, way_out = message.rpartition("ran in ")[2].partition("; ")
    assert Path(folder).name.startswith("toolkit-probe-")
    assert way_out == "set HAWK_CXX to a C++23 compiler, or CXX, or put g++ on PATH"


def test_a_first_build_logs_one_line_naming_its_directory_and_a_second_build_logs_nothing(
    tmp_path, caplog
):
    """The first build of a kernel states where it builds and how to rebuild; the repeat is served from the cache silently."""
    with caplog.at_level(logging.INFO, logger="hawk"):
        build_on_host(tmp_path)
        first = [r.getMessage() for r in caplog.records]
        build_on_host(tmp_path)
        second = [r.getMessage() for r in caplog.records][len(first) :]
    assert len(first) == 1 and "delete the directory to rebuild" in first[0]
    assert second == []
