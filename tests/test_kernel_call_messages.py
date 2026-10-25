# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dead ends of a first run name the next step."""

from __future__ import annotations

import logging

import pytest

from hawk.compile import toolchain as tc
from hawk.ir import HawkError


def test_a_missing_host_compiler_names_the_install_and_hawk_cxx(monkeypatch, tmp_path):
    monkeypatch.setenv("HAWK_CXX", str(tmp_path / "no-such-c++"))
    with pytest.raises(HawkError, match=r"install g\+\+ 11\+ .*HAWK_CXX=/path/to/c\+\+"):
        tc.host_compiler()
    monkeypatch.delenv("HAWK_CXX")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "ziglang", None)
    with pytest.raises(HawkError, match=r"install g\+\+ 11\+ .*HAWK_CXX=/path/to/c\+\+"):
        tc.host_compiler()


def test_a_compile_error_ends_with_the_build_directory_and_hawk_cxx(tmp_path):
    hint = tc.compile_hint(tmp_path)
    assert str(tmp_path) in hint and "HAWK_CXX=/path/to/c++" in hint


def test_the_first_build_logs_one_info_line(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="hawk"):
        tc.announce_build("scale", "host", tmp_path)
    (record,) = caplog.records
    assert record.getMessage() == (
        f"building scale (host) in {tmp_path} — delete the directory to rebuild")


def test_a_first_build_logs_once_and_a_second_logs_nothing(tmp_path, caplog):
    from hawk.compile import drivers
    from hawk.compile.drivers import CompileOptions

    source = 'extern "C" int hawk_probe_once() { return 1; }\n'
    with caplog.at_level(logging.INFO, logger="hawk"):
        for _ in range(2):
            drivers.compile_source(source, "probe_once", CompileOptions(
                target="host", cache_dir=str(tmp_path)))
    lines = [r.getMessage() for r in caplog.records if "building probe_once" in r.getMessage()]
    assert len(lines) == 1 and lines[0].startswith("building probe_once (host) in ")
