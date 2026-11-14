# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dead ends of a first run name the next step."""

from __future__ import annotations

import logging

import pytest

from hawk.compile import toolchain as tc
import ospreys.cpu.toolchain as ctc
from raptor.errors import RaptorError as HawkError


def test_a_missing_host_compiler_names_the_install_and_hawk_cxx(monkeypatch, tmp_path):
    monkeypatch.setenv("HAWK_CXX", str(tmp_path / "no-such-c++"))
    with pytest.raises(HawkError, match=r"install g\+\+ 11\+ .*HAWK_CXX=/path/to/c\+\+"):
        ctc.host_compiler()
    monkeypatch.delenv("HAWK_CXX")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "ziglang", None)
    with pytest.raises(HawkError, match=r"install g\+\+ 11\+ .*HAWK_CXX=/path/to/c\+\+"):
        ctc.host_compiler()


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


def _compile_with(cxx, tmp_path, monkeypatch):
    from hawk.compile import drivers
    from hawk.compile.drivers import CompileOptions

    monkeypatch.setenv("HAWK_CXX", str(cxx))
    return pytest.raises(HawkError), lambda: drivers.compile_source(
        'extern "C" int hawk_probe_cc() { return 1; }\n', "probe_cc",
        CompileOptions(target="host", cache_dir=str(tmp_path / "c")))


def test_a_missing_compiler_says_not_found_not_the_address_cap(monkeypatch, tmp_path):
    from hawk.compile import drivers

    monkeypatch.setenv("HAWK_COMPILE_ADDRESS_CAP", str(12 * 1024**3))
    with pytest.raises(HawkError) as excinfo:
        argv = ["/no/such/dir/c++", "-c", "x.cpp"]
        drivers._refuse(drivers.run_compiler(argv, "host compile", "host"), argv, "host compile", "", tmp_path)
    msg = str(excinfo.value)
    assert "compiler not found or not executable: /no/such/dir/c++" in msg, msg
    assert "$HAWK_CXX" in msg and "address-space cap" not in msg, msg


def test_a_compiler_script_exiting_127_says_not_found_not_the_address_cap(monkeypatch, tmp_path):
    script = tmp_path / "wrapper-c++"
    script.write_text("#!/bin/sh\nexec /no/such/clang++ \"$@\"\n")
    script.chmod(0o755)
    ctx, run = _compile_with(script, tmp_path, monkeypatch)
    monkeypatch.setenv("HAWK_COMPILE_ADDRESS_CAP", str(12 * 1024**3))
    with ctx as excinfo:
        run()
    msg = str(excinfo.value)
    assert f"compiler not found or not executable: {script}" in msg, msg
    assert "address-space cap" not in msg, msg
