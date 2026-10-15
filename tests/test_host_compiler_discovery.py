# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host-compiler discovery order: $HAWK_CXX > $CXX > g++ on PATH > zig."""

import os
import stat
import sys

import pytest

from hawk.compile import toolchain as tc
from hawk.ir import HawkError


def _exe(directory, name):
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HAWK_CXX", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return tmp_path


def test_cxx_honoured_without_gxx_on_path(clean_env, monkeypatch):
    bindir = clean_env / "bin"
    bindir.mkdir()
    exe = _exe(bindir, "x86_64-conda-linux-gnu-g++")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("CXX", "x86_64-conda-linux-gnu-g++ -O2")
    assert tc.host_compiler() == exe


def test_cxx_absolute_path(clean_env, monkeypatch):
    exe = _exe(clean_env, "mycxx")
    monkeypatch.setenv("CXX", exe)
    assert tc.host_compiler() == exe


def test_hawk_cxx_beats_cxx_and_path_gxx(clean_env, monkeypatch):
    bindir = clean_env / "bin"
    bindir.mkdir()
    gxx = _exe(bindir, "g++")
    other = _exe(bindir, "other-cxx")
    hawk = _exe(bindir, "hawk-cxx")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("CXX", other)
    monkeypatch.setenv("HAWK_CXX", hawk)
    assert tc.host_compiler() == hawk
    monkeypatch.delenv("HAWK_CXX")
    assert tc.host_compiler() == other      # CXX beats g++ on PATH
    monkeypatch.delenv("CXX")
    assert tc.host_compiler() == gxx


def test_unresolvable_cxx_falls_through(clean_env, monkeypatch):
    bindir = clean_env / "bin"
    bindir.mkdir()
    gxx = _exe(bindir, "g++")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("CXX", "no-such-compiler-xyz --flag")
    assert tc.host_compiler() == gxx


def test_not_found_message_lists_every_candidate(clean_env, monkeypatch):
    monkeypatch.setenv("CXX", "no-such-compiler-xyz --flag")
    monkeypatch.setitem(sys.modules, "ziglang", None)   # import raises ImportError
    with pytest.raises(HawkError) as err:
        tc.host_compiler()
    msg = str(err.value)
    for needle in ("HAWK_CXX", "CXX", "no-such-compiler-xyz --flag", "g++", "ziglang"):
        assert needle in msg
