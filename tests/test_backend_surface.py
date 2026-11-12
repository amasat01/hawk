# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The driver reaches a backend through ``emit`` and ``build`` only, over one backend-neutral ``Lowered``."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent / "hawk"

#: the names the driver may still call on a backend outside a build's two calls, each with why it cannot fold in
ALLOWED = {
    "emit": "the build's cheap half",
    "build": "the build's costly half",
    "key_terms": "the unit key is computed before anything is emitted, and a memo hit must emit nothing",
    "own_options": "the unit key hashes the options a backend takes, before any emit",
    "refuse_kernels": "refuses a whole kernel list before the key, the memo or any emit",
    "resolve_arch": "the device arch is resolved once, through the device backend, before any emit",
    "report_file": "hawk.runtime finds the report of an existing unit with no build",
    "probe_device": "the device cards read the machine with no build",
    "process_env": "the compiler subprocess environment of every backend, outside any one build",
}
RECEIVERS = {"need", "need_device"}
NAMES = {"row", "backend"}


def _is_backend(node) -> bool:
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id == "TARGETS"
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id in RECEIVERS
    return isinstance(node, ast.Name) and node.id in NAMES


def _calls(files) -> dict:
    found = {}
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _is_backend(node.func.value):
                found.setdefault(node.func.attr, []).append(f"{path.name}:{node.lineno}")
    return found


@pytest.mark.repo_local  # reads the source tree
def test_the_driver_calls_a_backend_only_through_its_surface():
    found = _calls([*(ROOT / "artifact").glob("*.py"), ROOT / "runtime.py"])
    assert {"emit", "build"} <= set(found)
    assert not set(found) - set(ALLOWED), {k: v for k, v in found.items() if k not in ALLOWED}
    assert not {"package", "facts", "render", "compile"} & set(found)


def test_the_census_sees_a_package_call():
    probe = Path(__file__).parent / "_census_probe.tmp"
    probe.write_text("def f(t):\n    return TARGETS[t].package(1)\n")
    try:
        assert "package" in _calls([probe])
    finally:
        probe.unlink()


def test_lowered_names_no_backend():
    from hawk.backends import Lowered

    names = [f.name for f in fields(Lowered)]
    assert names == ["kernel", "name", "smode", "kind", "body", "segment", "one_step", "box", "active", "rows"]


def test_a_backend_reads_only_its_own_namespace():
    from hawk.backends import need

    assert need("host").own_options({"cuda.literal_table": True}) == {}
