# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The exact host profiles stay exact: one table spells the contraction flag, and a profile's compile line carries what it says.

``native``, ``portable`` and ``native-vector-math-nofma`` contract nothing (``-ffp-contract=off``); the shipped default,
``native-vector-math``, lets the compiler fuse (``-ffp-contract=fast``). Flipping an exact entry makes the bit-for-bit
profiles differ from NumPy without any other row moving, so the table and every place that spells the flag are read here.
"""

from __future__ import annotations

import ast
import pathlib
import platform

import pytest

from hawk.compile import toolchain as tc
import osprey_cpu.toolchain as ctc

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXACT = ("native", "portable", "native-vector-math-nofma")
DEFAULT = "native-vector-math"
#: The one place a HOST profile's contraction is spelled, and the embedded reference build line (not a host profile).
HOST_SITE = "staging/osprey_cpu/toolchain.py"
REFERENCE_SITES = {"extensions/hawk_c/hawk_c/toolchain.py": "-ffp-contract=off"}


def _flags(profile, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ctc, "host_compiler", lambda: "/fake/g++")
    monkeypatch.setattr(ctc, "compiler_is_gcc", lambda c: True)
    monkeypatch.setattr(ctc, "masked_call_flag", lambda flags, extra=(): ["-DAETHER_HVM_MASKED=1"])
    return ctc.host_codegen_flags(profile)


@pytest.mark.parametrize("profile", EXACT)
def test_an_exact_profile_compiles_with_contraction_off(profile, monkeypatch):
    flags = _flags(profile, monkeypatch)
    assert "-ffp-contract=off" in flags and not [f for f in flags if f.startswith("-ffp-contract=") and f != "-ffp-contract=off"], flags


def test_the_default_profile_compiles_with_contraction_fast(monkeypatch):
    flags = _flags(DEFAULT, monkeypatch)
    assert [f for f in flags if f.startswith("-ffp-contract=")] == ["-ffp-contract=fast"], flags


def _spellings(path: pathlib.Path) -> list:
    """The string constants of ``path`` that spell ``ffp-contract`` (docstrings and comments are prose, not spellings)."""
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "ffp-contract" in n.value and id(n) not in docs]


def test_the_host_table_is_the_only_place_a_host_contraction_flag_is_spelled():
    found = {}
    for sub in ("hawk", "extensions", "staging"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if spelled := _spellings(path):
                found[str(path.relative_to(ROOT))] = spelled
    assert set(found) == {HOST_SITE, *REFERENCE_SITES}, f"a contraction flag is spelled in a new place: {sorted(found)}"
    assert sorted(found[HOST_SITE]) == sorted(ctc._HOST_CONTRACTION.values())
    for site, flag in REFERENCE_SITES.items():        # the embedded reference build line
        assert found[site] == [flag], (site, found[site])


def test_the_table_names_exactly_the_profiles_and_only_the_default_fuses():
    assert set(ctc._HOST_CONTRACTION) == {*EXACT, DEFAULT}
    assert {p for p, f in ctc._HOST_CONTRACTION.items() if f != "-ffp-contract=off"} == {DEFAULT}
