# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The generated header of a raptor wing (staging/raptor/wings/<name>/raptor_facts.h): regenerate-and-diff, the number is the one the
wing's chain states, and a unit that uses linalg or rays includes it and reaches the compiler through the ``-I`` of its folder."""
import os
import subprocess
import sys
from pathlib import Path

from hawk import facts

TREE = Path(__file__).resolve().parent.parent


def test_generated_headers_do_not_drift():
    done = subprocess.run([sys.executable, str(TREE / "tools" / "gen_facts_header.py"), "--check"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert done.returncode == 0, done.stdout + done.stderr


def test_header_carries_the_wings_number():
    for name in ("cpu", "x86", "arm", "simt", "cuda"):
        text = (TREE / "staging" / "raptor" / "wings" / name / "raptor_facts.h").read_text()
        assert f"#define RAPTOR_LINALG_MAX_DIM {facts.load(name)['linalg']['max_dim']}\n" in text


def test_include_flag_names_the_wing_folder():
    assert facts.include_flags(facts.load("x86")) == (f"-I{facts.location('x86')}",)
    assert facts.include_flags({}) == ()
