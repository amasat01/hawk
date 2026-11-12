# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The variant's generated header (variants/<name>/raptor_variant.h): regenerate-and-diff, the number is the variant's, and a unit that
uses linalg or rays includes it and reaches the compiler through the ``-I`` of its directory."""
import os
import subprocess
import sys
from pathlib import Path

from hawk import variants

TREE = Path(__file__).resolve().parent.parent


def test_generated_headers_do_not_drift():
    done = subprocess.run([sys.executable, str(TREE / "tools" / "gen_variant_header.py"), "--check"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert done.returncode == 0, done.stdout + done.stderr


def test_header_carries_the_variants_number():
    for name in ("cpu", "x86", "arm", "simt", "cuda"):
        text = (TREE / "variants" / name / "raptor_variant.h").read_text()
        assert f"#define RAPTOR_LINALG_MAX_DIM {variants.load(name)['linalg']['max_dim']}\n" in text


def test_include_flag_names_the_variant_directory():
    assert variants.include_flags(variants.load("x86")) == (f"-I{variants.location('x86')}",)
    assert variants.include_flags({}) == ()
