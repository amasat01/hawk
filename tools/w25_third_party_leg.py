# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The third-party leg of the install smoke: ``python tools/w25_third_party_leg.py <python of an install venv> <work dir>``.

Builds a tiny distribution ``acme-toy`` (a variant ``acme`` that extends ``cpu`` with ONE different number, and a backend ``acme_toy`` that
names it), installs its wheel into that environment and, with ``python -I`` from an empty folder, builds one kernel for ``host`` and for
``acme_toy``: the emitted texts must differ, and only where the number says (the fold threshold: a 12-term sum is unrolled on one and
rolled on the other). Exit code 0 only on success."""

from __future__ import annotations

import difflib
import json
import subprocess
import sys
from pathlib import Path

CHECK = '''
import hashlib, os, sys, tempfile
work = tempfile.mkdtemp()
os.environ["HAWK_CACHE_DIR"] = os.path.join(work, "cache")
import hawk
from hawk import Mutable, Vector, Scalar, registry, variants
from hawk.math import dot

@hawk.kernel
def d12(u: Vector[12], v: Vector[12], y: Mutable[Scalar]):
    y = dot(u, v)

text = {}
for target in ("host", "acme_toy"):          # two host-entered rows are not one bundle: one build each
    b = hawk.build([d12], os.path.join(work, "out_" + target), targets=(target,), cache_dir=os.path.join(work, "cache"))
    (art,) = b.artifacts
    (entry,) = art.entries.values()
    text[target] = entry.source.read_text()
print("variant acme at", variants.location("acme"), "fold", variants.load("acme")["form"], "cpu fold", variants.load("cpu")["form"])
print("registry:", registry.backend_ids(), registry.refused())
open(sys.argv[1], "w").write(__import__("json").dumps(text))
'''


def main(python: str, work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    src = work / "src"
    base = json.loads(subprocess.run([python, "-I", "-c", "import json, hawk.variants as v; print(json.dumps({'arm': dict(v.load('arm')), 'header': (v.location('arm') / 'raptor_variant.h').read_text()}, default=dict))"],
                                     capture_output=True, text=True, check=True, cwd=work).stdout)
    arm, header = base["arm"], base["header"]
    (src / "acme_toy_variants" / "acme").mkdir(parents=True, exist_ok=True)
    (src / "acme_toy").mkdir(exist_ok=True)
    (src / "acme_toy_variants" / "__init__.py").write_text("")
    (src / "acme_toy_variants" / "acme" / "__init__.py").write_text("")
    data = {"name": "acme", "extends": "cpu", "isa": arm["isa"], "fusion_host": arm["fusion_host"], "form": {"rolled_fold_min": 8}}
    (src / "acme_toy_variants" / "acme" / "data.py").write_text(repr(data) + "\n")
    (src / "acme_toy_variants" / "acme" / "raptor_variant.h").write_text(header.replace("ARM", "ACME").replace('"arm"', '"acme"').replace("/arm/", "/acme/"))
    (src / "acme_toy" / "__init__.py").write_text("")
    (src / "acme_toy" / "backend.py").write_text(
        "from osprey_cpu.backend import Host\n\n\nclass Toy(Host):\n    id = 'acme_toy'\n    about = 'a third-party CPU backend'\n\n\nBACKEND = Toy()\n")
    (src / "acme_toy" / "extension.json").write_text(json.dumps({
        "name": "acme-toy", "version": "0.0", "contract": "1", "requires": [],
        "backends": [{"name": "acme_toy", "ref": "acme_toy.backend:BACKEND", "about": "a third-party CPU backend", "device": False,
                      "builds": True, "variant": "acme", "components": ["lowering"], "priority": 7, "options": ["profile"],
                      "policy": ["emit_form"], "reads": ["form.rolled_fold_min"]}]}))
    (src / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=64"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "acme-toy"\nversion = "0.0"\n\n'
        '[project.entry-points."raptor.extensions"]\nacme-toy = "acme_toy:extension.json"\n\n'
        '[project.entry-points."raptor.variants"]\nacme = "acme_toy_variants.acme"\n\n'
        '[tool.setuptools]\npackages = ["acme_toy", "acme_toy_variants", "acme_toy_variants.acme"]\n\n'
        '[tool.setuptools.package-data]\nacme_toy = ["extension.json"]\n"acme_toy_variants.acme" = ["raptor_variant.h"]\n')
    wheels = work / "wheels"
    subprocess.run([python, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(wheels), str(src)], check=True, cwd=work)
    wheel = next(wheels.glob("acme_toy-*.whl"))
    subprocess.run([python, "-m", "pip", "install", "--no-deps", "-q", str(wheel)], check=True, cwd=work)
    empty = work / "empty"
    empty.mkdir(exist_ok=True)
    out = work / "texts.json"
    done = subprocess.run([python, "-I", "-c", CHECK, str(out)], capture_output=True, text=True, cwd=empty)
    print(done.stdout + done.stderr[-2500:])
    if done.returncode:
        print("THIRD-PARTY LEG FAILED: the build did not run")
        return 1
    text = json.loads(out.read_text())
    host, toy = text["host"], text["acme_toy"]
    changed = [line for line in difflib.unified_diff(host.splitlines(), toy.splitlines(), "host", "acme_toy", n=0) if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    print("\n".join(changed[:24]))
    if host == toy or not changed:
        print("THIRD-PARTY LEG FAILED: the texts do not differ although the fold threshold does")
        return 1
    print(f"THIRD-PARTY LEG OK: {len(changed)} differing lines of {len(host.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], Path(sys.argv[2]).resolve()))
