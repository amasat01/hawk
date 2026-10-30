# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fail when a public name of hawk has no test that mentions it.

    python tools/public_names.py [--list]

A public name is each entry of ``hawk.__all__`` and ``dir(hawk)``'s submodule doors that does not start with an underscore; a test
mentions it when the identifier appears in any ``tests/*.py`` (static: nothing is run). The names that have no test today are
in ci/public_names_untested.txt, one per line as ``hawk.name  # reason``: the file may only shrink, so a name that gained a test
(or left the API) must be removed from it, and a new untested name fails the gate.  --list prints the untested names.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW = ROOT / "ci" / "public_names_untested.txt"


def public_names():
    import hawk

    return sorted({n for n in (*hawk.__all__, *dir(hawk)) if not n.startswith("_")
                   and (n in hawk.__all__ or type(getattr(hawk, n, None)).__name__ == "module")})


def mentioned(names):
    text = "\n".join(p.read_text(errors="replace") for p in sorted((ROOT / "tests").glob("*.py")))
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return {n for n in names if n in words}


def allowed():
    out = {}
    for line in ALLOW.read_text().splitlines() if ALLOW.exists() else ():
        name, _, reason = line.partition("#")
        if name.strip():
            out[name.strip()] = reason.strip()
    return out


def main(argv):
    names = public_names()
    untested = sorted(set(names) - mentioned(names))
    if "--list" in argv:
        print("\n".join(untested))
        return 0
    allow = allowed()
    new = [n for n in untested if n not in allow]
    stale = sorted(n for n in allow if n not in untested)
    noreason = sorted(n for n, r in allow.items() if not r)
    if new:
        print(f"{len(new)} public name(s) have no test mentioning them (add a test, or list them in {ALLOW.name} with a reason):")
        print("\n".join(f"  hawk.{n}" for n in new))
    if stale:
        print(f"{len(stale)} name(s) in {ALLOW.name} are tested now or gone: remove them (the list only shrinks):")
        print("\n".join(f"  {n}" for n in stale))
    if noreason:
        print(f"{len(noreason)} name(s) in {ALLOW.name} lack a reason: {', '.join(noreason)}")
    if new or stale or noreason:
        return 1
    print(f"public names: {len(names)}, tested {len(names) - len(untested)}, allowlisted untested {len(untested)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
