# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Classify the paths a change touches against the confinement rule of a backend's own work: a backend's optimization touches only
its leaf and its own wings; a shared wing changes only for a feature every backend using it gains; anything else is a
violation to be explained.

    python tools/confined_diff.py <backend> [<base> [<head>]]        (reads `git diff --name-only <base>..<head>`; default main..HEAD)
    git diff --name-only A..B | python tools/confined_diff.py <backend> -

Rules are data (RULES below): per backend, the path prefixes that are its leaf and its own wings, and the shared wings it names.
Exit status: 0 when every path is CONFINED, 1 when a path is SHARED (needs the wing-wide reason in the change's description),
2 when a path is a VIOLATION."""

from __future__ import annotations

import subprocess
import sys

#: backend -> (its own prefixes, the shared wings it may change with a stated reason)
RULES = {
    "cuda": (("staging/ospreys/cuda/", "staging/raptor/wings/cuda/"), ("staging/raptor/wings/simt/",)),
    "c": (("staging/ospreys/c/",), ()),
    "host": (("staging/raptor/wings/cpu/", "staging/raptor/wings/x86/", "staging/raptor/wings/arm/"), ()),
}
#: paths any change may carry with it
NEUTRAL = ("CHANGELOG.md", "docs/")


def classify(backend: str, path: str) -> str:
    own, shared = RULES[backend]
    if path.startswith(own):
        return "CONFINED"
    if path.startswith("tests/") or path.startswith(NEUTRAL):
        return "CONFINED"
    if path.startswith(shared):
        return "SHARED"
    return "VIOLATION"


def main(argv) -> int:
    if len(argv) < 2 or argv[1] not in RULES:
        print(f"usage: confined_diff.py <{'|'.join(RULES)}> [<base> [<head>] | -]", file=sys.stderr)
        return 64
    backend = argv[1]
    if len(argv) > 2 and argv[2] == "-":
        paths = [line.strip() for line in sys.stdin if line.strip()]
    else:
        base = argv[2] if len(argv) > 2 else "main"
        head = argv[3] if len(argv) > 3 else "HEAD"
        paths = subprocess.run(["git", "diff", "--name-only", f"{base}..{head}"], check=True, capture_output=True,
                               text=True).stdout.split()
    worst = 0
    for path in paths:
        kind = classify(backend, path)
        worst = max(worst, {"CONFINED": 0, "SHARED": 1, "VIOLATION": 2}[kind])
        print(f"{kind:9} {path}")
    print(f"{backend}: {len(paths)} paths, verdict {('CONFINED', 'SHARED (state the wing-wide reason)', 'VIOLATION')[worst]}")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv))
