"""Fails when the environment holds something other than the sibling a build was given.

A sibling is given as a source folder (``--source``, a folder with a ``pyproject.toml``) or as a folder of wheels (``--wheels``).
Installing hawk afterwards re-resolves its own requirements, and a sibling whose version is below the floor hawk asks for is then
replaced by the released package of the index. The installed distribution records where it came from (``direct_url.json``,
PEP 610); this script compares that record with what was given:

* a source: the installed distribution of the project's name was installed from that folder;
* a wheel: the installed distribution of the wheel's name has the wheel's version and was installed from a file of its hash.

usage: python tools/check_siblings.py [--source DIR]... [--wheels DIR]
Prints one line per mismatch, naming the sibling, what is installed and the requirement of hawk that the installed version meets or
misses. Exit status: the number of mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path


def project_name(source: Path) -> str:
    """The distribution name a source folder's ``pyproject.toml`` declares."""
    text = (source / "pyproject.toml").read_text()
    found = re.search(r'^name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if found is None:
        raise ValueError(f"{source}/pyproject.toml declares no project name")
    return found.group(1)


def origin(dist) -> dict:
    """The ``direct_url.json`` record of an installed distribution; empty when it came from an index."""
    text = dist.read_text("direct_url.json")
    return json.loads(text) if text else {}


def hawk_requirement(name: str, lookup) -> str:
    """The requirement of hawk on distribution ``name``, or a note that it has none."""
    try:
        requires = lookup("raptor-hawk").requires or []
    except metadata.PackageNotFoundError:
        return "hawk is not installed"
    wanted = re.sub(r"[-_.]+", "-", name).lower()
    for line in requires:
        if (
            re.sub(
                r"[-_.]+", "-", re.split(r"[ ;<>=!~\[(]", line, maxsplit=1)[0]
            ).lower()
            == wanted
        ):
            return line.split(";")[0].strip()
    return "no requirement of hawk"


def installed_as(dist, name: str) -> str:
    """What is installed under ``name``, as one phrase."""
    if dist is None:
        return "nothing"
    where = origin(dist).get("url", "the index")
    return f"{dist.version} from {where}"


def mismatches(sources, wheels, lookup=metadata.distribution) -> list:
    """One line for each given sibling the environment does not hold as given; ``lookup`` finds an installed distribution."""

    def installed(name):
        try:
            return lookup(name)
        except metadata.PackageNotFoundError:
            return None

    lines = []
    for source in sources:
        source = Path(source).resolve()
        name = project_name(source)
        dist = installed(name)
        url = origin(dist).get("url") if dist else None
        if url != source.as_uri():
            lines.append(
                f"{name}: the source {source} is not what is installed ({installed_as(dist, name)}); "
                f"hawk requires {hawk_requirement(name, lookup)}"
            )
    for wheel in sorted(Path(wheels).glob("*.whl")) if wheels else []:
        name, version = wheel.name.split("-")[:2]
        dist = installed(name)
        digest = "sha256=" + hashlib.sha256(wheel.read_bytes()).hexdigest()
        held = origin(dist).get("archive_info", {}).get("hash") if dist else None
        if dist is None or dist.version != version or held != digest:
            lines.append(
                f"{name}: the wheel {wheel} is not what is installed ({installed_as(dist, name)}); "
                f"hawk requires {hawk_requirement(name, lookup)}"
            )
    return lines


def main(argv=None) -> int:
    """Check the environment this interpreter runs in; return the number of mismatches."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", action="append", default=[], metavar="DIR")
    parser.add_argument("--wheels", metavar="DIR")
    given = parser.parse_args(argv)
    lines = mismatches(given.source, given.wheels)
    for line in lines:
        print(f"check_siblings: {line}", file=sys.stderr)
    return len(lines)


if __name__ == "__main__":
    sys.exit(main())
