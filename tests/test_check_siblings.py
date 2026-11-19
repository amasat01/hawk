# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The sibling check of the gate environment: a sibling installed from anywhere but what the build was given is a mismatch."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import check_siblings  # noqa: E402


def installed(root, name, version, direct_url=None, requires=()):
    """A fake installed distribution ``name`` in ``root``, with the ``direct_url.json`` given."""
    folder = root / f"{name}-{version}.dist-info"
    folder.mkdir(parents=True)
    lines = [f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}"]
    lines += [f"Requires-Dist: {r}" for r in requires]
    (folder / "METADATA").write_text("\n".join(lines) + "\n")
    if direct_url is not None:
        (folder / "direct_url.json").write_text(json.dumps(direct_url))
    return metadata.Distribution.at(folder)


def environment(dists):
    """A lookup over fake distributions by name."""

    def lookup(name):
        for dist in dists:
            if dist.metadata["Name"].lower() == name.lower().replace("_", "-"):
                return dist
        raise metadata.PackageNotFoundError(name)

    return lookup


def source_of(root, name):
    folder = root / name
    folder.mkdir()
    (folder / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    return folder


def test_a_sibling_installed_from_its_source_is_no_mismatch(tmp_path):
    source = source_of(tmp_path, "raptor-eagle")
    dist = installed(
        tmp_path / "env", "raptor-eagle", "0.7.4", {"url": source.as_uri()}
    )
    assert check_siblings.mismatches([source], None, environment([dist])) == []


def test_a_sibling_replaced_by_the_index_names_it_and_the_requirement_it_misses(
    tmp_path,
):
    source = source_of(tmp_path, "raptor-eagle")
    hawk = installed(
        tmp_path / "env", "raptor-hawk", "1", requires=["raptor-eagle>=0.7.5,<0.8"]
    )
    replaced = installed(tmp_path / "env", "raptor-eagle", "0.7.5")
    (line,) = check_siblings.mismatches([source], None, environment([hawk, replaced]))
    assert (
        "raptor-eagle" in line
        and "0.7.5 from the index" in line
        and "raptor-eagle>=0.7.5,<0.8" in line
    )


def test_a_sibling_given_and_not_installed_is_a_mismatch(tmp_path):
    source = source_of(tmp_path, "raptor-eagle")
    (line,) = check_siblings.mismatches([source], None, environment([]))
    assert "installed (nothing)" in line


def test_a_wheel_is_held_only_by_its_version_and_its_hash(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "raptor_osprey-1.2-py3-none-any.whl"
    wheel.write_bytes(b"wheel bytes")
    digest = "sha256=" + hashlib.sha256(b"wheel bytes").hexdigest()
    held = installed(
        tmp_path / "a", "raptor-osprey", "1.2", {"archive_info": {"hash": digest}}
    )
    other = installed(
        tmp_path / "b", "raptor-osprey", "1.2", {"archive_info": {"hash": "sha256=0"}}
    )
    older = installed(
        tmp_path / "c", "raptor-osprey", "1.1", {"archive_info": {"hash": digest}}
    )
    assert check_siblings.mismatches([], wheels, environment([held])) == []
    assert len(check_siblings.mismatches([], wheels, environment([other]))) == 1
    assert len(check_siblings.mismatches([], wheels, environment([older]))) == 1
