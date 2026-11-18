"""Writes the three wheels of the OSPREY libraries and checks that they install and are found.

``python ci/wheels.py build``: after ``ci/dist.sh`` and ``binding/build.sh``, writes into
``dist/wheels/`` ``raptor-osprey`` (the binding module ``_osprey`` with the base OSPREY inside,
tagged for the stable ABI), ``raptor-osprey-cpu`` and ``raptor-osprey-cuda``. The two backend
wheels hold their folder as package data and an entry point of the group ``hawk.ospreys`` whose
value is that folder; they hold no importable module.
``python ci/wheels.py check``: run by the interpreter of an environment holding only these wheels;
imports ``_osprey`` and opens every folder the entry points name.

Only the standard library is used: a wheel is a zip file with a RECORD of hashes.
"""

import base64
import hashlib
import importlib.metadata
import json
import re
import sys
import sysconfig
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ROOT / "dist" / "wheels"
GROUP = "hawk.ospreys"
BACKENDS = ("cpu", "cuda")
# a fixed stamp keeps two builds of the same files byte-identical
STAMP = (1980, 1, 1, 0, 0, 0)


def platform_tag() -> str:
    """Return the platform tag of this machine in wheel spelling."""
    return re.sub(r"[-.]", "_", sysconfig.get_platform())


def manifest_version() -> str:
    """Return the family version the manifests state; they must state one."""
    manifests = [
        json.loads((ROOT / "dist" / f"osprey_{b}" / "osprey.json").read_text()) for b in BACKENDS
    ]
    versions = {m["version"] for m in manifests}
    if len(versions) != 1:
        raise SystemExit(
            f"the manifests state different versions: {sorted(versions)}; run ci/dist.sh"
        )
    return versions.pop()


def write_wheel(
    distribution: str, version: str, tag: str, files: dict, entry_points: str = ""
) -> Path:
    """Write ``dist/wheels/<distribution>-<version>-<tag>.whl`` of ``files``; return its path.

    ``files`` maps the path inside the wheel to the file on disk; ``entry_points`` is the text of
    ``entry_points.txt`` when the distribution declares any.
    """
    stem = f"{distribution.replace('-', '_')}-{version}"
    info = f"{stem}.dist-info"
    members = {
        name: (source.read_bytes(), source.stat().st_mode & 0o777) for name, source in files.items()
    }
    members[f"{info}/METADATA"] = (
        (
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n"
            "Requires-Python: >=3.9\n"
        ).encode(),
        0o644,
    )
    members[f"{info}/WHEEL"] = (
        (
            "Wheel-Version: 1.0\nGenerator: osprey ci/wheels.py\nRoot-Is-Purelib: false\n"
            f"Tag: {tag}\n"
        ).encode(),
        0o644,
    )
    if entry_points:
        members[f"{info}/entry_points.txt"] = (entry_points.encode(), 0o644)
    record = []
    for name, (data, _) in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")
    record.append(f"{info}/RECORD,,")
    members[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode(), 0o644
    WHEELS.mkdir(parents=True, exist_ok=True)
    out = WHEELS / f"{stem}-{tag}.whl"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, (data, mode) in members.items():
            row = zipfile.ZipInfo(name, STAMP)
            row.external_attr = mode << 16
            row.compress_type = zipfile.ZIP_DEFLATED
            wheel.writestr(row, data)
    return out


def folder_files(name: str) -> dict:
    """Return ``{path inside the wheel: file on disk}`` for every file of ``dist/osprey_<name>``."""
    folder = ROOT / "dist" / f"osprey_{name}"
    if not folder.is_dir():
        raise SystemExit(f"{folder} does not exist; run ci/dist.sh first")
    return {
        f"osprey_{name}/{path.relative_to(folder).as_posix()}": path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def build() -> None:
    """Write the binding wheel and one wheel per backend."""
    version = manifest_version()
    binding = ROOT / "binding" / "build" / "_osprey.abi3.so"
    if not binding.is_file():
        raise SystemExit(f"{binding} does not exist; run binding/build.sh first")
    platform = platform_tag()
    built = [
        write_wheel("raptor-osprey", version, f"cp39-abi3-{platform}", {"_osprey.abi3.so": binding})
    ]
    for name in BACKENDS:
        entry = f"[{GROUP}]\n{name} = osprey_{name}\n"
        built.append(
            write_wheel(
                f"raptor-osprey-{name}", version, f"py3-none-{platform}", folder_files(name), entry
            )
        )
    for wheel in built:
        print(f"{wheel.name}  {wheel.stat().st_size} bytes")


def check() -> None:
    """Import ``_osprey`` and open the folder of each backend the entry points name."""
    import _osprey

    suffix = "dylib" if sys.platform == "darwin" else "so"
    points = importlib.metadata.entry_points(group=GROUP)
    found = {}
    for point in points:
        folder = Path(point.dist.locate_file(point.value))
        manifest = json.loads((folder / "osprey.json").read_text())
        library = folder / f"libosprey_{manifest['name']}.{suffix}"
        if not library.is_file():
            raise SystemExit(f"{folder} holds no {library.name}")
        found[point.name] = manifest["name"]
    if found != {b: b for b in BACKENDS}:
        raise SystemExit(f"the entry points of {GROUP} name {found}, expected {list(BACKENDS)}")
    print(f"checks: _osprey imported from {Path(_osprey.__file__).parent}; opened {sorted(found)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        raise SystemExit("usage: python ci/wheels.py build|check")
