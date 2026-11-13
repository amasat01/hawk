# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Throwaway extensions for the registry rows: a dist-info, a package with a manifest, and the modules it names, under one folder."""

from __future__ import annotations

import json

COPY = "import copy\nfrom {base} import BACKEND as _b\nBACKEND = copy.copy(_b)\nBACKEND.id = {id!r}\n{extra}"


def copy_of(base: str, backend_id: str, extra: str = "") -> str:
    """Module text: a backend that is a copy of ``base``'s ``BACKEND`` under ``backend_id``."""
    return COPY.format(base=base, id=backend_id, extra=extra)


def make(root, name: str, manifest: dict | None = None, *, modules: dict | None = None, raw: str | None = None,
         value: str | None = None) -> None:
    """Write the extension ``name`` under ``root``: a dist-info (METADATA and an entry point whose value is ``value``,
    default ``<package>:extension.json``), the package ``name`` with ``-`` read as ``_`` holding the manifest
    ``{name, version, contract}`` plus ``manifest`` (or the text ``raw``), and ``modules`` (module name -> source)."""
    package = name.replace("-", "_")
    dist = root / f"{package}-0.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0\n")
    (dist / "entry_points.txt").write_text(f"[raptor.extensions]\n{name} = {value or package + ':extension.json'}\n")
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("")
    (root / package / "extension.json").write_text(
        raw if raw is not None else json.dumps({"name": name, "version": "0.0", "contract": "1", **(manifest or {})}))
    for module, text in (modules or {}).items():
        (root / f"{module}.py").write_text(text)


def backend(name: str, ref: str, **traits) -> dict:
    """A backends item of a manifest; it reads no variant key unless ``reads`` says otherwise (the manifest must state it)."""
    return {"name": name, "ref": ref, "about": "a test backend", "reads": [], **traits}


def host_reads() -> list:
    """The variant keys the first-party host row states under ``reads``: a planted row that renders as the host does states the same."""
    from hawk import registry

    return list(registry.traits("host")["reads"])
