# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The import directions (design section 2a), each test as strong as its line, no exemption list, no counted debt.

All four share one ast scanner (``_w25_scan``): ``import`` / ``from`` at module level and inside functions, classes and try
blocks, relative imports resolved, and string literals given to ``importlib.import_module`` / ``__import__`` / ``find_spec``.

1. no module of raptor imports anything but the standard library and raptor;
2. hawk imports no runtime (eagle) at import time;
3. nothing in raptor, a variant, a component, an extension or eagle imports hawk, at any level (one test per piece);
4. a writer (a component, the emission modules of an extension) imports the standard library, raptor and its own package,
   plus the units its manifest names, directly or through the chain of the units they name; its variant comes as an argument (one test per piece).
"""

from __future__ import annotations

import pytest
import _w25_scan as S


def _scan_counts(pred) -> dict[str, int]:
    return {p: sum(1 for e in S.piece_edges(p) if pred(p, e)) for p in S.piece_ids()}


# ---- 2a.1 ------------------------------------------------------------------------------------------------------------

def test_raptor_imports_only_the_standard_library_and_raptor():
    root, pkg = S.pieces()["raptor"]
    bad = [e for e in S.piece_edges("raptor") if e.top != "raptor" and not S.is_stdlib(e.top)]
    assert not bad, S.report("raptor", "2a.1 raptor imports only the standard library and raptor", bad, {"raptor": len(bad)})


# ---- 2a.2 ------------------------------------------------------------------------------------------------------------

def test_hawk_imports_no_runtime_at_import_time():
    root = S.REPO / "hawk"
    assert root.is_dir(), f"expected the installable package at {root}"
    edges = S.scan_tree(root, "hawk", include_type_checking=False)
    bad = [e for e in edges if e.at_import and e.top == "eagle"]
    assert not bad, "2a.2 hawk imports eagle at import time:\n" + "\n".join(f"  {e.where(S.REPO)}: {e.how} {e.target}" for e in bad)


def test_hawk_has_one_lazy_importer_of_the_runtime():
    """``(one lazy importer, inside a function)``: at most one module of hawk names eagle at any level."""
    files = sorted({e.path for e in S.scan_tree(S.REPO / "hawk", "hawk") if e.top == "eagle"})
    assert len(files) <= 1, "2a.2 hawk imports eagle from several modules, one lazy door allowed:\n" + "\n".join(
        f"  {f.relative_to(S.REPO)}" for f in files)


# ---- 2a.3 ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("piece", S.piece_ids())
def test_piece_imports_no_hawk_at_any_level(piece):
    def pred(p, e):
        return e.top == "hawk"

    bad = [e for e in S.piece_edges(piece) if pred(piece, e)]
    assert not bad, S.report(piece, "2a.3 nothing below hawk imports hawk", bad, _scan_counts(pred))


# ---- eagle imports: the standard library, itself, run-time libraries, raptor and raptor wings only ------------------------

_EAGLE_FORBIDDEN = {
    "eagle->backend": lambda t: t.split(".")[0] in ("hawk_c", "hawk_cuda") or t.startswith("osprey_"),
    "eagle->aether-wing": lambda t: t == "aether.wings" or t.startswith("aether.wings."),
    "eagle->aether": lambda t: (t == "aether" or t.startswith("aether.")) and not (t == "aether.wings" or t.startswith("aether.wings.")),
}


@pytest.mark.parametrize("family", list(_EAGLE_FORBIDDEN))
def test_eagle_imports_no_backend_or_aether(family):
    match = _EAGLE_FORBIDDEN[family]
    bad = [e for e in S.piece_edges("eagle") if match(e.target)]
    base = S.pieces()["eagle"][0].parent
    assert not bad, f"{family}: {len(bad)} import(s) NOT HELD:\n" + "\n".join(f"  {e.where(base)}: {e.how} {e.target}" for e in bad)


# ---- 2a.4 ------------------------------------------------------------------------------------------------------------

def _writer_edges(piece):
    files = S.writer_files(piece)
    return [e for e in S.piece_edges(piece) if files is None or e.path in files]


def _writer_ok(piece, e) -> bool:
    _, pkg = S.pieces()[piece]
    named = S.named_components().get(piece, set())
    named_unit = any(e.target == m or e.target.startswith(m + ".") for m in named)
    # a wing of the family package reaches another wing only through a unit its manifest names (the base wing names none: it holds what every backend needs)
    own = S.unit_module(piece.split(":", 1)[1]) if piece.startswith("wing:") else None
    other_wing = own is not None and e.target.startswith(f"{pkg}.wings.") and not (e.target == own or e.target.startswith(own + "."))
    return (S.is_stdlib(e.top) or (e.top in ("raptor", pkg) and not (other_wing and not named_unit)) or named_unit
            or e.target == "<non-literal>")      # a module named by manifest data (the spelling arm)


@pytest.mark.parametrize("piece", [p for p in S.piece_ids() if p.startswith(("component:", "extension:", "wing:"))])
def test_writer_imports_only_the_standard_library_raptor_and_its_named_components(piece):
    bad = [e for e in _writer_edges(piece) if not _writer_ok(piece, e)]
    counts = {p: sum(1 for e in _writer_edges(p) if not _writer_ok(p, e))
              for p in S.piece_ids() if p.startswith(("component:", "extension:", "wing:"))}
    assert not bad, S.report(piece, "2a.4 a writer imports the standard library, raptor and its named components only", bad, counts)


def test_the_host_and_c_rows_name_nothing_of_the_cells_mode():
    """The cells (banded) writer and team forms are the SIMT wing's: the host row (its backend and its wing) and the C row never name them."""
    import re

    names = re.compile(r"cells_writer|CellsWriter|team_cells|writer_cells|aether\.wings\.simt")
    roots = [S.pieces()[p][0] for p in ("wing:aether_cpu", "extension:osprey_cpu", "extension:hawk_c")]
    hits = [f"{f.relative_to(S.REPO)}:{n}" for root in roots for f in sorted(root.rglob("*.py")) if "tests" not in f.parts
            for n, line in enumerate(f.read_text().splitlines(), 1) if names.search(line)]
    assert not hits, hits
