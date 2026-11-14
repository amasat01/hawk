# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The import directions (design section 2a), each test as strong as its line, no exemption list, no counted debt.

All four share one ast scanner (``_w25_scan``): ``import`` / ``from`` at module level and inside functions, classes and try
blocks, relative imports resolved, and string literals given to ``importlib.import_module`` / ``__import__`` / ``find_spec``.

1. no module of raptor imports anything but the standard library and raptor;
2. hawk imports no runtime (eagle) at import time;
3. nothing in raptor, a variant, a wing, an OSPREY or eagle imports hawk, at any level (one test per piece);
4. the emission modules of an OSPREY import the standard library, raptor and its own package, plus the wings its rows name, through the
   chains of their leaves (one test per OSPREY); a wing is held to the chain rule (R2) by the test of its own, by manifests.
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
    named = S.named_wings().get(piece, set())
    named_unit = any(e.target == m or e.target.startswith(m + ".") for m in named)
    return (S.is_stdlib(e.top) or e.top in ("raptor", pkg) or named_unit
            or e.target == "<non-literal>")      # a module named by manifest data (the spelling arm)


@pytest.mark.parametrize("piece", [p for p in S.piece_ids() if p.startswith("osprey:")])
def test_osprey_writer_imports_only_the_standard_library_raptor_and_its_named_wings(piece):
    bad = [e for e in _writer_edges(piece) if not _writer_ok(piece, e)]
    counts = {p: sum(1 for e in _writer_edges(p) if not _writer_ok(p, e)) for p in S.piece_ids() if p.startswith("osprey:")}
    assert not bad, S.report(piece, "2a.4 an OSPREY's writer imports the standard library, raptor and its named wings only", bad, counts)


def _wing_table():
    """``{(family, name): (folder, module, manifest)}`` of every wing found by its manifest, and the problems of the manifests themselves."""
    table, problems = {}, []
    for folder, module in S.wing_homes():
        m = S.wing_manifest(folder)
        if list(m)[:3] != ["family", "name", "parent"]:
            problems.append(f"{module}: its wing.json does not start with family, name, parent")
            continue
        if (m["family"], m["name"]) in table:
            problems.append(f"{module}: the wing {m['family']}.{m['name']} is found twice")
        table[m["family"], m["name"]] = (folder, module, m)
    return table, problems


def _chain(table, key):
    """The wings of ``key`` from itself up to the root, as keys; stops (and says so) at a missing parent or a cycle."""
    out, seen = [], set()
    while key and key not in seen:
        seen.add(key)
        out.append(key)
        parent = table[key][2]["parent"] if key in table else None
        key = (key[0], parent) if parent else None
    return out


def _under(target: str, module: str) -> bool:
    return target == module or target.startswith(module + ".")


def test_a_wing_imports_the_standard_library_its_family_core_its_parent_chain_and_its_raptor_chain():
    """R2: a wing imports the standard library, its family's core (the family package outside ``wings``), raptor's core (the contract every
    wing of every family shares), its own parent chain, and
    (an aether or eagle wing) the raptor wings of the chain of the SAME name; never a sibling, a child, a wing off its chain, hawk."""
    table, problems = _wing_table()
    report, counts = [], {}
    for (family, name), (folder, module, _) in table.items():
        chain = _chain(table, (family, name))
        allowed = [table[k][1] for k in chain if k in table]
        if family != "raptor":
            allowed += [table[k][1] for k in _chain(table, ("raptor", name)) if k in table]
        bad = []
        for e in S.scan_tree(folder, module):
            core = any(e.top == f and not _under(e.target, f"{f}.wings") for f in {family, "raptor"})
            if not (S.is_stdlib(e.top) or core or e.target == "<non-literal>" or any(_under(e.target, m) for m in allowed)):
                bad.append(e)
        counts[f"{family}.{name}"] = len(bad)
        if bad:
            report.append(f"{family}.{name} (chain {' -> '.join(f'{f}.{n}' for f, n in reversed(chain))}): {len(bad)}\n"
                          + "\n".join(f"    {e.where(S.REPO)}: {e.how} {e.target}" for e in bad))
    total = len(problems) + sum(counts.values())
    assert not total, (f"R2 (stdlib, family core, raptor core, parent chain, same-name raptor chain): {total} NOT HELD ({len(problems)} manifest problems; edges per wing: {counts})\n" + "\n".join(problems)
                       + "\n" + "\n".join(report))


def test_the_host_and_c_rows_name_nothing_of_the_cells_mode():
    """The cells (banded) writer and team forms are the SIMT wing's: the host row (its backend and its wing) and the C row never name them."""
    import re

    names = re.compile(r"cells_writer|CellsWriter|team_cells|writer_cells|aether\.wings\.simt")
    roots = [S.pieces()[p][0] for p in ("wing:aether.cpu", "osprey:cpu", "osprey:c")]
    hits = [f"{f.relative_to(S.REPO)}:{n}" for root in roots for f in sorted(root.rglob("*.py")) if "tests" not in f.parts
            for n, line in enumerate(f.read_text().splitlines(), 1) if names.search(line)]
    assert not hits, hits
