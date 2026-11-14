# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A new module-level mutable (dict, list, set, deque, ...) in the ``hawk`` package fails this test until it is
written into ``ALLOWED`` with the reason it is safe to share between threads: a read-only table, or a lock that
guards it, or an idempotent memo whose every operation is a single atomic dict or set call. ``__all__`` is exempt.
The scan reads source only (no import)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from contract_kit import is_mutable as _is_mutable, module_level as _module_level, nonfrozen_classes as _nonfrozen_classes  # noqa: E402,F401  (the one scanner)

ROOT = Path(importlib.util.find_spec("hawk").origin).resolve().parent
IR_ROOT = Path(importlib.util.find_spec("raptor.ir").origin).resolve().parent

TABLE = "read-only table, never mutated after import"
MEMO = "memo of a pure function: single atomic dict calls, a lost race recomputes the same value"

#: Module-level mutables that predate this guard, by file: not reviewed one by one. A new one is not added here.
BASELINE = {
    '__init__.py': ('_BACKEND_DOORS', '_RETIRED', '_TOP_LEVEL_DOORS'),
    '_bind_checks.py': ('_FORMAT_KIND', '_NP_DTYPE_LITERAL', '_WIRE_SPEC'),
    '_roots.py': ('_SDIST_PROBES',),
    'artifact/sidecar.py': ('MUTABLE_DTYPES', '_FIXED_WIRE_DTYPE'),
    'artifact/unit_cache.py': ('COUNTERS', '_PUBLISH_LOG', '_UNIT_MEMO', '_UNIT_STATS'),
    'compile/cache.py': ('_ARTIFACT_MEMO', '_CLOSURE_VERDICT_MEMO', '_MEMO', '_STATS'),
    'compile/drivers.py': ('_INFLIGHT',),
    'compile/toolchain.py': ('_IDENTITY_MEMO',),
    'diff/event.py': ('_OPS', '_SIGN'),
    'diff/jump.py': ('_ORDERED',),
    'diff/transform.py': ('_READ_ROLE',),
    'ext/__init__.py': ('SINK_POLICIES', '_UNBUILT_SINKS'),
    'ext/primitive.py': ('_REGISTRY',),
    'ir/access.py': ('_RANK',),
    'ir/compound.py': ('_KIND_OF_ROLE',),
    'passes/contraction.py': ('_OUTPUT_KIND',),
    'lowering/cost.py': ('_QUAT_OPS',),
    'ir/linalg.py': ('FACTOR_TAG', '_FACTOR_BYTES', '_WIDTH'),
    
    'ir/rays.py': ('_COUNT',),
    'runtime.py': ('_KIND_OF_MIRROR', '_MUTABLE_DTYPE', '_WIRE_TO_DTYPE'),
    'trace/astpass.py': ('_BANNED_EXPR', '_STORE_OPS'),
    'trace/decl.py': ('_FORM_ROLE', '_IN_ROLE'),
    'trace/helpers.py': ('_ACTIVE', '_NEEDS_REWRITE', '_REWRITTEN'),
    'trace/kernel.py': ('_LEAF_KIND', '_TRANSFORM_MEMO'),
    'trace/lane_axes.py': ('_trace',),
    'trace/value.py': ('_INT_RANGE',),
}

ALLOWED = {
    **{("osprey_cpu/toolchain.py", n): "predates the guard, moved with its module from hawk/compile/toolchain.py" for n in
       ('_GCC_FAMILY_MEMO', '_HOST_CONTRACTION', '_LIBC_MEMO', '_MACRO_MEMO', '_TARGET_MEMO', '_WITNESS_MEMO')},
    ("osprey_cpu/backend.py", "BACKEND"): "the backend object, a stateless row never written after import",
    ("registry.py", "KNOWN"): "read-only table, never mutated after import",
    ("registry.py", "_STATE"): "the registry scan, rebuilt under _LOCK; hawk.registry.reset() forgets it",
    ("compile/payload.py", "_PAYLOAD_MEMO"): "memo of the sealed payload; hawk.compile.payload.reset_payload_memo() forgets it",
    ("compile/toolchain.py", "_PIP_ONLY_ROOT"): "process-lifetime served directory, made once under _PIP_ONLY_LOCK and removed at exit; no reset function exists, tests put it back with monkeypatch",
    ("options.py", "LEGACY"): "read-only table, never mutated after import",
    ("base/wrapper_text.py", "HOST_ARMS"): "a table read once at emission, never written after import",
    ("simt/wrapper_text.py", "ARMS"): "a table read once at emission, never written after import",
    **{("cpu/host.py", n): "predates the guard, moved with its module from hawk/emit" for n in ("_ELEM_BYTES",)},
    **{("base/spelling.py", n): "predates the guard, moved with its module from hawk/emit" for n in
       ('_CWISE', 'CWISE_UNARY', 'INFIX_OPS', 'MATH_UNARY', 'MATH_BINARY', 'MATH_TERNARY', '_METHOD0', '_METHOD1', '_RANDOM_FN', 'IDENT_PREFIX')},
    ("lowering/exchange.py", "MIRROR_OF"): TABLE,
    **{("base/wrapper.py", n): "predates the guard, moved with its module from hawk/emit" for n in ('SAMPLE_SUM_OP_CODE', 'SCALAR_MODES')},
    ("simt/wrapper_text.py", "_BANDED_UNCERTIFIED"): "read-only table, never mutated after import",
    ("simt/writer_cells.py", "_RAW_TWIN"): "read-only table, never mutated after import",
    **{("base/writer_aether.py", n): "predates the guard, moved with its module from hawk/emit" for n in
       ('_ENTRY', '_INT_CALLS', '_INT_WRAPPING', '_RANDOM_SALT', '_RAY', '_SOLVE', '_TRIANGULAR')},
    **{("base/team_emit.py", n): "predates the guard, moved with its module from emit/team.py" for n in ("_FACTORISE", "_SCATTER", "_SOLVE", "_STRUCT")},
    **{(file, name): "predates the guard, not reviewed" for file, names in BASELINE.items() for name in names},
}


REPO = ROOT.parent


def pieces() -> dict:
    """piece id -> [(path, key)]: hawk, the IR in raptor, EVERY component root and EVERY extension package, found by their folders.
    Component files keep ``<name>/`` keys, the IR keeps ``ir/``, an extension keeps its package name."""
    out = {"hawk": [(p, p.relative_to(ROOT).as_posix()) for p in sorted(ROOT.rglob("*.py"))],
           "raptor.ir": [(p, "ir/" + p.relative_to(IR_ROOT).as_posix()) for p in sorted(IR_ROOT.rglob("*.py"))]}
    roots = [(f"component:{c.name}", c / c.name) for c in sorted((REPO / "components").glob("*")) if (c / c.name / "__init__.py").is_file()]
    roots += [(f"wing:{d.parent.parent.name}_{d.name}", d) for d in sorted((REPO / "staging").glob("*/wings/*")) if (d / "__init__.py").is_file()]
    roots += [(f"extension:{e.name}", e) for d in sorted((REPO / "extensions").glob("*")) for e in sorted(d.iterdir()) if (e / "__init__.py").is_file()]
    roots += [(f"extension:{d.name}", d) for d in sorted((REPO / "staging").glob("osprey_*")) if (d / "__init__.py").is_file()]
    for piece, root in roots:
        out[piece] = [(p, f"{root.name}/" + p.relative_to(root).as_posix()) for p in sorted(root.rglob("*.py")) if "tests" not in p.parts]
    return out


def global_rebound(tree: ast.Module) -> set:
    """The module-level names some function of the module rebinds through ``global``: process-wide state like a mutable literal."""
    named = set()
    for node in _module_level(tree.body):
        if isinstance(node, ast.Assign):
            named |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            named.add(node.target.id)
    return {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names if name in named}


def found_mutables(piece: str | None = None) -> set:
    found = set()
    every = pieces()
    assert len(every["hawk"]) > 20, f"the scan found {len(every['hawk'])} modules under {ROOT}"
    paths = [pr for name, prs in every.items() if piece in (None, name) for pr in prs]
    trees = {path: ast.parse(path.read_text()) for path, _ in paths}
    # a class made in one piece and instantiated in another is state too: its definition is looked up in every piece
    classes = _nonfrozen_classes([ast.parse(p.read_text()) for prs in every.values() for p, _ in prs])
    for path, rel in paths:
        found |= {(rel, name) for name in global_rebound(trees[path])}
        for node in _module_level(trees[path].body):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if _is_mutable(value, classes):
                found |= {(rel, t.id) for t in targets
                          if isinstance(t, ast.Name) and t.id != "__all__"}
    return found


def test_the_pieces_cover_every_component_and_extension():
    assert {"hawk", "raptor.ir", "component:lowering", "component:simt", "extension:hawk_c", "extension:hawk_cuda"} <= set(pieces())


@pytest.mark.parametrize("piece", sorted(pieces()))
def test_every_module_level_mutable_is_allow_listed_with_a_reason(piece):
    new = sorted(found_mutables(piece) - ALLOWED.keys())
    assert not new, (
        f"{piece}: new module-level mutable(s) {new}: make them per-object state, or add each to "
        "ALLOWED in tests/test_module_mutable_state.py with the reason it is thread-safe")


def test_the_allow_list_has_no_stale_entries_and_every_entry_a_reason():
    stale = sorted(ALLOWED.keys() - found_mutables())
    assert not stale, f"ALLOWED names mutables that no longer exist: {stale}"
    assert all(len(reason) > 8 for reason in ALLOWED.values())


def test_the_scan_sees_a_planted_mutable():
    tree = ast.parse("import threading\n_cache = {}\nif True:\n    _more: list = []\nX = (1, 2)\n")
    names = [t.id for n in _module_level(tree.body) if isinstance(n, (ast.Assign, ast.AnnAssign))
             for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
             if _is_mutable(n.value)]
    assert names == ["_cache", "_more"]


def test_the_scan_sees_a_name_a_function_rebinds_through_global():
    tree = ast.parse("_memo = None\n_other = None\ndef set_it():\n    global _memo\n    _memo = 1\ndef local():\n    _other = 2\n")
    assert global_rebound(tree) == {"_memo"}


def test_the_scan_sees_state_held_by_an_instance_a_context_variable_or_a_thread_local():
    tree = ast.parse("import contextvars, threading\nclass Reg:\n    pass\n@dataclass(frozen=True)\nclass Val:\n    pass\n"
                     "KINDS = Reg()\nV = Val()\nA = contextvars.ContextVar('a')\nB = threading.local()\nC = (1, 2)\n")
    classes = _nonfrozen_classes([tree])
    names = [n.targets[0].id for n in _module_level(tree.body) if isinstance(n, ast.Assign) and _is_mutable(n.value, classes)]
    assert names == ["KINDS", "A", "B"]
