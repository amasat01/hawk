# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A new module-level mutable in the ``hawk`` package fails here until it is written into ``ALLOWED`` with the reason it is safe
to share between threads: a read-only table, a lock that guards it, or a memo whose every operation is one atomic call.

The scan reads source text only (no import). It finds a dict, list or set literal or comprehension, a call of a mutable
container, ``ContextVar`` or ``threading.local``, an instance of a class defined in ``hawk`` that is not a frozen value, and a
name that some function rebinds through ``global``.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

HAWK = Path(importlib.util.find_spec("hawk").origin).resolve().parent

MUTABLE_CALLS = {
    "dict",
    "list",
    "set",
    "bytearray",
    "defaultdict",
    "OrderedDict",
    "deque",
    "Counter",
    "ChainMap",
    "WeakSet",
    "WeakKeyDictionary",
    "WeakValueDictionary",
    "array",
    "local",
    "ContextVar",
}
#: Bases that make a class a value type or a refusal: its instances are not shared mutable state.
VALUE_BASES = {
    "NamedTuple",
    "Enum",
    "IntEnum",
    "Flag",
    "IntFlag",
    "StrEnum",
    "TypedDict",
    "Protocol",
}

KEPT_AS_FOUND = "held since before this guard existed; not reviewed one by one"
VIEW = (
    "a read-only view over what the served OSPREYs state; it holds nothing of its own"
)
VALUE = "an immutable payload type: a slotless TensorType, never changed"
BASELINE = {
    "__init__.py": ("_RETIRED", "_TOP_LEVEL_DOORS"),
    "_bind_checks.py": ("_FORMAT_KIND", "_NP_DTYPE_LITERAL", "_WIRE_SPEC"),
    "artifact/sidecar.py": ("MUTABLE_DTYPES", "_FIXED_WIRE_DTYPE"),
    "diff/event.py": ("_OPS", "_SIGN"),
    "diff/jump.py": ("_ORDERED",),
    "ext/__init__.py": ("SINK_POLICIES", "_UNBUILT_SINKS"),
    "ext/primitive.py": ("_REGISTRY",),
    "options.py": ("LEGACY",),
    "trace/astpass.py": ("_BANNED_EXPR", "_STORE_OPS"),
    "trace/decl.py": ("_FORM_ROLE",),
    "trace/helpers.py": ("_ACTIVE", "_NEEDS_REWRITE", "_REWRITTEN"),
    "trace/kernel.py": ("_LEAF_KIND", "_TRANSFORM_MEMO"),
    "trace/lane_axes.py": ("_trace",),
    "trace/value.py": ("_INT_RANGE",),
}
ALLOWED = {
    ("graph_builder.py", "_ACTIVE"): "the recorder of the running builder: a context variable, one value per thread or task",
    ("graph_builder.py", "_HG"): "the runtime the rewritten builder code calls: static methods only, no state of its own",
    ("graph_builder.py", "_MEMO"): "the compiled code of each builder function, keyed by its code object; a repeat only recompiles",
    (
        "ext/primitive.py",
        "_RULES",
    ): "the rule templates of declared primitives; a name is claimed under one lock",
    ("targets/__init__.py", "TARGETS"): VIEW,
    ("targets/__init__.py", "DEFAULT_TARGETS"): VIEW,
    **{
        ("trace/decl.py", name): VALUE
        for name in ("Index", "Int32", "Quat", "Scalar", "UInt32", "_INDEX64")
    },
    **{
        (file, name): KEPT_AS_FOUND
        for file, names in BASELINE.items()
        for name in names
    },
}


def at_module_level(body):
    """Yield the statements of ``body`` that run at import, looking inside ``if``, ``try`` and ``with`` blocks."""
    for node in body:
        if isinstance(node, ast.If | ast.Try):
            for part in (
                node.body,
                node.orelse,
                *(h.body for h in getattr(node, "handlers", ())),
            ):
                yield from at_module_level(part)
        elif isinstance(node, ast.With):
            yield from at_module_level(node.body)
        else:
            yield node


def mutable_classes(trees) -> set:
    """Names of the classes defined in ``trees`` whose instances can change: not frozen, not a value or refusal base."""
    names = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            frozen = any("frozen=True" in ast.unparse(d) for d in node.decorator_list)
            bases = {
                b.id
                if isinstance(b, ast.Name)
                else b.attr
                if isinstance(b, ast.Attribute)
                else ""
                for b in node.bases
            }
            refusal = any(b.endswith(("Error", "Exception")) for b in bases)
            if not frozen and not refusal and not bases & VALUE_BASES:
                names.add(node.name)
    return names


def is_mutable(value, classes=frozenset()) -> bool:
    """Whether the expression ``value`` makes a container or an instance of one of ``classes``."""
    if isinstance(
        value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp
    ):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        return name in MUTABLE_CALLS or name in classes
    return False


def assigned_names(tree: ast.Module) -> dict:
    """The module-level names of ``tree`` mapped to the expressions assigned to them (``__all__`` left out)."""
    out = {}
    for node in at_module_level(tree.body):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id != "__all__":
                out[target.id] = value
    return out


def rebound_through_global(tree: ast.Module) -> set:
    """The module-level names some function of the module rebinds through ``global``: process-wide state."""
    named = set(assigned_names(tree))
    return {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Global)
        for name in node.names
        if name in named
    }


def found_in(trees: dict, classes: set) -> set:
    """``{(file, name)}`` of every module-level mutable of the parsed modules ``trees`` (file -> tree)."""
    found = set()
    for file, tree in trees.items():
        found |= {(file, name) for name in rebound_through_global(tree)}
        found |= {
            (file, name)
            for name, value in assigned_names(tree).items()
            if is_mutable(value, classes)
        }
    return found


def hawk_modules() -> dict:
    """The parsed modules of the installed ``hawk`` package, by path relative to the package."""
    paths = sorted(HAWK.rglob("*.py"))
    assert len(paths) > 20, f"the scan found {len(paths)} modules under {HAWK}"
    return {p.relative_to(HAWK).as_posix(): ast.parse(p.read_text()) for p in paths}


def test_every_module_level_mutable_of_hawk_is_allow_listed_with_a_reason():
    """A mutable made at import and shared by every thread is named in ``ALLOWED``; a new one is a failure here."""
    trees = hawk_modules()
    new = sorted(found_in(trees, mutable_classes(trees.values())) - ALLOWED.keys())
    assert not new, (
        f"new module-level mutable(s) {new}: make them per-object state, or add each to ALLOWED with its reason"
    )


def test_the_allow_list_has_no_stale_entry_and_every_entry_states_a_reason():
    """An entry whose mutable no longer exists is removed; every reason is a sentence."""
    trees = hawk_modules()
    stale = sorted(ALLOWED.keys() - found_in(trees, mutable_classes(trees.values())))
    assert not stale, f"ALLOWED names mutables that no longer exist: {stale}"
    assert all(len(reason) > 8 for reason in ALLOWED.values())


def test_the_scan_sees_a_planted_container_inside_an_if_block():
    """A dict at the top, a list inside ``if`` and a tuple: the first two are found, the tuple is not."""
    tree = ast.parse(
        "import threading\n_cache = {}\nif True:\n    _more: list = []\nX = (1, 2)\n"
    )
    found = {name for name, value in assigned_names(tree).items() if is_mutable(value)}
    assert found == {"_cache", "_more"}


def test_the_scan_sees_a_name_a_function_rebinds_through_global():
    """``_memo`` is rebound through ``global``; ``_other`` is only a local of the function that assigns it."""
    tree = ast.parse(
        "_memo = None\n_other = None\ndef set_it():\n    global _memo\n    _memo = 1\ndef local():\n    _other = 2\n"
    )
    assert rebound_through_global(tree) == {"_memo"}


def test_the_scan_sees_state_held_by_an_instance_a_context_variable_or_a_thread_local_and_not_a_frozen_value():
    """A plain class instance, a ``ContextVar`` and ``threading.local`` are found; a frozen dataclass instance and a tuple are not."""
    tree = ast.parse(
        "import contextvars, threading\n"
        "class Reg:\n    pass\n"
        "@dataclass(frozen=True)\nclass Val:\n    pass\n"
        "KINDS = Reg()\nV = Val()\nA = contextvars.ContextVar('a')\nB = threading.local()\nC = (1, 2)\n"
    )
    classes = mutable_classes([tree])
    found = {
        name
        for name, value in assigned_names(tree).items()
        if is_mutable(value, classes)
    }
    assert found == {"KINDS", "A", "B"}
