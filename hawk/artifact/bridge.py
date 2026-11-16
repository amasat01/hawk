# This module goes with the tracer: it is the only one that reads the tracer's nodes for the library, and nothing else grows around it.
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The node graph of a kernel as the text the library loads.

:func:`dump` takes the sinks of a kernel, walks every node they reach once (operands and node-valued fields alike), and returns
the library's own dump of that graph, the dialect declarations the nodes' kinds need, and the position of each node in the dump
(the handle a request names it by). The graph lives in a context of its own that is made and dropped inside the call."""

from __future__ import annotations

import dataclasses
import json

from raptor.ir.nodes import Node

from .._errors import HawkError

#: The packages whose slotted objects are records a node field may hold.
_OWN = ("raptor", "hawk", "aether")


def _slots(cls) -> list:
    """Every slot a class and its bases declare, once each, base first."""
    out: list = []
    for base in reversed(cls.__mro__):
        own = base.__dict__.get("__slots__", ())
        out.extend([own] if isinstance(own, str) else own)
    return list(dict.fromkeys(out))


def _is_record(value) -> bool:
    kind = type(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return True
    if isinstance(value, tuple) and hasattr(kind, "_fields"):
        return True
    return kind.__module__.split(".")[0] in _OWN and hasattr(kind, "__slots__") and not isinstance(value, Node)


def _fields(value) -> list:
    kind = type(value)
    if dataclasses.is_dataclass(value):
        return [f.name for f in dataclasses.fields(value)]
    return list(kind._fields) if hasattr(kind, "_fields") else _slots(kind)


def _held(value, out: list) -> None:
    """The nodes a value holds directly (no walk through the nodes themselves)."""
    if isinstance(value, Node):
        out.append(value)
    elif type(value) in (tuple, list, frozenset):
        for item in value:
            _held(item, out)
    elif type(value) is dict:
        for key, item in value.items():
            _held(key, out)
            _held(item, out)
    elif value is None or type(value) in (int, float, str, bool):
        return
    elif _is_record(value):
        for name in _fields(value):
            if hasattr(value, name):
                _held(getattr(value, name), out)


def _text(value, position: dict) -> str:
    """A field value as the library's value text; a node is its position in the dump."""
    kind = type(value)
    if value is None:
        return "None"
    if kind is bool:
        return "True" if value else "False"
    if kind is int:
        return str(value)
    if kind is float:
        return repr(value)
    if kind is str:
        return repr(value)
    if isinstance(value, Node):
        return "#%d" % position[id(value)]
    if kind in (tuple, list, frozenset):
        items = [_text(item, position) for item in (sorted(value, key=repr) if kind is frozenset else value)]
        if kind is tuple:
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        return "[" + ", ".join(items) + "]"
    if kind is dict:
        pairs = sorted((_text(k, position), _text(v, position)) for k, v in value.items())
        return "[" + ", ".join("(%s, %s)" % pair for pair in pairs) + "]"
    if _is_record(value):
        fields = sorted((name, getattr(value, name)) for name in _fields(value) if hasattr(value, name))
        return "%s(%s)" % (kind.__name__, ", ".join("%s=%s" % (n, _text(v, position)) for n, v in fields))
    raise HawkError(f"a node field holds a {kind.__module__}.{kind.__qualname__}, which the library cannot read")


def _plain(value):
    """A declaration's value as JSON data; what is not plain data becomes null."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    return [_plain(item) for item in value] if type(value) in (tuple, list) else None


def _manifests(classes: list) -> list:
    """``[(manifest JSON text, key text)]`` for the declared kinds of ``classes``, one per extension, parents first."""
    by_ext: dict = {}
    seen: set = set()
    for cls in classes:
        for base in reversed([b for b in cls.__mro__ if "decl" in b.__dict__]):
            decl = base.__dict__["decl"]
            if decl.name in seen:
                continue
            seen.add(decl.name)
            low, high, families = decl.operands
            row = {
                "name": decl.name,
                "family": decl.family,
                "refines": decl.refines,
                "operands": {"min": low, "max": "many" if high is None else high, "families": _plain(families)},
                "fields": [
                    {"name": name, "type": _plain(kind), **({} if required is True else {"default": _plain(default)})}
                    for name, kind, default, required in decl.fields
                ],
                "result": _plain(decl.result),
                "identity": _plain(decl.identity),
                "feathers": _plain(decl.feathers),
                "spelling": None if not decl.spelling else dict(zip(("dialect", "name", "args"), _plain(decl.spelling))),
                "wire": decl.wire,
                "no_writer": decl.no_writer,
                "diff": decl.diff,
            }
            by_ext.setdefault(decl.ext, (decl.dialect, []))[1].append(row)
    return [
        (json.dumps({"name": ext, "dialect": dialect, "kinds": rows}, separators=(",", ":")), '("%s",)' % ext)
        for ext, (dialect, rows) in by_ext.items()
    ]


def dump(sinks) -> tuple:
    """``(dump text, dialects text, position)`` of the graph ``sinks`` reach: ``position[id(node)]`` is the handle of a node in
    the dump."""
    import _osprey

    nodes: list = []
    position: dict = {}
    classes: list = []
    stack: list = []
    for root in reversed(list(sinks)):
        found: list = []
        _held(root, found)
        stack.extend((node, False) for node in reversed(found))
    while stack:
        node, done = stack.pop()
        if id(node) in position:
            continue
        if done:
            position[id(node)] = len(nodes)
            nodes.append(node)
            if type(node) not in classes:
                classes.append(type(node))
            continue
        stack.append((node, True))
        found = []
        for slot in _slots(type(node)):
            try:
                _held(object.__getattribute__(node, slot), found)
            except AttributeError:
                pass
        stack.extend((kid, False) for kid in reversed(found) if id(kid) not in position)
    manifests = _manifests(classes)
    context = _osprey.Ctx()
    for manifest, key in manifests:
        context.declare_dialect(manifest, key)
    made: list = []
    for node in nodes:
        values = {}
        for slot in _slots(type(node)):
            try:
                values[slot] = object.__getattribute__(node, slot)
            except AttributeError:
                continue
        kind = values["ttype"]
        fields = [f"({name!r}, {_text(item, position)})" for name, item in sorted(values.items()) if name not in ("ttype", "operands")]
        made.append(context.make(type(node).__name__, kind.dtype, list(kind.shape), kind.tag,
                                 [made[position[id(o)]] for o in values["operands"]],
                                 "(" + ", ".join(fields) + ("," if len(fields) == 1 else "") + ")"))
    return context.dump(made), "".join(f"{m}\n{k}\n" for m, k in manifests), position
