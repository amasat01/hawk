"""A throwaway lowered sink class and expression class, written as a dialect author would: each carries its render entry."""
from _specs import HOST
import _kernels as K
from hawk.emit import FLOAT64, render_source
import pytest
from hawk.dialects import declare_kinds, make_node, spells
from hawk_c.writer import CWriter
from hawk.emit import render_body
from hawk.emit.wrapper import scalar_mode
from hawk.ir import canonical
from hawk import HawkError
from hawk.ir.nodes import Assign
from hawk.targets import TARGETS


DEMO = {"name": "demo", "dialect": "demo", "kinds": [
    {"name": "demo.twice", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["same_as", 0],
     "spelling": {"dialect": "demo", "name": "twice", "args": ["ref0"]}},
    {"name": "demo.loud_store", "family": "sink", "refines": "assign", "operands": {"min": 1, "max": 1},
     "result": ["carried"], "spelling": {"dialect": "demo", "name": "loud_store", "args": ["target", "ref0"]}},
    {"name": "demo.spelled", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["same_as", 0],
     "spelling": {"dialect": "demo", "name": "spelled", "args": ["ref0"]}},
    {"name": "demo.aether_only", "family": "expr", "refines": "demo.spelled",
     "spelling": {"dialect": "demo", "name": "aether_only", "args": ["ref0"]}}]}
declare_kinds(DEMO)


def Twice(x):
    """A lowered expression: twice its operand."""
    return make_node("demo.twice", x.ttype, (x,))


def LoudStore(name, value, ttype):
    """A lowered sink: the own-column store, spelled with a marker comment."""
    return make_node("demo.loud_store", ttype, (value,), name=name)


def Spelled(x):
    """A lowered expression whose text comes from the spelling each writer registers for it."""
    return make_node("demo.spelled", x.ttype, (x,))


def AetherOnly(x):
    return make_node("demo.aether_only", x.ttype, (x,))


@spells("aether", "demo", "twice")
def _twice(w, r, x: str) -> str:
    return f"(2 * {x})"


@spells("aether", "demo", "loud_store")
def _loud(w, r, target: str, value: str) -> None:
    r._emit(f"/* demo */ {target}[i] = {value};")


@spells("aether", "demo", "spelled")
def _spelled_aether(w, r, x: str) -> str:
    return f"(AETHER_SPELLED {x})"


@spells("c", "demo", "spelled")
def _spelled_c(w, r, x: str) -> str:
    return f"(C_SPELLED {x})"


@spells("aether", "demo", "aether_only")
def _aether_only(w, r, x: str) -> str:
    return x


def test_demo_dialect_renders_on_host():
    kern = next(iter(K.CORPUS.values()))
    sinks = tuple(LoudStore(s.name, Twice(s.value), s.ttype) if isinstance(s, Assign) else s for s in kern.sinks)
    text = render_source("demo", sinks, canonical(sinks), TARGETS["host"], mode=FLOAT64).text
    assert "/* demo */" in text and "(2 * " in text


def _with(node_class):
    kern = next(iter(K.CORPUS.values()))
    sinks = tuple(Assign(s.name, node_class(s.value), s.ttype) if isinstance(s, Assign) else s for s in kern.sinks)
    return sinks, canonical(sinks)


def test_each_writer_spells_the_dialect_node():
    sinks, walk = _with(Spelled)
    assert "AETHER_SPELLED" in render_source("demo", sinks, walk, TARGETS["host"], mode=FLOAT64).text
    body = render_body(sinks, walk, writer=CWriter(scalar_mode("float64"), "demo", {}, max_dim=32), spec=HOST).text
    assert "C_SPELLED" in body and "AETHER_SPELLED" not in body


def test_a_writer_without_a_spelling_refuses_by_name():
    sinks, walk = _with(AetherOnly)
    with pytest.raises(HawkError, match="the 'c' writer has no spelling for demo.aether_only"):
        render_body(sinks, walk, writer=CWriter(scalar_mode("float64"), "demo", {}, max_dim=32), spec=HOST)
