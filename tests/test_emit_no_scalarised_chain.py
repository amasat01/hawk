# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""No scalarised per-element chain in an emitted body.

HAWK emits aether expression trees and lets aether fuse them: a sink lowers
to ONE assignment whose right-hand side is ONE expression, never a sequence
of named per-element scalars each feeding the next. A node whose fan-out is
>= 2 IS materialised, leaves included, because an aether ``View`` subscript
is a ``layout_stride`` address computation
(``data_ + c*compStride_ + i*sampleStride_``) and re-subscripting a leaf
read N times pays it N times.

THE DETECTOR, stated so it can fail. Over an emitted body:

  a. SINK SHAPE -- every sink is exactly one ``target[index] = <rhs>;``
     statement, and the number of such statements equals the number of sinks;
  b. NO SCALARISED CHAIN -- no named temporary's right-hand side performs a
     per-ELEMENT read (``.eval<`` with a component index), which is the
     signature of unrolling a rank>=1 value into per-component scalars;
  c. NO CHAIN OF NAMES -- in a body whose every interior node has fan-out 1,
     there are ZERO named temporaries besides the leaf materialisations, so
     there is no "sequence of named scalars each feeding the next" to have;
  d. NO RE-SUBSCRIPT -- no leaf view identifier is subscripted (``ident[``)
     more than once;
  e. POSITIVE half -- a fan-out-2 node IS named once and READ twice.

This test previously failed via both plants in ``_Renderer._materialised``
(``hawk/emit/aether.py``):
  * ``return True`` -- the "materialise everything" scalarising emitter. It
    fired (c) AND (d):
      AssertionError: (c): 'scale' has a fan-out-1 interior node bound to a
      name -- the chain was split into a sequence of named temporaries instead
      of ONE aether expression. Named: ['t2', 't4'], expected []
            const auto psc_x_i = psc_x[i].eval();
            const auto t2 = (uni_a * psc_x_i);
            const auto t4 = (t2 + uni_b);
      AssertionError: (d): 'gather' subscripts leaf 'lut_table' 2 times --
      an aether View subscript is a layout_stride address computation, so a leaf
      read N times is hoisted ONCE
  * ``return self._leaf_valued(node)`` -- the fan-out rule switched off. It
    fired (c) in the other direction and (e):
      AssertionError: (e): 'chain6''s fan-out-2 node was not materialised --
       requires a name at fan-out >= 2 (named: [])
            mut_out[i] = (((((vin_state * static_cast<Real>(uni_k)) +
              (vin_state * static_cast<Real>(uni_k))) * ...
  both plants were then removed.
"""

from __future__ import annotations

import re

from _emitted import cases

from hawk.emit import render_body
from hawk.emit.aether import binding_name

#: A named temporary the renderer binds: ``const auto <name> = <rhs>;``
_BINDING = re.compile(r"^\s*const auto (\w+) = (.*);$")
#: A sink: ``<target>[<index>] = <rhs>;``, or — since sink seam — the
#: call ``hawk_abi::accum_add(<target>, <index>, <rhs>);`` a scattered
#: accumulate commits through (the target's own atomic add, one subscript
#: on the target). Still exactly ONE statement per sink, which is what the
#: row counts; the compensated kind's companion plane keeps its ``+=``.
_SINK = re.compile(r"^\s*(?:\w+\[.*\] (?:\+)?= .*|hawk_abi::accum_add\(\w+, .*\));$")
#: A per-ELEMENT read -- the scalarisation signature.
_ELEMENT_READ = re.compile(r"\.eval<\d+>\(")


def _statements(text: str):
    return [line for line in text.splitlines() if line.strip()]


def test_every_sink_is_exactly_one_assignment():
    for name, sinks, walk in cases():
        body = render_body(sinks, walk).text
        assignments = [s for s in _statements(body) if _SINK.match(s)]
        assert len(assignments) == len(sinks), (
            f"{name!r} emitted {len(assignments)} assignment(s) for "
            f"{len(sinks)} sink(s) -- a sink lowers to ONE assignment:\n{body}"
        )


def test_no_named_temporary_performs_a_per_element_read():
    for name, sinks, walk in cases():
        body = render_body(sinks, walk).text
        bad = [s for s in _statements(body)
               if _BINDING.match(s) and _ELEMENT_READ.search(s)]
        assert not bad, (
            f"{name!r} binds a per-ELEMENT scalar temporary -- the "
            "signature of unrolling a tensor into components:\n"
            + "\n".join(bad)
        )


def _fanout(walk, sinks):
    from hawk.ir.walk import canonical_nodes

    nodes, position = canonical_nodes(sinks)
    counts: dict = {}
    for node in nodes:
        for child in node.operands:
            p = position[id(child)]
            counts[p] = counts.get(p, 0) + 1
    return nodes, position, counts


def test_a_fanout_one_chain_binds_no_interior_name():
    """(c) The six-op chain: only the fan-out-2 node and the leaf are named."""
    from hawk.ir import Leaf

    for name, sinks, walk in cases():
        body = render_body(sinks, walk).text
        nodes, position, counts = _fanout(walk, sinks)
        expected = {f"t{position[id(n)]}" for n in nodes
                    if not isinstance(n, Leaf) and counts.get(position[id(n)], 0) >= 2}
        named = {m.group(1) for m in map(_BINDING.match, _statements(body)) if m}
        interior = {n for n in named if n.startswith("t")}
        assert interior == expected, (
            f"{name!r} has a fan-out-1 interior node bound to a name -- the "
            "chain was split into a sequence of named temporaries instead of ONE "
            f"aether expression. Named: {sorted(interior)}, expected "
            f"{sorted(expected)}\n{body}"
        )


def test_no_leaf_is_subscripted_more_than_once():
    for name, sinks, walk in cases():
        body = render_body(sinks, walk).text
        for role, slot in walk.arg_spec:
            ident = binding_name(role, slot)
            hits = len(re.findall(rf"\b{ident}\[", body))
            assert hits <= 1, (
                f"{name!r} subscripts leaf {ident!r} {hits} times -- an "
                "aether View subscript is a layout_stride address computation, so a "
                "leaf read N times is hoisted ONCE:\n" + body
            )


def test_a_fanout_two_node_is_materialised_and_read_twice():
    """(e) The positive half: rule must actually fire."""
    body = render_body(*_chain6()).text
    named = [m.group(1) for m in map(_BINDING.match, _statements(body)) if m]
    interior = [n for n in named if n.startswith("t")]
    assert len(interior) == 1, (
        "'chain6''s fan-out-2 node was not materialised -- requires a "
        f"name at fan-out >= 2 (named: {named})\n" + body
    )
    assert body.count(interior[0]) >= 3, (
        f"{interior[0]!r} is bound but read fewer than twice -- the "
        "materialisation is not the shared node\n" + body
    )
    leaf = [m.group(1) for m in map(_BINDING.match, _statements(body))
            if m and m.group(1).endswith("_i")]
    assert leaf == ["vin_state_i"], (
        "leaves included: the fan-out-2 LEAF was not materialised "
        f"(bound: {leaf})\n" + body
    )


def _chain6():
    for name, sinks, walk in cases():
        if name == "chain6":
            return sinks, walk
    raise AssertionError("the corpus lost its six-op chain kernel")
