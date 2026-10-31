# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Stores at a runtime index into a vector (``set_component_at`` / ``add_at``).

The IR node is functional: the vector with one component replaced. A run of
them, each the only user of the one before, is ONE mutable local written in
place: the loop's own carried local when nothing else reads it, else one copy
made for the run. Each store is then written one of two ways:

* in place, ``v(k) += e``: one indexed access, which keeps the whole vector in
  local memory on a device;
* as selects, ``v(c) = k == c ? v(c) + e : v(c)`` over the components the index
  can reach (:func:`_interval`): registers only, but work proportional to the
  reach. Stores that share one runtime base and differ by static offsets fuse
  into one pass over the reachable components.

:data:`SELECT_COST_RATIO` picks between them for each run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..ir.loop_nodes import Loop, LoopCarry
from ..ir.nodes import At, Const, Node, Op
from .spelling import _through, carry_type, element_spelling

#: The kinds of node this module writes.
STORE_KINDS = ("set_component_at", "add_at")

#: A run is written as selects when (components its selects touch x the card's
#: select weight) is at most this many times its indexed accesses (a
#: read-modify-write counts two). Measured on a Quadro P2000 with keys that differ
#: across the warp: selects win by 1.5-3x up to a ratio of 16, tie near 32 and
#: lose by 3x at 64.
SELECT_COST_RATIO = 32.0

#: What an indexed access costs when its key is proven uniform over the warp
#: (the whole warp reads one address): on the same card float32 selects win up to
#: a ratio of about 2 against 32 for a divergent key.
UNIFORM_ACCESS = 1 / 16

#: The widest vector whose single indexed accumulate keeps the form that reads the
#: slot back through selects and rebuilds the whole vector (the select-chain spelling)
#: when ``Real`` is four bytes. At width 4 with one store per trip ptxas allocates
#: 32 registers for it against 38 for per-component selects (no local memory in
#: either), the lower count keeps a full 2048 threads resident per SM on sm_61 and
#: the kernel runs 5-8 % faster (float, 60 and 2048 trips). With eight-byte ``Real``
#: the per-component form is faster (0.58x), so it stays.
LEAN_WIDTH = 4


def _select_weight(scalar: str) -> float:
    """The card's weight of a select add for ``scalar`` (``float64``/``float32``)
    on the arch of the build being emitted."""
    from ..targets.cards import select_weight

    return select_weight(scalar)


def _weights(dtype: str) -> tuple:
    """The select weight when ``Real`` is eight bytes wide, and when four: the
    scalar type is chosen when the unit is compiled, so a ``Real`` vector may
    have both."""
    if dtype == "f64":
        return _select_weight("float64"), _select_weight("float32")
    w = _select_weight("float32") if dtype == "f32" else 1.0
    return w, w


@dataclass
class _Link:
    """One store: ``node`` writes ``value`` (added to the slot when ``acc``) at
    ``key``, which is ``base + offset`` (``base`` None for a static key) and
    reaches components ``lo..hi``."""

    node: Op
    key: Node
    base: Node | None
    offset: int
    lo: int
    hi: int
    acc: bool
    value: Node
    read: Node | None
    add: Node | None


@dataclass
class _Run:
    """The chain of stores one mutable local takes, bottom to top."""

    links: list
    base: Node          # the vector the first store starts from


def _interval(node: Node) -> tuple:
    """``(lo, hi)`` bounds of an index-valued node, from its clamps and
    constant offsets alone; unknown is ``(-inf, inf)``."""
    node = _through(node)
    if isinstance(node, Const):
        v = float(node.literal)
        return v, v
    if isinstance(node, Op) and node.kind in ("add", "min", "max", "cast"):
        parts = [_interval(o) for o in node.operands]
        if node.kind == "add":
            return parts[0][0] + parts[1][0], parts[0][1] + parts[1][1]
        if node.kind == "min":
            return min(parts[0][0], parts[1][0]), min(parts[0][1], parts[1][1])
        if node.kind == "max":
            return max(parts[0][0], parts[1][0]), max(parts[0][1], parts[1][1])
        lo, hi = parts[0]
        return (math.trunc(lo) if math.isfinite(lo) else lo,
                math.trunc(hi) if math.isfinite(hi) else hi)
    return -math.inf, math.inf


def _same(a: Node, b: Node, seen: dict | None = None) -> bool:
    """Whether two index expressions are the same value: one node, or equal
    kind, literal and operands (a key written twice is traced twice). ``seen``
    keeps a shared subexpression from being compared once per path."""
    if a is b:
        return True
    seen = {} if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen:
        return seen[pair]
    out = (type(a) is type(b) and a.ttype == b.ttype
           and len(a.operands) == len(b.operands))
    if out and isinstance(a, Const):
        out = a.literal == b.literal
    elif out:
        out = (isinstance(a, (Op, At)) and getattr(a, "kind", None) == getattr(b, "kind", None)
               and getattr(a, "literal", None) == getattr(b, "literal", None)
               and all(_same(_through(x), _through(y), seen)
                       for x, y in zip(a.operands, b.operands)))
    seen[pair] = out
    return out


def _split_key(key: Node) -> tuple:
    """``(base, offset)`` of ``key = base + constants``; ``base`` None when
    the key is all constant."""
    key, offset = _through(key), 0
    while isinstance(key, Op) and key.kind == "add":
        a, b = (_through(o) for o in key.operands)
        if isinstance(b, Const):
            key, offset = a, offset + int(b.literal)
        elif isinstance(a, Const):
            key, offset = b, offset + int(a.literal)
        else:
            break
    if isinstance(key, Const):
        return None, offset + int(key.literal)
    return key, offset


class StorePlan:
    """What the stores of a body are, found once per render: the runs, the
    nodes a run swallows, and which node reads which."""

    def __init__(self, r) -> None:
        self.uses: dict = {}
        self.scope: dict = {}
        self.links: dict = {}
        #: id(node) of the intermediate stores and of the reads a run absorbs
        #: (they render nothing), of the keys only a store reads (never named),
        #: and id(top store) -> its run.
        self.skip: set = set()
        self.inline: set = set()
        self.runs: dict = {}
        if not getattr(r.writer, "in_place_stores", False):
            return
        self._walk(r)
        for gone in self._committed(self.links):
            self.links.pop(gone, None)
        for node in self.links.values():
            self._link(node)
        self._chain()

    def skips(self, node: Node) -> bool:
        return id(node) in self.skip

    def unnamed(self, node: Node) -> bool:
        return id(node) in self.skip or id(node) in self.inline

    # -- the use graph ----------------------------------------------------

    def _use(self, child: Node, by) -> None:
        self.uses.setdefault(id(_through(child)), []).append(by)

    def _walk(self, r) -> None:
        seen: set = set()
        stack = [(n, None) for n in r.nodes]
        while stack:
            node, scope = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            self.scope[id(node)] = scope
            if isinstance(node, Op) and node.kind in STORE_KINDS:
                self.links[id(node)] = node
            for child in node.operands:
                self._use(child, node)
            stack.extend((b, scope) for b in getattr(node, "branches", ()))
            if isinstance(node, Loop):
                roots = [*node.nexts, *node.exit_values]
                if node.exit_cond is not None:
                    roots.append(node.exit_cond)
                for root in roots:
                    self._use(root, node)
                for sink in node.body_sinks:
                    for operand in sink.operands:
                        self._use(operand, sink)
                stack.extend((b, id(node)) for b in node.body_nodes())

    @staticmethod
    def _committed(links: dict) -> set:
        """The ``add_at`` chains over a ``vec``/``splat``: a sink commits them
        link by link (``_plan_commits``), so a run does not take them."""
        gone: set = set()
        for node in links.values():
            chain, base = [], node
            while isinstance(base, Op) and base.kind == "add_at":
                chain.append(id(base))
                base = _through(base.operands[0])
            if chain and isinstance(base, Op) and base.kind in ("vec", "splat"):
                gone.update(chain)
        return gone

    # -- one store ----------------------------------------------------------

    def _link(self, node: Op) -> None:
        base, key, value = (_through(o) for o in node.operands)
        acc, read, add = node.kind == "add_at", None, None
        if node.kind == "set_component_at":
            acc, value, read, add = self._accumulate(node, base, key, value)
        b, offset = _split_key(key)
        n = node.ttype.shape[0]
        if b is None:
            lo = hi = offset
        else:
            ilo, ihi = _interval(b)
            lo = 0 if ilo == -math.inf else max(0, int(ilo) + offset)
            hi = n - 1 if ihi == math.inf else min(n - 1, int(ihi) + offset)

        self.links[id(node)] = _Link(node, key, b, offset, lo, hi, acc, value,
                                     read, add)

    def _accumulate(self, node, base, key, value) -> tuple:
        """``(True, x, read, add)`` when ``node`` stores ``base[key] + x`` at
        ``key``, else ``(False, value, None, None)``. ``read`` is the read of
        that slot the ``add`` alone uses."""
        if (isinstance(value, Op) and value.kind == "add"
                and self.uses.get(id(value)) == [node]):
            for read, x in (value.operands, value.operands[::-1]):
                read = _through(read)
                if (isinstance(read, Op) and self.uses.get(id(read)) == [value]
                        and _through(read.operands[0]) is base):
                    same = (read.kind == "component_at"
                            and _same(_through(read.operands[1]), key)
                            or read.kind == "component" and isinstance(key, Const)
                            and tuple(read.literal if isinstance(read.literal, tuple)
                                      else (read.literal,)) == (key.literal,))
                    if same:
                        return True, _through(x), read, value
        return False, value, None, None

    # -- the chains ------------------------------------------------------

    def _owned(self, link: _Link) -> list:
        """The users of ``link``'s base that belong to ``link`` itself."""
        return [link.node] + ([link.read] if link.read is not None else [])

    def _sole(self, vec: Node, link: _Link) -> bool:
        have = sorted(id(u) for u in self.uses.get(id(vec), ()))
        return have == sorted(id(u) for u in self._owned(link))

    def _chain(self) -> None:
        below: dict = {}            # id(store) -> the store that consumes it
        for link in self.links.values():
            base = _through(link.node.operands[0])
            prev = self.links.get(id(base))
            if (prev is not None and self._sole(base, link)
                    and self.scope[id(base)] == self.scope[id(link.node)]):
                below[id(base)] = link
        for link in list(self.links.values()):
            if id(link.node) in below:
                self.skip.add(id(link.node))
            if link.read is not None:
                self.skip.update((id(link.read), id(link.add)))
            if link.base is not None and all(u is link.node or u is link.read
                                             for u in self.uses[id(link.key)]):
                self.inline.add(id(link.key))
        for link in self.links.values():
            if id(link.node) in below:
                continue
            chain = [link]
            while True:
                base = _through(chain[0].node.operands[0])
                prev = self.links.get(id(base))
                if prev is None or below.get(id(base)) is not chain[0]:
                    break
                chain.insert(0, prev)
            self.runs[id(link.node)] = _Run(chain, _through(chain[0].node.operands[0]))


def plan(r) -> StorePlan:
    """The stores of a render's body (empty for a writer that does not run them)."""
    return StorePlan(r)


# -- emission ------------------------------------------------------------


def emit(w, r, top: Op, args) -> str:
    """Emit the run that ends at ``top`` as statements and return the name of
    the local holding the result. A store inside a run renders nothing; one
    outside every run (a sink's own chain) is a copy written at the index."""
    run = r.stores.runs.get(id(top))
    t = top.ttype
    if run is None:
        if r.stores.skips(top):
            return ""
        name = f"hawk_sv{r.hoists}"
        r.hoists += 1
        r._emit(f"{carry_type(t)} {name} = {args[0]};")
        r._emit(f"{name}({args[1]}) {'+=' if top.kind == 'add_at' else '='} "
                f"{w._cast(args[2], t.dtype)};")
        return name
    first = run.links[0]
    token = f"hawk_sv{r.hoists}"
    r.hoists += 1
    carry = (isinstance(run.base, LoopCarry) and id(run.base) in r.mutable_carries
             and (r.stores._sole(run.base, first) or r.local.get(id(run.base)) in r.proxies))
    held = r._ref(run.base) if r.placed else None
    if carry:
        name = r.local[id(run.base)]
    elif held in r.proxies and r.stores._sole(run.base, first):     # a value placed in a plane, only these stores read it
        name = held
    else:
        name = token
        r._emit(f"{carry_type(t)} {name} = {r._ref(run.base)};")
    elem = element_spelling(t.dtype)
    zero = "-0.0" if t.dtype in ("f64", "f32") else "0"
    groups = _groups(run.links)
    dynamic = [g for g in groups if g[0].base is not None]
    accesses = sum(2 if link.acc else 1 for g in dynamic for link in g)
    touched = sum(len(_reach(g)) for g in dynamic)
    uniform = bool(dynamic) and all(r.uniform(link.key) for g in dynamic for link in g)
    budget = SELECT_COST_RATIO * accesses * (UNIFORM_ACCESS if uniform else 1.0)
    pick8, pick4 = (bool(dynamic) and touched * w <= budget for w in _weights(t.dtype))
    lean = (len(dynamic) == 1 and len(dynamic[0]) == 1 and dynamic[0][0].acc
            and t.shape[0] <= LEAN_WIDTH and t.dtype in ("f64", "f32"))
    if name in r.proxies:       # a value in a plane is written in place: one access per store
        pick8 = pick4 = lean = False
    if lean:
        pick4 = "lean"
        if t.dtype == "f32":
            pick8 = pick4
    if pick8 == pick4:
        _write(w, r, top, name, token, groups, pick8, elem, zero)
    else:
        r._emit("if constexpr (sizeof(Real) == 8) {")
        r.indent += "    "
        _write(w, r, top, name, token, groups, pick8, elem, zero)
        r.indent = r.indent[:-4]
        r._emit("} else {")
        r.indent += "    "
        _write(w, r, top, name, token, groups, pick4, elem, zero)
        r.indent = r.indent[:-4]
        r._emit("}")
    return name


def _write(w, r, top: Op, name: str, token: str, groups: list, select,
           elem: str, zero: str) -> None:
    """The statements of the groups of a run: as selects over the reachable
    components when ``select``, else as indexed writes."""
    for j, group in enumerate(groups):
        tag = f"{token}_{j}"
        names = []
        for i, link in enumerate(group):
            v = f"{tag}_v{i}"
            value = w._cast(r._ref(link.value), top.ttype.dtype)
            r._emit(f"const {elem} {v} = {value};")
            names.append(v)
        first = group[0]
        if first.base is None:
            r._emit(f"{_at(r, name, first.lo)} {'+=' if first.acc else '='} {names[0]};")
        elif select == "lean":
            k = f"{tag}_k"
            old = f"{tag}_old"
            r._emit(f"const auto {k} = {r._ref(first.key)};")
            r._emit(f"const {carry_type(top.ttype)} {old} = {name};")
            r._emit(f"const {elem} {tag}_sum = {old}({k}) + {names[0]};")
            n = top.ttype.shape[0]
            parts = ", ".join(f"({k} == {c} ? {tag}_sum : {old}({c}))"
                              for c in range(n))
            r._emit(f"{name} = {carry_type(top.ttype)}{{{parts}}};")
        elif not select:
            for link, v in zip(group, names):
                r._emit(f"{_at(r, name, r._ref(link.key))} {'+=' if link.acc else '='} {v};")
        elif len(group) == 1:
            k = f"{tag}_k"
            r._emit(f"const auto {k} = {r._ref(first.key)};")
            for c in range(first.lo, first.hi + 1):
                new = f"{name}({c}) + {names[0]}" if first.acc else names[0]
                r._emit(f"{name}({c}) = {k} == {c} ? {new} : {name}({c});")
        else:
            k = f"{tag}_k"
            r._emit(f"const auto {k} = {r._ref(first.base)};")
            for c in _reach(group):
                pick = f"static_cast<{elem}>({zero})"
                for link, v in reversed(list(zip(group, names))):
                    if link.lo <= c <= link.hi:
                        pick = f"{k} == {c - link.offset} ? {v} : {pick}"
                r._emit(f"{name}({c}) += {pick};")


def _at(r, name: str, k) -> str:
    """Element ``k`` of the mutable local ``name``: an ``Item``'s own subscript, or the element of a plane's proxy."""
    return f"{name}.at({k})" if name in r.proxies else f"{name}({k})"


def _reach(group: list) -> list:
    """The components some store of the group can write, ascending."""
    return sorted({c for link in group for c in range(link.lo, link.hi + 1)})


def _groups(links: list) -> list:
    """The stores of a run in order, each a list: a run of accumulates that share
    one runtime base and take distinct offsets is one fused group."""
    groups: list = []
    for link in links:
        last = groups[-1] if groups else None
        if (last and link.acc and link.base is not None and last[0].acc
                and _same(last[0].base, link.base)
                and all(link.offset != other.offset for other in last)):
            last.append(link)
        else:
            groups.append([link])
    return groups
