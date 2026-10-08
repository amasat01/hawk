# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How a :class:`~hawk.ir.nodes.Dispatch` node renders: the rank-0
``predicated``/``switch`` text both backends share, and the renderer's
dispatch methods as a :class:`hawk.emit.aether._Renderer` mixin."""

from __future__ import annotations

from collections.abc import Sequence

from ..ir import HawkError, Node
from ..ir.loop_nodes import Loop, LoopCarry, LoopIndex
from ..ir.nodes import (
    Const,
    Dispatch,
    Leaf,
    Primitive,
    SampleIndex,
)
from ..types import TensorType
from .spelling import _is_literal_zero, carry_type, element_spelling


def _clamp_expr(kind_spelling: str, k: int) -> str:
    """ONE clamp, identical under every policy. Spelled as a ternary
    rather than ``min(max(...))``: the host TU includes no
    ``<algorithm>``, so an unqualified ``min``/``max`` has no overload
    there, while CUDA's device compiler carries ambient builtins — a
    body string shared by both targets must never depend on that gap."""
    return f"({kind_spelling} < 0 ? 0 : ({kind_spelling} > {k - 1} ? {k - 1} : "\
           f"static_cast<std::int32_t>({kind_spelling})))"


def render_dispatch_rank0(kind_spelling: str, branch_spellings: Sequence[str],
                          policy: str, *, elided: frozenset = frozenset(),
                          cpp_type: str = "", name: str = "",
                          zero_spelling: str = "") -> str:
    """The per-node dispatch text for ``predicated``/``switch`` — identical
    C++ on both targets (no ABI shape, no launch geometry), which is why
    :class:`~hawk.emit.backend.Backend` gains one member both built
    backends implement by calling this same function.

    Every input here is already rendered: ``kind_spelling`` names the
    clamped selector local (:func:`_clamp_expr`, materialised once so a
    K-branch chain doesn't recompute an address K times), and ``elided``
    is the set of branch indices whose IR was literal zero — known only
    to the caller, so this function stays a pure string assembler.

    ``predicated`` needs nothing beyond the two required arguments: a
    right-associative ternary chain over every branch. ``switch`` needs
    the three keyword arguments too: ``cpp_type``/``name`` size and name
    the local the ``case`` labels assign into, and ``zero_spelling`` is
    what ``default:`` writes when a branch was elided — its own index
    never appears in the text at all, which is what makes an elided
    branch's absence auditable in the emitted source.

    A rank>=1 ``predicated`` dispatch does NOT reach this function: it
    needs ``aether::select``'s broadcast condition hoisted into a named
    local per nesting level, which needs a SCOPE this bare string
    function has none of. ``_Renderer._dispatch`` builds that case
    directly instead."""
    if policy == "predicated":
        expr = branch_spellings[-1]
        for j in range(len(branch_spellings) - 2, -1, -1):
            expr = f"({kind_spelling} == {j} ? {branch_spellings[j]} : {expr})"
        return expr
    if policy != "switch":
        raise HawkError(
            f"dispatch: policy {policy!r} has no per-node text — 'segmented' "
            "replaces the node with its own branch inside a per-unit walk "
            "before the renderer ever sees it"
        )
    last = len(branch_spellings) - 1
    lines = [f"{cpp_type} {name};", f"switch ({kind_spelling}) {{"]
    for j, text in enumerate(branch_spellings):
        if j in elided:
            continue
        if not elided and j == last:
            continue          # falls to `default` below
        lines.append(f"    case {j}: {name} = {text}; break;")
    if elided:
        lines.append(f"    default: {name} = {zero_spelling};")
    else:
        lines.append(f"    default: {name} = {branch_spellings[last]};")
    lines.append("}")
    return "\n".join(lines)


class _DispatchMixin:
    """The renderer's dispatch methods."""

    def _dispatch(self, node: Dispatch) -> str:
        """Finite dispatch: one clamp (the contract every policy shares),
        then the policy's own control shape.

        ``switch`` is built by :meth:`_dispatch_switch`: each case renders
        its own branch fresh, scoped to that ``case j: { ... }`` block,
        rather than through :func:`render_dispatch_rank0`'s pure-string
        assembly — a plain string function has no scope to hoist a
        branch's own temporaries into, so without this every temporary
        used to land before the switch regardless of which case needed it.

        ``predicated`` is untouched: it's the evaluate-all policy, so
        every branch is always live and there's no per-branch scope to
        gain. At rank 0 it's pure text through :func:`render_dispatch_rank0`.
        A rank>=1 ``predicated`` dispatch is the one shape that function
        can't build — ``aether::select`` needs its condition broadcast
        into a named local, which needs this scope — so it's built here,
        nesting the same ``aether::select`` calls :meth:`_select` renders.

        ``segmented`` never reaches a renderer: by the time a segmented
        unit is built, the dispatch node has already been replaced by its
        own branch."""
        pos = self.position.get(id(node))
        if pos is None:                            # pragma: no cover - a loop body
            pos = f"b{self.temps}"
            self.temps += 1
        kind_expr = self._ref(node.selector)
        k = node.K
        kname = f"hawk_disp{pos}_k"
        self._emit(f"const std::int32_t {kname} = {_clamp_expr(kind_expr, k)};")
        if node.policy == "switch":
            return self._dispatch_switch(node, pos, kname)
        if node.policy != "predicated":
            raise HawkError(
                f"dispatch: policy {node.policy!r} reached the renderer — "
                "'segmented' replaces the node with its own branch inside a "
                "per-unit walk before this point")
        branch_texts = [self._ref(b) for b in node.branches]
        if not node.ttype.shape:
            return render_dispatch_rank0(kname, branch_texts, "predicated")
        expr = branch_texts[-1]
        for j in range(len(branch_texts) - 2, -1, -1):
            cond = self._hoist(f"aether::constant<aether::extents<>>"
                               f"({kname} == {j})")
            expr = f"aether::select({cond}, {branch_texts[j]}, {expr})"
        return expr

    def _dispatch_switch(self, node: Dispatch, pos, kname: str) -> str:
        """``switch`` with per-CASE scoped hoisting: same control shape
        :func:`render_dispatch_rank0` assembled as one string, except
        each case's expression renders inside its own ``{ }`` block
        (:meth:`_emit_case_body`), so a temporary is forgotten the
        instant that block closes. Duplication across cases is free; a
        shared hoist would put it back before the switch."""
        elided = frozenset(j for j, b in enumerate(node.branches)
                           if _is_literal_zero(b))
        last = node.K - 1
        name = f"hawk_disp{pos}"
        self._emit(f"{carry_type(node.ttype)} {name};")
        self._emit(f"switch ({kname}) {{")
        outer, self.indent = self.indent, self.indent + "    "
        for j, branch in enumerate(node.branches):
            if j in elided:
                continue
            if not elided and j == last:
                continue          # falls to `default:` below
            self._emit(f"case {j}: {{")
            self._emit_case_body(branch, name)
            self._emit("break;")
            self._emit("}")
        if elided:
            self._emit(f"default: {name} = {self._typed_zero(node.ttype)};")
        else:
            self._emit("default: {")
            self._emit_case_body(node.branches[last], name)
            self._emit("}")
        self.indent = outer
        self._emit("}")
        return name

    def _emit_case_body(self, branch: Node, name: str) -> None:
        """Render ONE ``case``'s own scope — a fresh :attr:`local`/
        :attr:`items` snapshot (:meth:`_loop`'s save-a-copy/restore idiom):
        a temporary this case mints must not leak to the next one, while
        a name read from an enclosing scope must still resolve. Popped
        the instant the block closes, so another case's equal temporary
        is a fresh duplicate, not a stale reference."""
        inner, self.indent = self.indent, self.indent + "    "
        saved_local, saved_items = self.local, self.items
        self.local, self.items = dict(saved_local), dict(saved_items)
        text = self._render_branch(branch)
        self._emit(f"{name} = {text};")
        self.local, self.items = saved_local, saved_items
        self.indent = inner

    def _branch_order(self, root: Node) -> tuple:
        """Post-order over ``root``'s dependency subgraph, excluding
        whatever :meth:`_ref` can already resolve — an id already bound
        in the current :attr:`local`, or a position already in
        :attr:`_outside`. What's left is this branch's exclusive work.

        A nested `switch` :class:`Dispatch` contributes only its own
        ``selector`` here; its ``branches`` stay unvisited, so the same
        scoping applies recursively the next time :meth:`_dispatch`
        reaches it. A nested `predicated` Dispatch is walked through
        every operand, branches included, the same exemption
        :meth:`_outside_reachable` grants it."""
        seen: set = set()
        order: list = []
        stack: list = [(root, False)]
        while stack:
            n, expanded = stack.pop()
            if id(n) in seen or id(n) in self.local:
                continue
            p = self.position.get(id(n))
            if p is not None and p in self._outside:
                continue
            if not expanded:
                stack.append((n, True))
                if isinstance(n, Dispatch) and n.policy == "switch":
                    children = (n.selector,)
                else:
                    children = n.operands
                for child in reversed(children):
                    stack.append((child, False))
                continue
            seen.add(id(n))
            order.append(n)
        return tuple(order)

    def _should_name(self, node: Node, fanout: int) -> bool:
        """HAWK's materialisation predicate (:meth:`_materialised`),
        parameterised on which fanout count to read — the kernel-wide
        :attr:`fanout` there, a scope-local tally here."""
        if isinstance(node, Leaf) and node.role in ("uniform", "nsamples"):
            return False
        if isinstance(node, (SampleIndex, LoopIndex, LoopCarry)):
            return False
        if isinstance(node, (Loop, Const, Primitive)):
            return False
        return fanout >= 2 or self._leaf_valued(node)

    def _render_branch(self, root: Node) -> str:
        """Render ONE dispatch branch's exclusive dependency subgraph
        fresh, into the case scope :meth:`_emit_case_body` just opened.
        Mirrors :meth:`_render_scope`'s loop-body pass (a scope-local
        fan-out tally; a :class:`Loop` lowered via :meth:`_loop`; a
        :class:`Primitive` inlined to its forward) — the one difference
        is that :meth:`_branch_order` already excluded anything resolvable
        through an enclosing scope, so only this branch's own nodes
        remain."""
        order = self._branch_order(root)
        local_fanout: dict = {}
        for node in order:
            if isinstance(node, Primitive):
                continue
            for child in node.operands:
                local_fanout[id(child)] = local_fanout.get(id(child), 0) + 1
        for node in order:
            if isinstance(node, Loop):
                self._loop(node)
                self.local[id(node)] = ""
                continue
            if isinstance(node, Primitive):
                self.local[id(node)] = self._ref(node.forward)
                continue
            named = self._should_name(node, local_fanout.get(id(node), 0))
            text = self._expr(node, materialised=named)
            if named:
                name = f"tb{self.temps}"
                self.temps += 1
                self._emit(f"const auto {name} = {text};")
                self.local[id(node)] = name
            else:
                self.local[id(node)] = text
        return self._ref(root)

    def _typed_zero(self, ttype: TensorType) -> str:
        """The typed zero the ``switch`` ``default:`` writes when at
        least one branch was elided — the same spelling :meth:`_broadcast`
        builds, inlined rather than hoisted, since it's assigned straight
        into the declared local, which copies it out immediately."""
        elem = element_spelling(ttype.dtype)
        if not ttype.shape:
            return f"static_cast<{elem}>(0)"
        ext = ", ".join(str(e) for e in ttype.shape)
        return f"aether::constant<aether::extents<{ext}>>(static_cast<{elem}>(0))"
