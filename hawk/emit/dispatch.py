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
    SWITCH_RENDERED,
    is_param_cell,
    At,
    Const,
    Dispatch,
    Leaf,
    Primitive,
    SampleIndex,
)
from .spelling import _is_literal_zero, binding_name


def _speculable(node: Node) -> bool:
    """Whether ``node`` may be computed in a case that does not need it, on the host
    or the device: it cannot fault and has no effect. Refused: a :class:`Loop`;
    integer ``floordiv``/``mod`` (SIGFPE on the host); a runtime component store;
    an ``at`` read of a lookup plane (its extent is a runtime quantity, so no
    index can be proven in range) except a parameter-cell read (constant index 0
    of a one-cell plane); a ``component_at`` whose index is not a constant inside
    the vector's static extent."""
    if isinstance(node, Loop):
        return False
    kind = getattr(node, "kind", "")
    if kind in ("floordiv", "mod", "set_component_at", "add_at"):
        return False
    if isinstance(node, At):
        return is_param_cell(node)
    if kind == "component_at":
        vec, idx = node.operands[0], node.operands[1]
        return (isinstance(idx, Const) and isinstance(idx.literal, int)
                and 0 <= idx.literal < vec.ttype.shape[0])
    return True


def _clamp_expr(kind_spelling: str, k: int) -> str:
    """ONE clamp, identical under every policy. Spelled as a ternary
    rather than ``min(max(...))``: the host TU includes no
    ``<algorithm>``, so an unqualified ``min``/``max`` has no overload
    there, while CUDA's device compiler carries ambient builtins — a
    body string shared by both targets must never depend on that gap."""
    return f"({kind_spelling} < 0 ? 0 : ({kind_spelling} > {k - 1} ? {k - 1} : "\
           f"static_cast<std::int32_t>({kind_spelling})))"


def _in_range(selector: Node, k: int) -> bool:
    """Whether ``selector`` is a select between two integer literals that are
    branch indices already, so the clamp has nothing to clamp."""
    if getattr(selector, "kind", None) != "select":
        return False
    return all(isinstance(o, Const) and isinstance(o.literal, int)
               and not isinstance(o.literal, bool) and 0 <= o.literal < k
               for o in selector.operands[1:])


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
    if policy == "auto":
        policy = "switch"
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

        ``switch`` is built by :meth:`_dispatch_switches`: each case renders
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
        if node.policy in SWITCH_RENDERED:
            return self._dispatch_switches((node,))[0]
        if node.policy != "predicated":
            raise HawkError(
                f"dispatch: policy {node.policy!r} reached the renderer — "
                "'segmented' replaces the node with its own branch inside a "
                "per-unit walk before this point")
        kname = f"hawk_disp{self._disp_pos(node)}_k"
        self._selector(kname, node.selector, node.K)
        branch_texts = [self._ref(b) for b in node.branches]
        if not node.ttype.shape:
            return self.dialect.predicated_rank0(kname, branch_texts)
        return self.dialect.predicated_tensor(self, kname, branch_texts)

    def _selector(self, kname: str, selector: Node, k: int) -> None:
        """Emit ``kname``, the selector clamped to ``[0, k)``; no clamp for a
        select between two in-range literals."""
        ref = self._ref(selector)
        self._emit(self.dialect.selector_decl(self, kname, ref, k,
                                              in_range=_in_range(selector, k)))

    def _disp_pos(self, node: Dispatch):
        """The suffix a dispatch's emitted names carry: its canonical position,
        or a fresh one for a loop-body dispatch (which has none)."""
        pos = self.position.get(id(node))
        if pos is None:                            # pragma: no cover - a loop body
            pos = f"b{self.temps}"
            self.temps += 1
        return pos

    def _mix_plane(self, node: Dispatch) -> str | None:
        """The binding an ``auto`` group counts its picks into, else ``None``
        (a declared policy counts nothing: its body stays the same text)."""
        if node.policy != "auto":
            return None
        from ..ir.segment import mix_name
        from ..ir.walk import dispatch_groups

        for group, held in dispatch_groups(self.walk).items():
            if held is node:
                return binding_name("lookup", mix_name(group))
        return None                    # inside a loop body: not a group

    def _dispatch_switches(self, nodes: Sequence[Dispatch]) -> list:
        """ONE ``switch`` for ``nodes`` — switch-rendered dispatches that share a
        selector and ``K`` and do not depend on each other (a lone dispatch is the
        group of one). Each ``case j`` renders every member's branch ``j`` in one
        scope (a temporary the members share is computed once) and assigns each
        member's output; a temporary is forgotten when its block closes.

        Work every non-elided case needs is hoisted above the switch
        (:meth:`_hoist_common`): computed once, never speculatively. A member's
        elided branch writes its typed zero; a member with an elided branch takes
        the zero in ``default:``, one without it takes its last branch there.
        Returns each member's output name."""
        poses = [self._disp_pos(n) for n in nodes]
        k = nodes[0].K
        links = self._inline_links.pop(self.position.get(id(nodes[0])), [])
        hidden = {id(n) for _t, a, b in links for n in (a, b)}
        kname = f"hawk_disp{poses[0]}_k"
        self._selector(kname, nodes[0].selector, k)
        names = [f"hawk_disp{p}" for p in poses]
        for node, name in zip(nodes, names):
            mix = self._mix_plane(node)
            if mix is not None:
                self._emit(self.dialect.mix_count(mix, k, kname))
        elided = [frozenset(j for j, b in enumerate(n.branches)
                            if _is_literal_zero(b)) for n in nodes]
        shown = [i for i, n in enumerate(nodes) if id(n) not in hidden]
        plain = any(not elided[i] for i in shown)
        last = k - 1
        active = {j: [i for i in shown if j not in elided[i]] for j in range(k)}
        # a link (:meth:`_plan_commits`) whose index and value are members of this
        # group is a statement of the case that owns it, not a value of every case
        stmts = {j: [(t, a.branches[j], b.branches[j]) for t, a, b in links
                     if not _is_literal_zero(b.branches[j])] for j in range(k)}
        self._hoist_common([[nodes[i].branches[j] for i in active[j]]
                            + [r for _t, a, b in stmts[j] for r in (a, b)]
                            for j in range(k)])
        for i in shown:
            self._emit(f"{carry_type(nodes[i].ttype)} {names[i]};")
        self._emit(f"switch ({kname}) {{")
        outer, self.indent = self.indent, self.indent + "    "
        for j in range(k):
            if (not active[j] and not stmts[j]) or (j == last and plain):
                continue          # falls to `default:` below
            self._emit(f"case {j}: {{")
            self._emit_case_body(
                [(names[i], nodes[i].branches[j]) for i in active[j]],
                [names[i] + " = " + self._typed_zero(nodes[i].ttype)
                 for i in shown if j in elided[i]], stmts[j])
            self._emit("break;")
            self._emit("}")
        tail = self.dialect.switch_default_tail()
        if plain:
            self._emit("default: {")
            self._emit_case_body(
                [(names[i], nodes[i].branches[last]) for i in active[last]],
                [names[i] + " = " + self._typed_zero(nodes[i].ttype)
                 for i in shown if last in elided[i]], stmts[last])
            self._emit("}")
        else:
            for n, i in enumerate(shown):
                label = "default: " if n == 0 else ""
                self._emit(f"{label}{names[i]} = {self._typed_zero(nodes[i].ttype)};")
        self.indent = outer
        self._emit("}")
        return [None if id(n) in hidden else name for n, name in zip(nodes, names)]

    def _hoist_common(self, case_roots: list) -> None:
        """Render, above the switch, every node reachable from ALL of its cases
        (``case_roots``: the roots each case renders), or, with three or more
        cases, from all but one. All: exact common-subexpression elimination, the
        value is computed on every path anyway. All but one: the extra case computes
        a side-effect-free value it does not use, which is cheaper than re-emitting
        it in every other case. A case with nothing to render (every member elided)
        is a path that computes nothing, so hoisting for it would be speculative:
        no hoist."""
        if len(case_roots) < 2 or not all(case_roots):
            return
        orders = [self._branch_order(roots) for roots in case_roots]
        n = len(orders)
        count: dict = {}
        for order in orders:
            for x in order:
                count[id(x)] = count.get(id(x), 0) + 1
        floor = n - 1 if n >= 3 else n
        hoisted: list = []
        seen: set = set()
        for order in orders:
            for x in order:
                if count[id(x)] >= floor and id(x) not in seen:
                    seen.add(id(x))
                    hoisted.append(x)
        # What only some cases read runs in a case that never needed it, so it
        # must be unable to fault or change a result: drop each such node that is
        # not provably safe, and everything computed from it.
        dropped: set = set()
        kept: list = []
        for x in hoisted:
            if count[id(x)] < n and (not _speculable(x) or any(
                    id(o) in dropped for o in x.operands)):
                dropped.add(id(x))
            else:
                kept.append(x)
        hoisted = kept
        if not hoisted:
            return
        hoist_ids = {id(x) for x in hoisted}
        tally: dict = {}
        for x in hoisted:
            self._tally_operands(x, tally)
        for order, roots in zip(orders, case_roots):
            for x in order:
                if id(x) not in hoist_ids:
                    self._tally_operands(x, tally)
            for r in roots:
                if id(r) in hoist_ids:
                    tally[id(r)] = tally.get(id(r), 0) + 1
        self._render_nodes(hoisted, tally)

    def _emit_case_body(self, assigns: list, zeros: list, stmts: Sequence = ()) -> None:
        """Render ONE ``case``'s own scope — a fresh :attr:`local`/
        :attr:`items` snapshot (:meth:`_loop`'s save-a-copy/restore idiom):
        a temporary this case mints must not leak to the next one, while
        a name read from an enclosing scope must still resolve. Popped
        the instant the block closes, so another case's equal temporary
        is a fresh duplicate, not a stale reference. ``assigns`` is
        ``[(output name, branch root)]``, ``zeros`` the zero assignments of
        members elided here, ``stmts`` the ``[(plane, index root, value root)]``
        accumulates this case owns."""
        inner, self.indent = self.indent, self.indent + "    "
        saved_local, saved_items = self.local, self.items
        self.local, self.items = dict(saved_local), dict(saved_items)
        self._render_branch([root for _, root in assigns]
                            + [r for _t, a, b in stmts for r in (a, b)])
        for name, root in assigns:
            self._emit(f"{name} = {self._ref(root)};")
        for target, at, value in stmts:
            self._emit(f"{target}[i].at({self._ref(at)}) += {self._ref(value)};")
        for zero in zeros:
            self._emit(f"{zero};")
        self.local, self.items = saved_local, saved_items
        self.indent = inner

    def _branch_order(self, roots: Sequence[Node]) -> tuple:
        """Post-order over ``roots``' dependency subgraph, excluding
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
        stack: list = [(r, False) for r in reversed(roots)]
        while stack:
            n, expanded = stack.pop()
            if id(n) in seen or id(n) in self.local:
                continue
            p = self.position.get(id(n))
            if p is not None and p in self._outside:
                continue
            if not expanded:
                stack.append((n, True))
                if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
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
        if self.dialect.names_every_tensor and node.ttype.shape:
            return True
        return fanout >= 2 or self._leaf_valued(node)

    @staticmethod
    def _tally_operands(node: Node, tally: dict) -> None:
        if isinstance(node, Primitive):
            return
        for child in node.operands:
            tally[id(child)] = tally.get(id(child), 0) + 1

    def _render_branch(self, roots: Sequence[Node]) -> None:
        """Render ``roots``' exclusive dependency subgraph fresh, into the case
        scope :meth:`_emit_case_body` just opened (a scope-local fan-out tally, like
        :meth:`_render_scope`'s loop-body pass; :meth:`_branch_order` already
        excluded anything resolvable through an enclosing scope)."""
        order = self._branch_order(roots)
        tally: dict = {}
        for node in order:
            self._tally_operands(node, tally)
        for root in roots:          # each root is read once by its assignment
            tally[id(root)] = tally.get(id(root), 0) + 1
        self._render_nodes(order, tally)

    def _render_nodes(self, order: Sequence[Node], tally: dict) -> None:
        """Render ``order`` into the open scope, naming a node the ``tally`` says
        is read twice; a :class:`Loop` lowers via :meth:`_loop`, a
        :class:`Primitive` inlines to its forward."""
        for node in order:
            if isinstance(node, Loop):
                self._loop(node)
                self.local[id(node)] = ""
                continue
            if isinstance(node, Primitive):
                self.local[id(node)] = self._ref(node.forward)
                continue
            named = self._should_name(node, tally.get(id(node), 0))
            text = self._expr(node, materialised=named)
            if named:
                name = f"tb{self.temps}"
                self.temps += 1
                self._emit(self.dialect.declare(name, node.ttype, text))
                self.local[id(node)] = name
            else:
                self.local[id(node)] = text
