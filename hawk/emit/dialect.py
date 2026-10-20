# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The spelling seam: what a DIALECT supplies to the one body renderer.

:class:`hawk.emit.aether._Renderer` owns the traversal — the walk order,
fan-out naming, loop scopes and carries, the per-case ``switch`` scopes, the
mask guard and the finish epilogue. Every piece of TARGET TEXT it produces
comes from a :class:`Dialect`: the aether C++ spelling
(:mod:`hawk.emit.dialect_aether`, the default, and what both built backends
wrap) or a plain-C spelling (:mod:`hawk.embedded`). The traversal is written
once; a dialect never re-walks the IR.

Every member takes already-rendered operand TEXT and returns text (or, for
the sink members, emits statements through the renderer's ``_emit``). ``r``
is the renderer: a dialect reads ``r.walk``, ``r.kind``, ``r.sinks``,
``r.items``, ``r.hoists`` and calls ``r._emit``/``r._ref`` exactly as the
moved aether code did through ``self``. Nothing here imports the renderer,
so a dialect module never cycles through :mod:`hawk.emit.aether`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

#: Every spelling member a dialect supplies, in the order the renderer
#: reaches them: declarations and loops, dispatch, expressions, sinks.
DIALECT_MEMBERS: tuple[str, ...] = (
    "declare", "carry_type", "tape_decl", "counter_decl", "for_header",
    "tail_break", "sample_index_value", "hoist",
    "selector_decl", "mix_count", "predicated_rank0", "predicated_tensor",
    "switch_default_tail", "typed_zero",
    "leaf", "const", "at", "select", "op",
    "guard", "store", "scatter_store", "accum_scattered", "store_compensated",
    "sample_sum", "tape_write", "finish", "exit_if",
)


class Dialect(Protocol):
    """One target's spelling of every text the renderer produces.

    The docstring of each member states the aether dialect's text, which is
    the renderer's output today; a second dialect answers the same question
    in its own language."""

    #: The dialect's id (``"aether"``, ``"c"``).
    id: str
    #: Whether EVERY rank>=1 value is named (a C struct value), beyond the
    #: renderer's own fan-out rule. ``False`` for aether, whose lazy
    #: expressions inline until the fan-out rule names them.
    names_every_tensor: bool

    # -- declarations and loops ----------------------------------------
    def declare(self, name: str, ttype: Any, text: str) -> str:
        """The named local: ``const auto {name} = {text};``."""

    def carry_type(self, ttype: Any) -> str:
        """A loop-carried local's declared type (``Real``, ``aether::Item<...>``)."""

    def tape_decl(self, elem: str, name: str, trip: int) -> str:
        """A forward loop's tape array: ``{elem} {name}[{trip}];``."""

    def counter_decl(self, counter: str) -> str:
        """A loop's per-sample iteration counter: ``Int {counter} = 0;``."""

    def for_header(self, r: Any, index: str, loop: Any, count: str | None) -> str:
        """The lowered loop's ``for`` header (``spelling._for_header``). Takes
        the renderer because a runtime-count header may need a statement of
        its own ahead of the ``for`` (the C dialect's clamped count)."""

    def tail_break(self, flag: str) -> str:
        """The tail-form exit: ``if ({flag}) break;``."""

    def exit_if(self, cond: str) -> str:
        """The line opening a head/middle-form ``break``: ``if ({cond}) {``."""

    def sample_index_value(self) -> str:
        """A ``SampleIndex`` node's value: ``static_cast<Int>(hawk_i)``."""

    def hoist(self, r: Any, text: str, ttype: Any) -> str:
        """Bind ``text`` to a fresh local ahead of the current line; its name."""

    # -- dispatch -------------------------------------------------------
    def selector_decl(self, r: Any, kname: str, kind_expr: str, k: int, *,
                      in_range: bool = False) -> str:
        """The clamped selector local: ``const std::int32_t {kname} = ...;``.
        ``in_range``: the selector is a select between two in-range literals,
        so a dialect may skip the clamp (the aether dialect does)."""

    def mix_count(self, mix: str, k: int, kname: str) -> str:
        """An ``auto`` dispatch's branch-mix count statement."""

    def predicated_rank0(self, kname: str, texts: Sequence[str]) -> str:
        """The rank-0 ``predicated`` chain (``render_dispatch_rank0``)."""

    def predicated_tensor(self, r: Any, kname: str, texts: Sequence[str]) -> str:
        """The rank>=1 ``predicated`` chain (nested ``aether::select``)."""

    def switch_default_tail(self) -> str:
        """Text after a ``switch``'s ``default`` clause (empty for aether)."""

    def typed_zero(self, ttype: Any) -> str:
        """The typed zero an elided ``switch`` branch assigns."""

    # -- expressions ----------------------------------------------------
    def leaf(self, r: Any, leaf: Any, materialised: bool) -> str:
        """A slot read at the body's own sample (``view[i].eval()``)."""

    def const(self, node: Any) -> str:
        """A literal (``static_cast<Real>(1.5)``)."""

    def at(self, r: Any, node: Any) -> str:
        """A plane read at a computed index."""

    def select(self, r: Any, node: Any, c: str, a: str, b: str) -> str:
        """``cond ? a : b`` at the node's rank."""

    def op(self, r: Any, node: Any, args: Sequence[str], ranks: Sequence[int]) -> str:
        """Every ``Op`` kind: raw blocks, rank-0 math, casts, reductions, tensors."""

    # -- sinks and the guard --------------------------------------------
    def guard(self, bound: Sequence[str]) -> str | None:
        """The body-wide mask condition over ``bound`` mask bindings, or ``None``."""

    def store(self, r: Any, sink: Any, target: str, value: str) -> None:
        """An own-column store."""

    def scatter_store(self, r: Any, sink: Any, target: str, index: str,
                      value: str) -> None:
        """A scattered store at a computed index."""

    def accum_scattered(self, r: Any, sink: Any, target: str, index: str,
                        value: str) -> None:
        """A scattered ``plain``/``atomic`` accumulate."""

    def store_compensated(self, r: Any, sink: Any, target: str, comp: str,
                          value: str) -> None:
        """The ``compensated`` kind's commit (own column or scattered)."""

    def sample_sum(self, r: Any, sink: Any, column: int) -> None:
        """A ``SampleSum``'s per-sample term into column ``column`` of the row
        the wrapper reduces."""

    def tape_write(self, r: Any, sink: Any, target: str) -> None:
        """A tape row write under ``if (on)``."""

    def finish(self, r: Any, sink: Any) -> None:
        """The finish epilogue: mark the mask, count the sample."""
