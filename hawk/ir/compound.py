# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Compound quantities: declaration, wire NAMESPACING and wire expansion.

A :class:`Quantity` binds ONE authored name to an ordered tuple of
:class:`~hawk.types.Wire` declarations plus a reconstruction expression,
namespacing every wire (``<kind-slug>__<instance>__<wire>``, or
``<kind-slug>__<wire>`` when not instanced) so two quantities may share a
wire name without colliding. Consumers never write, see or construct a
wire's IR: they call :meth:`Quantity.read`, which mints the leaves and
hands them to ``reconstruct``. Binding is all-or-nothing on the ACCESS:
reaching any wire binds every wire in one span (:mod:`hawk.ir.walk`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..types import Slot, TensorType, Wire
from .nodes import NAMESPACE_SEP, HawkError, Leaf

#: How a wire's ROLE picks the leaf KIND it mints (the leaf-kind table).
_KIND_OF_ROLE = {
    "vec_in": "vocab_read",
    "mat_in": "vocab_read",
    "wide_in": "vocab_read",
    "per_sample": "vocab_read",
    "lookup": "table_read",
    "terminated": "terminated",
    "uniform": "uniform",
    "nsamples": "nsamples",
}


@dataclass(frozen=True)
class QuantitySpan:
    """One reached quantity's slots in a :class:`~hawk.ir.walk.Walk`: the
    namespaced wire names and their ``arg_spec`` indices, both in
    DECLARATION order (the contiguous-run invariant is the walk's to assert)."""

    prefix: str
    wires: tuple[str, ...]
    slots: tuple[int, ...]


class Quantity:
    """A compound quantity declaration (seam).

    ``slug`` is the kind-slug namespace segment; ``name`` the INSTANCE
    segment (``None`` namespaces as ``<slug>__<wire>``). ``reconstruct``
    takes the wire values in order and returns what the author sees."""

    __slots__ = ("slug", "name", "wires", "reconstruct")

    def __init__(self, slug: str, name: str | None = None, *,
                 wires: Sequence[Wire] = (),
                 reconstruct: Callable[..., Any | None] = None) -> None:
        if not slug:
            raise HawkError(
                "Quantity: slug is required — it is the namespace's first "
                "segment"
            )
        wires = tuple(wires)
        if not wires:
            raise HawkError(f"Quantity {slug!r}: declares no wires")
        seen = set()
        for w in wires:
            if not isinstance(w, Wire):
                raise HawkError(f"Quantity {slug!r}: {w!r} is not a hawk.types.Wire")
            if w.role not in _KIND_OF_ROLE:
                raise HawkError(
                    f"Quantity {slug!r}: wire {w.name!r} has role {w.role!r}, which "
                    f"no leaf kind mints; declared read roles are "
                    f"{sorted(_KIND_OF_ROLE)}"
                )
            if w.name in seen:
                raise HawkError(f"Quantity {slug!r}: duplicate wire name {w.name!r}")
            seen.add(w.name)
        self.slug = slug
        self.name = name
        self.wires = wires
        self.reconstruct = reconstruct

    @property
    def prefix(self) -> str:
        """The namespace this quantity owns."""
        if self.name is None:
            return self.slug
        return f"{self.slug}{NAMESPACE_SEP}{self.name}"

    def wire_name(self, wire: Wire) -> str:
        """``<kind-slug>__<instance>__<wire>`` / ``<kind-slug>__<wire>``."""
        return f"{self.prefix}{NAMESPACE_SEP}{wire.name}"

    def expand(self) -> tuple[Slot, ...]:
        """Every wire's slot, in DECLARATION order — the full span binds
        whenever the quantity is reached, unread wires included."""
        return tuple(
            Slot(w.role, self.wire_name(w), TensorType(tuple(w.shape), w.dtype, w.tag))
            for w in self.wires
        )

    def leaves(self) -> tuple[Leaf, ...]:
        """The wire LEAVES, in declaration order, each carrying its owning
        quantity so the walk can recover the span."""
        return tuple(
            Leaf(_KIND_OF_ROLE[w.role], w.role, self.wire_name(w),
                 TensorType(tuple(w.shape), w.dtype, w.tag),
                 quantity=self, wire_index=i)
            for i, w in enumerate(self.wires)
        )

    def read_with(self, wrap: Callable[[Leaf], Any]) -> Any:
        """:meth:`read`, with every wire leaf passed through ``wrap`` —
        the seam a TRACER binds a quantity parameter through."""
        wires = tuple(wrap(leaf) for leaf in self.leaves())
        if self.reconstruct is not None:
            return self.reconstruct(*wires)
        return wires[0] if len(wires) == 1 else wires

    def read(self) -> Any:
        """What a body reads: ``reconstruct(*wire_leaves)``, or the wire
        leaves themselves (a tuple, or the single leaf) with none."""
        leaves = self.leaves()
        if self.reconstruct is not None:
            return self.reconstruct(*leaves)
        return leaves[0] if len(leaves) == 1 else leaves

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Quantity {self.prefix} wires={[w.name for w in self.wires]}>"
