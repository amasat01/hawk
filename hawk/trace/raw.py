# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@raw_device`` — the bounded escape hatch for opaque author text.

The ONLY way raw text enters an emitted body; it REQUIRES an explicit
access annotation since the walk can't see inside the text (an
unannotated block would default to ``sample_local`` and could return a
wrong answer under a stricter partitioning). The refusal fires at TRACE
time and again in the walk's classifier, joining the
most-restrictive-wins fold. The emitter splices the text verbatim."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..ir import HawkError, Op
from ..ir.access import ACCESS_CLASSES
from ..ir.nodes import RAW_KIND
from ..types import TensorType
from .decl import Scalar
from .value import Value, node_of


class RawBlock:
    """One spliceable block of author text plus its declared access."""

    __slots__ = ("text", "access", "reads", "writes", "returns", "name")

    def __init__(self, text: str, access: str, reads: tuple[str, ...],
                 writes: tuple[str, ...], returns: TensorType) -> None:
        self.text = text
        self.access = access
        self.reads = reads
        self.writes = writes
        self.returns = returns
        self.name = "raw"

    def __call__(self, *operands: Any) -> Value | RawBlock:
        """Splice the block over ``operands``, or adopt a stub ``def``'s
        name and return the block itself (the decorator spelling)."""
        if len(operands) == 1 and callable(operands[0]):
            self.name = getattr(operands[0], "__name__", self.name)
            return self
        return Value(Op(RAW_KIND, tuple(node_of(o) for o in operands),
                        self.returns, literal=self))

    def __repr__(self) -> str:
        return f"<RawBlock {self.name} access={self.access!r}>"


def raw_device(text: str, *, access: str | None = None, reads: Sequence[str] = (),
               writes: Sequence[str] = (), returns: TensorType = Scalar) -> RawBlock:
    """Declare a raw device block (the escape hatch for code the
    vocabulary cannot spell): ``text`` spliced verbatim, with ``access=``
    required (the walk can't see inside text), ``reads``/``writes`` naming
    the touched slots, and ``returns`` the value type."""
    if access is None:
        raise HawkError(
            "@raw_device requires an explicit access= annotation: the walk "
            "cannot see inside spliced text, so its access class cannot be inferred "
            f" — declare one of {ACCESS_CLASSES}"
        )
    if access not in ACCESS_CLASSES:
        raise HawkError(
            f"@raw_device: access={access!r} is not an access class; declare one of "
            f"{ACCESS_CLASSES}"
        )
    return RawBlock(text, access, tuple(reads), tuple(writes), returns)
