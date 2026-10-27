# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The emit policy: which spelling an op site takes.

A site is spelled STATIC (an ``aether`` expression, every element unrolled) or
ROLLED (an ``aether::rt`` twin, a loop over the element index kept rolled,
:mod:`aether/expr/Rt.h`). Both return the same bits; the rolled form keeps one
element live where the static one keeps a fold's worth, so a long fold or a
wide result does not spill. The choice depends only on shapes the IR carries:
the site's fold length ``K`` and its result element count ``E``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from math import prod

from ..ir.nodes import Node, Op

#: The forms a build may ask for: ``auto`` picks per site by the threshold below, ``static``
#: spells every site as the unrolled expression (the text before the policy), and ``rolled``
#: spells every site that has a twin as its loop (a fast-compile mode: the compiler does no
#: per-element work, and a fold or result of any size builds).
FORMS = ("auto", "static", "rolled")

_FORM: ContextVar[str] = ContextVar("hawk_emit_form", default="auto")


@contextmanager
def using(form: str) -> Iterator[None]:
    """Render under ``form`` (one of :data:`FORMS`) for the duration of the block."""
    if form not in FORMS:
        raise ValueError(f"emit_form must be one of {FORMS}, got {form!r}")
    token = _FORM.set(form)
    try:
        yield
    finally:
        _FORM.reset(token)


#: A site is static while its fold length and its result size are both at most
#: this many elements, rolled above it.
ROLLED_FOLD_MIN = 16


def rolled(node: Node) -> bool:
    """Whether ``node`` is spelled as its ``aether::rt`` twin: a matrix-vector or
    matrix-matrix product whose fold length or result size passes
    :data:`ROLLED_FOLD_MIN`, or a ``dot``, ``sum`` or ``norm`` of a vector longer than it
    (under ``emit_form="rolled"`` every such site, whatever its size; under ``"static"``, none)."""
    if not isinstance(node, Op):
        return False
    form = _FORM.get()
    if node.kind in ("mv", "mm"):
        fold, size = node.operands[0].ttype.shape[1], prod(node.ttype.shape)
    elif node.kind in ("dot", "sum", "norm") and len(node.operands[0].ttype.shape) == 1:
        fold, size = node.operands[0].ttype.shape[0], 1
    else:
        return False
    if form != "auto":
        return form == "rolled"
    return fold > ROLLED_FOLD_MIN or size > ROLLED_FOLD_MIN
