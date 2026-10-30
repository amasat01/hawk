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

#: The forms a build may ask of per-lane linear algebra (``linalg_form``): ``auto`` picks by size
#: (one thread per matrix up to :data:`~hawk.ir.linalg.MAX_DIM`, a team above it), ``thread`` keeps
#: every matrix in one lane (refused above that size, by name) and ``team`` runs every
#: factorisation as a team stage whatever its size.
LINALG_FORMS = ("auto", "thread", "team")

_LINALG_FORM: ContextVar[str] = ContextVar("hawk_linalg_form", default="auto")

#: The forms a build may ask of a per-lane value wider than a thread's registers (``wide_form``):
#: ``auto`` places each such value where the card keeps it (:mod:`hawk.ir.wide`), ``thread``
#: keeps every value in the lane's own registers and local memory (the text before the pass),
#: and ``team`` is a warp per lane, which a later release builds and this one refuses by name.
WIDE_FORMS = ("auto", "thread", "team")

_WIDE_FORM: ContextVar[str] = ContextVar("hawk_wide_form", default="auto")

#: The device feature class the placement pass reads its thresholds from (:mod:`hawk.ir.wide_class`); ``""`` is the default arch's.
_WIDE_CLASS: ContextVar[str] = ContextVar("hawk_wide_class", default="")


@contextmanager
def wide_class_scope(name: str) -> Iterator[None]:
    """Place wide values with the thresholds of class ``name`` for the duration of the block."""
    token = _WIDE_CLASS.set(name)
    try:
        yield
    finally:
        _WIDE_CLASS.reset(token)


def wide_class() -> str:
    """The class the current build places by (``""``: the default arch's)."""
    return _WIDE_CLASS.get()


@contextmanager
def using(form: str = "auto", *, linalg_form: str = "auto", wide_form: str = "auto") -> Iterator[None]:
    """Render under ``form`` (one of :data:`FORMS`), ``linalg_form`` (one of
    :data:`LINALG_FORMS`) and ``wide_form`` (one of :data:`WIDE_FORMS`) for the duration of
    the block."""
    if form not in FORMS:
        raise ValueError(f"emit_form must be one of {FORMS}, got {form!r}")
    if linalg_form not in LINALG_FORMS:
        raise ValueError(f"linalg_form must be one of {LINALG_FORMS}, got {linalg_form!r}")
    if wide_form not in WIDE_FORMS:
        raise ValueError(f"wide_form must be one of {WIDE_FORMS}, got {wide_form!r}")
    if wide_form == "team":
        raise ValueError("wide_form='team' (a warp per lane) is not built in this release; "
                         "use 'auto' or 'thread'")
    token, linalg_token, wide_token = _FORM.set(form), _LINALG_FORM.set(linalg_form), _WIDE_FORM.set(wide_form)
    try:
        yield
    finally:
        _WIDE_FORM.reset(wide_token)
        _LINALG_FORM.reset(linalg_token)
        _FORM.reset(token)


def linalg_form() -> str:
    """The ``linalg_form`` the current build asks for."""
    return _LINALG_FORM.get()


def wide_form() -> str:
    """The ``wide_form`` the current build asks for."""
    return _WIDE_FORM.get()


#: Whether the text being rendered keeps wide values in their planes (:mod:`hawk.ir.wide`): the device
#: text does; the host text renders a placement as the value it places, as it was before the pass.
_PLACED: ContextVar[bool] = ContextVar("hawk_placed", default=False)


@contextmanager
def placing(on: bool) -> Iterator[None]:
    """Render placements as planes (``on``) or as their values for the duration of the block."""
    token = _PLACED.set(bool(on))
    try:
        yield
    finally:
        _PLACED.reset(token)


def placed() -> bool:
    """Whether a placement renders as its plane."""
    return _PLACED.get()


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
