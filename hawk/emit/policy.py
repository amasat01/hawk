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
#: and ``team`` is a warp per lane (:mod:`hawk.ir.lane`), for the kernels it has a rule for; the others build as ``auto``.
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


#: How a dense contraction is run (``contraction``): ``direct`` is the kernel's own loop; ``gemm`` asks eagle for a
#: matrix-multiply route, for the kernels that have one (opt in: its values are within a few ulp, not the same bits).
CONTRACTION_FORMS = ("direct", "gemm")

_CONTRACTION: ContextVar[str] = ContextVar("hawk_contraction", default="direct")

#: The scalar mode the build compiles for (``"float32"``, ``"float64"``; ``""`` when the build does not say).
_SCALAR: ContextVar[str] = ContextVar("hawk_scalar", default="")

#: ``(threads, blocks_per_sm)`` the device kernel is compiled for (``launch_bounds``), or ``None`` for no bound.
_LAUNCH_BOUNDS: ContextVar[tuple[int, int] | None] = ContextVar("hawk_launch_bounds", default=None)


@contextmanager
def using(form: str = "auto", *, linalg_form: str = "auto", wide_form: str = "auto", contraction: str = "direct",
          launch_bounds: tuple[int, int] | None = None, scalar: str = "") -> Iterator[None]:
    """Render under ``form`` (one of :data:`FORMS`), ``linalg_form`` (one of
    :data:`LINALG_FORMS`), ``wide_form`` (one of :data:`WIDE_FORMS`), ``contraction`` (one of
    :data:`CONTRACTION_FORMS`) and ``launch_bounds`` (``(threads, blocks_per_sm)``: ``threads`` a multiple of
    32, at most 2048 with the blocks) for the duration of the block."""
    if form not in FORMS:
        raise ValueError(f"emit_form must be one of {FORMS}, got {form!r}")
    if linalg_form not in LINALG_FORMS:
        raise ValueError(f"linalg_form must be one of {LINALG_FORMS}, got {linalg_form!r}")
    if wide_form not in WIDE_FORMS:
        raise ValueError(f"wide_form must be one of {WIDE_FORMS}, got {wide_form!r}")
    if contraction not in CONTRACTION_FORMS:
        raise ValueError(f"contraction must be one of {CONTRACTION_FORMS}, got {contraction!r}")
    if launch_bounds is not None:
        threads, blocks = launch_bounds
        if threads < 32 or threads % 32 or blocks < 1 or threads * blocks > 2048:
            raise ValueError(f"launch_bounds=(threads, blocks) needs threads a multiple of 32 and threads*blocks <= 2048, got {launch_bounds!r}")
        launch_bounds = (int(threads), int(blocks))
    token, linalg_token, wide_token = _FORM.set(form), _LINALG_FORM.set(linalg_form), _WIDE_FORM.set(wide_form)
    contraction_token, bounds_token = _CONTRACTION.set(contraction), _LAUNCH_BOUNDS.set(launch_bounds)
    scalar_token = _SCALAR.set(scalar)
    try:
        yield
    finally:
        _SCALAR.reset(scalar_token)
        _LAUNCH_BOUNDS.reset(bounds_token)
        _CONTRACTION.reset(contraction_token)
        _WIDE_FORM.reset(wide_token)
        _LINALG_FORM.reset(linalg_token)
        _FORM.reset(token)


def scalar() -> str:
    """The scalar mode the current build compiles for (``""``: not stated)."""
    return _SCALAR.get()


def linalg_form() -> str:
    """The ``linalg_form`` the current build asks for."""
    return _LINALG_FORM.get()


def wide_form() -> str:
    """The ``wide_form`` the current build asks for."""
    return _WIDE_FORM.get()


def contraction() -> str:
    """The ``contraction`` the current build asks for."""
    return _CONTRACTION.get()


def launch_bounds() -> tuple[int, int] | None:
    """The ``(threads, blocks_per_sm)`` the current build compiles for, or ``None``."""
    return _LAUNCH_BOUNDS.get()


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


#: Whether the text being rendered is the host's, where a float select is a bitwise ``hawk_abi::blend`` (a ``?:`` is a
#: branch the host vectoriser refuses; the device keeps ``?:``). Off unless a host row of a build turns it on.
_BLEND: ContextVar[bool] = ContextVar("hawk_blend", default=False)


@contextmanager
def blending(on: bool) -> Iterator[None]:
    """Spell host float selects as ``hawk_abi::blend`` (``on``) or as ``?:`` for the duration of the block."""
    token = _BLEND.set(bool(on))
    try:
        yield
    finally:
        _BLEND.reset(token)


def blend() -> bool:
    """Whether float selects render as ``hawk_abi::blend``."""
    return _BLEND.get()


_STAGED: ContextVar[int] = ContextVar("hawk_staged_rows", default=1)


@contextmanager
def staging(rows: int) -> Iterator[None]:
    """Render a warp kernel's vector outputs through shared memory, ``rows`` samples to a block, for the duration of the block."""
    token = _STAGED.set(int(rows))
    try:
        yield
    finally:
        _STAGED.reset(token)


def staged_rows() -> int:
    """The samples a block of the warp form being rendered shares one output tile for (1: stores go straight to the planes)."""
    return _STAGED.get()


_WARPS: ContextVar[int] = ContextVar("hawk_block_warps", default=1)


@contextmanager
def block_warps(k: int) -> Iterator[None]:
    """Render a warp-per-lane kernel for a block of ``k`` warps, one lane each, for the duration of the block."""
    token = _WARPS.set(int(k))
    try:
        yield
    finally:
        _WARPS.reset(token)


def warps_per_block() -> int:
    """The warps a block of the warp-per-lane kernel being rendered holds (1: the block is one warp)."""
    return _WARPS.get()


#: The registers a one-column sample-sum warp keeps, and the cost of each further column
#: (ptxas, Fast2Sum combine, width 1: 20 registers for 1 row, 26 for 4, 43 for 8), against the register line below
#: which every arch keeps full residency.
FOLD_REGS_BASE, FOLD_REGS_COLUMN, RESIDENCY_REGS, FOLD_K_MAX = 20, 2, 32, 8

_FOLD_K: ContextVar[int] = ContextVar("hawk_fold_k", default=0)


@contextmanager
def fold_rows(k: int) -> Iterator[None]:
    """Force ``k`` rows per warp in every cell-sum kernel rendered in the block (the card kit's switch; 0 restores the policy)."""
    token = _FOLD_K.set(int(k))
    try:
        yield
    finally:
        _FOLD_K.reset(token)


def fold_shape(width: int, regs: int = RESIDENCY_REGS) -> int:
    """The rows (32 samples each) one warp folds in ONE batched ``sampleSumWarp_`` call, for a cell sink of ``width``
    columns: the largest power of two ``k <= 8`` whose ``k * width`` shuffle columns keep the warp under the residency
    line (``FOLD_REGS_BASE + FOLD_REGS_COLUMN * (k * width - 1) <= regs``), 1 when none fits. Every ``k`` stores the same
    rows with the same bits; only the shuffle latency per row changes."""
    forced = _FOLD_K.get()
    if forced:
        return forced
    k = FOLD_K_MAX
    while k > 1 and FOLD_REGS_BASE + FOLD_REGS_COLUMN * (k * width - 1) > regs:
        k //= 2
    return k


_BLOCK_FOLD: ContextVar[bool] = ContextVar("hawk_block_fold", default=False)


@contextmanager
def block_fold(on: bool) -> Iterator[None]:
    """Let a top-level keyed compensated run commit once per block (``on``: the frame being rendered is the plain index frame, which keeps every thread of a full block alive to the commit)."""
    token = _BLOCK_FOLD.set(bool(on))
    try:
        yield
    finally:
        _BLOCK_FOLD.reset(token)


def block_folding() -> bool:
    """Whether the body being rendered sits in a frame that allows the block-level commit."""
    return _BLOCK_FOLD.get()


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
