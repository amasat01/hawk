# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether writer of the cells (banded) scalar mode: a Real literal is the three-limb register value ``Band(h, l, t)``, a store into a
plane packs (``Real(value)``), and a named rank-0 Real local is bound as a ``Band``, normalised once where it is named. The device
row holds an instance (``cells_writer``); the base writer's hooks are the points this subclass overrides."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from fractions import Fraction
from math import isfinite

from raptor.errors import RaptorError
from raptor.ir.nodes import At, Const, Leaf, Op, through
from raptor.ir.types import TensorType

from aether.wings.base.spelling import INFIX_OPS, MATH_BINARY, MATH_TERNARY, MATH_UNARY, real_literal
from aether.wings.base.writer_aether import AetherWriter

#: Op kinds whose banded result is a normalised ``Band`` (a facade door or a division).
_DOORS = frozenset(MATH_UNARY) | frozenset(MATH_BINARY) | frozenset(MATH_TERNARY) | {"div"}
#: The certified ops that have a raw-exit twin in ``aether::banded::detail`` (no exit normalise), and the readers
#: that take the twin: an operator of the chain. Any other reader gets the facade entry, which normalises.
_RAW_TWIN = {"sin": "sinRaw", "cos": "cosRaw", "exp": "expRaw"}
_TWIN_READERS = frozenset(("add", "sub", "mul", "neg", "div"))
_RAW_READERS = frozenset(("add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos", "sqrt"))
_LITERAL_KINDS = ("add", "sub", "mul", "div")
_DETAIL = "aether::banded::detail::"


def _real_const(operand) -> Fraction | None:
    """The exact value of a finite rank-0 Real literal operand, else ``None``."""
    if isinstance(operand, Const) and not operand.ttype.shape and operand.ttype.dtype in ("f64", "f32"):
        value = float(operand.literal)
        return Fraction(value) if isfinite(value) else None
    return None


def _float_exact(value: Fraction) -> float | None:
    """The float32 equal to ``value`` exactly, else ``None``."""
    try:
        f = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return None
    return f if Fraction(f) == value else None


def _is_pow2(value: Fraction) -> bool:
    n, d = abs(value).numerator, abs(value).denominator
    return n > 0 and (n & (n - 1)) == 0 and (d & (d - 1)) == 0


def _banded_literal_op(kind: str, args: Sequence[str], node: Op) -> str | None:
    """``x <op> c`` (either side for ``+``/``*``) with a literal ``c``, spelled by its exact class: the identities
    (``x*1``, ``x+0``, ``x-0``, ``x/1``) are ``x``, ``x*-1`` is ``-x``, a power of two scales exactly
    (``scalePow2f``), a float-exact ``c`` takes the one-limb ``mulRawS``/``addRawS``, and ``x / c`` is ``x`` times
    the three-limb reciprocal of ``c`` split from the exact quotient. ``None``: not a literal case, spelled as an
    operator on a three-limb ``Band``."""
    lc, rc = (_real_const(o) for o in node.operands)
    if (lc is None) == (rc is None):
        return None                      # no literal, or two: nothing to classify
    left_lit = lc is not None
    c = lc if left_lit else rc
    x = args[1] if left_lit else args[0]
    if kind in ("sub", "div") and left_lit:
        return None
    if kind in ("add", "sub"):
        if c == 0:
            return x
        f = _float_exact(-c if kind == "sub" else c)
        return f"{_DETAIL}addRawS({x}, {real_literal(f)}f)" if f is not None else None
    if kind == "div":
        if c == 0:
            return None
        c = 1 / c
    if c == 1:
        return x
    if c == -1:
        return f"(-{x})"
    f = _float_exact(c)
    if f is not None:
        helper = "scalePow2f" if _is_pow2(c) else "mulRawS"
        return f"{_DETAIL}{helper}({x}, {real_literal(f)}f)"
    if kind == "div":
        limbs = _split3(c)
        if limbs is not None:
            return f"({x} * aether::banded::Band({', '.join(real_literal(h) + 'f' for h in limbs)}))"
    return None


def _split3(value: Fraction):
    """``value`` as three floats, each the float nearest the rest; ``None`` out of the FP32 range."""
    limbs, rest = [], value
    try:
        for _ in range(3):
            limbs.append(struct.unpack("<f", struct.pack("<f", float(rest)))[0])
            rest -= Fraction(limbs[-1])
    except OverflowError:
        return None
    if abs(limbs[0]) < 2.0 ** -94 or abs(limbs[0]) >= 2.0 ** 126:
        return None
    return limbs


def band_literal(value: float) -> str:
    """``value`` as the three-limb ``Band(h, l, t)``: ``h`` the float nearest it, ``l`` the float nearest the rest,
    ``t`` the rest after that, from the exact value (:class:`~fractions.Fraction`). Exact for every double in the
    banded range; a literal the three limbs cannot hold exactly is refused by name."""
    if not isfinite(value):
        return f"aether::banded::Band({real_literal(value)}, 0.0f, 0.0f)"
    if value == 0:
        return f"aether::banded::Band({real_literal(value)}f, 0.0f, 0.0f)"     # keeps the sign of -0.0
    limbs, rest = [], Fraction(value)
    try:
        for _ in range(3):
            limbs.append(struct.unpack("<f", struct.pack("<f", float(rest)))[0])
            rest -= Fraction(limbs[-1])
    except OverflowError:
        rest = 1
    if rest != 0:
        raise RaptorError(f"the literal {value!r} is outside the banded range: a banded value lives in the FP32 "
                        "exponent envelope [2^-94, 2^126); scale the quantity, or build in float64")
    return "aether::banded::Band(" + ", ".join(real_literal(h) + "f" for h in limbs) + ")"


class CellsWriter(AetherWriter):
    """:class:`~aether.wings.base.writer_aether.AetherWriter` in the cell form of the ``banded`` scalar mode."""

    plain_items = False

    @staticmethod
    def reads_raw(kind: str) -> bool:
        """Does the op ``kind`` take an unnormalised chain operand: arithmetic and the facade doors that normalise
        at entry. The limb-order readers (``abs``, ``min``/``max``, ``copysign``, compares, ``select``) do not."""
        return kind in _RAW_READERS

    def _local_type(self, ttype: TensorType | None) -> str:
        """``auto``, or a rank-0 Real's ``Band``: a later op that reads the limbs in order (``abs``,
        a compare, a division, a root, a transcendental, a store) needs a normalised value, and naming the local
        normalises it once, here (a ``Band`` already normalised copies)."""
        if ttype is not None and not ttype.shape and ttype.dtype in ("f64", "f32"):
            return "aether::banded::Band"
        return "auto"

    def _packed(self, ttype: TensorType | None, value: str) -> str:
        """``value`` as a plane stores it: a rank-0 Real packs into its cell (``Real(value)``)."""
        if ttype is None or (not ttype.shape and ttype.dtype in ("f64", "f32")):
            return f"Real({value})"
        return value

    def sample_sum(self, r, sink, column: int) -> None:
        """The sample's term into its column of the row the wrapper sums, as a ``Band`` term."""
        from aether.wings.base.render import SAMPLE_SUM_TERMS

        term = f"aether::banded::Band({r._ref(sink.value)})"
        r._emit(f"{SAMPLE_SUM_TERMS}[{column}].addTerm({term});" if sink.op == "sum" else f"{SAMPLE_SUM_TERMS}[{column}] = {term};")

    def _real_literal(self, value: float) -> str:
        return band_literal(value)

    def typed_zero(self, ttype: TensorType) -> str:
        if not ttype.shape and ttype.dtype in ("f64", "f32"):
            return band_literal(0.0)
        return super().typed_zero(ttype)

    def _special_op(self, r, kind: str, args: Sequence[str], node: Op) -> str | None:
        """The shared ``sincos``, the raw-exit twins and the literal-operand forms, before the math tables are read."""
        if kind in ("sin", "cos"):
            shared = self._sincos(r, kind, args[0], node)
            if shared is not None:
                return shared
        if kind in _RAW_TWIN and r.readers.get(id(node)) and r.readers[id(node)] <= _TWIN_READERS:
            return f"{_DETAIL}{_RAW_TWIN[kind]}(aether::banded::Band({args[0]}))"
        if kind in INFIX_OPS and len(args) == 2 and kind in _LITERAL_KINDS:
            folded = _banded_literal_op(kind, args, node)
            if folded is not None:
                return folded
            if kind == "add":
                return self._fused_add(r, node)
        return None

    def _fused_add(self, r, node: Op) -> str | None:
        """``a*b + c`` (either order) as the fused raw intake ``fma(a, b, c)`` when ``c`` is a normalised ``Band``
        (a plane element or the result of a door), the product is read by this add alone and neither factor is a
        literal (those take their own raw forms). ``None``: left to the raw chain, which accepts a raw addend."""
        for j in (0, 1):
            prod_node, addend = (through(o) for o in (node.operands[j], node.operands[1 - j]))
            pos = r.position.get(id(prod_node))
            if (not (isinstance(prod_node, Op) and prod_node.kind == "mul" and len(prod_node.operands) == 2)
                    or r.fanout.get(pos, 0) != 1 or r._materialised(prod_node, pos)
                    or any(_real_const(o) is not None for o in prod_node.operands)
                    or not (isinstance(addend, (Leaf, At)) or (isinstance(addend, Op) and addend.kind in _DOORS))
                    or addend.ttype.shape or addend.ttype.dtype not in ("f64", "f32")):
                continue
            a, b = (r._ref(o) for o in prod_node.operands)
            return f"{_DETAIL}fma({a}, {b}, aether::banded::Band({r._ref(addend)}))"
        return None

    def _sincos(self, r, kind: str, arg: str, node: Op) -> str | None:
        """Under ``cells``, the ``sin`` and ``cos`` of ONE operand node, both at the top scope, share one reduction:
        the first reached binds ``sincos`` once, the other reads its result. ``None``: not such a pair."""
        operand = node.operands[0]
        partner = "cos" if kind == "sin" else "sin"
        if r._scope_depth or not any(isinstance(o, Op) and o.kind == partner and o.operands[0] is operand
                                     and r.position.get(id(o)) in r._outside for o in r.nodes):
            return None
        if r.position.get(id(node)) not in r._outside:
            return None
        names = r.sincos.get(id(operand))
        if names is None:
            k = len(r.sincos)
            names = r.sincos[id(operand)] = (f"hawk_sin{k}", f"hawk_cos{k}")
            r._emit(f"aether::banded::Band {names[0]}, {names[1]};")
            r._emit(f"aether::math::sincos(aether::banded::Band({arg}), &{names[0]}, &{names[1]});")
        return names[0] if kind == "sin" else names[1]
