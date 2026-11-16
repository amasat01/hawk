# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The block-shaped keyed sum, two kinds of this wing: the sum over the lanes of a key of a product of per-row rank-1
values (``block_sum``) and the store of one to a plane (``block_sum_write``). The front end builds them where it sees the
shape; whether a build keeps a block as one team stage or expands it per entry is a lowering decision
(:func:`raptor.wings.base.team.plain_blocks`). A block is the rows of a key added in lane order: no atomics, the same bits on every
launch. Its operands are the payload, the key and the number of keys (an all-lane sum: key 0 of 1)."""

from __future__ import annotations

from raptor.errors import RaptorError
from raptor.ir.kinds import is_a
from raptor.ir.nodes import CONTRACTS, Const, Op, ReduceSamples
from raptor.ir.types import TensorType

from . import make_node
from .cells import stored

_I32 = TensorType((), "i32")


def BlockSum(value, key=None, count=None, contract: str = "auto", kept: tuple = ()):
    """The sum of ``value`` over the lanes of ``key`` (``count`` keys); no ``key``: one key, every lane."""
    if value.ttype.dtype not in ("f64", "f32") or len(value.ttype.shape) > 2:
        raise RaptorError(f"reduce over samples: an f64/f32 scalar, vector or matrix is reduced, got {value.ttype!r}")
    if contract not in CONTRACTS:
        raise RaptorError(f"reduce over samples: contract {contract!r} is not one of {CONTRACTS}")
    key = Const(0, _I32) if key is None else key
    count = Const(1, _I32) if count is None else count
    return make_node("block_sum", value.ttype, (value, key, count), contract=contract, kept=tuple(kept))


def BlockSumWrite(name, block):
    """The store of ``block`` to the plane ``name`` (``(*payload, *kept)``, payload first)."""
    return make_node("block_sum_write", TensorType((), block.ttype.dtype), (block,), name=name)


def is_block(node) -> bool:
    return is_a(node, "block_sum")


def is_block_write(node) -> bool:
    return is_a(node, "block_sum_write")


def block_of(sink):
    """The block a ``block_sum_write`` stores."""
    return sink.operands[0]


def key_of(block):
    return block.operands[1]


def count_of(block):
    return block.operands[2]


def recount(block, count):
    """``block`` over ``count`` kept coordinates."""
    return block.rebuilt((block.operands[0], key_of(block), count), {})


def factors(payload) -> tuple | None:
    """``(u, v, rows)`` when ``payload`` is a sum over per-lane rows of rank-1 products, whose sum
    over the lanes of a key is ``sum_r u_r v_r^T``: ``outer(u, v)`` (a ``p x q`` block), ``u * v`` of
    a rank-1 ``u`` and a rank-0 ``v`` (a ``p`` vector: the ``q = 1`` block), and, for a per-lane
    ``rows x p`` matrix ``M``, ``transpose(M) @ M`` (``u = v = M``, a ``p x p`` block) and
    ``transpose(M) @ r`` (``v = r``, a ``p`` vector). ``rows`` is how many rows a lane holds (1 for
    the first two spellings); else ``None``."""
    if not isinstance(payload, Op):
        return None
    kind, ops = payload.kind, payload.operands
    rank = lambda x: len(x.ttype.shape)        # noqa: E731
    if kind == "outer":
        return ops[0], ops[1], 1
    if kind == "mul":
        a, b = ops
        if rank(a) == 1 and rank(b) == 0:
            return a, b, 1
        if rank(b) == 1 and rank(a) == 0:
            return b, a, 1
    elif kind in ("mm", "mv"):
        t, other = ops
        if isinstance(t, Op) and t.kind == "transpose":
            m = t.operands[0]
            if rank(m) == 2:
                rows = m.ttype.shape[0]
                if kind == "mm" and other is m:
                    return m, m, rows
                if kind == "mv" and tuple(other.ttype.shape) == (rows,):
                    return m, other, rows
    return None


def factors_of(block) -> tuple:
    """The factors of a block's payload; an ordered sum of a plain scalar or vector is ``(payload, None, 1)``."""
    payload = block.operands[0]
    return factors(payload) or (payload, None, 1)


def ordered_case(payload, contract: str, max_dim: int | None = None) -> bool:
    """Whether the ordered contract keeps ``payload`` (a floating scalar or vector of at most ``max_dim`` elements; ``None``:
    any length) as one block."""
    t = payload.ttype
    return (contract == "ordered" and t.dtype in ("f64", "f32") and len(t.shape) <= 1
            and all(max_dim is None or e <= max_dim for e in t.shape))


def block_shaped(payload, contract: str = "auto", max_dim: int | None = None) -> bool:
    """Whether the keyed sum of ``payload`` CAN be one block, by structure alone: a product of per-row
    rank-1 values of a float type of at most ``max_dim`` elements per extent (``None``: any); under ``contract="ordered"`` any
    floating scalar or vector payload is (rows of a key added in lane order: no atomics, the same bits on every launch)."""
    if ordered_case(payload, contract, max_dim):
        return True
    t = payload.ttype
    return (t.dtype in ("f64", "f32") and all(max_dim is None or e <= max_dim for e in t.shape)
            and factors(payload) is not None)


def _reduction(block, keyed: bool) -> ReduceSamples:
    payload, contract = block.operands[0], block.contract
    if not keyed and not block.kept:
        return ReduceSamples("sum", payload, contract=contract)
    return ReduceSamples("sum", payload, key=key_of(block), count=count_of(block), contract=contract, kept=block.kept)


def plain(block) -> ReduceSamples:
    """The plain sum of ``block``: an all-lane block is the unkeyed sum, a keyed one the keyed sum."""
    return _reduction(block, False)


def expand_stores(sinks) -> tuple:
    """``sinks`` with every ``block_sum_write`` replaced by its per-entry accumulates (the keyed form, an all-lane block at key 0)."""
    out = []
    for sink in sinks:
        if is_block_write(sink):
            out.extend(stored(_reduction(block_of(sink), True), sink.name))
        else:
            out.append(sink)
    return tuple(out)

