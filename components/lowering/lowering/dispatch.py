# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dispatch strategy of this wing: a dispatch pinned to ``predicated``, ``segmented`` or ``auto`` is a kind of its own, each
refining the core ``dispatch`` and stating one feather; a dispatch that pins nothing (``switch``) is the core kind. Whoever reads
a strategy asks :func:`raptor.ir.nodes.has_feather`; the words of the strategies live here, in :func:`strategy`."""

from __future__ import annotations

from raptor.errors import RaptorError
from raptor.ir.nodes import Dispatch, has_feather

from . import make_node


#: the strategies a dispatch can be pinned to; ``switch`` is the core kind itself
DISPATCH_STRATEGIES = ("predicated", "switch", "segmented", "auto")

_WORDS = (("splits_by_branch", "segmented"), ("counts_branch_mix", "auto"), ("evaluates_all_branches", "predicated"))


def strategy(node) -> str:
    """The word of the strategy ``node`` is pinned to, from its feathers: a plain dispatch is the ``switch`` form."""
    return next((word for feather, word in _WORDS if has_feather(node, feather)), "switch")


def switch_form(node) -> bool:
    """Whether ``node`` renders as a ``switch`` statement: neither every branch evaluated nor the branches split apart."""
    return not (has_feather(node, "evaluates_all_branches") or has_feather(node, "splits_by_branch"))


def dispatch(selector, branches, strategy: str, ttype) -> Dispatch:
    """A dispatch of ``selector`` over ``branches`` (each of ``ttype``) pinned to ``strategy``: the core kind for ``switch``, else the
    wing's kind of that strategy."""
    if strategy not in DISPATCH_STRATEGIES:
        raise RaptorError(f"dispatch: unknown strategy {strategy!r}; declared are {DISPATCH_STRATEGIES}")
    node = Dispatch(selector, branches, ttype)          # the core's checks
    return node if strategy == "switch" else make_node(f"{strategy}_dispatch", ttype, node.operands)
