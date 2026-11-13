# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What the text writer and the runner exchange on: the by-value shape each argument role travels in
(:data:`MIRROR_OF`, :func:`mirror_of`), the entry that folds a companion plane into its target
(:func:`fold_entry_name`) and the refusal for a toolchain or host feature that is absent here (:class:`ToolchainAbsent`).
The writer spells them, the runner reads them back."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raptor.errors import RaptorError

if TYPE_CHECKING:
    from raptor.ir.types import TensorType

MIRROR_OF = {
    "out": "GRefMirror", "vec_in": "GRefMirror", "mat_in": "GRefMirror",
    "per_sample": "ScalarHandle", "lookup": "ScalarHandle",
    "terminated": "ScalarHandle", "wide_in": "ScalarHandle",
    "wide_out": "ScalarHandle", "accum_out": "ScalarHandle",
    "uniform": "value", "nsamples": "value",
}


#: Roles pinned to a 32-byte ``ScalarHandle``; a rank>=1 plane there would
#: need the 40-byte ``GRefMirror``, which the map forbids.
_HANDLE_ONLY = ("per_sample", "lookup", "terminated", "wide_in", "wide_out",
                "accum_out")


def mirror_of(role: str, ttype: TensorType) -> str:
    """The pinned by-value ABI shape: ``GRefMirror``, ``ScalarHandle`` or
    ``value``; ``mutable`` resolves by rank."""
    if role == "mutable":
        return "GRefMirror" if ttype.shape else "ScalarHandle"
    if role in _HANDLE_ONLY and ttype.shape:
        raise RaptorError(
            f"role {role!r} carries a rank-{len(ttype.shape)} plane typed {ttype!r}, "
            "but the map pins it to the 32-byte ScalarHandle (eagle's classify_arg, "
            "eagle/python/eagle/roles.py:292-341) — a wider plane needs the "
            "40-byte GRefMirror, which that role has no packing for. Until eagle's "
            "packer grows a GRef accum/wide role, such a kernel cannot "
            "be emitted; a rank-0 plane in this role can."
        )
    return MIRROR_OF[role]


def fold_entry_name(kernel: str, target: str) -> str:
    """The entry that folds ``target``'s owned companion into it (the host
    spelling appends ``_host``, as every host entry does)."""
    return f"{kernel}__fold_{target}"


class ToolchainAbsent(RaptorError):
    """A toolchain or host feature this build needs is absent here.

    ``toolchain`` names which one (``"cppcheck"``, ``"x86-64"``, a compiler or a device, ...).
    It is a :class:`~raptor.errors.RaptorError`, so a caller that catches that, or reads the message, is unaffected; a gate
    that wants the early exit reads the type and the name.
    """

    toolchain = ""

    def __init__(self, message: str = "", *, toolchain: str = "") -> None:
        super().__init__(message)
        if toolchain:
            self.toolchain = toolchain
