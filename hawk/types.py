# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dependency-free type tier.

Small, frozen dataclasses describing HAWK's tensor-typed vocabulary —
the records a policy core (a downstream routing/dispatch layer) needs
to describe its own kernel shapes without paying for HAWK's
kernel-authoring machinery. This module is importable at near-zero
cost: importing ``hawk.types`` pulls in no other ``hawk`` module, so a
downstream consumer avoids paying for
``hawk.ir``/``hawk.emit``/``hawk.compile``.

Nothing here performs validation, tracing or codegen — types only.
``TensorType``/``Wire`` are the static slice of the vocabulary read off
a traced/emitted IR; ``Slot`` is one entry of ``Walk.leaves``.
:class:`LaneMeta` lives here rather than in the emitter so a policy core
composing lanes pays the cheap import, never the codegen one.
"""

from __future__ import annotations

from dataclasses import dataclass

#: IR value dtypes. The compiled scalar mode (float64/float32/banded) is
#: a separate axis carried by the artifact, never by a node. ``i32``/``i64``
#: are INDEX integers (addresses, counts; stored 64-bit as ``Int``);
#: ``int32``/``uint32`` are the 32-bit integer PLANES (:data:`INT_PLANE_DTYPES`).
DTYPES = ("f64", "f32", "i32", "i64", "bool", "int32", "uint32")

#: The 32-bit integer plane dtypes: stored at their own width, wrapping
#: arithmetic, never promoted to or from a float without an explicit cast,
#: and never differentiated.
INT_PLANE_DTYPES = ("int32", "uint32")

#: The roles whose plane is addressed at the current sample index by
#: construction: every leaf role here is read ``plane[i]``, so these are
#: the only roles whose plane's trailing extent is n_samples.
PER_SAMPLE_ROLES = ("per_sample", "vec_in", "mat_in", "terminated", "mutable", "out")

@dataclass(frozen=True)
class TensorType:
    """A node's type: ``(shape, dtype, tag)``.

    ``shape=`` is rank-0 (scalar), ``shape=(w,)`` rank-1, ``shape=(r,
    c)`` rank-2; both extents are static. ``tag`` is an optional
    semantic refinement that participates in type identity: two leaves
    sharing ``(role, name)`` with disagreeing ``(shape, dtype, tag)`` are
    never the same slot."""

    shape: tuple[int, ...]
    dtype: str
    tag: str | None = None

    def __getitem__(self, width: object) -> TensorType:
        """``Index[int64]``: the 64-bit index plane (``Index`` alone is the 32-bit one).
        ``int32`` spells the default; any other width is refused by name."""
        if self != _INDEX:
            raise TypeError(f"{self!r} is not subscriptable; only Index takes a width, Index[int64]")
        name = getattr(width, "name", None) or getattr(width, "__name__", None) or str(width)
        if name == "int64":
            return TensorType((), "i64")
        if name == "int32":
            return self
        raise TypeError(f"Index[...] takes int32 (the default) or int64, got {width!r}")


_INDEX = TensorType((), "i32")


@dataclass(frozen=True)
class Wire:
    """One wire of a compound :class:`TensorType` declaration:
    ``Wire("q", role="vec_in", shape=(4,), tag="quaternion")``. A
    ``Quantity`` declaration binds an ordered tuple of these to one
    authored name and namespaces every wire as
    ``<kind-slug>__<instance>__<wire>``."""

    name: str
    role: str
    shape: tuple[int, ...] = ()
    dtype: str = "f64"
    tag: str | None = None


@dataclass(frozen=True)
class Slot:
    """One bound leaf: the ``(role, name)`` identity a slot keys on,
    plus its :class:`TensorType`. Two leaves sharing ``(role, name)``
    merge into one slot only if their ``ttype`` also agrees; a
    disagreement is a walk-time refusal."""

    role: str
    name: str
    ttype: TensorType


@dataclass(frozen=True)
class LaneMeta:
    """One lane of a fused group, as a consumer sees it.

    ``slots`` is that lane's own ``Walk.arg_spec`` — the ``(role, name)``
    sequence in HAWK's canonical role order — and ``digest`` its walk's
    content hash, identifying the lane's body across a re-emission.
    Composition (which lanes fuse) is the consumer's decision; this
    record carries none, only what the emitter observed."""

    name: str
    slots: tuple
    digest: str
