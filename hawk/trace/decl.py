# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The declaration vocabulary a kernel author annotates parameters with.

A parameter's annotation IS its vocabulary declaration: a payload type
declares a per-sample input plane whose ROLE follows its rank; a wrapper
(``Mutable[...]``, ``Table[...]``, ``Accum[...]``, ``Param``,
``Reduce(op)``) declares the output/lookup/accumulate/uniform/mapreduce
forms; a :class:`~hawk.ir.compound.Quantity` declares a compound access,
namespaced and all-or-nothing. Nothing here builds IR; it resolves an
annotation to the ``(form, ttype)`` pair :mod:`hawk.trace.kernel` binds.

NAMED AXES: ``Table["row":rows, "col":cols]`` and
``Staged[Vector[W], count]`` declare a plane's axis LAYOUT as a fact, not a
convention, with each stride minted as a :class:`~hawk.ir.nodes.RoleConst`
naming its axis, rather than a hand-written index that re-derives the
producer's layout uncheckably. Layout is ROW-MAJOR, last axis innermost.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import HawkError, Quantity
from ..ir.nodes import REDUCE_OPS
from ..types import TensorType

#: Payload types. ``Vector``/``Matrix`` are subscripted with static extents —
#: an extent is a TYPE, never a runtime value.
Scalar = TensorType((), "f64")
Index = TensorType((), "i32")
Quat = TensorType((4,), "f64", "quaternion")


class Vector:
    """``Vector[W]`` — a rank-1 payload of static width ``W``."""

    def __class_getitem__(cls, w: int) -> TensorType:
        return TensorType((int(w),), "f64")


class Matrix:
    """``Matrix[R, C]`` — a rank-2 payload of static extents."""

    def __class_getitem__(cls, rc: tuple[int, int]) -> TensorType:
        r, c = rc
        return TensorType((int(r), int(c)), "f64")


@dataclass(frozen=True)
class Plane:
    """A resolved declaration: its ``form`` plus the payload type it
    carries. ``dims`` is the declared axis layout (empty for a positional
    plane); ``staged`` marks the one role whose adjoint FAN-IN is a
    derivable number."""

    form: str
    ttype: TensorType
    op: str | None = None
    dims: tuple = ()
    staged: bool = False

    def __setitem__(self, key: object, value: object) -> None:
        from ..ir import HawkError

        if self.form == "terminated":
            from .value import FinishRefusal

            raise FinishRefusal(
                "a `Terminated` plane the kernel did not declare is finished: "
                "declare the mask as a parameter (`terminated: Terminated`) and "
                "write `terminated = cond` on it")
        raise HawkError(
            f"a `{self.form}` DECLARATION is assigned to; commit through the "
            "kernel's own parameter instead")


class Mutable:
    """``Mutable[...]`` — the output plane a body assigns (the ``assign``)."""

    def __class_getitem__(cls, t: TensorType) -> Plane:
        return Plane("mutable", _payload("Mutable", t))


class Table:
    """``Table[...]`` — a lookup plane read with ``at()``.

    ``Table[Scalar]`` is the POSITIONAL plane, read at a flat index;
    ``Table["row":rows, "col":cols]`` is the NAMED-AXIS plane, read
    ``t.at(row=…, col=…)`` against declaration-minted strides."""

    def __class_getitem__(cls, key: object) -> Plane:
        if isinstance(key, TensorType):
            return Plane("table", _payload("Table", key))
        return Plane("table", Scalar, dims=_axes("Table", key))


class Staged:
    """``Staged[Vector[W], count]`` — a buffer one launch PRODUCES and the
    next CONSUMES, declared as the producer wrote it.

    ``Vector[W]`` is the producing ``Mutable``'s element type, ``count`` its
    sample count — a ``(W, count)`` plane-major rectangle read as
    ``feat.at(plane=p, sample=s)``, binding the SAME ``lookup`` role a table
    does, with a DECLARED offset and a derivable reverse-pass fan-in."""

    def __class_getitem__(cls, key: object) -> Plane:
        if not isinstance(key, tuple) or len(key) != 2:
            raise HawkError(
                "Staged[...] takes Staged[Vector[W], count] — the producing "
                "Mutable's element type and its sample count, e.g. "
                "Staged[Vector[5], 8192]"
            )
        elem, count = key
        if not isinstance(elem, TensorType) or len(elem.shape) != 1:
            raise HawkError(
                f"Staged[...]: the element type must be a Vector[W], got {elem!r}")
        return Plane("table", Scalar, staged=True,
                     dims=_axes("Staged", (slice("plane", elem.shape[0]),
                                           slice("sample", count))))


class Wide:
    """``Wide[W]`` — a per-sample WIDE plane of RUNTIME extent, stride ``W``.

    Only the per-edge STRIDE is declared (the caller binds the runtime
    length); a read is ``buf.at(edge=e, component=c)``, folding to
    ``e * W + c``. Binds the ``wide_in`` role, same 32-byte
    ``ScalarHandle`` a ``lookup`` plane rides."""

    def __class_getitem__(cls, w: int) -> Plane:
        width = _extent("Wide", w)
        return Plane("wide", Scalar,
                     dims=(("edge", None, width), ("component", width, 1)))


class WideOut:
    """``WideOut[...]`` — the wide OUTPUT plane a body commits to, assigned
    ``buf = v`` for the own column or scattered with
    ``buf.write(v, at=lane)``, for the same reason :class:`Wide` is here: a
    role an author cannot write is a hole in the language."""

    def __class_getitem__(cls, t: TensorType) -> Plane:
        return Plane("wide_out", _payload("WideOut", t))


class Accum:
    """``Accum[...]`` — a cross-sample accumulate target (the walk's scatter)."""

    def __class_getitem__(cls, t: TensorType) -> Plane:
        return Plane("accum", _payload("Accum", t))


#: ``Param`` — one broadcast by-value scalar (the ``uniform``).
Param = Plane("uniform", Scalar)

#: ``Terminated`` — the reserved per-sample boolean MASK (role
#: ``terminated``). Declared, never written: HAWK's guard seam gates on it.
Terminated = Plane("terminated", TensorType((), "bool"))


def Reduce(op: str, of: TensorType = Scalar) -> Plane:  # a declaration
    """``Reduce(op)`` — a mapreduce partial sink, EXCLUSIVE per; ``op``
    must be a declared aether functor."""
    if op not in REDUCE_OPS:
        raise HawkError(
            f"Reduce({op!r}): not a declared aether functor; declared ops "
            f"are {REDUCE_OPS}"
        )
    return Plane("reduce", _payload("Reduce", of), op)


def _extent(what: str, w: object) -> int:
    if not isinstance(w, int) or isinstance(w, bool) or w <= 0:
        raise HawkError(
            f"{what}[...]: an extent is a positive compile-time int — it is a TYPE, "
            f"never a runtime value; got {w!r}")
    return int(w)


def _axes(what: str, key: object) -> tuple:
    """``key`` as a ROW-MAJOR axis layout ``((label, size, stride), …)``.

    Spelled ``Table["row":rows, "col":cols]`` via Python's slice syntax.
    The last declared axis is innermost (stride 1), so the axis the lane
    varies fastest over goes LAST."""
    items = key if isinstance(key, tuple) else (key,)
    axes = []
    for item in items:
        if not isinstance(item, slice) or item.step is not None\
                or not isinstance(item.start, str) or not item.start:
            raise HawkError(
                f"{what}[...]: a named axis is spelled <label>:<extent> (e.g. "
                f'{what}["row":rows, "col":cols]); got {item!r}')
        axes.append((item.start, _extent(what, item.stop)))
    labels = [a for a, _ in axes]
    if len(set(labels)) != len(labels):
        raise HawkError(
            f"{what}[...]: duplicate axis label(s) in {labels} — an axis label "
            "names the stride constant the reads fold against, so two axes may "
            "not share one")
    out, stride = [], 1
    for label, size in reversed(axes):
        out.append((label, size, stride))
        stride *= size
    return tuple(reversed(out))


def _payload(what: str, t: object) -> TensorType:
    if not isinstance(t, TensorType):
        raise HawkError(
            f"{what}[...] takes a payload type (Scalar, Index, Quat, Vector[W], "
            f"Matrix[R,C]), got {t!r}"
        )
    return t


def resolve(param: str, annotation: object) -> Plane | Quantity:
    """The declaration a parameter's annotation names."""
    if isinstance(annotation, (Plane, Quantity)):
        return annotation
    if isinstance(annotation, TensorType):
        return Plane("in", annotation)
    raise HawkError(
        f"parameter {param!r}: {annotation!r} is not a HAWK declaration. Annotate "
        "it with a payload type (Scalar / Index / Quat / Vector[W] / Matrix[R,C]), "
        "a Param / Mutable[...] / Table[...] / Staged[...] / Wide[W] / "
        "WideOut[...] / Accum[...] / Reduce(op) form, or a Quantity"
    )


#: How an input plane's RANK picks its role.
_IN_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}
_FORM_ROLE = {"uniform": "uniform", "table": "lookup", "mutable": "mutable",
              "accum": "accum_out", "reduce": "accum_out",
              "terminated": "terminated", "wide": "wide_in",
              "wide_out": "wide_out"}


def role_of(plane: Plane) -> str:
    """The role a resolved declaration binds its slot under."""
    if plane.form == "in":
        return _IN_ROLE[len(plane.ttype.shape)]
    return _FORM_ROLE[plane.form]
