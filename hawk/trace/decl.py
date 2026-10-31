# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The declaration vocabulary a kernel author annotates parameters with.

A parameter's annotation IS its vocabulary declaration: a payload type
declares a per-sample input plane whose ROLE follows its rank; a wrapper
(``Mutable[...]``, ``Table[...]``, ``Param``) declares the
output/lookup/uniform forms; a :class:`~hawk.ir.compound.Quantity` declares a compound access,
namespaced and all-or-nothing. Nothing here builds IR; it resolves an
annotation to the ``(form, ttype)`` pair :mod:`hawk.trace.kernel` binds.

NAMED AXES: ``Table["row":rows, "col":cols]`` and
``Staged[Vector[W], count]`` declare a plane's axis LAYOUT as a fact, not a
convention, with each stride minted as a :class:`~hawk.ir.nodes.RoleConst`
naming its axis, rather than a hand-written index that re-derives the
producer's layout uncheckably. Layout is ROW-MAJOR, last axis innermost.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, replace

from ..ir import HawkError, Quantity
from ..types import TensorType

#: Payload types. ``Vector``/``Matrix`` are subscripted with static extents —
#: an extent is a TYPE, never a runtime value.
Scalar = TensorType((), "f64")
#: ``Index``: a 32-bit index plane on the wire (an ``int32`` column), widened to the 64-bit
#: ``Int`` the body computes in at every read. ``Index[int64]`` is the 64-bit plane (an
#: ``int64`` column), for values past ``2**31 - 1``.
Index = TensorType((), "i32")
_INDEX64 = TensorType((), "i64")
#: The 32-bit integer plane payloads: stored at 4 bytes (``np.int32`` /
#: ``np.uint32``), wrapping ``+ - *``, Python's ``//``/``%`` sign rules, the
#: bitwise operators and shifts; mixed with a float only through
#: ``hawk.math.real``/``int32``/``uint32``; never differentiated.
Int32 = TensorType((), "int32")
UInt32 = TensorType((), "uint32")
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
    dims: tuple = ()
    staged: bool = False
    #: ``Grid[...]``: ``((label, extent|None), ...)``, the launch shape.
    grid: tuple = ()
    #: ``Sorted[Index]``: the values are non-decreasing along the flat lane (a
    #: promise the bind checks only under ``HAWK_CHECK_SORTED``).
    sorted: bool = False
    #: ``Param[Vector[W]]``/``Param[Matrix[R,C]]``/``Param[Table[...]]``: the
    #: shape of the ONE value every lane shares (empty for any other plane).
    shared: tuple = ()
    #: ``Annotated[T, Meta(...)]``: opaque ``(key, value)`` pairs the sidecar carries for a reader.
    meta: tuple = ()

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Never called: a declaration is only annotated. Defined so that ``typing.Annotated``
        accepts it as its first argument before Python 3.11, which requires a callable."""
        raise HawkError(f"a `{self.form}` DECLARATION is annotated, not called")

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
    """``Mutable[...]`` — the output plane a body assigns (the ``assign``).

    ``Mutable["bin":256]`` and ``Mutable[Vector[3], "channel":64, "input":64]``
    name the sink's own axes: a plane the body writes by key, bound
    ``(*payload, *axes)`` (the payload is ``Scalar`` unless given first). A sink
    you pass keeps its contents; one you omit starts at zero. A bare label
    (``Mutable["node"]``) is a runtime extent and may only be first, as in
    ``Grid``: the caller gives it as ``node=N`` or by the plane it passes."""

    def __class_getitem__(cls, key: object) -> Plane:
        items = key if isinstance(key, tuple) else (key,)
        if isinstance(items[0], (slice, str)):
            return Plane("mutable", Scalar, dims=_axes("Mutable", items))
        dims = _axes("Mutable", items[1:]) if items[1:] else ()
        return Plane("mutable", _payload("Mutable", items[0]), dims=dims)


class Table:
    """``Table[...]`` — a lookup plane read with ``at()``.

    ``Table[Scalar]`` is the POSITIONAL plane, read at a flat index;
    ``Table["row":rows, "col":cols]`` is the NAMED-AXIS plane, read
    ``t.at(row=…, col=…)`` against declaration-minted strides. A bare first
    label (``Table["elem", "c":3]``) is a runtime extent, as in ``Grid``: no
    stride depends on it."""

    def __class_getitem__(cls, key: object) -> Plane:
        if isinstance(key, TensorType):
            return Plane("table", _payload("Table", key))
        return Plane("table", Scalar, dims=_axes("Table", key))


class Grid:
    """``Grid["channel", "sample":8192]`` — the shape of the launch: one lane per
    point, the body is one lane's code.

    Labels are in numpy order and the LAST axis is the fastest (consecutive
    lanes). A bare label is a runtime extent and may only be first; ``"l":k`` is
    static, and every axis may be. The body reads an axis as an Index:
    ``lane.channel``, ``lane[0]`` (0 is the slowest, -1 the fastest), and
    ``lane.flat`` is the flat lane index. A coordinate, never a neighbour: read
    another lane with ``u.at(...)``."""

    def __class_getitem__(cls, key: object) -> Plane:
        items = key if isinstance(key, tuple) else (key,)
        if not items:
            raise HawkError('Grid[...] needs at least one axis, e.g. Grid["i"]')
        axes = []
        for pos, item in enumerate(items):
            if isinstance(item, str) and item and not pos:
                axes.append((item, None))
            elif isinstance(item, slice) and item.step is None \
                    and isinstance(item.start, str) and item.start:
                axes.append((item.start, _extent("Grid", item.stop)))
            elif isinstance(item, str) and item:
                raise HawkError(
                    f"Grid[...]: axis {item!r} at position {pos} has no extent; only "
                    f'the FIRST axis may be bare (a runtime extent), write "{item}":<extent>')
            else:
                raise HawkError(
                    f'Grid[...]: an axis is "<label>" (first only) or "<label>":<extent>, '
                    f'e.g. Grid["channel", "sample":8192]; got {item!r}')
        labels = [a for a, _ in axes]
        for label in labels:
            why = _label_refusal(label)
            if why:
                raise HawkError(f"Grid[...]: the axis label {label!r} {why}")
        if len(set(labels)) != len(labels):
            raise HawkError(f"Grid[...]: duplicate axis label(s) in {labels}")
        return Plane("in", Index, grid=tuple(axes))


class Sorted:
    """``Sorted[Index]`` — a per-sample Index plane whose values never decrease
    along the flat lane (segment ids): a promise the compiler may build on.
    Set ``HAWK_CHECK_SORTED`` to have each bind check it."""

    def __class_getitem__(cls, t: TensorType) -> Plane:
        if t not in (Index, _INDEX64):
            raise HawkError(f"Sorted[...] takes Index (a per-sample segment id), got {t!r}")
        return Plane("in", t, sorted=True)


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
    ``buf = v`` for the own column or stored by key with
    ``hawk.scatter(buf)[k] = v``, for the same reason :class:`Wide` is here: a
    role an author cannot write is a hole in the language."""

    def __class_getitem__(cls, t: TensorType) -> Plane:
        return Plane("wide_out", _payload("WideOut", t))


@dataclass(frozen=True)
class _ParamPlane(Plane):
    """``Param`` — one value shared by every lane, whatever type it wraps.

    ``Param[T]`` is the one rule: the bare type ``T`` (``Scalar``, ``Vector[W]``,
    ``Matrix[R,C]``, ``Table[...]``) is per-lane, and ``Param`` in front of it makes it
    ONE value that all lanes read. A scalar is passed by value (``Param[Int32]``,
    ``Param[UInt32]`` an integer one, as a Python ``int``); a vector, matrix or
    named-axis table is passed as an array of its own shape and read in the body as
    ``p[j]``, ``p[i, j]``, ``p.at(...)`` or whole, from memory, never copied per lane."""

    def __getitem__(self, t: object) -> Plane:
        """``Param[t]``: ``t`` as one value shared by all lanes.

        :param t: ``Scalar``, ``Int32``, ``UInt32``, ``Vector[W]``, ``Matrix[R,C]`` or a
            ``Table`` with static named axes.
        :returns: the shared declaration (``uniform`` for a scalar, else a ``lookup``
            plane of the value's cells, row-major).
        :raises HawkError: for any other payload, naming the supported ones.

        Example: ``def k(x: Scalar, n: Param[Int32], A: Param[Matrix[2, 2]], y: Mutable[Scalar]): ...``"""
        if t in (Scalar, Int32, UInt32):
            return Plane("uniform", t)
        if isinstance(t, TensorType) and t.shape:
            return _shared_value(t)
        if isinstance(t, Plane) and t.form == "table":
            return _shared_table(t)
        raise HawkError(
            f"Param[...] takes Scalar, Int32 or UInt32, a float Vector[W] or Matrix[R,C], or "
            f"a Table with static named axes, got {t!r}")


def _shared_value(t: TensorType) -> Plane:
    """The shared declaration of a vector or matrix payload: its cells, row-major."""
    if t.dtype != "f64" or t.tag is not None or len(t.shape) not in (1, 2):
        raise HawkError(
            f"Param[{spell(t)}]: a shared vector or matrix is a float Vector[W] or "
            "Matrix[R,C] (rank 1 or 2); an integer or tagged value is passed as "
            "separate Scalar/Int32 Params, or as a Table of its cells")
    dims = (("component", t.shape[0], 1),) if len(t.shape) == 1 \
        else _axes("Param", (slice("row", t.shape[0]), slice("col", t.shape[1])))
    return Plane("table", t, dims=dims, shared=tuple(t.shape))


def _shared_table(plane: Plane) -> Plane:
    """A ``Table`` as one shared value: named axes, every extent static (the
    caller's array is checked against them)."""
    if plane.ttype != Scalar or plane.staged or not plane.dims \
            or any(size is None for _label, size, _stride in plane.dims):
        raise HawkError(
            'Param[Table[...]] takes named axes with static extents, '
            'Param[Table["row":R, "col":C]]; a plain Table is already one plane '
            "shared by all lanes, with a runtime length")
    return replace(plane, shared=tuple(size for _label, size, _stride in plane.dims))


#: ``Param`` — one broadcast by-value scalar (the ``uniform``).
Param = _ParamPlane("uniform", Scalar)

#: ``Terminated`` — the reserved per-sample boolean MASK (role
#: ``terminated``). Declared, never written: HAWK's guard seam gates on it.
Terminated = Plane("terminated", TensorType((), "bool"))


def _extent(what: str, w: object) -> int:
    if not isinstance(w, int) or isinstance(w, bool) or w <= 0:
        raise HawkError(
            f"{what}[...]: an extent is a positive compile-time int — it is a TYPE, "
            f"never a runtime value; got {w!r}")
    return int(w)


#: Words a ``Grid`` label may not be: the flat lane, and the launch keywords an
#: extent passed by label would collide with.
RESERVED_LABELS = ("flat", "n", "n_samples", "base", "count", "layout")


def _label_refusal(label: str) -> str:
    if not label.isidentifier() or keyword.iskeyword(label):
        return "must be a plain identifier (not a Python keyword): it is read as lane.<label>"
    if label in RESERVED_LABELS:
        return f"is reserved ({', '.join(RESERVED_LABELS)}); pick another"
    return ""


def _axes(what: str, key: object) -> tuple:
    """``key`` as a ROW-MAJOR axis layout ``((label, size, stride), …)``.

    Spelled ``Table["row":rows, "col":cols]`` via Python's slice syntax.
    The last declared axis is innermost (stride 1), so the axis the lane
    varies fastest over goes LAST. A ``Mutable``/``Table`` may leave its FIRST
    axis bare (``(label, None)``, a runtime extent): every stride is the
    product of the axes after it, so none depends on that extent."""
    items = key if isinstance(key, tuple) else (key,)
    axes = []
    for pos, item in enumerate(items):
        if what in ("Mutable", "Table") and isinstance(item, str) and item:
            if pos:
                raise HawkError(
                    f"{what}[...]: axis {item!r} at position {pos} has no extent; only "
                    f'the FIRST axis may be bare (a runtime extent), write "{item}":<extent>')
            axes.append((item, None))
            continue
        if not isinstance(item, slice) or item.step is not None\
                or not isinstance(item.start, str) or not item.start:
            raise HawkError(
                f"{what}[...]: a named axis is spelled <label>:<extent> (e.g. "
                f'{what}["row":rows, "col":cols]); got {item!r}')
        axes.append((item.start, _extent(what, item.stop)))
    labels = [a for a, _ in axes]
    if axes and axes[0][1] is None and _label_refusal(labels[0]):
        raise HawkError(f"{what}[...]: the axis label {labels[0]!r} {_label_refusal(labels[0])}")
    if len(set(labels)) != len(labels):
        raise HawkError(
            f"{what}[...]: duplicate axis label(s) in {labels} — an axis label "
            "names the stride constant the reads fold against, so two axes may "
            "not share one")
    out, stride = [], 1
    for label, size in reversed(axes):
        out.append((label, size, stride))
        if size is not None:
            stride *= size
    return tuple(reversed(out))


def _payload(what: str, t: object) -> TensorType:
    if not isinstance(t, TensorType):
        raise HawkError(
            f"{what}[...] takes a payload type (Scalar, Index, Index[int64], Int32, UInt32, Quat, "
            f"Vector[W], Matrix[R,C]), got {t!r}"
        )
    return t


def spell(t: TensorType) -> str:
    """``t`` as an author writes it: ``Scalar``, ``Vector[3]``, ``Matrix[2, 3]``."""
    for name, named in (("Scalar", Scalar), ("Index", Index), ("Index[int64]", _INDEX64), ("Int32", Int32),
                        ("UInt32", UInt32), ("Quat", Quat)):
        if t == named:
            return name
    if t.dtype == "f64" and len(t.shape) == 1:
        return f"Vector[{t.shape[0]}]"
    if t.dtype == "f64" and len(t.shape) == 2:
        return f"Matrix[{t.shape[0]}, {t.shape[1]}]"
    return repr(t)


class Meta:
    """``Annotated[Vector[3], Meta(unit="km")]`` — opaque notes on a plane.

    HAWK never reads them: they are written under ``plane_meta`` in the sidecar, by plane
    name, and eagle hands them back from the loaded kernel. Values must be JSON."""

    def __init__(self, **notes: object) -> None:
        import json

        try:
            json.dumps(notes)
        except TypeError as exc:
            raise HawkError(f"Meta(...) values must be JSON: {exc}") from None
        self.notes = tuple(sorted(notes.items()))


def resolve(param: str, annotation: object) -> Plane | Quantity:
    """The declaration a parameter's annotation names."""
    if getattr(annotation, "__metadata__", None) is not None:        # typing.Annotated
        notes = tuple(n for m in annotation.__metadata__ if isinstance(m, Meta) for n in m.notes)
        declared = resolve(param, annotation.__origin__)
        if not notes:
            return declared
        if not isinstance(declared, Plane):
            raise HawkError(f"parameter {param!r}: Meta(...) annotates a plane, not a Quantity")
        return replace(declared, meta=notes)
    if isinstance(annotation, (Plane, Quantity)):
        return annotation
    if isinstance(annotation, TensorType):
        return Plane("in", annotation)
    raise HawkError(
        f"parameter {param!r}: {annotation!r} is not a HAWK declaration. Annotate "
        "it with a payload type (Scalar / Index / Int32 / UInt32 / Quat / Vector[W] / "
        "Matrix[R,C]), "
        "a Param / Mutable[...] / Table[...] / Staged[...] / Wide[W] / "
        "WideOut[...] form, or a Quantity"
    )


#: How an input plane's RANK picks its role.
_IN_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}
_FORM_ROLE = {"uniform": "uniform", "table": "lookup", "mutable": "mutable",
              "terminated": "terminated", "wide": "wide_in",
              "wide_out": "wide_out"}


def role_of(plane: Plane) -> str:
    """The role a resolved declaration binds its slot under."""
    if plane.form == "in":
        return _IN_ROLE[len(plane.ttype.shape)]
    return _FORM_ROLE[plane.form]
