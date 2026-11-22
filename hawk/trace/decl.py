# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The declaration vocabulary a kernel author annotates parameters with.

A parameter's annotation IS its vocabulary declaration: a payload type
declares a per-sample input plane whose ROLE follows its rank; a wrapper
(``Mutable[...]``, ``Table[...]``, ``Param``) declares the
output/lookup/uniform forms; a :class:`~hawk._ir.Quantity` declares a compound access,
namespaced and all-or-nothing. Nothing here builds IR; it resolves an
annotation to the ``(form, ttype)`` pair :mod:`hawk.trace.kernel` binds.

NAMED AXES: ``Table["row":rows, "col":cols]`` and
``Staged[Vector[W], count]`` declare a plane's ordered axis list as a fact of the plane: a read or a
store takes one index per axis, and no stride is ever written down. Storage is row-major, last axis
innermost.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass, replace

from .. import _ir
from ..errors import HawkError

#: Payload types. ``Vector``/``Matrix`` are subscripted with static extents —
#: an extent is a TYPE, never a runtime value.
class PayloadType(_ir.TensorType):
    """A payload type an author annotates with: a ``TensorType`` that is never called.

    Only ``Index`` can be subscripted, by a width.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Write the type as the plain record it equals."""
        return f"TensorType(shape={self.shape!r}, dtype={self.dtype!r}, tag={self.tag!r})"

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Refuse a call: a declaration is only annotated.

        ``typing.Annotated`` wants a callable as its first argument before Python 3.11.
        """
        raise TypeError(f"{self!r} is a type declaration: annotate with it, do not call it")

    def __getitem__(self, width: object) -> _ir.TensorType:
        """Return ``Index[int64]``, the 64-bit index plane (``Index`` alone is the 32-bit one).

        A position or a slice reads the record as the tuple it is; any other type is refused by name.
        """
        if isinstance(width, (int, slice)):
            return tuple(self)[width]
        if self != Index:
            raise TypeError(f"{self!r} is not subscriptable; only Index takes a width, Index[int64]")
        name = getattr(width, "name", None) or getattr(width, "__name__", None) or str(width)
        if name == "int64":
            return _INDEX64
        if name == "int32":
            return self
        raise TypeError(f"Index[...] takes int32 (the default) or int64, got {width!r}")


Scalar = PayloadType((), "f64")
#: ``Index``: a 32-bit index plane on the wire (an ``int32`` column), widened to the 64-bit
#: ``Int`` the body computes in at every read. ``Index[int64]`` is the 64-bit plane (an
#: ``int64`` column), for values past ``2**31 - 1``.
Index = PayloadType((), "i32")
_INDEX64 = PayloadType((), "i64")
#: The 32-bit integer plane payloads: stored at 4 bytes (``np.int32`` /
#: ``np.uint32``), wrapping ``+ - *``, Python's ``//``/``%`` sign rules, the
#: bitwise operators and shifts; mixed with a float only through
#: ``hawk.math.real``/``int32``/``uint32``; never differentiated.
Int32 = PayloadType((), "int32")
UInt32 = PayloadType((), "uint32")
Quat = PayloadType((4,), "f64", "quaternion")


def _items(key: object) -> tuple:
    """The items of a subscript ``Plane[key]``: a tuple key is its items, anything else one item.

    A type is one item although it is a tuple too (a record).
    """
    return key if isinstance(key, tuple) and not isinstance(key, _ir.TensorType) else (key,)


class Vector:
    """``Vector[W]`` — a rank-1 payload of static width ``W``."""

    def __class_getitem__(cls, w: int) -> PayloadType:
        return PayloadType((int(w),), "f64")


class Matrix:
    """``Matrix[R, C]`` — a rank-2 payload of static extents."""

    def __class_getitem__(cls, rc: tuple[int, int]) -> PayloadType:
        r, c = rc
        return PayloadType((int(r), int(c)), "f64")


class RecordView:
    """What a kernel body sees for a ``Record`` parameter: ``rec.pos`` is the plane ``<parameter>_pos``."""

    __slots__ = ("_name", "_fields")

    def __init__(self, name: str, fields: dict) -> None:
        """Group the traced planes of the record parameter ``name`` by field."""
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, field: str) -> object:
        """The traced plane of ``field``; an unknown field is refused naming the record's fields."""
        try:
            return self._fields[field]
        except KeyError:
            raise HawkError(
                f"record {self._name!r} has no field {field!r}; its fields are {sorted(self._fields)}"
            ) from None

    def __setattr__(self, field: str, value: object) -> None:
        """Refuse an assignment: a record is a name for planes, and a field is one plane."""
        raise HawkError(f"record {self._name!r}: fields are read, not assigned (`{self._name}.{field} = ...`)")


class Record:
    """``Record(pos=Vector[3], mass=Scalar)`` — a NAME for a group of per-sample planes of one extent.

    A kernel parameter annotated with a record expands to one plane per field, named
    ``<parameter>_<field>``, and the body reads ``rec.pos``. The record has no storage and no node of
    its own: the kernel equals the same kernel written with the separate planes. A call binds one array
    per expanded name, or the parameter's own name given a mapping or an object with exactly the fields
    the kernel reads.
    The sidecar lists the grouping under ``records`` for a tool that shows it.
    """

    __slots__ = ("fields",)

    def __init__(self, **fields: _ir.TensorType) -> None:
        """Declare the fields, each a payload type (``Scalar``, ``Vector[W]``, ``Index``, ...)."""
        if not fields:
            raise HawkError("Record(...) needs at least one field, e.g. Record(pos=Vector[3], mass=Scalar)")
        for field, ttype in fields.items():
            if field.startswith("_") or keyword.iskeyword(field):
                raise HawkError(f"Record(...): {field!r} cannot be a field name; it starts with an underscore or is reserved")
            if not isinstance(ttype, _ir.TensorType):
                raise HawkError(f"Record(...): field {field!r} takes a payload type (Scalar, Vector[W], ...), got {ttype!r}")
        self.fields = tuple(fields.items())

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Refuse a call: a declaration is only annotated (``typing.Annotated`` wants a callable before Python 3.11)."""
        raise HawkError("a Record DECLARATION is annotated, not called")

    def planes(self, parameter: str) -> dict:
        """The planes the parameter ``parameter`` expands to: ``{<parameter>_<field>: Plane}``, in field order."""
        return {
            f"{parameter}_{field}": Plane("in", ttype, group=(parameter, field, "record"))
            for field, ttype in self.fields
        }


class Rows:
    """``Rows(Index, Scalar)`` — ragged rows: an offsets plane related to a values plane.

    A kernel parameter annotated with it expands to two planes of runtime length, ``<parameter>_offsets`` and
    ``<parameter>_values``. ``offsets`` holds one start per row plus the end (CSR style), so row ``i`` is
    ``values[offsets[i]:offsets[i + 1]]``. In the body ``row = rows[i]`` gives ``row.length`` and the loop
    ``for x in row.bounded(64):``, which visits at most 64 entries of the row: a longer row is TRUNCATED at the
    bound, an empty row visits none. The relation is listed in the sidecar under ``rows``. A call binds the
    two planes by their names, or the parameter's own name given a mapping or an object with ``offsets`` and
    ``values``.
    """

    __slots__ = ("offsets", "values")

    def __init__(self, offsets: _ir.TensorType, values: _ir.TensorType) -> None:
        """Declare the offsets (``Index``, 32-bit starts) and the payload type of the values."""
        if offsets != Index:
            raise HawkError(f"Rows(offsets, values): the offsets are 32-bit starts, Index, got {offsets!r}")
        if not isinstance(values, _ir.TensorType) or values.shape:
            raise HawkError(f"Rows(offsets, values): the values are a rank-0 payload (Scalar, Index, ...), got {values!r}")
        self.offsets, self.values = offsets, values

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Refuse a call: a declaration is only annotated (``typing.Annotated`` wants a callable before Python 3.11)."""
        raise HawkError("a Rows DECLARATION is annotated, not called")

    def planes(self, parameter: str) -> dict:
        """The planes the parameter ``parameter`` expands to: ``{<parameter>_offsets: Plane, <parameter>_values: Plane}``."""
        return {
            f"{parameter}_{field}": Plane("table", ttype, group=(parameter, field, "rows"))
            for field, ttype in (("offsets", self.offsets), ("values", self.values))
        }


class Tensor:
    """``Tensor[A, B, C, ...]`` — a payload of any rank with static extents, e.g. ``Tensor[3, 3, 3]``.

    Rank 1 and 2 are ``Vector[W]`` and ``Matrix[R, C]``. A value of rank 3 or more is built and
    computed in the body (elementwise operations, ``t[a, b, c]``, ``t[a, b, c] = e``) or shared by all
    lanes as ``Param[Tensor[...]]``; as a per-lane plane it is declared as a ``Table`` with named axes.
    """

    def __class_getitem__(cls, extents: object) -> PayloadType:
        items = _items(extents)
        if not items or not all(isinstance(e, int) and not isinstance(e, bool) and e > 0 for e in items):
            raise HawkError(f"Tensor[...] takes one positive static extent per axis, got {extents!r}")
        return PayloadType(tuple(items), "f64")


@dataclass(frozen=True)
class Plane:
    """A resolved declaration: its ``form`` plus the payload type it
    carries. ``axes`` is the declared axis list ``((label, extent), ...)``, an extent of ``None``
    being known at run time (empty for a positional
    plane); ``staged`` marks the one role whose adjoint FAN-IN is a
    derivable number."""

    form: str
    ttype: _ir.TensorType
    axes: tuple = ()
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
    #: ``(parameter, field, form)`` of a plane that is one field of a :class:`Record` parameter (form ``"record"``) or
    #: one of the two planes of a :class:`Rows` parameter (form ``"rows"``), else empty.
    group: tuple = ()
    #: The name of the appended plane a one-element count belongs to (``hawk.append``), else empty.
    counts: str = ""
    #: The name of the plane a plane of winner lanes ranks (``hawk.scatter(y, contract="ordered")[k].min(v)`` makes one), else empty.
    winner_of: str = ""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Never called: a declaration is only annotated. Defined so that ``typing.Annotated``
        accepts it as its first argument before Python 3.11, which requires a callable."""
        raise HawkError(f"a `{self.form}` DECLARATION is annotated, not called")

    def __setitem__(self, key: object, value: object) -> None:
        from ..errors import HawkError

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
        items = _items(key)
        if isinstance(items[0], (slice, str)):
            return Plane("mutable", Scalar, axes=_axes("Mutable", items))
        axes = _axes("Mutable", items[1:]) if items[1:] else ()
        return Plane("mutable", _payload("Mutable", items[0]), axes=axes)


class Table:
    """``Table[...]`` — a lookup plane read with ``at()``.

    ``Table[Scalar]`` is the POSITIONAL plane, read at a flat index;
    ``Table["row":rows, "col":cols]`` is the NAMED-AXIS plane, read
    ``t.at(row=…, col=…)``, one index per axis; ``Table[Vector[3], "row":rows]``
    holds a value of that type at each position, bound ``(*payload, *axes)``. A bare first
    label (``Table["elem", "c":3]``) is a runtime extent, as in ``Grid``."""

    def __class_getitem__(cls, key: object) -> Plane:
        items = _items(key)
        if isinstance(items[0], (slice, str)):
            return Plane("table", Scalar, axes=_axes("Table", items))
        axes = _axes("Table", items[1:]) if items[1:] else ()
        return Plane("table", _payload("Table", items[0]), axes=axes)


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
        items = _items(key)
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

    def __class_getitem__(cls, t: _ir.TensorType) -> Plane:
        if t not in (Index, _INDEX64):
            raise HawkError(f"Sorted[...] takes Index (a per-sample segment id), got {t!r}")
        return Plane("in", t, sorted=True)


class Staged:
    """``Staged[Vector[W], count]`` — a buffer one launch PRODUCES and the
    next CONSUMES, declared as the producer wrote it.

    ``Vector[W]`` is the producing ``Mutable``'s element type, ``count`` its
    sample count — a ``(W, count)`` plane-major rectangle read as
    ``feat.at(plane=p, sample=s)``, binding the SAME ``lookup`` role a table
    does, with a DECLARED offset and a derivable reverse-pass fan-in.

    ``count`` is an int, or ``m.n_samples`` for a buffer of exactly the launch's samples: the kernel is then
    built once for every launch size (an int count is part of the build, so each size is its own). An int smaller
    than the launch is a staging window."""

    def __class_getitem__(cls, key: object) -> Plane:
        if not isinstance(key, tuple) or len(key) != 2:
            raise HawkError(
                "Staged[...] takes Staged[Vector[W], count] — the producing "
                "Mutable's element type and its sample count, e.g. "
                "Staged[Vector[5], 8192]"
            )
        elem, count = key
        if not isinstance(elem, _ir.TensorType) or len(elem.shape) != 1:
            raise HawkError(
                f"Staged[...]: the element type must be a Vector[W], got {elem!r}")
        from .value import n_samples
        if count is n_samples:      # the count is the launch count: the sample axis has a run-time extent
            return Plane("table", Scalar, staged=True, axes=(("plane", elem.shape[0]), ("sample", None)))
        return Plane("table", Scalar, staged=True,
                     axes=_axes("Staged", (slice("plane", elem.shape[0]),
                                           slice("sample", count))))


class Wide:
    """``Wide[W]`` — a per-sample WIDE plane of RUNTIME extent and ``W`` components.

    The edge axis has a run-time extent (the caller binds it); a read is
    ``buf.at(edge=e, component=c)``. Binds the ``wide_in`` role, same 32-byte
    ``ScalarHandle`` a ``lookup`` plane rides."""

    def __class_getitem__(cls, w: int) -> Plane:
        width = _extent("Wide", w)
        return Plane("wide", Scalar,
                     axes=(("edge", None), ("component", width)))


class WideOut:
    """``WideOut[...]`` — the wide OUTPUT plane a body commits to, assigned
    ``buf = v`` for the own column or stored by key with
    ``hawk.scatter(buf)[k] = v``, for the same reason :class:`Wide` is here: a
    role an author cannot write is a hole in the language."""

    def __class_getitem__(cls, t: _ir.TensorType) -> Plane:
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
        if isinstance(t, _ir.TensorType) and t.shape:
            return _shared_value(t)
        if isinstance(t, Plane) and t.form == "table":
            return _shared_table(t)
        raise HawkError(
            f"Param[...] takes Scalar, Int32 or UInt32, a float Vector[W] or Matrix[R,C], or "
            f"a Table with static named axes, got {t!r}")


def _shared_value(t: _ir.TensorType) -> Plane:
    """The shared declaration of a vector, matrix or higher-rank payload: its cells, row-major."""
    if t.dtype != "f64" or t.tag is not None:
        raise HawkError(
            f"Param[{spell(t)}]: a shared value is a float Vector[W], Matrix[R,C] or Tensor[...]; "
            "an integer or tagged value is passed as "
            "separate Scalar/Int32 Params, or as a Table of its cells")
    labels = {1: ("component",), 2: ("row", "col")}.get(len(t.shape)) \
        or tuple(f"axis{k}" for k in range(len(t.shape)))
    axes = (("component", t.shape[0]),) if len(t.shape) == 1 \
        else _axes("Param", tuple(slice(label, extent) for label, extent in zip(labels, t.shape)))
    return Plane("table", t, axes=axes, shared=tuple(t.shape))


def _shared_table(plane: Plane) -> Plane:
    """A ``Table`` as one shared value: named axes, every extent static (the
    caller's array is checked against them)."""
    if plane.ttype != Scalar or plane.staged or not plane.axes \
            or any(extent is None for _label, extent in plane.axes):
        raise HawkError(
            'Param[Table[...]] takes named axes with static extents, '
            'Param[Table["row":R, "col":C]]; a plain Table is already one plane '
            "shared by all lanes, with a runtime length")
    return replace(plane, shared=tuple(extent for _label, extent in plane.axes))


#: ``Param`` — one broadcast by-value scalar (the ``uniform``).
Param = _ParamPlane("uniform", Scalar)

#: ``Terminated`` — the reserved per-sample boolean MASK (role
#: ``terminated``). Declared, never written: HAWK's guard seam gates on it.
Terminated = Plane("terminated", _ir.TensorType((), "bool"))


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
    """``key`` as an ordered axis list ``((label, extent), …)``.

    Spelled ``Table["row":rows, "col":cols]`` via Python's slice syntax.
    The last declared axis is innermost, so the axis the lane varies fastest
    over goes LAST. A ``Mutable``/``Table`` may leave its FIRST axis bare
    (``(label, None)``, a runtime extent)."""
    items = _items(key)
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
            "names the axis a read or a store addresses, so two axes may "
            "not share one")
    return tuple(axes)


def _payload(what: str, t: object) -> _ir.TensorType:
    if not isinstance(t, _ir.TensorType):
        raise HawkError(
            f"{what}[...] takes a payload type (Scalar, Index, Index[int64], Int32, UInt32, Quat, "
            f"Vector[W], Matrix[R,C]), got {t!r}"
        )
    return t


def spell(t: _ir.TensorType) -> str:
    """``t`` as an author writes it: ``Scalar``, ``Vector[3]``, ``Matrix[2, 3]``, ``Tensor[2, 3, 4]``."""
    for name, named in (("Scalar", Scalar), ("Index", Index), ("Index[int64]", _INDEX64), ("Int32", Int32),
                        ("UInt32", UInt32), ("Quat", Quat)):
        if t == named:
            return name
    if t.dtype == "f64" and len(t.shape) == 1:
        return f"Vector[{t.shape[0]}]"
    if t.dtype == "f64" and len(t.shape) == 2:
        return f"Matrix[{t.shape[0]}, {t.shape[1]}]"
    if t.dtype == "f64" and len(t.shape) > 2 and t.tag is None:
        return f"Tensor[{', '.join(map(str, t.shape))}]"
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


def resolve(param: str, annotation: object) -> Plane | _ir.Quantity:
    """The declaration a parameter's annotation names."""
    if getattr(annotation, "__metadata__", None) is not None:        # typing.Annotated
        notes = tuple(n for m in annotation.__metadata__ if isinstance(m, Meta) for n in m.notes)
        declared = resolve(param, annotation.__origin__)
        if not notes:
            return declared
        if not isinstance(declared, Plane):
            raise HawkError(f"parameter {param!r}: Meta(...) annotates a plane, not a Quantity")
        return replace(declared, meta=notes)
    if isinstance(annotation, (Plane, _ir.Quantity)):
        return annotation
    if isinstance(annotation, _ir.TensorType):
        return Plane("in", annotation)
    raise HawkError(
        f"parameter {param!r}: {annotation!r} is not a HAWK declaration. Annotate "
        "it with a payload type (Scalar / Index / Int32 / UInt32 / Quat / Vector[W] / "
        "Matrix[R,C]), "
        "a Param / Mutable[...] / Table[...] / Staged[...] / Wide[W] / "
        "WideOut[...] form, or a Quantity"
    )


_FORM_ROLE = {"uniform": "uniform", "table": "lookup", "mutable": "mutable",
              "terminated": "terminated", "wide": "wide_in",
              "wide_out": "plane_out"}


def role_of(plane: Plane) -> str:
    """The role a resolved declaration binds its slot under."""
    if plane.form == "in":
        return "plane_in"
    return _FORM_ROLE[plane.form]
