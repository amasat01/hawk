# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How a plane's layout is classified (component-major, free sample-major
view, single sample), and the refusal texts for the rest."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import MappingProxyType

from .errors import HawkError

# -- Sample-major planes. A per-sample plane of declared width w >= 2 is
# component-major, (w, L); most array code stores vectors sample-major,
# (L, w). HAWK binds by ADDRESS and never copies, so it accepts a
# sample-major plane only when its transpose is C-contiguous (its bytes
# ARE the native (w, L) layout); any other is refused with the fix named,
# or copied by eagle's doors under an `eagle.LayoutWarning`.
#
# A shape that reads BOTH ways -- (w, w), i.e. N == w -- is AMBIGUOUS and is
# refused, never guessed. The caller says which axis holds the samples, zero
# copy, in two ways: per call, `layout="samples_first"` (planes are (N, w)) or
# `layout="samples_last"` (planes are (w, N), native), which resolves every
# ambiguous plane of the call; per array, `hawk.samples_first(x)` /
# `hawk.samples_last(x)`, honoured for ANY shape, refused when it contradicts
# the shape, and winning over the call's `layout=`. The marker PROTOCOL is
# shared with eagle (which hawk does not import): any object carrying
# `__raptor_samples_axis__` ("first" or "last") and the wrapped array as
# `.array` is a marker, so each package accepts the other's. A marker on one
# sample's value is ignored. ----------------------------------------------- #

MARK_ATTR = "__raptor_samples_axis__"
LAYOUTS = ("samples_first", "samples_last")


class _Marked:
    """One array with the axis that holds its samples declared."""

    __slots__ = ("array", "__raptor_samples_axis__")

    def __init__(self, array, axis):
        self.array = array
        self.__raptor_samples_axis__ = axis

    def __repr__(self):
        return f"hawk.samples_{self.__raptor_samples_axis__}({self.array!r})"


def samples_first(x):
    """Mark ``x`` as sample-major: its FIRST axis holds the samples,
    ``(N, w)``. Zero-copy, honoured for any shape, accepted by every hawk
    (and eagle) door that binds per-sample planes, and it beats the call's
    ``layout=``; a shape that contradicts it is refused naming the argument.
    hawk still binds by address, so a ``samples_first`` plane must be a free
    transposed view (its ``.T`` C-contiguous), as any sample-major plane."""
    return _Marked(x, "first")


def samples_last(x):
    """Mark ``x`` as component-major (native): its LAST axis holds the samples,
    ``(w, N)``. Zero-copy, honoured for any shape, beats the call's ``layout=``;
    a shape that contradicts it is refused naming the argument."""
    return _Marked(x, "last")


def check_layout(layout) -> None:
    """Refuse a ``layout=`` that is not ``None``, ``"samples_first"`` or
    ``"samples_last"``."""
    if layout is not None and layout not in LAYOUTS:
        raise HawkError(
            f"layout={layout!r} is not one of 'samples_first' (planes are "
            f"(N, w)), 'samples_last' (planes are (w, N)), or None")


def split_marks(arrays: dict, layout=None):
    """``(plain, axes)``: ``arrays`` with each marker unwrapped to its array,
    and ``{name: "first"|"last"}`` for the marked names. ``layout`` is
    validated. ``arrays`` itself is returned when nothing is marked."""
    check_layout(layout)
    axes = {}
    for name, value in arrays.items():
        axis = getattr(value, MARK_ATTR, None)
        if axis in ("first", "last") and hasattr(value, "array"):
            axes[name] = axis
    if not axes:
        return arrays, axes
    return {n: (v.array if n in axes else v) for n, v in arrays.items()}, axes


#: The ``memoryview`` format character of a DLPack ``(type code, bits)``: 0 signed, 1 unsigned, 2 float, 6 bool.
_FORMAT_OF_DLPACK = MappingProxyType({(0, 8): "b", (0, 16): "h", (0, 32): "i", (0, 64): "q", (1, 8): "B", (1, 16): "H",
                                      (1, 32): "I", (1, 64): "Q", (2, 16): "e", (2, 32): "f", (2, 64): "d", (6, 8): "?"})


class _Described:
    """What the host library states of an array that is not a NumPy array and exports DLPack: the members of a ``memoryview`` that
    hawk's checks read (``shape``, ``ndim``, ``format``, ``itemsize``, ``readonly``, the two contiguities)."""

    def __init__(self, layout: dict) -> None:
        code, bits, _lanes = layout["dtype"]
        self.shape = tuple(layout["shape"])
        self.ndim = len(self.shape)
        self.format = _FORMAT_OF_DLPACK.get((code, bits), "?")
        self.itemsize = max(bits // 8, 1)
        self.readonly = bool(layout["read_only"])
        self._strides = layout["strides"]

    def _contiguous(self, order) -> bool:
        if 0 in self.shape:
            return True         # no element: contiguous in both orders, whatever strides the library states
        if self._strides is None:
            return order == "C" or self.ndim <= 1
        expect, shape = 1, self.shape
        for axis in (reversed(range(self.ndim)) if order == "C" else range(self.ndim)):
            if shape[axis] != 1 and self._strides[axis] != expect:
                return False
            expect *= shape[axis]
        return True

    @property
    def c_contiguous(self) -> bool:
        return self._contiguous("C")

    @property
    def f_contiguous(self) -> bool:
        return self._contiguous("F")


def is_ndarray(value) -> bool:
    """Whether ``value`` is a NumPy array: the one kind of array hawk makes views of (transpose, reshape, slice) with NumPy's own
    methods. An array of any other library is handed over as it is. NumPy is not imported here: a value cannot be one of its
    arrays before the process has loaded it."""
    numpy = sys.modules.get("numpy")
    return numpy is not None and isinstance(value, numpy.ndarray)


def describe(value):
    """The shape, dtype and flags of ``value``: its ``memoryview`` for a NumPy array (or a plain buffer, which the checks refuse
    by name), else what the host library reads from the DLPack tensor of an array of another library, whether or not it also
    exports a buffer (:class:`_Described`)."""
    if is_ndarray(value) or not hasattr(value, "__dlpack__"):
        try:
            return memoryview(value)
        except TypeError:
            raise TypeError(f"{type(value).__name__} exports neither the buffer protocol nor DLPack") from None
    from . import ospreys
    from .targets import host_target

    return _Described(ospreys.serving(host_target().id).runtime_door().array_layout(value))


def _is_single(shape: tuple, width: int) -> bool:
    """Whether ``shape`` is ONE sample of a plane of declared ``width``.
    A shape that is also a batch is a batch: ``(1,)``, ``(w, 1)``,
    ``(1, w)`` and ``(w, w)`` are never one sample."""
    if width <= 1:
        return shape == ()
    if shape == (width,):
        return True
    return (len(shape) == 2 and shape[0] * shape[1] == width
            and shape[0] not in (1, width))


def plane_layout(mv: memoryview, width: int, axis=None, layout=None) -> str:
    """How a bound plane of declared per-sample ``width`` is laid out.
    ``"single"``: one sample. ``"native"``: anything not sample-major 2-D.
    ``"view"``: a sample-major plane whose transpose is C-contiguous,
    bound zero-copy. ``"copy"``: one only a copy could bind, refused.
    ``"ambiguous"``: ``(w, w)``, which reads both ways, refused unless ``axis``
    (the array's own marker, ``"first"``/``"last"``) or ``layout`` (the call's,
    ``"samples_first"``/``"samples_last"``) says which; the marker wins.
    ``"conflict"``: a marker the shape contradicts, refused."""
    shape = tuple(mv.shape)
    if _is_single(shape, width):
        return "single"
    if width <= 1 or len(shape) != 2:
        return "native"
    is_native, is_sample_major = shape[0] == width, shape[1] == width
    if axis is None and is_native and is_sample_major:
        axis = {"samples_first": "first", "samples_last": "last"}.get(layout)
        if axis is None:
            return "ambiguous"
    if axis == "first" and not is_sample_major:
        return "conflict"
    if axis == "last" and not is_native:
        return "conflict"
    if axis == "last" or (axis is None and (is_native or not is_sample_major)):
        return "native"
    return "view" if mv.f_contiguous else "copy"


@dataclass(frozen=True)
class LayoutChoice:
    """One call's layout choices: the per-call ``layout`` and the per-array
    markers. :meth:`of` classifies one bound plane under them."""

    layout: str | None = None
    axes: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "axes", self.axes or {})

    def of(self, name: str, mv: memoryview, width: int) -> str:
        return plane_layout(mv, width, self.axes.get(name), self.layout)


_NO_CHOICE = LayoutChoice()


def _native_extent(mv: memoryview, width: int, layout: str | None = None) -> int:
    """The trailing extent the plane has in its native ``(w, L)`` layout;
    ``layout`` is the plane's already-settled classification."""
    layout = layout or plane_layout(mv, width)
    if layout == "single":
        return 1
    shape = tuple(mv.shape)
    if not shape:
        return 0
    return shape[0] if layout in ("view", "copy") else shape[-1]


def number_refusal(name: str, value) -> str:
    """The refusal for a Python number bound to a plane (hawk binds by
    address; one sample is a 0-d array, never a number)."""
    return (
        f"argument {name!r} was given a Python {type(value).__name__}, which "
        f"cannot be bound by address or receive a write; pass a writable 0-d "
        f"array (np.array({name}))"
    )


def mixed_refusal(single: tuple, batch: tuple) -> str:
    """The refusal for a call that mixes one sample with a batch."""
    (s_name, s_shape), (b_name, b_shape, b_n) = single, batch
    tile = (f"np.full({b_n}, {s_name})" if s_shape == ()
            else f"np.repeat({s_name}.reshape(-1, 1), {b_n}, axis=1)")
    return (
        f"{s_name!r} is one sample (shape {s_shape}) but {b_name!r} is a batch "
        f"of {b_n} (shape {b_shape}); a call is one sample or one batch, never "
        f"both. A value shared by every sample is a Param (scalar) or a Table "
        f"(vector); a Grid kernel's per-lane input also takes a scalar; or tile it "
        f"once: {tile}"
    )


def sample_major_refusal(name: str, shape: tuple, width: int) -> str:
    """The refusal for a sample-major plane only a copy could bind."""
    return (
        f"argument {name!r}: given sample-major, shape {shape}, but this "
        f"kernel's plane is component-major ({width}, L). hawk.runtime binds "
        f"by address and never copies, so it accepts a sample-major plane only "
        f"as a zero-copy transposed view (the .T of a contiguous ({width}, L) "
        f"array). Pass np.ascontiguousarray({name}.T), mark a free view with "
        f"hawk.samples_first({name}), allocate it as "
        f"({width}, L), or launch through eagle, which copies it with an "
        f"eagle.LayoutWarning"
    )


def ambiguous_refusal(name: str, shape: tuple, width: int) -> str:
    """The refusal for a ``(w, w)`` plane that reads both ways."""
    return (
        f"argument {name!r} has shape {shape}, which reads both as "
        f"component-major ({width}, L) with L = {shape[-1]} (samples last) and "
        f"as sample-major (L, {width}) with L = {shape[-1]} (samples first); "
        f"hawk will not guess. Say which axis holds the samples, zero-copy: "
        f"pass layout=\"samples_first\" or layout=\"samples_last\" to the "
        f"call (resolves every ambiguous plane), or mark this array: "
        f"hawk.samples_first({name}) / hawk.samples_last({name})"
    )


def conflict_refusal(name: str, shape: tuple, width: int, axis: str) -> str:
    """The refusal for a marker its array's shape contradicts."""
    want = f"(L, {width})" if axis == "first" else f"({width}, L)"
    return (
        f"argument {name!r} is marked samples_{axis} (shape {want}) but its "
        f"shape is {shape}; mark it with hawk.samples_"
        f"{'last' if axis == 'first' else 'first'}({name}) or drop the marker"
    )


def layout_refusal(kind: str, name: str, shape: tuple, width: int, axis=None) -> str:
    """The refusal for a plane classified ``"ambiguous"``, ``"conflict"`` or
    ``"copy"``."""
    if kind == "ambiguous":
        return ambiguous_refusal(name, shape, width)
    if kind == "conflict":
        return conflict_refusal(name, shape, width, axis)
    return sample_major_refusal(name, shape, width)
