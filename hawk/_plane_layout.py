# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How a plane's layout is classified (component-major, free sample-major
view, single sample), and the refusal texts for the rest."""

from __future__ import annotations

# -- Sample-major planes. A per-sample plane of declared width w >= 2 is
# component-major, (w, L); most array code stores vectors sample-major,
# (L, w). HAWK binds by ADDRESS and never copies, so it accepts a
# sample-major plane only when its transpose is C-contiguous (its bytes
# ARE the native (w, L) layout); any other is refused with the fix named,
# or copied by eagle's doors under an `eagle.LayoutWarning`. --------------- #

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


def plane_layout(mv: memoryview, width: int) -> str:
    """How a bound plane of declared per-sample ``width`` is laid out.
    ``"single"``: one sample. ``"native"``: anything not sample-major 2-D
    (the ambiguous ``(w, w)`` included, taken as native, never guessed).
    ``"view"``: a sample-major plane whose transpose is C-contiguous,
    bound zero-copy. ``"copy"``: one only a copy could bind, refused."""
    shape = tuple(mv.shape)
    if _is_single(shape, width):
        return "single"
    if width <= 1 or len(shape) != 2 or shape[0] == width or shape[1] != width:
        return "native"
    return "view" if mv.f_contiguous else "copy"


def _native_extent(mv: memoryview, width: int) -> int:
    """The trailing extent the plane has in its native ``(w, L)`` layout."""
    layout = plane_layout(mv, width)
    if layout == "single":
        return 1
    shape = tuple(mv.shape)
    if not shape:
        return 0
    return shape[0] if layout != "native" else shape[-1]


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
        f"(vector); or tile it once: {tile}"
    )


def sample_major_refusal(name: str, shape: tuple, width: int) -> str:
    """The refusal for a sample-major plane only a copy could bind."""
    return (
        f"argument {name!r}: given sample-major, shape {shape}, but this "
        f"kernel's plane is component-major ({width}, L). hawk.runtime binds "
        f"by address and never copies, so it accepts a sample-major plane only "
        f"as a zero-copy transposed view (the .T of a contiguous ({width}, L) "
        f"array). Pass np.ascontiguousarray({name}.T), allocate it as "
        f"({width}, L), or launch through eagle, which copies it with an "
        f"eagle.LayoutWarning"
    )
