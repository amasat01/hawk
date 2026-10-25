# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The launch shape a kernel declares (``hawk.Grid[...]``), read at the host bind door.

The sidecar's ``launch.lane_space`` is ``{"axes": [{"label", "extent"}, ...],
"group"}``: the FIRST extent may be ``None`` (runtime-sized), the later ones are
static, and the LAST axis varies fastest. A kernel that declares no ``Grid`` has
one unlabelled runtime axis, and none of this applies to it.

A ``Grid`` kernel takes its runtime extent BY LABEL (``k(x, channel=64)``) or from
a lane-shaped per-sample plane: ``head + lane shape``, which C-contiguous IS the
flat plane, so it binds as a view through the buffer protocol (no copy, no
numpy). ``head`` is the component axis of a ``Vector[w]`` plane; the rank and the
declared head tell the lane form from the flat ``(w, N)`` one. ``n_samples=`` and
``n=`` are the flat kernels' totals and are refused here, naming the label.
"""

from __future__ import annotations

from math import prod

from .ir import HawkError


def lane_axes(sidecar: dict) -> list | None:
    """``[(label, extent|None), ...]`` of a kernel that declares a ``Grid``, else ``None``."""
    space = (sidecar.get("launch") or {}).get("lane_space")
    if not space or space["axes"][0]["label"] is None:
        return None
    return [(ax["label"], ax["extent"]) for ax in space["axes"]]


def spelling(axes) -> str:
    return "Grid[" + ", ".join(f'"{a}"' if e is None else f'"{a}":{e}' for a, e in axes) + "]"


def output_shape(sidecar: dict, name: str, width: int, n: int) -> tuple:
    """The shape to allocate for the output plane ``name`` of ``n`` lanes and
    per-lane ``width``: lane-shaped for a ``Grid`` kernel (``(width, *lane
    shape)``, the component axis only when ``width > 1``), the named-axis
    ``(*payload, *axes)`` for a sink declared with axes (a reduction's output:
    the payload, then the kept lane axes), else the flat ``(width, n)`` / ``(n,)``."""
    declared = (sidecar.get("plane_axes") or {}).get(name)
    axes = lane_axes(sidecar)
    if declared is not None:
        # a kept runtime axis (a reduction's output) is as long as the lane axis
        runtime = n // prod(e for _a, e in axes[1:]) if axes else n
        return tuple(declared["payload"]) + tuple(
            runtime if ax["extent"] is None else ax["extent"] for ax in declared["axes"])
    head = (width,) if width > 1 else ()
    if axes is None:
        return head + (n,)
    static = tuple(e for _a, e in axes[1:])
    return head + (n // prod(static),) + static


def check_total(sidecar: dict, n: int) -> None:
    """Refuse a flat lane count ``n`` that is not a whole number of rows of a
    ``Grid``'s static axes."""
    axes = lane_axes(sidecar)
    if axes is None:
        return
    row = prod(e for _a, e in axes[1:])
    if n % row if axes[0][1] is None else n != row * axes[0][1]:
        raise HawkError(
            f"hawk.runtime.run: {n} lanes do not fit {spelling(axes)} "
            + (f"(a whole number of rows of {row})" if axes[0][1] is None
               else f"({row * axes[0][1]} lanes)"))


def bind(sidecar: dict, arg_spec, arrays: dict, n_samples, per_sample_roles) -> tuple:
    """``(arrays, n_samples)`` for a ``Grid`` kernel: the extent by label consumed,
    lane-shaped planes flattened, the total lane count settled. A kernel with no
    ``Grid`` is returned as it came."""
    axes = lane_axes(sidecar)
    if axes is None:
        return arrays, n_samples
    first, runtime = axes[0][0], axes[0][1] is None
    static = tuple(e for _a, e in axes[1:])
    slots = {name for _r, name in arg_spec}
    out = dict(arrays)
    where = f"hawk.runtime.run: {spelling(axes)}"
    if n_samples is not None or ("n" in out and "n" not in slots):
        out.pop("n", None)
        raise HawkError(
            f"{where} takes its extent by label, not as a total: pass "
            + (f"{first}=<extent of {first!r}>" if runtime else "no extent (every axis is static)")
            + " instead of n=/n_samples=")
    extent = None
    for label, fixed in axes:
        if label in out and label not in slots:
            if not (runtime and label == first):
                raise HawkError(
                    f"{where}: the axis {label!r} is "
                    + (f"static ({fixed}); drop {label}=" if fixed is not None
                       else "the runtime one"))
            extent = out.pop(label)
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
                raise HawkError(f"{where}: {first}= is the extent of the first axis, "
                                f"a positive int; got {extent!r}")
    shapes = sidecar.get("arg_shapes", {})
    seen = None
    for role, name in arg_spec:
        if role not in per_sample_roles or role == "mat_in" or name not in out or len(axes) < 2:
            continue
        try:
            mv = memoryview(out[name])
        except TypeError:
            continue                                  # a marked array: the layout rules read it
        width = int(shapes.get(name) or 1)
        head = (width,) if width > 1 else ()
        got = tuple(mv.shape)
        if len(got) != len(head) + len(axes):
            continue                                  # a flat plane: the layout rules read it
        dims = got[len(head):]
        declared = tuple(e for _a, e in axes)
        if got[:len(head)] != head or (dims[1:] != declared[1:] if runtime else dims != declared):
            want = "(" + ", ".join(map(str, head + ((first, *static) if runtime else declared))) + ")"
            raise HawkError(
                f"{where}: argument {name!r} has shape {got}, which contradicts the "
                f"declaration; expected {want} (the static extents are part of the kernel"
                + (f", only {first!r} is the caller's)" if runtime else ")"))
        if not mv.c_contiguous:
            raise HawkError(
                f"{where}: lane-shaped argument {name!r} (shape {got}) must be C-contiguous "
                f"to bind without a copy — pass np.ascontiguousarray({name})")
        if seen is not None and dims != seen:
            raise HawkError(f"{where}: lane-shaped arguments disagree on the lane shape: "
                            f"{seen} vs {dims} ({name!r})")
        seen = dims
        flat = mv.cast("B").cast(mv.format, shape=list(head) + [prod(dims)]) if head \
            else mv.cast("B").cast(mv.format)
        out[name] = flat
    total = None if seen is None else prod(seen)
    if extent is not None:
        want = extent * prod(static)
        if total is not None and total != want:
            raise HawkError(f"{where}: {first}={extent} gives {want} lanes, but the "
                            f"lane-shaped planes bound hold {total}")
        total = want
    elif total is None and not runtime:
        total = prod(static)                          # every axis static: no extent to pass
    elif total is None and not any(r in per_sample_roles and nm in out for r, nm in arg_spec):
        raise HawkError(
            f"{where} has a runtime first axis {first!r} and this launch binds no "
            f"per-sample plane to read it from; pass {first}=<extent of {first!r}> "
            "(a Table's or buffer's length is never the lane count)")
    return out, total
