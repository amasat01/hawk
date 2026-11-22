# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.param_planes(kernel)`` — a kernel whose ``uniform`` parameters
are read from one-cell parameter planes.

A ``uniform`` is a by-value argument: changing it between two launches of a
recorded graph means re-capturing the launch. The same value held in a
one-cell ``lookup`` plane is changed with a memcpy, and the capture stays
valid. This is the same promotion the time step gets in
:func:`hawk.diff.tape`: one traced body, no second compile of the same text.

The plane keeps the uniform's own name (``k`` stays ``k``), so ``rebind(k=...)``
reads naturally. The sidecar lists the promoted planes under ``param_planes``;
a launching runtime binds each to a one-element array.

:func:`hawk.diff.vjp` of the promoted kernel gives the gradient with respect
to each floating-point promoted parameter as one sum per 32 samples
(``bar_<name>`` plus its compensation ``bar_<name>_comp``, ``ceil(n/32)``
values each, a ``SampleSum`` node), never a scatter into the cell,
so the gradient is the same bits on every launch. The plain kernel is never
changed.

A uniform read inside a ``for`` is promoted the same way: the loop's body reads
it as a loop-invariant boundary value, so the cell is read once before the loop.
"""

from __future__ import annotations

from typing import Any

from .. import _ir
from ..errors import HawkError
from .tape import TAPE_ON

#: The per-sample plane that names each sample's parameter cell (read by the framework bridges).
CELL_PLANE = _ir.CELL_PLANE

#: The promoted kernel's name suffix, and the one a per-cell kernel adds to it.
SUFFIX = "_params"
CELLS_SUFFIX = "_cells"


def param_planes(kernel: Any, names: Any = None, per_cell: bool = False) -> Any:
    """``kernel`` with its ``uniform`` parameters read from one-cell planes.

    ``names`` selects which uniforms (default: every one the body reads, the
    taping flag aside); an unknown name, or a plane of that name that is not
    the uniform, is refused. The result is named ``<name>_params``. A default
    automatic kernel is promoted through its single step and rebuilt.

    With ``per_cell=True`` a sample reads cell ``param_cell[i]`` of each plane
    (an ``i32`` per-sample plane) instead of cell 0, so one launch can run many
    parameter values; the result is named ``<name>_params_cells``, and ``vjp``
    gives the gradient of each cell, summed over the samples that read it. A
    kernel with reductions of its own, or a ``jump=`` plan, is refused."""
    from ..trace.kernel import Kernel
    from ..trace.steps import default_step, default_steps

    step = default_step(kernel)
    step = getattr(step, "one_step", None) or step
    nodes = _ir.post_order(step.sinks)
    leaves = [
        n
        for n in nodes
        if _ir.is_a(n, "leaf")
        and n.field_text("role") == "uniform"
        and n.field_text("name") != TAPE_ON
    ]
    by_name = {leaf.field_text("name"): leaf for leaf in leaves}
    have = set(by_name)
    wanted = sorted(have) if names is None else list(names)
    unknown = [n for n in wanted if n not in have]
    if unknown:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): {unknown} are not uniform "
            f"parameters this kernel reads; its uniforms are {sorted(have)}"
        )
    clash = [n for role, n in _ir.arg_spec(step.walk) if n in wanted and role != "uniform"]
    if clash:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): a plane named {clash} "
            "already exists; the parameter plane takes the uniform's own name, "
            "so rename one of them"
        )
    if not wanted:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): the kernel reads no uniform "
            "parameter, so there is nothing to promote"
        )
    if per_cell:
        if any(_ir.is_a(n, "sample_sum") for n in nodes):
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): the kernel "
                "already reduces over samples into cells of its own, and a "
                "parameter cell index would be a second use of the word. Promote "
                "its parameters with per_cell=False, or move the reduction out"
            )
        if step.jump is not None:
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): the kernel "
                "has a `jump=` plan, whose switching comparisons are written per "
                "launch, not per parameter cell. Promote with per_cell=False"
            )
        if CELL_PLANE in {n for _, n in _ir.arg_spec(step.walk)}:
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): a plane "
                f"named {CELL_PLANE!r} already exists; the cell index takes that "
                "name, so rename the plane"
            )
    cells = {
        n: _ir.param_cell(by_name[n].ctx, n, _ir.type_of(by_name[n]), per_cell) for n in wanted
    }
    mapping = {leaf: cells[name] for name, leaf in by_name.items() if name in cells}
    sinks = _ir.swap(step.sinks, mapping)
    out = Kernel(
        step.name + SUFFIX + (CELLS_SUFFIX if per_cell else ""), sinks, step.planes, kind=step.kind
    )
    out.time, out.time_step = step.time, step.time_step
    out.jump = step.jump
    return default_steps(out) if step is not kernel else out


def _per_sample_read(cell: Any) -> Any:
    """The per-sample read of the plane a parameter-cell read ``cell`` reads."""
    plane = cell.operands[0]
    return _ir.leaf(
        cell.ctx,
        "vocab_read",
        "plane_in",
        plane.field_text("name"),
        _ir.type_of(cell),
        sorted=plane.field_flag("sorted"),
    )


def demote(sinks: Any) -> tuple:
    """``sinks`` with every parameter cell read as a per-sample plane of the
    same name — the form whose reverse yields a per-sample gradient."""
    nodes = _ir.post_order(sinks)
    mapping = {n: _per_sample_read(n) for n in nodes if _ir.is_param_cell(n)}
    return tuple(sinks) if not mapping else _ir.swap(sinks, mapping)


def promote(sinks: Any, names: Any) -> tuple:
    """``sinks`` with every per-sample read of a ``names`` plane turned back
    into a parameter-cell read."""
    nodes = _ir.post_order(sinks)
    mapping = {
        n: _ir.param_cell(n.ctx, n.field_text("name"), _ir.type_of(n))
        for n in nodes
        if _ir.is_a(n, "leaf")
        and n.field_text("role") == "plane_in"
        and n.field_text("name") in names
    }
    try:
        return _ir.swap(sinks, mapping)
    except HawkError as refusal:
        if getattr(refusal, "code", None) != "derived_loop_rewrite":
            raise
        raise HawkError(
            "a kernel that reads a parameter cell cannot also be differentiated through a "
            "`for` that commits a gradient per iteration (a plane read at a runtime index "
            "inside the loop): the per-sample accumulate that feeds the per-32-sample sum "
            "of the parameter is not built. Differentiate the kernel with the parameter "
            "passed as a plain `Param` (not through param_planes), or read the plane outside "
            "the loop"
        ) from refusal
