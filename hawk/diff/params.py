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
values each, :class:`hawk.ir.nodes.SampleSum`), never a scatter into the cell,
so the gradient is the same bits on every launch. The plain kernel is never
changed.

A uniform read inside a ``for`` is promoted the same way: the loop's body reads
it as a loop-invariant boundary value, so the cell is read once before the loop.
"""

from __future__ import annotations

from typing import Any

from ..ir import HawkError, Leaf
from ..ir.nodes import PARAM_CELL, At, RoleConst, SampleSum, is_param_cell
from ..ir.walk import substitute
from ..types import TensorType
from .tape import TAPE_ON, _reachable, _resink

#: The promoted kernel's name suffix, and the one a per-cell kernel adds to it.
SUFFIX = "_params"
CELLS_SUFFIX = "_cells"

#: The per-sample ``i32`` plane that holds each sample's parameter cell.
CELL_PLANE = "param_cell"

_INDEX = TensorType((), "i32")


def param_cell(name: str, ttype: TensorType, per_cell: bool = False) -> At:
    """The read of parameter ``name``: cell 0 of a one-cell ``lookup`` plane,
    or, with ``per_cell``, the cell the sample's ``param_cell`` plane names."""
    plane = Leaf("table_read", "lookup", name, ttype)
    if per_cell:
        return At(plane, Leaf("vocab_read", "per_sample", CELL_PLANE, _INDEX), ttype)
    return At(plane, RoleConst(0, PARAM_CELL, _INDEX), ttype)


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
    from ..trace.steps import default_steps

    from ..trace.steps import default_step

    step = default_step(kernel)
    step = getattr(step, "one_step", None) or step
    nodes = _reachable(step.sinks)
    leaves = [n for n in nodes if isinstance(n, Leaf) and n.role == "uniform"
              and n.name != TAPE_ON]
    have = {n.name for n in leaves}
    wanted = sorted(have) if names is None else list(names)
    unknown = [n for n in wanted if n not in have]
    if unknown:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): {unknown} are not uniform "
            f"parameters this kernel reads; its uniforms are {sorted(have)}")
    clash = [n for role, n in step.walk.arg_spec if n in wanted and role != "uniform"]
    if clash:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): a plane named {clash} "
            "already exists; the parameter plane takes the uniform's own name, "
            "so rename one of them")
    if not wanted:
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): the kernel reads no uniform "
            "parameter, so there is nothing to promote")
    if per_cell:
        if any(isinstance(n, SampleSum) for n in nodes):
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): the kernel "
                "already reduces over samples into cells of its own, and a "
                "parameter cell index would be a second use of the word. Promote "
                "its parameters with per_cell=False, or move the reduction out")
        if step.jump is not None:
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): the kernel "
                "has a `jump=` plan, whose switching comparisons are written per "
                "launch, not per parameter cell. Promote with per_cell=False")
        if CELL_PLANE in {n for _, n in step.walk.arg_spec}:
            raise HawkError(
                f"hawk.diff.param_planes({step.name!r}, per_cell=True): a plane "
                f"named {CELL_PLANE!r} already exists; the cell index takes that "
                "name, so rename the plane")
    cells = {n: param_cell(n, next(l.ttype for l in leaves if l.name == n), per_cell)
             for n in wanted}
    mapping = {id(l): cells[l.name] for l in leaves if l.name in cells}
    sinks = swap(step.sinks, mapping)
    out = Kernel(step.name + SUFFIX + (CELLS_SUFFIX if per_cell else ""), sinks,
                 step.planes, kind=step.kind)
    out.time, out.time_step = step.time, step.time_step
    out.jump = step.jump
    return default_steps(out) if step is not kernel else out


def demote(sinks: Any) -> tuple:
    """``sinks`` with every parameter cell read as a per-sample plane of the
    same name — the form whose reverse yields a per-sample gradient."""
    nodes = _reachable(sinks)
    mapping = {id(n): Leaf("vocab_read", "per_sample", n.plane.name, n.ttype,
                           sorted=n.plane.sorted)
               for n in nodes if is_param_cell(n)}
    return tuple(sinks) if not mapping else swap(sinks, mapping)


def promote(sinks: Any, names: Any) -> tuple:
    """``sinks`` with every per-sample read of a ``names`` plane turned back
    into a parameter-cell read."""
    nodes = _reachable(sinks)
    mapping = {id(n): param_cell(n.name, n.ttype) for n in nodes
               if isinstance(n, Leaf) and n.role == "per_sample" and n.name in names}
    return swap(sinks, mapping)


def swap(sinks: Any, mapping: dict) -> tuple:
    """``sinks`` with the nodes in ``mapping`` (by ``id``) replaced."""
    if mapping and any(not hasattr(s, "value") for s in sinks):
        raise HawkError(
            "a kernel that reads a parameter cell cannot also be differentiated "
            "through a `for` that commits a gradient per iteration (a plane "
            "read at a runtime index inside the loop): the per-sample "
            "accumulate that feeds the per-32-sample sum of the parameter is "
            "not built. Differentiate the kernel with the parameter passed as a "
            "plain `Param` (not through param_planes), or read the plane outside "
            "the loop")
    values = substitute(tuple(s.value for s in sinks), mapping)
    return tuple(_resink(s, v) for s, v in zip(sinks, values))
