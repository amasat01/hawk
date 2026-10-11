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
to each floating-point promoted parameter as a PER-SAMPLE plane
(``bar_<name>``, one value per sample), never a scatter into the cell, so the
caller sums it in a fixed sample order and the gradient is reproducible. The
plain kernel is never changed.
"""

from __future__ import annotations

from typing import Any

from ..ir import HawkError, Leaf
from ..ir.nodes import PARAM_CELL, At, RoleConst, is_param_cell
from ..ir.walk import substitute
from ..types import TensorType
from .tape import TAPE_ON, _reachable, _resink

#: The promoted kernel's name suffix.
SUFFIX = "_params"

_INDEX = TensorType((), "i32")


def param_cell(name: str, ttype: TensorType) -> At:
    """The read of parameter ``name``: cell 0 of a one-cell ``lookup`` plane."""
    return At(Leaf("table_read", "lookup", name, ttype),
              RoleConst(0, PARAM_CELL, _INDEX), ttype)


def param_planes(kernel: Any, names: Any = None) -> Any:
    """``kernel`` with its ``uniform`` parameters read from one-cell planes.

    ``names`` selects which uniforms (default: every one the body reads, the
    taping flag aside); an unknown name, or a plane of that name that is not
    the uniform, is refused. The result is named ``<name>_params``. A default
    automatic kernel is promoted through its single step and rebuilt."""
    from ..trace.kernel import Kernel
    from ..trace.steps import default_steps

    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    nodes = _reachable(step.sinks)
    if any(getattr(n, "kind", None) == "loop" for n in nodes):
        raise HawkError(
            f"hawk.diff.param_planes({step.name!r}): promoting a uniform through "
            "a loop in the step body is not built")
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
    cells = {n: param_cell(n, next(l.ttype for l in leaves if l.name == n))
             for n in wanted}
    mapping = {id(l): cells[l.name] for l in leaves if l.name in cells}
    sinks = swap(step.sinks, mapping)
    out = Kernel(step.name + SUFFIX, sinks, step.planes, kind=step.kind)
    out.time, out.time_step = step.time, step.time_step
    return default_steps(out) if step is not kernel else out


def demote(sinks: Any) -> tuple:
    """``sinks`` with every parameter cell read as a per-sample plane of the
    same name — the form whose reverse yields a per-sample gradient."""
    nodes = _reachable(sinks)
    mapping = {id(n): Leaf("vocab_read", "per_sample", n.plane.name, n.ttype)
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
    values = substitute(tuple(s.value for s in sinks), mapping)
    return tuple(_resink(s, v) for s, v in zip(sinks, values))
