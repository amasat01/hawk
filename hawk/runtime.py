# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Loading a built host kernel and running it over the runtime door of the host OSPREY: the one-call :func:`run`, and the
:class:`BoundPlan` that :func:`hawk.plan` makes, which both end in the same door calls.

hawk's part is what no target decides: it resolves a plane's name to its slot from the kernel's sidecar,
classifies each array's layout, infers and checks the sample count and refuses a wrong argument by name, all
the failures of one call together. Everything that runs is the library's, reached through one
:class:`_osprey.Door` (:meth:`hawk.ospreys.Osprey.runtime_door`): the unit is loaded, each array is wrapped as a plane,
a plan is made over the range, the planes are bound and the plan is launched. The scratch planes a kernel
owns (the finished counter, the step word, the compensation companions, the rows of a sum over samples, the
planes between the stages of a split kernel) are made and folded by the door, so no array library is needed
here.

Arrays are NumPy arrays or anything else that exports DLPack and the buffer protocol; the plane is the caller's
memory, written in place. A plain buffer (``array.array``, ``bytes``) exports no DLPack and is refused by name.
Every bound argument is validated against the sidecar's declaration before the door is called
(:meth:`HostKernel.arguments`, which :func:`run` and :meth:`hawk.plans.Plan.bind` both call): dtype is exact and fixed by role (every integer-valued role rides a 64-bit wire, and
``Terminated`` is always ``bool``), a per-sample plane holds exactly ``n_samples`` samples, every plane is
C-contiguous and a sink role's plane writable, except a sample-major plane whose transpose is C-contiguous,
which binds as that view (the rule is ``hawk._plane_layout``'s). A ``(w, w)`` plane reads both ways and is refused unless
``layout="samples_first"``/``"samples_last"`` (per call) or ``hawk.samples_first(x)``/``hawk.samples_last(x)``
(per array, which wins) says which axis holds the samples.

The shape a caller passes and the shape the door is handed, by role (``w`` the declared width, ``P`` the payload axes the unit
states for the slot, ``N`` the samples; the door binds by the unit's declared shape):

==============================  ==========================================  ==================================
role                            the caller passes                           the door is handed
==============================  ==========================================  ==================================
per-sample plane, scalar        ``(N,)``                                    ``(N,)`` as passed
per-sample plane, width ``w``   ``(w, N)``, or ``(N, w)`` whose transpose   ``(*P, N)``: ``(w, N)``; a matrix
(``per_sample``, ``mutable``,   is contiguous; ``layout=`` / markers        payload ``(R, C)`` gets
``out``, ``vec_in``,            settle ``N == w``                           ``(R, C, N)``; ``(N, w)`` as its
``mat_in``, ``terminated``)                                                 transpose view, no copy
one sample                      the head alone: ``()``, ``(w,)``            ``(*P, 1)``, a batch of one
a matrix read at lane indices   ``(R*C, m)`` compact                        ``(R, C, m)``
(``gather_mats``)
table (``lookup``)              any contiguous array; a shared parameter    as passed
                                in its declared shape
wide plane (``wide_in``,        any contiguous array; a flat ``(W*N,)``     as passed, never reshaped (the
``wide_out``)                   holds element ``(e, c)`` at ``e*W + c``     door's handle takes its trailing
                                                                            extent)
cell of a sum over samples      ``(columns,)``, one value per column        one ``(1,)`` view per column slot
(``cells``)
scatter target (``accum_out``)  ``(*payload, *axes)`` as declared           as passed
uniform                         a Python number or a 0-d array              its bytes at the wire dtype
==============================  ==========================================  ==================================

The library makes and the caller does not bind: the sample count, the step word, the finished counter, a dispatch group's
branch counters, compensation planes, a cell's rows and the planes between the stages of a split kernel.

Threading. The door's context admits one calling thread at a time: a second thread calling while another is
inside the door is refused by the door, with its reason.
"""

from __future__ import annotations

import json
import struct
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import _osprey

from . import _broadcast, _cells, _ir, _lane_bind, ospreys
from ._bind_checks import _PER_SAMPLE_ROLES, _check_bind, _n_from, narrow_index, refuse_index_width, widen_index
from ._plane_layout import LayoutChoice, describe, is_ndarray, samples_last, split_marks
from .errors import HawkError
from .targets import ENTRY_KINDS, host_target

__all__ = ["BoundPlan", "HostKernel", "check_scatter_status", "dispatch_costs", "load", "run", "steps_reverse_table"]


#: The struct format a uniform's value is packed with, by the wire dtype the sidecar states for it.
_UNIFORM_FORMAT = MappingProxyType({"float64": "<d", "float32": "<f", "int64": "<q", "int32": "<i", "uint32": "<I", "bool": "<?"})


def through_door(call, *args, **kwargs):
    """Return ``call(*args, **kwargs)``; a refusal of the door is raised as :class:`~hawk.errors.HawkError`, its reason as given."""
    try:
        return call(*args, **kwargs)
    except (ValueError, _osprey.Fault) as refusal:
        raise HawkError(str(refusal)) from refusal



class _Arguments(NamedTuple):
    """One call's arguments, checked and ready to wrap: what :meth:`HostKernel.arguments` returns."""

    n_samples: int
    #: ``{name: the caller's value}``, the outputs the call made included; a marked array stays marked.
    given: dict
    #: ``{slot name: the array the slot's plane wraps}``.
    planes: dict
    #: ``{name: the wire bytes of a uniform}``.
    uniforms: dict
    #: ``{name: the caller's writable counter array}``, filled after each launch.
    counters: dict
    #: ``{label: extent}`` of the runtime axes the caller stated by label; a rebind states them again.
    extents: dict


class HostKernel:
    """One built host kernel: its sidecar and its unit, loaded through the host OSPREY's door.

    Construction is per kernel; it holds no bound planes. :func:`hawk.plan` makes a plan over it, :func:`run` runs it once.
    """

    _FINISHED = _ir.FINISHED_PLANE

    def __init__(self, so_path, sidecar: dict) -> None:
        """Load the unit the sidecar names from the library at ``so_path``, with the units of its stages."""
        self.path = Path(so_path)
        self.sidecar = sidecar
        self.arg_spec = tuple(tuple(pair) for pair in sidecar["arg_spec"])
        self.scalar_type = sidecar.get("scalar_type", "float64")
        if not sidecar.get("host_entry"):
            raise HawkError(
                f"{self.path}: its sidecar declares no 'host_entry', so it carries "
                "no host target to run"
            )
        door = ospreys.serving(host_target().id).runtime_door()
        self._door = door
        source = sidecar.get("source")
        if source is None:
            #: the unit a run loads and binds, and the specification of the planes it takes
            members = self._group_members(sidecar)
            self.unit = _load_unit(door, self.path, sidecar,
                                   deps=[_load_unit(door, self.path, member) for member in members])
            self._stages = []
            slots = list(sidecar["slots"])
            for member in members:      # the planes of the group's other units follow the branch's own in the slot table
                slots += [name for name in member["slots"] if name not in slots]
            role_of = {}
            for pair in (*self.arg_spec, *(tuple(p) for member in members for p in member["arg_spec"])):
                role_of.setdefault(pair[1], pair[0])
            self._slots = slots
            self.bind_spec = tuple((role_of[name], name) for name in slots if name in role_of)
            self.declared = _declared_over([sidecar, *members]) if members else sidecar
        else:
            # a split kernel runs as the unit that lists its stages: the stage units are loaded first and named as its deps
            later = [load(self.path.parent, record["kernel"]) for record in sidecar["stages"][1:]]
            self._stages = later
            stage_units = [_load_unit(door, self.path, sidecar), *(stage.unit for stage in later)]
            self.unit = _load_unit(door, self.path, {"kernel": source["name"], "digest": source["digest"]}, deps=stage_units)
            role_of = {}
            for pair in (*self.arg_spec, *(p for stage in later for p in stage.arg_spec)):
                role_of.setdefault(pair[1], pair[0])
            self._slots = list(source["slots"])
            self.bind_spec = tuple((role_of[name], name) for name in source["slots"] if name in role_of)
            self.declared = _declared_over([sidecar, *(stage.sidecar for stage in later)])
        self._slot_named = {name: i for i, name in enumerate(self._slots)}
        #: ``{(role, name): slot}``: a slot is the index of the plane in the unit's slot table.
        self.slot_of = {pair: self._slot_named[pair[1]] for pair in self.bind_spec}
        self._made = _door_made_names(self.declared)
        self._one_step_word = None

    def _group_members(self, sidecar: dict) -> list:
        """The sidecars of the other units of the segmented dispatch group this unit is a branch of: its other branches and
        its selector, which the library plans together with it (the sidecar's ``group_units`` names them)."""
        members = []
        for entry in sidecar.get("dispatch", ()):
            group = entry.get("group_units") or ()
            if any(member["name"] == sidecar["kernel"] for member in group):
                members += [
                    json.loads((self.path.parent / f"{member['name']}.json").read_text())
                    for member in group
                    if member["name"] != sidecar["kernel"]
                ]
        return members

    # -- per call ----------------------------------------------------------
    def arguments(self, arrays: dict, n_samples: int | None = None, layout: str | None = None, *,
                  keep: bool, outputs: bool = False) -> _Arguments:
        """Check ``arrays`` against the sidecar and return what a plan wraps and binds.

        ``arrays`` maps a plane's name to an array and a ``uniform``'s to a Python number. Refuses naming every failing
        argument, not just the first. A kernel that declares a ``Grid`` takes its runtime extent by label, and ``n_samples``
        is the count every per-sample plane agrees on when it is not stated. The planes stay the caller's memory.
        ``keep``: they stay bound after the call, so an ``Index`` plane of another integer width than the kernel declares is
        refused (a copy would stay behind when the caller changes the column); a call that binds once converts it instead.
        ``outputs``: an output plane the caller left out is made, zeroed, in the shape and dtype the sidecar declares.
        An automatic kernel's ``fused_steps`` word, a finishing kernel's ``finished_count`` counter and the other planes the
        library owns are not bound.
        """
        sidecar = self.declared
        spec = self.bind_spec
        arrays = _shared_shapes_checked(sidecar, arrays)
        stated = _lane_bind.stated_extents(sidecar, arrays)
        arrays, lane_n = _lane_bind.bind(sidecar, spec, arrays, n_samples, _PER_SAMPLE_ROLES)
        plain, axes = split_marks(arrays, layout)
        choice = LayoutChoice(layout, axes)
        if lane_n is not None:
            n_samples = lane_n
        if n_samples is None:
            compact = sidecar.get("gather_mats") or {}
            n_samples = _n_from([(r, n) for r, n in spec if not (r == "mat_in" and n in compact)], plain,
                                sidecar.get("arg_shapes", {}), choice)
            _lane_bind.check_total(sidecar, n_samples)
        if outputs:
            arrays = {**arrays, **self._made_outputs(plain, n_samples)}
            plain, axes = split_marks(arrays, layout)
            choice = LayoutChoice(layout, axes)
        declared = _stage_names(self) if sidecar.get("stages") else {name for _, name in spec} | set(sidecar.get("cells") or ())
        _refuse_unbound_names(self, arrays, declared)
        return self._checked(arrays, n_samples, layout, keep=keep)._replace(extents=stated)

    def output_names(self) -> tuple:
        """The names of the output planes a caller sees, in slot order: the sinks the library does not make, a cell by its own
        name (a cell of a kernel in stages is the caller's only when it binds it)."""
        cells = self.declared.get("cells") or {}
        rows = {plane for cell, spec in cells.items() for plane in _ir.planes(cell, spec["columns"])}
        named = [name for role, name in self.bind_spec if role in _ir.SINK_ROLES and name not in self._made
                 and (name not in rows or name in cells)]
        return tuple(dict.fromkeys([*named, *(cell for cell in cells if not self.declared.get("stages"))]))

    def _made_outputs(self, arrays: dict, n_samples: int) -> dict:
        """The zeroed arrays of the outputs ``arrays`` leaves out, by name, each as long as the lanes unless the sidecar
        declares its axes. A wide plane, whose extent only the caller knows, is refused by name: the caller binds it."""
        import numpy as np

        sidecar = self.declared
        cells = sidecar.get("cells") or {}
        labels = _lane_bind.free_labels(sidecar)
        extents = {label: arrays[label] for label in labels if label in arrays}
        roles = {name: role for role, name in self.bind_spec}
        made = {}
        for name in self.output_names():
            if name in arrays:
                continue
            if name in cells:
                made[name] = np.zeros(cells[name]["columns"], dtype=self.scalar_type)
                continue
            if roles[name] == "wide_out":
                raise HawkError(f"kernel {sidecar.get('kernel', '?')!r}: argument {name!r}: the extent of a wide plane is the "
                                "caller's, so it is not made; bind it")
            width = int(sidecar.get("arg_shapes", {}).get(name) or 1)
            shape = _lane_bind.output_shape(sidecar, name, width, n_samples, extents)
            plane = np.zeros(shape, dtype=sidecar["arg_dtypes"][name])
            made[name] = samples_last(plane) if width > 1 and shape[0] == width else plane
        return made

    def _checked(self, arrays: dict, n_samples: int, layout, *, keep: bool) -> _Arguments:
        """Validate ``arrays`` and return the views and uniform values a plan binds.

        ``keep``: see :meth:`arguments`.
        """
        sidecar = self.declared
        given = arrays
        arrays, axes = split_marks(arrays, layout)
        choice = LayoutChoice(layout, axes)
        arrays = self._with_seed(arrays)
        arrays, counters = self._split_counters(arrays)
        if _ir.FUSED_STEPS_PLANE in self._slot_named and _ir.FUSED_STEPS_PLANE not in arrays:
            arrays = {**arrays, _ir.FUSED_STEPS_PLANE: self._one_step()}
        index_inputs = tuple(
            name
            for role, name in self.bind_spec
            if role in ("per_sample", "lookup", "wide_in") and sidecar.get("arg_dtypes", {}).get(name) == "int64"
        )
        if keep:
            refuse_index_width(index_inputs, sidecar.get("index_planes") or (), arrays, sidecar.get("kernel", ""))
        else:
            if index_inputs:
                arrays = widen_index(index_inputs, arrays)
            if sidecar.get("index_planes"):
                arrays = narrow_index(sidecar["index_planes"], arrays, sidecar.get("kernel", ""))
        arrays = self._with_cell_columns(arrays)
        spec = tuple(pair for pair in self.bind_spec if pair[1] in arrays or pair[1] not in self._made)
        _check_bind(sidecar, spec, arrays, n_samples, choice)
        planes, uniforms = {}, {}
        for role, name in spec:
            if name not in arrays or role == "nsamples":
                continue
            if role == "uniform":
                uniforms[name] = self._uniform_bytes(name, arrays[name])
            else:
                planes[name] = self._plane_view(role, name, arrays[name], choice)
        return _Arguments(n_samples, given, planes, uniforms, counters, {})


    def _one_step(self):
        """The one-cell step word of ``1`` an automatic-step kernel reads when the caller binds none: a launch
        outside a runner takes one step. Made once per kernel."""
        if self._one_step_word is None:
            import numpy as np

            self._one_step_word = np.ones(1, dtype=np.int64)
        return self._one_step_word

    def _split_counters(self, arrays: dict) -> tuple:
        """``(arrays without the counters, {name: array})``: the finished counter and a dispatch group's branch counters
        a caller binds are the library's to count, so they are not bound as planes; they are checked here (one writable
        array of the declared length) and filled from the plan's counters after the launch."""
        names = {self._FINISHED: 1}
        for entry in self.declared.get("dispatch", ()):
            if entry.get("mix") is not None:
                names[entry["mix"]] = int(entry["K"])
        given = {name: arrays[name] for name in names if name in arrays}
        errors = []
        for name, value in given.items():
            try:
                view = describe(value)
            except TypeError:
                view = None
            if view is None:
                errors.append(f"argument {name!r}: the library fills this counter after the launch; pass a writable array, "
                              f"numpy.zeros({names[name]}, dtype=numpy.uint32)")
            elif view.readonly:
                errors.append(f"argument {name!r}: the library fills this counter after the launch, so it must be "
                              f"writable, got a read-only array")
            elif tuple(view.shape) != (names[name],):
                errors.append(f"argument {name!r}: this counter has {names[name]} cell(s), got shape {tuple(view.shape)}")
        if errors:
            raise HawkError(f"kernel {self.declared.get('kernel', '?')!r}: " + "; ".join(errors))
        return {name: value for name, value in arrays.items() if name not in names}, given

    def _with_seed(self, arrays: dict) -> dict:
        """``arrays`` plus a ``jump_seed`` of ``0`` when the kernel reads one and the caller left it out (an
        alternate-path ``jump=`` derivative runs with seed 0 unless told otherwise)."""
        from .diff.jump import SEED

        if SEED in arrays or SEED not in self._slot_named:
            return arrays
        return {**arrays, SEED: 0}

    def _with_cell_columns(self, arrays: dict) -> dict:
        """``arrays`` with each cell the caller bound replaced by its columns, one one-cell view for each row plane.

        A kernel that reduces over samples names a cell (``(columns,)``, one value per column); the unit's slot
        for column ``k`` is the plane ``name`` or ``name_k``, one element.
        """
        cells = self.declared.get("cells") or {}
        if not cells:
            return arrays
        missing = [c for c in cells if c not in arrays]
        if missing and not self.declared.get("stages"):
            raise HawkError(
                f"{self.path.name}: this kernel reduces over samples into the cells "
                f"{sorted(cells)}; bind {missing} (one value per column), not its rows"
            )
        out = {name: value for name, value in arrays.items() if name not in cells}
        for cell, spec in cells.items():
            if cell not in arrays:
                continue
            if not is_ndarray(arrays[cell]):
                if spec["columns"] != 1 or tuple(describe(arrays[cell]).shape) != (1,):
                    raise HawkError(
                        f"kernel {self.declared.get('kernel', '?')!r}: the cell {cell!r} exports DLPack and is not a NumPy array, so "
                        f"hawk cannot slice it per column; pass shape ({spec['columns']},) as a one-cell array of one "
                        f"column, or wrap it with numpy.asarray({cell})")
                out[cell] = arrays[cell]
                continue
            column = arrays[cell].reshape(-1)
            for k, plane in enumerate(_ir.planes(cell, spec["columns"])):
                out[plane] = column[k : k + 1]
        return out

    def _uniform_bytes(self, name: str, value) -> bytes:
        """The wire bytes of the uniform ``name``: its value at the width the sidecar states."""
        wire = self.declared["arg_dtypes"][name]
        if wire == "bool":
            return struct.pack(_UNIFORM_FORMAT[wire], bool(value))
        kind = float if wire.startswith("float") else int
        return struct.pack(_UNIFORM_FORMAT[wire], kind(value))

    def _plane_view(self, role: str, name: str, value, choice):
        """The array ``value`` as the plane the unit's slot ``name`` takes (the table in the module docstring): the
        caller's memory, never a copy.

        A sample-major plane whose transpose is contiguous is bound as that transpose, one sample as a batch of one,
        and a plane of per-sample payload axes (a vector, a matrix) as ``(*payload, samples)``. Any other role is
        handed over as the caller passed it. An array that exports DLPack and is not a NumPy array (whether or not it also exports a buffer) is always handed over as
        passed: hawk has no array library to make a view of it with, so one that needs a view is refused by name.
        """
        if role not in _PER_SAMPLE_ROLES:
            return value
        sidecar = self.declared
        width = int(sidecar.get("arg_shapes", {}).get(name) or 1)
        payload = tuple((sidecar.get("slot_shapes") or {}).get(name) or ())
        view = describe(value)
        kind = choice.of(name, view, width)
        shape = tuple(view.shape)
        native = (*payload, shape[0] if kind == "view" else shape[-1]) if shape else ()
        if not is_ndarray(value):
            if kind != "native" or (payload and shape != native):
                want = (*payload, "N") if payload else ("N",)
                raise HawkError(
                    f"kernel {sidecar.get('kernel', '?')!r}: argument {name!r} exports DLPack and is not a NumPy array, so hawk can "
                    f"hand it over as it is and no other way, but this plane needs a view ({kind}, shape {shape}); "
                    f"pass it as {want}, or wrap it with numpy.asarray({name})")
            return value
        if kind == "single":
            return value.reshape(payload + (1,)) if payload else value.reshape((1,))
        if kind == "view":
            value = value.T
        if payload:
            value = value.reshape(payload + (value.shape[-1],))
        return value

    def refuse_partial_stages(self, partial: bool) -> None:
        """Refuse a launch over part of the range of a kernel that reads a value reduced over every sample."""
        if partial and self.sidecar.get("stages"):
            raise HawkError(
                f"{self.path.name}: this kernel reads a value reduced over "
                "every sample, so it runs over every sample (no base= or count=)")

    def _fill_counters(self, counted: dict, counters: dict) -> None:
        """Add what the last launch counted (``counted``, :meth:`BoundPlan.counters`) to the arrays the caller bound for the
        finished count and the branch counters: a cell holds its value before the launch plus what the launch counted."""
        if self._FINISHED in counters and counted["finished"] is not None:
            counters[self._FINISHED][0] += counted["finished"]
        offset = 0
        for entry in self.declared.get("dispatch", ()):
            if entry.get("mix") is None:
                continue
            width = int(entry["K"])
            if entry["mix"] in counters:
                for i, value in enumerate(counted["mix"][offset : offset + width]):
                    counters[entry["mix"]][i] += value
            offset += width

    def _own_cell_planes(self, planes: dict) -> list:
        """The one-cell planes of a split kernel's cells the caller did not bind (a cell only a later stage reads
        is the kernel's own): the unit's slots for them are bound to planes the door makes, zeroed."""
        if not self.sidecar.get("stages"):
            return []
        return [
            plane
            for cell, spec in self.sidecar["cells"].items()
            for plane in _ir.planes(cell, spec["columns"])
            if plane not in planes and plane in self._slot_named
        ]


def _release(door, plans: list, planes: list) -> None:
    """Free the plan handles in ``plans``, then the plane handles in ``planes`` (a plan reads its planes)."""
    for plan in plans:
        door.plan_free(plan)
    for plane in planes:
        door.plane_free(plane)
    plans.clear()
    planes.clear()


class BoundPlan:
    """A plan over one kernel with its planes bound: launch it as often as needed, rebind planes between launches.

    The planes are the caller's memory, read and written in place; a launch moves no data. The door's handles live until
    :meth:`close` or until this object is collected.
    """

    def __init__(self, kernel: HostKernel, arguments: _Arguments, structure: int, ranges: dict, *,
                 n_samples: int | None = None, layout: str | None = None) -> None:
        """Make the library's plan over ``arguments.n_samples`` samples under ``structure`` (a door structure code) and
        partitioned by ``ranges`` (the door's ``part_base`` / ``part_count`` or ``split_evenly``, else none: the whole
        range), then bind the planes. ``n_samples`` and ``layout`` are what the caller stated, kept for :meth:`rebind`."""
        self._kernel = kernel
        self._door = kernel._door
        self._stated = (n_samples, layout)
        self._n = arguments.n_samples
        #: the handles the door made, kept in lists the finalizer frees without holding this object
        self._plans: list = []
        self._planes: list = []
        self._finalizer = weakref.finalize(self, _release, self._door, self._plans, self._planes)
        try:
            self._plan = through_door(self._door.plan_new, kernel.unit, structure, _osprey.EG_REDUCE_COMPENSATED, self._n,
                                       **ranges)
            self._plans.append(self._plan)
            self._wire(arguments)
        except BaseException:
            self.close()
            raise

    def _wire(self, arguments: _Arguments) -> None:
        """Wrap ``arguments``' planes and bind them with its uniforms, replacing the planes of the earlier bind."""
        door, kernel = self._door, self._kernel
        wrapped = {}
        try:
            for name, view in arguments.planes.items():
                wrapped[name] = through_door(door.plane_wrap, view)
            for name in kernel._own_cell_planes(arguments.planes):
                wrapped[name] = through_door(door.plane_new, [1], 2, 32 if kernel.scalar_type == "float32" else 64,
                                              flags=_osprey.EG_PLANE_ZERO)
            through_door(
                door.bind,
                self._plan,
                [kernel._slot_named[name] for name in wrapped],
                list(wrapped.values()),
                [kernel._slot_named[name] for name in arguments.uniforms],
                list(arguments.uniforms.values()),
            )
        except BaseException:
            for plane in wrapped.values():
                door.plane_free(plane)
            raise
        replaced = list(self._planes)
        self._planes[:] = wrapped.values()
        for plane in replaced:
            door.plane_free(plane)
        self._arguments = arguments

    @property
    def planes(self) -> MappingProxyType:
        """The arrays bound, by name, as the caller gave them (a plane made for an output left out included), read-only."""
        return MappingProxyType({name: getattr(value, "array", value) for name, value in self._arguments.given.items()})

    def launch(self) -> int:
        """Run the plan once over the bound planes and return the host tiles the library ran. A counter array the caller
        bound (``finished_count``, a dispatch group's branch counters) is filled from the plan's counters."""
        if not self._finalizer.alive:
            raise HawkError(f"{self._kernel.path.name}: this plan is closed; bind a new one")
        tiles = through_door(self._door.launch, self._plan)
        if self._arguments.counters:
            self._kernel._fill_counters(self.counters(), self._arguments.counters)
        return tiles

    def counters(self) -> dict:
        """The library's counters of the plan: ``finished`` and ``steps`` (``None`` when the kernel has none) and ``mix``.

        A counter array the caller bound holds its value before a launch plus what that launch counted."""
        return through_door(self._door.plan_counters, self._plan)

    def rebind(self, **planes) -> BoundPlan:
        """Replace the named planes (and uniforms) with new values, every other bound value staying, and return this plan.
        An extent stated by label at bind (``e=``) stays too; a rebind may state it again, and a value that contradicts a bound
        plane is refused by name.

        The same checks as :meth:`hawk.plans.Plan.bind` run over the whole set; the sample count cannot change.
        """
        n_samples, layout = self._stated
        for name, value in planes.items():
            _broadcast.guard_rebind(self._kernel.declared, name, self._arguments.given.get(name), value)
        held = self._arguments
        arguments = self._kernel.arguments({**held.given, **held.extents, **planes}, n_samples, layout, keep=True)
        if arguments.n_samples != self._n:
            raise HawkError(f"{self._kernel.path.name}: rebind cannot change the sample count, bound for {self._n}, "
                            f"got {arguments.n_samples}; bind a new plan")
        self._wire(arguments)
        return self

    def close(self) -> None:
        """Free the library's plan and planes at once; a closed plan cannot launch. Closing twice is harmless."""
        self._finalizer()


def _load_unit(door, path: Path, sidecar: dict, deps=()):
    """The unit ``sidecar`` names (``kernel``, ``digest``) from the library at ``path``; ``deps`` are the loaded units it runs."""
    return through_door(door.unit_load, str(path), sidecar["kernel"], bytes.fromhex(sidecar["digest"]), deps=list(deps))


#: The declarations of a split kernel's stages that are dicts keyed by plane name, lists of records named by plane, and lists of names.
_DICT_DECLARATIONS = ("arg_dtypes", "arg_shapes", "arg_widths", "slot_shapes", "mat_shapes", "vec_widths", "param_shapes",
                      "plane_axes", "gather_mats", "adjoint_like", "companions")
_RECORD_DECLARATIONS = ("mutables", "params")
_NAME_DECLARATIONS = ("index_planes", "sorted_inputs", "unused_planes")


def _declared_over(sidecars: list) -> dict:
    """The first sidecar with the plane declarations of every stage merged in: a split kernel is bound as one, by the
    planes of all its stages. Where stages state the same plane, the first stage's word stands."""
    merged = dict(sidecars[0])
    for key in _DICT_DECLARATIONS:
        merged[key] = {name: value for sidecar in reversed(sidecars) for name, value in (sidecar.get(key) or {}).items()}
        merged[key].update(sidecars[0].get(key) or {})
    for key in _RECORD_DECLARATIONS:
        seen = {}
        for sidecar in sidecars:
            for record in sidecar.get(key) or ():
                seen.setdefault(record["name"], record)
        merged[key] = list(seen.values())
    for key in _NAME_DECLARATIONS:
        merged[key] = list(dict.fromkeys(name for sidecar in sidecars for name in sidecar.get(key) or ()))
    return merged


def _door_made_names(sidecar: dict) -> set:
    """The names of the slots the library makes and the caller need not bind: the finished
    counter, a dispatch group's branch counters, the compensation companions, the scratch planes between the
    stages, the compensation of a cell's rows and, in a split kernel, the cells themselves (the library keeps the
    ones the caller does not read)."""
    names = {_ir.FINISHED_PLANE}
    names |= {entry["mix"] for entry in sidecar.get("dispatch", ()) if entry.get("mix") is not None}
    for entry in sidecar.get("dispatch", ()):
        if entry["strategy"] == "segmented":
            names |= {entry["class"], entry["stale"], entry["offsets"]}   # the group's sorted-sample planes
    names |= set(sidecar.get("companions") or ()) | set(sidecar.get("team_planes") or ()) | set(sidecar.get("keyed") or ())
    for cell, spec in (sidecar.get("cells") or {}).items():
        planes = _ir.planes(cell, spec["columns"])
        if sidecar.get("stages"):
            names |= set(planes)
        if spec["op"] in ("sum", "mean"):
            names |= {_cells.comp_name(plane) for plane in planes}
    return names


def load(directory, kernel: str, sidecar: dict | None = None) -> HostKernel:
    """Load ``<directory>/<kernel>.so``, reading its sidecar JSON if omitted."""
    directory = Path(directory)
    if sidecar is None:
        sidecar = json.loads((directory / f"{kernel}.json").read_text())
    return HostKernel(directory / f"{kernel}.so", sidecar)


def steps_reverse_table(sidecar: dict, device: bool) -> dict:
    """The measured rows of ``steps_reverse="auto"`` for the host or the device, as the build stored them in the sidecar."""
    table = sidecar.get("steps_reverse_table") or {}
    device_kind, host_kind = ENTRY_KINDS
    return dict(table.get(device_kind if device else host_kind) or {})


def dispatch_costs(sidecar: dict) -> dict:
    """The dispatch-cost record of the sidecar, as the build stored it (empty when the answer stated none)."""
    return dict(sidecar.get("dispatch_costs") or {})


def check_scatter_status(kernel: str, team: dict, status, cells: int) -> None:
    """Refuse, by name, a run whose ordered scatter met a key outside ``[0, cells)`` that was not the skip key -1.

    ``kernel`` names the kernel in the refusal, ``team`` is the sidecar's team record of the scatter (its ``of`` names the
    plane the scatter adds into), ``status`` the plane ``team["status"]`` after the run (flags from element 1) and ``cells``
    the extent of the plane ``team["of"]``. Raises :class:`~hawk.errors.HawkError`; returns ``None`` when no key was dropped."""
    if any(int(f) for f in status[1:]):
        raise HawkError(
            f"{kernel}: hawk.scatter({team['of']}, contract='ordered') got a key outside [0, {cells}) "
            f"that is not the skip key -1; those adds were dropped, the rest are in {team['of']}")


def run(kernel_artifact: HostKernel, *, base: int = 0, count: int | None = None,
        n_samples: int | None = None, layout: str | None = None,
        **arrays) -> None:
    """Bind ``arrays`` and run the WHOLE range in ONE serial launch.

    ``count`` defaults to ``n_samples``, and ``n_samples`` to the count
    every bound per-sample plane agrees on (:func:`_n_from`). Output
    planes are the caller's, written in place; this returns nothing. A
    thread-box kernel (sidecar ``launch.box``) and a kernel that reads an index set run over the
    whole range: the library refuses ``base=`` and ``count=`` for them.

    A per-sample plane whose shape reads both as component-major ``(w, N)`` and
    as sample-major ``(N, w)`` (``(w, w)``) is refused unless the call says
    which axis holds the samples: ``layout="samples_first"`` or
    ``layout="samples_last"`` resolves every such plane of the call, and
    ``hawk.samples_first(x)`` / ``hawk.samples_last(x)`` say it for one array
    (any shape; a contradiction is refused; the array's marker beats ``layout``).
    Neither copies.

    A kernel declaring ``hawk.Grid[...]`` takes its runtime extent by label
    (``run(k, x=x, y=y, channel=64)``) or from lane-shaped planes, and refuses
    ``n_samples=``."""
    arguments = kernel_artifact.arguments(arrays, n_samples, layout, keep=False)
    n = arguments.n_samples
    kernel_artifact.refuse_partial_stages(base != 0 or count not in (None, n))
    ranges = {} if base == 0 and count in (None, n) else {"part_base": [base], "part_count": [n - base if count is None else count]}
    bound = BoundPlan(kernel_artifact, arguments, _osprey.EG_STRUCTURE_HOST_SERIAL, ranges)
    try:
        bound.launch()
    finally:
        bound.close()


def _stage_names(first: HostKernel) -> set:
    """Every plane name a stage of the split kernel ``first`` declares."""
    names = {name for _, name in first.arg_spec}
    for stage in first._stages:
        names |= {name for _, name in stage.arg_spec}
    return names | set(first.sidecar.get("cells") or ())


def _shared_shapes_checked(sidecar: dict, arrays: dict) -> dict:
    """``arrays``, each ``Param[Vector/Matrix/Table]`` plane (sidecar ``param_shapes``) checked against its declared
    shape: one value for every lane, bound in that natural shape."""
    for name, shape in (sidecar.get("param_shapes") or {}).items():
        if name not in arrays:
            continue
        got = tuple(describe(arrays[name]).shape)
        if got != tuple(shape):
            raise HawkError(
                f"kernel {sidecar.get('kernel', '?')!r}: {name!r} is a shared parameter of shape "
                f"{tuple(shape)} (one value for every lane), got shape {got}")
    return arrays


def _refuse_unbound_names(kernel_artifact: HostKernel, arrays: dict, declared: set) -> None:
    """Refuse a bound name the kernel declares no slot for, as eagle's ``bind`` does: a plane passed
    under a wrong name is otherwise never written, and its zeros read as a result."""
    unknown = sorted(set(arrays) - declared - _lane_bind.free_labels(kernel_artifact.declared))   # ``<label>=N`` is no plane
    if not unknown:
        return
    name = kernel_artifact.sidecar.get("kernel", "?")
    pruned = [n for n in unknown if n in (kernel_artifact.declared.get("unused_planes") or ())]
    stray = [n for n in unknown if n not in pruned]
    if pruned:
        raise HawkError(
            f"kernel {name!r}: {pruned} are declared but unused (the body never reads or writes "
            f"them, so the compiled kernel has no slot for them); do not bind them "
            f"(declared and used: {sorted(declared)})"
            + (f"; {stray} are not declared at all" if stray else ""))
    raise HawkError(
        f"kernel {name!r}: {stray} are not planes it declares "
        f"(declared: {sorted(declared)})")
