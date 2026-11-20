# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Loading a built host kernel and running it over the runtime door of the host OSPREY.

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
(:func:`_check_bind`): dtype is exact and fixed by role (every integer-valued role rides a 64-bit wire, and
``Terminated`` is always ``bool``), a per-sample plane holds exactly ``n_samples`` samples, every plane is
C-contiguous and a sink role's plane writable, except a sample-major plane whose transpose is C-contiguous,
which binds as that view (:func:`plane_layout`). A ``(w, w)`` plane reads both ways and is refused unless
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
from pathlib import Path
from types import MappingProxyType

import _osprey

from . import _cells, _ir, _lane_bind, ospreys
from ._bind_checks import _PER_SAMPLE_ROLES, _check_bind, _n_from, narrow_index, refuse_index_width, widen_index
from ._plane_layout import LayoutChoice, describe, is_buffered, split_marks
from .errors import HawkError
from .targets import ENTRY_KINDS, host_target

__all__ = ["HostKernel", "check_scatter_status", "dispatch_costs", "load", "run", "steps_reverse_table"]

#: The struct format a uniform's value is packed with, by the wire dtype the sidecar states for it.
_UNIFORM_FORMAT = MappingProxyType({"float64": "<d", "float32": "<f", "int64": "<q", "int32": "<i", "uint32": "<I", "bool": "<?"})


def _through_door(call, *args, **kwargs):
    """Return ``call(*args, **kwargs)``; a refusal of the door is raised as :class:`~hawk.errors.HawkError`, its reason as given."""
    try:
        return call(*args, **kwargs)
    except (ValueError, _osprey.Fault) as refusal:
        raise HawkError(str(refusal)) from refusal


class HostKernel:
    """One built host kernel: its sidecar and its unit, loaded through the host OSPREY's door.

    Construction is per kernel; every :meth:`bind_all` and :meth:`launch` after it is per bind or per launch.
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
        self._bound = None
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

    # -- per bind ----------------------------------------------------------
    def bind_all(self, arrays: dict, n_samples: int, *, layout=None) -> None:
        """Check ``arrays`` against the sidecar and keep them for :meth:`launch`.

        ``arrays`` maps a slot name to an array; a ``uniform`` maps to a Python number. Refuses naming every
        failing argument, not just the first. The planes stay the caller's memory, so a column of another
        integer width than the kernel declares is refused (a copy would stay behind when the caller changes
        the column); :func:`run` converts it instead. An automatic kernel's ``fused_steps`` word, a finishing
        kernel's ``finished_count`` counter and the other planes the library owns are not bound.
        """
        self._bind_all(arrays, n_samples, layout, keep=True)

    def _bind_all(self, arrays: dict, n_samples: int, layout, *, keep: bool = False) -> None:
        """Validate ``arrays`` and keep the views and uniform values to launch with.

        ``keep``: the planes stay bound after the call (``bind_all``), so an ``Index`` plane of another width
        is refused instead of copied; ``run`` binds per call and converts.
        """
        sidecar = self.declared
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
        self._bound = (planes, uniforms, counters)

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
            if not is_buffered(arrays[cell]):
                if spec["columns"] != 1 or tuple(describe(arrays[cell]).shape) != (1,):
                    raise HawkError(
                        f"kernel {self.declared.get('kernel', '?')!r}: the cell {cell!r} exports DLPack and no buffer, so "
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
        handed over as the caller passed it. An array that exports DLPack and no buffer is always handed over as
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
        if not is_buffered(value):
            if kind != "native" or (payload and shape != native):
                want = (*payload, "N") if payload else ("N",)
                raise HawkError(
                    f"kernel {sidecar.get('kernel', '?')!r}: argument {name!r} exports DLPack and no buffer, so hawk can "
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

    # -- per launch --------------------------------------------------------
    def launch(self, base: int, count: int, n_samples: int) -> None:
        """Launch over the samples ``[base, base + count)`` of the planes :meth:`bind_all` kept.

        Serial, always: the reference every partitioned, tiled or ranked run is judged against.
        """
        if self._bound is None:
            raise HawkError(f"{self.path.name}: bind_all(arrays, n_samples) first: nothing is bound to launch")
        self._launch(base, count, n_samples)

    def _fill_counters(self, plan, counters: dict) -> None:
        """Write the plan's finished count and branch counters into the arrays the caller bound for them."""
        counted = _through_door(self._door.plan_counters, plan)
        if self._FINISHED in counters and counted["finished"] is not None:
            counters[self._FINISHED][0] = counted["finished"]
        offset = 0
        for entry in self.declared.get("dispatch", ()):
            if entry.get("mix") is None:
                continue
            width = int(entry["K"])
            if entry["mix"] in counters:
                for i, value in enumerate(counted["mix"][offset : offset + width]):
                    counters[entry["mix"]][i] = value
            offset += width

    def _launch(self, base: int, count: int | None, n_samples: int) -> None:
        """Wrap the bound arrays as planes, plan the range, bind and launch; the handles are freed whatever happens."""
        door = self._door
        planes, uniforms, counters = self._bound
        wrapped = {}
        plan = None
        try:
            for name, view in planes.items():
                wrapped[name] = _through_door(door.plane_wrap, view)
            for name in self._own_cell_planes(planes):
                wrapped[name] = _through_door(door.plane_new, [1], 2, 32 if self.scalar_type == "float32" else 64,
                                              flags=_osprey.EG_PLANE_ZERO)
            whole = base == 0 and count in (None, n_samples)
            ranges = {} if whole else {"part_base": [base], "part_count": [count]}
            plan = _through_door(
                door.plan_new, self.unit, _osprey.EG_STRUCTURE_HOST_SERIAL, _osprey.EG_REDUCE_COMPENSATED, n_samples, **ranges
            )
            _through_door(
                door.bind,
                plan,
                [self._slot_named[name] for name in wrapped],
                list(wrapped.values()),
                [self._slot_named[name] for name in uniforms],
                list(uniforms.values()),
            )
            _through_door(door.launch, plan)
            if counters:
                self._fill_counters(plan, counters)
        finally:
            if plan is not None:
                door.plan_free(plan)
            for plane in wrapped.values():
                door.plane_free(plane)


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


def _load_unit(door, path: Path, sidecar: dict, deps=()):
    """The unit ``sidecar`` names (``kernel``, ``digest``) from the library at ``path``; ``deps`` are the loaded units it runs."""
    return _through_door(door.unit_load, str(path), sidecar["kernel"], bytes.fromhex(sidecar["digest"]), deps=list(deps))


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
    sidecar = kernel_artifact.declared
    spec = kernel_artifact.bind_spec
    arrays = _shared_shapes_checked(sidecar, arrays)
    arrays, lane_n = _lane_bind.bind(sidecar, spec, arrays, n_samples, _PER_SAMPLE_ROLES)
    plain, axes = split_marks(arrays, layout)
    if lane_n is not None:
        n_samples = lane_n
    if n_samples is None:
        compact = sidecar.get("gather_mats") or {}
        n_samples = _n_from([(r, n) for r, n in spec
                             if not (r == "mat_in" and n in compact)], plain,
                            sidecar.get("arg_shapes", {}),
                            LayoutChoice(layout, axes))
        _lane_bind.check_total(sidecar, n_samples)
    cells = sidecar.get("cells") or {}
    if sidecar.get("stages"):
        declared = _stage_names(kernel_artifact)
        if base != 0 or count not in (None, n_samples):
            raise HawkError(
                f"{kernel_artifact.path.name}: this kernel reads a value reduced over "
                "every sample, so it runs over every sample (no base= or count=)")
    else:
        declared = {name for _, name in spec} | set(cells)
    _refuse_unbound_names(kernel_artifact, arrays, declared)
    kernel_artifact._bind_all(arrays, n_samples, layout)
    kernel_artifact._launch(base, count, n_samples)


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
