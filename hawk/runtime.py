# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Loading a built kernel and running it over the runtime door of the OSPREY that serves the memory its arrays live in (host or
device, :mod:`hawk._place`): the one-call :func:`run`, and the :class:`BoundPlan` that :func:`hawk.plan` makes, which both end
in the same door calls.

hawk's part is what no target decides: it resolves a plane's name to its slot from the kernel's sidecar,
classifies each array's layout, infers and checks the sample count and refuses a wrong argument by name, all
the failures of one call together. Everything that runs is the library's, reached through one
:class:`_osprey.Door` (:meth:`hawk.ospreys.Osprey.runtime_door`): the unit is loaded, each array is wrapped as a plane,
a plan is made over the range, the planes are bound and the plan is launched. The scratch planes a kernel
owns (the finished counter, the active set of a runner's plans, the compensation companions, the rows of a sum over samples, the
planes between the stages of a split kernel) are made and folded by the door, so no array library is needed
here.

Arrays are NumPy arrays or anything else that exports DLPack and the buffer protocol; the plane is the caller's
memory, written in place. A plain buffer (``array.array``, ``bytes``) exports no DLPack and is refused by name.
Every bound argument is validated against the sidecar's declaration before the door is called
(:meth:`LoadedKernel.arguments`, which :func:`run` and :meth:`hawk.plans.Plan.bind` both call): dtype is exact and fixed by role (every integer-valued role rides a 64-bit wire, and
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
(``plane_in``, ``mutable``,     is contiguous; ``layout=`` / markers        payload ``(R, C)`` gets
``out``,                         settle ``N == w``                           ``(R, C, N)``; ``(N, w)`` as its
``terminated``)                                                              transpose view, no copy
one sample                      the head alone: ``()``, ``(w,)``            ``(*P, 1)``, a batch of one
a matrix read at lane indices   ``(R*C, m)`` compact                        ``(R, C, m)``
(``gather_mats``)
table (``lookup``)              any contiguous array; a shared parameter    as passed
                                in its declared shape
wide plane (``wide_in``,        any contiguous array; a flat ``(W*N,)``     as passed, never reshaped (the
``plane_out``)                  holds element ``(e, c)`` at ``e*W + c``     door's handle takes its trailing
                                                                            extent)
cell of a sum over samples      ``(columns,)``, one value per column        one ``(1,)`` view per column slot
(``cells``)
zeroed scatter target           ``(*payload, *axes)`` as declared           as passed
uniform                         a Python number or a 0-d array              its bytes at the wire dtype
==============================  ==========================================  ==================================

The library makes and the caller does not bind: the sample count, the finished counter, a dispatch group's branch counters, compensation planes, a cell's rows and the planes between the stages of a split kernel.

Threading. The door's context admits one calling thread at a time: a second thread calling while another is
inside the door is refused by the door, with its reason.
"""

from __future__ import annotations

import json
import math
import struct
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import _osprey

from . import _broadcast, _cells, _ir, _lane_bind, _place, ospreys
from ._bind_checks import _PER_SAMPLE_ROLES, _check_bind, _n_from, narrow_index, refuse_index_width, widen_index
from ._place import Maker
from ._plane_layout import LayoutChoice, describe, is_ndarray, samples_last, split_marks
from .errors import HawkError
from .targets import ENTRY_KINDS, host_target

__all__ = ["BoundPlan", "LoadedKernel", "PlaneStore", "check_scatter_status", "dispatch_costs", "load", "run", "steps_reverse_table"]

#: The two planes a kernel built under ``Guard(active_set=True)`` reads: the live samples and their count. A plain plan
#: takes the caller's; a runner makes and compacts its own, and refuses a plan that holds the caller's.
_ACTIVE_SET_PLANES = ("active_map", "active_count")


#: The door's role code of each derived unit a recorded run takes, by the word the sidecar's ``derived`` records state it.
_DERIVED_ROLE_CODES = MappingProxyType({
    "tape": _osprey.EG_DERIVED_TAPE,
    "tape_fused": _osprey.EG_DERIVED_TAPE_FUSED,
    "step_vjp": _osprey.EG_DERIVED_STEP_VJP,
    "fused_vjp": _osprey.EG_DERIVED_FUSED_VJP,
})


#: The struct format a uniform's value is packed with, by the wire dtype the sidecar states for it.
_UNIFORM_FORMAT = MappingProxyType({"float64": "<d", "float32": "<f", "int64": "<q", "int32": "<i", "uint32": "<I", "bool": "<?"})


def through_door(call, *args, **kwargs):
    """Return ``call(*args, **kwargs)``; a refusal of the runtime (its own exception types) is raised as
    :class:`~hawk.errors.HawkError`, its reason as given. Any other exception, one a step of the user's raised included, passes
    unchanged."""
    try:
        return call(*args, **kwargs)
    except (_osprey.NotWritten, _osprey.BadRequest, _osprey.Fault) as refusal:
        raise HawkError(str(refusal)) from refusal



class _Arguments(NamedTuple):
    """One call's arguments, checked and ready to wrap: what :meth:`LoadedKernel.arguments` returns."""

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


class LoadedKernel:
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
        #: the host door the kernel was loaded through; a plan over device memory loads the unit through its own door
        self._door = ospreys.serving(host_target().id).runtime_door()
        self._units: dict = {}
        self._source = source = sidecar.get("source")
        if source is None:
            #: the unit a run loads and binds, and the specification of the planes it takes
            members = self._members = self._group_members(sidecar)
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
            self._members = []
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
        #: the names of the active map and count when the unit reads an active set the caller or a runner makes, else ()
        self.active_set = (() if self.declared.get("launch", {}).get("index_set") is not None
                           else tuple(name for name in _ACTIVE_SET_PLANES if name in self._slot_named))
        self._one_step_words: dict = {}
        self._vjps: dict = {}
        #: the unit on the host door, which the load checks
        self.unit = self.unit_on(self._door)

    def unit_on(self, door):
        """The unit loaded through ``door``, once for each door: a plan runs on the door of the place its planes live in. A unit
        the library built for the other kind of memory is refused by the door, by name."""
        held = self._units.get(door)
        if held is None:
            if self._source is None:
                deps = [_load_unit(door, self.path, member) for member in self._members]
                held = _load_unit(door, self.path, self.sidecar, deps=deps)
            else:
                # a split kernel runs as the unit that lists its stages: the stage units are loaded first and named as its deps
                stages = [_load_unit(door, self.path, self.sidecar), *(stage.unit_on(door) for stage in self._stages)]
                held = _load_unit(door, self.path, {"kernel": self._source["name"], "digest": self._source["digest"]},
                                  deps=stages)
            self._units[door] = held
        return held

    def derived_roles(self, steps_reverse: str = "per_step") -> list:
        """The words of the derived units a recorded run of this kernel takes, in the order :meth:`derived_units` loads them: the
        taping twin (the K-step one when the kernel takes other than one step per launch), the step vjp, and the K-step vjp for
        ``steps_reverse="fused"`` or, when the build has one, for ``"auto"`` (the library then chooses between the two)."""
        twin = "tape" if self.sidecar.get("finish", {}).get("steps") == 1 else "tape_fused"
        built = {record["role"] for record in self.sidecar.get("derived") or ()}
        many = steps_reverse == "fused" or (steps_reverse == "auto" and "fused_vjp" in built)
        return [twin, "step_vjp", *(["fused_vjp"] if many else [])]

    def _derived_records(self, roles) -> list:
        """The sidecar's ``derived`` records of ``roles``, in order; a role the build did not record is refused by name."""
        recorded = {record["role"]: record for record in self.sidecar.get("derived") or ()}
        missing = [role for role in roles if role not in recorded]
        if missing:
            raise HawkError(f"{self.path.name}: its build records no derived unit for {missing}, which a recorded run takes; "
                            "the kernel's library must hold the taping twin and the vjp of its step, listed in the sidecar's "
                            "'derived'")
        return [recorded[role] for role in roles]

    def recorded_window(self) -> int:
        """The steps between the states a recorded run of this kernel keeps: the window its taping twin was built with, which the
        library takes as the one window of a recorded run."""
        (twin,) = self._derived_records(self.derived_roles()[:1])
        return int(twin["window"])

    def derived_units(self, roles, door=None) -> list:
        """The units of ``roles`` (words of :meth:`derived_roles`) as ``(door role code, unit loaded through door)`` pairs, in order;
        ``door`` defaults to the host door.

        The units are the ones the sidecar's ``derived`` list names, ``{"role", "kernel", "digest"}`` each, loaded from this
        kernel's own library: the door takes a derived unit only from the library of its primal.
        """
        door = self._door if door is None else door
        return [(_DERIVED_ROLE_CODES[role], _load_unit(door, self.path, record))
                for role, record in zip(roles, self._derived_records(roles))]

    def graph_vjp(self, door=None):
        """The unit, loaded through ``door``, that reverses one launch of this kernel in a recorded graph, from this kernel's own
        library, or ``None`` when the build holds none (:func:`hawk.deploy` with ``reverse=True`` makes it). The library takes a
        vjp only from the library of its launch. ``door`` defaults to the host door."""
        door = self._door if door is None else door
        if door not in self._vjps:
            records = [record for record in self.sidecar.get("derived") or () if record["role"] == "step_vjp"]
            self._vjps[door] = _load_unit(door, self.path, records[0]) if records else None
        return self._vjps[door]

    def gradient_names(self) -> tuple:
        """The planes a recorded run's reverse differentiates with respect to: the ``wrt`` of the step vjp's own sidecar that are
        planes of the kernel."""
        (record,) = self._derived_records(["step_vjp"])
        vjp = json.loads((self.path.parent / f"{record['kernel']}.json").read_text())
        # the start copy of a time plane (a ``wrt`` of a time-stepped kernel) is no plane: its gradient joins the time plane's
        return tuple(name for name in vjp["derivative"]["wrt"] if name in self._slot_named)

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
                  keep: bool, make: Maker, outputs: bool = False, running: bool = False,
                  shown: dict | None = None) -> _Arguments:
        """Check ``arrays`` against the sidecar and return what a plan wraps and binds.

        ``arrays`` maps a plane's name to an array and a ``uniform``'s to a Python number. Refuses naming every failing
        argument, not just the first. A kernel that declares a ``Grid`` takes its runtime extent by label, and ``n_samples``
        is the count every per-sample plane agrees on when it is not stated. The planes stay the caller's memory.
        ``keep``: they stay bound after the call, so an ``Index`` plane of another integer width than the kernel declares is
        refused (a copy would stay behind when the caller changes the column); a call that binds once converts it instead.
        ``make`` makes, in the memory the planes live in, every array the call needs and the caller left out.
        ``outputs``: an output plane the caller left out is made, zeroed, in the shape and dtype the sidecar declares.
        ``running``: the call runs its samples to done (a runner), so a ``terminated`` mask the caller left out starts
        all-false, and the active set and the step word are the runner's, not bound here.
        ``shown``: the caller's own arrays for the planes of ``arrays`` that are views of them (a broadcast input's compact buffer);
        the result's ``given`` reports these.
        An automatic kernel's ``fused_steps`` word, a finishing kernel's ``finished_count`` counter and the other planes the
        library owns are not bound.
        """
        sidecar = self.declared
        spec = self.bind_spec
        arrays = _shared_shapes_checked(sidecar, arrays)
        stated = _lane_bind.stated_extents(sidecar, arrays)
        given = {name: value for name, value in {**arrays, **(shown or {})}.items() if name not in stated}   # not lane or compact views
        arrays, lane_n = _lane_bind.bind(sidecar, spec, arrays, n_samples, _PER_SAMPLE_ROLES)
        plain, axes = split_marks(arrays, layout)
        choice = LayoutChoice(layout, axes)
        if lane_n is not None:
            n_samples = lane_n
        if n_samples is None:
            compact = sidecar.get("gather_mats") or {}
            n_samples = _n_from([(r, n) for r, n in spec if n not in compact], plain,
                                sidecar.get("arg_shapes", {}), choice)
            _lane_bind.check_total(sidecar, n_samples)
        if outputs:
            made = self._made_outputs(plain, n_samples, make)
            given |= made
            arrays = {**arrays, **self._lane_views(made)}
            plain, axes = split_marks(arrays, layout)
            choice = LayoutChoice(layout, axes)
        if running:
            made = self._made_for_running(arrays, n_samples, make)
            given |= made
            arrays = {**arrays, **made}
        _refuse_unbound_names(self, arrays, self._plane_names())
        return self._checked(arrays, n_samples, layout, keep=keep, running=running, make=make)._replace(given=given, extents=stated)

    def output_names(self) -> tuple:
        """The names of the output planes a caller sees, in slot order: the sinks the library does not make, a cell by its own
        name (a cell of a kernel in stages is the caller's only when it binds it)."""
        cells = self.declared.get("cells") or {}
        rows = {plane for cell, spec in cells.items() for plane in _ir.planes(cell, spec["columns"])}
        named = [name for role, name in self.bind_spec if role in _ir.SINK_ROLES and name not in self._made
                 and (name not in rows or name in cells)]
        return tuple(dict.fromkeys([*named, *(cell for cell in cells if not self.declared.get("stages"))]))

    def _made_outputs(self, arrays: dict, n_samples: int, make: Maker) -> dict:
        """The zeroed arrays, made by ``make``, of the outputs ``arrays`` leaves out, by name, each as long as the lanes unless the
        sidecar declares its axes, and the gradient of a lookup plane as large as that plane. A wide plane whose axes the sidecar
        does not declare, whose extent only the caller knows, is refused by name: the caller binds it."""
        sidecar = self.declared
        cells = sidecar.get("cells") or {}
        labels = _lane_bind.free_labels(sidecar)
        extents = {label: arrays[label] for label in labels if label in arrays}
        roles = {name: role for role, name in self.bind_spec}
        adjoint = sidecar.get("adjoint_like") or {}
        made = {}
        for name in self.output_names():
            if name in arrays:
                continue
            if name in cells:
                made[name] = make.zeros((cells[name]["columns"],), self.scalar_type)
                continue
            if adjoint.get(name) in arrays:     # the gradient of a lookup plane is as large as that plane
                made[name] = make.zeros(describe(arrays[adjoint[name]]).shape, sidecar["arg_dtypes"][name])
                continue
            if (roles[name] == "plane_out" and name not in sidecar["zeroed_outputs"]
                    and name not in (sidecar.get("plane_axes") or {})):
                raise HawkError(f"kernel {sidecar.get('kernel', '?')!r}: argument {name!r}: the extent of a wide plane is the "
                                "caller's, so it is not made; bind it")
            width = int(sidecar.get("arg_shapes", {}).get(name) or 1)
            shape = _lane_bind.output_shape(sidecar, name, width, n_samples, extents)
            plane = make.zeros(shape, sidecar["arg_dtypes"][name])
            made[name] = samples_last(plane) if width > 1 and shape[0] == width else plane
        return made

    def _lane_views(self, made: dict) -> dict:
        """``made`` (:meth:`_made_outputs`) as the unit binds it: each plane of a kernel that declares a ``Grid`` flattened to
        ``(*component axis, lanes)``, a view of the same memory; a plane the unit declares the axes of, a cell and the gradient of
        a lookup plane stay as they are."""
        sidecar = self.declared
        if _lane_bind.lane_axes(sidecar) is None:
            return made
        kept = {*(sidecar.get("plane_axes") or ()), *(sidecar.get("cells") or ()), *(sidecar.get("adjoint_like") or ())}
        views = {}
        for name, plane in made.items():
            marked = hasattr(plane, "array")
            array = plane.array if marked else plane
            width = int(sidecar.get("arg_shapes", {}).get(name) or 1)
            view = array if name in kept else array.reshape((*array.shape[:1][:width > 1], -1))
            views[name] = samples_last(view) if marked else view
        return views

    def _made_for_running(self, arrays: dict, n_samples: int, make: Maker) -> dict:
        """What a runner starts from when the caller left it out: an all-false ``terminated`` mask."""
        return {name: make.zeros((n_samples,), "bool")
                for role, name in self.bind_spec if role == "terminated" and name not in arrays}

    def _plane_names(self) -> set:
        """The names of the planes and cells a call to this kernel may bind: every stage's, for a split kernel."""
        if self.declared.get("stages"):
            return _stage_names(self)
        return {name for _, name in self.bind_spec} | set(self.declared.get("cells") or ())

    def declared_names(self) -> set:
        """The names of every plane, cell and runtime axis a call to this kernel may state."""
        return self._plane_names() | _lane_bind.free_labels(self.declared)

    def _checked(self, arrays: dict, n_samples: int, layout, *, keep: bool, running: bool, make: Maker) -> _Arguments:
        """Validate ``arrays`` and return the views and uniform values a plan binds.

        ``keep``: see :meth:`arguments`.
        """
        sidecar = self.declared
        given = arrays
        arrays, axes = split_marks(arrays, layout)
        choice = LayoutChoice(layout, axes)
        arrays = self._with_seed(arrays)
        arrays, counters = self._split_counters(arrays)
        if _ir.FUSED_STEPS_PLANE in self._slot_named and _ir.FUSED_STEPS_PLANE not in arrays and not running:
            arrays = {**arrays, _ir.FUSED_STEPS_PLANE: self._one_step(make)}
        index_inputs = tuple(
            name
            for role, name in self.bind_spec
            if role in ("plane_in", "lookup", "wide_in") and sidecar.get("arg_dtypes", {}).get(name) == "int64"
        )
        if keep:
            refuse_index_width(index_inputs, sidecar.get("index_planes") or (), arrays, sidecar.get("kernel", ""))
        else:
            if index_inputs:
                arrays = widen_index(index_inputs, arrays)
            if sidecar.get("index_planes"):
                arrays = narrow_index(sidecar["index_planes"], arrays, sidecar.get("kernel", ""))
        arrays = self._with_cell_columns(arrays)
        made = self._made | {*self.active_set, _ir.FUSED_STEPS_PLANE} if running else self._made   # a runner makes both
        spec = tuple(pair for pair in self.bind_spec if pair[1] in arrays or pair[1] not in made)
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


    def _one_step(self, make: Maker):
        """The one-cell step word of ``1`` an automatic-step kernel reads when the caller binds none: a launch
        outside a runner takes one step. Made once for each place."""
        if make.device not in self._one_step_words:
            import numpy as np

            self._one_step_words[make.device] = make.copy_of(np.ones(1, dtype=np.int64))
        return self._one_step_words[make.device]

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
            value, columns = arrays[cell], spec["columns"]
            held = math.prod(describe(value).shape)
            if held != columns:
                raise HawkError(f"kernel {self.declared.get('kernel', '?')!r}: the cell {cell!r} holds {held} value(s); it is one "
                                f"value per column, {columns} here")
            if columns == 1 and not is_ndarray(value) and tuple(describe(value).shape) == (1,):
                out[cell] = value         # the one-column cell of any array is its own one-element view
                continue
            # the runtime binds one one-element plane per column and takes no offset, so each column is the array's own view of
            # that element: its indexing, whatever library the array comes from
            try:
                column = value.reshape(-1)
                views = [column[k : k + 1] for k in range(columns)]
            except (AttributeError, TypeError) as failure:
                raise HawkError(f"kernel {self.declared.get('kernel', '?')!r}: the cell {cell!r} ({type(value).__name__}) cannot "
                                f"be viewed per column: hawk reshapes it to one dimension and slices it, and its library "
                                f"refuses ({failure})") from failure
            out.update(zip(_ir.planes(cell, columns), views))
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
        handed over as the caller passed it. An array that exports DLPack and is not a NumPy array (whether or not it also exports a buffer) is handed over as
        passed, or as its own reshape to the payload axes when only those differ: hawk has no array library to make another
        view of it with, so one that needs a transpose is refused by name.
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
            reshaped = shape != native and shape[-1:] == native[-1:] and math.prod(shape) == math.prod(native)
            if kind == "native" and payload and reshaped and hasattr(value, "reshape"):
                # component-major memory in another shape: the array's own reshape views it, as NumPy's does below
                return value.reshape(native)
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

    def slot(self, name: str) -> int:
        """The index of the plane ``name`` in the unit's slot table, which the library binds by."""
        return self._slot_named[name]

    def plane_view(self, name: str, value, layout: str | None = None) -> object:
        """``value`` as the plane the unit's slot ``name`` takes, by the rules of a bind (the table in the module docstring):
        the caller's memory, never a copy. ``layout`` is that of :func:`run`."""
        role = next(role for role, plane in self.bind_spec if plane == name)
        return self._plane_view(role, name, value, LayoutChoice(layout, {}))

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


def _free_planes(door, planes: list) -> None:
    """Free the plane handles in ``planes``."""
    for plane in planes:
        door.plane_free(plane)
    planes.clear()


class PlaneStore:
    """The door planes over the caller's arrays that the launches of one graph share: one plane per region of memory.

    A graph is reversed over the planes its launches name, and the library refuses two planes over one region, so every launch
    of a graph is bound through the graph's store. The store keeps its planes, and the arrays they wrap, until it and every
    plan bound through it are collected.
    """

    def __init__(self, door) -> None:
        """Make an empty store over ``door``."""
        self._door = door
        self._held: dict = {}
        self._handles: list = []
        self._finalizer = weakref.finalize(self, _free_planes, door, self._handles)

    @staticmethod
    def region(view) -> tuple:
        """What names the memory ``view`` covers: a NumPy array's address, shape, strides and dtype, else the object itself. Two
        views of the same region have one name."""
        if is_ndarray(view):
            interface = view.__array_interface__
            return (interface["data"][0], interface["shape"], interface["strides"], interface["typestr"])
        return ("object", id(view))

    def entries(self) -> list:
        """The ``(plane, view)`` pairs the store holds, in the order the planes were made."""
        return list(self._held.values())

    def plane(self, view):
        """The plane over ``view``, made on the first ask for its region and the same one after."""
        key = self.region(view)
        held = self._held.get(key)
        if held is None:
            handle = through_door(_place.wrap, self._door, view)
            self._handles.append(handle)
            held = self._held[key] = (handle, view)
        return held[0]


class BoundPlan:
    """A plan over one kernel with its planes bound: launch it as often as needed, rebind planes between launches.

    The planes are the caller's memory, read and written in place; a launch moves no data. The door's handles live until
    :meth:`close` or until this object is collected.
    """

    def __init__(self, kernel: LoadedKernel, arguments: _Arguments, structure: int, ranges: dict, *, place: Maker,
                 n_samples: int | None = None, layout: str | None = None, broadcast: tuple | None = None,
                 store: PlaneStore | None = None) -> None:
        """Make the library's plan, through the door of ``place`` (where the planes live), over ``arguments.n_samples`` samples
        under ``structure`` (a door structure code) and partitioned by ``ranges`` (the door's ``part_base`` / ``part_count`` or
        ``split_evenly``, else none: the whole range), then bind the planes. ``n_samples`` and ``layout`` are what the caller
        stated, kept for :meth:`rebind`.
        ``broadcast`` is ``(traced kernel, patterns)`` of a plan deployed from a kernel: the patterns the planes were bound in
        (none for the exact shapes), which a rebind keeps; ``None`` for a plan over a loaded artifact. ``store`` is the graph's
        :class:`PlaneStore` when the plan is bound for a graph being written: its planes are the store's."""
        self._kernel = kernel
        self._broadcast = broadcast
        self._place = place
        self._door = place.door
        self._store = store
        self._stated = (n_samples, layout)
        self._n = arguments.n_samples
        #: the handles the door made, kept in lists the finalizer frees without holding this object
        self._plans: list = []
        self._planes: list = []
        self._finalizer = weakref.finalize(self, _release, self._door, self._plans, self._planes)
        try:
            self._plan = through_door(self._door.plan_new, kernel.unit_on(self._door), structure, _osprey.EG_REDUCE_COMPENSATED, self._n,
                                       **ranges)
            self._plans.append(self._plan)
            self._wire(arguments)
        except BaseException:
            self.close()
            raise

    def _wire(self, arguments: _Arguments) -> None:
        """Wrap ``arguments``' planes and bind them with its uniforms, replacing the planes of the earlier bind. A plane over the
        caller's array comes from the graph's store when this plan has joined one, else it is this plan's own."""
        door, kernel = self._door, self._kernel
        wrapped, owned = {}, []
        try:
            for name, view in arguments.planes.items():
                if self._store is None:
                    wrapped[name] = through_door(_place.wrap, door, view)
                    owned.append(wrapped[name])
                else:
                    wrapped[name] = self._store.plane(view)
            for name in kernel._own_cell_planes(arguments.planes):
                kind = _osprey.EG_MEM_HOST if self._place.device == _place.HOST else _osprey.EG_MEM_DEVICE
                wrapped[name] = through_door(door.plane_new, [1], 2, 32 if kernel.scalar_type == "float32" else 64, kind=kind,
                                              flags=_osprey.EG_PLANE_ZERO)
                owned.append(wrapped[name])
            through_door(
                door.bind,
                self._plan,
                [kernel._slot_named[name] for name in wrapped],
                list(wrapped.values()),
                [kernel._slot_named[name] for name in arguments.uniforms],
                list(arguments.uniforms.values()),
            )
        except BaseException:
            for plane in owned:
                door.plane_free(plane)
            raise
        replaced = list(self._planes)
        self._planes[:] = owned
        for plane in replaced:
            door.plane_free(plane)
        self._arguments = arguments

    def join(self, store: PlaneStore) -> None:
        """Bind this plan's planes through ``store`` from now on, so that a graph over it shares one plane per region of memory
        with the other launches of the graph. The caller's arrays and what a launch reads and writes do not change."""
        if self._store is not store:
            self._store = store
            self._wire(self._arguments)

    @property
    def names(self) -> tuple:
        """The names of the kernel's planes and values, in the order the kernel declares them."""
        return tuple(name for role, name in self._kernel.arg_spec if role != "extent")

    @property
    def n(self) -> int:
        """The sample count this plan was bound for."""
        return self._n

    @property
    def planes(self) -> MappingProxyType:
        """The arrays bound, by name, as the caller gave them (a plane made for an output left out included), read-only."""
        return MappingProxyType({name: getattr(value, "array", value) for name, value in self._arguments.given.items()})

    @property
    def door(self):
        """The runtime door this plan and its planes belong to: the door of the place the planes live in."""
        return self._door

    @property
    def kernel(self) -> LoadedKernel:
        """The loaded kernel this plan launches."""
        return self._kernel

    @property
    def views(self) -> MappingProxyType:
        """The arrays the library reads and writes, by plane name: the caller's memory as the plan takes it (a broadcast input
        compact, a sample-major plane as its transpose), read-only. A graph's reverse seeds and gradients are in these shapes."""
        return MappingProxyType(dict(self._arguments.planes))

    @property
    def handle(self):
        """The library's handle of the plan, which a runner or a recording composes with others of its context."""
        self._check_open()
        return self._plan

    def _check_open(self) -> None:
        """Refuse a closed plan, by name."""
        if not self._finalizer.alive:
            raise HawkError(f"{self._kernel.path.name}: this plan is closed; bind a new one")

    def launch(self) -> int:
        """Run the plan once over the bound planes and return the host tiles the library ran (0 on a device). A launch over device
        memory is queued: :meth:`wait` or :func:`hawk.synchronize` returns when it has run. A counter array the caller bound
        (``finished_count``, a dispatch group's branch counters) is filled from the plan's counters."""
        self._check_open()
        tiles = through_door(self._door.launch, self._plan)
        self.settle_counters()
        return tiles

    def wait(self) -> None:
        """Return when every launch queued on this plan's door has run; a host launch has run when it returns."""
        through_door(self._door.sync)

    def enqueue(self) -> None:
        """Queue one launch into the recording in progress: it runs at every replay of the recorded graph, and
        :meth:`settle_counters` after a replay adds what it counted."""
        self._check_open()
        through_door(self._door.launch, self._plan)

    def settle_counters(self, counted: dict | None = None) -> None:
        """Add what a run counted to the counter arrays the caller bound (``finished_count``, a dispatch group's branch
        counters); nothing when it bound none. ``counted`` has the shape of :meth:`counters` and defaults to it."""
        if self._arguments.counters:
            self._kernel._fill_counters(self.counters() if counted is None else counted, self._arguments.counters)

    def counters(self) -> dict:
        """The library's counters of the plan: ``finished`` and ``steps`` (``None`` when the kernel has none) and ``mix``.

        A counter array the caller bound holds its value before a launch plus what that launch counted."""
        return through_door(self._door.plan_counters, self._plan)

    def rebind(self, **planes) -> BoundPlan:
        """Replace the named planes (and uniforms) with new values, every other bound value staying, and return this plan.
        An extent stated by label at bind (``e=``) stays too; a rebind may state it again, and a value that contradicts a bound
        plane is refused by name.

        The same checks as :meth:`hawk.plans.Plan.bind` run over the whole set; the sample count cannot change. A plane bound
        broadcast keeps its pattern: it takes any shape of it, or the compact buffer it holds, and another pattern is refused.
        """
        n_samples, layout = self._stated
        held = self._arguments
        shown = merged = {**held.given, **planes}
        if self._broadcast is not None:
            traced, patterns = self._broadcast
            shown, merged = _broadcast.rebound(traced, patterns, held.given, planes, self._n, traced.name)
        arguments = self._kernel.arguments({**held.extents, **merged}, n_samples, layout, keep=True, make=self._place, shown=shown)
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
_DICT_DECLARATIONS = ("arg_dtypes", "arg_shapes", "arg_widths", "slot_shapes", "param_shapes",
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
    ones the caller does not read) and the extent of each runtime axis, which the runtime fills from the shape of the bound plane."""
    names = {_ir.FINISHED_PLANE} | {name for role, name in sidecar["arg_spec"] if role == "extent"}
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


def load(directory, kernel: str, sidecar: dict | None = None) -> LoadedKernel:
    """Load ``<directory>/<kernel>.so``, reading its sidecar JSON if omitted."""
    directory = Path(directory)
    if sidecar is None:
        sidecar = json.loads((directory / f"{kernel}.json").read_text())
    return LoadedKernel(directory / f"{kernel}.so", sidecar)


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


def run(kernel_artifact: LoadedKernel, *, base: int = 0, count: int | None = None,
        n_samples: int | None = None, layout: str | None = None,
        **arrays) -> None:
    """Bind ``arrays`` and run the WHOLE range in ONE serial launch (one device launch over arrays in device memory) and wait for it.

    Anything but a loaded kernel is run once in the order written and waited for: a kernel or a plan, a bound
    :func:`hawk.host_step`, a built :class:`hawk.Graph`, or an iterable of them, whose kernels and plans bind the ``arrays`` by
    name (``base=`` and ``count=`` are a loaded kernel's).

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
    from .plans import DeviceKernel, HostSerial, Plan    # plans builds on this module

    if not isinstance(kernel_artifact, LoadedKernel):
        if base != 0 or count is not None:
            raise HawkError("hawk.run: base= and count= are the range of a loaded kernel; a plan, a host step, a graph or a list "
                            "of them runs whole")
        from .runner import run_once    # runner builds on this module

        return run_once(kernel_artifact, arrays, n_samples, layout)
    found = _place.place_of(arrays, kernel_artifact.path.name)
    structure = HostSerial if found is None or found[0] == _place.HOST else DeviceKernel
    bound = Plan(kernel_artifact, structure, None, None).bound(
        arrays, n_samples=n_samples, layout=layout, keep=False, span=(base, count))
    try:
        bound.launch()
        bound.wait()
    finally:
        bound.close()


def _stage_names(first: LoadedKernel) -> set:
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


def _refuse_unbound_names(kernel_artifact: LoadedKernel, arrays: dict, declared: set) -> None:
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
