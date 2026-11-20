# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Running a model until every sample is done: :func:`until_done` builds the loop, :func:`simulate` runs it and returns
the state.

hawk binds the planes by the rules of :meth:`hawk.plans.Plan.bind` and states how a runner compacts the live samples
(:class:`ActiveSet`); the run itself is the host library's: its picks, its compaction, its counting of steps and finished
samples. The model is a :class:`~hawk.plans.Plan`, a kernel (deployed first) or a list of them run as one step, in order,
over one namespace of planes: a name is one plane in every plan that declares it.

A run can be recorded (``record=True``) and reversed (:func:`hawk.reverse`): the library keeps the states the reverse
needs and runs the vjp of the step backwards over them. The taping twin and the vjp of the step are units of the kernel's own
library, named by the ``derived`` records of its sidecar (:func:`hawk.deploy` with ``record=True`` builds them). Not offered yet:
``events="refine"``.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import _osprey

from .errors import HawkError
from ._plane_layout import describe
from .plans import Plan, deploy
from .runtime import BoundPlan, LoadedKernel, through_door

__all__ = ["ActiveSet", "GradientReport", "Gradients", "RunRecord", "RunReport", "Runner", "SimResult", "simulate", "until_done"]

#: The door's choice of how a recorded run is reversed, by the word the caller states.
_STEPS_REVERSE = MappingProxyType({"per_step": _osprey.EG_STEPS_REVERSE_PER_STEP,
                                   "fused": _osprey.EG_STEPS_REVERSE_FUSED, "auto": _osprey.EG_STEPS_REVERSE_AUTO})


class ActiveSet:
    """The live samples of a kernel built under ``Guard(active_set=True)``: how a runner compacts them, or the map and count
    a plain launch runs.

    For a runner, ``every`` is the compaction cadence in steps (``None``: the library's choice) and ``reorder`` the locality
    threshold ``theta`` at which it also reorders the bound planes (``None``: never); the library makes and compacts the
    set itself, so ``mask`` and ``positions`` are not given. For a plain launch, ``mask`` (a one-dimensional bool or
    ``uint8`` array, set = finished, read where it is every time :meth:`planes` is called) or ``positions`` (the entries of
    the map, in the order the launch runs them, each at most once) state the set, and :meth:`planes` returns the planes to
    bind. A kernel emitted for groups of ``G`` lanes reads group indices: a mask is compacted by group, and ``positions``
    are group indices.
    """

    def __init__(self, mask=None, *, positions=None, every: int | None = None, reorder: float | None = None) -> None:
        """Keep the stated set or cadence; refuse a set stated twice and a cadence that is not a whole number."""
        if mask is not None and positions is not None:
            raise HawkError("ActiveSet takes mask= or positions=, not both")
        if (mask is not None or positions is not None) and (every is not None or reorder is not None):
            raise HawkError("ActiveSet(every=, reorder=) states a runner's compaction; a mask or positions state a plain "
                            "launch's set: give one or the other")
        if every is not None and (isinstance(every, bool) or not isinstance(every, int)):
            raise HawkError(f"ActiveSet(every=...) is a whole number of steps or None, got {every!r}")
        if reorder is not None and (isinstance(reorder, bool) or not isinstance(reorder, (int, float))):
            raise HawkError(f"ActiveSet(reorder=...) is a locality threshold or None, got {reorder!r}")
        self.mask, self.positions, self.every, self.reorder = mask, positions, every, reorder
        self._planes: dict | None = None

    def planes(self, kernel) -> dict:
        """The planes ``{map name: array, count name: array}`` of this set for ``kernel`` (a :class:`~hawk.plans.Plan` or a
        loaded kernel), to bind with :meth:`hawk.plans.Plan.bind`. The names and the map's dtype are the kernel's.

        The arrays are made on the first call; a later call writes the set again into the same arrays from the mask as it
        is then, so a plan bound earlier sees it. The map is as long as the samples (rounded up to whole groups) and the
        count says how many lanes the launch runs.
        """
        import numpy as np

        loaded = kernel.kernel if isinstance(kernel, Plan) else kernel
        if not isinstance(loaded, LoadedKernel) or not loaded.active_set:
            raise HawkError("ActiveSet.planes: the kernel reads no active set (build it under Guard(active_set=True))")
        if self.mask is None and self.positions is None:
            raise HawkError("ActiveSet.planes: this set states a runner's compaction; give mask= or positions= to make planes")
        map_name, count_name = loaded.active_set
        group = int(loaded.declared["launch"]["lane_space"]["group"])
        entries = self._entries(group)
        capacity = max(entries.size * group, self._capacity(group))
        if self._planes is None:
            dtype = loaded.declared["arg_dtypes"][map_name]
            self._planes = {map_name: np.zeros(max(capacity, 1), dtype=dtype), count_name: np.zeros(1, dtype=np.uint32)}
        elif set(self._planes) != {map_name, count_name}:
            raise HawkError(f"ActiveSet.planes: the planes were made for a kernel reading {sorted(self._planes)}, "
                            f"this one reads {[map_name, count_name]}")
        made_map, made_count = self._planes[map_name], self._planes[count_name]
        if entries.size > made_map.size:
            raise HawkError(f"ActiveSet.planes: {entries.size} entries do not fit the {made_map.size} the map was made for")
        made_map[: entries.size] = entries
        made_count[0] = entries.size * group
        return dict(self._planes)

    def _entries(self, group: int):
        """The map's entries: the live positions of the mask in ascending order (their groups, for a group kernel), or the
        positions as stated."""
        import numpy as np

        if self.positions is not None:
            return np.asarray(self.positions).reshape(-1)
        live = np.flatnonzero(~np.asarray(self.mask, dtype=bool).reshape(-1))
        return np.unique(live // group) if group > 1 else live

    def _capacity(self, group: int) -> int:
        """The entries the map is made with: one per lane of the mask, rounded up to whole groups, because the launch's count
        is in lanes and the library refuses a count above the map's length."""
        import numpy as np

        if self.mask is None:
            return 0
        return -(-np.asarray(self.mask).size // group) * group

    def __repr__(self) -> str:
        """The set as stated."""
        stated = [f"{name}={value!r}" for name, value in (("every", self.every), ("reorder", self.reorder)) if value is not None]
        if self.mask is not None:
            stated.append("mask=...")
        if self.positions is not None:
            stated.append("positions=...")
        return f"ActiveSet({', '.join(stated)})"


@dataclass(frozen=True)
class RunReport:
    """What one :meth:`Runner.run` did, as the library counted it.

    ``n`` samples, ``finished`` of them done, ``launches`` of the loop and ``steps`` the loop ran (``exact_steps``: the
    count is exact), the ``compactions`` and ``reorders`` of the set, the samples ``mapped`` over all launches, ``done``
    (every sample finished) and the wall clock ``wall_s`` of the call.
    """

    n: int
    finished: int
    launches: int
    steps: int
    compactions: int
    reorders: int
    mapped: int
    done: bool
    exact_steps: bool
    wall_s: float


class GradientReport(NamedTuple):
    """How a :class:`Gradients` was made: ``bitwise`` says, for each plane returned, whether its sums over samples are reproducible
    bit for bit (``False`` for a plane fed by a cross-sample scatter-add, whose summing order varies from run to run), and
    ``chunks`` the passes over the memory the library made."""

    bitwise: dict
    chunks: int = 1


class Gradients(Mapping):
    """The gradients of one reverse pass, read-only, by plane name: one row per sample for a per-sample plane, in sample order,
    and a 0-d array, the sum over samples, for a shared parameter. :attr:`report` is the :class:`GradientReport`."""

    __slots__ = ("_values", "report")

    def __init__(self, values: dict, report: GradientReport) -> None:
        """Hold ``values`` (``{plane name: array}``) and the ``report`` of how they were made."""
        self._values = dict(values)
        self.report = report

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"<hawk.Gradients {sorted(self._values)}>"


#: The words of ``policy=`` of a reverse and what the library does for each: it makes the sums over samples of a gradient in one
#: deterministic compensated order, which ``"compensated"`` and ``"auto"`` ask for. ``"ordered"`` asks for a plain sum in sample
#: order, which the library does not do.
_POLICIES = MappingProxyType({"compensated": True, "auto": True, "ordered": False})


def check_reverse_options(strict, policy, who: str) -> bool:
    """``strict`` as a bool, after refusing by name a ``strict`` that is no bool and a ``policy`` the library does not do."""
    if not isinstance(strict, bool):
        raise HawkError(f"{who}: strict= is True or False, got {strict!r}")
    if policy not in _POLICIES:
        raise HawkError(f"{who}: policy={policy!r} is not one of {sorted(_POLICIES)}")
    if not _POLICIES[policy]:
        raise HawkError(f"{who}: policy={policy!r} asks for the sums over samples in plain sample order; the library sums them in "
                        "one deterministic compensated order ('compensated' or 'auto'), and strict=True refuses a gradient that "
                        "is not reproducible instead of reporting it")
    return strict


def _difference(latest: dict, earlier: dict) -> dict:
    """The counters a run added: ``latest`` less ``earlier`` (both as :meth:`BoundPlan.counters` answers), a counter the
    unit lacks staying ``None``."""
    finished = None if latest["finished"] is None else latest["finished"] - (earlier["finished"] or 0)
    mix = tuple(value - (earlier["mix"][i] if i < len(earlier["mix"]) else 0) for i, value in enumerate(latest["mix"]))
    return {"finished": finished, "steps": latest["steps"], "mix": mix}


def _release(door, handles: list) -> None:
    """Free the runner handles in ``handles``."""
    for handle in handles:
        door.runner_free(handle)
    handles.clear()


class Runner:
    """A loop that runs bound plans until every sample is done or ``max_steps`` is reached: what :func:`until_done` returns.

    The library owns the loop; the planes stay the caller's memory, written in place. A counter array the caller bound
    (``finished_count``) receives what each :meth:`run` finished. A runner made with ``record=True`` keeps what the last run
    needs for :meth:`reverse`.
    """

    def __init__(self, plans: list[Plan], bounds: list[BoundPlan], max_steps: int, active: ActiveSet | None, *,
                 layout: str | None = None, record: bool = False, window: int | None = None,
                 memory_budget: int | None = None, steps_reverse: str = "per_step") -> None:
        """Make the library's runner over ``bounds``, the plans of one step in order, bound from ``plans``; ``layout`` is the
        call's, which a reverse reads its planes by; ``record``, ``window``, ``memory_budget`` and ``steps_reverse`` are those
        of :func:`until_done`."""
        self._plans = plans
        self._bounds = bounds
        self._layout = layout
        self._door = bounds[0].door
        if any(bound.door is not self._door for bound in bounds):
            raise HawkError("Runner: the plans of one model run on one door, and their planes are in more than one memory; "
                            "bind them all in the same place")
        self._counted = [{"finished": 0, "steps": None, "mix": ()} for _ in bounds]
        options = {}
        if active is not None:
            options["every"] = active.every or 0
            if active.reorder is not None:
                options["flags"] = _osprey.EG_RUN_REORDER
                options["reorder_theta"] = float(active.reorder)
        if record:
            options["flags"] = options.get("flags", 0) | _osprey.EG_RUN_RECORD
        options["window"] = window or 0
        options["memory_budget"] = memory_budget or 0
        options["steps_reverse"] = _STEPS_REVERSE[steps_reverse]
        #: the derived units a recording takes, held for as long as the runner lives
        self._derived = []
        if record:
            kernel = plans[0].kernel
            self._derived = kernel.derived_units(kernel.derived_roles(steps_reverse), self._door)
            options["window"] = window or kernel.recorded_window()   # the twin keeps the window it was built with
            options["derived_roles"] = [role for role, _ in self._derived]
            options["derived_units"] = [unit for _, unit in self._derived]
        self._recorded = record
        self._handle = through_door(self._door.runner_new, [bound.handle for bound in bounds], max_steps, **options)
        #: the handle in a list the finalizer frees without holding this object
        self._handles = [self._handle]
        self._finalizer = weakref.finalize(self, _release, self._door, self._handles)

    @property
    def outputs(self) -> tuple:
        """The names of the output planes of the model, in order: what :func:`simulate` returns as its state."""
        return tuple(dict.fromkeys(name for plan in self._plans for name in plan.kernel.output_names()))

    @property
    def planes(self) -> dict:
        """The arrays bound, by name, as the caller gave them (a plane made for an omitted one included)."""
        return {name: value for bound in reversed(self._bounds) for name, value in bound.planes.items()}

    def run(self) -> RunReport:
        """Run until every sample is done or ``max_steps`` is reached and report what the library counted. A second run
        continues from where the first stopped; :meth:`reset` starts the batch over."""
        started = time.perf_counter()
        report = through_door(self._door.run, self._handle)
        through_door(self._door.sync)
        wall = time.perf_counter() - started
        for i, bound in enumerate(self._bounds):
            latest = bound.counters()
            bound.settle_counters(_difference(latest, self._counted[i]))
            self._counted[i] = latest
        return RunReport(n=report["n"], finished=report["finished"], launches=report["launches"], steps=report["steps"],
                         compactions=report["compactions"], reorders=report["reorders"], mapped=report["mapped"],
                         done=bool(report["flags"] & _osprey.EG_REPORT_DONE),
                         exact_steps=bool(report["flags"] & _osprey.EG_REPORT_EXACT_STEPS), wall_s=wall)

    def reset(self) -> None:
        """Put the loop back as the library made it: the done mask and the counters cleared, every sample live. The caller rewrites
        the state planes it wants run again."""
        through_door(self._door.runner_reset, self._handle)
        self._counted = [{"finished": 0, "steps": None, "mix": ()} for _ in self._bounds]

    def figures(self) -> dict:
        """The library's figures of the loop: its ``mode`` and, where the loop has one, the launch it resolved."""
        return through_door(self._door.runner_figures, self._handle)

    @property
    def record(self) -> RunRecord | None:
        """The :class:`RunRecord` of the last run of a runner made with ``record=True``, else ``None``."""
        return RunRecord(self) if self._recorded else None

    def reverse(self, cotangents: dict, *, strict: bool = False) -> Gradients:
        """The gradients of the last run, given the cotangents ``{state plane name: array}`` of the final state, as :class:`Gradients`.

        A cotangent has the layout of the state plane it belongs to. The result maps each plane the step's vjp differentiates
        with respect to (its ``wrt``) to its gradient: an array like the plane for a per-sample plane, in sample order, and a
        0-d array, the sum over samples, for a shared parameter. A runner that was not made with ``record=True``, or that has
        not run, is refused by the library by name. The report states, for each gradient, the library's one answer for the run:
        its sums over samples are the canonical compensated pair, so bitwise, in the passes it states; ``strict=True`` therefore
        refuses nothing here.
        """
        strict = check_reverse_options(strict, "compensated", "Runner.reverse")
        if not self._recorded:
            raise HawkError("Runner.reverse: this runner records nothing; make it with record=True")
        import numpy as np

        kernel = self._plans[0].kernel
        names = kernel.gradient_names()
        roles = {name: role for role, name in kernel.bind_spec}
        given = self.planes
        unknown = sorted((set(cotangents) | set(names)) - set(roles))
        if unknown:
            raise HawkError(f"Runner.reverse: {unknown} are not planes of the kernel {kernel.sidecar['kernel']!r}")
        seeds = {name: kernel.plane_view(name, value, self._layout) for name, value in cotangents.items()}
        gradients, grad_views = {}, {}
        for name in names:
            if roles[name] == "uniform":
                gradients[name] = np.zeros(1, dtype=kernel.declared["arg_dtypes"][name])
                grad_views[name] = gradients[name]
                gradients[name] = gradients[name].reshape(())
            else:
                gradients[name] = np.zeros(tuple(describe(given[name]).shape), dtype=kernel.declared["arg_dtypes"][name])
                grad_views[name] = kernel.plane_view(name, gradients[name], self._layout)
        door, wrapped = self._door, []
        try:
            sides = []
            for views in (seeds, grad_views):
                planes = [through_door(door.plane_wrap, view) for view in views.values()]
                wrapped += planes
                sides.append(([kernel.slot(name) for name in views], planes))
            through_door(door.reverse, self._handle, *sides)
        finally:
            for plane in wrapped:
                door.plane_free(plane)
        figures = self.figures()
        return Gradients(gradients, GradientReport(dict.fromkeys(gradients, bool(figures["reverse_bitwise"])),
                                                   max(int(figures["reverse_passes"]), 1)))

    def close(self) -> None:
        """Free the library's runner and plans at once. Closing twice is harmless."""
        self._finalizer()
        for bound in self._bounds:
            bound.close()


class RunRecord:
    """What the library keeps of a recorded runner's last run, the thing :func:`hawk.reverse` differentiates.

    ``runner`` is the :class:`Runner` it belongs to; reversing after the runner is closed is refused.
    """

    def __init__(self, runner: Runner) -> None:
        """Hold the runner the record belongs to."""
        self.runner = runner


def _plans_of(model, record: bool = False, window: int | None = None) -> list[Plan]:
    """The plans of ``model`` in order: a :class:`~hawk.plans.Plan` as it is, a kernel deployed with the others in one
    bundle (recordable, with ``window`` steps, when ``record``)."""
    items = list(model) if isinstance(model, (list, tuple)) else [model]
    if not items:
        raise HawkError("the model is empty; pass a plan, a kernel, or a list of them")
    for item in items:
        if isinstance(item, LoadedKernel):
            raise HawkError("the model is a loaded kernel; plan it first (hawk.plan(loaded)) or pass the kernel itself")
    kernels = [item for item in items if not isinstance(item, Plan)]
    deployed = iter(deploy(kernels, record=record, window=window if record else None) if kernels else ())
    return [item if isinstance(item, Plan) else next(deployed) for item in items]


def _bind_model(plans: list[Plan], planes: dict, n_samples, layout) -> list[BoundPlan]:
    """Bind every plan over the one namespace ``planes``: each takes the names it declares, and what an earlier plan made
    (an omitted mask or output) is the same array in the next."""
    unknown = sorted(set(planes) - set().union(*(plan.kernel.declared_names() for plan in plans)))
    if unknown:
        raise HawkError(f"{unknown} are not planes any plan of the model declares")
    namespace, bounds = dict(planes), []
    try:
        for plan in plans:
            names = plan.kernel.declared_names()
            bound = plan.bound({name: value for name, value in namespace.items() if name in names},
                               n_samples=n_samples, layout=layout, keep=False, outputs=True, running=True)
            bounds.append(bound)
            namespace |= bound.planes   # a plane made for an omitted one is the same array in the next plan
    except BaseException:
        for bound in bounds:
            bound.close()
        raise
    return bounds


def _recording_options(record, window, memory_budget, steps_reverse) -> None:
    """Refuse an option of a recorded run that is not of its kind, by name."""
    if not isinstance(record, bool):
        raise HawkError(f"record= is True or False, got {record!r}")
    for name, value in (("window", window), ("memory_budget", memory_budget)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise HawkError(f"{name}= is a positive whole number or None (the library's choice), got {value!r}")
    if steps_reverse not in _STEPS_REVERSE:
        raise HawkError(f"steps_reverse= is one of {sorted(_STEPS_REVERSE)}, got {steps_reverse!r}")


def until_done(model, *, max_steps: int, active: ActiveSet | None = None, n_samples: int | None = None,
               layout: str | None = None, record: bool = False, window: int | None = None,
               memory_budget: int | None = None, steps_reverse: str = "per_step", **planes) -> Runner:
    """Bind ``model`` and make the loop that runs it until every sample is done; ``.run()`` of the result runs it.

    ``model`` is a plan, a kernel or a list of them run as one step in order (see the module). ``planes`` are bound by
    the rules of :meth:`hawk.plans.Plan.bind`, except the finished counter and the active set, which the library makes; an
    omitted ``terminated`` mask starts all-false and an omitted output is made, zeroed. ``max_steps`` caps the steps a
    sample may take. ``active`` states the compaction of a kernel built under ``Guard(active_set=True)``.

    ``record=True`` keeps what :func:`hawk.reverse` needs of each run, for a model of one plan built recordable
    (``hawk.deploy(..., record=True)``; a kernel passed here is deployed so). ``window`` is the steps between kept states, which
    the build fixes: ``None`` is the window built, and another is refused by the library by name. ``memory_budget`` is the bytes
    the record may hold (``None``: the library's choice), over which the reverse runs the samples in passes; both are refused
    without ``record``. ``steps_reverse`` is ``"per_step"``, ``"fused"`` (one K-step unit for K steps, for a kernel taking a fixed
    K, built with window K) or ``"auto"`` (the library picks from the kernel's measured table).
    """
    _recording_options(record, window, memory_budget, steps_reverse)
    if active is not None and (active.mask is not None or active.positions is not None):
        raise HawkError("until_done: a runner makes its own active set; pass ActiveSet(every=, reorder=), or bind "
                        "ActiveSet(mask=...).planes(kernel) to a plain plan")
    plans = _plans_of(model, record, window)
    bounds = _bind_model(plans, planes, n_samples, layout)
    try:
        return Runner(plans, bounds, max_steps, active, layout=layout, record=record, window=window,
                      memory_budget=memory_budget, steps_reverse=steps_reverse)
    except BaseException:
        for bound in bounds:
            bound.close()
        raise


#: The names a result states itself; a state plane of one of them is reached through ``state``.
_RESULT_FIELDS = ("state", "report", "record", "n", "finished", "done", "steps", "status", "wall_s")


@dataclass(frozen=True)
class SimResult:
    """What :func:`simulate` returns: the state planes and the report of the run.

    A state plane is an attribute of the result by its name (``result.x``) and an item of ``state``. ``finished`` samples
    of ``n``, ``steps`` the loop ran, ``wall_s`` seconds; ``status`` is ``"done"``, or ``"max_steps"`` when the cap is
    reached first (a result, not an error). ``record`` is the :class:`RunRecord` of a run made with ``record=True``, else
    ``None``.
    """

    state: dict
    report: RunReport
    record: RunRecord | None = None

    def __getattr__(self, name: str):
        """A state plane by its name."""
        try:
            return self.__dict__["state"][name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def n(self) -> int:
        """The samples run."""
        return self.report.n

    @property
    def finished(self) -> int:
        """The samples done at the end."""
        return self.report.finished

    @property
    def done(self) -> bool:
        """Whether every sample finished."""
        return self.report.done

    @property
    def steps(self) -> int:
        """The steps the loop ran."""
        return self.report.steps

    @property
    def status(self) -> str:
        """``"done"`` or ``"max_steps"``."""
        return "done" if self.report.done else "max_steps"

    @property
    def wall_s(self) -> float:
        """The seconds the run took."""
        return self.report.wall_s


def simulate(model, *, max_steps: int, active: ActiveSet | None = None, n_samples: int | None = None,
             layout: str | None = None, record: bool = False, window: int | None = None,
             memory_budget: int | None = None, steps_reverse: str = "per_step", **planes) -> SimResult:
    """Run ``model`` until every sample is done or ``max_steps`` is reached and return the state.

    The model, the planes, ``active`` and the recording options are those of :func:`until_done`. The state is every output
    plane of the model, the caller's own array where it bound one and a zeroed one where it left one out. With
    ``record=True`` the result's ``record`` goes to :func:`hawk.reverse` and keeps the library's runner alive until the
    result is collected.
    """
    runner = until_done(model, max_steps=max_steps, active=active, n_samples=n_samples, layout=layout, record=record,
                        window=window, memory_budget=memory_budget, steps_reverse=steps_reverse, **planes)
    try:
        report = runner.run()
        bound = runner.planes
        state = {name: bound[name] for name in runner.outputs if name in bound}
    except BaseException:
        runner.close()
        raise
    clash = sorted(set(state) & set(_RESULT_FIELDS))
    if clash or not record:
        runner.close()
    if clash:
        raise HawkError(f"hawk.simulate: the planes {clash} are names of the result; read them from the arrays you bound")
    return SimResult(state=state, report=report, record=runner.record)
