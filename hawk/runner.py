# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Running a model until every sample is done: :func:`until_done` builds the loop, :func:`simulate` runs it and returns
the state.

hawk binds the planes by the rules of :meth:`hawk.plans.Plan.bind` and states how the live samples are compacted
(:class:`ActiveSet`); the run itself is the host library's: its picks, its compaction, its counting of steps and finished
samples. The model is a :class:`~hawk.plans.Plan`, a kernel (deployed first) or a list of them run as one step, in order,
over one namespace of planes: a name is one plane in every plan that declares it.

Not offered yet: recording a run and reversing it.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass

import _osprey

from . import ospreys
from .errors import HawkError
from .plans import Plan, bind_for_run, deploy
from .runtime import BoundPlan, HostKernel, through_door
from .targets import host_target

__all__ = ["ActiveSet", "RunReport", "Runner", "SimResult", "simulate", "until_done"]


@dataclass(frozen=True)
class ActiveSet:
    """How a runner compacts the live samples of a kernel built under ``Guard(active_set=True)``.

    ``every`` is the compaction cadence in steps (``None``: the library's choice) and ``reorder`` the locality threshold
    ``theta`` at which it also reorders the bound planes (``None``: never). The library makes and keeps the set itself;
    both are refused by it, with its reason, for a model that reads none.
    """

    every: int | None = None
    reorder: float | None = None

    def __post_init__(self) -> None:
        """Refuse a cadence that is not a whole number and a threshold that is not a number."""
        if self.every is not None and (isinstance(self.every, bool) or not isinstance(self.every, int)):
            raise HawkError(f"ActiveSet(every=...) is a whole number of steps or None, got {self.every!r}")
        if self.reorder is not None and (isinstance(self.reorder, bool) or not isinstance(self.reorder, (int, float))):
            raise HawkError(f"ActiveSet(reorder=...) is a locality threshold or None, got {self.reorder!r}")


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
    (``finished_count``) receives what each :meth:`run` finished.
    """

    def __init__(self, plans: list[Plan], bounds: list[BoundPlan], max_steps: int, active: ActiveSet | None) -> None:
        """Make the library's runner over ``bounds``, the plans of one step in order, bound from ``plans``."""
        self._plans = plans
        self._bounds = bounds
        self._door = ospreys.serving(host_target().id).runtime_door()
        self._counted = [{"finished": 0, "steps": None, "mix": ()} for _ in bounds]
        options = {}
        if active is not None:
            options["every"] = active.every or 0
            if active.reorder is not None:
                options["flags"] = _osprey.EG_RUN_REORDER
                options["reorder_theta"] = float(active.reorder)
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

    def close(self) -> None:
        """Free the library's runner and plans at once. Closing twice is harmless."""
        self._finalizer()
        for bound in self._bounds:
            bound.close()


def _plans_of(model) -> list[Plan]:
    """The plans of ``model`` in order: a :class:`~hawk.plans.Plan` as it is, a kernel deployed with the others in one
    bundle."""
    items = list(model) if isinstance(model, (list, tuple)) else [model]
    if not items:
        raise HawkError("the model is empty; pass a plan, a kernel, or a list of them")
    for item in items:
        if isinstance(item, HostKernel):
            raise HawkError("the model is a loaded kernel; plan it first (hawk.plan(loaded)) or pass the kernel itself")
    kernels = [item for item in items if not isinstance(item, Plan)]
    deployed = iter(deploy(kernels) if kernels else ())
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
            bound = bind_for_run(plan, {name: value for name, value in namespace.items() if name in names},
                                 n_samples, layout)
            bounds.append(bound)
            # a step word belongs to the plan that reads it; every other plane is one array across the step
            namespace |= {name: value for name, value in bound.planes.items() if name != "fused_steps"}
    except BaseException:
        for bound in bounds:
            bound.close()
        raise
    return bounds


def until_done(model, *, max_steps: int, active: ActiveSet | None = None, n_samples: int | None = None,
               layout: str | None = None, **planes) -> Runner:
    """Bind ``model`` and make the loop that runs it until every sample is done; ``.run()`` of the result runs it.

    ``model`` is a plan, a kernel or a list of them run as one step in order (see the module). ``planes`` are bound by
    the rules of :meth:`hawk.plans.Plan.bind`, except the finished counter and the active set, which the library makes; an
    omitted ``terminated`` mask starts all-false and an omitted output is made, zeroed. ``max_steps`` caps the steps a
    sample may take. ``active`` states the compaction of a kernel built under ``Guard(active_set=True)``. A run cannot be
    recorded for a reverse pass yet.
    """
    plans = _plans_of(model)
    bounds = _bind_model(plans, planes, n_samples, layout)
    try:
        return Runner(plans, bounds, max_steps, active)
    except BaseException:
        for bound in bounds:
            bound.close()
        raise


#: The names a result states itself; a state plane of one of them is reached through ``state``.
_RESULT_FIELDS = ("state", "report", "n", "finished", "done", "steps", "status", "wall_s")


@dataclass(frozen=True)
class SimResult:
    """What :func:`simulate` returns: the state planes and the report of the run.

    A state plane is an attribute of the result by its name (``result.x``) and an item of ``state``. ``finished`` samples
    of ``n``, ``steps`` the loop ran, ``wall_s`` seconds; ``status`` is ``"done"``, or ``"max_steps"`` when the cap is
    reached first (a result, not an error).
    """

    state: dict
    report: RunReport

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
             layout: str | None = None, **planes) -> SimResult:
    """Run ``model`` until every sample is done or ``max_steps`` is reached and return the state.

    The model, the planes and ``active`` are those of :func:`until_done`. The state is every output plane of the model, the
    caller's own array where it bound one and a zeroed one where it left one out. A run cannot be recorded for a reverse
    pass yet.
    """
    runner = until_done(model, max_steps=max_steps, active=active, n_samples=n_samples, layout=layout, **planes)
    try:
        report = runner.run()
        bound = runner.planes
        state = {name: bound[name] for name in runner.outputs if name in bound}
    finally:
        runner.close()
    clash = sorted(set(state) & set(_RESULT_FIELDS))
    if clash:
        raise HawkError(f"hawk.simulate: the planes {clash} are names of the result; read them from the arrays you bound")
    return SimResult(state=state, report=report)
