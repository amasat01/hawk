# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The verbs that plan, bind and launch a loaded host kernel, deploy a kernel, and move arrays in and out of it.

A :class:`Plan` is a loaded kernel (:func:`hawk.load`) under a stated structure and partitioning; nothing is decided here
about how it runs. Every run form ends in :meth:`Plan.bound`: :meth:`Plan.bind` returns the
:class:`~hawk.runtime.BoundPlan` to launch, :meth:`Plan.run` binds, launches once and returns the outputs, and calling the
plan (``plan(x, a=3)``, which a kernel's own call ``k(x, a=3)`` is) does so with the kernel's own signature. A plan
:func:`deploy` made keeps its kernel, so a per-lane input of a ``Grid`` kernel may be given broadcast: the plan builds, once
per pattern, the kernel's twin (:mod:`hawk._broadcast`) and binds the caller's buffer compact. The tiles, threads and
memory of a launch are the host library's.

Where a plan runs is where its arrays live. Every array crosses by the DLPack protocol, NumPy's and any other library's, host
memory or device memory alike, and the plan's door is that of the memory the arrays are in (:mod:`hawk._place`): the host
library's for host memory, the device library's for device memory, one launch never mixing the two. An output the caller
leaves out is made by that door and handed back by the same protocol: a NumPy array for host inputs, an array of the library the
caller's own arrays come from for device inputs. A launch over device memory is queued; the one-call forms (:meth:`Plan.run`,
calling the plan) wait before they return, and :meth:`~hawk.runtime.BoundPlan.launch` leaves it to :func:`synchronize`.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import _osprey

from . import _broadcast, _place, graph_builder, ospreys
from .errors import HawkError
from .runtime import BoundPlan, LoadedKernel, load, through_door
from .targets import TARGETS, host_target, runs_here
from .trace.kernel import Kernel

__all__ = ["DeviceKernel", "HostSerial", "HostTeam", "Partition", "Plan", "asarray", "banded_pack", "banded_unpack", "deploy", "plan", "synchronize", "to_numpy"]


@dataclass(frozen=True)
class Structure:
    """How a plan launches its partitions: a marker the caller names, never a decision about hardware.

    ``code`` is the door's structure code.
    """

    name: str
    code: int

    def __repr__(self) -> str:
        return f"hawk.{self.name}"


#: The host library's team of workers over contiguous tiles of the range.
HostTeam = Structure("HostTeam", _osprey.EG_STRUCTURE_HOST_TEAM)
#: One serial call per partition: the reference every other structure is judged against.
HostSerial = Structure("HostSerial", _osprey.EG_STRUCTURE_HOST_SERIAL)
#: One device launch per partition, on the device library's door.
DeviceKernel = Structure("DeviceKernel", _osprey.EG_STRUCTURE_DEVICE)


@dataclass(frozen=True)
class Partition:
    """The samples ``[base, base + count)`` of a range of ``n_samples``.

    For a kernel that computes several lanes per thread the positions are threads, and for a kernel that reads an index set
    they are positions of the set; the host library says which and refuses a partition outside the range.
    """

    base: int
    count: int
    n_samples: int

    @classmethod
    def whole(cls, n_samples: int) -> Partition:
        """The one partition that covers all ``n_samples``."""
        return cls(0, n_samples, n_samples)


@dataclass(frozen=True)
class _Deployed:
    """What a plan built by :func:`deploy` keeps of its build: the traced ``kernel``, the ``settings`` it was deployed with
    (which its twins are deployed with) and the ``bundle`` digest that names its unit to a derivative of it."""

    kernel: Kernel
    settings: dict
    bundle: str


class Plan:
    """A loaded kernel under a structure and a partitioning: what :func:`plan` and :func:`deploy` return.

    The sample count belongs to the planes, so the library's plan is made at :meth:`bind`, once the count is known.
    """

    def __init__(self, kernel: LoadedKernel, structure: Structure, npartitions: int | None,
                 partitions: tuple | None, *, deployed: _Deployed | None = None) -> None:
        """Keep the stated structure and partitioning; ``npartitions`` and ``partitions`` are never both given. ``deployed``
        is set by :func:`deploy` only: a plan over a loaded artifact has no source kernel."""
        self.kernel = kernel
        self.structure = structure
        self.npartitions = npartitions
        self.partitions = partitions
        self._deployed = deployed
        self._twins: dict = {}
        self._lock = threading.Lock()

    def _ranges(self, n_samples: int, span: tuple | None = None) -> dict:
        """The partitioning in the host library's words for a plan over ``n_samples``: its ``part_base`` and
        ``part_count``, its ``split_evenly``, or nothing (the whole range). ``span`` is ``(base, count)`` of one launch over
        part of the range, which replaces the plan's own partitioning."""
        if span is not None:
            base, count = span
            if self._is_partial(n_samples, span):
                return {"part_base": [base], "part_count": [n_samples - base if count is None else count]}
            return {}
        if self.partitions is not None:
            wrong = {p.n_samples for p in self.partitions if p.n_samples != n_samples}
            if wrong:
                raise HawkError(f"{self.kernel.path.name}: the partitions are over n_samples={sorted(wrong)[0]}, "
                                f"the planes hold {n_samples}")
            return {"part_base": [p.base for p in self.partitions], "part_count": [p.count for p in self.partitions]}
        return {} if self.npartitions is None else {"split_evenly": self.npartitions}

    def _is_partial(self, n_samples: int, span: tuple | None = None) -> bool:
        """Whether a launch covers less than the whole range in one piece."""
        if span is not None:
            base, count = span
            return base != 0 or count not in (None, n_samples)
        if self.partitions is not None:
            return [(p.base, p.count) for p in self.partitions] != [(0, n_samples)]
        return self.npartitions not in (None, 1)

    def _twin(self, patterns: dict) -> Plan:
        """The plan of the kernel's twin for ``patterns`` (:mod:`hawk._broadcast`), deployed with this plan's settings on the
        first use of the pattern and kept; a repeat compiles nothing, the build being cached on disk."""
        key = tuple(sorted(patterns.items()))
        with self._lock:
            held = self._twins.get(key)
            if held is None:
                twin = _broadcast.twin(self._deployed.kernel, patterns)
                held = self._twins[key] = deploy(twin, **self._deployed.settings)
        return held

    def _reversible(self) -> Plan:
        """This plan when it carries the vjp of its kernel or has no kernel to take it from, else the plan of the same kernel built
        with it, made on the first use and kept: what a graph kept for :func:`hawk.reverse` launches."""
        deployed = self._deployed
        if deployed is None or deployed.settings["reverse"] or deployed.settings["record"]:
            return self
        with self._lock:
            held = self._twins.get("reverse")
            if held is None:
                held = self._twins["reverse"] = deploy(deployed.kernel, **{**deployed.settings, "reverse": True})
        return held

    def bound(self, planes: dict, *, n_samples: int | None = None, layout: str | None = None, keep: bool,
              outputs: bool = False, running: bool = False, span: tuple | None = None) -> BoundPlan:
        """Check ``planes`` and bind them: the one path every run form takes (:meth:`bind`, :meth:`run`, a call of the plan,
        :func:`hawk.run`, a runner's bind).

        A plan from :func:`deploy` first classifies the read-only per-lane inputs by shape: when one is given broadcast the
        planes bind to that pattern's twin, compact. The checks are those of :meth:`~hawk.runtime.LoadedKernel.arguments`
        (``keep``, ``outputs`` and ``running`` as there); ``span=(base, count)`` launches over part of the range.
        """
        recorder = graph_builder.current()
        if recorder is not None and recorder.reverse and self._reversible() is not self:
            return self._reversible().bound(planes, n_samples=n_samples, layout=layout, keep=keep, outputs=outputs,
                                            running=running, span=span)
        deployed = self._deployed
        place = self._place(planes)
        plan, patterns, shown = self, {}, None
        if deployed is not None:
            patterns, shown, planes = _broadcast.routed(deployed.kernel, planes, deployed.kernel.name)
            if patterns:
                plan = self._twin(patterns)
                taken = {name for _role, name in plan.kernel.arg_spec}     # a derivative linear in an input does not take it
                planes = {name: value for name, value in planes.items() if name in taken or name not in patterns}
        arguments = plan.kernel.arguments(planes, n_samples, layout, keep=keep, make=place, outputs=outputs, running=running,
                                          shown=shown)
        n = arguments.n_samples
        plan.kernel.refuse_partial_stages(plan._is_partial(n, span))
        route = None if deployed is None else (deployed.kernel, patterns)
        return BoundPlan(plan.kernel, arguments, plan._structure_code(place.device), plan._ranges(n, span), place=place,
                         n_samples=n_samples, layout=layout, broadcast=route,
                         store=None if recorder is None else recorder.store(place.door))

    def _place(self, planes: dict) -> _place.Maker:
        """The maker, and with it the door, of the memory ``planes`` live in: the host's when none is an array. Planes in two
        places are refused by name."""
        found = _place.place_of(planes, self.kernel.path.name)
        return _place.Maker.at(_place.HOST, None) if found is None else _place.Maker.at(found[0], found[1])

    def _structure_code(self, device: tuple) -> int:
        """The door's structure code of a launch over memory on ``device``: the plan's own, which a plan from :func:`deploy`
        replaces by the device structure when its planes are in device memory."""
        return DeviceKernel.code if self._deployed is not None and device != _place.HOST else self.structure.code

    def bind(self, /, *, n_samples: int | None = None, layout: str | None = None, **planes) -> BoundPlan:
        """Bind the caller's own planes by name and return the :class:`~hawk.runtime.BoundPlan` to launch.

        The checks are those of :func:`hawk.run`, which every argument failing at once is named in one refusal; the planes are
        the caller's memory, written in place, and an ``Index`` plane of another width than the kernel declares is refused
        because a copy would go stale. A plan from :func:`deploy` takes a read-only per-lane input of a ``Grid`` kernel
        broadcast (a scalar, axes of size 1, fewer axes) as the caller's own buffer, never copied; a plan made from a loaded
        artifact refuses the shape by name. ``n_samples`` (default: the count every per-sample plane agrees on) and
        ``layout`` are as in :func:`hawk.run`. ``self`` is positional-only so a plane may be named ``self``.
        """
        return self.bound(planes, n_samples=n_samples, layout=layout, keep=True)

    def _outputs(self, planes: dict, n_samples: int | None, layout: str | None) -> dict:
        """Bind ``planes``, launch once and return the outputs by name: the planes the kernel's sinks are, the ones the caller
        left out made, zeroed, in the shape and dtype the kernel declares. Inside a graph builder the launch is recorded into
        the graph instead, and the outputs are the arrays the recorded launches write."""
        bound = self.bound(planes, n_samples=n_samples, layout=layout, keep=False, outputs=True)
        recorder = graph_builder.current()
        if recorder is not None:                  # inside a builder a call is recorded, and the graph keeps the plan
            recorder.record(bound, self.kernel.sidecar["kernel"])
            return {name: bound.planes[name] for name in self.kernel.output_names() if name in bound.planes}
        try:
            bound.launch()
            bound.wait()
            return {name: bound.planes[name] for name in self.kernel.output_names() if name in bound.planes}
        finally:
            bound.close()

    def run(self, /, *, n_samples: int | None = None, layout: str | None = None, returns: str | None = None, **planes):
        """Bind, launch once and return the outputs: the output plane itself for a kernel with one, else a ``dict`` of
        every output by name.

        An output the caller leaves out is made, zeroed, in the shape and dtype the kernel declares (a plane whose length only
        the caller knows is refused: bind it); an output the caller binds is written in place and returned as given.
        ``returns="numpy"`` hands every output back as a NumPy array; ``None`` leaves each as it is. Other arguments as in
        :meth:`bind`.
        """
        if returns not in (None, "numpy"):
            raise HawkError(f"returns={returns!r}: 'numpy' is the one conversion this build has")
        if not self.kernel.output_names():
            raise HawkError(f"{self.kernel.path.name}: this kernel declares no output plane, so run() has nothing to return")
        outputs = self._outputs(planes, n_samples, layout)
        if returns == "numpy":
            outputs = {name: to_numpy(value) for name, value in outputs.items()}
        return next(iter(outputs.values())) if len(outputs) == 1 else outputs

    def __call__(self, /, *args, **kwargs):
        """Run the kernel once as a call to it, where the arrays live: ``plan(x, y, a=3)`` takes the planes in the order the
        kernel declares them (a ``Grid`` declaration is no plane), the rest by name, and a runtime extent by its label.

        Returns what the kernel body returns, lane-shaped; a kernel that returns nothing gives its one output plane, or a
        ``dict`` of them by name. An output left out is made, zeroed; an argument the kernel reads before it writes needs a
        value. Only a plan from :func:`deploy` has the kernel's signature, and its inputs bind as in :meth:`bind`, broadcast
        included; a plan made from a loaded artifact refuses the call by name.
        """
        deployed = self._deployed
        if deployed is None:
            raise HawkError(f"{self.kernel.path.name}: this plan was made from a loaded artifact, which carries no call "
                            "signature; bind its planes by name with plan.bind(...) or plan.run(...)")
        outputs = self._outputs(self._named(deployed.kernel, args, kwargs), None, None)
        tree = deployed.kernel.returns
        if tree is not None:
            return _filled(tree, outputs)
        return next(iter(outputs.values())) if len(outputs) == 1 else (outputs or None)

    def _named(self, kernel: Kernel, args: tuple, kwargs: dict) -> dict:
        """``{plane: value}`` of a call ``kernel(*args, **kwargs)``: ``args`` in the order the kernel declares its planes."""
        order = [name for name, declared in kernel.planes.items() if not getattr(declared, "grid", None)]
        if len(args) > len(order):
            raise HawkError(f"{kernel.name}(): {len(args)} positional arguments, but the kernel declares "
                            f"{len(order)} planes{'; pass them by name' if not order else ''}")
        given = dict(zip(order, args))
        twice = sorted(given.keys() & kwargs.keys())
        if twice:
            raise HawkError(f"{kernel.name}(): argument {twice[0]!r} is given both by position and by name")
        planes = {**given, **kwargs}
        prior = [m["name"] for m in self.kernel.declared.get("mutables") or () if m.get("prior") and m["name"] not in planes]
        if prior:
            raise HawkError(f"{kernel.name}(): {prior[0]!r} is updated from its previous value, so it needs an initial "
                            f"value: pass {prior[0]}=...")
        return planes


def _filled(tree, outputs: dict):
    """``tree`` (nested tuples, lists and dicts of plane names, or one name) with each name replaced by its output."""
    if isinstance(tree, str):
        return outputs[tree]
    if isinstance(tree, dict):
        return {key: _filled(value, outputs) for key, value in tree.items()}
    return type(tree)(_filled(value, outputs) for value in tree)


def plan(loaded: LoadedKernel, structure: Structure = HostTeam, *, npartitions: int | None = None,
         partitions=None) -> Plan:
    """Plan a loaded kernel (:func:`hawk.load`) under ``structure``: :data:`HostTeam` (default), :data:`HostSerial` or, over
    arrays in device memory, :data:`DeviceKernel`; the door refuses a structure that does not fit the memory.

    ``npartitions=k`` splits the range evenly into ``k`` partitions; ``partitions`` states them as
    :class:`Partition` values instead (never both). Neither: the whole range, one partition.
    """
    if isinstance(loaded, Plan):
        raise HawkError("hawk.plan takes a loaded kernel (hawk.load); a plan, deployed or not, already carries its structure")
    if not isinstance(loaded, LoadedKernel):
        raise HawkError(f"hawk.plan takes a loaded kernel (hawk.load), got {type(loaded).__name__}")
    if not isinstance(structure, Structure):
        raise HawkError(f"structure={structure!r} is none of hawk.HostTeam, hawk.HostSerial, hawk.DeviceKernel")
    if npartitions is not None and partitions is not None:
        raise HawkError("npartitions= and partitions= are two ways to partition the range; give one")
    if npartitions is not None and (not isinstance(npartitions, int) or npartitions < 1):
        raise HawkError(f"npartitions must be a positive integer, got {npartitions!r}")
    if partitions is not None:
        partitions = tuple(partitions)
        if not partitions or not all(isinstance(p, Partition) for p in partitions):
            raise HawkError("partitions= is a non-empty sequence of hawk.Partition(base, count, n_samples)")
    return Plan(loaded, structure, npartitions, partitions)


def deploy(kernels, *, targets=None, scalar_type: str = "float64", cache_dir=None, contraction=None, record: bool = False,
           window: int | None = None, reverse: bool = False):
    """Build one kernel, or a list of them as one bundle, and return its :class:`Plan`, or a tuple of plans in order.

    A repeat deploy of an unchanged kernel compiles nothing (the build is cached under ``cache_dir``, default
    :func:`hawk.artifact.default_cache_dir`). ``targets`` defaults to every target whose OSPREY says it runs on this machine; a
    stated target (or list) is exactly those, and one that does not run here is refused with its OSPREY's reason.
    ``scalar_type`` is ``"float64"``, ``"float32"`` or, for a device target, ``"banded"``. The plan takes no ``structure=``: it
    runs under :data:`HostTeam` over arrays in host memory and :data:`DeviceKernel` over arrays in device memory; :func:`plan`
    over a loaded kernel is the call that states one. The plan keeps the kernel and these settings: calling it runs the kernel (``k(...)`` is ``deploy(k)(...)``), and a per-lane input given
    broadcast builds the kernel's twin for that shape, once, with the same settings.

    ``record=True`` builds every kernel recordable (:func:`hawk.until_done`, :func:`hawk.reverse`): in the same library as the
    taping twin that keeps ``window`` steps (default 16) and the vjp of its step. A recorded run takes that window, so it is
    built in; ``window`` without ``record`` is refused.

    ``reverse=True`` builds every kernel with its vjp in the same library, which :func:`hawk.reverse` needs for a launch of the
    kernel in a graph kept with ``record=True``; a kernel called inside a builder of such a graph is deployed so on its own.
    It is not combined with ``record``.
    """
    from .artifact import build_bundle, build_key, default_cache_dir
    from .build_settings import BuildSettings

    single = not isinstance(kernels, (list, tuple))
    kernels = [kernels] if single else list(kernels)
    ids = _targets_here(targets)
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    if window is not None and not record:
        raise HawkError("deploy: window= is the steps a recorded run keeps, so it goes with record=True")
    asked = (16 if window is None else window) if record else None
    settings = {"targets": ids, "scalar_type": scalar_type, "cache_dir": cache_dir, "contraction": contraction,
                "record": record, "window": window, "reverse": reverse}
    primals = _primal_units(kernels, settings) or None
    key = build_key(kernels, targets=ids, mode=scalar_type, record=asked, derivative=primals, reverse=reverse)
    directory = root / "deploy" / hashlib.sha256(key.encode()).hexdigest()
    bundle = build_bundle(kernels, directory, mode=scalar_type, targets=ids, cache_dir=cache_dir,
                          settings=BuildSettings(contraction=contraction), record=asked, derivative=primals, reverse=reverse)
    built = {artifact.name for artifact in bundle.artifacts}
    plans = []
    for kernel in kernels:
        if kernel.name not in built:
            raise HawkError(f"{kernel.name!r} builds as several units ({sorted(built)}); build it with hawk.build and "
                            "plan each unit of the bundle after hawk.load")
        plans.append(Plan(load(bundle.directory, kernel.name), HostTeam, None, None,
                          deployed=_Deployed(kernel, settings, bundle.digest)))
    return plans[0] if single else tuple(plans)


def _primal_units(kernels: list, settings: dict) -> dict:
    """``{derivative: (primal name, primal unit)}`` for each derivative of ``kernels`` whose primal is not among them: the
    primal is deployed first, in a bundle of its own, and its unit is what the derivative names."""
    names = {kernel.name for kernel in kernels}
    named = {}
    for kernel in kernels:
        primal = getattr(kernel.sinks, "source", None)
        if isinstance(primal, Kernel) and primal.name not in names and kernel.sinks.primal_unit is None:
            named[kernel.name] = (primal.name, deploy(primal, **settings)._deployed.bundle)
    return named


def _targets_here(targets) -> tuple:
    """The target ids to build and run: every target whose OSPREY says it runs here when ``targets`` is ``None``, else exactly
    the stated ones. An OSPREY that does not run here is refused with its own reason, which is carried as it states it."""
    if targets is None:
        states = {target: runs_here(target) for target in TARGETS}
        ids = tuple(target for target, (runs, _why) in states.items() if runs)
        if not ids:
            raise HawkError("no installed OSPREY runs here: " + "; ".join(f"{t}: {why}" for t, (_runs, why) in states.items()))
        return ids
    ids = tuple(ospreys.target_id(t) for t in ([targets] if isinstance(targets, str) else targets))
    for target in ids:
        runs, why = runs_here(target)
        if not runs:
            raise HawkError(f"target {target!r} does not run here: {why}")
    return ids


def to_numpy(x):
    """``x``, an array from NumPy or any DLPack producer, as a NumPy array: the same memory for host memory, never a copy; an
    array in device memory is copied into host memory by its device's door."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if not hasattr(x, "__dlpack__"):
        raise HawkError(f"hawk.to_numpy: {type(x).__name__} is not an array (it exports no DLPack)")
    if _place.device_of(x) == _place.HOST:
        return np.from_dlpack(x)
    return through_door(_place.to_host, x)


def asarray(x, device: str | None = None):
    """``x`` as an array. ``device=None`` gives a NumPy array on the host: an array or DLPack producer shares its memory, a
    sequence or a number is copied.

    ``device="<target>:<ordinal>"`` (``"cuda:0"``) gives the array in the memory of that device, copied by the door of the
    OSPREY serving the target: an object that exports DLPack, which the caller's array library reads with its ``from_dlpack``.
    An array already in that memory is returned as it is.
    """
    import numpy as np

    if device is None:
        return to_numpy(x) if hasattr(x, "__dlpack__") and not isinstance(x, np.ndarray) else np.asarray(x)
    target, _, ordinal = device.partition(":")
    served = TARGETS.get(target)
    if served is None or not served.device or not (ordinal or "0").isdigit():
        raise HawkError(f"hawk.asarray: device={device!r} is not '<target>:<ordinal>' of a device target "
                        f"({', '.join(t.id for t in TARGETS.values() if t.device) or 'none is served'})")
    ordinal = int(ordinal or 0)
    here = _place.device_of(x)
    if here not in (None, _place.HOST):
        if through_door(_place.door_of, here) is through_door(_place.open_door, served.id, ordinal):
            return x
        raise HawkError(f"hawk.asarray: the array is on the DLPack device {here}, not on {device!r}")
    return through_door(_place.to_device, np.asarray(x), served.id, ordinal)


def synchronize() -> None:
    """Return when every launch made so far has finished, on every door in use. A host launch has finished when it returns; a
    launch over device memory that was queued (:meth:`~hawk.runtime.BoundPlan.launch`, :meth:`hawk.Graph.launch`) has when this
    returns."""
    for osprey in ospreys.found().values():
        for door in osprey.open_doors().values():
            through_door(door.sync)


def banded_pack(values):
    """The 8-byte banded cells (a ``uint64`` array of the same shape) of the float64 ``values``."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    door = ospreys.serving(host_target().id).runtime_door()
    cells = through_door(door.banded_pack, values.ravel().tolist())
    return np.array(cells, dtype=np.uint64).reshape(values.shape)


def banded_unpack(cells):
    """The float64 values (an array of the same shape) of the banded ``cells``, the inverse of :func:`banded_pack`."""
    import numpy as np

    cells = np.asarray(cells, dtype=np.uint64)
    door = ospreys.serving(host_target().id).runtime_door()
    values = through_door(door.banded_unpack, cells.ravel().tolist())
    return np.array(values, dtype=np.float64).reshape(cells.shape)
