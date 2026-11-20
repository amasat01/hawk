# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The verbs that plan, bind and launch a loaded host kernel, deploy a kernel, and move arrays in and out of it.

A :class:`Plan` is a loaded kernel (:func:`hawk.load`) under a stated structure and partitioning; nothing is decided here
about how it runs. :meth:`Plan.bind` checks the caller's planes exactly as :func:`hawk.run` does and returns a
:class:`~hawk.runtime.BoundPlan`; :meth:`Plan.run` binds, launches once and returns the outputs. The tiles, threads and
memory of a launch are the host library's.

Only the host runtime is in this build: :data:`DeviceKernel` and a device target are refused by name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import _osprey

from . import ospreys
from .errors import HawkError
from .runtime import BoundPlan, HostKernel, load, through_door
from .targets import TARGETS, host_target

__all__ = ["DeviceKernel", "HostSerial", "HostTeam", "Partition", "Plan", "asarray", "banded_pack", "banded_unpack", "deploy",
           "plan", "synchronize", "to_numpy"]

#: What a refusal of anything that needs a device says.
_NO_DEVICE = "the device runtime is not in this build"

#: The DLPack device types a host array lives on: ordinary host memory and pinned host memory.
_HOST_DEVICE_TYPES = (1, 3)


@dataclass(frozen=True)
class Structure:
    """How a plan launches its partitions: a marker the caller names, never a decision about hardware.

    ``code`` is the host library's structure code, ``None`` for a structure this build does not run.
    """

    name: str
    code: int | None

    def __repr__(self) -> str:
        return f"hawk.{self.name}"


#: The host library's team of workers over contiguous tiles of the range.
HostTeam = Structure("HostTeam", _osprey.EG_STRUCTURE_HOST_TEAM)
#: One serial call per partition: the reference every other structure is judged against.
HostSerial = Structure("HostSerial", _osprey.EG_STRUCTURE_HOST_SERIAL)
#: One device launch per partition; refused by name until the device runtime is in this build.
DeviceKernel = Structure("DeviceKernel", None)


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


class Plan:
    """A loaded kernel under a structure and a partitioning: what :func:`plan` and :func:`deploy` return.

    The sample count belongs to the planes, so the library's plan is made at :meth:`bind`, once the count is known.
    """

    def __init__(self, kernel: HostKernel, structure: Structure, npartitions: int | None,
                 partitions: tuple | None) -> None:
        """Keep the stated structure and partitioning; ``npartitions`` and ``partitions`` are never both given."""
        self.kernel = kernel
        self.structure = structure
        self.npartitions = npartitions
        self.partitions = partitions

    def _ranges(self, n_samples: int) -> dict:
        """The partitioning in the host library's words for a plan over ``n_samples``: its ``part_base`` and
        ``part_count``, its ``split_evenly``, or nothing (the whole range)."""
        if self.partitions is not None:
            wrong = {p.n_samples for p in self.partitions if p.n_samples != n_samples}
            if wrong:
                raise HawkError(f"{self.kernel.path.name}: the partitions are over n_samples={sorted(wrong)[0]}, "
                                f"the planes hold {n_samples}")
            return {"part_base": [p.base for p in self.partitions], "part_count": [p.count for p in self.partitions]}
        return {} if self.npartitions is None else {"split_evenly": self.npartitions}

    def _is_partial(self, n_samples: int) -> bool:
        """Whether a launch covers less than the whole range in one piece."""
        if self.partitions is not None:
            return [(p.base, p.count) for p in self.partitions] != [(0, n_samples)]
        return self.npartitions not in (None, 1)

    def _bound(self, arguments, n_samples, layout) -> BoundPlan:
        """The plan over ``arguments``, bound; ``n_samples`` and ``layout`` as the caller stated them."""
        n = arguments.n_samples
        self.kernel.refuse_partial_stages(self._is_partial(n))
        return BoundPlan(self.kernel, arguments, self.structure.code, self._ranges(n), n_samples=n_samples, layout=layout)

    def bind(self, /, *, n_samples: int | None = None, layout: str | None = None, **planes) -> BoundPlan:
        """Bind the caller's own planes by name and return the :class:`~hawk.runtime.BoundPlan` to launch.

        The checks are those of :func:`hawk.run`, which every argument failing at once is named in one refusal; the planes are
        the caller's memory, written in place, and an ``Index`` plane of another width than the kernel declares is refused
        because a copy would go stale. ``n_samples`` (default: the count every per-sample plane agrees on) and ``layout``
        are as in :func:`hawk.run`. ``self`` is positional-only so a plane may be named ``self``.
        """
        return self._bound(self.kernel.arguments(planes, n_samples, layout, keep=True), n_samples, layout)

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
        names = self.kernel.output_names()
        if not names:
            raise HawkError(f"{self.kernel.path.name}: this kernel declares no output plane, so run() has nothing to return")
        bound = self._bound(self.kernel.arguments(planes, n_samples, layout, keep=False, outputs=True), n_samples, layout)
        try:
            bound.launch()
            outputs = {name: bound.planes[name] for name in names if name in bound.planes}
        finally:
            bound.close()
        if returns == "numpy":
            outputs = {name: to_numpy(value) for name, value in outputs.items()}
        return next(iter(outputs.values())) if len(outputs) == 1 else outputs


def plan(loaded: HostKernel, structure: Structure = HostTeam, *, npartitions: int | None = None,
         partitions=None) -> Plan:
    """Plan a loaded kernel (:func:`hawk.load`) under ``structure``: :data:`HostTeam` (default) or :data:`HostSerial`.

    ``npartitions=k`` splits the range evenly into ``k`` partitions; ``partitions`` states them as
    :class:`Partition` values instead (never both). Neither: the whole range, one partition.
    """
    if isinstance(loaded, Plan):
        raise HawkError("hawk.plan takes a loaded kernel (hawk.load); a plan, deployed or not, already carries its structure")
    if not isinstance(loaded, HostKernel):
        raise HawkError(f"hawk.plan takes a loaded kernel (hawk.load), got {type(loaded).__name__}")
    if not isinstance(structure, Structure):
        raise HawkError(f"structure={structure!r} is none of hawk.HostTeam, hawk.HostSerial, hawk.DeviceKernel")
    if structure.code is None:
        raise HawkError(f"hawk.{structure.name}: {_NO_DEVICE}")
    if npartitions is not None and partitions is not None:
        raise HawkError("npartitions= and partitions= are two ways to partition the range; give one")
    if npartitions is not None and (not isinstance(npartitions, int) or npartitions < 1):
        raise HawkError(f"npartitions must be a positive integer, got {npartitions!r}")
    if partitions is not None:
        partitions = tuple(partitions)
        if not partitions or not all(isinstance(p, Partition) for p in partitions):
            raise HawkError("partitions= is a non-empty sequence of hawk.Partition(base, count, n_samples)")
    return Plan(loaded, structure, npartitions, partitions)


def deploy(kernels, *, targets=None, scalar_type: str = "float64", cache_dir=None, contraction=None):
    """Build one kernel, or a list of them as one bundle, and return its :class:`Plan`, or a tuple of plans in order.

    A repeat deploy of an unchanged kernel compiles nothing (the build is cached under ``cache_dir``, default
    :func:`hawk.artifact.default_cache_dir`). ``targets`` defaults to the host; a device target is refused. ``scalar_type`` is
    ``"float64"`` or ``"float32"``; ``"banded"`` is a device type and is refused on the host. The plan runs under
    :data:`HostTeam` and takes no ``structure=``: :func:`plan` over a loaded kernel is the call that states one.
    """
    from .artifact import build_bundle, build_key, default_cache_dir
    from .build_settings import BuildSettings

    single = not isinstance(kernels, (list, tuple))
    kernels = [kernels] if single else list(kernels)
    ids = _host_targets(targets)
    if scalar_type == "banded":
        raise HawkError("scalar_type='banded' is a device type; the host runs float64 or float32")
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    key = build_key(kernels, targets=ids, mode=scalar_type)
    directory = root / "deploy" / hashlib.sha256(key.encode()).hexdigest()
    bundle = build_bundle(kernels, directory, mode=scalar_type, targets=ids, cache_dir=cache_dir,
                          settings=BuildSettings(contraction=contraction))
    built = {artifact.name for artifact in bundle.artifacts}
    plans = []
    for kernel in kernels:
        if kernel.name not in built:
            raise HawkError(f"{kernel.name!r} builds as several units ({sorted(built)}); build it with hawk.build and "
                            "plan each unit of the bundle after hawk.load")
        plans.append(Plan(load(bundle.directory, kernel.name), HostTeam, None, None))
    return plans[0] if single else tuple(plans)


def _host_targets(targets) -> tuple:
    """The target ids to build: the host's when ``targets`` is ``None``; a device target is refused by name."""
    if targets is None:
        return (host_target().id,)
    ids = tuple(ospreys.target_id(t) for t in ([targets] if isinstance(targets, str) else targets))
    for target in ids:
        served = TARGETS.get(target)
        if served is not None and not served.host_entry:
            raise HawkError(f"target {target!r}: {_NO_DEVICE}")
    return ids


def to_numpy(x):
    """``x``, an array of host memory from NumPy or any DLPack producer, as a NumPy array: the same memory, never a copy.
    A device array is refused by name."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if not hasattr(x, "__dlpack__"):
        raise HawkError(f"hawk.to_numpy: {type(x).__name__} is not an array (it exports no DLPack)")
    if x.__dlpack_device__()[0] not in _HOST_DEVICE_TYPES:
        raise HawkError(f"hawk.to_numpy: {type(x).__name__} lives on a device; {_NO_DEVICE}")
    return np.from_dlpack(x)


def asarray(x):
    """``x`` as a NumPy array on the host: an array or DLPack producer shares its memory, a sequence or a number is copied."""
    import numpy as np

    if hasattr(x, "__dlpack__") and not isinstance(x, np.ndarray):
        return to_numpy(x)
    return np.asarray(x)


def synchronize() -> None:
    """Return when every launch made so far has finished. A host launch has finished when it returns; this is the call a
    program that also runs on a device keeps in the same place."""
    through_door(ospreys.serving(host_target().id).runtime_door().sync)


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
