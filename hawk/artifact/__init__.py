# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Manifest + sidecar + the ABI exports + bundles.

Nothing in this package imports eagle or raptor: :func:`plan_view`
projects a sidecar into the dict shape ``eagle.plan`` reads without
importing it, so a consumer can hand a HAWK artifact to eagle while HAWK
stays severed. :func:`plugins` assembles the whole plan-able object per
kernel the same way, loading the device side through a caller-supplied
loader rather than an eagle import.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

from ..ir import HawkError
from ..targets import ENTRY_DEVICE, ENTRY_HOST, TARGETS
from .bundle import (
    STAMP_NAME,
    Artifact,
    Bundle,
    Entry,
    arch,
    build,
    build_bundle,
    reset_unit_memo,
    unit_stats,
)
from .layout import LAYOUT_FIELDS, exports
from .manifest import (
    EXEC_ACCESS_CLASSES,
    EXEC_OPS,
    EXEC_TARGETS,
    TOP_LEVEL_KEY_ORDER_V2,
    write_manifest,
)
from .sidecar import PARAMS_SCHEMA

__all__ = ["Artifact", "Bundle", "Entry", "EXEC_ACCESS_CLASSES", "EXEC_OPS",
           "EXEC_TARGETS", "LAYOUT_FIELDS", "PARAMS_SCHEMA", "STAMP_NAME",
           "TOP_LEVEL_KEY_ORDER_V2", "arch", "build", "build_bundle", "exports",
           "plan_view", "plugins", "reset_unit_memo", "unit_stats",
           "write_manifest"]


def plan_view(sidecar: dict) -> dict:
    """The plan-time declaration a v2 consumer reads off a HAWK sidecar.

    Exactly the attributes ``eagle.plan`` looks for on a plugin —
    ``arg_spec``, ``exec_access``/``exec_op``, ``scalar_type``,
    ``arg_widths``, and the vector-/matrix-shaped ``mutable`` name sets
    — projected with zero eagle import, plus ``derivative`` (the
    sidecar's derivative tag, ``None`` for a primal). The consumer adds the loaded
    entry point; HAWK holds no launch path.

    ``arg_widths`` stays int-only, never ``None`` for a runtime-length
    role: ``eagle.plan``'s ``_declared_widths`` reads it through an
    unconditional ``int(v)``, including straight off a deployed sidecar,
    and a ``None`` there crashes that path. That fact lives in
    ``arg_shapes`` instead; ``arg_dtypes`` is passed through whole, for
    a consumer that wants to tell an integer-valued handle from a Real
    one."""
    mutables = sidecar.get("mutables", ())
    return {
        "arg_spec": tuple(tuple(pair) for pair in sidecar["arg_spec"]),
        "exec_access": sidecar["exec_access"],
        "exec_op": sidecar.get("exec_op"),
        "scalar_type": sidecar.get("scalar_type", "float64"),
        "arg_widths": dict(sidecar.get("arg_widths", {})),
        "arg_shapes": dict(sidecar.get("arg_shapes", {})),
        "arg_dtypes": dict(sidecar.get("arg_dtypes", {})),
        # The v2 ``params`` block verbatim: eagle.plan boxes an "int"
        # uniform as ``long long``.
        "params": tuple(dict(p) if isinstance(p, dict) else p
                        for p in sidecar.get("params", ())),
        "vec_mutables": frozenset(m["name"] for m in mutables
                                  if m.get("dtype") == "vector"),
        "mat_mutables": frozenset(m["name"] for m in mutables
                                  if m.get("dtype") == "matrix"),
        # a matrix output's (rows, cols): the head a one-sample result takes
        "mat_shapes": {m["name"]: tuple(int(e) for e in m["shape"])
                       for m in mutables if m.get("shape")},
        "kernel": sidecar["kernel"],
        # the launch shape and compaction group (``launch.lane_space``), as written
        "lane_space": sidecar.get("launch", {}).get("lane_space"),
        # a thread box's ``{axis, T, stride}``: base and count count threads
        **({"box": dict(sidecar["launch"]["box"])}
           if sidecar.get("launch", {}).get("box") else {}),
        # the exact threads per block an emit that fixed one needs (``launch.block``)
        **({"block": sidecar["launch"]["block"]} if sidecar.get("launch", {}).get("block") else {}),
        # the bound ``Sorted[Index]`` inputs and the named-axis sinks, when declared
        **({"sorted_inputs": tuple(sidecar["sorted_inputs"])}
           if sidecar.get("sorted_inputs") else {}),
        **({"plane_axes": dict(sidecar["plane_axes"])} if sidecar.get("plane_axes") else {}),
        # the companion planes the runtime owns (compensated keyed sums)
        **({"companions": dict(sidecar["companions"])} if sidecar.get("companions") else {}),
        **({"adjoint_like": dict(sidecar["adjoint_like"])} if sidecar.get("adjoint_like") else {}),
        **({"gather_mats": dict(sidecar["gather_mats"])} if sidecar.get("gather_mats") else {}),
        "host_entry": sidecar.get("host_entry"),
        # A finishing kernel's ``{mask, counter, steps}``, if present.
        **({"finish": dict(sidecar["finish"])} if "finish" in sidecar else {}),
        # One step's operation count, if the kernel steps.
        **({"step_ops": int(sidecar["step_ops"])} if "step_ops" in sidecar else {}),
        # One step's 64-bit floating-point operations in the IR, if the kernel
        # steps (``ops_per_step``: the same count under its earlier development name).
        **({"fp64_ops_ir": int(sidecar.get("fp64_ops_ir", sidecar.get("ops_per_step")))}
           if "fp64_ops_ir" in sidecar or "ops_per_step" in sidecar else {}),
        # The fast entries count samples finished on entry into the finish counter.
        **({"entry_counts_finished": True} if sidecar.get("entry_counts_finished") else {}),
        # A derivative kernel's ``{kind, wrt, primal, primal_unit}`` tag (its
        # sidecar ``derivative`` block), ``None`` for a primal: a launching
        # runtime refuses a prebuilt derivative at a forward-only door.
        "derivative": (dict(sidecar["derivative"], wrt=tuple(sidecar["derivative"]
                                                           .get("wrt", ())))
                       if sidecar.get("derivative") is not None else None),
    }


def fold_entries(cdll, sidecar: dict) -> dict:
    """``{"fold_entries": {target: address}}``: the host entry that folds each
    owned companion into its target (:func:`hawk.emit.wrapper.fold_entries`),
    for a kernel that has any; else ``{}``."""
    from ..emit.wrapper import fold_entry_name

    found = {target: ctypes.cast(getattr(cdll, fold_entry_name(sidecar["kernel"], target)
                                         + "_host"), ctypes.c_void_p).value
             for target in (sidecar.get("companions") or {}).values()}
    return {"fold_entries": found} if found else {}


def plugins(bundle: Bundle, *, device_loader=None) -> dict:
    """``{name: plugin}`` for every kernel of ``bundle``: the objects
    ``eagle.plan.plan`` and ``eagle.deploy`` take, assembled in one place.

    Each plugin is a :class:`types.SimpleNamespace` carrying the kernel's
    :func:`plan_view` declaration, its ``name``, and the entry points the
    bundle was built for: ``host_entry`` (a ``host`` target, after
    :class:`hawk._core.HostLibrary` checks the object's ABI tag and
    layout table) and ``device_function`` (a device target, only when
    ``device_loader`` is given).

    ``device_loader`` loads the bundle's device side, since hawk holds
    no device launch path: called once with the manifest path, it
    returns a by-name mapping whose entries carry ``fn.kernel.ptr``.
    Without one a plugin carries its host entry only.

    Everything an entry point points into is held on the plugin's
    ``_keepalive``, so it stays valid as long as the caller holds it.

    A kernel built only for an exported target such as ``c`` (a standalone C
    unit, :mod:`hawk.embedded`) has no entry point to plan and is refused,
    as :func:`hawk.load` refuses its directory."""
    from .. import _core

    def entry_of(art, kind):
        return next((e for t, e in art.entries.items()
                     if TARGETS[t].entry == kind), None)

    registry = None
    if device_loader is not None and any(entry_of(a, ENTRY_DEVICE)
                                         for a in bundle.artifacts):
        registry = device_loader(bundle.manifest_path)
    out = {}
    for art in bundle.artifacts:
        if not any(TARGETS[t].entry for t in art.entries):
            built = ", ".join(repr(t) for t in art.entries)
            raise HawkError(
                f"{art.name!r} was built for the {built} target only: it is a "
                "standalone C unit (<name>.c, <name>.h, <name>.embedded.json, "
                "see hawk.embedded) for an embedded build, with no host or "
                "device entry point to plan; build it with targets=('host',) "
                "or ('cuda',) for plugins")
        view = plan_view(art.sidecar)
        symbol = view.pop("host_entry", None)
        entries, keep = {}, []
        host = entry_of(art, ENTRY_HOST)
        if host is not None and symbol:
            library = _core.HostLibrary(str(host.artifact))   # refuses a mismatch
            cdll = ctypes.CDLL(str(host.artifact))
            entries["host_entry"] = ctypes.cast(getattr(cdll, symbol),
                                                ctypes.c_void_p).value
            entries.update(fold_entries(cdll, art.sidecar))
            keep += [library, cdll]
        if registry is not None and entry_of(art, ENTRY_DEVICE):
            loaded = registry[art.name]
            entries["device_function"] = loaded.fn.kernel.ptr
            keep += [registry, loaded]
        out[art.name] = SimpleNamespace(name=art.name, _keepalive=tuple(keep),
                                        **entries, **view)
    return out
