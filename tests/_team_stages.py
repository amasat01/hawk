# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Run a split kernel with team stages on the device, stage by stage.

eagle launches a team stage with one block per lane (the launch of the stage's sidecar
``team.threads``); until it does, this driver makes that launch itself: each stage is bound
through an eagle plan (which packs the arguments), and a team stage is then launched through the
driver with ``grid = lanes`` and ``block = threads``. The scratch planes are the kernel's own
(sidecar ``team_planes``), made here once per call.
"""

from __future__ import annotations

import ctypes

import _deploy as L
import numpy as np
from conftest import sidecar_of
from hawk._lane_bind import check_scatter_status, target_size, team_grid, team_units


def _cudart():
    import re

    path = re.search(r"\S*libcudart\S*\.so\S*", open("/proc/self/maps").read())
    lib = ctypes.CDLL(path.group(0) if path else "libcudart.so")
    lib.cudaGetLastError.restype = ctypes.c_int
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    return lib


def check_launch(name: str) -> None:
    """Raise, by name, if the last CUDA call failed: a launch refused for resources (local memory,
    registers) leaves its error in the runtime and the stream sync alone reports nothing."""
    lib = _cudart()
    code = lib.cudaGetLastError()
    if code:
        raise RuntimeError(f"kernel {name!r}: the launch failed ({lib.cudaGetErrorString(code).decode()}, "
                           f"cuda error {code}); the planes it was to write are unchanged")


def run_device(bundle, name: str, planes: dict, lanes: int, *, scalar=np.float64, scratch: dict | None = None, only: int | None = None) -> dict:
    """Run kernel ``name`` of ``bundle`` over ``lanes`` lanes on the device, ``planes`` being the
    cupy arrays of its declared planes. Returns the scratch planes (cupy) by name; ``scratch`` hands in
    the planes of an earlier run to run over them; ``only`` stops after that many stages (an aborted launch)."""
    import cupy as cp
    import eagle.exec as eexec
    from cupy.cuda import driver
    from eagle import plan as eplan

    first = sidecar_of(bundle, name)
    scratch = scratch or {n: cp.full(s.get("per_lane", 1) * team_units(first, s, lanes, planes), s.get("fill", 0),
                          dtype=np.int32 if s["dtype"] == "int32" else scalar)
               for n, s in (first.get("team_planes") or {}).items()}
    for fresh, spec in (first.get("team_planes") or {}).items():
        if "fill" in spec:              # a plane that holds its fill at the start of every run
            scratch[fresh].fill(spec["fill"])
    checks = []
    for record in (first.get("stages") or [{"kernel": name}])[:only]:
        sidecar = dict(sidecar_of(bundle, record["kernel"]))
        sidecar.pop("stages", None)
        plugin = L.device_plugin(bundle.directory, record["kernel"], sidecar)
        names = {n for _, n in sidecar["arg_spec"]}
        args = {k: v for k, v in {**planes, **scratch}.items() if k in names}
        axes = (first.get("launch") or {}).get("lane_space", {}).get("axes")
        if axes and axes[0]["label"] and axes[0]["extent"] is None and record.get("team") is None:
            # a Grid's runtime first axis: a stage that binds no lane-sized plane is told its extent
            args[axes[0]["label"]] = lanes // int(np.prod([a["extent"] for a in axes[1:]]))
        bound = eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**args)
        team = record.get("team")
        if team is None:
            bound.launch()
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception as exc:
                raise RuntimeError(f"stage {record['kernel']}: {exc}") from exc
            check_launch(record["kernel"])
            continue
        stream = int(cp.cuda.get_current_stream().ptr)
        grid = team_grid(first, team, lanes, planes)
        if grid == 0:
            continue
        if team.get("grid") == "chunks":
            checks.append(team)
        tail = (ctypes.c_longlong * 3)(0, grid, lanes)
        addrs = list(bound._addrs) + [ctypes.addressof(tail) + 8 * k for k in range(3)]
        params = (ctypes.c_void_p * len(addrs))(*addrs)
        driver.launchKernel(bound._entry, grid, 1, 1, team["threads"], 1, 1, 0, stream,
                            ctypes.addressof(params), 0)
        cp.cuda.get_current_stream().synchronize()
        check_launch(record["kernel"])
    cp.cuda.get_current_stream().synchronize()
    for team in checks:
        check_scatter_status(name, team, scratch[team["status"]].get(), target_size(planes, team["of"]))
    return scratch
