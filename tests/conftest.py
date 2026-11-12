# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Session fixtures for the artifact rows.

ONE built bundle per (kernel, kind, scalar mode) shape, in a session tmpdir,
under ONE session cache directory — so the content-closure cache is exercised
across the whole session and no row pays for another's compile. Every row that
LOADS or RUNS an artifact takes ``built``; rows that only read text take the
piece they need.
"""

from __future__ import annotations

import os
import pathlib
import platform
import threading

import _deployable as D
import numpy as np
import pytest

from hawk.artifact import build_bundle
from raptor.protocols.refusal import ToolchainAbsent
from hawk.targets import TARGETS

BOTH = ("cuda", "host")

#: ``(bundle name, kernels, kwargs)`` — the fixture set rows ///
#: / build between them.
_UNITS = (
    ("axpb", [D.axpb], {}),
    ("vec3", [D.vec3_scale], {}),
    ("vec3_f32", [D.vec3_scale], {"mode": "float32"}),
    ("mat", [D.mat_apply], {}),
    ("multi", [D.two_outputs], {}),
    ("gather", [D.gather], {}),
    ("energy", [D.energy], {}),
    ("speed", [D.speed], {}),
    ("scatter", [D.scatter], {}),
    ("fraction", [D.fraction], {}),
    ("spin", [D.spin], {}),
    ("two_wire", [D.two_wire], {}),
    ("vocab", [D.vocab], {}),
    ("scatter_c_plain", [D.scatter_c], {}),
    ("scatter_c_comp", [D.scatter_c], {"kind": D.COMPENSATED}),
    ("diag_guarded", [D.diagnostic], {}),
    ("diag_free", [D.diagnostic], {"kind": D.MASK_FREE}),
    ("diag_one_mask", [D.diagnostic_two], {}),
    ("diag_two_masks", [D.diagnostic_two], {"kind": D.TWO_MASKS}),
)


#: The MPI bed is EXCLUDED from the default collection, exactly as eagle's is:
#: its rows only mean anything under `mpirun -np 2`, they raise (never skip) when
#: their artifact is unset, and initialising MPI inside a session that is also
#: driving CUDA is how an interpreter dies at finalize. `tests/mpi/
#: check_hawk_rank_bed.sh` is the ONE way to run them, and it names the directory
#: directly — pytest's own rule for initial arguments, which this hook does not
#: apply to.
collect_ignore = ["mpi"]


@pytest.fixture
def spec():
    """The build spec of a host and device build on this machine (device arch ``sm_61``): what a test that calls a pass outside a build passes."""
    from hawk.targets import build_spec

    return build_spec(("host", "cuda"), "sm_61")


@pytest.fixture
def host_spec():
    """The build spec of a host-only build on this machine."""
    from hawk.targets import build_spec

    return build_spec(("host",))


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("hawk_compile_cache"))


@pytest.fixture(scope="session")
def built(tmp_path_factory, cache_dir) -> dict:
    """Every fixture bundle, built once. ``{name: Bundle}``.

    Host side under the EXACT ``native`` profile: the rows that run these
    bundles compare the host results with NumPy references and the serial
    oracle bit for bit, which only the exact profiles promise (the default
    fast profile contracts ``a*b + c`` into FMAs)."""
    root = tmp_path_factory.mktemp("hawk_artifacts")
    out = {}
    for name, kernels, kw in _UNITS:
        out[name] = build_bundle(kernels, pathlib.Path(root) / name,
                                 targets=BOTH, cache_dir=cache_dir,
                                 host_profile="native", **kw)
    return out


def pytest_collection_modifyitems(config, items):
    """Skip ``wall_clock`` rows (measured run times) unless HAWK_WALL_CLOCK_ROWS is set."""
    if os.environ.get("HAWK_WALL_CLOCK_ROWS"):
        return
    skip = pytest.mark.skip(reason="measured-time row; set HAWK_WALL_CLOCK_ROWS=1")
    for item in items:
        if "wall_clock" in item.keywords:
            item.add_marker(skip)


def sidecar_of(bundle, name: str) -> dict:
    """The named artifact's sidecar out of a built bundle."""
    return next(a.sidecar for a in bundle.artifacts if a.name == name)


def declared(kernel, arrays: dict) -> dict:
    """``arrays`` cut to the planes ``kernel`` declares. A derivative binds what its body reads (a primal
    input, or a seed, that its body does not read is no slot of it), and a run refuses a name that is none."""
    names = {n for _, n in kernel.arg_spec}
    return {k: v for k, v in arrays.items() if k in names}


def _launch(bundle, name, target, planes):
    import _deploy as L
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(**planes)


def _xp(target):
    if target == "host":
        return np
    import cupy

    return cupy


__all__ = ["BOTH", "sidecar_of"]
assert TARGETS  # the targets table is what `targets` names


def run_threads(fns):
    """Start every callable at once; return ``(results, errors)`` by index."""
    barrier = threading.Barrier(len(fns))
    results, errors = [None] * len(fns), [None] * len(fns)

    def work(i):
        barrier.wait()
        try:
            results[i] = fns[i]()
        except BaseException as exc:                      # noqa: BLE001
            errors[i] = exc

    pool = [threading.Thread(target=work, args=(i,)) for i in range(len(fns))]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return results, errors


def avx512() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo") as f:
            return "avx512f" in f.read()
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------------------------------
# The gate's outcome for an absent toolchain.
#
# A row whose failure is the product's own refusal for a toolchain this machine lacks is reported `toolchain-absent` (letter T):
# its own count, not passed, failed, error or skipped. It exits early (the refusal is raised by the first thing that needs the
# toolchain, in the test or in the fixture that builds for it); a session with only passed and toolchain-absent rows exits 0, and
# "N toolchain-absent" is always printed. Every other failure, including one in a row that needs an absent toolchain but fails
# for another reason, stays failed.
# --------------------------------------------------------------------------------------------------------------------------
import functools
import shutil
import sys

_ABSENT = "toolchain-absent"


def _on_path(name_or_path: str | None) -> bool:
    return bool(name_or_path) and (os.path.isfile(name_or_path) or shutil.which(name_or_path) is not None)


@functools.lru_cache(maxsize=None)
def toolchains() -> dict:
    """The probes, once per session, cheap (no compile): ``name -> (present, what)``."""
    x86 = platform.machine().lower() in ("x86_64", "amd64")
    linux = sys.platform.startswith("linux")
    try:
        from cuda.bindings import nvrtc as _nvrtc        # noqa: F401
        nvrtc = True
    except Exception:                                    # noqa: BLE001 - any failure to load means absent
        nvrtc = False
    nvcc = any(_on_path(c) for c in (os.environ.get("HAWK_NVCC"), "nvcc"))
    cppcheck = _on_path(os.environ.get("HAWK_CPPCHECK") or "cppcheck")
    device = linux and (os.path.exists("/dev/nvidiactl") or shutil.which("nvidia-smi") is not None)
    return {
        "cuda-nvrtc": (nvrtc, "cuda python bindings with NVRTC are not loadable"),
        "nvcc": (nvcc, "nvcc is not on PATH ($HAWK_NVCC)"),
        "cppcheck": (cppcheck, "cppcheck is not on PATH ($HAWK_CPPCHECK)"),
        "cuda-device": (device, "no CUDA device"),
        "x86-64": (x86, f"host is {platform.machine()}, not x86-64"),
        "linux": (linux and os.path.isdir("/proc"), f"host is {sys.platform}: no /proc, ulimit -v address cap, ldd or dynamic symbol table"),
        "linux-tools": (linux and shutil.which("ldd") is not None, "ldd is not on PATH"),
        "avx512": (avx512(), "no AVX-512"),
        "aether-src": (os.path.isdir(os.environ.get("W25_AETHER_SRC") or ""), "$W25_AETHER_SRC is unset or not a directory (the aether source tree)"),
    }


def absent_toolchain(exc: BaseException) -> str | None:
    """The absent toolchain this failure is the typed refusal for, else ``None``.

    Only by type: the product's :class:`~raptor.protocols.refusal.ToolchainAbsent` (its ``toolchain`` attribute names which)
    anywhere in the cause chain. A row that asserts on a tool the product does not itself refuse for declares it with
    ``@pytest.mark.needs_toolchain("<name>")``, which raises the same type from setup when the probe says it is absent.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ToolchainAbsent):
            return exc.toolchain or "unknown"
        exc = exc.__cause__ or exc.__context__
    return None


def pytest_runtest_setup(item):
    for mark in item.iter_markers("needs_toolchain"):
        name = mark.args[0]
        ok, what = toolchains()[name]
        if not ok:
            raise ToolchainAbsent(f"{item.name} needs {name}: {what}", toolchain=name)


def pytest_report_header(config):
    return [f"toolchain absent: {name}: {what}" for name, (ok, what) in toolchains().items() if not ok]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.outcome == "failed" and call.excinfo is not None and not hasattr(rep, "wasxfail"):
        name = absent_toolchain(call.excinfo.value)
        if name:
            rep.outcome = _ABSENT
            rep.toolchain = name


def pytest_report_teststatus(report):
    if report.outcome == _ABSENT:
        return _ABSENT, "T", "TOOLCHAIN-ABSENT"
    return None


def pytest_terminal_summary(terminalreporter):
    rows = terminalreporter.stats.get(_ABSENT, [])
    by: dict = {}
    for r in rows:
        terminalreporter.write_line(f"TOOLCHAIN-ABSENT {r.nodeid} - {r.toolchain}")
        by[r.toolchain] = by.get(r.toolchain, 0) + 1
    terminalreporter.write_line(f"{len(rows)} {_ABSENT}" + (" (" + ", ".join(f"{k}={v}" for k, v in sorted(by.items())) + ")" if by else ""))
