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

import hawk
from hawk.artifact import build_bundle
from hawk.errors import ToolchainAbsent

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
    """The plan of ``name`` bound over ``planes``: a host row through hawk's own verbs, a device row through eagle's."""
    if target == "host":
        return hawk.plan(hawk.load(bundle.directory, name)).bind(**planes)
    import _deploy as L
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.device_plugin(bundle.directory, name, sidecar_of(bundle, name))
    return eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes)


def _xp(target):
    if target == "host":
        return np
    try:
        import cupy
    except ModuleNotFoundError as absent:
        if absent.name != "cupy":
            raise
        raise ToolchainAbsent(f"a device row needs cupy: {absent}", toolchain="cupy") from absent
    return cupy


__all__ = ["BOTH", "sidecar_of"]


def sum_bound(terms, dtype=np.float64) -> float:
    """Largest gap between two orderings of one sum of ``terms`` in ``dtype``.

    Each ordering is within ``(k-1)u*sum|x|`` of the exact sum (``k`` terms, ``u = eps/2``).
    """
    magnitude = np.abs(np.asarray(terms, dtype=np.float64))
    return max(magnitude.size - 1, 0) * np.finfo(dtype).eps * float(magnitude.sum())


def fd_bound(step, magnitude) -> float:
    """Roundoff of a central difference of step ``step``: two evaluations each carry
    ``eps*magnitude``, divided by ``2*step``. Exact for a quadratic, which has no truncation."""
    return np.finfo(np.float64).eps * float(magnitude) / step


def assert_within(got, want, bound):
    """``|got - want| <= bound`` element by element, ``bound`` a scalar or one value per element."""
    gap = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64))
    assert (gap <= bound).all(), f"gap {gap} beyond the bound {bound}"


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


@functools.lru_cache(maxsize=None)
def toolchains() -> dict:
    """The host-feature probes of the rows that declare ``needs_toolchain``, once per session: ``name -> (present, what)``.

    No compiler is probed here: a missing compiler reaches a row as the ``ToolchainAbsent`` an OSPREY's ``machine`` answers.
    """
    x86 = platform.machine().lower() in ("x86_64", "amd64")
    linux = sys.platform.startswith("linux")
    return {
        "x86-64": (x86, f"host is {platform.machine()}, not x86-64"),
        "linux": (linux and os.path.isdir("/proc"), f"host is {sys.platform}: no /proc, ulimit -v address cap, ldd or dynamic symbol table"),
        "linux-tools": (linux and shutil.which("ldd") is not None, "ldd is not on PATH"),
        "avx512": (avx512(), "no AVX-512"),
    }


def absent_toolchain(exc: BaseException) -> str | None:
    """The absent toolchain this failure is the typed refusal for, else ``None``.

    Only by type: the product's :class:`~hawk.errors.ToolchainAbsent` (its ``toolchain`` attribute names which)
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
