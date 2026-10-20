# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Helpers the autodiff card shares: the median/IQR statistic, the
device/software/compile facts a card's header carries, and the
no-absolute-path check every card passes before it is written.

The card imports this file BY PATH (this directory is not a package).
Nothing here imports a GPU library (``cupy``, ``torch``) at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess

#: Schema tag every card's JSON carries (bump the suffix on a
#: breaking layout change; a reader checks this field, never the filename).
SCHEMA = "raptor-perf-card/1"

# --------------------------------------------------------------------------- #
# Statistics and small facts
# --------------------------------------------------------------------------- #
def median_iqr(values):
    """Median and interquartile range (inclusive quartiles), with the raw
    per-repetition samples alongside -- a card keeps BOTH, never the
    aggregate alone (a scalar is blind to a bimodal run)."""
    values = list(values)
    if len(values) < 2:
        v = values[0] if values else None
        return {"median": v, "iqr": 0.0, "q1": v, "q3": v, "samples": values}
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "iqr": q[2] - q[0],
            "q1": q[0], "q3": q[2], "samples": values}


def md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def version_of(module):
    try:
        return __import__(module).__version__
    except Exception:  # not installed: the arm cannot have run
        return None


def find_absolute_paths(value, _key=""):
    """[(dotted key path, value), ...] for every string anywhere inside
    ``value`` (recursively, through dicts/lists/tuples) that is an absolute
    filesystem path. A real card run refuses to write while this is
    non-empty."""
    hits = []
    if isinstance(value, str):
        if os.path.isabs(value):
            hits.append((_key or "<root>", value))
    elif isinstance(value, dict):
        for k, v in value.items():
            hits.extend(find_absolute_paths(v, f"{_key}.{k}" if _key else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            hits.extend(find_absolute_paths(v, f"{_key}[{i}]"))
    return hits


def cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_commit(module, short=False):
    """``<commit>[-dirty]`` of the repository holding ``module``, or None."""
    root = pathlib.Path(module.__file__).resolve().parent
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse",
                              *(["--short"] if short else []), "HEAD"],
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
        if rev.returncode == 0:
            return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def host_compile_facts():
    """The EXACT host code-generation recipe hawk's own compile toolchain
    uses for every host-compiled kernel in these cards (``hawk.compile.
    toolchain.host_codegen_flags``) -- recorded verbatim here rather than
    re-asserted, so a card never has to guess what optimisation level
    produced its numbers. The effective ``opt_level`` (``hawk.compile.
    toolchain.opt_level``, user-controllable via ``$HAWK_OPT_LEVEL``) is
    recorded next to it: it is what the host flags' ``-O<n>`` now comes
    from, and it ALSO changes a device (nvcc) build's own flags the same
    way (``-O<n>`` for nvcc's host side, ``-Xptxas -O<n>`` for the device
    side) -- there used to be no explicit ``-O`` there at all (nvcc's own
    default), which is why this used to be a note rather than a flag; NVRTC
    device compiles always optimise and ignore ``opt_level`` entirely."""
    try:
        from hawk.compile import toolchain
    except Exception as exc:  # hawk not importable: the arm cannot have run
        return {"host_profile": None, "host_flags": None, "error": repr(exc)}
    profile = toolchain.host_profile()
    level = toolchain.opt_level()
    return {
        "host_profile": profile,
        "host_flags": toolchain.host_codegen_flags(profile, level),
        "opt_level": level,
        "device_flags_note": (
            f"hawk.compile.toolchain.device_flags now carries the same "
            f"opt_level explicitly: -{level} for nvcc's own host-side code "
            f"generation and -Xptxas -{level} for ptxas; NVRTC device "
            "compiles always optimise and ignore opt_level"),
    }


def software(modules=("numpy", "torch", "cupy")):
    """Versions and commits of the cards' software."""
    import hawk

    out = {"python": platform.python_version(),
          **{m: version_of(m) for m in modules},
          "hawk": getattr(hawk, "__version__", None),
          "hawk_commit": git_commit(hawk),
          "platform": platform.platform(), "cpu": cpu_model(),
          "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    try:
        import eagle
        out["eagle_commit"] = git_commit(eagle)
    except Exception:
        out["eagle_commit"] = None
    out["compile"] = host_compile_facts()
    return out


def write_card(card: dict, json_path: pathlib.Path, md_text: str, md_path: pathlib.Path):
    """Write ``card`` (checked absolute-path-free) and its rendered markdown.
    Raises, writing neither file, if any value in ``card`` is an absolute
    path (the generic half of the "no absolute path in a card" gate)."""
    hits = find_absolute_paths(card)
    if hits:
        raise ValueError(f"card carries an absolute path: {hits}")
    json_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    md_path.write_text(md_text)


def utc_now_iso():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def device_facts():
    """``({"name": ...}, peak)`` of the visible CUDA device through cupy, or
    ``(None, None)`` with no GPU or no cupy. Peak figures are not derived
    here: these cards report walls and ratios, not a fraction of peak."""
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() == 0:
            return None, None
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    except Exception:
        return None, None
    return {"name": name.decode() if isinstance(name, bytes) else str(name)}, None


