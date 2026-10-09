# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Writes the compile-time card: one row per consumer translation unit (a
hawk device kernel, a hawk host TU, and a hand-written eagle-side exemplar
compiled with the same recipe), each with its parse/instantiate/codegen
split (``g++ -ftime-report`` on the host, ``nvcc --time`` on the device) and
wall time. The numbers are written by the run into a md5-fenced file, and
every measured compile is a genuine cold one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import pytest

import _deployable as D
from _cards import card_path

from hawk.artifact.layout import exports
from hawk.compile import (
    aether_include,
    compiler_identity,
    device_compiler,
    eagle_include,
    host_compiler,
    host_flags,
)
from hawk.compile.toolchain import resolve_arch, device_flags, subprocess_env
from hawk.emit import BACKENDS, FLOAT64, render_source

CARD = card_path("CARD_COMPILE_TIME.md")

#: One kernel per CLASS the card reports: the smallest scalar body, a rank-1
#: chain, a compound-quantity read and the aether-vocabulary sweep.
CLASSES = ("axpb", "vec3_scale", "spin", "vocab")

_GPP_PHASE = re.compile(r"^\s*(phase [a-z. ]+|TOTAL)\s*:\s+([\d.]+)")


def _emit(name: str, backend: str, tmp: Path) -> Path:
    kernel = getattr(D, name)
    src = render_source(name, kernel.sinks, kernel.walk, BACKENDS[backend],
                        mode=FLOAT64, exports=exports(backend))
    path = tmp / f"{name}.{'cu' if backend == 'cuda' else 'cpp'}"
    path.write_text(src.text)
    return path


def _host_row(path: Path, tmp: Path) -> dict:
    argv = [host_compiler(), *host_flags(extra=("-ftime-report",)),
            str(path), "-o", str(tmp / f"{path.stem}.so")]
    start = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=subprocess_env())
    wall = time.perf_counter() - start
    assert done.returncode == 0, done.stderr[-2000:]
    phases = {}
    for line in done.stderr.splitlines():
        m = _GPP_PHASE.match(line)
        if m:
            phases[m.group(1).strip()] = float(m.group(2))
    return {"wall_s": round(wall, 3), "phases_s": phases}


def _device_row(path: Path, tmp: Path) -> dict:
    csv_path = tmp / f"{path.stem}.time.csv"
    argv = [device_compiler(), "--time", str(csv_path), *device_flags(),
            str(path), "-o", str(tmp / f"{path.stem}.ptx")]
    start = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=subprocess_env())
    wall = time.perf_counter() - start
    assert done.returncode == 0, done.stderr[-2000:]
    phases = {}
    if csv_path.is_file():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) >= 7 and row[6].strip().replace(".", "", 1).isdigit():
                    phases[row[1].strip()] = float(row[6])
    return {"wall_s": round(wall, 3), "phases_ms": phases}


#: The eagle-side exemplar: the hand-written aether-abi/2 fixture, compiled with
#: the SAME host recipe. It includes eagle's ABI header and NO aether, so it is
#: the floor a HAWK host TU's aether cost is read against.
_EAGLE_TU = Path(eagle_include()) / "tests" / "fixtures" / "host_plugin_execv2.cpp"


@pytest.mark.repo_local  # compiles the eagle repository's fixture TU
def test_the_compile_time_card_is_written(tmp_path):
    rows = {}
    for name in CLASSES:
        rows[f"hawk/{name}/host"] = _host_row(_emit(name, "host", tmp_path), tmp_path)
        rows[f"hawk/{name}/device"] = _device_row(_emit(name, "cuda", tmp_path),
                                                 tmp_path)
    assert _EAGLE_TU.is_file(), f"the eagle exemplar TU is missing: {_EAGLE_TU}"
    rows["eagle/host_plugin_execv2/host"] = _host_row(_EAGLE_TU, tmp_path)

    payload = {
        "card": "HAWK compile time (evidence)",
        "produced": time.strftime("%Y-%m-%d"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "host_compiler": compiler_identity(host_compiler()),
        "device_compiler": compiler_identity(device_compiler()),
        "device_arch": resolve_arch(),
        "aether_include": Path(aether_include()).name,
        "eagle_include": Path(eagle_include()).name,
        "note": ("every compile below is COLD (a fresh tmpdir, no cache slot); "
                 "the machine was shared at measurement time, so these are "
                 "ORDERS OF MAGNITUDE for a deferred decision, not a gate"),
        "rows": rows,
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    fence = hashlib.md5(body.encode()).hexdigest()
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(
        "# CARD_COMPILE_TIME — one card per consumer TU\n\n"
        "Written BY THE RUN (`tests/test_compile_time_card.py`). Prose cites this "
        "file; no number here may be restated without citing it. It DECIDES "
        "NOTHING — it is the evidence gating aether's header partition "
        " and whether the compile/cache driver moves to C++.\n\n"
        "Host phases come from `g++ -ftime-report` (seconds, wall column); device "
        "phases from `nvcc --time` (milliseconds, per compilation phase).\n\n"
        f"md5: `{fence}`\n\n```json\n{body}\n```\n"
    )
    assert CARD.is_file()
    text = CARD.read_text()
    assert f"md5: `{fence}`" in text
    assert hashlib.md5(text.split("```json\n")[1].split("\n```")[0]
                       .encode()).hexdigest() == fence


@pytest.mark.repo_local  # compiles the eagle repository's fixture TU
def test_the_card_reports_a_phase_split_for_every_row():
    """A card whose rows carry only a wall time cannot answer either deferred
    question, so the split is part of what the row observes."""
    assert CARD.is_file(), "run test_the_compile_time_card_is_written first"
    payload = json.loads(CARD.read_text().split("```json\n")[1].split("\n```")[0])
    for name, row in payload["rows"].items():
        assert row["wall_s"] > 0, name
        split = row.get("phases_s") or row.get("phases_ms")
        assert split, f"{name} carries no parse/instantiate/codegen split"
        assert len(split) >= 2, f"{name}: {split}"
