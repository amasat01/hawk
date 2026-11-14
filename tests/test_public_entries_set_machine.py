# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every public entry that can reach a lowering pass sets the machine descriptor itself.

The surface (hawk, ospreys.c): ``hawk.artifact.emit_source``, ``hawk.artifact.build_bundle``, ``aether.wings.base.wrapper.render_source``,
``aether.wings.base.render.render_body``, ``ospreys.c.render`` and ``ospreys.c.export``. Each runs on a tiny kernel in a fresh process, so no
test fixture and no earlier ``policy.using`` block supplies a machine."""

from __future__ import annotations

import json
import os
import subprocess
import sys

SCRIPT = """
import contextvars, json, sys
import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import build_bundle, emit_source
from aether.wings.base.render import render_body
from aether.wings.base.writer_aether import AetherWriter
from aether.wings.base.wrapper import render_source
from hawk.ext import DEFAULT_KIND
from hawk.targets import build_spec
from hawk.targets import TARGETS
import ospreys.c

@hawk.kernel
def entry_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b

k, work = entry_affine, sys.argv[1]
calls = {
    "hawk.artifact.emit_source": lambda: emit_source(k, targets=("host",)),
    "hawk.artifact.build_bundle": lambda: build_bundle([k], work + "/bundle", targets=("host",), cache_dir=work + "/cache"),
    "hawk.emit.render_source": lambda: render_source(k.name, tuple(k.sinks), k.walk, TARGETS["host"], spec=build_spec(("host",)), kind=DEFAULT_KIND),
    "hawk.emit.render_body": lambda: render_body(tuple(k.sinks), k.walk, spec=build_spec(("host",)), kind=DEFAULT_KIND, writer=AetherWriter()),
    "ospreys.c.render": lambda: ospreys.c.render(k),
    "ospreys.c.export": lambda: ospreys.c.export(k, work + "/c", check=False),
}
out = {}
for name, call in calls.items():
    try:
        contextvars.copy_context().run(call)
        out[name] = "ok"
    except Exception as e:
        out[name] = f"{type(e).__name__}: {e}"[:300]
print(json.dumps(out))
"""


def test_public_entries_need_no_ambient_machine(tmp_path):
    src = tmp_path / "entries.py"
    src.write_text(SCRIPT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    done = subprocess.run([sys.executable, str(src), str(tmp_path)], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert {k: v for k, v in got.items() if v != "ok"} == {}
