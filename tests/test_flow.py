# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""hawk.trace.flow: source lifts for ``python -c``, ``exec`` and the REPL."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading

from hawk.trace import flow

_PROG = (
    "import inspect, hawk.trace.flow as f\n"
    "def k(x):\n    return x + 1\n"
    "def other(x):\n    return x\n"
    "print(f.source_available(k), f.lift_dash_c(k), f.source_available(k))\n"
    "print(inspect.getsource(k).strip().splitlines()[0])\n"
    "ns = {}\nexec(compile('def g(y):\\n    return y', '<string>', 'exec'), ns)\n"
    "print(f.lift_dash_c(ns['g']))\n"
)


def test_dash_c_lift_is_scoped_to_verified_functions(tmp_path):
    # the child imports what this process imports, from a folder that holds no package of its own
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", _PROG], capture_output=True, text=True, check=True, cwd=tmp_path,
                         env=env).stdout.split("\n")
    # CPython 3.13+ keeps the -c text in linecache itself, so the source is there before the lift.
    assert out[0] == ("True True True" if sys.version_info >= (3, 13) else "False True True")
    assert out[1] == "def k(x):"
    assert out[2] == "False"  # an exec'd '<string>' def is not from the -c text: not relabelled


def test_lift_dash_c_outside_dash_c_run_is_false(monkeypatch):
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "script.py"], raising=False)
    assert flow.lift_dash_c() is False


def test_exec_source_registers_text():
    ns = flow.exec_source("def kern(a):\n    return a * 2\n")
    assert flow.source_available(ns["kern"])
    assert inspect.getsource(ns["kern"]).startswith("def kern")


def test_exec_source_names_unique_under_threads():
    names = []

    def run():
        ns = flow.exec_source("def f():\n    pass\n")
        names.append(ns["f"].__code__.co_filename)

    ts = [threading.Thread(target=run) for _ in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(names)) == 16


def test_source_available_false_without_source():
    assert flow.source_available(compile("1", "<nowhere>", "eval")) is False
    assert flow.source_available(eval("lambda: 0", {})) is False
