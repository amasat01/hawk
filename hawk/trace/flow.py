# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Source lifts: make ``inspect.getsourcelines(fn)`` work for functions not defined in a file
(``python -c``, ``exec``, the REPL). Everything goes through :mod:`linecache`, which is what
:mod:`inspect` reads. Zero dependencies.

Public: :func:`lift_dash_c`, :func:`exec_source`, :func:`register_source`, :func:`source_available`.
"""

from __future__ import annotations

import inspect
import itertools
import linecache
import sys
import threading

__all__ = ["exec_source", "lift_dash_c", "register_source", "source_available"]

_lock = threading.Lock()
_counter = itertools.count(1)
_DASH_C = "<string>"


def register_source(filename: str, text: str) -> str:
    """Make ``filename`` resolvable by inspect.

    A full ``(size, None, lines, path)`` linecache entry is never dropped by
    ``linecache.checkcache`` (mtime ``None`` means not a real file).
    """
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    with _lock:
        linecache.cache[filename] = (len(text), None, lines, filename)
    return filename


def exec_source(text: str, ns: dict | None = None, name: str | None = None) -> dict:
    """``exec`` generated source under a unique registered name, so its functions can be traced."""
    ns = {} if ns is None else ns
    if name is None:
        with _lock:
            name = f"<flow-exec-{next(_counter)}>"
    register_source(name, text)
    exec(compile(text, name, "exec"), ns)
    return ns


def lift_dash_c(*fns) -> bool:
    """Under ``python -c "<source>"`` code is labelled ``<string>`` and has no text; the text is the
    argument after ``-c`` in ``sys.orig_argv`` (which exists from 3.10: on 3.9 this returns False).

    Every ``<string>`` code object shares that label, so the text is registered only when each
    ``fns`` entry is verified to live in it (its ``def`` line matches at ``co_firstlineno``).
    Returns False for a non-``-c`` run or a function that is not from the ``-c`` text.
    Call it after the kernels are defined: ``python -c "import hawk.trace.flow as f; ...; f.lift_dash_c(k)"``.
    """
    argv = getattr(sys, "orig_argv", None)
    if not argv or "-c" not in argv:
        return False
    i = argv.index("-c")
    if i + 1 >= len(argv):
        return False
    text = argv[i + 1]
    lines = text.splitlines()
    for fn in fns:
        code = getattr(fn, "__code__", None)
        if code is None or code.co_filename != _DASH_C:
            return False
        head = lines[code.co_firstlineno - 1 : code.co_firstlineno + 8]  # a decorator stack may precede the def
        if not any(line.lstrip().startswith((f"def {fn.__name__}(", f"async def {fn.__name__}(")) for line in head):
            return False
    register_source(_DASH_C, text)
    return True


def source_available(fn) -> bool:
    """True when ``inspect.getsourcelines(fn)`` and ``getsourcefile(fn)`` succeed (the plain REPL
    before 3.13 is not supported: its lines are not in linecache)."""
    try:
        inspect.getsourcelines(fn)
        return inspect.getsourcefile(fn) is not None
    except (OSError, TypeError):
        return False
