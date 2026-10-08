# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The two builds one host TU carries for a fused-step kernel.

An interchanged host TU (:mod:`hawk.emit.host`) holds its fused loop twice
under ``#if HAWK_HOST_INTERCHANGE``: the interchanged loop, and after the
``#else`` the plain per-sample loop a ``-DHAWK_HOST_INTERCHANGE=0`` build
compiles. :func:`builds` cuts the text into the two programs the
preprocessor would see, so a row about one shape reads that shape alone.
"""

from __future__ import annotations

MACRO = "#if HAWK_HOST_INTERCHANGE"


def builds(text: str) -> dict:
    """``{"interchanged": ..., "plain": ...}``: ``text`` with every
    ``#if HAWK_HOST_INTERCHANGE`` block resolved one way or the other. A TU
    without the macro answers itself for both."""
    out = {}
    for keep in ("interchanged", "plain"):
        lines, stack = [], []
        for line in text.splitlines():
            word = line.strip()
            if word == MACRO:
                stack.append(["macro", keep == "interchanged"])
                continue
            if word.startswith("#if"):
                stack.append(["other", True])
            elif word == "#else" and stack and stack[-1][0] == "macro":
                stack[-1][1] = not stack[-1][1]
                continue
            elif word == "#endif" and stack:
                if stack.pop()[0] == "macro":
                    continue
            if all(on for kind, on in stack if kind == "macro"):
                lines.append(line)
        out[keep] = "\n".join(lines)
    return out
