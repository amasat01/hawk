# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The emitted-text audit of the standalone C target.

The verify pass checks the IR; this pass reads the ``.c`` and ``.h`` text the
renderer produced, so a spelling no IR check can see (a helper template, the
core closure, the ABI blocks) still answers to the rules. Each hit is
``(line, rule, text)`` with ``rule`` one of:

* ``audit.token`` — a banned token in code (comments are prose and are not
  read): allocation, ``goto``, stdio, inline assembly, ``union``,
  ``setjmp``/``longjmp``, ``volatile``, C++ casts and namespaces, ``eagle``,
  ``#pragma``, ``tgmath``, ``extern "C"`` and every launch-geometry token of
  :data:`hawk.emit.spelling.GEOMETRY_TOKENS` (imported, never respelled).
  The bare word ``aether`` is allowed: the inlined core functions are named
  ``aether_c_*``;
* ``audit.include`` — an ``#include`` outside :data:`INCLUDE_WHITELIST` and
  the unit's own ``"<name>.h"``;
* ``audit.break`` — a ``for`` body with more than one ``break;`` at its own
  level (a ``break`` inside a nested ``for``/``while``/``do``/``switch``
  belongs to that statement; the tail form ``if (f) { break; }`` sits one
  brace deeper and still counts), or a ``for`` body that is not a compound
  statement (MISRA C:2012 Rules 15.4 and 15.6);
* ``audit.switch`` — a ``switch`` without ``default:`` or a clause that does
  not end in ``break;`` (Rules 16.3 and 16.4);
* ``audit.unused`` — a ``static`` function of the ``.c`` that nothing else in
  the unit references (Rule 2.2);
* ``audit.ident`` — two identifiers the header declares (the entry, the args
  struct typedef, the shared types, the status enumerators) that agree in
  their first 31 characters (Rule 5.1);
* ``audit.comment`` — a ``//`` anywhere outside a string literal, in either
  file (block comments only; Rule 3.1);
* ``audit.init`` — an aggregate (brace) initialiser ``= {`` in the ``.c``.
  hawk never emits one, and this rule keeps it so: it is what makes the
  unit-wide MISRA C:2012 Rule 9.3 deviation in
  :data:`hawk.embedded.toolchain.DEVIATIONS` (a checker false positive on
  ``const T t = f(...);``) unable to hide a real partial brace initialiser.

Line numbers are 1-based in the file the hit's text names.
"""

from __future__ import annotations

import re

from ..emit.spelling import GEOMETRY_TOKENS

#: Tokens no emitted unit may spell in code. Word tokens match at word
#: boundaries; the ``::`` namespaces, ``#pragma`` and ``extern "C"`` match as
#: literal text; the launch-geometry tokens match as a word PREFIX, as
#: :func:`hawk.emit.spelling.geometry_hits` matches them.
BANNED_TOKENS: tuple[str, ...] = (
    "malloc", "calloc", "realloc", "free", "goto", "printf", "sprintf",
    "fprintf", "asm", "__asm__", "union", "setjmp", "longjmp", "volatile",
    "static_cast", "reinterpret_cast", "eagle", "aether::", "std::", "#pragma",
    "tgmath", 'extern "C"', *GEOMETRY_TOKENS,
)

#: The system headers a unit may include; the unit's own ``"<name>.h"`` is
#: the only other include.
INCLUDE_WHITELIST: tuple[str, ...] = (
    "<math.h>", "<stdbool.h>", "<stdint.h>", "<stddef.h>", "<float.h>",
)

#: Rule 5.1: external identifiers distinct in this many leading characters.
IDENT_SIGNIFICANT = 31

_WORD = re.compile(r"\w")


def _token_pattern(token: str) -> str:
    if token in GEOMETRY_TOKENS:
        return rf"(?<!\w){re.escape(token)}"
    if _WORD.fullmatch(token[0]) and _WORD.fullmatch(token[-1]):
        return rf"(?<!\w){re.escape(token)}(?!\w)"
    return re.escape(token)


_BANNED = re.compile("|".join(_token_pattern(t) for t in BANNED_TOKENS))
_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include\b(.*)$", re.M)
_INCLUDE_TARGET = re.compile(r"\s*(<[^>\n]*>|\"[^\"\n]*\")\s*$")
_FOR = re.compile(r"\bfor\s*\(")
_NESTED = re.compile(r"\b(?:for|while|switch)\s*\(|\bdo\b|\bbreak\s*;")
_SWITCH = re.compile(r"\bswitch\s*\(")
_LABEL = re.compile(r"\bcase\b|\bdefault\s*:")
_BREAK_END = re.compile(r"\bbreak\s*;\Z")
_BRACE_INIT = re.compile(r"(?<![=!<>])=\s*\{")
_STATIC_FN = re.compile(
    r"^static[ \t]+(?:inline[ \t]+)?(?:const[ \t]+)?[A-Za-z_]\w*[ \t*]+"
    r"([A-Za-z_]\w*)[ \t]*\(", re.M)
_TYPEDEF_NAME = re.compile(r"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*;")
_STRUCT_TYPEDEF = re.compile(r"}\s*([A-Za-z_]\w*)\s*;")
_ENUM_BODY = re.compile(r"\benum\b[^{;]*{([^}]*)}")
_PROTOTYPE = re.compile(r"^[A-Za-z_][\w \t*]*?\b([A-Za-z_]\w*)[ \t]*\([^;{]*\)[ \t]*;",
                        re.M)


# --------------------------------------------------------------------------
# lexing: blank comments and/or strings, keeping every offset and newline


def _blank(span: str) -> str:
    return re.sub(r"[^\n]", " ", span)


def _views(text: str) -> tuple[str, str, list[int]]:
    """``(code, structure, line_comment_offsets)``: ``code`` is ``text`` with
    every comment blanked (string literals kept, so ``extern "C"`` and the
    include targets stay visible); ``structure`` also blanks string and
    character literals; the offsets are every ``//`` outside a literal
    (a line comment's start, or one inside a block comment)."""
    code, structure = list(text), list(text)
    slashes: list[int] = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "/*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            slashes += [m.start() + i for m in re.finditer("//", text[i:end])]
            code[i:end] = structure[i:end] = _blank(text[i:end])
            i = end
        elif two == "//":
            end = text.find("\n", i)
            end = n if end < 0 else end
            slashes.append(i)
            code[i:end] = structure[i:end] = _blank(text[i:end])
            i = end
        elif text[i] in "\"'":
            quote, j = text[i], i + 1
            while j < n and text[j] != quote and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
            structure[i + 1:end - 1] = _blank(text[i + 1:end - 1])
            i = end
        else:
            i += 1
    return "".join(code), "".join(structure), slashes


def _match(text: str, start: int) -> int:
    """The offset of the bracket closing the one at ``start`` (``(`` or
    ``{``); ``len(text)`` when it never closes."""
    pair = {"(": ")", "{": "}"}[text[start]]
    depth = 0
    for j in range(start, len(text)):
        if text[j] == text[start]:
            depth += 1
        elif text[j] == pair:
            depth -= 1
            if depth == 0:
                return j
    return len(text)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


# --------------------------------------------------------------------------
# the rules


class _File:
    def __init__(self, label: str, text: str) -> None:
        self.label = label
        self.text = text
        self.lines = text.splitlines()
        self.code, self.structure, self.slashes = _views(text)

    def line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

    def hit(self, offset: int, rule: str, why: str) -> tuple[int, str, str]:
        line = self.line_of(offset)
        source = self.lines[line - 1].strip() if line <= len(self.lines) else ""
        return (line, rule, f"{self.label} `{source}` ({why})")


def _tokens(f: _File) -> list:
    return [f.hit(m.start(), "audit.token", f"banned token {m.group(0)!r}")
            for m in _BANNED.finditer(f.code)]


def _includes(f: _File, own: str) -> list:
    out = []
    for m in _INCLUDE.finditer(f.code):
        target = _INCLUDE_TARGET.fullmatch(m.group(1))
        spelled = target.group(1) if target else m.group(1).strip()
        if spelled not in INCLUDE_WHITELIST and spelled != own:
            out.append(f.hit(m.start(), "audit.include",
                             f"{spelled} is not on the include whitelist"))
    return out


def _body(s: str, head: int) -> tuple[int, int] | None:
    """``(open, close)`` brace offsets of the compound statement after the
    parenthesised header opening at ``head``; ``None`` when the statement
    is not a compound statement."""
    after = _skip_ws(s, _match(s, head) + 1)
    if after >= len(s) or s[after] != "{":
        return None
    return after, _match(s, after)


def _own_breaks(s: str, start: int, end: int) -> list[int]:
    """Every ``break;`` in ``s[start:end]`` not inside a nested loop or
    ``switch``."""
    found: list[int] = []
    i = start
    while True:
        m = _NESTED.search(s, i, end)
        if m is None:
            return found
        word = m.group(0)
        if word.startswith("break"):
            found.append(m.start())
            i = m.end()
        elif word == "do":
            brace = _skip_ws(s, m.end())
            i = _match(s, brace) + 1 if brace < end and s[brace] == "{" else m.end()
        else:
            body = _body(s, m.end() - 1)
            i = body[1] + 1 if body else m.end()


def _breaks(f: _File) -> list:
    s, out = f.structure, []
    for m in _FOR.finditer(s):
        body = _body(s, m.end() - 1)
        if body is None:
            out.append(f.hit(m.start(), "audit.break",
                             "the loop body is not a compound statement"))
            continue
        out += [f.hit(b, "audit.break", "a second `break;` at the loop's own level")
                for b in _own_breaks(s, body[0] + 1, body[1])[1:]]
    return out


def _clauses(s: str, start: int, end: int) -> list[tuple[int, int, int]]:
    """``(label, content start, content end)`` per clause of the switch body
    ``s[start:end]``: labels are read at the body's own brace depth."""
    labels: list[tuple[int, int]] = []
    depth, i = 0, start
    while i < end:
        c = s[i]
        if c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
        elif depth == 0 and (i == start or not _WORD.fullmatch(s[i - 1])):
            m = _LABEL.match(s, i)
            if m is not None:
                colon = s.find(":", m.start())
                labels.append((i, colon + 1))
                i = colon + 1
                continue
        i += 1
    return [(lab, body, labels[k + 1][0] if k + 1 < len(labels) else end)
            for k, (lab, body) in enumerate(labels)]


def _unwrap(s: str, a: int, b: int) -> str:
    """The clause content, stripped of every block wrapping the whole of it."""
    while True:
        a = _skip_ws(s, a)
        while b > a and s[b - 1].isspace():
            b -= 1
        if a < b and s[a] == "{" and _match(s, a) == b - 1:
            a, b = a + 1, b - 1
            continue
        return s[a:b]


def _switches(f: _File) -> list:
    s, out = f.structure, []
    for m in _SWITCH.finditer(s):
        body = _body(s, m.end() - 1)
        if body is None:
            out.append(f.hit(m.start(), "audit.switch",
                             "the switch body is not a compound statement"))
            continue
        clauses = _clauses(s, body[0] + 1, body[1])
        if not any(s.startswith("default", lab) for lab, _a, _b in clauses):
            out.append(f.hit(m.start(), "audit.switch", "the switch has no `default:`"))
        for lab, a, b in clauses:
            if not _BREAK_END.search(_unwrap(s, a, b)):
                out.append(f.hit(lab, "audit.switch",
                                 "the clause does not end in `break;`"))
    return out


def _unused(f: _File) -> list:
    out = []
    for m in _STATIC_FN.finditer(f.structure):
        name = m.group(1)
        uses = len(re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", f.structure))
        if uses < 2:
            out.append(f.hit(m.start(), "audit.unused",
                             f"static function {name} is never referenced"))
    return out


def _idents(f: _File) -> list:
    s = f.structure
    found: dict[str, int] = {}
    for pattern in (_TYPEDEF_NAME, _STRUCT_TYPEDEF, _PROTOTYPE):
        for m in pattern.finditer(s):
            found.setdefault(m.group(1), m.start(1))
    for m in _ENUM_BODY.finditer(s):
        for part in m.group(1).split(","):
            ident = re.match(r"\s*([A-Za-z_]\w*)", part)
            if ident:
                found.setdefault(ident.group(1), m.start(1) + m.group(1).find(part))
    out, names = [], sorted(found, key=lambda k: (found[k], k))
    for k, later in enumerate(names):
        for earlier in names[:k]:
            if later[:IDENT_SIGNIFICANT] == earlier[:IDENT_SIGNIFICANT]:
                out.append(f.hit(found[later], "audit.ident",
                                 f"{later} and {earlier} agree in their first "
                                 f"{IDENT_SIGNIFICANT} characters"))
    return out


def _inits(f: _File) -> list:
    return [f.hit(m.start(), "audit.init",
                  "an aggregate initialiser `= {`; the C target emits none")
            for m in _BRACE_INIT.finditer(f.structure)]


def _comments(f: _File) -> list:
    return [f.hit(i, "audit.comment",
                  "`//` outside a string literal; block comments only")
            for i in f.slashes]


def audit(c_text: str, h_text: str, name: str) -> list[tuple[int, str, str]]:
    """Every rule hit in the ``<name>.c``/``<name>.h`` pair, ``.c`` first,
    each as ``(line, rule, text)``; an empty list is a clean unit."""
    c, h = _File(f"{name}.c", c_text), _File(f"{name}.h", h_text)
    own = f'"{name}.h"'
    hits = []
    for f in (c, h):
        hits += sorted(_tokens(f) + _includes(f, own) + _comments(f)
                       + _breaks(f) + _switches(f)
                       + (_unused(f) + _inits(f) if f is c else _idents(f)))
    return hits


__all__ = ["BANNED_TOKENS", "INCLUDE_WHITELIST", "audit"]
