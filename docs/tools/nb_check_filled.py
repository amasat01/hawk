#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Gate: every notebook under the given directories must be EXECUTED — each
code cell with non-blank source carries an execution count AND no `error`
output. deploy:pages runs this on the notebooks it received from
test:pages before rendering them with Sphinx execution switched off, so an
unfilled notebook (or one that silently ran a cell to a traceback) can
never reach the published pages (nor trigger a second, hidden execution).

Also refuses a notebook whose OUTPUT (stream text, `text/plain`,
`text/html`, or an error traceback) contains an absolute local-machine
path. hawk's docs publish to a public site; a committed output with
`/local/...`, `/home/<user>/...`, `~/.cache/hawk/...` (as an
absolute path) etc. leaks the machine that built them. Only rendered
OUTPUT is scanned, not cell source. A stored Python warning (``UserWarning:``
and the like) is refused too: it is noise from the machine that ran the
notebook, and it usually carries such a path.

Finally, refuses a stored image larger than IMAGE_BUDGET: the docs show the
stored plots (nothing re-executes at build), so a runaway figure would bloat
the repository.
"""
import pathlib
import re
import sys

import nbformat

# Deliberately narrow: absolute local-filesystem roots a committed output
# should never contain, not a general "looks like a path" heuristic.
# Bytes of base64 per stored image output. Measured on the 9 shipped notebooks:
# mean ~30 KB, largest 109 KB.
IMAGE_BUDGET = 120_000

_LEAK_PATTERNS = [re.compile(p) for p in (r"/local/", r"/home/", r"/tmp/", r"/dev/shm/", r"/root/", r"/Users/",
                                          r"C:\\Users")]
_WARNING = re.compile(r"\b\w*Warning: ")


def _text_of(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def _leaked_path(text: str) -> str | None:
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


def _output_leaks(cell) -> list[str]:
    leaks = []
    for out in cell.get("outputs", []):
        texts = []
        if out.get("output_type") == "stream":
            texts.append(_text_of(out.get("text", "")))
        elif out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            texts.extend(_text_of(data[mime]) for mime in ("text/plain", "text/html") if mime in data)
        elif out.get("output_type") == "error":
            texts.append(_text_of(out.get("traceback", [])))
        for text in texts:
            pattern = _leaked_path(text)
            if pattern:
                leaks.append(pattern)
            if _WARNING.search(text):
                leaks.append("a Python warning")
    return leaks


def _oversized_images(cell) -> list[int]:
    return [len(_text_of(out["data"][mime])) for out in cell.get("outputs", [])
            for mime in ("image/png", "image/jpeg")
            if mime in out.get("data", {}) and len(_text_of(out["data"][mime])) > IMAGE_BUDGET]


def main(argv: list[str]) -> int:
    """Refuse (return 1) if any notebook under ``argv``'s roots carries an
    unexecuted code cell or an output that leaks a local path; 2 when no
    notebook was found at all."""
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path(".")]
    notebooks = sorted(
        p for r in roots
        for p in ([r] if r.suffix == ".ipynb" else r.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in p.parts
    )
    if not notebooks:
        print("nb_check_filled: no notebooks found under", [str(r) for r in roots])
        return 2
    bad = []
    errored = []
    leaked = []
    heavy = []
    for p in notebooks:
        nb = nbformat.read(p, as_version=4)
        unexecuted = [
            i for i, c in enumerate(nb.cells)
            if c.cell_type == "code" and c.source.strip()
            and c.get("execution_count") is None
            and "colab-setup" not in c.get("metadata", {}).get("tags", [])  # generated, no output
        ]
        if unexecuted:
            bad.append((p, unexecuted))
        has_error = [
            i for i, c in enumerate(nb.cells)
            if c.cell_type == "code"
            and any(out.get("output_type") == "error" for out in c.get("outputs", []))
        ]
        if has_error:
            errored.append((p, has_error))
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            for pattern in _output_leaks(c):
                leaked.append((p, i, pattern))
            heavy.extend((p, i, size) for size in _oversized_images(c))
    for p, cells in bad:
        shown = cells[:6]
        more = "…" if len(cells) > 6 else ""
        print(f"[nb_check_filled] UNEXECUTED {p}: code cells {shown}{more}")
    for p, cells in errored:
        shown = cells[:6]
        more = "…" if len(cells) > 6 else ""
        print(f"[nb_check_filled] ERROR OUTPUT {p}: code cells {shown}{more}")
    for p, i, pattern in leaked:
        print(f"[nb_check_filled] LEAKED OUTPUT {p}: code cell {i} output matched {pattern!r}")
    for p, i, size in heavy:
        print(f"[nb_check_filled] IMAGE OVER BUDGET {p}: code cell {i} image is {size} bytes "
              f"(budget {IMAGE_BUDGET})")
    filled = len(notebooks) - len({p for p, _ in bad} | {p for p, _ in errored})
    print(f"[nb_check_filled] {filled}/{len(notebooks)} notebooks fully executed, no errors")
    return 1 if (bad or errored or leaked or heavy) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
