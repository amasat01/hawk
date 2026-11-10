# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two emitter rules a long body with several outputs needs.

A vector value read more than once is an ``aether::Item``, evaluated once: as a lazy expression each reader
would nest it again, and the compile would grow exponentially with the length of a chain of such values.
An output is stored right after its value is complete, so a body with several outputs keeps none of them
live to its end.
"""

from __future__ import annotations

import linecache
import re

import numpy as np
from conftest import sidecar_of

from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.emit.render import render_body


def _source(text: str, tag: str):
    """Trace the one kernel in ``text`` (a generated source, registered so the tracer can read it)."""
    name = f"<{tag}>"
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)
    ns: dict = {}
    exec(compile(text, name, "exec"), ns)          # noqa: S102 - a generated test fixture
    return ns["k"]


def fan_in(n: int, w: int = 8):
    """``n`` vector stages, each read twice by the next, then summed in reverse order."""
    lines = ["import hawk", "from hawk import Scalar, Mutable, Vector, math as m", "", "@hawk.kernel",
             f"def k(x: Scalar, out: Mutable[Vector[{w}]]):",
             f"    r0 = m.vec({', '.join(['x'] * w)}) * 0.5"]
    for j in range(1, n):
        lines.append(f"    r{j} = m.sin(r{j - 1}) * r{j - 1} + {0.25 + 0.01 * j:.2f}")
    lines.append(f"    out = {' + '.join(f'r{j}' for j in reversed(range(n)))}")
    return _source("\n".join(lines) + "\n", f"fan_in_{n}_{w}")


def three_outputs():
    return _source("import hawk\nfrom hawk import Scalar, Mutable, math as m\n\n@hawk.kernel\n"
                   "def k(x: Scalar, a: Mutable[Scalar], b: Mutable[Scalar], c: Mutable[Scalar]):\n"
                   "    a = m.sin(x)\n    b = m.exp(a)\n    c = m.cos(b)\n", "three_outputs")


def _body(kernel) -> str:
    return render_body(kernel.sinks, kernel.walk).text


def test_item_temps_linear_compile():
    texts = {n: _body(fan_in(n)) for n in (6, 12)}
    for n, text in texts.items():
        held = re.findall(r"^\s*const aether::Item<Real, 8> t\d+ = ", text, re.M)
        lazy = re.findall(r"^\s*const auto t\d+ = (?!aether::Item<)", text, re.M)   # a vec literal is a leaf already
        assert len(held) == n - 1 and not lazy, (n, held, lazy)
    assert len(texts[12]) <= 2.2 * len(texts[6]), {n: len(t) for n, t in texts.items()}


def test_sink_at_producer():
    lines = _body(three_outputs()).splitlines()
    first = {op: next(i for i, line in enumerate(lines) if f"{op}(" in line) for op in ("sin", "exp", "cos")}
    store = {name: next(i for i, line in enumerate(lines) if line.lstrip().startswith(f"mut_{name}[i] =")) for name in "abc"}
    assert first["sin"] <= store["a"] < first["exp"] <= store["b"] < first["cos"] <= store["c"], lines


def test_early_stores_keep_the_values(tmp_path, cache_dir):
    kernel = three_outputs()
    bundle = build_bundle([kernel], tmp_path / "k", targets=("host",), cache_dir=cache_dir, host_profile="native")
    art = rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    x = np.linspace(-2.0, 2.0, 33)
    a, b, c = (np.zeros_like(x) for _ in range(3))
    rt.run(art, x=x, a=a, b=b, c=c)
    np.testing.assert_allclose(a, np.sin(x), rtol=4e-16, atol=1e-300)
    np.testing.assert_allclose(b, np.exp(a), rtol=4e-16)
    np.testing.assert_allclose(c, np.cos(b), rtol=4e-16, atol=1e-300)
