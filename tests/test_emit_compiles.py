# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The compile smoke: every emitted translation unit is real C++.

The only thing that turns "the emitter produced a plausible string" into
"the emitter produced a program". Each unit is checked with the toolchain
names: ``g++
-std=c++23 -fsyntax-only`` for the host TU (AETHER_CPP_MODE -- the standalone
artifact mode; see ``_toolchain.py``) and ``nvcc -std=c++20 -arch=<a real arch the toolkit supports> -c``
for the device TU (AOT, never NVRTC; C++20 is nvcc's cap). No GPU is used and
nothing is executed: this is a syntax/object check of a single TU.

Shapes covered, one per lowering the emitter must get right: a rank-0 chain
(``scale``), a rank-1 chain (``drag``), a compound quaternion quantity
(``spin``/``rotate_quat``/``as_body``), a ``lookup`` gather
(``gather``, the ``at``), a mapreduce partial (``energy``), a
scatter (``scatter``), the six-op chain (``chain6``) and a VJP
(``drag_vjp``) -- x both backends x both scalar modes.

This test previously failed. Two observations, one of which changed the emitter:
  * the first hand-written probe of this shape bound every plane ``const auto``.
    aether's ``View::operator[]`` has a const overload returning a READ-ONLY
    ``ConstSampleRef``, so every write sink
    was rejected. Re-planted afterwards as ``WRITABLE_ROLES =`` to observe
    the row itself go red --
      scale_host_float64.cpp:95:54: error: no match for 'operator=' (operand
      types are 'aether::ConstSampleRef<const aether::View<double,
      aether::extents<...>, aether::layout_stride, false, false> >' and 'Real'
      {aka 'double'})
    ``hawk/emit/backend.WRITABLE_ROLES`` (binding a write target NON-const) is
    the fix that observation bought; the plant was then removed.
  * a second, environmental red worth recording because it makes this row
    non-vacuous: a conda activation exports ``NVCC_PREPEND_FLAGS`` pointing at
    its own gcc-14, which CUDA 12.6 refuses ("gcc versions later than 13 are not
    supported"). The device arm runs under the caller's own
    ``NVCC_PREPEND_FLAGS`` (or ``$HAWK_NVCC_CCBIN``), so a wrong host compiler
    fails here loudly rather than being papered over.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _emitted import sources
from _toolchain import compile_check

#: The shapes the smoke must cover.
_REQUIRED = {"scale", "drag", "spin", "gather", "energy", "drag_vjp"}


@pytest.fixture(scope="module")
def emitted(tmp_path_factory) -> list:
    root = tmp_path_factory.mktemp("hawk_emit")
    units = []
    for name, backend, mode, src in sources():
        suffix = "cu" if backend.id == "cuda" else "cpp"
        path = Path(root) / f"{name}_{backend.id}_{mode.id}.{suffix}"
        path.write_text(src.text)
        units.append((name, backend.id, mode.id, path))
    return units


def test_the_smoke_covers_every_required_shape(emitted):
    covered = {name for name, _b, _m, _p in emitted}
    missing = _REQUIRED - covered
    assert not missing, f"the compile smoke lost its {sorted(missing)} kernel(s)"


def test_every_emitted_unit_compiles(emitted):
    failures = []
    for name, backend, mode, path in emitted:
        rc, stderr, argv = compile_check(path, backend)
        if rc != 0:
            failures.append(f"--- {name}/{backend}/{mode}\n$ {' '.join(argv)}\n"
                            + stderr[:2000])
    assert not failures, (
        "an emitted translation unit does not compile -- the emitter is wrong, not "
        "the test:\n" + "\n".join(failures)
    )
