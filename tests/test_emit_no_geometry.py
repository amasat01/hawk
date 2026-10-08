# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""No launch geometry in an emitted body.

HAWK owns no launch geometry: the grid is eagle's. The structural
guarantee is that the renderer emits a body STRING with no geometry token in it
at all, and the ONE index prologue that names one is the BACKEND's — so
the audit has two halves, and the second is what keeps the first from being
vacuous:

  1. no geometry token appears in ANY emitted body;
  2. the cuda backend's index prologue DOES name them, and in a full emitted
     translation unit every occurrence lies inside that prologue's text.

Without (2) a renderer that emitted nothing at all would pass (1).

This test previously failed when the device flat index moved out
of ``CudaBackend.index_prologue`` and into the renderer (one ``self._emit`` line
at the top of ``_Renderer.run``) --

    AssertionError: : a launch-geometry token escaped the backend's ONE
    index prologue into the body. Found:
      scale:1: 'blockIdx' in `const long long hawk_flat = blockIdx.x *
        blockDim.x + threadIdx.x;`
      scale:1: 'blockDim' in `...`
      scale:1: 'threadIdx' in `...`... 3 hits x 10 kernels

-- the plant was then removed. The audit matches at a WORD boundary, not as a
bare substring: ``omp`` is a substring of the ABI mirror's own ``compStride_``
field, and the first form of this row reported the view reconstruction as
launch geometry (measured, before the boundary was added).
"""

from __future__ import annotations

from _emitted import bodies, sources

from hawk.emit import BACKENDS, CUDA
from hawk.emit.aether import geometry_hits


def test_no_body_contains_a_geometry_token():
    hits = []
    for name, body in bodies():
        hits += [f"  {name}:{n}: {token!r} in `{line}`"
                 for n, token, line in geometry_hits(body.text)]
    assert not hits, (
        "a launch-geometry token escaped the backend's ONE index prologue "
        "into the body (EC-I1). Found:\n" + "\n".join(hits)
    )


def test_the_detector_can_see_the_tokens_it_looks_for():
    """Non-vacuity: the cuda prologue is the ONE place they legitimately appear."""
    present = sorted({t for _n, t, _l in geometry_hits(CUDA.index_prologue())})
    assert present == ["blockDim", "blockIdx", "threadIdx"], (
        "the cuda index prologue no longer names the geometry builtins the audit "
        f"greps for — it names {present}, so the first half proves nothing"
    )
    assert not geometry_hits(BACKENDS["host"].index_prologue()), (
        "the SERIAL host entry named a geometry token: threading and "
        "tiling are eagle's HostTeam, never HAWK's"
    )


def test_every_geometry_token_in_a_full_unit_lies_in_the_index_prologue():
    for name, backend, mode, src in sources():
        prologue = backend.index_prologue()
        outside = src.text.replace(prologue, "")
        stray = sorted({t for _n, t, _l in geometry_hits(outside)})
        assert not stray, (
            f"{name}/{backend.id}/{mode.id}: geometry token(s) {stray} appear outside "
            "the backend's ONE index prologue"
        )
