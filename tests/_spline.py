# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The spline edge cell, at HAWK's own authoring surface — TEST SCAFFOLD.

ONE definition of the shape and were both ruled over, shared by the
compile card (``tests/test_loop_lowering_card.py``) and the scale entry point below, so the two can never drift into
measuring different subjects and reporting them under one name.

WHY IT IS COPIED AND NOT IMPORTED. The real subject is a downstream
package's spline edge kernel,
and ``hawk`` may not depend on it (the dependency runs the other way). So the shape is
reproduced here from HAWK's own vocabulary, with the two properties that made it
the subject preserved exactly:

* the ``for`` over EDGES is in the kernel body, so the ``ast`` pass lowers it to
  a real loop and the trip count never enters the IR;
* everything inside one edge — the Cox-de Boor basis recursion and the
  basis-term sum — is built by TRACE-TIME python in a helper, exactly as the
  downstream package's ``_edge_phi_loop_wide_hawk`` builds it, so the loop BODY is several hundred
  nodes deep. That depth is the whole point: a ``term1 + term2`` node whose two
  children are both large is what made an early body key double in size once
  per recursion level.

At ``(n_in=64, G=20, k=3)`` — the point measured here — the body is 727
nodes and the walk is 89.
"""

from __future__ import annotations

import hawk
import hawk.math as m

__all__ = ["knots", "basis_levels", "edge_phi", "spline_edge_cell",
           "shallow_spline_cell", "reference", "main"]


def knots(G: int, k: int) -> tuple:
    """``G + 2k + 1`` uniformly spaced knots extending ``[-1, 1]`` by ``k``
    intervals on each side — a downstream package's ``cox_de_boor_knots``. Uniform spacing makes
    every Cox-de Boor denominator the constant ``d * step``, so the recursion
    below needs no divide-by-zero guard."""
    step = 2.0 / G
    return tuple(-1.0 + (j - k) * step for j in range(G + 2 * k + 1))


def basis_levels(x, kn: tuple, k: int) -> list:
    """The order-``k`` Cox-de Boor basis at a traced scalar ``x``.

    The DEEP half of the body: each level's element combines TWO elements of the
    level below, so the dependency graph fans in at every one of the ``k``
    levels. Written against whatever ``x`` is — a traced value inside a lowered
    loop's scope, in this file's only caller."""
    level = [m.where(m.land(x >= kn[i], x < kn[i + 1]), 1.0, 0.0)
             for i in range(len(kn) - 1)]
    for d in range(1, k + 1):
        level = [
            ((x - kn[i]) / (kn[i + d] - kn[i])) * level[i]
            + ((kn[i + d + 1] - x) / (kn[i + d] - kn[i])) * level[i + 1]
            for i in range(len(level) - 1)
        ]
    return level


def edge_phi(x, theta, e, kn: tuple, k: int, n_basis: int):
    """One edge's spline value. A TRACE-TIME helper: a helper's body is ordinary
    Python the ``ast`` pass never sees, so its ``for`` runs once while the
    enclosing lowered ``for`` body is being traced and builds its chain INSIDE
    that scope — a downstream package's own ``_edge_phi_loop_wide_hawk`` arrangement."""
    phi = basis_levels(x, kn, k)
    spline = phi[0] * theta.at(edge=e, basis=0)
    for b in range(1, n_basis):
        spline = spline + phi[b] * theta.at(edge=e, basis=b)
    return spline


def spline_edge_cell(n_in: int, G: int, k: int):
    """The spline edge cell at ``(n_in, G, k)``: ONE lowered ``for`` over edges
    whose body is a Cox-de Boor basis expansion folded through a nonlinearity."""
    kn, n_basis = knots(G, k), G + k

    @hawk.kernel
    def cell(x: hawk.Table["edge":n_in],
             theta: hawk.Table["edge":n_in, "basis":n_basis],
             g: hawk.Param,
             y: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for e in range(n_in):
            total = total + m.tanh(
                edge_phi(x.at(edge=e) * g, theta, e, kn, k, n_basis))
        y = total

    return cell


def reference(x, theta, g: float, n_in: int, G: int, k: int) -> float:
    """The same arithmetic in numpy, in the loop's own association order — the
    third arm beside the compiled kernel and the IR interpreter."""
    import numpy as np

    kn, n_basis = knots(G, k), G + k
    total = 0.0
    for e in range(n_in):
        xe = x[e] * g
        level = [1.0 if (xe >= kn[i] and xe < kn[i + 1]) else 0.0
                 for i in range(len(kn) - 1)]
        for d in range(1, k + 1):
            level = [
                ((xe - kn[i]) / (kn[i + d] - kn[i])) * level[i]
                + ((kn[i + d + 1] - xe) / (kn[i + d] - kn[i])) * level[i + 1]
                for i in range(len(level) - 1)
            ]
        spline = level[0] * theta[e * n_basis]
        for b in range(1, n_basis):
            spline = spline + level[b] * theta[e * n_basis + b]
        total = total + np.tanh(spline)
    return float(total)


def shallow_spline_cell(n_in: int, basis: int):
    """The card's ORIGINAL synthetic family — two nested lowered loops over a
    shallow body — kept beside the deep one so a row can cover both shapes."""
    @hawk.kernel
    def cell(x: hawk.Table["edge":n_in],
             theta: hawk.Table["edge":n_in, "basis":basis],
             g: hawk.Param,
             y: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for e in range(n_in):
            xe = x.at(edge=e)
            acc = 0.0
            for b in range(basis):
                acc = acc + theta.at(edge=e, basis=b) * m.cos(g * (b + 1) * xe)
            total = total + m.tanh(acc)
        y = total

    return cell


# --------------------------------------------------------------------------- #
# The capped child.
#
# WHY THIS ENTRY POINT EXISTS AT ALL. The spline-scale point is the one that took
# a 31 GB box down before /, and rule is that a trace or
# canonicalisation of a body that size is exercised only behind an address-space
# cap. A `ulimit -v` is applied to a PROCESS, so the whole end-to-end — trace,
# canonicalise, emit, compile both targets, build the host artifact, run it,
# and evaluate the same IR with the scratch interpreter — has to happen in ONE
# child that the parent starts under that cap. `RLIMIT_AS` is inherited, so the
# compilers this child spawns run under the same ceiling without being told.
#
# It prints ONE json line, which `tests/test_loop_lowering_card.py` folds into
# CARD_LOOP_LOWERING.md. It measures; it decides nothing.
# --------------------------------------------------------------------------- #

#: Samples the compiled host kernel is run over for the timing column.
_SAMPLES = 256

#: Peak RSS, in KiB, out of `/usr/bin/time -v`'s own report. Read from the
#: COMPILER's own process rather than from `getrusage(RUSAGE_CHILDREN)`, which
#: reports the high-water mark over every child this process has ever reaped and
#: would therefore attribute the first compiler's peak to the second one too.
