# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contraction forward kernel of a spline-shaped edge sum, shared by the lane-facts and thread-box rows."""

from __future__ import annotations

import hawk.math as m
from hawk import Index, Mutable, Scalar, Table
import hawk

N_IN, N_OUT, BATCH, G, K = 8, 8, 256, 5, 2
NB = G + K
S_ROW = NB + 2
E_ALL = N_IN * N_OUT
CHUNK = BATCH // 2
L_ALL = N_IN * BATCH
O_N, O_S = 1, 2 * K + 3


def _feat(feat, slot, l):
    return feat.at(slot * L_ALL + l)


def _edge(feat, th, l, b):
    i = m.index(_feat(feat, 0, l), G)
    acc = 0.0
    for r in range(K + 1):
        acc = acc + _feat(feat, O_N + r, l) * th.at(b + 2 + i + r)
    return acc


@hawk.kernel
def kg_fwd(lane: Index, feat: Table[Scalar], th: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(BATCH)
    acc = 0.0 * m.real(lane)
    for e in range(N_IN):
        l = e * BATCH + s
        b = (j * N_IN + e) * S_ROW
        acc = acc + th.at(b) * _feat(feat, O_S, l) + th.at(b + 1) * _edge(feat, th, l, b)
    y = acc
