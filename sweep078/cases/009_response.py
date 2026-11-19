"""009_response: response (docs/content/across_lanes.ipynb: cell 10 (notebook))."""
# ruff: noqa: F821
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Grid, Scalar, Table, Vector

NC, NT, B, S = 3, 4, 64, 2


@hawk.kernel
def response(basis: Vector[S], theta: Table["slot":S]) -> Scalar:
    w = m.vec(theta.at(slot=0), theta.at(slot=1))
    return (basis * w).sum()                                   # inside the lane: this lane's S terms


@hawk.kernel
def coef_grad(lane: Grid["channel", "term":NT, "sample":B],
               feat: Table["slot":S, "term":NT, "sample":B],
               resid: Table["channel":NC, "sample":B]) -> Vector[S]:
    g = resid.at(channel=lane.channel, sample=lane.sample)
    terms = m.vec(feat.at(slot=0, term=lane.term, sample=lane.sample),
                  feat.at(slot=1, term=lane.term, sample=lane.sample)) * g
    return hawk.sum(terms, over="sample")                      # over an axis: B lanes become one


CASE = {
    "id": "009_response",
    "source": [
        "docs/content/across_lanes.ipynb: cell 10 (notebook)",
    ],
    "kernels": [
        "response",
        "coef_grad",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
