"""K models in one launch (``python _net_axis_kernels.py``): lanes are (net, sample), ``W[net, :]``
are per-net weights, a mask row zeroes a net's unused width, ``over="sample"`` keeps the net label."""
import pathlib
import tempfile

import numpy as np

import hawk
import hawk.math as m
from hawk import Grid, Kernel, Scalar, Table, kernel
from hawk import runtime as rt
from hawk.artifact import build_bundle

H, P = 8, 24                       # padded hidden width; weights per net: (a, b, c) per unit


def net_loss_for(S, policy="compensated"):
    @kernel
    def net_loss(lane: Grid["net", "sample":S], W: Table["net", "p":P], mask: Table["net", "h":H],
                 x: Scalar, y: Scalar) -> Scalar:
        pred = 0.0 * x
        for h in range(H):
            a = W.at(net=lane.net, p=3 * h)
            b = W.at(net=lane.net, p=3 * h + 1)
            c = W.at(net=lane.net, p=3 * h + 2)
            pred = pred + mask.at(net=lane.net, h=h) * c * m.tanh(a * x + b)
        return hawk.sum((pred - y) * (pred - y), over="sample", policy=policy)
    return Kernel(f"net_loss_{S}_{policy}", net_loss.sinks, net_loss.planes)


if __name__ == "__main__":
    K, S = 3, 40
    rng = np.random.default_rng(0)
    W, x, y = rng.normal(size=(K, P)), rng.normal(size=(K, S)), rng.normal(size=(K, S))
    mask = (np.arange(H)[None, :] < np.array([[2], [5], [8]])).astype(float)
    k = net_loss_for(S)
    home = build_bundle([k], pathlib.Path(tempfile.mkdtemp()), targets=("host",))
    out = np.zeros(K)
    rt.run(rt.load(home.directory, k.name), W=W.ravel(), mask=mask.ravel(), x=x, y=y, out=out)
    print("per-net loss", out)
