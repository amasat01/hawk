"""071_layer: layer (docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 16 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp

n_in, W = 4, 16


@hawk.kernel
def layer(x: Vector[n_in], theta: Vector[n_in * W], kinds: Vector[n_in], y: Mutable[Scalar]):
    total = 0.0
    for e in range(n_in):
        u = x[e] * 15.0
        f = m.floor(u)
        i = m.index(f, 15)
        w = u - f
        spline = theta[e * W + i] * (1.0 - w) + theta[e * W + i + 1] * w
        cases = [spline, x[e] * x[e], m.sin(x[e])]
        total = total + m.dispatch(m.int32(kinds[e]), cases, policy="switch")
    y = total


layer_vjp = Kernel("layer_vjp", vjp(layer, wrt=("theta",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    rng = np.random.default_rng(1)
    ns = 3
    lx = rng.uniform(0.0, 1.0, (n_in, ns))
    ltheta = rng.normal(size=(n_in * W, ns))
    kinds = np.array([[0, 1, 0], [1, 0, 2], [0, 0, 0], [2, 0, 0]], dtype=float)   # a kind per edge and sample
    return {"x": lx, "theta": ltheta, "kinds": kinds, "bar_y": np.ones(ns)}


CASE = {
    "id": "071_layer",
    "source": [
        "docs/content/tutorials/07_gradients_through_branches_and_stops.ipynb: cell 16 (notebook)",
    ],
    "kernels": [
        "layer",
        "layer_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "layer_vjp",
    "call_form": True,
    "build_groups": [
        [
            "layer",
            "layer_vjp",
        ],
    ],
}
