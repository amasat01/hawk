"""084_ramp_only: ramp_only (docs/content/vocabulary/03_your_own_derivative.ipynb: cell 16 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Kernel, Mutable, Scalar
from hawk.diff import vjp
from hawk.ext import primitive
from hawk.math import exp, log


@primitive("speed_ramp",
           vjp=lambda s, bar: bar / (1.0 + exp(-s)),
           jvp=lambda s, ds: ds / (1.0 + exp(-s)))
def speed_ramp(s):
    return log(1.0 + exp(s))


@hawk.kernel
def ramp_only(x: Scalar, y: Mutable[Scalar]):
    y = speed_ramp(x)


ramp_only_vjp = Kernel("ramp_only_vjp", vjp(ramp_only, wrt=("x",)))


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    xs = np.linspace(-6.0, 6.0, 25)
    H = 1e-5
    return {"x": xs + H}


CASE = {
    "id": "084_ramp_only",
    "source": [
        "docs/content/vocabulary/03_your_own_derivative.ipynb: cell 16 (notebook)",
    ],
    "kernels": [
        "ramp_only",
        "ramp_only_vjp",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "ramp_only",
    "call_form": True,
    "build_groups": [
        [
            "ramp_only",
            "ramp_only_vjp",
        ],
    ],
}
