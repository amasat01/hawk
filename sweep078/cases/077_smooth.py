"""077_smooth: smooth (docs/content/tutorials/09_per_sample_control.ipynb: cell 5 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar
from hawk.math import n_samples, sample_index, select


@hawk.kernel
def smooth(x: Scalar, y: Mutable[Scalar]):
    i = sample_index()
    left = x.at(select(i > 0, i - 1, i))
    right = x.at(select(i < n_samples() - 1, i + 1, i))
    y = 0.25 * left + 0.5 * x + 0.25 * right


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    n = 48
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0.0, 2.0 * np.pi, n)) + 0.3 * rng.normal(size=n)
    return {"x": x}


CASE = {
    "id": "077_smooth",
    "source": [
        "docs/content/tutorials/09_per_sample_control.ipynb: cell 5 (notebook)",
    ],
    "kernels": [
        "smooth",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": None,
    "run_kernel": "smooth",
    "call_form": True,
}
