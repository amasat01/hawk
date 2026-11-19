"""064_histogram: histogram (docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 9 (notebook))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk.artifact
from hawk import Index, Mutable, Scalar


@hawk.kernel
def histogram(lane: Index, acc: Mutable[Scalar]):
    hawk.scatter(acc)[lane] += 1.0


def inputs():
    """Planes of the run the source shows, from a fixed seed."""
    import numpy as np
    rng = np.random.default_rng(0)
    n_samples, n_bins = 2000, 10
    lane = rng.integers(0, n_bins, size=n_samples).astype(np.int32)
    acc = np.zeros(n_bins)
    return {"lane": lane, "acc": acc}


def oracle(planes):
    """Expected value of the checked plane, as the source computes it."""
    lane = planes["lane"]
    import numpy as np
    n_samples, n_bins = 2000, 10
    expect = np.bincount(lane, minlength=n_bins).astype(float)
    return {"acc": expect}


CASE = {
    "id": "064_histogram",
    "source": [
        "docs/content/tutorials/06_lookups_scatters_and_sums.ipynb: cell 9 (notebook)",
    ],
    "kernels": [
        "histogram",
    ],
    "targets": ('host',),
    "inputs": "inputs",
    "oracle": "oracle",
    "run_kernel": "histogram",
}
