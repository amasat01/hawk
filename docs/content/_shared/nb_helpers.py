# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Small, repeated mechanics shared by every hawk docs notebook.

Every tutorial/example notebook needs the same handful of things before
its first real cell: whether a GPU is usable, the matching array library,
a plot style that stays legible on both a light and a dark page, and a
way to size a batch that does not force every lesson cell to branch on
the device. None of that is part of the lesson, so it lives here behind
one hidden import instead of being re-typed (and re-explained, and
re-branched-on) on every page.

This file is this site's own copy on purpose: each docs site in the
family keeps an independent copy rather than depending on another repo's
docs tree.
"""
from __future__ import annotations

import warnings

# An optional dependency's own notice (seen when importing torch alongside
# certain NumPy/SciPy builds); not something any notebook's lesson is about.
warnings.filterwarnings("ignore", category=FutureWarning)


def gpu_available() -> bool:
    """``True`` iff CuPy is importable and sees at least one GPU."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def xp_for(device: str):
    """The array module for ``device`` (``"cpu"`` or ``"gpu"``): NumPy or CuPy."""
    if device == "gpu":
        import cupy as cp
        return cp
    import numpy as np
    return np


#: Resolved once, at import time: every notebook that imports this module
#: gets the SAME device decision, so no lesson cell has to compute it (or
#: branch on it) itself.
DEVICE = "gpu" if gpu_available() else "cpu"
xp = xp_for(DEVICE)

try:
    import torch as _torch
    torch_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
except Exception:
    torch_device = None


def to_numpy(a):
    """``a`` as a NumPy array, regardless of which device it came from."""
    return xp.asnumpy(a) if DEVICE == "gpu" else a


def batch_size(gpu: int, cpu: int) -> int:
    """``gpu`` on a GPU run, ``cpu`` otherwise -- the one place a notebook's
    sample count depends on the device, so no lesson cell needs an
    ``if DEVICE == "gpu"`` of its own."""
    return gpu if DEVICE == "gpu" else cpu


def plot_style() -> None:
    """Apply the family's matplotlib style: transparent figure background,
    muted foreground colors legible on both a light and a dark page."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "none", "savefig.facecolor": "none", "axes.facecolor": "none",
        "text.color": "#898781", "axes.labelcolor": "#898781", "axes.edgecolor": "#898781",
        "xtick.color": "#898781", "ytick.color": "#898781", "grid.color": "#c3c2b7",
    })
