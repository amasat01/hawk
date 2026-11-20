# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra op kinds: one factorisation node, consumers and
projections over it, the type rules, the cost formulas (operations and bytes),
and the general-shape descriptor a sidecar yields.

The formulas are the ones ``aether::linalg`` documents: ``lu`` n^3/3, ``chol``
n^3/6, ``eigh`` S(n) n(n-1)/2 (6n+10), a solve n^2 k.
"""

from __future__ import annotations


































# --- cost ---------------------------------------------------------------------



















# --- the sidecar descriptor reads declared shapes ----------------------------------

def _sidecar(**kw):
    base = {"scalar_type": "float64", "mutables": [], "params": [], "arg_dtypes": {}}
    base.update(kw)
    # the slots these rows describe are the mirror-shaped ones: outputs, vectors and matrices
    base["arg_mirrors"] = {name: "GRefMirror" for _role, name in base.get("arg_spec", ())}
    return base


# --- a factorisation is never an arm ---------------------------------------------------

