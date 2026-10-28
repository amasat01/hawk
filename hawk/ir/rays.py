# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Ray intersection: the op kinds, their type rule and their cost.

Three kinds, each ONE node whose value is a rank-1 ``(4,)`` result read with the
``component`` op, like the quaternion kinds:

* ``ray_tri``    ``(o, d, a, b, c, tmax)`` -> ``[hit, t, u, v]``
* ``ray_aabb``   ``(o, d, lo, hi, tmax)``  -> ``[hit, t_enter, t_exit, 0]``
* ``ray_sphere`` ``(o, d, c, r, tmax)``    -> ``[hit, t0, t1, 0]``

``o``, ``d``, ``a``, ``b``, ``c``, ``lo``, ``hi`` are length-3 vectors; ``r`` and
``tmax`` are rank-0 reals. The arithmetic and the degenerate-input rules are
``aether::geometry``'s (``aether/geometry/Ray.h``): values are the computed ones also
on a miss, and a degenerate input gives finite values.
"""

from __future__ import annotations

from ..types import TensorType
from .nodes import HawkError, Op

#: The ray kinds.
RAY_OPS = ("ray_tri", "ray_aabb", "ray_sphere")

#: Operand counts.
ARITY = {"ray_tri": 6, "ray_aabb": 5, "ray_sphere": 5}

#: The rank-0 operands of each kind, by position (the rest are length-3 vectors).
_SCALARS = {"ray_tri": (5,), "ray_aabb": (4,), "ray_sphere": (3, 4)}

#: Floating-point operations one lane runs (multiplies, adds, compares, selects).
_COUNT = {"ray_tri": 70.0, "ray_aabb": 60.0, "ray_sphere": 45.0}


def result_type(kind: str, ts: list[TensorType]) -> TensorType:
    """The type ``kind`` returns over operands typed ``ts``: ``(4,)`` of the widest operand dtype."""
    scalars = _SCALARS[kind]
    for k, t in enumerate(ts):
        want = () if k in scalars else (3,)
        if t.shape != want or t.dtype not in ("f64", "f32"):
            raise HawkError(
                f"op {kind!r}: operand {k} is "
                + ("a rank-0" if want == () else "a length-3")
                + f" f64/f32 value, got {t!r}")
    return TensorType((4,), "f64" if any(t.dtype == "f64" for t in ts) else "f32")


def op_count(node: Op) -> float:
    """Operations ``node`` runs."""
    return _COUNT[node.kind]
