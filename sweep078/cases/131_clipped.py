"""131_clipped: clipped (docs/content/vocabulary/when_hawk_cant_spell_it.md: line 10 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
from hawk import Mutable, Scalar


def soft_floor(x, lo):
    r = x
    if x < lo:
        r = lo
    return r


@hawk.kernel
def clipped(a: Scalar, b: Scalar, y: Mutable[Scalar]):
    y = soft_floor(a, 0.0) + soft_floor(b, 1.0)


CASE = {
    "id": "131_clipped",
    "source": [
        "docs/content/vocabulary/when_hawk_cant_spell_it.md: line 10 (markdown)",
    ],
    "kernels": [
        "clipped",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
