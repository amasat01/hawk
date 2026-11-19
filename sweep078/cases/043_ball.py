"""043_ball: ball (docs/content/howto_rays_and_meshes.md: line 15 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Mutable, Vector


@hawk.kernel
def ball(o: Vector[3], d: Vector[3], y: Mutable[Vector[4]]):
    y = m.ray_sphere(o, d, m.vec(0.0, 0.0, 3.0), 1.0)


CASE = {
    "id": "043_ball",
    "source": [
        "docs/content/howto_rays_and_meshes.md: line 15 (markdown)",
    ],
    "kernels": [
        "ball",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
