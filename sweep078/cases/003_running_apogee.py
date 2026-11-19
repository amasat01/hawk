"""003_running_apogee: running_apogee (README.md: line 112 (readme))."""

import hawk
from hawk import Mutable, Scalar, Terminated, Vector
from hawk.math import maximum, norm


@hawk.kernel
def running_apogee(position: Vector[3], terminated: Terminated,
                    r_max: Mutable[Scalar]):
    r_max = maximum(r_max, norm(position))


CASE = {
    "id": "003_running_apogee",
    "source": [
        "README.md: line 112 (readme)",
    ],
    "kernels": [
        "running_apogee",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
