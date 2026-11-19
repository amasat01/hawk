"""035_bump: bump (docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 11 (notebook))."""

import hawk
from hawk import Mutable, Scalar


@hawk.kernel
def bump(x: Scalar, acc: Mutable[Scalar]):
    acc = acc + x                                      # update in place: no return, the call gives back None


@hawk.kernel
def bump_and_double(x: Scalar, acc: Mutable[Scalar]):
    acc = acc + x                                      # in place, as above
    twice = 2.0 * x
    return twice                                       # and one returned output, named `twice`


CASE = {
    "id": "035_bump",
    "source": [
        "docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 11 (notebook)",
    ],
    "kernels": [
        "bump",
        "bump_and_double",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
