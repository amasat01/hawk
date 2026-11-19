"""034_one: one (docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 7 (notebook))."""

import hawk
from hawk import Scalar


@hawk.kernel
def one(x: Scalar):
    return x + 1.0                                     # one value: you get one array


@hawk.kernel
def pair(x: Scalar):
    return x + 1.0, x * 3.0                            # a tuple: you get a tuple


@hawk.kernel
def many(x: Scalar):
    return [x + 1.0, x * 3.0]                          # a list: you get a list


@hawk.kernel
def named(x: Scalar):
    return {"up": x + 1.0, "triple": x * 3.0}          # a dict: you get a dict, with your keys


@hawk.kernel
def nested(x: Scalar):
    up = x + 1.0
    return {"up": up, "rest": [x * 3.0, (x * 4.0, x * 5.0)]}   # any nesting of the above


CASE = {
    "id": "034_one",
    "source": [
        "docs/content/examples/06_every_form_of_a_kernel.ipynb: cell 7 (notebook)",
    ],
    "kernels": [
        "one",
        "pair",
        "many",
        "named",
        "nested",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
