"""041_draws: draws (docs/content/howto_random_draws.md: line 5 (markdown))."""
# ruff: noqa: F841
# Kernel bodies assign to output parameters and name axes in string annotations: the DSL reads both.

import hawk
import hawk.math as m
from hawk import Index, Mutable, Vector


@hawk.kernel
def draws(seed: Index, step: Index, y: Mutable[Vector[7]]):
    y = m.vec(m.random.uniform(seed, step, 1.0, 3.0),     # open interval (1, 3)
              m.random.normal(seed, step, 10.0, 2.0),     # mean 10, sd 2
              m.random.lognormal(seed, step),             # exp(mu + sigma z)
              m.random.exponential(seed, step, 0.5),      # rate 0.5
              m.random.bernoulli(seed, step, 0.25),       # 1.0 or 0.0
              m.random.uniform_int(seed, step, 1, 6),     # 1..6, both ends reachable
              m.random.poisson1(seed, step))              # counts, mean 1


CASE = {
    "id": "041_draws",
    "source": [
        "docs/content/howto_random_draws.md: line 5 (markdown)",
    ],
    "kernels": [
        "draws",
    ],
    "targets": ('host',),
    "inputs": None,
    "oracle": None,
}
