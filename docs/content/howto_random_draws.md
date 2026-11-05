# Draw random numbers in a kernel

`hawk.math.random` draws are counter-based: a draw is a function of `(seed, lane, counter)` and nothing else, so there is no generator state to carry. Pass an integer `seed` and a `counter` (both `Index` values, or Python integers in the body); every lane is its own stream.

```python
import hawk
import hawk.math as m
from hawk import Index, Mutable, Scalar, Vector

@hawk.kernel
def draws(seed: Index, step: Index, y: Mutable[Vector[7]]):
    y = m.vec(m.random.uniform(seed, step, 1.0, 3.0),     # open interval (1, 3)
              m.random.normal(seed, step, 10.0, 2.0),     # mean 10, sd 2
              m.random.lognormal(seed, step),             # exp(mu + sigma z)
              m.random.exponential(seed, step, 0.5),      # rate 0.5
              m.random.bernoulli(seed, step, 0.25),       # 1.0 or 0.0
              m.random.uniform_int(seed, step, 1, 6),     # 1..6, both ends reachable
              m.random.poisson1(seed, step))              # counts, mean 1
```

Every draw returns a real value, so a count or a coin is `1.0`, `2.0`, and so on. The flat names (`m.random_uniform`, `m.random_normal`, ...) are the same functions. Scale and shift with the optional arguments or with ordinary arithmetic; both are differentiable in `lo`, `hi`, `mean` and `sd`.

## Correlated normals

`m.random.multivariate_normal(seed, counter, mean, cholesky)` returns `mean + L z`, `L` the lower Cholesky factor you pass. `m.mvn_sample(seed, counter, mean, cov)` takes the covariance and factors it for you.

```python
from hawk import Matrix

@hawk.kernel
def pair(seed: Index, step: Index, L: Matrix[2, 2], w: Mutable[Vector[2]]):
    w = m.random.multivariate_normal(seed, step, m.vec(1.0, 2.0), L)
```

The vector draw uses sub-counters `counter * D + k`, and `poisson1` uses `counter * 16 + k`. Do not mix a counter `c` with a draw that already uses its sub-counters, or the two will share uniforms; give each use its own counter.

## What stays the same

- **Lanes.** Lane `t` draws from its own position in the launch, so the same `(seed, counter)` gives a different value in each lane, and the same value in the same lane on every run.
- **Kinds.** A uniform and a normal on the same `(seed, counter)` are unrelated: each kind keys its own stream.
- **Slices.** Running part of a launch (`hawk.run(..., base=2, count=3)`) gives the lane values the whole launch gives.
- **Host and device.** `uniform` is bit-identical on the CPU and on the GPU, on every run and every replay of a captured graph. The other draws go through the math library and agree to the ULP contract.
- **Seeds.** A new `seed` is a new set of streams. To step through a simulation, keep the seed and raise the counter.

A seed is an integer: a float seed is refused by name. Draws have no derivative with respect to `seed` or `counter`; `vjp` and `jvp` carry the rest.
