# hawk in your domain

Teach hawk the objects of your field once: a spacecraft state, a portfolio, a
reaction network. Every kernel you write then speaks those terms, compiles for
CPU and GPU, and differentiates.

The mechanism is a **vocabulary**: the named per-sample arrays and parameters a
family of kernels shares. Here is the same small kernel, one step of a body
falling toward a central mass, written twice.

With raw planes, every kernel repeats every name:

```python
@hawk.kernel
def kick(x: Scalar, y: Scalar, vx: Scalar, vy: Scalar, dt: Param, mu: Param,
         x_next: Mutable[Scalar], y_next: Mutable[Scalar],
         vx_next: Mutable[Scalar], vy_next: Mutable[Scalar]):
    d3 = (x * x + y * y) ** 1.5
    vx_next = vx - dt * mu * x / d3
    vy_next = vy - dt * mu * y / d3
    x_next = x + dt * vx_next
    y_next = y + dt * vy_next
```

With a vocabulary, the state is declared once and the kernel reads like the
physics:

```python
class TwoBody(KernelKind, slug="two_body"):
    r: Vector[2]
    v: Vector[2]
    dt: Param
    mu: Param
    guard = DATA_ONLY


@TwoBody
def kick(r, v, dt, mu, r_next: Mutable[Vector[2]], v_next: Mutable[Vector[2]]):
    v_next = v - dt * mu * r / norm(r) ** 3
    r_next = r + dt * v_next
```

Both build with `hawk.build` and give the same numbers. The vocabulary version
is the one you extend, share across models, and give derivatives to.

```{toctree}
:maxdepth: 1

vocabulary/01_a_family_of_kernels
vocabulary/02_extend_by_inheritance
vocabulary/03_your_own_derivative
vocabulary/04_capstone
vocabulary/05_tapes_and_events
vocabulary/06_orbits
```

The tutorials go in order; [Orbits](vocabulary/06_orbits) is a short worked
example of a vocabulary for orbital mechanics.
