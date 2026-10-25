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
vocabulary/07_fem_assembly
vocabulary/08_particle_binning
vocabulary/09_heat_diffusion
vocabulary/10_fluids
vocabulary/11_batched_contraction
vocabulary/12_monte_carlo_ensemble
```

The tutorials go in order; [Orbits](vocabulary/06_orbits) is a short worked
example of a vocabulary for orbital mechanics.

## The same words in other fields

Each of these is a short notebook that runs on a CPU at a small size and on a GPU at a larger one.

- [Finite elements](vocabulary/07_fem_assembly): assemble a load vector from a triangle mesh. One lane per
  (triangle, corner), `hawk.scatter` into the nodes, `t[e, a]` and `.at(...)` for reads, and the ways to read
  where a lane is.
- [Particle binning](vocabulary/08_particle_binning): add each particle's weight into its cell, with
  `policy=` for the sum and `Sorted[Index]` for keys that are already in order.
- [Heat diffusion](vocabulary/09_heat_diffusion): a 2-D `Grid` stencil, with neighbours read by `.at(...)`.
  Animated.
- [Fluids](vocabulary/10_fluids): shallow water on a 2-D `Grid`, with the total water checked.
  Animated.
- [Batched small matrices](vocabulary/11_batched_contraction): a matrix-vector product and a quadratic form per batch
  element on `Grid["batch", "i":3, "j":3]`, with `hawk.sum(..., over=...)`, checked against `einsum`.
- [Monte Carlo trajectories](vocabulary/12_monte_carlo_ensemble): one projectile's step with drag and a stop rule, run until every
  sample has landed or left the field, checked against the exact solution. Animated.

## Words for lanes and targets

Declaring the shape of your lanes brings a few more words; the [glossary](glossary) has each in a line.

- A **lane** is one work item, and the body of a kernel is one lane's code.
- The **lane space** is the shape a kernel gives its lanes. You declare it with a **`Grid`** parameter,
  `lane: Grid["channel", "sample":8192]`. The **fastest axis** is the last one declared: consecutive lanes
  along it are consecutive threads or SIMD lanes.
- A **group** is the lane count in which finished lanes are compacted away (`@hawk.kernel(group=32)`).
- A **policy** says how a sum over lanes is added up: `"auto"`, `"compensated"` or `"ordered"`.
  See [Across lanes](across_lanes).
- `Sorted[Index]` is an index plane whose values never decrease. It lets hawk add each run of equal keys once.
- A **target** is a Writer and a Wrapper with their compile facts, one thing `hawk.build(targets=...)` builds
  (`cuda`, `host`, `c`). The **Writer** says how a language writes each construct, the **Wrapper** how a kernel is
  wrapped for one execution model. [Add a target](contributing/add_a_target) shows the row.

`Table` (a shared read-only plane) and `Param` (one value for the run) keep their names. Calling them `Const`
and `Uniform` was considered and not taken.
