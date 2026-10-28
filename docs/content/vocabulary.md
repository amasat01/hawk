# Your own vocabulary

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
vocabulary/04_capstone
```

The three notebooks go in order; the two on derivatives, [your own derivative](vocabulary/03_your_own_derivative) and [tapes and stop rules](vocabulary/05_tapes_and_events), are under Derivatives in the tutorials. The worked cases, one per field, are on the [Domains](domains) pages;
[Orbits](vocabulary/06_orbits) is a good first one.

## Words for lanes and targets

Declaring the shape of your lanes brings a few more words; the [glossary](glossary) has each in a line.

- A **lane** is one work item, and the body of a kernel is one lane's code.
- The **lane space** is the shape a kernel gives its lanes. You declare it with a **`Grid`** parameter,
  `lane: Grid["channel", "sample":8192]`. The **fastest axis** is the last one declared: consecutive lanes
  along it are consecutive threads or SIMD lanes. A per-lane input you only read may be given **broadcast**,
  as in NumPy: a scalar, an array with axes of size 1, or one with fewer axes, lined up from the last axis.
  hawk binds your array as it is, without copying it, and compiles the kernel to read it along only the axes
  it varies along. The gradient of such an input comes back in its own shape. Planes a kernel writes, and the
  `terminated` mask, keep the exact lane shape.
- A **bare label** (`Grid["channel", ...]`, `Mutable["node"]`, `Table["elem", "c":3]`) is an extent given when
  you run, and only the first axis may be bare. Give it as `node=N` or by passing a plane of that length; one
  build serves every size. A label that a `Grid` axis shares takes the lane extent.
- The **pairs of N items** (`i < j`, no lane for `i >= j`) are `lane: Grid["pair"]` and `i, j = hawk.math.pair_index()`,
  run as `pair=N * (N - 1) // 2`. Lane `k` is the pair with `k = j * (j - 1) / 2 + i`, so `j` is the slow index and
  neighbouring lanes read neighbouring `i`. The index is exact in 64-bit integers on every target (`float32` builds: every
  `k < 2**31`; `float64`: every `k < 2**53`). Items live in a `Table["node"]` read as `x.at(i)` and `x.at(j)`; the outputs
  are one value per pair, a keyed `+=` gives a row sum (`hawk.scatter(rs)[j] += t`) and `policy="or"` an adjacency bit mask:

  ```python
  @hawk.kernel
  def near(lane: Grid["pair"], x: Table["node"], mask: Mutable[UInt32]):
      i, j = hawk.math.pair_index()
      bit = hawk.math.select(x.at(i) * x.at(j) > 0.5, hawk.math.uint32(1) << hawk.math.uint32(j & 31), hawk.math.uint32(0))
      hawk.scatter(mask, policy="or")[i * 4 + (j >> 5)] |= bit      # 4 words of 32 bits per row: up to 128 items
  ```
- An **index set** (`Kind("screen", lanes=IndexSet())`) runs a list of lanes you built: launch lane `t` runs lane
  `map[t]`, with the map and its count bound as `active_map` and `active_count`. `hawk.lane_position()` is `t`
  where `hawk.math.sample_index()` is `map[t]`. See [Run only the lanes you list](howto_index_set).
- A **run-time trip count** is `for k in range(n, max=K)` with `n` an `Index` per lane: `n` iterations, held to
  `[0, K]`, never unrolled. Without `max=` `jvp` works and `vjp` is refused.
- A **group** is the lane count in which finished lanes are compacted away (`@hawk.kernel(group=32)`).
- A **policy** says how a sum over lanes is added up: `"auto"`, `"compensated"` or `"ordered"`.
  See [Across lanes](across_lanes).
- `Sorted[Index]` is an index plane whose values never decrease. It lets hawk add each run of equal keys once.
- A **target** is a Writer and a Wrapper with their compile facts, one thing `hawk.build(targets=...)` builds
  (`cuda`, `host`, `c`). The **Writer** says how a language writes each construct, the **Wrapper** how a kernel is
  wrapped for one execution model. [Add a target](contributing/add_a_target) shows the row.

`Table` (a shared read-only plane) and `Param` (one value for the run) keep their names. Calling them `Const`
and `Uniform` was considered and not taken.
