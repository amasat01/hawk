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
- **`u.at(row=r, col=c, edge="clamp")`** reads a per-lane input `u` at another lane of the `Grid`: name every axis, and `edge` (`"clamp"`, the default, `"wrap"` or `"zero"`) is what a coordinate outside its axis reads. **`u.near(col=1)`** is the short form: this lane's coordinates moved by the integer offsets on the axes you name (an axis you leave out keeps this lane's coordinate). `u.at(i)` reads at a flat lane index, which the same `edge` rules bring into the launch. Only an input that existed before the launch can be read at another lane.
- **`table.sample(x, y, ...)`** reads a `Table` at real coordinates in index units, one per axis (one to three): `filter="linear"` (the default) weighs the corners of the cell the point is in, `"nearest"` reads the element at `floor(x + 0.5)`; `edge="clamp"` (the default), `"wrap"` or `"zero"` is what a read past an axis gives. A whole coordinate is exactly the stored element.
- A **bare label** (`Grid["channel", ...]`, `Mutable["node"]`, `Table["elem", "c":3]`) is an extent given when
  you run, and only the first axis may be bare. Give it as `node=N` or by passing a plane of that length; one
  build serves every size. A label that a `Grid` axis shares takes the lane extent.
- The **pairs of N items** (`i < j`, no lane for `i >= j`) are `lane: Grid["pair"]` and `i, j = hawk.math.pair_index()`,
  run as `pair=N * (N - 1) // 2`. Lane `k` is the pair with `k = j * (j - 1) / 2 + i`, so `j` is the slow index and
  neighbouring lanes read neighbouring `i`. The index is exact in 64-bit integers on every target (`float32` builds: every
  `k < 2**31`; `float64`: every `k < 2**53`). Items live in a `Table["node"]` read as `x.at(i)` and `x.at(j)`; the outputs
  are one value per pair, a keyed `+=` gives a row sum (`hawk.scatter(rs)[j] += t`) and `|=` an adjacency bit mask:

  ```python
  @hawk.kernel
  def near(lane: Grid["pair"], x: Table["node"], mask: Mutable[UInt32]):
      i, j = hawk.math.pair_index()
      bit = hawk.math.select(x.at(i) * x.at(j) > 0.5, hawk.math.uint32(1) << hawk.math.uint32(j & 31), hawk.math.uint32(0))
      hawk.scatter(mask)[i * 4 + (j >> 5)] |= bit      # 4 words of 32 bits per row: up to 128 items
  ```
- A keyed **fold** other than a sum is a method of the same object: `hawk.scatter(y)[k].min(v)`, `.max(v)`, `.and_(bits)`
  and `.xor(bits)` combine every lane's value with the plane's start value in any order. `hawk.scatter(y, conflict="any")[k] = v`
  lets lanes share a key (some lane's value wins). `old = hawk.scatter(y)[k].exchange(v)`, `.compare_exchange(expected, v)` and
  `.add(v, old=True)` return what the cell held, in the order the lanes arrived. None of them has a derivative.
  `hawk.scatter(y, contract="ordered")[k].min(v)` (or `.max(v)`, scalar planes) fixes which lane wins each element: the lowest lane
  among those whose value equals the finished element. The plane is the one the unordered fold gives; the winners are made with
  the run (`<y>_winner`, unsigned, all ones where no lane wrote) and `plan.bind(...).winner("y")` reads them after a launch as an
  `int32` array with `-1` for none. A NaN element has no winner, and `+0.0` and `-0.0` tie.
- **`slot = hawk.append(out, value=v, when=cond)`** adds `v` to the list `out` (a plane declared with one axis, whose extent is
  its capacity) for each lane whose `when` holds, in no promised order, and gives the lane's place as an integer (`-1` when it
  asked for none). The run counts the lanes that asked in a one-element plane made with it: `plan.bind(...).count("out")` after
  a launch. More lanes than the capacity is not an error: the values past it are dropped and the count says how many asked.
  `jvp` stores the tangent at the place of its value. `vjp` reads the adjoint of the list at the place each lane took, which only the primal run knows: store the slot in an integer plane (`place = slot` with `place: Mutable[Int32]`) and bind that plane, as the primal run left it, to the reverse kernel; a kernel that stores no slot is refused by name.
- An **index set** (`Kind("screen", lanes=IndexSet())`) runs a list of lanes you built: launch lane `t` runs lane
  `map[t]`, with the map and its count bound as `active_map` and `active_count`. `hawk.sample_position()` is `t`
  where `hawk.math.sample_index()` is `map[t]`. See [Run only the lanes you list](howto_index_set).
- A **run-time trip count** is `for k in range(n, max=K)` with `n` an `Index` per lane: `n` iterations, held to
  `[0, K]`, never unrolled. Without `max=` `jvp` works and `vjp` is refused.
- A **group** is the lane count in which finished lanes are compacted away (`@hawk.kernel(group=32)`).
- A **contract** says how a sum over lanes is added up: `"auto"`, `"compensated"` or `"ordered"`.
  See [Across lanes](across_lanes).
- `Sorted[Index]` is an index plane whose values never decrease. It lets hawk add each run of equal keys once.
- A **target** is what an OSPREY library serves, one thing `hawk.build(targets=...)` builds (`cuda`, `host`).
  [Add a target](contributing/add_a_target) shows what hawk needs to find one.

`Table` (a shared read-only plane) and `Param` (one value for the run) keep their names. Calling them `Const`
and `Uniform` was considered and not taken.

## Notes on a plane

`Annotated[Vector[3], hawk.Meta(unit="km")]` attaches notes to a plane. hawk never reads them: they are written to the sidecar as `plane_meta`, by plane name, and a program that loads the kernel through eagle gets them back as `plane_meta` on the loaded kernel. Values must be JSON.

## Conditions

A comparison is a bool. Combine bools with `&`, `|`, `^`, `~` or with `and`, `or`, `not` (every side is evaluated: a kernel
body has no short circuit), and parenthesise each comparison around `&` and `|`: `x >= 0 & x <= 3` parses as
`x >= (0 & x) <= 3`, so write `(x >= 0) & (x <= 3)`. Arithmetic on a bool (`+ - * / **`, unary `-`, `abs`, `min`, `max`) is
refused: use `m.select(cond, x, 0.0)` for a number.
