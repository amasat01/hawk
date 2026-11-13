# Run only the lanes you list

A kernel's lanes are all of its lane space. When you already know which lanes matter, a conjunction screen that kept 40 pairs of 100 000 items, or the cells that are occupied, declare an **index set** and pass the list. Launch lane `t` runs lane `map[t]`, for `t` below the count, and nothing else runs.

```python
import hawk
import hawk.math as m
from hawk import Grid, Mutable, Table
from hawk.ext import IndexSet, Kind

SCREEN = Kind("screen", lanes=IndexSet())

@SCREEN
def miss(lane: Grid["pair"], pos: Table["node"], d: Mutable[hawk.Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = (pos.at(i) - pos.at(j)) ** 2
    d = t                              # written at the pair's own id
    hawk.scatter(rs)[j] += t           # a keyed sum over the listed pairs only
```

Bind two extra planes next to your own: `active_map` (the lane ids) and `active_count` (a one-element `uint32`). The names are the ones `eagle.ActiveSet` already fills, so a map it built feeds the same kernel. Every id must be in `[0, extent)` and listed once; the bind refuses the first one that is not and names its position. The check runs at bind. If you rewrite a bound map in place afterwards, its contents are your contract until you run `eagle.check_index_set(map, count, extent)`; a captured graph replays without checking. The extent is the size of the dense space, here `pair=N * (N - 1) // 2`.

```python
hawk.run(hawk.load(directory, "miss"), pos=pos, d=d, rs=rs, pair=n_pairs,
         active_map=np.array([3, 17, 44], dtype=np.int32),
         active_count=np.array([3], dtype=np.uint32))
```

## What changes

- `hawk.math.sample_index()` is the lane id, `map[t]`. `hawk.sample_position()` is `t`, so `hawk.scatter(out)[hawk.sample_position()] += v` writes a compacted output in launch order.
- Keyed sums and scatters are allowed. They are added in the order the lanes arrive, so they match a dense launch to the ULP contract, not bit for bit. `vjp` and `jvp` carry the map.
- A kernel with `sample_sum`, a segmented dispatch and the `c` target refuse an index set by name.

## A batch that changes size: `IndexSet(prefix=True)`

When the lanes you want are always the first `count` of them, skip the map. Declare `lanes=IndexSet(prefix=True)`
and bind one plane, `active_count` (a one-element `uint32`). Launch lane `t` runs lane `t` for `t < count`, and the
count is read from the plane on every run, so one build, and one captured graph, serves any batch size up to the
bound:

```python
BATCH = Kind("batch", lanes=IndexSet(prefix=True))
```

- Sums, keyed sums and `vjp` cover the live lanes only. Lanes past the count are not touched: their outputs keep the
  values they held.
- Bind at the largest size. A count above it is refused at bind. Under eagle, `bind(..., capacity=n_max)` makes the
  count plane and `set_count(m)` changes it between replays.
- Writing the count plane directly, without `set_count`, is your contract: nothing then checks the count.

## When a map kernel needs more registers than you have

A map-form lane index is a loaded value, so it stays live for the whole kernel, one register beyond the prefix form. When registers are tight, use the prefix form. When the kernel must fit an occupancy step instead, build with `launch_bounds=(threads, blocks_per_sm)`, for example `build_bundle(..., launch_bounds=(256, 2))`. It puts `__launch_bounds__` on the kernel (the compiler fits the registers to it, the values do not change), records the block in the sidecar so a launch uses it, and needs `threads` a multiple of 32 with `threads * blocks_per_sm` at most 2048. Check the result: a bound that is too tight spills, and the sidecar reports the local memory.

## Emit without compiling

To compile the source yourself, ask for the text and the plane list instead of a build: `hawk.artifact.emit_source(kernel, kind=...)` returns `.sources["cuda"].text` and `.sources["host"].text`, the same text `build` writes (the map or count planes included), and `.sidecar`, the sidecar without what only a compile knows (`ptxas`, the local-memory and wide-value figures; `format` is the target's default).

## A map past 2**31 lanes

A pair grid of 65 537 items has more than `2**31 - 1` pairs. The map word is then 64-bit: build with `IndexSet(width=64)`, bind an `int64` map. `IndexSet()` is 32-bit, and `hawk._lane_bind.index_width(extent)` says which one an extent needs. A `float32` kernel cannot hold such a lane id and is refused; an `int64` map over a 32-bit extent is refused rather than narrowed.

## A count known only at run time

`for k in range(n, max=K)` runs `n` iterations, `n` held to `[0, K]`. `n` is an `Index` per lane, so each lane can run its own segment:

```python
from hawk import Index, Scalar

@hawk.kernel
def segment_sum(lo: Index, hi: Index, t: Table[Scalar], y: Mutable[Scalar]):
    s = 0.0
    for k in range(hi - lo, max=8):
        s = s + t.at(lo + k)
    y = s
```

The loop is never unrolled. Without `max=` the bound is the int range: forward and `jvp` work, and `vjp` is refused ("give `max=` so the tape has a size").
