# Whole-plane operations: reduce, scan, compact, sort, reorder, copy, matmul and fft

Some steps of a computation are not a body run for every sample: a sum over all samples, a copy, a matrix product, a Fourier
transform. They are called like a kernel. Outside a graph they run once; inside a `@hawk.Graph.builder` they record one node
of the graph, in order with the kernel launches, and a replay runs them again over the planes as they are then.

```python
import numpy as np
import hawk

x = np.random.default_rng(0).normal(size=(3, 50))      # three columns of fifty samples (samples last)
hawk.reduce(x, "sum")                                  # shape (3,): the sum of each column; "min", "max", "all", "any" too
hawk.matmul(a, b, transpose_b=True, accumulate=False)  # op(a) @ op(b)
hawk.scan(x[0], "sum")                                 # running sum of one axis; "min", "max"; exclusive=True shifts it
hawk.compact(keep)                                     # (kept, count): the lanes a bool plane keeps, in lane order
hawk.sort(keys, bound=K)                               # (order, starts): the stable sort of the lanes by integer key
hawk.reorder([pos, vel], order)                        # in place: afterwards p[j] holds what p[order[j]] held
hawk.copy(src, dst)
hawk.fft.r2c(y, axes=None, norm="backward")            # also hawk.fft.c2c and hawk.fft.c2r
```

The result is a new array in the memory of the first operand, or the array you pass as `out=`. A complex plane is a float plane
whose last axis has extent 2 (real part, imaginary part). `hawk.reduce(p, "sum", contract="fixed")` adds in the fixed leaf
order; the default is `"compensated"`. The runtime checks every shape and dtype, and its reason for a refusal is shown as it
gave it. The reverse of a recording that holds a copy, a matrix product, a transform or a reduce over one flat plane with
`op="sum"` is built by the library, and so is that of a minimum or a maximum over one flat plane (the cotangent goes to the first
lane that holds the extreme, and nowhere else); a reduce over several columns is refused by name when the recording is made with
`record=True`.

A scan runs along one axis, `hawk.scan(p, "sum" | "min" | "max", exclusive=False, out=None)`: `out[i]` folds `p[0..i]`, or with
`exclusive=True` folds `p[0..i-1]` and starts with the identity (0 for a sum, the largest value for a minimum, the smallest for a
maximum). It takes int32, int64, uint32, uint64, float32 and float64 planes of one axis and scalar values; a plane of several
axes is refused by name. An integer sum wraps and a minimum or maximum is exact; a real sum has the same bits on every run and at
any thread count but is not the bits of a left-to-right loop, and stays within `i * u * sum(|p[0..i]|)` of the exact value
(`u` the unit roundoff). The reverse of a recording that holds a scan is built by the library: a sum scan gives the cotangent of `in` the suffix scan of the
cotangent of `out` (the same shifted by one lane when exclusive), and a minimum or maximum scan gives the cotangent of `out[i]` to the
running winner of `i`, the lowest lane among equals.

`kept, count = hawk.compact(keep, out=None)` lists the lanes a bool plane `keep` leaves set: `kept[:count]` holds their indices in
lane order (an int32 plane the shape of `keep`) and `count` is a one-element uint32 plane. Entries of `kept` from `count` on are not
written, and the result is the same on every run and at any thread count. Inside a graph builder `count` is a condition cell
(`if count:`) and a loop limit, read at every replay. A `keep` of several axes is refused by name; a recording that holds a compaction
reverses, and no cotangent flows through `kept` or `count`.

`order, starts = hawk.sort(keys, bound=K, out=None)` sorts the lanes of an integer plane `keys` of one axis, every key in `[0, K)`:
`order[j]` is the lane that lands at sorted place `j` and `starts[k]` is the first sorted place of key `k`, with `starts[K]` the
lane count, so the lanes of key `k` are `order[starts[k]:starts[k + 1]]`. Lanes of equal key keep their lane order, and the result
is the same on every run and at any thread count. `order` is an int32 plane the shape of `keys` and `starts` an int32 plane of
`K + 1` entries; `out=(order, starts)` passes your own. A key outside `[0, K)` fails the call, or in a graph builder the replay,
naming how many there are; a real `keys` plane, a `K` below 1 and a `keys` of several axes are refused by name. A recording that holds a
sort reverses, and no cotangent flows through `order` or `starts`.

`hawk.reorder(planes, order)` permutes a list of planes in place so that `p[j]` holds what `p[order[j]]` held. `order` is an int32
or uint32 plane of `n` entries and every plane has `n` lanes as its last axis; a plane of a vector value, shape `(rows, n)`,
moves each row by the same order. Reordering by the output of a sort puts every plane in key order:

```python
order, starts = hawk.sort(keys, bound=K)
hawk.reorder([pos, vel, mass], order)                  # the same as pos[order] for each plane, in place
```

The result is the same on every run and at any thread count. An entry of `order` outside `[0, n)` fails the call, or in a graph
builder the replay, naming how many there are, and moves nothing. A repeated entry is not detected and leaves a lane twice, so an
`order` that is not a permutation is the author's promise. A recording that holds a reorder replays it over the planes and the
order as they are then; its reverse scatters the cotangent of every real plane back through the same `order`.

`value, where = hawk.reduce(p, "min" | "max", with_index=True)` also returns an int32 plane with the lane of each column's winner:
a tie goes to the lowest lane and a NaN never wins; a column of nothing but NaN has the value NaN and the lane `n`. `"any"` and
`"all"` take a bool plane and give a bool plane; pass a one-element uint32 plane as `out=` to get a cell that `if cell:` tests in a
graph builder, read at every replay. A position with another op is refused by name; the reverse of a recording that holds a
minimum or a maximum goes to the lane it chose.

## Streams

A device queues its work on one ordered queue, its own. `hawk.Stream(device="cuda:0")` makes another that the library owns, and
`hawk.Stream(theirs)` adopts a stream of yours, a raw handle or an object with `__cuda_stream__`, which stays yours to free.
`with stream:` puts every launch made inside it on that stream (a kernel call, a whole-plane operation, a graph replay), scopes
nest, and the outer one is back on leaving, an exception included. `stream.synchronize()` returns when the work queued on it has
run. A scope belongs to the thread that opened it: a thread you start inside it is in no scope, and one that should be in it runs inside `contextvars.copy_context().run`. A stream scope cannot be opened inside a graph builder or block: there the queues follow from `parallel()` branches and
`wait(planes)`. A host context is one queue, so `hawk.Stream(device=...)` on it is refused with the library's reason.

## Devices

`hawk.asarray(x, device="cuda:1")` puts an array on a device by its number; without `device` it is on the host. A call runs on the
device of its planes, so it follows where you put them; `hawk.copy(src, dst)` between two devices is the transfer, and a call
whose planes sit on two devices is refused by name with that copy as the way. The library answers for every device: a host
context has one member.

## Events

`hawk.Event()` is a mark on a stream. `e.record()` marks the stream in scope, `e.wait()` makes the stream in scope wait for the
mark (the calling thread does not), and `stop.elapsed_ms(since=start)` is the time between two recorded marks, taken once both
have been reached. An event lives on the device of the stream it was first recorded on, so the stream must be one the library
made, or one that has had a launch. On a host a record stamps a steady clock and a wait returns at once. Waiting for or timing an
event that was never recorded is refused by name, and so is any use of an event inside a graph builder or block: there the order
is `wait(planes)` and the time is taken around the replay.
