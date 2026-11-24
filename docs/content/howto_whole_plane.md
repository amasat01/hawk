# Whole-plane operations: reduce, copy, matmul and fft

Some steps of a computation are not a body run for every sample: a sum over all samples, a copy, a matrix product, a Fourier
transform. They are called like a kernel. Outside a graph they run once; inside a `@hawk.Graph.builder` they record one node
of the graph, in order with the kernel launches, and a replay runs them again over the planes as they are then.

```python
import numpy as np
import hawk

x = np.random.default_rng(0).normal(size=(3, 50))      # three columns of fifty samples (samples last)
hawk.reduce(x, "sum")                                  # shape (3,): the sum of each column; "min", "max", "all" too
hawk.matmul(a, b, transpose_b=True, accumulate=False)  # op(a) @ op(b)
hawk.copy(src, dst)
hawk.fft.r2c(y, axes=None, norm="backward")            # also hawk.fft.c2c and hawk.fft.c2r
```

The result is a new array in the memory of the first operand, or the array you pass as `out=`. A complex plane is a float plane
whose last axis has extent 2 (real part, imaginary part). `hawk.reduce(p, "sum", contract="fixed")` adds in the fixed leaf
order; the default is `"compensated"`. The runtime checks every shape and dtype, and its reason for a refusal is shown as it
gave it. The reverse of a recording that holds a copy, a matrix product, a transform or a reduce over one flat plane with
`op="sum"` is built by the library; a minimum, a maximum or a sum over several columns is refused by name when the recording
is made with `record=True`.

## Streams

A device queues its work on one ordered queue, its own. `hawk.Stream(device="cuda:0")` makes another that the library owns, and
`hawk.Stream(theirs)` adopts a stream of yours, a raw handle or an object with `__cuda_stream__`, which stays yours to free.
`with stream:` puts every launch made inside it on that stream (a kernel call, a whole-plane operation, a graph replay), scopes
nest, and the outer one is back on leaving, an exception included. `stream.synchronize()` returns when the work queued on it has
run. A stream scope cannot be opened inside a graph builder or block: there the queues follow from `parallel()` branches and
`wait(planes)`. A host context is one queue, so `hawk.Stream(device=...)` on it is refused with the library's reason.

## Events

`hawk.Event()` is a mark on a stream. `e.record()` marks the stream in scope, `e.wait()` makes the stream in scope wait for the
mark (the calling thread does not), and `stop.elapsed_ms(since=start)` is the time between two recorded marks, taken once both
have been reached. An event lives on the device of the stream it was first recorded on, so the stream must be one the library
made, or one that has had a launch. On a host a record stamps a steady clock and a wait returns at once. Waiting for or timing an
event that was never recorded is refused by name, and so is any use of an event inside a graph builder or block: there the order
is `wait(planes)` and the time is taken around the replay.
