# Your own Python step, and how to give it a gradient

A graph records kernel launches. When one step of your computation is easier to write in plain Python (a call into another
library, a lookup, a piece of bookkeeping), wrap the function with `hawk.host_step` and add it to the graph next to the kernels.
hawk's library runs the function where a launch would run, in the order the graph records.

## The step

```python
import numpy as np
import hawk

@hawk.host_step
def double(x, y: hawk.Mutable):
    np.multiply(x, 2.0, out=y)
```

An argument annotated `hawk.Mutable` (bare, or subscripted as in a kernel) may be written, and read first. An argument with no
mark is read-only: it arrives as a read-only array, and writing to it is refused. The arguments arrive as arrays of the library
your own arrays come from (a NumPy array for NumPy), made from the memory you bound; they are valid only during the call.

A step may return the values of its `Mutable` arguments instead of writing them: one array for one argument, a tuple in the
order of the arguments for several. hawk copies each returned array into its argument; a wrong count, shape or dtype is refused,
naming the step and the argument.

## Binding and adding

Calling the step binds it to arrays, by keyword or by position; every argument is given.

```python
a, b = np.arange(4.0), np.zeros(4)
g = hawk.Graph()
g.add(double(x=a, y=b))
g.build()
g()                          # b is now 2 * a
```

Inside a `@hawk.Graph.builder` the call records the step and returns the arrays it writes (`b` here, a tuple for several, `None`
for none), exactly as a kernel call returns its outputs:

```python
@hawk.Graph.builder
def run(a, b) -> hawk.Graph:
    g = hawk.Graph()
    double(a, b)
    return g
```

## Giving it a gradient

`@<step>.reverse` takes a function of three objects, whose attributes are the step's arguments:

```python
@double.reverse
def double_reverse(value, cotangent, gradient):
    gradient.x += 2.0 * cotangent.y
```

- `value.<argument>` holds the arguments as the step saw them.
- `cotangent.<argument>` holds the incoming adjoint of each `Mutable` argument (read-only).
- `gradient.<argument>` holds a zeroed array for every argument the step may read, which is every argument. Add to it in place
  (`gradient.x += ...` or `gradient.x[...] = ...`); the library accumulates it into the gradients `hawk.reverse` returns.

A name the object does not hold is refused by name, and so is replacing an array instead of writing into it.

A worked example with a closed-form check: `z = 3 y^2` as a step between two kernels, differentiated with `hawk.reverse`.

```python
from hawk import Mutable, Scalar

@hawk.kernel
def scale(x: Scalar, w: Scalar, y: Mutable[Scalar]):
    y = w * x

@hawk.kernel
def square(y: Scalar, z: Mutable[Scalar]):
    z = y * y

@hawk.host_step
def triple_square(y, t: Mutable):
    t[...] = 3.0 * y * y

@triple_square.reverse
def triple_square_reverse(value, cotangent, gradient):
    gradient.y += 6.0 * value.y * cotangent.t

x, w = np.linspace(0.5, 2.5, 9), np.linspace(1.7, 2.5, 9)
y, t, q = np.zeros(9), np.zeros(9), np.zeros(9)
g = hawk.Graph(record=True)
g.add(hawk.deploy(scale, reverse=True).bind(x=x, w=w, y=y))
g.add(triple_square(y=y, t=t))
g.add(hawk.deploy(square, reverse=True).bind(y=t, z=q))
g.build()
g()
grad = hawk.reverse(g.record, {"square.z": np.ones(9)})
# q = 9 (w x)^4, so dq/dx = 36 (w x)^3 w
np.testing.assert_allclose(grad["x"], 36.0 * (w * x) ** 3 * w, rtol=1e-12)
```

A step with no reverse runs and replays as usual; asking for a gradient that flows through it is refused, naming the step.

## In a run

`hawk.run` and the stepping run (`hawk.until_done`, `hawk.simulate`) take a kernel or a plan, a bound step, a built graph, or any
iterable of them (a list, a tuple, a generator). A graph a `@hawk.Graph.builder` function made is passed as it is.

- `hawk.run` runs the items once, in the order written, and waits: `hawk.run([double(x=a, y=b), kernel], x=b, ...)`. A step alone
  runs alone: `hawk.run(double(x=a, y=b))`.
- A stepping run runs each step and graph once per step, at its place among the kernels:
  `[before, kernel, after]` runs in that order in every step. A recorded run takes steps ahead of the kernels only and
  refuses one after a kernel by name. It needs at least one kernel or plan, which is what counts the steps.
  `hawk.until_done([double(x=a, y=b), kernel], max_steps=100, ...)`. A step must not write a sample that has finished.
- With `record=True`, `hawk.reverse(run.record, cotangents)` goes through each kernel and, with its `@<step>.reverse`, each step.

## Limits

- Every argument must be given; an omitted `Mutable` argument is not allocated for you.
- An exception raised by your function reaches the caller as it was raised, and the graph stays usable.
- Not available: a step that is not differentiable by declaration, a step inside a conditional or loop on a device, and device
  modes other than the synchronized one.
- The arrays of a bound step are recorded with the graph: `Graph.bind` refuses a parameter a step holds; call the builder again.
