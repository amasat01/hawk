# Quickstart

The shortest path from install to a first answer: one kernel, built for
the CPU, run, and read back. New words (plane, kernel kind, ...) are in the
[glossary](glossary).

Install first (`pip install raptor-hawk raptor-eagle matplotlib` for the CPU route; see [installation](installation)).
Then write a kernel. It is a plain Python function decorated
`@hawk.kernel`; each parameter's type says what role it plays for every
sample, and `return` is the output:

```python
import hawk
import numpy as np
from hawk import Param, Scalar


@hawk.kernel
def scale(x: Scalar,              # a per-sample input
          a: Param, b: Param      # a shared constant: one value for the run
          ) -> Scalar:            # optional, and checked: what the body returns
    return a * x + b              # the output, one value per sample


x = np.linspace(0.0, 7.0, 8)
scale(x, 2.0, 1.0)                # call it: the first call builds and caches it
```

The printed answer, as recorded in the
[first tutorial's](tutorials/01_your_first_kernel) executed run:

```text
array([ 1.,  3.,  5.,  7.,  9., 11., 13., 15.])
```

That is `2x + 1` for every `x`, on the CPU because `x` is a NumPy array. Give it a
CuPy array and it runs on the GPU, with no change to its source. A kernel can return a
tuple, a list or a dict, and can update a `Mutable` in place:
[every form, executed](examples/06_every_form_of_a_kernel). Building a kernel yourself
(`hawk.build`, `hawk.load`, `hawk.run`) is what the tutorials start from.

## Where next

- [Tutorials](tutorials): nine notebooks that build on each other,
  starting with this kernel in full.
- {doc}`hawk in your domain <vocabulary>`: declare your own kernel family, derivative and
  tape.
- [How-to guides](examples): one task per notebook.
