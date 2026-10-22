# Quickstart

The shortest path from install to a first answer: one kernel, built for
the CPU, run, and read back. New words (plane, kernel kind, ...) are in the
[glossary](glossary).

Install first (`pip install raptor-hawk raptor-eagle matplotlib` for the CPU route; see [installation](installation)).
Then write a kernel. It is a plain Python function decorated
`@hawk.kernel`; each parameter's type says what role it plays for every
sample:

```python
import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def scale(x: Scalar,              # a per-sample input
          a: Param, b: Param,     # a shared constant: one value for the run
          y: Mutable[Scalar]):    # an output plane the body assigns
    y = a * x + b
```

Build it for the host (the CPU alone, no GPU and no eagle involved), load
it and run it:

```python
import pathlib
import tempfile

import hawk
import numpy as np

work_dir = pathlib.Path(tempfile.mkdtemp())
hawk.build([scale], work_dir, targets=("host",))

host_kernel = hawk.load(work_dir, "scale")
x = np.linspace(0.0, 7.0, 8)
y = np.zeros(8)
hawk.run(host_kernel, x=x, a=2.0, b=1.0, y=y)
y
```

The printed answer, as recorded in the
[first tutorial's](tutorials/01_your_first_kernel) executed run:

```text
array([ 1.,  3.,  5.,  7.,  9., 11., 13., 15.])
```

That is `2x + 1` for every `x`. The same kernel runs on a GPU through
eagle with no change to its source; the tutorial shows how.

## Where next

- [Tutorials](tutorials): nine notebooks that build on each other,
  starting with this kernel in full.
- {doc}`hawk in your domain <vocabulary>`: declare your own kernel family, derivative and
  tape.
- [How-to guides](examples): one task per notebook.
