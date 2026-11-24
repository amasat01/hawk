# When hawk can't spell it

Four ways to reach past hawk's own arithmetic, in the order to try them.

## 0. A plain function (try this first)

Most of the time a shared piece of a kernel needs no hawk feature at all. Write it as an ordinary Python
function and call it from the kernel with traced values:

```python
def soft_floor(x, lo):
    r = x
    if x < lo:
        r = lo
    return r

@hawk.kernel
def clipped(a: Scalar, b: Scalar, y: Mutable[Scalar]):
    y = soft_floor(a, 0.0) + soft_floor(b, 1.0)
```

A function you define yourself gets the same treatment as the kernel body: `if`, `for` and `break` are traced, once
per function, and the call is inlined. `vjp` and `jvp` go through its body, so it needs no rule. Library functions
(hawk, installed packages, the standard library) are never rewritten. Reach for this before anything below, and
factor repeated blocks of a kernel this way.

## 1. A custom primitive

If the function is a **composition of ops hawk already has** — `log`,
`exp`, `norm`, `dot`, the arithmetic operators — and you know its
derivative in closed form, `@hawk.primitive` ([`hawk.ext.primitive`](03_your_own_derivative))
is the seam — the extension point where your own code plugs into hawk's:
register the forward, supply `vjp=`/`jvp=`, done. Bare `@hawk.primitive` names it after the function; with no
rules it differentiates through its own body, like a plain function; running a notebook cell twice registers the
same function again without an error. The forward is inlined
(no function call, no separate compiled translation unit), and the
supplied rule replaces
the table's for that subgraph. This covers almost everything a kernel
author reaches for: a smoothed activation, a regularised norm, a custom
loss term.

## 2. `hawk.native` (a function written as text)

`hawk.native(name, reads=..., returns=..., text=...)` declares a function whose body is your own text for each
target. `reads` names the parameters and their types, `returns` is the type of the one result, and `text` maps a
target name to the body of the function. It exists for arithmetic hawk's vocabulary cannot express at all: a
hand-tuned numerical trick, a call into a library hawk has no binding for.

```python
fma_sq = hawk.native(
    "fma_sq",
    reads={"x": hawk.Scalar, "y": hawk.Scalar},
    returns=hawk.Scalar,
    text={"host": "return std::fma(x, x, y);"},
)

@hawk.kernel
def combine(a: Scalar, b: Scalar, y: Mutable[Scalar]):
    y = fma_sq(a, b)
```

The text sees its parameters and nothing else: a name of a plane or of anything in the kernel is a compile error
that names the function. A target the function has no text for is refused by the function's and the target's names. It is
written once per unit and called, like [a function](../composition). Two limits, today:

- **It is not differentiable.** Differentiating a kernel that calls one is refused by name; there is no rule for
  opaque text.
- **The device text does not write functions yet.** A kernel that calls one is refused by name when it is built for
  the device.

Reach for `hawk.native` only when a primitive genuinely cannot express what you need, and expect to lose autodiff
on that kernel.

## 3. Contribute a native op (the two-repo path)

If the function is not a composition of existing ops at all — it needs a
new `aether::math` call, with its own host and device implementation — a
primitive cannot help: there is nothing to compose it from. This is a
**contributor** change, not an authoring one: a new name across `aether`
and `hawk`, six files, two repositories. See
[Add a native function end to end](../contributing/add_a_native_function) for the exact
steps.

```text
                 I need a function hawk's kernels don't have
                                    |
            Is it a COMPOSITION of ops hawk already has
                (log, exp, norm, dot, the operators...)?
                 /                                  \
              yes                                    no -- it needs its
               |                                      own aether::math call
     Do I know its derivative                               |
       in closed form?                             contribute a native op
        /            \                              (3. above --
     yes              no                             two repos, six files)
      |                |
  hawk.ext.primitive   hawk.native
  (1. above --         (2. above --
   the common case)     not differentiable)
```

## Next

[Add a native function end to end](../contributing/add_a_native_function) — the
contributor path, six files across `aether` and `hawk`. Back to
[the capstone](04_capstone).
