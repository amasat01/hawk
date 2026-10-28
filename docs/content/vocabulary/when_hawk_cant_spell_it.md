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

## 2. `hawk.raw_device` (concept only — not a runnable cell here)

`hawk.raw_device(text, access=..., reads=..., writes=..., returns=...)`
splices literal C++ text into the generated kernel body, with the access
class (`sample_local`, `cross_sample_write`, …) declared explicitly — never
inferred, because spliced text is opaque to the walk (the pass that
inspects a kernel's IR to infer things like this automatically). It
exists for
arithmetic hawk's vocabulary cannot express at all: a hand-tuned numerical
trick, a call into a library hawk has no binding for.

Two honest limits, today:

- **It is not differentiable.** `hawk.diff.vjp`/`jvp` refuse a kernel that
  calls a `raw_device` block, by name — there is no rule for opaque text,
  and there will not be one (the point of the seam is that the text is
  outside hawk's own IR).
- **Little of it is tested end to end.** Its own test suite
  (`tests/test_trace_raw_device.py`) stops at the trace and IR
  level: it checks that an unannotated block is refused and that a
  declared access class folds into the walk's inferred class. The only
  place a compiled kernel that calls one runs is the multi-process (MPI)
  rank test bed. This page shows no runnable example because none is
  tested in the regular suite.

Reach for `raw_device` only when a primitive genuinely cannot express what
you need, and expect to lose autodiff on that kernel.

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
  hawk.ext.primitive   hawk.raw_device
  (1. above --         (2. above -- concept only,
   the common case)     not differentiable, no
                         runnable cell here)
```

## Next

[Add a native function end to end](../contributing/add_a_native_function) — the
contributor path, six files across `aether` and `hawk`. Back to
[the capstone](04_capstone).
