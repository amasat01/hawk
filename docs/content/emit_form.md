# Emit form

`emit_form` picks how a product, a dot, a sum or a norm is written in the
generated code. It is an argument of `build_bundle`.

```python
bundle = hawk.build_bundle(kernel, emit_form="auto")   # the default
```

| Value | What it writes |
|---|---|
| `"auto"` | each site by size: unrolled when small, a loop when large |
| `"static"` | every site unrolled, element by element |
| `"rolled"` | every site as a loop over the element index |

Any other value raises `HawkError`.

## The size rule

Under `"auto"` a site is unrolled when its fold length and its result element
count are both at most 16, and written as a loop above that. The fold is the
length that is summed over: the inner size of a matrix-vector or matrix-matrix
product, or the length of the vector in a dot, sum or norm.

## What does not change

The three forms give the same bits. They differ only in compile time and in
how many values a lane keeps live. A loop keeps one element live where the
unrolled text keeps a whole fold, so a long fold or a wide result does not
spill.

`"static"` is the text hawk wrote before the option existed. A wide product
can fail to compile in that form. `"rolled"` is the fast-compile mode: the
compiler does no per-element work, and a fold or result of any size builds.

## The spill report

For a CUDA target the sidecar's `ptxas` rows carry two more fields beside the
register and local-frame figures: `working_set_B`, the working set the kernel
is expected to need, and `spills`. `spills` is true when registers were spilled
or the local frame is more than 1.25 times the working set. A kernel flagged
this way also warns when it is built; if it does, try `emit_form="rolled"`.
