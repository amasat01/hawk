# Vocabulary and composition

A new problem should usually need a new composition of the vocabulary, not new machinery in hawk.

You write one sample's step in the names the vocabulary gives you: planes, `Matrix` and `Vector` values, loops, `dispatch`,
reductions, `m.solve`. hawk traces it, differentiates it and writes code for the host, CUDA and C from the same source.
None of that knows what your problem is about.

So a domain is a set of kernels, not a patch to hawk. When a domain needs something the vocabulary lacks, the fix is a
generic primitive that every domain then has, or a refusal that names the limit. There is no domain special case in the
engine, and a kernel without the new feature keeps emitting the same text.

The [domain pages](domains) are the test. Each one is written against the existing vocabulary, and the places where it
stretched it are listed with the primitive that closed them on [Why RAPTOR is built this way](https://amasat01.github.io/why.html).

How to extend the vocabulary yourself: [Add a native function](contributing/add_a_native_function). Why the choice of
spelling inside a kernel is the compiler's, not yours: [Emit form](emit_form).

## Helpers and masked sums

A plain `def` called from a kernel is inlined when it is given traced values, so a repeated block is written once: a right-hand side `def rhs(x, k): return m.vec(x[1], -k * x[0])` and a step `def rk4(f, x, dt, *a): ...` that calls it four times are ordinary helpers, and give the bits of the same lines written out. A helper that returns the new state is used as `x, v = step(x, v, dt)`. A helper may branch on traced values, but only in the kernel-body form (one trailing `return`, no comprehension over a traced value); the error names the helper and the fix.

A helper defined in the kernel's own module and traced as a kernel has no parameter declarations, and the error says so: move the helper to another module, or call it with traced values so it is inlined.

A helper used many times can be written once in the generated text instead of at every use: decorate it with `@hawk.function` and annotate its parameters (and, optionally, its result) with payload types. The function has one definition in the unit's source and one call per use; a plain helper stays inlined. A scalar is passed by value, a vector or a matrix by constant reference.

```python
@hawk.function
def drag(v: hawk.Vector[3], rho: hawk.Scalar) -> hawk.Vector[3]:
    return v * (rho * m.sqrt(m.dot(v, v)))

@hawk.kernel
def push(v: hawk.Vector[3], rho: hawk.Scalar, out: hawk.Mutable[hawk.Vector[3]]):
    out = drag(v, rho) + drag(v, 2.0)
```

A function body may read only its parameters; it may call another function, but not itself. A wrong number or type of arguments is refused by the function's name. A compiler that keeps the call may fuse multiply-adds differently from the inlined text, so under the default host profile the two agree to rounding, and under the `native` profile (no fusing) they agree bit for bit. The device text does not write functions yet and refuses a kernel that calls one.

A function can instead be your own text per target: `hawk.native(name, reads={...}, returns=T, text={target: body})` declares it, and a call in a kernel body records one call, like a `@hawk.function`. The text is the BODY of the function and sees only its parameters, so it cannot name a plane or a lane index of the kernel; a name it uses that is not a parameter fails the compile, and the failure names the function. A target the function has no text for is refused by the function's and the target's names. A native function can be called from a `@hawk.function` and the other way round. It is not differentiable, and the device text does not write it yet.

```python
fma_sq = hawk.native("fma_sq", reads={"x": hawk.Scalar, "y": hawk.Scalar}, returns=hawk.Scalar,
                     text={"host": "return std::fma(x, x, y);"})

@hawk.kernel
def combine(a: hawk.Scalar, b: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = fma_sq(a, b)
```

A sum that skips some terms (a masked sum) writes each term as `m.dispatch(live, [0.0, term], strategy="predicated")`, with `live` an integer 0 or 1 per element: both branches are evaluated and one kept, with no divergent branch. See [Choosing a dispatch strategy](tutorials/08_choosing_a_dispatch_strategy).

## Kernels in sequence: `hawk.chain`

Several kernels that run one after another on the same lanes can be one kernel:

```python
step = hawk.chain(predict, correct, finish, name="step")
```

A member reads a value an earlier member wrote by using the same name; any other input is the chain's. A name no later
member reads is an output. The values passed between members are not planes of yours: they stay in the kernel.

Fusing saves the launches and the plane traffic between members. Fusing everything is not always faster, so the build
decides from a per-device row (`fusion.json`): it cuts where a body would spill registers. A cut is an ordinary launch
with the crossing values in planes the runtime makes; you see no difference in the answer. Members must agree on
kind, launch grid, compaction group and jump width. A member that reads another lane's element of a chained value is
refused, with the name of the member and the value.

Keep branches that do not depend on each other as a graph, not a chain. `hawk.diff.jvp` and `hawk.diff.vjp` of a
chain are chains of the members' derivatives, cut by the same rule.
