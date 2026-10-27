# Vocabulary and composition

A new problem should usually need a new composition of the vocabulary, not new machinery in hawk.

You write one sample's step in the names the vocabulary gives you: planes, `Matrix` and `Vector` values, loops, `dispatch`,
reductions, `m.solve`. hawk traces it, differentiates it and writes code for the host, CUDA and C from the same source.
None of that knows what your problem is about.

So a domain is a set of kernels, not a patch to hawk. When a domain needs something the vocabulary lacks, the fix is a
generic primitive that every domain then has, or a refusal that names the limit. There is no domain special case in the
engine, and a kernel without the new feature keeps emitting the same text (`tests/test_render_baseline.py` pins it).

The [domain pages](domains) are the test. Each one is written against the existing vocabulary, and the places where it
stretched it are listed with the primitive that closed them on [Why RAPTOR is built this way](https://amasat01.github.io/why.html).

How to extend the vocabulary yourself: [Add a native function](contributing/add_a_native_function). Why the choice of
spelling inside a kernel is the compiler's, not yours: [Emit form](emit_form).
