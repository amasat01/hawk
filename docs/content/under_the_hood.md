# How a kernel is compiled and run

You write a function and call it. This page follows that call, so you can see
where each piece of the program lives. Nothing here is needed to use hawk.

```{code-block} text
@hawk.kernel         k(x, a, b)  first call              later calls
 trace the source →  eagle plan → generate C++/CUDA →     cache hit →  launch
 IR (what it does)   (host or      compile, store in      (no compile)
                      device)      the cache folder
```

## `@hawk.kernel`: your function becomes a description

At decoration hawk reads the **source text** of the function (so it must live in a file or a
notebook cell), rewrites it a little (an `if` on a value becomes a select, a bounded `for` stays a loop) and runs it once on stand-in values. Every operation
it sees is recorded. The result is an intermediate representation: a graph of operations on
named inputs and outputs, with no Python left in it. This is where the refusals come
from (a class instance, an annotation that does not match, a reserved name): they are
raised when you define the kernel, not when you run it.

`return` is how outputs are declared. Each returned value becomes an **output slot**, an array the
call allocates for you. A slot is named after the variable it was returned from, or `out`
(`out0`, `out1`, ... in a tuple) when the value is an expression. The shape of the
`return` (value, tuple, list, dict, nested) is remembered and given back from the call. A
`Mutable` parameter is the other kind of output: an array you pass in, read and written in
place. The derivative kernels of `hawk.diff` are built from the same description, which is
why `return` names show up as gradient keys.

## `k(x, a, b)`: a call reaches eagle

A call hands the arguments to eagle, which owns launching. eagle looks at where the arrays
live: NumPy arrays mean a host launch on CPU threads, CuPy arrays a device
launch. It builds the kernel for that side on the first call and keeps what it built, so later calls
only launch.

## The build and the cache

For the side chosen, hawk turns the description into C++ (host) or CUDA (device)
source, writes it with a small manifest, and compiles it with the host compiler (or the CUDA compiler for the GPU). The first build of a kernel logs one line saying where it goes:

```text
hawk: building scale (host) in ~/.cache/hawk — delete the directory to rebuild
```

(the line is at INFO level of the `hawk` logger; the folder is `$HAWK_CACHE_DIR`, else
`$XDG_CACHE_HOME/hawk`, else `~/.cache/hawk`). The compiled library is stored under a key made from the
generated source, the backend, and the compiler with its flags, and it is reused only
while the files it was compiled against still have the same content. So a second run of your
program, or a second kernel that generates the same code, does not compile again. Delete the
folder to start fresh; it is only a cache.

[Every form of a kernel, executed](examples/06_every_form_of_a_kernel) shows each of these
steps from the outside. For how eagle runs several kernels as one graph, see
[eagle's graph model](https://amasat01.github.io/eagle/content/userguide/graph_model.html).
