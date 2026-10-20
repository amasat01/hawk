# Changelog

## 0.6.3 (2026-10-20)

**Cells are the only record of a reduction over samples, and a reduced value can be read later in the body.** A kernel that reduces over samples is bound by its cells (`t` of shape `(1,)`, or `(columns,)` for a wider cell); `hawk.run` owns the row scratch (made once per sample count), launches, and finishes the rows on the host in eagle's ordered tree, bitwise the cell eagle's `ordered` policy gives. The sidecar no longer carries `sample_sums`; `cells` is the record, and a kernel with cells declares `exec_access = "mapreduce"` with its op (`sum` or `max`), so a bundle holds only kernels of one execution axis. `z = x / hawk.all_samples(x).sum()` builds as launches run in sequence, `<name>`, `<name>_2`, ...: each later launch reads the earlier cells as one-cell `lookup` planes, and a per-sample value it needs is computed again there. The first launch's sidecar lists every launch under `stages` (`kernel`, `cells_in`, `cells_out`, `recomputed`) and every cell under `cells`; a kernel that reads no reduction is one launch, unchanged. `hawk.run` runs the launches with the host finish between them; a cell only a later launch reads is the kernel's own, so the caller binds only its outputs. Each launch is its own compile, so a change after the cut recompiles only the later launch. Derivatives are taken on the whole kernel and then split; the derivative of a sum over samples read in the body is the sum of its adjoint, and a max or min refuses. A reduction inside a `for` still refuses.

**Gradients through a parameter-dependent switch, `@hawk.kernel(jump=...)`.** With `jump=<width>`, `vjp` and `jvp` differentiate an `if`, a conditional expression or `where` comparing a differentiated value (`x < t`, `<=`, `>`, `>=`, combined with `land`/`lor`/`not`, nested, entrywise on a vector, or inside a `for` body) as the smoothed loss: a sample within `sqrt(5) * width` of the switch passes on the jump between the branches, weighted by an Epanechnikov kernel of standard deviation `width`; farther samples cost one comparison. `jump="h"` reads the width from a `Param[Scalar]` (no rebuild per width, no gradient for it) and the derivative also writes spread cells `jump_sd<i>_0/1/2` for each switch outside a loop (sum of `l - r`, of its square, and the count). The primal kernel and its digest are unchanged, and a kernel without `jump=` derives exactly as before. The derived sidecar lists the switches under `derivative.jump` (each with the comparison it came from as `group`, and its loop). A `dispatch` whose selector is a select between two branch indices is no longer clamped. Refused by name: `==`/`!=` on a differentiated value, a `break` depending on one, a rounding or integer cast before the comparison, a second derivative of a `jump=` derivative, and a width that is not a positive number, `"auto"` or a `Param[Scalar]`. `jump="auto"` picks each comparison's width from the batch, `0.02·√12` times the standard deviation of `l - r` (a vector comparison's entries pooled), in a first stage of the derived kernel that sums `l - r`, its square and the count into the cells `jump_sd<g>_0/1/2`; the second stage writes the width it used to `jump_h<g>`. One derived kernel serves every batch. A switch inside a `for` takes a number or a `Param` width.

**Removed: `Accum`, `Reduce`, `.add`, `.contribute`, `.write(at=)`.** Declare `Mutable[...]` (or `WideOut[...]`) and use `hawk.all_samples`: `hawk.all_samples(bins)[key] += term` accumulates by key, `hawk.all_samples(buf)[key] = v` stores by key, and `total = hawk.all_samples(term).sum()` (or `.max()`, `.min()`, `.mean()`) reduces over samples. `x.add(v)` with no `at` is `x += v`; `buf.write(v)` is `buf = v`. `hawk.Accum` and `hawk.Reduce` raise an error that names the replacement, as do `.add`, `.contribute` and `.write` on an output. Forward mode (`jvp`) through a max or min over samples is refused, as reverse already was.

**Reductions and `hawk.all_samples`.** A value's own axes reduce with `.sum/.max/.min/.mean(axis)` (`axis` is a number, `"components"`, `"rows"` or `"columns"`), `hawk.math.sum/mean`, and the views `.components`, `.rows`, `.columns`; `axis="samples"` on one sample's value refuses and names `hawk.all_samples`. `max(a, b)` and `min(a, b)` on traced values are `hawk.math.max/min` (a tie keeps the first operand). `total = hawk.all_samples(x).sum()` (also `.max()`, `.min()`, `.mean()`) stored to a `Mutable` writes one row per 32 samples (sum and mean: with `_comp`; max and min: exact), listed in the sidecar under `cells`; a kernel reduces with one op. The derivative of a sum over samples broadcasts its adjoint; a max or min refuses.

**A parameter's gradient is summed in the derived kernel.** `vjp` of a `param_planes` kernel writes each parameter's gradient `bar_<name>` as one sum per 32 samples (a warp), plus its compensation term in `bar_<name>_comp`: `ceil(n/32)` values each instead of `n`, in the same kernel as the per-sample `bar_x`. A row is a fixed compensated tree over its 32 samples, bitwise across launches, and the host gives the device's rows within rounding. The sidecar lists these planes under `cells`. Add `bar_<name>` and `bar_<name>_comp` over the rows to get the gradient. A launch covers whole rows (its first sample is a multiple of 32), and the segmented, active-set and persistent entries refuse such a kernel.

## 0.5.1 (2026-10-17)

**One `HostKernel` shared across threads is safe.** In 0.5.0 `hawk.runtime.run` bound and launched in two steps, so a second thread calling `run` on the same kernel could rebind its planes between the two and the first thread's launch then wrote to, or read from, the wrong arrays (wrong results, and heap corruption with unequal sizes). It hit anyone calling one loaded kernel from several threads, such as an eight-thread framework call. `run`, `bind_all` and `launch` now hold a per-kernel lock for the whole call.

**Nested-derivative host kernels no longer read a stale value under GCC.** A host kernel with a termination guard, differentiated twice (`jvp` of `jvp`, as `jacfwd` of `jacfwd` emits), could read a stack slot it had not written when built with GCC 11.5 or 14.3, so its result was wrong or changed between calls. The fix is in aether-dsc 0.2.4, now the minimum; 0.5.0 users get it by upgrading aether-dsc. Vectorised loops and speed are unchanged.

**Fused, persistent and range launches now agree within floating-point rounding, not bit for bit.** The compiler may contract a multiply-add differently in each entry; step counts and flags stay exact.

**Derivatives of derivatives, to any order.** `vjp` and `jvp` of a derivative kernel (any mix, up to three levels) name each level's planes apart (`bar_x`, `bar2_x`, `bar2_bar_x`, `dot3_x`) instead of clashing on the primal's names; each plane is found under its own prefix, in float64 and float32, on the host and the device.

**`param_planes` reaches uniforms read inside loops.** A `Param` read in a loop body is promoted to a one-cell plane like any other, the loop is rebuilt over it, and its `vjp`, `jvp` and second-order routes match the `Param` kernel.

**Two examples, with notebooks and a benchmark card.** The autodiff-against-torch and hybrid orbit examples (moved here from raptor-core) show hawk gradients inside a torch program and a model with shared weights as one `Table`.

**NumPy 1.26 or newer.** The package declares `numpy>=1.26`, the oldest NumPy the host route and the examples are tested with.

**docs-fresh tells its failures apart.** `tools/docs_fresh.sh` exits with a different code for stale docs, a missing card and a missing tool. The CI test steps point at the eagle checkout, so the docs-fresh card tests run there instead of skipping.

**Docs.** The Colab and Kaggle setup cell installs the CUDA 12 or 13 extras, can test a release candidate and upgrades the family's own packages, and the notebooks are checked in a Colab-like environment. The capability map names JAX among the certified zero-copy frameworks. Pages actions moved to the Node 24 releases.

## 0.5.0 (2026-10-14)

GCC 13.1-13.3 and 14.1-14.2 miscompile a conditional call of a function that has masked
vector variants (fixed in GCC 13.4 and 14.3). aether-dsc 0.2.3, now the minimum, avoids
it automatically by declaring no masked vector-math variants on those compilers, and
hawk reports the affected loops (an interchanged trip loop with a math call, under
`native-vector-math`) as not `vectorised`. Only loops with a conditional math call run
scalar there, and on GCC 12, which never if-converts such a call; everything else stays vectorised, and no compile flag is added.

Every `for` loop of a `sample_local` kernel runs interchanged on the host: the trips
outside, a tile of samples inside, the carried values in lane arrays. The tile spans
eight vector registers of the kernel's float type (64 float64 or 128 float32 samples
under AVX-512), so a dependent loop body keeps eight chains in flight and the lane loop
vectorises whatever the trip count. A sample's own `break` or runtime trip count stops
it in its lane; a nested `for` stays each lane's own loop. Under `native` and
`portable` the results are bit-identical to the per-sample loop, which
`-DHAWK_HOST_INTERCHANGE=0` still builds. `vectorised` (on a host `Source`, in
`kernel_cost()` and per `dispatch_costs()` branch) reports these loops as vectorised;
a loop pair that reads a tape or another loop's count, and a loop inside one `switch`
case, keep the per-sample loop and the unroll limit.
Host-compiler discovery honours the standard `$CXX` (precedence: `$HAWK_CXX`, `$CXX`, `g++` on `PATH`, zig); an unresolvable `$CXX` falls through, and the not-found error names every candidate tried.

`build()` and `build_bundle()` take `literal_table=` (default `False`). On, the device
source reads its `double` literals from one `__constant__` table per translation unit,
the same bits as operand immediates; off, the device code is byte-identical to before.
It pays on architectures that issue a move pair per literal (sm_90); a `float32` build
ignores it.

`hawk.ir.dispatch_costs()` takes the host profile, and reports `fused_vectorised`
only for a sample-local kernel whose branches either call no library math or are built
under `native-vector-math`: under an exact profile `exp`, `sin` and the like stay scalar,
so the fused host loop no longer counts as running every branch in lockstep.

An automatic kernel's `_persist` entry runs warp-converged. At each chunk head the
lanes of a warp without a sample claim the next ones together: one ballot, one
`atomicAdd` on the sample counter by the warp's leader, the base shuffled out. A lane
whose sample finishes mid-chunk waits, masked, for the next head; the step body holds
no atomic and no warp collective. The chunk length is a new trailing launch argument,
`hawk_chunk` (`uint32`, after `hawk_stepsum`; a power of two, 0 runs each sample
whole), and the entry is launched in whole warps. The slot pair now counts each
warp's slots at each head (32 lanes times the chunk), so steps over slots is the
run's slot fill. Results are bit-identical to the fused entry for any chunk, and the
entry holds no more registers than before. The sidecar of a stepping kernel gains
`fp64_ops_ir`, one step's 64-bit floating-point operation count from the IR.

The `_persist` entry writes a sample back at the next chunk head instead of where it
ends: a lane whose sample exits or spends its budget only marks it in its trip count,
and the warp's ending lanes commit together at the head, ahead of their claim, instead
of one lane at a time in a branch of the step. The commit reads the carried values and
rebuilds the loop's finish flag from the trip count, so the entry holds no more
registers; a kernel committing a value the step neither carries nor sets as that flag
keeps its commit in the chunk. Results stay bit-identical to the fused entry.

## 0.4.0 (2026-10-09)

**Free-threaded CPython, GIL-free.** The compiled core now declares free-threading
support, so on 3.13t and 3.14t hawk runs without the GIL and the GIL stays off after
import. The compile drivers, the compile cache and hawk's Python-side state are
thread-safe: any number of threads may build, compile and cache concurrently, distinct
objects need no synchronisation, and one stateful object shared between threads is
memory-safe with its calls serialised (the order of those calls is yours to decide).
CI gained free-threading stress legs, and the package tests against the shared
conformance harness in `raptor-core` 0.3 (the `test` extra now asks for
`raptor-core>=0.3`).

The sealed-path compile cache is now keyed on the user headers a kernel includes, so
editing a header you include recompiles the kernel; that path also gives a hint when
the address-space cap is what stops a compile.

New tutorial 07, "gradients through branches and stops", with tests that pin hawk's
tie conventions at a branch and at a stop.

A per-sample plane whose shape reads both as component-major `(w, N)` and as
sample-major `(N, w)` (a `(w, w)` array, or a square matrix head with
`N == R == C`) is now refused, naming the argument, its shape, both readings
and the fix; before, it was silently taken as component-major, which transposed
a C-contiguous sample-major array. Say which axis holds the samples, zero-copy:
per call with `hawk.run(..., layout="samples_first")` or
`layout="samples_last"` (also `HostKernel.bind_all(..., layout=...)`), or per
array with the new `hawk.samples_first(x)` / `hawk.samples_last(x)` markers,
which work for any shape, are refused when they contradict the shape, and win
over the call's `layout=`. The markers follow a small protocol shared with
eagle (`__raptor_samples_axis__` plus `.array`), so each package accepts the
other's. Shapes that are not ambiguous behave as before.

## 0.3.1

Wheels now cover CPython 3.9 through 3.14, plus the free-threaded 3.13t and
3.14t builds (manylinux_2_28, x86_64); `requires-python` is now `>=3.9`.
Free-threaded wheels first appeared here; 0.4.0 declares GIL-free support for
them. No API changes.

## 0.3.0 (first public release)

hawk is the RAPTOR family's kernel-authoring layer: a Python DSL for
writing a kernel once, deriving its forward- and reverse-mode derivatives,
and compiling it at run time for host or device through the eagle runtime
and aether.

Compiles kernels through NVRTC at run time, picking the CUBIN or PTX image
the installed driver can actually load rather than hard-coding one target,
and records the real device-image format and extension produced in
manifests and sidecars. Runtime arguments are validated — dtype, shape,
contiguity and writability — before any native call, with every problem
listed in one error.

Settles the public surface: every public name now has one canonical
import path (math functions under `hawk.math`, `Terminated` and
`HawkError` at top level), pinned by a public-API test so a future
internal reshuffle cannot change it silently.

Derived kernels now honour termination: a derived kernel of a kernel that
declares `terminated` takes a `terminated=` argument too (pass the same mask
you pass to the original kernel), and a
terminated sample now skips its kernel's work entirely rather than only
being excluded from the result. `Guard(active_set=True)` lets a kernel run
over eagle's active-set index map instead of the full sample range, so
live samples stay packed in warps as a population shrinks.

A sample can leave a loop early: `if cond: break` as a top-level statement
of a `for` over a compile-time `range` lowers to a real per-sample exit on
host and device. Forward mode differentiates every `break` position; reverse
mode supports a `break` that is the first or last statement of the body.
`while` stays refused (write a capped `for` with `if lnot(cond): break`).

Licensed under Apache-2.0, with a documentation site (install
including a pip-only path, quickstart, tutorials, examples, and the API
reference).

This is hawk's first public release.

The NVRTC device compile finds CCCL under `include/cccl` and `targets/*/include/cccl` as well as `include`, prefers the CUDA root of the NVRTC library actually loaded, and accepts both CCCL wheel layouts.

With no GPU reachable a device artifact is PTX at the toolkit's minimum compute arch (a virtual `compute_<N>` arch), a device newer than the toolkit gets PTX at its maximum, and a device older than the toolkit's minimum is refused with its name, arch and toolkit; `device_arch=` and `$HAWK_CUDA_ARCH` are checked the same way.

The device probe reads the current CUDA context's device, falling back to the first ordinal only when no context is current.

The nvcc host compiler is no longer forced to `/usr/bin/g++`: `$HAWK_NVCC_CCBIN`, else the caller's own `NVCC_PREPEND_FLAGS`/`NVCC_CCBIN`, else nvcc's default; its identity is part of the device cache key.

Cache writes go to a private temporary in the slot and are moved into place, so a concurrent reader never sees a partial artifact or record; the host cache key includes the C library version.

The package requires a Python supported by its classifiers (`requires-python` raised to match).

The `_persist` entry no longer runs an atomic per sample on the `hawk_steps` cell or the finish counter: each lane keeps its longest trip count and its finish count in registers and reports both once at exit, warp-reduced, into the same cells (same values, same bits).

Both fast entries (`_range`, `_persist`) of a self-finishing kernel count a
sample that is already finished on entry into the finish counter, as the
step's finish epilogue counts one that finishes during the run: the caller
zeroes the counter and reads the batch's finished count after the launch,
with no separate count of the mask. The sidecar (and plugin) declares this
with `entry_counts_finished`.

An automatic kernel's `_persist` entry runs its lanes in chunks of 32 steps
(`PERSIST_CHUNK`), and a lane whose sample finishes takes the next sample at
once, inside the chunk. The loop has no warp-wide synchronisation any more
(no `__syncwarp`), so it is correct however the GPU schedules the threads of a
warp. Its loop counters, and those of the `_range` entry and of a fused body's
runtime-count loop, are 32-bit (`hawk_clamp_count` caps a count at the int
range). A step of at most `SHORT_DEVICE_STEP_OPS` (128) operations unrolls the
`_persist` chunk loop and the `_range` step loop by 4. Results are
bit-identical; the GPU perf card records the speed.

Both fast entries take one more trailing argument, `hawk_stepsum`: a
nullable `uint64` cell into which the launch adds every step it runs (one
atomic per warp). A caller passing argument lists by hand adds it after
`hawk_steps`. A stepping kernel's sidecar (and its plugin) carries `step_ops`,
one step's operation count, for the launching runtime; the emitted code does
not read it.

The native module links the C++ runtime statically only when the toolchain
has a static `libstdc++`.

With no cuda-bindings installed, the device-arch resolver reads the running
GPU's arch from the CUDA driver library (through `ctypes`, no extra package)
before falling back to `sm_61`, so a newer GPU no longer gets kernels it cannot
load.

The default x86-64 host profile, `native-vector-math`, now compiles with FMA
contraction (`-ffp-contract=fast`): `a*b + c` becomes one fused
multiply-add, which rounds once instead of twice. Results are as accurate or
more, but no longer bit-identical to the exact profiles; `native` and
`portable` keep contraction off and stay bit-identical to the scalar build
(`hawk.compile.toolchain.EXACT_HOST_PROFILES`). Request one of them wherever
results must match bit for bit. Code that relies on exact rounding is
protected whatever the profile: the `compensated` sink's two-sum pins its
operands with `AETHER_FP_BARRIER`, so the compensation stays exact (pinned by
`test_compensation_stays_exact_under_fma_contraction`), and aether's vector
math stays contraction-free. Bit-identity tests now pin the `native` profile;
`test_the_fast_profile_is_within_its_derived_bound` gates the fast profile
with an error bound derived from the step's operation count.

A fused-step kernel's host build (`steps=K`, `steps="auto"` and the plain
`@hawk.kernel` default) runs its fused side as TILES: per tile of samples the
carried values sit in cache-resident lane arrays and each trip advances every
live sample of the tile in one vectorised loop, a finished sample keeping its
values, until the tile has no live sample or the launch's steps are done.
Under an exact host profile results are bit-identical to the per-sample
fused loop it replaces (same operations per sample, same order), the device source is
unchanged, and the tail break and the cap rule are untouched. The tile size
comes from `hawk.emit.host.host_tile_rule` (half of L1d, or of L2 for large
carry sets, in whole 64-sample blocks, clamped to 64-1024), recorded as a
comment in the emitted source; the entry's shared object exports the tile as
`<name>_host_tile` so a runner can tell the shapes apart. Building with
`-DHAWK_HOST_INTERCHANGE=0` (`build_bundle(..., defines=...)`) builds the SAME
source's per-sample loop. A kernel that is not `sample_local`, one whose step
holds a second lowered `for`, and an active-set kind keep the per-sample loop.

A ``KernelKind`` subclass now INHERITS its base's vocabulary entries and
seam fields (``output``/``sink``/``guard``/``quantities``/``compile_hook``),
most-base first — ``class Perturbed(Orbit, slug="perturbed"): ...`` carries
``Orbit``'s vocabulary forward and the extended kind works end to end
(trace, compile, run on host and device, derivatives), rather than the
vocabulary silently dropping. A class's own body adds entries and may
override a seam field; one it leaves unset keeps the nearest base's value,
never Kind's own bare default. Re-declaring an inherited vocabulary entry is
fine with an EQUAL resolved declaration; a differing one refuses, naming the
entry, the base, and both declarations — a diamond where two bases disagree
on the same entry refuses the same way, and the existing guard-mask check
(a guard's mask must be a `terminated`-form entry of the vocabulary) now
sees the MERGED vocabulary. `Kind.extend(slug, vocabulary=..., **seams)` is
the instance-form twin: same merge, same refusal rules, the same `Kind` the
class form would build for the same inputs.

`hawk.math` gains numpy's element-wise set, on host and device: `exp2`
`expm1` `log2` `log10` `log1p` `cbrt` `hypot` `power`/`pow`, `sinh` `cosh`
`asinh` `acosh` `atanh`, `ceil` `trunc` `round` `rint`, `absolute`/`abs`
`sign` `copysign` `fmod` `remainder` `fdim` `fma` `clip`, `isnan` `isinf`
`isfinite` (bool), and `erf` `erfc`. Each is an IR op with an aether spelling
and a derivative rule (zero for the rounding family, `sign` and the class
tests). `round` is C's (halves away from zero; `rint` is numpy's `round`),
`remainder` numpy's floor-mod. Every element-wise function, these and the
existing ones, takes vector and matrix operands as `+` does (equal shapes or
one rank-0 operand); a class test or a comparison on a matrix is a bool
matrix, which `where`/`select` take as a per-entry mask.

`hawk.runtime` now checks every bound argument against the kernel's OWN
declaration — dtype, shape, C-contiguity, writability — before taking its
address, for all three roles that used to trust it blindly: a
`Table[...].at()` lookup's integer index, a `Reduce(...)` output's element
count, and a `Terminated` mask's dtype. A wrong binding there used to run
anyway, either returning garbage (an `int32` index decoded 8 bytes at a
time out of a 4-byte-stride buffer) or segfaulting (a `Reduce("sum")`
output sized 1 instead of one slot per sample is an out-of-bounds write
for every sample past the first); `HostKernel.bind_all` (hence `run`) now
refuses instead, naming the kernel, the argument, what was expected, what
was given, and the fix.

`hawk.runtime` binds ONE sample: a plane whose shape is its per-sample head
(a 0-d array for a scalar plane, `(w,)` for a vector, `(R, C)` for a matrix)
binds at its own address as a batch of one and is written in place
(`plane_layout` answers `"single"`). A shape that is also a batch stays a
batch (`(1,)`, `(w, 1)`, `(1, w)`, `(w, w)`). A call mixing one sample with a
batch, and a Python number bound to a plane, are refused naming the argument
and the fix. `plan_view` also projects a matrix output's `(rows, cols)` as
`mat_shapes`. The host-vs-device tutorial debugs one trajectory on the CPU and
runs the same code on a million samples on the GPU through `eagle.plan.auto`.

`hawk.artifact.plugins(bundle, *, device_loader=None)` assembles every kernel
of a bundle into the plugin `eagle.plan` takes, `{name: plugin}`: its
`plan_view` declaration, the host entry's address (after `hawk._core.HostLibrary`
checks the object's ABI tag and layout), and, given a device loader
(`eagle.registry.load_manifest`), the device function, with everything they
point into held alive on the plugin. hawk still imports no eagle.
`eagle.deploy(kernel)` (the same function as `eagle.plan.auto`) uses it, so the
host-vs-device tutorial builds its plans with one call. The quickstart's GPU
cell is `eagle.deploy(scale).run(...)`, and the tutorial runs its one trajectory
and its million through `eagle.simulate`.

`@hawk.kernel(steps=K)` is the decorator form of `hawk.steps`: it traces the
kernel and returns `hawk.steps(kernel, K)` (named `<name>_x<K>`), so a step
kernel that finishes its own sample takes `K` steps per launch with its state
held in registers, bit-identical to one step per launch on the host (on a GPU,
fused multiply-add contraction may differ by an ULP). `steps=1` (the
default) is the plain kernel. A `steps` that is not an int >= 1, and a kernel
`hawk.steps` refuses, are refused at decoration naming the kernel.
`hawk.kernel` called with keywords only (`hawk.kernel(kind=..., steps=...)`)
now returns the decorator. Kernels decorated without `steps=` emit exactly the
same source as before.

`hawk.steps(kernel, "auto")` and `@hawk.kernel(steps="auto")` build ONE
kernel, `<name>_xauto`, whose number of steps per launch is chosen at run
time: the fused loop has the static bound `hawk.ir.steps.AUTO_K_MAX` (64) and
reads its trip count once per launch from the reserved `lookup` word
`fused_steps` (one `int32`, written by the runner on the device and never
bound by the host, like `finished_count`), clamped to `[0, 64]`. A launch
with `fused_steps[0] = k` takes exactly `k` steps for `0 <= k <= 64` (a larger
word runs 64, a negative one none), and a sample that finishes earlier
stops at its own finishing step. The sidecar records `finish.steps = "auto"`
and `finish.steps_max = 64`; the kind is unchanged (build under
`Guard(active_set=True)` for compaction) and the kernel keeps its single
step's access class. Host results are bit-identical to one step per launch
for every sequence of `k`; on the device nvcc's default FMA contraction is
kept, so device results may differ from one step per launch by an ULP where a
multiply-add is contracted differently. A `steps` that is neither an int
>= 1 nor exactly `"auto"`, a parameter named `fused_steps`, and a derivative
of the auto kernel are refused naming the kernel. The emitted entry reads the
word once and branches on it for the whole launch: a word of 1 runs the single
step's own straight-line body (the plain kernel's text, vectorised on the
host), any other word the fused loop, so a one-step launch costs what the
plain kernel costs and is bit-identical to it on both targets.

`steps="auto"` is the default. `@hawk.kernel` without `steps=` builds the
automatic kernel for every kernel `hawk.steps(kernel, "auto")` admits (it
finishes its own sample, commits only `Mutable` planes and has no parameter
named `fused_steps`), under the author's own name (no `_xauto` suffix), with
the plain single step as `.step`. `eagle.until_done` paces it; any other launch
takes exactly one step (eagle binds the word to 1 when the caller does not).
`vjp`/`jvp` and `hawk.steps` of such a kernel derive from its single step
(the explicit `steps="auto"` kernel, named `_xauto`, keeps refusing a
derivative). `steps=1` opts out: the plain kernel, always. Every kernel auto
does not admit emits exactly the same source and sidecar as before.
`hawk.ir.steps.admits_auto` and `default_steps` state the rule.

Host builds now vectorise. The host target compiles at `-O3` under a
`native` profile (`-march=native`, for run-time builds and the cache; the
cache key names the CPU) or a `portable` one
(`-march=x86-64-v2`, for shipped artifacts), selected by
`build_bundle(host_profile=)`, `CompileOptions(host_profile=)` or
`$HAWK_HOST_PROFILE`. A `sample_local` kernel's per-sample loop is now
emitted in a form the compiler vectorises across samples; FMA contraction
stays off, so under these two profiles results are bit-identical to before. Under AVX-512 the loop
widens its `terminated` guard to one float64 lane per sample, so it runs 8
samples per register without spilling. An active-set loop stops at the live
count as its bound, and a tile whose map is the identity (eagle's reorder)
runs the contiguous vector loop.

On x86-64 the default host profile is now `native-vector-math`: `native`
plus aether's vectorised double-precision math (`-DAETHER_HOST_VECTOR_MATH
-fno-trapping-math`), so a per-sample loop that calls `exp`, `log`, `pow`,
`sin`, `cos` or another transcendental function vectorises instead of
staying scalar. Those functions are then aether's packet math, within 1 to 3
ULP of glibc (per-function bounds in aether's
`backend/cpu/simd/math/PacketMath.h`), so host results that involve them are
no longer bit-identical to libm or to the device; arithmetic and `sqrt` are
unchanged. For libm bit-identity, request `host_profile="native"` (or
`HAWK_HOST_PROFILE=native`). On other architectures the default stays
`native`. The profile is part of the compile-cache key, so entries cached
under the previous default are not served under the new one. The
`native-vector-math` key now also names the CPU, as the `native` key does.

Breaking: a Mutable is a plain variable in a kernel body; `x.prior` and
`x[...] = e` are replaced by reading and assigning `x` (the old spellings
raise with the rewrite).
