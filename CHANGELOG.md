# Changelog

## 0.4.0 (unreleased)

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
Free-threaded wheels ship without declaring GIL-free support, so CPython
re-enables the GIL when `hawk` is imported and prints a RuntimeWarning;
results are correct, just not parallel. No API changes.

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
