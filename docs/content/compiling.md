# Compiling

How hawk turns a kernel into something that runs.

- [The compile cache](compile_cache): why building the same kernel twice compiles once.
- [Compile options](compile_options): the flags and profiles a build takes.

The `c` target emits standalone MISRA-C for an embedded build instead of a
native library; see [Embedded C target](embedded.md).

## The compile guard

Two guards keep a compile from running away unseen: a very large source is refused before the
compiler sees it, and a compiler that does not finish is stopped.

- A translation unit over 8 MiB is refused with an error that names its size. Set
  `HAWK_COMPILE_SOURCE_BUDGET` (bytes) to change the limit; `0` turns it off.
- A compiler subprocess is stopped after 1800 seconds. Set `HAWK_COMPILE_TIMEOUT` (seconds) to
  change it; `0` turns it off.

A source this large is usually a wide kernel unrolled. Roll the loop, or read a wide table at a
runtime index with [`hawk.math.index`](tutorials/06_lookups_scatters_and_sums), which keeps the
source small and the gradient cheap.

## Wide values

A wide kernel is placed by the device it is built for; see [wide values](howto_wide_values).

## Banded scalar mode

`mode="banded"` builds a kernel whose real values carry about double precision on hardware with little or no
FP64: each value is stored as an 8-byte cell and computed as three single-precision parts. Pack your float64 planes
with `eagle.banded_pack` before a run and read the results back with `eagle.banded_unpack`. Banded is a device
feature for GPUs with weak or no FP64; on a CPU use float64. A build with a CPU target is refused.

Values live in the FP32 exponent envelope: a stored value's magnitude is in [2^-94, 2^126). A result outside it is
stored as ±Inf (too large) or ±0 (too small), so `isfinite` on the unpacked results tells you whether a kernel left
the envelope. −0 is stored as +0. Each operation keeps the accuracy aether states for it, and a chain of operations
stays within the sum of those bounds. Host and GPU results are bit-identical.

Every sum in a banded build runs through aether's compensated accumulator: `hawk.sum` over the samples of a warp row,
over lane axes (both policies), the ordered stages, a keyed `hawk.scatter` with the default or compensated policy, and
the sum across the lanes of a warp. The accumulate always takes the ordered stages, so it gives the same bits on every
run; a packed cell has no atomic add, so `hawk.scatter(..., policy="plain")` is refused. The accumulator's own figure,
within 0.01 ULP of the exact sum on a series that cancels by 2^30, is a figure for the carrier. The cell a sum delivers
adds one rounding of at most 0.5 ULP to it. `hawk.max` and `hawk.min` over a warp row are built too (the fold is exact,
so they match NumPy's bit for bit). A banded build refuses, by name, a block sum over matrices and the functions
whose banded versions have not landed yet (`erf`, `erfc`, `rint`, `remainder`, `fdim`, `isnan`/`isinf`/`isfinite` inside a
kernel, and the random draws). Arithmetic, roots, `exp`, `log`, `pow`, `expm1`, `log1p`, `exp2`, `log2`, `log10`, the trigonometric and
hyperbolic functions and their inverses, rounding, `fmod`, `fma`, `abs`, `min`/`max`, `copysign` and comparisons all build.

## Local memory

A thread on the GPU can hold only so much local memory (spilled registers, arrays and matrices it indexes at run
time): 512 KB. A kernel past that would fail at launch, so `hawk.build` refuses it by name, with the number and the
way out: hold fewer live matrices per lane, build with `linalg_form="team"` (see
[small linear algebra](linear_algebra)), or use a smaller `n`. The sidecar records the figure as
`launch.local_bytes`, with `local_source` (how it was measured) and `local_limit`. A launch of `n` threads needs about
`n * local_bytes` of device memory beside your planes, and eagle refuses a bind that cannot fit it.

```{toctree}
:maxdepth: 1
:hidden:

compile_cache
compile_options
```
