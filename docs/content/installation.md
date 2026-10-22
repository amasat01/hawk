# Installation

**Time:** ~2 min &middot; **You need:** a terminal and `pip`

```bash
pip install raptor-hawk raptor-eagle matplotlib             # CPU, no extras (eagle and matplotlib: the tutorials import them)
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"    # GPU route; [cuda13] on both for CUDA 13
```

Linux x86_64 or aarch64, or macOS arm64 (with Homebrew's LLVM clang: `brew install llvm` and `export HAWK_CXX=$(brew --prefix llvm)/bin/clang++`; Apple's clang 15 does not accept C++23 yet), CPython 3.9-3.15 (including free-threaded 3.13t, 3.14t and 3.15t), and on Linux a host `g++` 11 or newer (check with `g++ --version`). `raptor-hawk` pulls
`aether-dsc` (the sealed C++ headers hawk compiles against) in automatically. No
`nvcc` or CUDA toolkit is needed.

```{admonition} NVIDIA packages come only through the extras
:class: important
`raptor-hawk[cuda12]` pulls `cuda-bindings` 12, `nvidia-cuda-nvrtc-cu12` and
`nvidia-cuda-cccl-cu12`; `raptor-hawk[cuda13]` pulls `cuda-bindings` 13,
`nvidia-cuda-nvrtc` 13 and `nvidia-cuda-cccl` 13 (CUDA 13's wheels have no `-cu13`
suffix); `raptor-eagle[cuda12]` / `[cuda13]` pull CuPy (`cupy-cuda12x` /
`cupy-cuda13x`, with the CUDA headers CuPy compiles against). Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick the extra
matching the CUDA version your driver reports (`nvidia-smi`, top right).
```

**Platforms:** the wheels are Linux x86_64 and aarch64 and macOS arm64 (CPython 3.9–3.15, including free-threaded 3.13t, 3.14t and 3.15t; the exact versions are in `pyproject.toml`). The GPUs and CPUs hawk + eagle are measured on are listed on [eagle's performance page](https://amasat01.github.io/eagle/content/performance.html). There are no wheels for Windows, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere. On free-threaded 3.13t and 3.14t, `hawk` runs GIL-free.

To build from source, and for the supported-version table, see [build from source](https://amasat01.github.io/build_from_source.html).

To *run* a kernel on a GPU you also install eagle, which launches what hawk
compiles: see [eagle's installation page](https://amasat01.github.io/eagle/content/userguide/installation.html)
(its Python package is `raptor-eagle`). What it buys you depends on what
the *machine* has, and on which extra you ask pip for:

- **CPU-only, no GPU anywhere** — see [](project:#cpu-only) right below.
  Authoring, compiling and running a kernel on the host needs nothing
  more than this and a host `g++`.
- **With a GPU** — see [](project:#with-a-gpu) below. Compiling
  and running a *device* artifact additionally needs a GPU driver and the
  `[cuda12]` / `[cuda13]` extras, and `eagle` to launch it.

Verify the install itself either way:

```python
import hawk
print(hawk.__version__)
```

(cpu-only)=
## CPU-only: no GPU at all

Nothing about authoring, compiling for the **host**, or running through
`hawk.runtime` needs a GPU, a driver, NVRTC, `nvcc`, or `eagle` — only
`pip install raptor-hawk` (see above) and a host `g++`.
`hawk.artifact.build(kernel, dir, targets=("host",))` compiles the host
target alone (the default is both targets; naming `"host"` is what skips
the GPU side entirely), and `hawk.load`/`hawk.run` execute it directly
— no `eagle` import anywhere on that path, exactly as
[Your first kernel](tutorials/01_your_first_kernel)'s host cell shows.

```{dropdown} How the CPU route was verified
Verified: a clean venv built from this project's own wheels
(`raptor-hawk`, `aether-dsc`, `numpy`; `eagle` and `cupy` not installed),
run with `$PATH` stripped to the system's own `/usr/bin:/bin` (no `nvcc`,
no CUDA toolkit), `$LD_LIBRARY_PATH` unset and no GPU selected
(`$CUDA_VISIBLE_DEVICES` unset) — the first tutorial's `scale` kernel built
and ran, computing the right answer, with `eagle` and `cupy` both absent
and unneeded. The GPU route below is for compiling and *running* a device
artifact; it changes nothing about this one.
```

(with-a-gpu)=
## With a GPU (no `nvcc` needed)

```bash
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"    # or [cuda13] on both
```

Compiling a device artifact needs a GPU driver, a host `g++`, and NVRTC
from pip — the `[cuda12]` / `[cuda13]` extra above brings `cuda-bindings`,
`nvidia-cuda-nvrtc` and `nvidia-cuda-cccl` for you (or use an equivalent CUDA wheel set already on the
machine) — no `nvcc`, no `aether`/`eagle` source checkout, and no headers
under any prefix: `aether-dsc` ships a sealed copy of the headers hawk's
device compiler reads from directly. `hawk.compile.device` /
`hawk.compile.cubin` take this path automatically whenever
`hawk.artifact.build`/`build_bundle` compile a `cuda` target and no
`nvcc` is found on `$PATH`/`$HAWK_NVCC`.

That compiles a device artifact. To *run* one, you also need `eagle`
installed (see [eagle's installation page](https://amasat01.github.io/eagle/content/userguide/installation.html)) — hawk compiles it, eagle
launches it.

## Building from source (development)

To work on hawk itself, build it from a clone: see
[Build and run the gate](contributing) on the contributing page.

## Requirements

- Python 3.9 or newer (CPython 3.9–3.14, including free-threaded 3.13t and 3.14t).
- CUDA 12.6 or newer for the device target (the toolkit each performance card was measured with is recorded in the card).
- A CUDA-capable GPU is **optional** for authoring and host execution; it is
  required only to compile and run a kernel's device target. `hawk.compile`
  reports what is available on the current machine:

  ```python
  from hawk.compile import cubin_available
  cubin_available()   # NVRTC importable and a CUDA device present
  ```

## Going deeper

Everything above is enough to install and run hawk. This section is for
anyone calling the device compiler directly, or tuning host performance
and chasing bit-for-bit reproducibility — it changes nothing about
whether a kernel runs.

<details>
<summary>The pip-only device compile, directly</summary>

The same entry points `hawk.artifact.build`/`build_bundle` use are also
usable directly, for an ad hoc CUDA C++ `source` string that was never
authored as a traced hawk kernel at all — see
[Your first kernel](tutorials/01_your_first_kernel) for a kernel-authored
artifact, and the snippet below for the primitive NVRTC compile underneath
it:

```python
from hawk.compile import cubin, cubin_available

if cubin_available():
    image = cubin(source)   # DeviceImage(image=<SASS bytes>, target="cubin", arch="sm_XX")
```

`cubin_available()` never raises; `device(source, target="ptx")` is the
alternative when PTX (the portable intermediate form NVRTC can also
emit) is what a caller needs (guarded against a driver too old to JIT a
newer NVRTC's PTX — `target="cubin"` needs no such guard, it is already
SASS: the GPU's own machine code). Headers for any `#include` in
`source` are served from `aether_dsc`'s sealed payload, never from disk.
</details>

<details>
<summary>SIMD and the host profiles</summary>

The host target is compiled at run time by the host `g++` (or `$HAWK_CXX`;
hawk also honours the standard `$CXX`, with the precedence `$HAWK_CXX`, then `$CXX`, then `g++` on `PATH`, then zig)
under one of three profiles. On x86-64 with a GCC host compiler the default is `native-vector-math`
(below), which vectorises transcendental math with aether's own functions,
faithfully rounded (error < 1 ULP), and lets the compiler fuse `a*b + c`
into one FMA instruction; on other architectures, or with a non-GCC host compiler such as clang, the
default is `native` (asking for `native-vector-math` under such a compiler is
an error that says so). The other two profiles are the EXACT ones: they
keep results bit-identical to libm and to a scalar build:

- **`native`**: `-O3 -march=native -ffp-contract=off`, plus
  `-mprefer-vector-width=512` on x86-64. For anything compiled on the machine
  that runs it when results must be bit-identical to libm (and the default on
  architectures other than x86-64). The cache key includes the CPU's
  identity, so a cache directory shared between machines never serves one
  CPU's binary to another.
- **`portable`**: `-O3 -march=x86-64-v2 -ffp-contract=off`. For a host
  artifact that is prebuilt and shipped (a wheel, a bundle inside a package).
  x86-64-v2 (SSE4.2, POPCNT) runs on every x86-64 CPU still in service and is
  the baseline RHEL 9 itself targets.

Choose with `build_bundle(..., host_profile="portable")`,
`CompileOptions(host_profile=...)` or `$HAWK_HOST_PROFILE`.

Neither of these two profiles changes a result: FMA contraction is off and
nothing enables fast-math or reassociation, so host results are
bit-identical to a scalar build that calls libm.

**`native-vector-math`** (the default on x86-64 with a GCC host compiler; x86-64 with GCC only): `native` plus
`-DAETHER_HOST_VECTOR_MATH -fno-trapping-math`, and `-ffp-contract=fast` in
place of `-ffp-contract=off`. A per-sample loop that calls a
transcendental function (`exp`, `log`, `pow`, `sin`, `cos`, `tan`, `tanh`,
`atan`, `atan2`, `asin`, `acos`, `cbrt`, `hypot`, ...) normally stays scalar,
because GCC cannot vectorise a call into libm. Under this profile aether
declares those functions with the x86-64 vector function ABI and supplies
vector versions (its own packet math), so the loop vectorises; `fmax`/`fmin`
become inline selects with libm's results (up to the sign of a +0/-0 tie,
which C leaves unspecified). The accuracy trade:
the transcendental functions are aether's, faithfully rounded (error < 1 ULP,
verified against a 128-bit reference; the per-function accuracy table is in aether's
`aether/backend/cpu/simd/math/PacketMath.h`), but not glibc's, so results are
not bit-identical to `native` or to the device. Arithmetic is contracted:
the compiler fuses a multiply and an add into one FMA instruction (on every
CPU that has one), which rounds once instead of twice. Each fused pair is
within half an ULP of the exact value instead of one, so results are as
accurate or more, but not bit-identical to `native`; how far they move is
bounded by the kernel's operation count and condition, and a kernel's tests
should use a tolerance derived from those, not bit equality. Code that needs
exact rounding is protected: aether's vector math is compiled without
contraction, and the `compensated` sink's two-sum pins its operands, so the
compensation stays exact. `sqrt` and the loop structure are unchanged, and a
result does not depend on whether a sample landed in a vector or a scalar
iteration.
`-fno-trapping-math` lets GCC if-convert the guarded loop body; it changes no
floating-point value, only how floating-point exceptions may be raised.

One caveat on GCC: on GCC 13.1-13.3 and 14.1-14.2 a masked call to a vector
math function can be miscompiled, so aether-dsc 0.2.3 and newer declare no masked
variants there, and a loop with a conditional math call runs scalar (the rule is `AETHER_HVM_MASKED`: off on GCC 13 before 13.4 and on
GCC 14 before 14.3, on elsewhere). hawk reports those loops as not
`vectorised`, and results stay correct; other loops stay vectorised, and GCC 13.4,
14.3 or newer vectorises these too.

It is the default on x86-64 with GCC, so `build`/`build_bundle` and the compile cache
use it unless a profile is named; it is part of the compile-cache key (with
the CPU's identity, as for `native`), so entries compiled under another
profile, including those cached when `native` was the default, are not
served for it. For results bit-identical to libm, request `native`:
`build_bundle(..., host_profile="native")`,
`CompileOptions(host_profile="native")` or `HAWK_HOST_PROFILE=native`.

**What is vectorised.** Threading is eagle's (its OpenMP host team splits the
samples into tiles). Within a tile, a `sample_local` kernel's per-sample loop is
written so the compiler can vectorise across samples: a 32-bit `aether::offset_t`
loop index, a byte-wide `terminated` mask, and an annotation that iterations are
independent (`#pragma GCC ivdep`). With `native` on an AVX-512 CPU and GCC, the
loop runs in blocks of 64 samples whose mask bytes are first widened to one
float64 guard per sample: a byte guard would make the compiler process 64
samples per vector, holding every float64 value in eight registers and spilling
them, where a float64-wide guard keeps it at 8 samples per zmm register.

Under an active set (`Guard(active_set=True)`) the loop stops at the live count
as its bound. When a tile's index map is the identity, as it is after eagle's
physical reorder, the tile runs the same contiguous, vectorisable loop. Any
other map runs the map loop, whose loads are gathers that the compiler may
leave scalar.

A `for` loop in a `sample_local` kernel runs interchanged: its trips on the
outside, a tile of samples on the inside, each carried value held in a lane
array with one slot per sample. The tile spans eight vector registers of the
kernel's float type (64 float64 or 128 float32 samples under AVX-512, half that
under AVX2, a quarter on SSE2 or NEON), so eight independent chains of a
dependent loop body are in flight at each trip, whatever the trip count. A
sample that leaves at its own `break`, or runs its own trip count, stops in its
lane and keeps its values; the trips end once every lane has stopped. A nested
`for` stays each lane's own loop. Every sample runs the same operations in the
same order as the per-sample loop, so `native` and `portable` keep their bits;
the same source built with `-DHAWK_HOST_INTERCHANGE=0` runs the per-sample loop.
A kernel whose loop pair reads a tape or another loop's count (a derived
kernel), or whose loop sits in one `switch` case, keeps the per-sample loop.

The plain scalar loop is kept for kernels that read or write other samples
(`cross_sample_*`), for reductions (reordering them would change their bits),
and for the body of a lowered inner `for`. GCC 14 vectorises a masked float64
loop only with AVX-512's masked stores, so on CPUs without AVX-512, and under
`portable`, such a loop stays scalar and only `-O3` applies.
</details>
