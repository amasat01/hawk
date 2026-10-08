# Installation

**Time:** ~2 min &middot; **You need:** a terminal and `pip`

```bash
pip install raptor-hawk                                     # CPU only
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"    # GPU route; [cuda13] on both for CUDA 13
```

Linux x86_64, CPython 3.10-3.13, and a host `g++` 11 or newer. `raptor-hawk` pulls
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

**Platforms:** built and tested on Linux x86_64 only so far (CPython 3.10–3.13), on NVIDIA GPUs from Pascal (Quadro P2000) and Turing (Tesla T4). There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere.

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

Verified: a clean venv built from this project's own wheels
(`raptor-hawk`, `aether-dsc`, `numpy`; `eagle` and `cupy` not installed),
run with `$PATH` stripped to the system's own `/usr/bin:/bin` (no `nvcc`,
no CUDA toolkit), `$LD_LIBRARY_PATH` unset and no GPU selected
(`$CUDA_VISIBLE_DEVICES` unset) — the first tutorial's `scale` kernel built
and ran, computing the right answer, with `eagle` and `cupy` both absent
and unneeded. The GPU route below is for compiling and *running* a device
artifact; it changes nothing about this one.

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

To work on hawk itself, build it from a clone. hawk has a compiled half (`hawk._core`, a nanobind extension) and depends
on two build-time siblings, `aether` and `eagle`, taken from checkouts for development. Clone all three
next to each other —

```text
workspace/
├── aether/
├── eagle/
└── hawk/
```

— then install aether's sealed header payload before hawk itself:

```bash
pip install ../aether/dsc
pip install -e .[test]
```

`pip install -e .` builds `hawk._core` through `CMakeLists.txt`
(scikit-build-core + nanobind, the same packaging eagle's own binding uses),
resolving the `aether`/`eagle` C++ header roots from the sibling checkouts
above — or from `$HAWK_AETHER_INCLUDE` / `$HAWK_EAGLE_INCLUDE` when they live
somewhere else.

To run a DEVICE artifact end to end (not just author and compile it) you
also need `eagle` installed as a Python package — see the "Python
package" section of [eagle's installation page](https://amasat01.github.io/eagle/content/userguide/installation.html) — since hawk compiles
a device artifact but never launches one itself. A host artifact needs
no `eagle` at all: `hawk.load`/`hawk.run` run it directly, as
[CPU-only](#cpu-only) above shows.

## Requirements

- Python 3.10 or newer (CPython 3.10–3.13).
- CUDA 12.6 or newer for the device target (tested with 12.6 and 13.0).
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

The host target is compiled at run time by the host `g++` (or `$HAWK_CXX`)
under one of three profiles. On x86-64 the default is `native-vector-math`
(below), which vectorises transcendental math with aether's own functions,
faithfully rounded (error < 1 ULP), and lets the compiler fuse `a*b + c`
into one FMA instruction; on other architectures the default is
`native`. The other two profiles are the EXACT ones: they
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

**`native-vector-math`** (the default on x86-64; x86-64 with GCC only): `native` plus
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
verified against a 128-bit reference over more than 10^6 inputs per function;
the per-function table is in aether's
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

Measured before FMA contraction joined this profile, on the RKF7(8) attempt
kernel of eagle's `rk78_card` (10^5 Kepler orbits to t = 64, Xeon W-2125,
GCC 14.3): the loop vectorises (1514 packed zmm float64 instructions in the
kernel, masked 8-lane `pow` calls) where `native` keeps it scalar; the whole
run takes 6.5 s instead of 14.6 s on 1 thread and 5.1 s instead of 7.9 s on
8. Every sample made the same accept/reject decisions and step count
(49,797,416 attempts in both); final states differ by at most 1.3e-8 relative,
and the error against the analytic solution is the same (2.2e-7 maximum).
It is the default on x86-64, so `build`/`build_bundle` and the compile cache
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

The plain scalar loop is kept for kernels that read or write other samples
(`cross_sample_*`), for reductions (reordering them would change their bits),
and for the body of a lowered inner `for`. GCC 14 vectorises a masked float64
loop only with AVX-512's masked stores, so on CPUs without AVX-512, and under
`portable`, such a loop stays scalar and only `-O3` applies.
</details>
