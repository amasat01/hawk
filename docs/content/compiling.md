# Compiling

How hawk turns a kernel into something that runs.

Building is done by the backend libraries you installed (one for the host, one for the GPU): they lower the kernel,
run the compiler and keep the result in a cache. hawk asks them and hands you the files.

An embedded C build is not available in this version: asking for the `c` target, or giving an `embedded_profile=` or a `c.<key>` option, is refused by name:
`the embedded C target is not available in this version: build for 'host' or 'cuda'`.

## The compile guard

Four limits keep a compile from running away unseen. Each is read from an environment variable when you build, and the
same four apply to the host and the GPU library. A value that is not a non-negative integer is refused, naming the
variable.

| Limit | Default | Variable | What it does |
|---|---|---|---|
| Source size | 8 MiB | `HAWK_COMPILE_SOURCE_BUDGET` (bytes) | A larger source is refused before the compiler sees it, with its size. |
| Time | 1800 s | `HAWK_COMPILE_TIMEOUT` (seconds) | A compiler that has not finished is stopped. |
| Address space | 12 GiB | `HAWK_COMPILE_ADDRESS_CAP` (bytes) | A compiler that runs out is reported as a source too large for one compile, not as a machine out of memory. |
| Compiles at once | 2 | `HAWK_COMPILE_JOBS` | Compiles that run together wait for a free slot, across processes that share a cache folder. |

`0` turns a limit off; for `HAWK_COMPILE_JOBS` it means no cap. Each refusal names its variable.

A source this large is usually a wide kernel unrolled. Roll the loop, or read a wide table at a
runtime index with [`hawk.math.index`](tutorials/06_lookups_scatters_and_sums), which keeps the
source small and the gradient cheap.

## The tools a build uses

`hawk.targets.buildable("host")` says whether the library finds what it needs on this machine, and
`hawk.targets.machine("host")` returns what it says about it: the compiler and its identity, the profile, the flags, and for
the GPU the device architecture. For the host, `exact` says whether the resolved profile gives bit-exact values, and
`asked["options"]["profile"]` is that profile. The keys are the library's own. It raises `ToolchainAbsent` with the library's reason
when the tools are missing. The same options you give `hawk.build` (`options=`, `opt_level=`) can be given to it.

Which variables a library reads is in [Installation](installation): `HAWK_CXX`, `CXX`, `HAWK_OPT_LEVEL` and
`HAWK_HOST_PROFILE` for the host, `HAWK_NVCC`, `HAWK_NVCC_CCBIN`, `CUDAHOSTCXX`, `HAWK_CUDA_ARCH` and
`HAWK_DEVICE_COMPILER` for the GPU. A machine without a tool is refused with a sentence that lists the ways to name it.

## Asking before you build

`hawk.artifact.describe(kernels, targets=None, ...)` returns, for each kernel by name, the sidecar a build would write, with
no compiler run, so it answers on a machine with no tools. It leaves out what exists only after a compile (the unit files,
each tool's figures, the measured launch figures); the rest has the sidecar's own shape, so one reader serves a kernel
before and after its build. It refuses what `build` refuses.

Among its members:

- `arg_dtypes` has the dtype of every slot, uniform and lookup slots included, and `slot_shapes` the static shape of each.
- `digest` identifies the lowered unit: it covers what a kind adds when the unit is lowered (an active-set kind's two
  lookup reads, for one). The identity of the traced body alone is `Kernel.body_digest`, which two kernels with the same
  body share whatever their names and kinds.
- `planes_written` lists the planes the kernel writes, sorted.
- For an `auto` dispatch, `dispatches[].cost` states the operation counts of each branch.
- `plane_names` gives the `group` and `wire` name of every plane whose name joins the two, so you never split a name
  yourself.
- `index_bytes` is the width in bytes of the index type of a host unit. A build states the width it compiled with; `describe`
  states the library's default.
- `contractions` lists each dense contraction a unit is: the `unit`, the `route` the library chose (`gemm` for matrix
  products beside the kernel, `direct` for the kernel's own loop) and the `products` with their shapes. A kernel that is
  none has no such member.
- A reverse derivative (`hawk.diff.vjp`) states `gradient_of`, which maps each gradient plane to the input it is the
  gradient of, e.g. `{"bar_x": "x"}`; `adjoint_like`, the part of it whose planes are accumulated at lookup indices (each
  is the size of its lookup plane, not the lane count); and `cotangent_of`, which maps each cotangent (seed) plane to the
  forward output it seeds, e.g. `{"bar_out": "out"}`. None is stated for a kernel that is not a derivative.
- A step as a pure map (`hawk.diff.step_function`) and the derivative of one (`hawk.diff.step_vjp`) state `next_of`, which
  maps each plane that holds a carried plane's value after the step to that carried plane, e.g. `{"x_next": "x"}`. With
  `cotangent_of`, `{"bar_x_next": "x_next"}`, a reader reaches the carried plane `x` of a seed without reading a name.
- A forward derivative (`hawk.diff.jvp`, `step_jvp`, `with_derivative`) states `tangent_of`, which maps each tangent plane, inputs
  and outputs alike, to the plane it is the tangent of, e.g. `{"dot_x": "x", "dot_y": "y"}`.
- A kernel declaring `time_step=` has `refine_planes`: `{"t0": ..., "tau": ...}`, the names of the two extra planes its
  refine entry reads.
- A kernel that runs as several units answers with its first unit. Its `stages` list the others, and each stage
  carries `sidecar`: the document of its own unit.

## The cache

Built units are kept in one folder: `$HAWK_CACHE_DIR`, else `$XDG_CACHE_HOME/hawk`, else `~/.cache/hawk`; `cache_dir=` on a
build names another. Each library keeps its entries under `<folder>/<two hex digits>/<key>/`. The key covers the source,
the compile command with its flags and defines, the identity of each tool, the headers the library carries, the library
itself, the optimisation level and the options. The same request on the same machine is the same entry, and a default you
state is the same entry as one you leave out. The `probes/`, `headers/` and `jobs/` folders beside them belong to the
library: read nothing from them.

`hawk.artifact.cache_stats()` returns each library's counters and `hawk.artifact.reset()` empties the cache. Deleting the
folder does the same.

A unit published to a folder holds a stamp file, `.hawk_stamp` (its name is `hawk.artifact.STAMP_NAME`). It is JSON: the
unit's digest (`Bundle.digest`) and the digest of each published file, so you can check the files against what hawk
published. Building the same unit again into a folder that already holds it leaves the folder alone.

The first build of a kernel logs one line at INFO level on the `hawk` logger, naming the kernel, the target and the cache
folder; a build the cache answers logs nothing.

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
over lane axes (both contracts), the ordered stages, a keyed `hawk.scatter` with the default or compensated contract, and
the sum across the lanes of a warp. The accumulate always takes the ordered stages, so it gives the same bits on every
run; a packed cell has no atomic add, so `hawk.scatter(..., contract="plain")` is refused. The accumulator's own figure,
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
