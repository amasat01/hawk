<p align="center">
  <img src="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/brand/glyph_hawk.svg" alt="" height="56">
</p>
<h1 align="center">hawk</h1>
<p align="center">Write per-sample kernels in Python, with derivatives. Part of the RAPTOR family.</p>

<h3 align="center">hawk: one kernel, two machines, and its derivatives</h3>
<p align="center">Write a numerical kernel for one sample as a plain Python function.
hawk compiles it for the GPU or the CPU and derives its gradient (reverse mode) and tangent (forward mode).</p>

<p align="center">
  <a href="https://github.com/amasat01/hawk/actions/workflows/ci.yml"><img src="https://github.com/amasat01/hawk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://amasat01.github.io/hawk/"><img src="https://github.com/amasat01/hawk/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/amasat01/hawk/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/raptor-hawk/"><img src="https://img.shields.io/pypi/v/raptor-hawk.svg" alt="PyPI"></a>
  <a href="https://doi.org/10.5281/zenodo.23250242"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23250242.svg" alt="DOI"></a>
</p>

<p align="center">
  <a href="https://amasat01.github.io/"><b>The RAPTOR family</b></a><br>
  <a href="https://amasat01.github.io/hawk/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_hawk_hawk_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_hawk_hawk_light.svg" alt="hawk" width="430"></picture></a>
  <a href="https://amasat01.github.io/eagle/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_eagle_hawk_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_eagle_hawk_light.svg" alt="eagle" width="430"></picture></a><br>
  <a href="https://amasat01.github.io/aether/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_aether_hawk_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_aether_hawk_light.svg" alt="aether" width="430"></picture></a>
  <a href="https://amasat01.github.io/raptor/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_raptor_hawk_dark.svg"><img src="https://raw.githubusercontent.com/amasat01/hawk/main/docs/_static/ecosystem/ecosystem_card_raptor_hawk_light.svg" alt="raptor" width="430"></picture></a>
</p>

hawk is the kernel-authoring layer of the [RAPTOR family](https://amasat01.github.io/): write a kernel once at the
level of your application, and hawk derives its forward- and reverse-mode derivatives and compiles it for host or
device through eagle. See [where RAPTOR fits](https://amasat01.github.io/where_raptor_fits.html) for the full
four-part story.

**[Read the documentation](https://amasat01.github.io/hawk/)** — installation, a five-minute quickstart, tutorials,
examples and the full Python API reference.

## Write a kernel, get its gradient

```python
import pathlib, tempfile

import hawk
from hawk import Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp
from hawk.math import dot

@hawk.kernel
def energy(v: Vector[3], out: Mutable[Scalar]):
    out = 0.5 * dot(v, v)

energy_vjp = Kernel("energy_vjp", vjp(energy, wrt=("v",)))
work = pathlib.Path(tempfile.mkdtemp())
bundle = hawk.build([energy, energy_vjp], work, targets=("host",))
```

```python
import numpy as np

v = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
e = np.zeros(4)
hawk.run(hawk.load(work, "energy"), v=v, out=e)
print("energy:", e)
# energy: [0.5 2.5 5.  9. ]
```

```python
bar_out = np.ones(4)             # seed: d(loss)/d(energy) = 1 for every sample
bar_v = np.zeros((3, 4))
hawk.run(hawk.load(work, "energy_vjp"), v=v, bar_out=bar_out, bar_v=bar_v)
print("gradient d(energy)/dv:")
print(bar_v)
# [[1. 2. 3. 4.]
#  [0. 1. 0. 1.]
#  [0. 0. 1. 1.]]
```

`energy = 0.5 |v|^2`, so `d(energy)/dv = v` exactly — the printed gradient above *is* the input `v`. Output above
is the notebook's own: [`docs/content/tutorials/04_gradients_backward.ipynb`](https://github.com/amasat01/hawk/blob/main/docs/content/tutorials/04_gradients_backward.ipynb). `hawk.diff.jvp` derives the forward-mode (tangent) twin the same way.

**[Five-minute quickstart](https://amasat01.github.io/hawk/)** · install: `pip install raptor-hawk` (CPU route; it brings `aether-dsc` along), or `pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"` for the GPU route

## Writing kernels

A kernel is a plain Python function decorated `@hawk.kernel`, one sample at a
time; its parameters are its declared planes (`Scalar`, `Vector[3]`, `Mutable`,
...) and its body writes ordinary arithmetic:

```python
import hawk
from hawk import Mutable, Scalar, Terminated, Vector
from hawk.math import maximum, norm

@hawk.kernel
def running_apogee(position: Vector[3], terminated: Terminated,
                    r_max: Mutable[Scalar]):
    r_max = maximum(r_max, norm(position))
```

<details>
<summary>Running statistics: read it before you write it</summary>

A `Mutable` parameter such as `r_max` is an ordinary local variable: read
it before you assign it and you get the value it held when THIS launch
began, which is how a running statistic is written — a launch reads what
an earlier launch left behind, and writes a new value on top of it. Your
buffer remembers; the kernel reads what it remembered. The host owns that
memory: it must initialise `r_max` before the first launch and preserve it
between launches — hawk never zeroes or scratch-allocates a plane a kernel
reads from. That memory is not differentiable: a launch-start read is a
recurrence across launches, and hawk keeps no tape across them, so
`hawk.diff.vjp`/`jvp` refuse a kernel that reads one — differentiate the
per-launch term on its own instead.
</details>

<details>
<summary>Kernel kinds and returned outputs</summary>

A kernel may commit through several **sinks** — the places a kernel's
result goes: declared `Mutable`/`Accum` parameters (`Output.named(...)`),
a `return <expr>` committed into a **synthesised slot** (a result sink
hawk creates for you rather than one you declared as a parameter;
`Output.returned(ttype, slot=...)` names it), or both together — a
returned acceleration alongside an ordinary diagnostic `Mutable` the body
also writes:

```python
from hawk import Param
from hawk.ext import Kind, Output

ACCEL = Kind("drag", output=Output.returned(Vector[3], slot="acc"))

@ACCEL
def drag(velocity: Vector[3], k: Param, speed: Mutable[Scalar]):
    speed = norm(velocity)          # an ordinary named sink
    return (-k * norm(velocity)) * velocity   # the returned slot, "acc"
```

`Kind(...)` also has a class-form spelling — sugar over the same object, one
kernel-authoring style closer to a plugin author's own vocabulary: an
annotated class attribute is a vocabulary entry, a plain-assigned one sets
one of `Kind`'s own fields (`output`, `sink`, `guard`, ...), and anything
else refuses.

```python
from hawk.ext import KernelKind, Output

class Accel(KernelKind, slug="drag"):
    output = Output.returned(Vector[3], slot="acc")

@Accel
def drag(velocity: Vector[3], k: Param):
    return (-k * norm(velocity)) * velocity
```
</details>

## Install

```bash
pip install raptor-hawk                                     # CPU only: no GPU, driver or CUDA packages needed
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"    # GPU: hawk compiles, eagle runs; [cuda13] on both for CUDA 13
```

Linux x86_64, CPython 3.9-3.14 (including free-threaded 3.13t and 3.14t), and a host `g++` 11 or newer. `raptor-hawk` pulls `aether-dsc` (the sealed C++
headers hawk compiles against) automatically. No `nvcc` or CUDA toolkit is needed. Tested with CUDA 12.6 and
CUDA 13.0 (CUDA 12.6 or newer).

> **NVIDIA packages come only through the extras.** `raptor-hawk[cuda12]` pulls `cuda-bindings` 12,
> `nvidia-cuda-nvrtc-cu12` and `nvidia-cuda-cccl-cu12`; `raptor-hawk[cuda13]` pulls `cuda-bindings` 13,
> `nvidia-cuda-nvrtc` 13 and `nvidia-cuda-cccl` 13 (CUDA 13's wheels have no `-cu13` suffix);
> `raptor-eagle[cuda12]` / `[cuda13]` pull CuPy (`cupy-cuda12x` / `cupy-cuda13x`, with the CUDA headers CuPy compiles against). Without an extra pip installs no NVIDIA package: you get the CPU route, or the GPU route through a CUDA setup you already have. Pick the extra matching the CUDA version your driver reports (`nvidia-smi`, top right).

**Platforms:** built and tested on Linux x86_64 only so far (CPython 3.9–3.14, including free-threaded 3.13t and 3.14t), on NVIDIA GPUs from Pascal (Quadro P2000) and Turing (Tesla T4). There are no wheels for macOS, Windows or ARM yet, and WSL2 is untested. `raptor-core` and `aether-dsc` are pure Python and install anywhere. Free-threaded builds (3.13t, 3.14t) ship without declaring GIL-free support, so CPython re-enables the GIL when `hawk` is imported and prints a RuntimeWarning; results are correct, just not parallel.

See [installation](https://amasat01.github.io/hawk/content/installation.html)
for the CPU-only route (no GPU, driver or NVRTC needed at all) and every
other detail.

## Performance

Every number in this section is read from eagle's committed cards (`benchmarks/perf_card/card_quadro-p2000.md`, `benchmarks/rk78_card/card_quadro-p2000.md`).
Kernels authored with hawk are what eagle's performance card times: hawk + eagle (`eagle.simulate`) is
**3.9×–47× faster** than NVIDIA Warp (3.9×), JAX (8.7×), CuPy and PyTorch (47×) at 1,000,000 RK4
oscillators finishing at different times — **156 ms** on a
Quadro P2000, using **3.8×–5.2× less GPU memory** than CuPy and PyTorch (a million samples in 50 MiB, 1.31× the
bare minimum state; CuPy 5.1×, PyTorch 6.9×; NVIDIA Warp 64 MiB, 1.7×) — and the same hawk kernel, unmodified, integrates 1,000,000 adaptive
RK7(8) orbits in **13.0 s** on the GPU or **7.15 s** on 8 CPU threads alone. This is the regime hawk + eagle's
compaction targets (many samples stopping at different times); in other regimes the other tools hold their own — on a
dense workload where nothing finishes early Warp ties it (693 ms vs 699 ms at N = 1,000,000) and is level at
N = 10,000 (7.22 ms vs 7.22 ms), and on this FP64-weak development card the CPU arm is the faster one there. See
[eagle's performance page](https://amasat01.github.io/eagle/content/performance.html) for every number, every arm,
including the ones where it doesn't win.

## Going deeper

<details>
<summary>The pip-only device compile, directly</summary>

The device compile `hawk.artifact.build`/`build_bundle` use when no `nvcc`
is on `$PATH` goes through `hawk.compile.device`/`hawk.compile.cubin`; the
same entry points are usable directly, for an ad hoc CUDA C++ `source`
string that was never authored as a traced hawk kernel at all:

```python
from hawk.compile import cubin, cubin_available

if cubin_available():
    image = cubin(source)          # DeviceImage(image=<SASS bytes>, target="cubin", arch="sm_XX")
```

`cubin_available()` never raises; `device(source, target="ptx")` is the
alternative when PTX (the portable intermediate form NVRTC can also emit)
is what a caller needs, guarded against a driver too old to JIT a newer
NVRTC's PTX (`target="cubin"` needs no such guard — it is already SASS,
the GPU's own machine code). Headers come from `aether_dsc`'s **sealed
payload** — a copy of the headers hawk's device compiler reads from
directly, bundled into the wheel — never from disk.
</details>

<details>
<summary>Build and test from a source checkout (development)</summary>

Clone `aether` and `eagle` next to this checkout (so `../aether`
and `../eagle` sit beside `hawk/`), then install aether's sealed header
payload before hawk itself:

```bash
pip install ../aether/dsc
pip install -e .[test]
pytest -m "not gpu"
```

The compiled half (`hawk._core`) is a nanobind extension built through
`scikit-build-core`; `pip install -e .` builds it via `CMakeLists.txt`,
resolving the `aether`/`eagle` C++ header roots from those sibling checkouts
(or `$HAWK_AETHER_INCLUDE`/`$HAWK_EAGLE_INCLUDE`). `-m "not gpu"` deselects
the tests that need a CUDA GPU and `cupy`; most of the rest additionally
need `eagle` installed as a Python package to execute traced kernels on host
or device (see the "Python package" section of
[eagle's README](https://github.com/amasat01/eagle#readme)).
</details>

---
<p align="center">
  <b>hawk</b> write it (this repo) ·
  <b>eagle</b> run it ·
  <b>aether</b> the numerics core ·
  <b>raptor</b> the shared contracts —
  <a href="https://amasat01.github.io/">the RAPTOR family</a>
</p>

Apache-2.0 (see [`LICENSE`](https://github.com/amasat01/hawk/blob/main/LICENSE) and
[`NOTICE`](https://github.com/amasat01/hawk/blob/main/NOTICE)) · cite via "Cite this repository"
(`CITATION.cff`; every tagged release is archived on Zenodo: [doi:10.5281/zenodo.23250242](https://doi.org/10.5281/zenodo.23250242)) · built to
make GPU computing accessible on modest hardware, for research and education. Collaboration is the point, and a
citation is the currency — get in touch.
