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

Runs on CPU.

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

**[Five-minute quickstart](https://amasat01.github.io/hawk/content/quickstart.html)** · install: `pip install raptor-hawk raptor-eagle matplotlib` (CPU route, no extras), or `pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"` for the GPU route

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
per-launch term on its own instead. A STEP kernel's whole rollout is the
exception a runtime handles: `hawk.diff.tape(kernel, window)` builds the
step's taping twin, which records the state every step read, so a
launching runtime can reverse the rollout window by window
([Tapes and events](https://amasat01.github.io/hawk/content/vocabulary/05_tapes_and_events.html)).
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

## Threading

Free-threaded CPython (3.13t/3.14t) is supported: the compiled modules declare GIL-free operation and the GIL stays disabled after import. Any number of threads may call module-level functions, build, compile, plan and cache concurrently. Distinct objects may be used from distinct threads without synchronisation. One stateful object (a stream, capture, launcher, graph, composer, plan, pipeline, active set, host kernel or arg block) shared by several threads is memory-safe — its calls serialise and a consumed object raises — but the ORDER of those calls is the caller's responsibility, exactly as for a NumPy array or a CuPy stream. CUDA adds two rules of its own: a stream capture is begun, filled and ended by one thread, and while any capture is open no thread may synchronise the whole device (stream-level synchronisation is fine). GPU routes are supported on free-threaded 3.14; on 3.13t the CPU route runs. hawk's compile cache is shared safely between threads, and a sealed-path kernel's cache key follows the user headers it includes.

## Install

```bash
pip install raptor-hawk raptor-eagle matplotlib          # CPU: no GPU, driver or CUDA packages (matplotlib is for the tutorials' plots)
pip install "raptor-hawk[cuda12]" "raptor-eagle[cuda12]"    # GPU: hawk compiles, eagle runs; [cuda13] on both for CUDA 13
```

Extras, platforms, requirements and the CPU-only route: [installation](https://amasat01.github.io/hawk/content/installation.html).

## Performance

Every number in this section is generated from the performance cards committed in eagle (`benchmarks/perf_card` and `benchmarks/rk78_card`):
`tools/sync_perf_claims.py` rewrites them, and `tests/test_perf_claims.py` fails if this text, the snapshot `docs/perf_claims.json` or (with `RAPTOR_EAGLE_ROOT` set to an eagle checkout) the cards disagree.
Kernels authored with hawk are what eagle's performance card times.

<!-- hero:begin -->
hawk + eagle (`eagle.simulate`) integrates 1,000,000 RK4 oscillators, each stopping at its own step, in 613 µs on an NVIDIA H100 80GB HBM3 (199 µs for 100,000), 2.5×–600× faster than NVIDIA Warp (2.5×), JAX (76×), CuPy (600×), PyTorch with torch.compile (130×) at N = 1,000,000 (1.1×–1700× at N = 100,000), in 40 MiB of GPU memory
<!-- hero:end -->
— and, on the <!--c spec card=rk78 dev=h100 field=device.name-->NVIDIA H100 80GB HBM3<!--/c--> reference GPU, the same hawk kernel, unmodified, integrates 1,000,000 adaptive
RK7(8) orbits in <!--c wall card=rk78 dev=h100 arm=eagle_simulate n=1000000-->43.6 ms<!--/c-->; on a CPU alone, through eagle's host team, the same batch takes <!--c wall card=rk78cpu dev=amd-epyc-9b45 arm=eagle_cpu8 n=1000000-->3.76 s<!--/c--> on 8 threads of an <!--c spec card=rk78cpu dev=amd-epyc-9b45 field=cpu-->AMD EPYC 9B45<!--/c--> (the no-GPU route, for machines without a GPU). This is the regime hawk + eagle's
compaction targets (many samples stopping at different times); in other regimes the other tools hold their own. On a
dense workload where nothing finishes early, NVIDIA Warp is the fastest arm on that H100 at N = 1,000,000
(<!--c wall card=rk4 dev=h100 n=1000000 dist=uniform steps=1000 arm=warp_kernel-->1.71 ms<!--/c--> vs <!--c wall card=rk4 dev=h100 n=1000000 dist=uniform steps=1000 arm=eagle_simulate-->1.75 ms<!--/c--> for `eagle.simulate`, which takes <!--c pctgap card=rk4 dev=h100 n=1000000 dist=uniform steps=1000 arm=eagle_simulate den_arm=warp_kernel-->2.5%<!--/c--> longer), while at N = 10,000 on the same workload
`eagle.simulate` is ahead (<!--c wall card=rk4 dev=h100 n=10000 dist=uniform steps=1000 arm=eagle_simulate-->112 µs<!--/c--> vs <!--c wall card=rk4 dev=h100 n=10000 dist=uniform steps=1000 arm=warp_kernel-->121 µs<!--/c-->). The development card the tutorials teach on is the Quadro P2000; its numbers are on the performance page next to every other device. See
[eagle's performance page](https://amasat01.github.io/eagle/content/performance.html) for every number, every arm,
including the ones where it doesn't win.

<p align="center"><a href="https://amasat01.github.io/eagle/content/performance.html"><img src="https://raw.githubusercontent.com/amasat01/eagle/main/docs/_static/perf/readme_gpu.svg" alt="Wall time per tool for a million RK4 oscillators on the default GPU: eagle.simulate, NVIDIA Warp, JAX, CuPy and PyTorch; the numbers are on eagle's performance page." width="732"></a></p>

The chart shows one GPU; [the performance page](https://amasat01.github.io/eagle/content/performance.html) has a tab for every device and CPU measured.

The same hawk kernel also runs on a CPU alone, through eagle's host team: on an <!--c spec card=rk4cpu dev=amd-epyc-9b45 field=cpu.model-->AMD EPYC 9B45<!--/c--> (<!--c spec card=rk4cpu dev=amd-epyc-9b45 field=cpu.cores-->8<!--/c--> cores, <!--c spec card=rk4cpu dev=amd-epyc-9b45 field=cpu.logical_cpus-->16<!--/c--> logical CPUs) the 8-thread spread S=1000 batch of 1,000,000 samples takes **<!--c wall card=rk4cpu dev=amd-epyc-9b45 arm=eagle_term8 n=1000000 dist=spread steps=1000-->142 ms<!--/c-->**. eagle's [performance page](https://amasat01.github.io/eagle/content/performance.html) has a tab for every CPU and GPU measured, with Numba, JAX, PyTorch, NumPy and multiprocessing on the CPU cards, each with the setting where it fits best; on some cells and CPUs another tool is ahead, and the card shows which.

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
