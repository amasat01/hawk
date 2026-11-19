# Sweep of the released kernels

The released documentation, notebooks, examples and README promise that certain kernel shapes
build and run. This sweep turns those promises into cases and runs the same cases on two versions
of hawk, so the outcomes can be compared.

## Commands

```
python run_cases.py --label old [--targets host,cuda] [--only <id>,<id>] [--cases <dir>]
python run_cases.py --label new
python compare.py results_old.json results_new.json
```

`run_cases.py` writes `results_<label>.json` in the current directory: one row per case and
target with `id, target, describe, build, run, message, max_error` (plus `bound`, the output sums
for `compare.py`, and a short `trace` after an unexpected error). `describe` calls the source
emitter (no compiler), `build` builds, `run` runs on the host and compares with the case's
oracle. A refusal is an exception of hawk's own error type; any other exception is an error.
`--targets` replaces every case's own targets; only the host target is run, the others are built.
If a public call has another name in a version, change the tables at the top of `run_cases.py`.
`describe` is the version's `artifact.describe` over the list of a case's kernels; a version
without that call records describe as "skipped" (the released line's `emit_source` refuses kernels
that run as several launches, so it is not used). A leading `<file>.py:<line>: ` in a refusal
message is stripped, so messages compare across checkouts.

## Environment of a run

The runner needs hawk importable and numpy. Placeholders, set for the version under test:

```
HAWK_CACHE_DIR=<empty directory for the compile cache>
HAWK_OSPREY_PATH=<directory of the cpu binding>:<directory of the cuda binding>   (new line only)
PYTHONPATH=<hawk source or install>:<native binding build directory>:<overlay, if any>
<python of that environment> run_cases.py --label <name> --targets host
```

Run it from the sweep directory; results land in the current directory.

Tolerance: where the source states none, `1e-12` (float64) or `1e-5` (float32), relative and
scaled by `max(1, |expected|)`, printed as "default bound". A `tolerance` entry in a case is the
source's own `rtol`/`atol`. `compare.py` groups LOST, GAINED, CHANGED MESSAGE and VALUE DIFFERENCE
(output sums or maxima that differ by more than 1e-12 relative) and prints counts.

## Case files

`cases/<NNN>_<slug>.py` holds the kernel text as the source shows it, the imports and earlier
definitions it needs, and a `CASE` dict: `id`, `source` (every place the same text appears),
`kernels`, `targets`, `inputs`, `oracle`, plus where needed `run_kernel` (name loaded and run),
`call_form` (the source runs `kernel(**planes)` instead of `hawk.run`), `build_groups` (kernels
the source builds in one call), `build_options` (keywords of that build call), `tolerance`,
`needs`, and `definers` (functions whose call defines a kernel the source says is refused;
`kernels` is empty for those). `inputs()` returns the planes of the source's run, `oracle(planes)`
returns the expected value of a plane (or `"result"`, the return value of a call-form run).
A leading `# ruff: noqa` line marks the DSL idioms a linter reads as errors (assigning to an
output parameter, axis names written as strings in annotations).

## How sources were chosen

Every Python block of README.md, docs/content/**/*.md and the code cells of docs/content/**/*.ipynb,
plus examples/*.py, was parsed. A block that defines a kernel (decorated function, derived
`Kernel(...)`, `hawk.chain`, `hawk.resolve_dispatch`, or a call of a function that returns one)
gives a case; the earlier statements it depends on are kept by a dependency walk; identical text
from several pages is one case with several sources. Inputs come from the first `hawk.run` (or
call-form run) the source shows; an oracle comes from a `np.allclose`-style check against a pure
NumPy expression in the source, or from a formula the source states in words (six, written by
hand). A kernel and its derivative built together are one build group (the source builds them
as a list or by one call that builds the pair).

| Source kind | Code blocks scanned | Cases (any source / first source) |
|---|---|---|
| notebook cells | 445 | 119 / 115 |
| markdown pages | 41 | 25 / 24 |
| README | 9 | 6 / 6 |
| examples/*.py | 2 files | 2 / 2 |
| CHANGELOG | 0 code blocks | 0 |

146 cases: 48 with inputs, 21 with an oracle, 1 with `needs` (an eagle-only object). 12 of the 146
are refusal cases (kernels the source defines only to show they are refused).

## Promises not turned into cases

- CHANGELOG 0.7.8: staged plugins through `hawk.artifact.plugins` and eagle plans need eagle.
- 0.7.7: `hawk.sum`/`mean` with `policy="ordered"` over non-trailing lane axes is refused: no kernel text.
- 0.7.5: `hawk.chain`: the best-practices and composition snippets call it with members they never define, so no case.
- 0.7.5: `mode="banded"` builds: no kernel text, and it needs a GPU target.
- 0.7.5: `Table.at` clamping, `Vector` component read at an `m.int32` index: no kernel text.
- 0.7.5: arm hosts' default profile and AVX2 vectorisation are build behaviours, not kernel shapes.
- 0.7.4: wide-value proxy ops, a thousand-op chain for CUDA, bool arithmetic refusals: no kernel text.
- 0.7.3: wide per-sample values (`Vector[512]`), `launch_bounds`, `Staged[...]`: only prose in the source.
- 0.7.2: ray functions, lane sets, 4096-square matrices, ordered sums over kept axes: prose only,
  apart from what the docs pages show (those are cases).
- 0.7.1 and 0.7.0: compiler, cache and target-table changes: no kernel text.
- A kernel run inside a function or a loop of the source, with unpacked planes (`**bvh.planes()`) or
  positional arguments: kernel and build are cases, but no inputs are taken (nine cases whose planes
  were local to a function or loop have their inputs removed).
- Cells that need torch, jax, cupy, matplotlib or eagle objects for their planes: kernel kept, no inputs.
- examples/two_body_hybrid.py and autodiff_vs_torch.py need torch to run; only the kernels are cases.
- Reference pages with no code block (api, glossary, explanation, vocabulary prose).
