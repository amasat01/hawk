# Walk record, replay and shadow tools

Test tools around the compiled IR core of raptor (`rs/ir`, crate `raptor_ir`). They record every `canonical()` walk that a hawk
test run makes, replay the recorded walks against the compiled core through its C door, and mirror each walk in-process through
a generated CPython binding. They are not part of any package and nothing here ships.

- `shadow/`   pytest plugin (modes `record`, `shadow`), merge, rebuild check, statistics, refusal cases
- `osprey/`   generator of the C binding (Rust, std only), the hand-written type lifecycle, `build.sh`, `test_binding.py`
- `tools/run_shadow.sh`   runs a hawk tree's tests with the plugin

## Prerequisites
Rust (cargo), a C compiler (clang on macOS, gcc or cc on Linux), Python 3.10 or newer with its headers, pytest and pytest-xdist
in that Python, a raptor checkout (`RX_SRC`, default: the parent folder of this one) and a hawk tree. raptor and its wings must be
importable by the Python that runs the tests (installed, or through `EMIT_OVERLAY`, colon-separated path entries).

## Commands
Variables: `RX_SRC` raptor checkout, `PYTHON` interpreter (default `python3`), `OUT` output folder (default `./out`),
`EMIT_OVERLAY` extra PYTHONPATH entries, `PYTEST_MARK` marker expression (default `not gpu`; empty runs every test, GPU ones
included), `JOBS` pytest workers (default 8), `SHADOW_EXT_DIR` folder of the built extension (default `osprey/build`).
`TREE` is the hawk tree.

1. Build the crate and replay:
   `(cd $RX_SRC && cargo build -j 2 --release -p raptor_ir)`; the replay program is the binary `replay` of that crate; run it as
   `cargo run --release -p raptor_ir --bin replay -- <walks.jsonl>` (after step 3).
2. Build the binding, run its gate and its test:
   `sh osprey/build.sh` (prints the gate: four checks, ends with GATE GREEN), then
   `PYTHONPATH=osprey/build $PYTHON osprey/test_binding.py` (14 checks).
3. Record a corpus, merge it, check that it rebuilds:
   `SHADOW_MODE=record sh tools/run_shadow.sh $TREE record`, then
   `$PYTHON shadow/merge.py $OUT/corpus/record` (writes walks.jsonl and SUMMARY.json), then
   `$PYTHON shadow/rebuild.py $OUT/corpus/record` (prints `rebuild-equal K / U`; K must equal U).
4. Replay the corpus through the core, with the dump and load round trip:
   `cargo run --release -p raptor_ir --bin replay -- $OUT/corpus/record/walks.jsonl --roundtrip` (run in `$RX_SRC`; use absolute paths).
5. Shadow run: every walk is mirrored through the binding and compared with the real result:
   `SHADOW_MODE=shadow sh tools/run_shadow.sh $TREE shadow`, then
   `$PYTHON shadow/merge_shadow.py $OUT/corpus/shadow $OUT/corpus/record/walks.jsonl`.
   The test outcomes of the shadow run must equal those of a run without the plugin.

## What to report
- the pass and fail counts of each test log (`$OUT/logs/<name>.log`), and the ids of the failing tests;
- from step 3: N calls, U unique walks, raised calls, unmirrorable count, `rebuild-equal K / U`;
- from step 4: the lines `digest-equal`, `arg_spec-equal`, `access-equal`, `roundtrip-equal`, the skipped-by-feature line;
- from step 5: `shadow-equal`, `roundtrip-equal`, mismatches, shadow exceptions, the digests seen against the corpus;
- the machine (CPU, OS, compiler, Python version) and the raptor and hawk commit ids (`$OUT/corpus/<name>/PROVENANCE`).

Public API of the binding: module `_osprey` with the types `Ctx` and `Node` and the integer `ABI_VERSION`. `Node` is created only
by the binding. Values read from a node come back as text.
