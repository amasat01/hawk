# Tools for checking the compiled slice on another machine

This folder holds what is needed to build the compiled slice, record the render calls of two test suites, replay the
recordings through the compiled writers and lowering, and run the suites against the compiled host library. Nothing here needs
the network.

## What is new since the last push
- Device units: the writers now write the device text of a unit as well as the host text (`treplay --target cuda`, and the
  `cuda` lines of the proof).
- Machine facts are data: the library loads them from `osprey_host/facts/` (schema checked) and the caller asks for them through
  the table entry `facts` instead of sending a machine record.
- Every refusal inside the writers carries a kind (not written / bad request / fault) that travels to the library's answer.
- The lowering sits behind ONE request per kernel: the table entry `lower` (abi 6) takes a kernel and its targets and answers
  every unit (stages, segments, chains, warp per lane) with the text, the stage block and the run facts.
- `kreplay` replays a kernel-level recording through `lower` and counts each outcome in one category (equal, equal only to an
  earlier render, equal with an adapted request, accepted, ambiguous record, ...); the totals reconcile on one printed line.
- `shadow/k6_plugin.py` records, per build call, the kernel before lowering, its units and the options the build used.
- Memory fix of the per-unit tools: `merge_text.py` and the replays peak at about 2.5 times the largest recorded line (one 1.09 GB
  line took 20 GB before).
- The runners put `-o pythonpath=` on the pytest command line; the eagle runners mirror eagle's layout (see "Fixes" below).

## Contents
- `osprey_host/`: the host library (source, gate, binding, replay tools `treplay` and `kreplay`, live test plugin, runners).
- `shadow/`: the pytest plugins that record (`emit_record.py`, `shadow_plugin.py`, `text_plugin.py`, `k6_plugin.py`) and the
  scripts that merge and rebuild a recording (`merge_text.py`, `rebuild_text.py`, `rebuild.py`, `k6_stats.py`, `shadow_mirror.py`).
- `tools/`: `proof.sh` (the portable per-unit proof over three source trees), `run_shadow.sh` (record the hawk suite; the eagle recorder runner is
  `osprey_host/tools/run_eagle_text.sh`), `run_k_gpu.sh` (kernel-level recording of the GPU-marked tests only), `run_kreplay.sh` (the kernel replay),
  `expected.txt` (expected numbers, recorded on an Arm Mac). `MANIFEST.txt`: every file with its line count.

## Prerequisites
Rust (cargo, rustfmt, clippy), `clang`, `make`, Python 3 with pytest and pytest-xdist, `nm` (or `llvm-nm`), a checkout of the
hawk tree and of the eagle tree with their environments, ruff and clang-format, `clang-tidy` (optional). Layout: one work folder `WORK` that holds this tree's folders next to the source trees: `WORK/raptor`,
`WORK/aether`, `WORK/osprey_host`, `WORK/shadow`, `WORK/tools`. In the shell: `export RUN=<an empty run folder>` and
`export PY=<python of the hawk environment>`. Disk: do NOT record the CPU suite at kernel level (it is far too large).

## What a difference means
A unit that is `written but different` or that a writer refuses on this machine is a FINDING. Report it with the unit's name,
its test id and the first differing line pair. Do not fix anything on this machine. Units in `osprey_host/corpus_accepted.tsv`
are known text differences of the first machine; their keys differ here, so report differing units by NAME and test id.
The numbers in `tools/expected.txt` were recorded on an Arm Mac: host text that depends on the instruction set may differ.

## The run sheet
Report for every leg: the command's last 5 lines, and anything red with its first error line.

(B) raptor crates and gate
    cd WORK/raptor && cargo build --release -j 2 -p raptor_ir -p raptor_wings -p raptor_support
    NM=$(command -v llvm-nm || command -v nm) sh ci/gate_rs.sh
(W) writers crate and gate
    cd WORK/aether && RAPTOR_IR_LIB=WORK/raptor/target/release/libraptor_ir.a sh ci/gate_rs.sh
  For both, report the `GREEN:` and `RED:` lines (and for W the number of tests passed).

(H) host library, gate and binding
    cd WORK/osprey_host && cargo build --release -j 2 --features old-contract
    PYTHON=$PY CLANG_TIDY=<clang-tidy or leave unset> make gate
  The Python 3.9 leg skips with a printed line when no 3.9
  interpreter is named (`PYTHON39`, `PY39_INCLUDE`): not needed on Linux. The cformat leg is pinned to clang-format major 23 and
  skips with a printed line on another major. Report the leg list.

(P) per-unit replay
    export ROOT=$RUN RECORDER_DIR=WORK/shadow PYTHON=$PY
    EXTRA_PLUGINS=text_plugin TEXT_MODE=record sh WORK/tools/run_shadow.sh <hawk tree> hawk_text
    EXTRA_PLUGINS=text_plugin TEXT_MODE=record MARK=gpu RUN_PREFIX="<gpu wrapper>" TEXT_DIR=$RUN/corpus/hawk_text \
        sh WORK/tools/run_shadow.sh <hawk tree> hawk_text_gpu
    sh WORK/osprey_host/tools/run_eagle_text.sh eagle_text 4      # EAGLE_SRC, EAGLE_RAPTOR, EAGLE_OVERLAY, EAGLE_TREE as in its header
    cd WORK/shadow && PYTHONPATH=. $PY merge_text.py $RUN/corpus/hawk_text && PYTHONPATH=. $PY merge_text.py $RUN/corpus/eagle_text
    cd WORK/osprey_host
    for c in hawk_text eagle_text; do for t in host cuda; do
      ./target/release/treplay $RUN/corpus/$c/texts.jsonl --target $t --accepted corpus_accepted.tsv > $RUN/replay_${c}_$t.txt 2>&1
    done; done
  Per corpus report: `units`, `text identical (whole path)` (host) or `(from the recorded body)` (cuda), `written but different`,
  each refusal class with its count, the lines `typed contract:` and `typed wire:`, and each differing unit by name and test id.
  `RAPTOR=WORK/raptor AETHER=WORK/aether OSPREY=WORK/osprey_host CORPUS=$RUN/corpus SCRATCH=$RUN/scratch sh WORK/tools/proof.sh quick`
  compares with `$RUN/corpus/expected.txt` (the format of `tools/expected.txt`; write it from the first clean run) and with a
  baseline dump of every text (written on the first run, compared byte for byte after). It stops, saying so, when a build fails.

(K) kernel-level recording of the GPU-marked tests (only those)
    export ROOT=$RUN RECORDER_DIR=WORK/shadow PYTHON=$PY RUN_PREFIX="<gpu wrapper>"
    sh WORK/tools/run_k_gpu.sh <hawk tree> hawk_gpu_k
  The corpus is `$RUN/corpus/hawk_gpu_k` (`k*.jsonl` per build call, `t*.jsonl` the render records); the script ends with the
  counts of `k6_stats.py`. Report them and the failed tests.

(KR) kernel replay
    OSPREY=WORK/osprey_host sh WORK/tools/run_kreplay.sh $RUN/corpus/hawk_gpu_k
  The result is the reconciliation line of the host and of the device (`outcomes N = equal a + ... : adds up`); the full output
  is `kreplay.txt` in the corpus folder. Report both lines. `tools/expected.txt` holds the four lines recorded on the Mac for
  its whole-suite recordings (hawk and eagle): they are a format and a scale, not a target, because this recording is GPU-only.
  A line that says `DOES NOT ADD UP`, a `different` above 0 or any `refused` is a finding: list the unit and test id.

(L) the live legs on CPU (the binding of (H) and the library built with `old-contract`)
    export ROOT=$RUN RECORDER_DIR=WORK/shadow PYTHON=$PY
    HOST_MODE=off    sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_off
    HOST_MODE=shadow sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_shadow
    HOST_MODE=route  sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_route
    HOST_MODE=off    sh WORK/osprey_host/tools/run_host_eagle.sh eagle_off 4
    HOST_MODE=route  sh WORK/osprey_host/tools/run_host_eagle.sh eagle_route 4
    $PY WORK/osprey_host/tools/merge_host.py $RUN/snap/hawk_shadow/host_counters      # the counters line; same for the others
    for n in hawk_off hawk_shadow hawk_route; do grep -E '^(FAILED|ERROR) ' $RUN/logs/$n.log | sed 's/ - .*//' | sort > $RUN/failed_$n.txt; done
    diff $RUN/failed_hawk_off.txt $RUN/failed_hawk_shadow.txt; diff $RUN/failed_hawk_off.txt $RUN/failed_hawk_route.txt
  Report each run's counters line and whether its failed ids equal the baseline's (eagle: `$RUN/snap/<run>/eagle_run/failed.txt`);
  a test that fails only with the plugin is a finding.

(G) THE GPU VALUE CHECK: the kernel-level route mode (`HOST_MODE=kroute`, `osprey_host/tools/kroute.py`)
  For every build call of a suite the plugin mirrors the kernel into a library context, sends ONE request with every served
  target of the call (host and cuda), asks `facts` per target, calls `lower` (abi 6), matches the answer's units to the old
  units by name and walk digest, and swaps the library's text in at the render seam (`hawk.backends.Backend.render`) for the
  host AND the device. Everything after the text is the old code (compile, publish, load, run), so the suite's own value
  assertions judge the library's text; on a machine with a GPU that includes the device text through the compiler and the run.
  A kernel is routed whole or falls back whole to the old text, counted by reason; an exception inside the plugin is a
  crossing error (must be 0; `KROUTE_RAISE=1` re-raises it).
    HOST_MODE=kroute sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_kroute          # CPU tests
    HOST_MODE=kroute HOST_MARK=gpu EMIT_JOBS=1 RUN_PREFIX="<your GPU wrapper>" sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_kroute_gpu
    HOST_MODE=kroute sh WORK/osprey_host/tools/run_host_eagle.sh eagle_kroute 4
  (`HOST_MARK` and `RUN_PREFIX` of run_host.sh were added for this leg and have NOT been run with a GPU wrapper anywhere)
  The counters are printed at the end and written to `<run folder>/kroute_counters.json`: kernels, routed whole, fallbacks by
  reason, texts taken from the library per target, identical / different to the old text, crossing errors.
  Recorded on an Arm Mac without a GPU (device texts were only taken at builds that emit without compiling):
    hawk:  1693 kernels, routed whole 1638, texts from the library host 1561 / cuda 200, identical 1746, different 15
           (13 on the accepted list + the two `fill_mask_seg__k0` device units), crossing errors 0, failed ids inside the base set;
           fallbacks 55: a test replaces the old renderer 20, no served target 19, banded mode 9, a build without a thread box 2,
           own card 1, unit mismatch 1, other refusals 3.
    eagle: 1041 kernels, routed whole 1034, host 1117 / cuda 5, identical 1119, different 3 (accepted list), crossing errors 0;
           fallbacks 7: unit mismatch 5 (the split tests of two and three stages), a patched renderer 1, no thread box 1.
  NEVER RUN ANYWHERE: device facts from a live machine record, a device text through the compiler and on a GPU, calls whose
  targets are device-only, the device compile-failure path. Report: the counters of each run, the failed ids against your
  baseline, every fallback reason, and for the GPU run the device texts compiled / run and each failing test with its unit.

## Fixes
- The hawk runners (`run_shadow.sh`, `run_k_gpu.sh`, `osprey_host/tools/run_host.sh`) pass `-o pythonpath=`: pytest's ini puts the
  source tree first on the path otherwise, and the recorder modules are shadowed.
- The eagle runners linked each test file into a flat folder, and tests that locate fixtures, headers or docs by a path relative to
  the test file left the tree (13 failures). They now build a folder with eagle's own layout (links to the checkout's folders, the
  test files linked under `python/tests`). This is a path of our script; it was not run on the Mac after the change, so the first
  eagle run is also its test: compare its failed list with the Mac's before reading anything into a red.
- The strict-C11 error (`strndup` undeclared) is fixed in `osprey_host/loader/loader.c`; the writers crate's clippy red is fixed in
  the aether tree pushed with this one. The C format leg is pinned to clang-format 23 (see (H)).

Notes: the runners take every path from the environment (see their headers). Run one cargo command at a time (`-j 2`).
