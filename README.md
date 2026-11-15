# Tools for checking the compiled slice on another machine

This folder holds what is needed to build the compiled slice, record the render calls of two test suites on the
machine, replay those recordings through the compiled writers and run the suites against the compiled host library.
Nothing here needs the network.

## Contents
- `osprey_host/`: the host library (source, gate, binding, replay tool, live test plugin and its runners).
- `shadow/`: the pytest plugins that record render calls (`emit_record.py`, `shadow_plugin.py`, `text_plugin.py`) and the scripts
  that merge and rebuild a recording (`merge_text.py`, `rebuild_text.py`, `rebuild.py`).
- `tools/`: `run_shadow.sh` (record the hawk suite), `run_eagle_text.sh` (record the eagle suite), `proof.sh` (the portable
  proof over three source trees), `expected.txt` (the format of its expected numbers, with the numbers of the first machine).
- `MANIFEST.txt`: every file with its line count, and the files each leg uses.

## Prerequisites
Rust (cargo, rustfmt, clippy), a C compiler (`clang` for the binding), `make`, Python 3 with pytest and pytest-xdist, `nm`
(or `llvm-nm`), a checkout of the hawk tree and of the eagle tree with their environments, ruff and clang-format for the
host gate, `clang-tidy` (optional). Layout: one work folder `WORK` that holds this tree's folders next to the two source
trees: `WORK/raptor`, `WORK/aether`, `WORK/osprey_host`, `WORK/shadow`, `WORK/tools`. In the shell below:
`export RUN=<an empty run folder>` (records, logs and baselines go there) and `export PY=<python of the hawk environment>`.

## What a difference means
A unit that is `written but different` or that a writer refuses on this machine is a FINDING. Report it with the unit's name,
its test id and the first differing line pair. Do not fix anything on this machine. Units in `osprey_host/corpus_accepted.tsv`
are known text differences of the first machine; their keys differ here, so report differing units by NAME and test id.

## The run sheet
Report for every leg: the command's last 5 lines, and anything red with its first error line.

(B) raptor crates and gate
    cd WORK/raptor && cargo build --release -j 2 -p raptor_ir -p raptor_wings -p raptor_support
    NM=$(command -v llvm-nm || command -v nm) sh ci/gate_rs.sh
  The exports leg reads the release libraries with a symbol tool: `NM` names it (the first of `$NM`, `llvm-nm`, `nm` that
  lists symbols is used). Report the `GREEN:` and `RED:` lines.

(W) writers crate and gate
    cd WORK/aether && RAPTOR_IR_LIB=WORK/raptor/target/release/libraptor_ir.a sh ci/gate_rs.sh
  Report the `GREEN:` and `RED:` lines and the number of tests passed.

(H) host library, gate and binding
    cd WORK/osprey_host
    cargo build --release -j 2 --features old-contract
    PYTHON=$PY CLANG_TIDY=<clang-tidy or leave unset> make gate
  The feature `old-contract` lets the test builds read old-shape recordings (the replay tool and the live plugin need it).
  `make gate` builds the binding (`binding/build.sh`, needs `PYTHON` and `clang`) and leaves it in `binding/build`.
  Report the leg list. The Linux paths of `binding/build.sh`, `tools/cc` and `check_exports.sh` were written without a Linux
  machine at hand: a failure there is a finding about those scripts, report the error line.

(R) record the text corpus
    export ROOT=$RUN RECORDER_DIR=WORK/shadow PYTHON=$PY
    # hawk suite, CPU tests, then the GPU-marked tests into the same record folder through the machine's GPU wrapper
    EXTRA_PLUGINS=text_plugin TEXT_MODE=record sh WORK/tools/run_shadow.sh <hawk tree> hawk_text
    EXTRA_PLUGINS=text_plugin TEXT_MODE=record MARK=gpu RUN_PREFIX="<gpu wrapper>" TEXT_DIR=$RUN/corpus/hawk_text \
        sh WORK/tools/run_shadow.sh <hawk tree> hawk_text_gpu
    # eagle suite (EAGLE_SRC, EAGLE_RAPTOR, EAGLE_OVERLAY and EAGLE_TREE as in the header of the script)
    sh WORK/tools/run_eagle_text.sh eagle_text 4
    # merge each record folder
    cd WORK/shadow && PYTHONPATH=. $PY merge_text.py $RUN/corpus/hawk_text && PYTHONPATH=. $PY merge_text.py $RUN/corpus/eagle_text
  Each merge writes `texts.jsonl` and `SUMMARY.json` in the record folder. Report the merge's printed totals, the tests that
  failed (`$RUN/logs/*.log`, `$RUN/snap/eagle_text/eagle_run/failed.txt`) and the number of units per target.

(P) replay each corpus through the compiled writers
    cd WORK/osprey_host
    for c in hawk_text eagle_text; do for t in host cuda; do
      ./target/release/treplay $RUN/corpus/$c/texts.jsonl --target $t --accepted corpus_accepted.tsv > $RUN/replay_${c}_$t.txt 2>&1
    done; done
  Per corpus report: `units`, `text identical (whole path)` (host) or `text identical (from the recorded body)` (cuda),
  `written but different`, every refusal class with its count (the `body by feature` and `text by feature` lines), the two
  lines starting `typed contract:` and `typed wire:`, and each differing unit by name and test id.
  To compare against a baseline automatically, write `$RUN/corpus/expected.txt` in the format of `tools/expected.txt`
  after the first clean run, then: `RAPTOR=WORK/raptor AETHER=WORK/aether OSPREY=WORK/osprey_host CORPUS=$RUN/corpus SCRATCH=$RUN/scratch sh WORK/tools/proof.sh quick`
  (it writes a baseline dump of every text when none exists and says so; the next run compares byte for byte).

(L) the live legs on CPU (the binding of (H) and the library built with `old-contract`)
    export ROOT=$RUN RECORDER_DIR=WORK/shadow PYTHON=$PY
    # the machine's own baseline first, once: the plugin is loaded but changes nothing
    HOST_MODE=off    sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_off
    HOST_MODE=shadow sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_shadow
    HOST_MODE=route  sh WORK/osprey_host/tools/run_host.sh <hawk tree> hawk_route
    # eagle (the variables of the header of run_host_eagle.sh are required)
    HOST_MODE=off    sh WORK/osprey_host/tools/run_host_eagle.sh eagle_off 4
    HOST_MODE=route  sh WORK/osprey_host/tools/run_host_eagle.sh eagle_route 4
    $PY WORK/osprey_host/tools/merge_host.py $RUN/snap/hawk_shadow/host_counters      # the counters line; same for the others
    for n in hawk_off hawk_shadow hawk_route; do grep -E '^(FAILED|ERROR) ' $RUN/logs/$n.log | sed 's/ - .*//' | sort > $RUN/failed_$n.txt; done
    diff $RUN/failed_hawk_off.txt $RUN/failed_hawk_shadow.txt; diff $RUN/failed_hawk_off.txt $RUN/failed_hawk_route.txt
  Report the counters line of each run and whether its failed test ids equal the baseline's (for eagle compare
  `$RUN/snap/eagle_route/eagle_run/failed.txt` with `.../eagle_off/eagle_run/failed.txt`). Any test that fails only with the
  plugin is a finding: report its id and the first error line of its log.

## Notes
- The runners take every path from the environment; their headers list the variables.
- The library file is `libosprey_host.dylib` on macOS and `libosprey_host.so` elsewhere; the runners choose it by platform.
- Run one cargo command at a time on a small machine (`-j 2`).
