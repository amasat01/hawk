#!/bin/sh
# Runs eagle's own Python suite with the render recorder: records go to $ROOT/corpus/<name>, the log and the failed test
# ids to $ROOT/snap/<name>/eagle_run/{run.log,failed.txt}.
# usage: sh run_eagle_text.sh <name> [jobs]
# Environment, all required: ROOT (the run folder), PYTHON (the suite's interpreter), RECORDER_DIR (folder with the recorder
#   modules shadow_plugin.py and text_plugin.py), EAGLE_SRC (eagle's source checkout), EAGLE_RAPTOR (raptor's source for the
#   suite), EAGLE_OVERLAY and EAGLE_TREE (the overlay and the hawk tree the suite imports).
# Optional: MARK (marker expression, default "not gpu"), RUN_PREFIX (a command put before python, for example a GPU wrapper).
set -eu
unset PYTEST_ADDOPTS       # a child pytest would inherit it
: "${ROOT:?}" "${PYTHON:?}" "${RECORDER_DIR:?}" "${EAGLE_SRC:?}" "${EAGLE_RAPTOR:?}" "${EAGLE_OVERLAY:?}" "${EAGLE_TREE:?}"
id=$1; jobs=${2:-4}
run=$ROOT/snap/$id/eagle_run
# The run folder mirrors eagle's own layout (eagle_run/python/tests next to links to the checkout's other folders): tests find their
# fixtures, headers and docs by paths relative to the test file, and a flat folder of links to the test files left the tree.
mkdir -p "$run/python/tests" "$run/cache" "$ROOT/corpus/$id"
find "$run" -maxdepth 1 -type l -exec rm {} +
find "$run/python" -maxdepth 2 -type l -exec rm {} +
for f in "$EAGLE_SRC"/*; do [ "$(basename "$f")" = python ] || ln -s "$f" "$run/"; done
for f in "$EAGLE_SRC"/python/*; do case $(basename "$f") in tests|conftest.py) ;; *) ln -s "$f" "$run/python/";; esac; done
for f in "$EAGLE_SRC"/python/tests/*; do ln -s "$f" "$run/python/tests/"; done
sed -e '/^sys\.path\.insert(0, str(ROOT))$/d' -e '/^sys\.path\.insert(0, RAPTOR_ROOT)$/d' \
    -e 's/^SFX = .*/SFX = ""/' "$EAGLE_SRC/python/conftest.py" > "$run/python/conftest.py"
cd "$run/python"
RAPTOR_ROOT=$EAGLE_RAPTOR PYTHONPATH=$RECORDER_DIR:$EAGLE_OVERLAY:$EAGLE_TREE PYTHONDONTWRITEBYTECODE=1 HAWK_CACHE_DIR=$run/cache \
  TEXT_MODE=record TEXT_DIR=$ROOT/corpus/$id SHADOW_MODE=off SHADOW_DIR=$ROOT/corpus/$id \
  nice -n 10 ${RUN_PREFIX:-} "$PYTHON" -P -m pytest -o pythonpath= -p no:cacheprovider -p shadow_plugin -p text_plugin -q -m "${MARK:-not gpu}" -n "$jobs" \
  --dist loadfile --tb=line --basetemp="$run/basetemp" tests > "$run/run.log" 2>&1 || true
cd "$run"
grep -E '^(FAILED|ERROR) ' run.log | sed -e 's/ - .*//' | sort > failed.txt
tail -1 run.log; wc -l < failed.txt
