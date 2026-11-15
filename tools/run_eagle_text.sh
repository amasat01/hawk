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
mkdir -p "$run/tests" "$run/cache" "$ROOT/corpus/$id"
find "$run/tests" -maxdepth 1 -type l -exec rm {} +
for f in "$EAGLE_SRC"/python/tests/*; do ln -s "$f" "$run/tests/"; done
sed -e '/^sys\.path\.insert(0, str(ROOT))$/d' -e '/^sys\.path\.insert(0, RAPTOR_ROOT)$/d' \
    -e 's/^SFX = .*/SFX = ""/' "$EAGLE_SRC/python/conftest.py" > "$run/conftest.py"
cd "$run"
RAPTOR_ROOT=$EAGLE_RAPTOR PYTHONPATH=$RECORDER_DIR:$EAGLE_OVERLAY:$EAGLE_TREE PYTHONDONTWRITEBYTECODE=1 HAWK_CACHE_DIR=$run/cache \
  TEXT_MODE=record TEXT_DIR=$ROOT/corpus/$id SHADOW_MODE=off SHADOW_DIR=$ROOT/corpus/$id \
  nice -n 10 ${RUN_PREFIX:-} "$PYTHON" -P -m pytest -p no:cacheprovider -p shadow_plugin -p text_plugin -q -m "${MARK:-not gpu}" -n "$jobs" \
  --dist loadfile --tb=line --basetemp="$run/basetemp" tests > run.log 2>&1 || true
grep -E '^(FAILED|ERROR) ' run.log | sed -e 's/ - .*//' | sort > failed.txt
tail -1 run.log; wc -l < failed.txt
