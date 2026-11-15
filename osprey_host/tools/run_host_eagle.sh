#!/bin/sh
# usage: HOST_MODE=<off|shadow|route> run_host_eagle.sh <snapshot name> [jobs] [pytest args...]   (default: eagle's whole Python suite)
# Runs eagle's own Python suite with the host library plugin; log and failed ids in $ROOT/snap/<name>/eagle_run, counters in
# $ROOT/snap/<name>/host_counters (merge_host.py prints them).
# Environment, no defaults: ROOT (the run folder), PYTHON (the suite's interpreter), RECORDER_DIR (folder with the recorder modules
# shadow_plugin, text_plugin, shadow_mirror), EAGLE_SRC (eagle's source checkout), EAGLE_RAPTOR (raptor's source), EAGLE_OVERLAY and
# EAGLE_TREE (the overlay and the old hawk tree the suite imports).
set -eu
unset PYTEST_ADDOPTS
: "${ROOT:?}" "${PYTHON:?}" "${RECORDER_DIR:?}" "${EAGLE_SRC:?}" "${EAGLE_RAPTOR:?}" "${EAGLE_OVERLAY:?}" "${EAGLE_TREE:?}"
repo=$(cd "$(dirname "$0")/.." && pwd)
case $(uname -s) in Darwin) ext=dylib ;; *) ext=so ;; esac
id=$1; jobs=${2:-4}; shift; shift || true
[ $# -eq 0 ] && set -- tests
run=$ROOT/snap/$id/eagle_run
mkdir -p "$run/tests" "$run/cache"
find "$run/tests" -maxdepth 1 -type l -exec rm {} +
for f in "$EAGLE_SRC"/python/tests/*; do ln -s "$f" "$run/tests/"; done
sed -e '/^sys\.path\.insert(0, str(ROOT))$/d' -e '/^sys\.path\.insert(0, RAPTOR_ROOT)$/d' \
    -e 's/^SFX = .*/SFX = ""/' "$EAGLE_SRC/python/conftest.py" > "$run/conftest.py"
cd "$run"
RAPTOR_ROOT=$EAGLE_RAPTOR PYTHONPATH=$repo/tools:$RECORDER_DIR:$repo/binding/build:$EAGLE_OVERLAY:$EAGLE_TREE PYTHONDONTWRITEBYTECODE=1 \
  HAWK_CACHE_DIR=$run/cache HOST_MODE=${HOST_MODE-off} HOST_DIR=$ROOT/snap/$id/host_counters \
  HOST_LIB=$repo/target/release/libosprey_host.$ext \
  nice -n 10 "$PYTHON" -P -m pytest -p no:cacheprovider -p host_plugin -q -m "not gpu" -n "$jobs" --dist loadfile \
  --tb=line --basetemp="$run/basetemp" "$@" > run.log 2>&1 || true
grep -E '^(FAILED|ERROR) ' run.log | sed -e 's/ - .*//' | sort > failed.txt
tail -1 run.log; wc -l < failed.txt
