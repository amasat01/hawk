#!/bin/sh
# usage: HOST_MODE=<off|shadow|route|kroute> run_host_eagle.sh <snapshot name> [jobs] [pytest args...]   (default: eagle's whole Python suite)
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
# The run folder mirrors eagle's own layout (eagle_run/python/tests next to links to the checkout's other folders): tests find their
# fixtures, headers and docs by paths relative to the test file, and a flat folder of links to the test files left the tree.
# The conftest gets the overlay first on the path: pytest puts python/ (which holds the linked `eagle` source package, read by path
# by some tests) first, and the overlay's `eagle` (with its compiled part) must be the one imported; it also leaves python/ as the working
# directory once collection is done (the conftest is linked there too, for children that import it), so a child interpreter does not find the source package first either.
mkdir -p "$run/python/tests" "$run/cache"
find "$run" -maxdepth 1 -type l -exec rm {} +
find "$run/python" -maxdepth 2 -type l -exec rm {} +
for f in "$EAGLE_SRC"/*; do [ "$(basename "$f")" = python ] || ln -s "$f" "$run/"; done
for f in "$EAGLE_SRC"/python/*; do case $(basename "$f") in tests|conftest.py) ;; *) ln -s "$f" "$run/python/";; esac; done
for f in "$EAGLE_SRC"/python/tests/*; do ln -s "$f" "$run/python/tests/"; done
{ echo 'import os, sys'; echo 'sys.path[:0] = [p for p in os.environ.get("EAGLE_OVERLAY", "").split(":") if p]'; echo 'def pytest_collection_finish(session):'; echo '    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'; sed -e '/^sys\.path\.insert(0, str(ROOT))$/d' -e '/^sys\.path\.insert(0, RAPTOR_ROOT)$/d' \
    -e 's/^SFX = .*/SFX = ""/' "$EAGLE_SRC/python/conftest.py"; } > "$run/python/conftest.py"
ln -sf python/conftest.py "$run/conftest.py"
cd "$run/python"
RAPTOR_ROOT=$EAGLE_RAPTOR PYTHONPATH=$repo/tools:$RECORDER_DIR:$repo/binding/build:$EAGLE_OVERLAY:$EAGLE_TREE PYTHONDONTWRITEBYTECODE=1 \
  HAWK_CACHE_DIR=$run/cache HOST_MODE=${HOST_MODE-off} HOST_DIR=$ROOT/snap/$id/host_counters \
  HOST_LIB=$repo/target/release/libosprey_host.$ext \
  nice -n 10 "$PYTHON" -P -m pytest -o pythonpath= -p no:cacheprovider -p host_plugin -q -m "not gpu" -n "$jobs" --dist loadfile \
  --tb=line --basetemp="$run/basetemp" "$@" > "$run/run.log" 2>&1 || true
cd "$run"
grep -E '^(FAILED|ERROR) ' run.log | sed -e 's/ - .*//' | sort > failed.txt
tail -1 run.log; wc -l < failed.txt
