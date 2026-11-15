#!/bin/sh
# Runs a hawk tree's tests with the recorders (emit_record always; the render recorder when asked) and leaves the records
# in $ROOT/corpus/<name> (texts), $ROOT/snap/<name> (emit snapshot), the log in $ROOT/logs/<name>.log.
# usage: sh run_shadow.sh <hawk tree> <name> [pytest args...]      (default pytest args: tests)
# Environment: ROOT (the run folder), PYTHON (the interpreter of the suite), RECORDER_DIR (the folder with emit_record.py,
#   shadow_plugin.py, text_plugin.py), EMIT_OVERLAY (an overlay the suite needs, optional), EXTRA_PLUGINS (extra pytest
#   plugins, for example text_plugin), TEXT_MODE=record (what text_plugin needs; the records go to $ROOT/corpus/<name>),
#   MARK (the pytest marker expression, default "not gpu"; use MARK=gpu for the GPU-marked tests),
#   RUN_PREFIX (a command put before python, for example the machine's GPU wrapper), EMIT_JOBS (default 8),
#   CHUNK (a suffix for the log name when one run is split).
unset PYTEST_ADDOPTS       # a child pytest would inherit it
: "${ROOT:?}" "${PYTHON:?}" "${RECORDER_DIR:?}"
tree=$1; name=$2; shift 2
[ $# -eq 0 ] && set -- tests
jobs=${EMIT_JOBS:-8}
plug=""; for x in $EXTRA_PLUGINS; do plug="$plug -p $x"; done
mkdir -p "$ROOT/snap/$name" "$ROOT/logs" "$ROOT/corpus/$name"
echo "HEAD=$(git -C "$tree" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$tree" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" > "$ROOT/snap/$name/PROVENANCE"
log=$ROOT/logs/${name}${CHUNK:+_$CHUNK}.log
cd "$tree" || exit 2
HAWK_CACHE_DIR=$ROOT/cache/$name EMIT_RECORD_DIR=$ROOT/snap/$name SHADOW_MODE=${SHADOW_MODE-off} SHADOW_DIR=$ROOT/corpus/$name \
  TEXT_DIR=${TEXT_DIR-$ROOT/corpus/$name} PYTHONPATH=$RECORDER_DIR${EMIT_OVERLAY:+:$EMIT_OVERLAY} \
  ${RUN_PREFIX:-} "$PYTHON" -P -m pytest -q -m "${MARK:-not gpu}" -p no:cacheprovider \
  -p emit_record -p shadow_plugin $plug -n "$jobs" --dist "${EMIT_DIST:-worksteal}" "$@" > "$log" 2>&1
echo "RC=$?" >> "$log"
