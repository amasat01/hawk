#!/bin/sh
# Records, at kernel level, ONLY the GPU-marked tests of a hawk tree (the whole CPU suite at kernel level is far too large for a
# small disk). Per build call the kernel before lowering, the units made and every render record go to $ROOT/corpus/<name>
# (k<pid>.jsonl beside t<pid>.jsonl); the emit snapshot goes to $ROOT/snap/<name>, the log to $ROOT/logs/<name>.log.
# usage: sh run_k_gpu.sh <hawk tree> <name> [pytest args...]      (default pytest args: tests)
# Environment: ROOT (the run folder), PYTHON (the suite's interpreter), RECORDER_DIR (the folder with emit_record.py, shadow_plugin.py,
#   text_plugin.py, k6_plugin.py, k6_stats.py), RUN_PREFIX (the machine's GPU wrapper: a command put before python; required on a
#   machine where the GPU needs one), EMIT_JOBS (default 1: one GPU process), EMIT_OVERLAY (an overlay the suite needs, optional),
#   CHUNK (a suffix for the log name when one run is split).
# Afterwards `k6_stats.py` prints the counts of the corpus (calls, units, texts, joins).
unset PYTEST_ADDOPTS       # a child pytest would inherit it
: "${ROOT:?}" "${PYTHON:?}" "${RECORDER_DIR:?}"
tree=$1; name=$2; shift 2
[ $# -eq 0 ] && set -- tests
mkdir -p "$ROOT/snap/$name" "$ROOT/logs" "$ROOT/corpus/$name"
echo "HEAD=$(git -C "$tree" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$tree" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" > "$ROOT/snap/$name/PROVENANCE"
log=$ROOT/logs/${name}${CHUNK:+_$CHUNK}.log
cd "$tree" || exit 2
# -o pythonpath= empties the ini's pythonpath: pytest would otherwise put the source tree before the recorder modules
HAWK_CACHE_DIR=$ROOT/cache/$name EMIT_RECORD_DIR=$ROOT/snap/$name TEXT_MODE=record TEXT_DIR=$ROOT/corpus/$name \
  SHADOW_MODE=off SHADOW_DIR=$ROOT/corpus/$name PYTHONPATH=$RECORDER_DIR${EMIT_OVERLAY:+:$EMIT_OVERLAY} \
  ${RUN_PREFIX:-} "$PYTHON" -P -m pytest -q -m gpu -o pythonpath= -p no:cacheprovider \
  -p emit_record -p shadow_plugin -p text_plugin -p k6_plugin -n "${EMIT_JOBS:-1}" --dist worksteal "$@" > "$log" 2>&1
echo "RC=$?" >> "$log"
tail -3 "$log"
"$PYTHON" "$RECORDER_DIR/k6_stats.py" "$ROOT/corpus/$name" | tail -12
