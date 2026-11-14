#!/bin/sh
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# usage: run_shadow.sh <hawk tree> <run name> [pytest arguments...]      (default arguments: tests)
# Runs the tree's tests with the shadow plugin in mode SHADOW_MODE (record or shadow; default record).
# Output: $OUT/corpus/<name> (the plugin's files), $OUT/logs/<name>.log, $OUT/cache/<name> (the tree's cache).
# Environment (all optional):
#   OUT              output folder (default: ./out)
#   PYTHON           interpreter (default: python3)
#   EMIT_OVERLAY     path entries added to PYTHONPATH (colon separated), e.g. a folder that holds raptor and its wings
#   PYTEST_MARK      marker expression (default: not gpu; set it to the empty string to run every test; no single quotes)
#   JOBS             pytest-xdist workers (default: 8; 0 runs in one process); DIST is the xdist scheduler (default: worksteal)
#   EMIT_RECORD_DIR  with EMIT_RECORDER_DIR (the folder holding the emit_record plugin): also record emitted sources there
here=$(cd "$(dirname "$0")" && pwd)
shadow=$(cd "$here/../shadow" && pwd)
tree=$(cd "$1" && pwd); name=$2; shift 2
[ $# -eq 0 ] && set -- tests
mkdir -p "${OUT:-out}"
out=$(cd "${OUT:-out}" && pwd)
mkdir -p "$out/corpus/$name" "$out/logs" "$out/cache/$name"
echo "HEAD=$(git -C "$tree" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$tree" status --porcelain 2>/dev/null | wc -l | tr -d ' ')" > "$out/corpus/$name/PROVENANCE"
addopts="-q -p no:cacheprovider -p shadow_plugin"
path=$shadow
if [ -n "$EMIT_RECORD_DIR" ] && [ -n "$EMIT_RECORDER_DIR" ]; then
  addopts="$addopts -p emit_record"; path=$path:$EMIT_RECORDER_DIR; export EMIT_RECORD_DIR
fi
mark=${PYTEST_MARK-not gpu}
[ -n "$mark" ] && addopts="$addopts -m '$mark'"
jobs=${JOBS:-8}
[ "$jobs" != 0 ] && addopts="$addopts -n $jobs --dist ${DIST:-worksteal}"
export PYTEST_ADDOPTS="$PYTEST_ADDOPTS $addopts"
export HAWK_CACHE_DIR="$out/cache/$name" SHADOW_MODE=${SHADOW_MODE:-record} SHADOW_DIR="$out/corpus/$name"
export PYTHONPATH="$path${EMIT_OVERLAY:+:$EMIT_OVERLAY}${PYTHONPATH:+:$PYTHONPATH}"
cd "$tree" && "${PYTHON:-python3}" -m pytest "$@" > "$out/logs/$name.log" 2>&1
echo "RC=$?" >> "$out/logs/$name.log"
