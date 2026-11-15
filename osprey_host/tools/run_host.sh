#!/bin/sh
# usage: HOST_MODE=<off|shadow|route|kroute|cost> run_host.sh <tree> <snapshot name> [pytest args...]   (default: whole CPU suite)
# Runs the tree's tests with the host library plugin; snapshot lands in $ROOT/snap/<name>, log in $ROOT/logs/<name>.log,
# the plugin's counters in $ROOT/snap/<name>/host_counters (merge_host.py prints them).
# Environment, no defaults: ROOT (the run folder), PYTHON (the interpreter of the suite), RECORDER_DIR (the folder that holds the
# recorder modules shadow_plugin, text_plugin and shadow_mirror), EMIT_OVERLAY (the overlay the suite needs, optional).
# Optional: HOST_MARK (the pytest marker expression; default `not gpu`; `gpu` runs the GPU-marked tests), RUN_PREFIX (a command put
# before the interpreter: the machine's GPU wrapper), EMIT_JOBS (workers; use 1 with a GPU wrapper).
# The library and the binding are the ones built in this repository (cargo build --release --features old-contract, binding/build.sh): the plugin sends the old shape of the contract.
: "${ROOT:?}" "${PYTHON:?}" "${RECORDER_DIR:?}"
repo=$(cd "$(dirname "$0")/.." && pwd)
case $(uname -s) in Darwin) ext=dylib ;; *) ext=so ;; esac
tree=$1; name=$2; shift 2
[ $# -eq 0 ] && set -- tests
jobs=${EMIT_JOBS:-8}
mkdir -p $ROOT/snap/$name $ROOT/logs
echo "HEAD=$(git -C $tree rev-parse HEAD 2>/dev/null) dirty=$(git -C $tree status --porcelain 2>/dev/null | wc -l | tr -d ' ')" > $ROOT/snap/$name/PROVENANCE
cd $tree && HAWK_CACHE_DIR=$ROOT/cache/$name EMIT_RECORD_DIR=$ROOT/snap/$name HOST_MODE=${HOST_MODE-off} HOST_DIR=$ROOT/snap/$name/host_counters \
  HOST_LIB=$repo/target/release/libosprey_host.$ext PYTHONPATH=$repo/tools:$RECORDER_DIR:$repo/binding/build${EMIT_OVERLAY:+:$EMIT_OVERLAY} \
  ${RUN_PREFIX:-} $PYTHON -m pytest -q -m "${HOST_MARK:-not gpu}" -o pythonpath= -p no:cacheprovider \
  -p emit_record -p host_plugin -n $jobs --dist ${EMIT_DIST:-worksteal} "$@" > $ROOT/logs/${name}${CHUNK:+_$CHUNK}.log 2>&1
echo "RC=$?" >> $ROOT/logs/${name}${CHUNK:+_$CHUNK}.log
