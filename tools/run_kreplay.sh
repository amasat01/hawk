#!/bin/sh
# Replays a kernel-level corpus through the compiled lowering (the table entry `lower`) and prints the replay's reconciliation
# lines: one line per target, where every outcome is counted once and the line says whether the categories add up.
# usage: sh run_kreplay.sh <corpus folder>       (for example $ROOT/corpus/hawk_gpu_k)
# Environment: OSPREY (the host library tree; required), KREPLAY_OUT (where the full output goes; default <corpus folder>/kreplay.txt),
#   C3_JOBS (cargo jobs, default 2), NO_BUILD=1 (use the binary as it is).
: "${OSPREY:?}"
corpus=$1
out=${KREPLAY_OUT:-$corpus/kreplay.txt}
cd "$OSPREY" || exit 2
if [ -z "${NO_BUILD:-}" ]; then
  cargo build --release -j "${C3_JOBS:-2}" --features old-contract --bin kreplay > "$out.build" 2>&1 \
    || { grep -E "^error" -A 6 "$out.build" | head -12; echo "NOT OK: the build of kreplay failed"; exit 1; }
  rm -f "$out.build"
fi
./target/release/kreplay "$corpus" --accepted corpus_accepted.tsv > "$out" 2>&1
grep -E "^(host|cuda) units|reconciliation:" "$out" | sed 's/^ *//'
echo "full output: $out"
