#!/bin/sh
# The whole proof of the compiled slice on three source trees: builds, tests, replays the recorded corpora through the
# writers (host and device), compares every written text with a baseline dump, prints one line per check and ends with
# ALL OK, or NOT OK and what moved.   Usage: sh proof.sh [quick|full]      (quick: no full-only checks)
#
# Environment (nothing is assumed about where anything lives):
#   RAPTOR, AETHER, OSPREY   the three source trees (folders)
#   CORPUS                   the folder of recordings: <CORPUS>/<recording>/texts.jsonl
#   EXPECTED                 the expected numbers (default <CORPUS>/expected.txt), one tab-separated line per check:
#                              host   <recording>        <the replay's host summary, for example 1696 / 1709 (written but different 0) + 13>
#                              cuda   <recording>        <the device summary, for example 455 / 462>
#                              typed  <recording>        <number of units: all must convert and round-trip>
#                              full-host <recording>     <host summary>                            (full only)
#                              wreplay  <recording> <host|cuda>   <summary>                         (full only)
#                              replay   <recording>      <summary>                                  (full only)
#                            (a recording folder named <x>_record is dumped under <x>)
#   BASE                     the baseline dump folder (default <SCRATCH>/dump_base). When it does not exist the dump of
#                            this run is written there and the comparison is skipped, with a line saying so.
#   SCRATCH                  a writable folder for the two-slot lock (folders <SLOT_PREFIX>1 and 2, default proof_slot_) and temporary files
#   ACCEPTED                 the accepted-differences list (default <OSPREY>/corpus_accepted.tsv)
#   C3_JOBS (default 2)      cargo jobs        C3_FEATURES   cargo features of the host library (default: old-contract when the
#                                              library has it)
: "${RAPTOR:?}" "${AETHER:?}" "${OSPREY:?}" "${CORPUS:?}" "${SCRATCH:?}"
mode=${1:-full}; bad=""
EXPECTED=${EXPECTED:-$CORPUS/expected.txt}
ACCEPTED=${ACCEPTED:-$OSPREY/corpus_accepted.tsv}
BASE=${BASE:-$SCRATCH/dump_base}
jobs=${C3_JOBS:-2}
[ -f "$EXPECTED" ] || { echo "no expected numbers: $EXPECTED"; exit 2; }
mkdir -p "$SCRATCH"
if [ -z "${C3_FEATURES+x}" ] && grep -q "old-contract" "$OSPREY/Cargo.toml" 2>/dev/null; then C3_FEATURES=old-contract; fi
wf=""; grep -q "old-contract" "$AETHER/rs/writers/Cargo.toml" 2>/dev/null && wf="--features old-contract"
note() { echo "$1"; }
fail() { bad="$bad [$1]"; echo "MOVED: $1"; }
# the expected value of a check: the third column (or the fourth for a check with two names)
want_of() { awk -F'\t' -v k="$1" -v a="$2" '$1 == k && $2 == a { print $NF; exit }' "$EXPECTED"; }
chk() { got=$(echo "$2" | sed 's/ *$//'); [ "$got" = "$3" ] && note "$1: $got" || fail "$1: got '$got' want '$3'"; }

( cd "$RAPTOR" && cargo build --release -j "$jobs" -p raptor_ir -p raptor_wings -p raptor_support 2>&1 | grep -E "^error" -A 6 | head -12 )
( cd "$OSPREY" && cargo build --release -j "$jobs" ${C3_FEATURES:+--features "$C3_FEATURES"} 2>&1 | grep -E "^error" -A 6 | head -12 )
for r in "$RAPTOR" "$OSPREY"; do (cd "$r" && cargo fmt --all --check >/dev/null 2>&1) || fail "fmt $r"; done
(cd "$AETHER" && cargo fmt --manifest-path rs/writers/Cargo.toml --check >/dev/null 2>&1) || fail "fmt aether"
t=$(cd "$RAPTOR" && cargo test --release -j "$jobs" -p raptor_ir -p raptor_wings -p raptor_support 2>&1 | grep -cE "FAILED|panicked|^error|signal:")
[ "$t" = 0 ] || fail "raptor tests"
t=$(cd "$OSPREY" && cargo test --release -j "$jobs" ${C3_FEATURES:+--features "$C3_FEATURES"} 2>&1 | grep -cE "FAILED|panicked|^error|signal:")
[ "$t" = 0 ] || fail "osprey_host tests"
t=$(cd "$AETHER/rs/writers" && RAPTOR_IR_LIB=$RAPTOR/target/release/libraptor_ir.a cargo test --release -j "$jobs" $wf 2>&1 | grep -E "FAILED|panicked|^error|signal:|test result" | tr '\n' ' ')
case "$t" in *FAILED*|*panicked*|*error*|*signal*) fail "writers tests";; esac
note "writers tests: $(echo "$t" | grep -oE '[0-9]+ passed' | tr '\n' ' ')"
# the library exports one symbol; macOS spells it with a leading underscore
sym=$(cd "$OSPREY" && sh check_exports.sh 2>/dev/null | tail -1)
case "$sym" in _osprey_v1|osprey_v1) ;; *) fail "exports: '$sym'";; esac

# At most two proofs replay at once on a machine (one replay holds a lot of memory): wait for a slot.
slot=""
while [ -z "$slot" ]; do
  for n in 1 2; do
    if mkdir "$SCRATCH/${SLOT_PREFIX:-proof_slot_}$n" 2>/dev/null; then slot=$SCRATCH/${SLOT_PREFIX:-proof_slot_}$n; echo $$ > "$slot/pid"; break; fi
    owner=$(cat "$SCRATCH/${SLOT_PREFIX:-proof_slot_}$n/pid" 2>/dev/null)
    if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then rm -rf "$SCRATCH/${SLOT_PREFIX:-proof_slot_}$n"; fi
  done
  [ -z "$slot" ] && sleep 20
done
trap 'rm -rf "$slot"' EXIT INT TERM

cd "$OSPREY" || exit 2
tr_host() { ./target/release/treplay "$CORPUS/$1/texts.jsonl" --target host --accepted "$ACCEPTED" ${2:+--dump "$2"} 2>&1 | grep -v MISMATCH > "$SCRATCH/out_$$_$1.txt"
  grep -E 'whole path|accepted, text|stale' "$SCRATCH/out_$$_$1.txt" | sed -E 's/text identical \(whole path\) //; s/accepted, text differs \(value-proven list\)/+/' | tr -s ' \n' ' '; }
tr_cuda() { ./target/release/treplay "$CORPUS/$1/texts.jsonl" --target cuda --accepted "$ACCEPTED" ${2:+--dump "$2"} 2>&1 | grep -v MISMATCH \
  | grep -E 'from the recorded body' | sed -E 's/text identical \(from the recorded body\) //' | tr -s ' \n' ' '; }
dump=$SCRATCH/dump_$$; rm -rf "$dump"; mkdir -p "$dump"
for rec in $(awk -F'\t' '$1 == "host" { print $2 }' "$EXPECTED"); do
  chk "$rec host" "$(tr_host "$rec" "$dump/${rec%_record}")" "$(want_of host "$rec")"
  n=$(want_of typed "$rec")
  if [ -n "$n" ]; then
    got=$(grep -E '^typed (contract|wire):' "$SCRATCH/out_$$_$rec.txt" | tr '\n' ' ' | sed 's/ *$//')
    chk "$rec typed contract" "$got" "typed contract: $n of $n units converted, 0 refused typed wire: $n of $n units round-trip"
  fi
  rm -f "$SCRATCH/out_$$_$rec.txt"
done
for rec in $(awk -F'\t' '$1 == "cuda" { print $2 }' "$EXPECTED"); do
  chk "$rec cuda" "$(tr_cuda "$rec" "$dump/${rec%_record}")" "$(want_of cuda "$rec")"
done
if [ -d "$BASE" ]; then
  d=$(diff -rq "$BASE" "$dump" | wc -l | tr -d ' ')
  [ "$d" = 0 ] || diff -rq "$BASE" "$dump" | head -5
  [ "$d" = 0 ] && note "dump: identical to $BASE ($(find "$dump" -type f | wc -l | tr -d ' ') files)" || fail "dump: $d files differ from $BASE"
else
  mkdir -p "$(dirname "$BASE")" && cp -R "$dump" "$BASE"
  note "dump: no baseline existed; this run's $(find "$dump" -type f | wc -l | tr -d ' ') files are written to $BASE (nothing compared)"
fi
if [ "$mode" = full ]; then
  for rec in $(awk -F'\t' '$1 == "full-host" { print $2 }' "$EXPECTED"); do
    chk "$rec host" "$(tr_host "$rec")" "$(want_of full-host "$rec")"; rm -f "$SCRATCH/out_$$_$rec.txt"
  done
  cd "$RAPTOR" || exit 2
  awk -F'\t' '$1 == "wreplay" { print $2 }' "$EXPECTED" | while read -r rec tg; do
    for m in "" --own-kinds --own-kinds-merge; do
      f=$SCRATCH/wr_$$.txt; ./target/release/wreplay "$CORPUS/$rec/texts.jsonl" --target "$tg" $m > "$f" 2>&1
      got="$(grep -c '^analysis-equal' "$f") functions, $(grep '^analysis-equal' "$f" | awk '$2!=$4' | wc -l | tr -d ' ') unequal, $(grep '^skipped' "$f" | awk '{s+=$2} END{print s+0}') skipped"
      want=$(awk -F'\t' -v a="$rec $tg" '$1 == "wreplay" && $2 == a { print $3; exit }' "$EXPECTED")
      [ "$got" = "$want" ] && echo "wreplay $tg ${m:-plain}: $got" || echo "MOVED: wreplay $tg ${m:-plain}: got '$got' want '$want'"
    done
  done > "$SCRATCH/wr_lines_$$.txt"
  cat "$SCRATCH/wr_lines_$$.txt"
  grep -q '^MOVED:' "$SCRATCH/wr_lines_$$.txt" && bad="$bad [wreplay]"
  for rec in $(awk -F'\t' '$1 == "replay" { print $2 }' "$EXPECTED"); do
    got=$(./target/release/replay "$CORPUS/$rec/walks.jsonl" --roundtrip --loops "$CORPUS/$rec/loops_oracle.jsonl" 2>&1 | grep -E "roundtrip-equal|refusal-equal|loop-rebuild" | sed -E 's/  \(compared.*//' | tr '\n' ' ')
    chk "replay $rec" "$got" "$(want_of replay "$rec")"
  done
fi
rm -rf "$dump" "$SCRATCH/wr_$$.txt" "$SCRATCH/wr_lines_$$.txt"
for r in "$RAPTOR" "$AETHER" "$OSPREY"; do printf "%s %s  " "$(basename "$r")" "$(git -C "$r" rev-parse --short HEAD)"; done; echo
[ -z "$bad" ] && echo "ALL OK ($mode)" || echo "NOT OK:$bad"
