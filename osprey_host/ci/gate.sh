#!/bin/sh
# The gate of osprey_host. Usage: sh ci/gate.sh [leg ...]   (no argument: every leg, in this order)
#   fmt clippy test exports binding abi ruff cformat tidy replay
# Environment: PYTHON (the interpreter the binding is built for; the binding leg is red without it), CLANG_TIDY (the clang-tidy
# program; the tidy leg is RED without it). The siblings ../raptor and ../aether are inputs.
# Each leg prints what it examined and is RED when it examined nothing. The exit status is non-zero when any leg is red.
cd "$(dirname "$0")/.." || exit 2
JOBS=${JOBS:-2}
out=$(mktemp "${TMPDIR:-/tmp}/gate.XXXXXX")
trap 'rm -f "$out" "$out".*' EXIT
CFILES=$(git ls-files loader include binding/template | grep -E '\.(c|h)$')

need() { [ "$2" -gt 0 ] || { echo "RED: $1 examined nothing"; return 1; }; }
passed() { awk '/^test result:/ { p += $4 } END { print p + 0 }' "$out"; }

leg_fmt() {
  cargo fmt --all --check >"$out" 2>&1 || { head -5 "$out"; echo "RED: fmt differs"; return 1; }
  n=$(git ls-files | grep -c '\.rs$'); echo "examined: $n rust files"; need fmt "$n"
}

leg_clippy() {
  cargo clippy --release -j "$JOBS" --all-targets -- -D warnings >"$out" 2>&1
  rc=$?; grep -E "^(warning|error)" "$out" | sort | uniq -c | sort -rn | head -8
  n=$(git ls-files | grep -c '\.rs$'); echo "examined: clippy over $n rust files, warnings denied"
  [ $rc -eq 0 ] || { echo "RED: clippy"; return 1; }; need clippy "$n"
}

leg_test() {
  cargo test --release -j "$JOBS" >"$out" 2>&1
  rc=$?; n=$(passed); grep -E "FAILED|panicked|^error" "$out" | head -5
  echo "examined: $n tests passed"
  [ $rc -eq 0 ] || { echo "RED: cargo test"; return 1; }; need test "$n"
}

# the built library exports exactly one symbol (check_exports.sh prints it)
leg_exports() {
  sh check_exports.sh >"$out" 2>"$out.err" || { tail -3 "$out" "$out.err"; echo "RED: check_exports.sh"; return 1; }
  n=$(grep -c . "$out"); echo "examined: $n exported symbol: $(head -1 "$out")"
  [ "$n" -eq 1 ] || { echo "RED: expected exactly one symbol line, got $n"; return 1; }
}

# the binding's build and its gate; it must print GATE GREEN
leg_binding() {
  [ -n "$PYTHON" ] || { echo "RED: PYTHON is not set"; return 1; }
  sh binding/build.sh >"$out" 2>&1; rc=$?
  tail -4 "$out"
  g=$(grep -c "GATE GREEN" "$out"); echo "examined: binding build, GATE GREEN lines: $g"
  [ $rc -eq 0 ] || { echo "RED: binding build"; return 1; }; need binding "$g"
}

# the abi number has one source: the header's define. The loader and the table test read it from there.
leg_abi() {
  defs=$(git ls-files include loader src binding tests | xargs grep -E '^[ \t]*#[ \t]*define[ \t]+OB_ABI[ \t]' 2>/dev/null | wc -l | tr -d ' ')
  echo "examined: $defs definition(s) of OB_ABI (include/osprey_host.h is the one source)"
  [ "$defs" -eq 1 ] || { echo "RED: OB_ABI is defined $defs times"; return 1; }
  cargo test --release -j "$JOBS" --test table >"$out" 2>&1; rc=$?; n=$(passed)
  echo "examined: the table test, $n passed (header define equals the library's entry table abi)"
  [ $rc -eq 0 ] || { grep -E "FAILED|panicked|^error" "$out" | head -3; echo "RED: table test"; return 1; }; need abi "$n"
}

leg_ruff() {
  command -v ruff >/dev/null 2>&1 || { echo "RED: ruff is not on PATH"; return 1; }
  n=$(git ls-files | grep -c '\.py$'); echo "examined: $n python files"; need ruff "$n" || return 1
  ruff check >"$out" 2>&1 || { tail -5 "$out"; echo "RED: ruff check"; return 1; }
  ruff format --check >"$out" 2>&1 || { tail -5 "$out"; echo "RED: ruff format"; return 1; }
}

leg_cformat() {
  command -v clang-format >/dev/null 2>&1 || { echo "RED: clang-format is not on PATH"; return 1; }
  n=$(echo "$CFILES" | grep -c .); echo "examined: $n C files in loader/ include/ binding/template/"; need cformat "$n" || return 1
  # shellcheck disable=SC2086
  clang-format --dry-run -Werror $CFILES >"$out" 2>&1 || { grep -E "error" "$out" | head -5; echo "RED: clang-format differs"; return 1; }
}

leg_tidy() {
  [ -n "$CLANG_TIDY" ] || { echo "RED: CLANG_TIDY is not set (examined: 0 files)"; return 1; }
  [ -n "$PYTHON" ] || { echo "RED: PYTHON is not set (the binding template includes Python.h)"; return 1; }
  pyinc=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
  tfiles=$(git ls-files loader binding/template | grep '\.c$')
  n=$(echo "$tfiles" | grep -c .); echo "examined: $n C files of loader/ and binding/template/ with .clang-tidy"; need tidy "$n" || return 1
  # shellcheck disable=SC2086
  "$CLANG_TIDY" $tfiles -- -std=c11 -Iinclude -Ibinding/template -I"$pyinc" -I../raptor/rs/ir/include >"$out" 2>&1 \
    || { grep -E "error|warning" "$out" | head -5; echo "RED: clang-tidy"; return 1; }
}

# the recorded small file replays through the writers end to end: every unit is written and none differs from its recording
leg_replay() {
  cargo run --release -j "$JOBS" --features old-contract --bin treplay -- tests/data/units_small.jsonl --target host >"$out" 2>&1 \
    || { tail -5 "$out"; echo "RED: treplay"; return 1; }
  line=$(grep -E '^text identical \(whole path\)' "$out" | head -1)
  echo "examined: $line"
  n=$(echo "$line" | sed -E 's/^[^0-9]*([0-9]+) \/ ([0-9]+).*written but different ([0-9]+).*$/\1 \2 \3/')
  set -- $n
  [ "$#" -eq 3 ] && [ "$1" -gt 0 ] && [ "$1" -eq "$2" ] && [ "$3" -eq 0 ] \
    && grep -q '^body written but different 0$' "$out" && grep -q '^render errors 0$' "$out" \
    || { grep -E 'different|errors|refused' "$out" | head -5; echo "RED: a unit was not written or differs from its recording"; return 1; }
}

legs=${*:-fmt clippy test exports binding abi ruff cformat tidy replay}
green=; red=
for l in $legs; do
  echo "== $l"
  if "leg_$l"; then green="$green $l"; else red="$red $l"; fi
done
echo "GREEN:$green"; echo "RED:$red"
[ -z "$red" ]
