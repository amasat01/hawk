#!/bin/sh
# The gate of the OSPREY libraries (components/lowering, osprey_cpu, osprey_cuda) and the tools that prove them. Usage: sh ci/gate.sh [leg ...]   (no argument: every leg, in this order)
#   fmt clippy test exports binding py39 abi wheels ruff cformat tidy replay
# Environment: PYTHON (the interpreter the binding is built for; the binding leg is red without it), OSPREY_EAGLE_SRC (the eagle source root; every leg that builds is red without it), CXX (the C++ compiler of the cpu library's runtime), CLANG_TIDY (the clang-tidy
# program; the tidy leg is RED without it), the cformat leg skips, saying so, unless clang-format is major version 23, PYTHON39 and PY39_INCLUDE (a Python 3.9 interpreter and its headers; the py39 leg
# skips, saying so, when the interpreter is absent; the default is the macOS command line tools' one). The siblings ../raptor and ../aether are inputs (RAPTOR names the first);
# the binding legs build the contract's static libraries in it (raptor_ir, raptor_wings, raptor_support; a no-op when current), so they need no prior build.
# Eagle source root (OSPREY_EAGLE_SRC), the one input of the runtime: a checkout of eagle that holds eagle/door/ (the door's header, its source
# list sources.txt and its sources) and plugin/ (the DLPack header). The cpu library is built from its host group, the binding is compiled against its header,
# and the libraries embed the headers it holds. The binding and tidy legs refuse by name a root that lacks one of eagle/door/eg_door.h, eagle/door/sources.txt
# and plugin/dlpack.h. The build also needs CXX (a C++23 compiler with OpenMP: the cpu library links the OpenMP runtime that compiler names, as the file it
# finds, so which runtime a process shares is decided by that compiler) and the raptor sibling's cpu facts and host door rows (RAPTOR);
# no prebuilt door library is an input.
# Each leg prints what it examined and is RED when it examined nothing. The exit status is non-zero when any leg is red.
cd "$(dirname "$0")/.." || exit 2
JOBS=${JOBS:-2}
out=$(mktemp "${TMPDIR:-/tmp}/gate.XXXXXX")
trap 'rm -f "$out" "$out".*' EXIT
CFILES=$(git ls-files loader include binding/template | grep -E '\.(c|h)$')

# The libraries embed the headers their compiles include, from folders their build names (OSPREY_CPU_* and OSPREY_CUDA_* _AETHER_HEADERS,
# _EAGLE_HEADERS, _FACTS_HEADERS). This gate prunes them from the checkouts named by OSPREY_AETHER_SRC (the aether checkout) and
# OSPREY_EAGLE_SRC (the eagle checkout that holds eagle/door and plugin/) into target/headers, and exports the six variables; a leg that builds is red without the two.
build_env() {
  [ -z "$OSPREY_CPU_AETHER_HEADERS" ] || return 0
  [ -n "$OSPREY_AETHER_SRC" ] && [ -n "$OSPREY_EAGLE_SRC" ] || { echo "RED: OSPREY_AETHER_SRC and OSPREY_EAGLE_SRC name the checkouts whose headers the libraries embed"; return 1; }
  sh ci/prepare_headers.sh target/headers "$OSPREY_AETHER_SRC" "$OSPREY_EAGLE_SRC" >"$out.headers" 2>&1 || { tail -3 "$out.headers"; echo "RED: ci/prepare_headers.sh"; return 1; }
  cat "$out.headers"
  for kind in cpu cuda; do
    up=$(echo $kind | tr a-z A-Z)
    for tree in aether eagle facts; do
      tu=$(echo $tree | tr a-z A-Z)
      eval "export OSPREY_${up}_${tu}_HEADERS=$PWD/target/headers/$kind/$tree"
    done
  done
  export OSPREY_CUDA_RTC_HEADERS=$PWD/target/headers/cuda/rtc
}

leg_skips=0; test_skips=0
need() { [ "$2" -gt 0 ] || { echo "RED: $1 examined nothing"; return 1; }; }
passed() { awk '/^test result:/ { p += $4 } END { print p + 0 }' "$out"; }

leg_fmt() {
  cargo fmt --all --check >"$out" 2>&1 || { head -5 "$out"; echo "RED: fmt differs"; return 1; }
  n=$(git ls-files | grep -c '\.rs$'); echo "examined: $n rust files"; need fmt "$n"
}

leg_clippy() {
  build_env || return 1
  cargo clippy --release -j "$JOBS" --workspace --all-targets -- -D warnings >"$out" 2>&1
  rc=$?
  cargo clippy --release -j "$JOBS" --workspace --all-targets --features osprey_replay/old-contract -- -D warnings >>"$out" 2>&1
  rc2=$?; [ $rc -ne 0 ] || rc=$rc2
  grep -E "^(warning|error)" "$out" | sort | uniq -c | sort -rn | head -8
  n=$(git ls-files | grep -c '\.rs$'); echo "examined: clippy over $n rust files, default features and old-contract, warnings denied"
  [ $rc -eq 0 ] || { echo "RED: clippy"; return 1; }; need clippy "$n"
}

# Every test file is compiled in (the old-contract feature) and must contribute tests: a file that contributes none is red. A test that
# checks nothing says so with one `skipped: <test>: <reason> [<variable>]` line (tests/common/mod.rs); the count is shown, and a skip on a
# variable that is set in this environment (stated but not honoured) is red.
leg_test() {
  build_env || return 1
  cargo test --release -j "$JOBS" --workspace --features osprey_replay/old-contract --no-fail-fast -- --nocapture >"$out" 2>&1
  rc=$?; n=$(passed); grep -E "FAILED|panicked|^error" "$out" | head -5
  per_file=$(awk '/^ *Running tests\// { name = $2 } /^running [0-9]+ tests?$/ && name != "" { print name, $2; name = "" }' "$out")
  files=; empty=
  for f in replay/tests/*.rs; do
    count=$(echo "$per_file" | awk -v f="${f#replay/}" '$1 == f { c = $2 } END { print c + 0 }')
    files="$files $(basename "$f" .rs)=$count"
    [ "$count" -gt 0 ] || empty="$empty $f"
  done
  skips=$(grep -c '^skipped: ' "$out"); test_skips=$skips
  echo "examined: $n tests passed, $skips skipped; per file:$files"
  [ $rc -eq 0 ] || { echo "RED: cargo test"; return 1; }
  [ -z "$empty" ] || { echo "RED: test files that contribute no tests:$empty"; return 1; }
  for v in $(grep -o '^skipped: .*\[[A-Z_]*\]$' "$out" | sed -E 's/.*\[([A-Z_]*)\]$/\1/' | sort -u); do
    eval "state=\${$v+set}"
    [ "$state" != set ] || { echo "RED: a test was skipped on $v, which is set"; return 1; }
  done
  need test "$n"
}

# each built library exports its entry and, when its manifest states a runtime, exactly the door functions of that runtime (check_exports.sh prints one line per library)
leg_exports() {
  build_env || return 1
  sh check_exports.sh >"$out" 2>"$out.err" || { tail -3 "$out" "$out.err"; echo "RED: check_exports.sh"; return 1; }
  n=$(grep -c . "$out"); echo "examined: $n libraries: $(tr '\n' ';' < "$out")"
  [ "$n" -eq 2 ] || { echo "RED: expected one symbol line for each of the two libraries, got $n"; return 1; }
}

# refuses, by name, an eagle source root that lacks one of its entries (see "Eagle source root" in the header of this file)
door_root() {
  for entry in eagle/door/eg_door.h eagle/door/sources.txt plugin/dlpack.h; do
    [ -e "$OSPREY_EAGLE_SRC/$entry" ] || {
      echo "RED: OSPREY_EAGLE_SRC $OSPREY_EAGLE_SRC lacks $entry; the \"Eagle source root\" lines in the header of ci/gate.sh say what it holds"
      return 1
    }
  done
}

# builds the binding for the interpreter $1 (its headers $2; empty: the interpreter's own) and runs binding/test_binding.py on it
build_and_test_binding() {
  door_root || return 1
  cargo build --release -j "$JOBS" --manifest-path "${RAPTOR:-../raptor}/Cargo.toml" -p raptor_ir -p raptor_wings -p raptor_support >"$out" 2>&1 \
    || { grep -E "^error" "$out" | head -3; echo "RED: build of the contract libraries in ${RAPTOR:-../raptor}"; return 1; }
  PYTHON=$1 PY_INCLUDE=$2 sh binding/build.sh >"$out" 2>&1; rc=$?
  tail -4 "$out"
  g=$(grep -c "GATE GREEN" "$out"); echo "examined: binding build for $1, GATE GREEN lines: $g"
  [ $rc -eq 0 ] || { echo "RED: binding build"; return 1; }; need binding "$g" || return 1
  JOBS=$JOBS sh ci/dist.sh >"$out" 2>&1 || { grep -E "^error" "$out" | head -3; echo "RED: library build (ci/dist.sh)"; return 1; }
  case $(uname -s) in Darwin) ext=dylib ;; *) ext=so ;; esac
  (cd binding && PYTHONPATH=build OSPREY_DIST="$PWD/../dist" HOST_LIB="$PWD/../dist/osprey_cpu/libosprey_cpu.$ext" CUDA_LIB="$PWD/../dist/osprey_cuda/libosprey_cuda.$ext" ABI1_LIB="build/libabi1.$ext" \
    CONTRACT0_LIB="build/libcontract0.$ext" "$1" test_binding.py) >"$out" 2>&1 || { tail -5 "$out"; echo "RED: test_binding.py"; return 1; }
  line=$(grep -E "^test_binding: [0-9]+ checks passed" "$out"); echo "examined: $line"
  need binding "$(echo "$line" | tr -dc '0-9 ' | awk '{print $1 + 0}')"
}

# the binding's build and its gate, then its tests; it must print GATE GREEN
leg_binding() {
  build_env || return 1
  [ -n "$PYTHON" ] || { echo "RED: PYTHON is not set"; return 1; }
  build_and_test_binding "$PYTHON" ""
}

# the same for Python 3.9, the floor of the binding's stable ABI; skipped, with a line saying so, when no such interpreter exists
leg_py39() {
  build_env || return 1
  py39=${PYTHON39:-/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9}
  if [ ! -x "$py39" ]; then echo "skipped: no Python 3.9 interpreter at $py39 (set PYTHON39 and PY39_INCLUDE)"; leg_skips=$((leg_skips + 1)); return 0; fi
  "$py39" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 9))' || { echo "RED: $py39 is not Python 3.9"; return 1; }
  build_and_test_binding "$py39" "${PY39_INCLUDE:-$(dirname "$(dirname "$py39")")/Headers}"
}

# the abi number has one source: the header's define. The loader and the table test read it from there.
leg_abi() {
  build_env || return 1
  defs=$(git ls-files include loader components osprey_cpu osprey_cuda replay binding | xargs grep -E '^[ \t]*#[ \t]*define[ \t]+OB_ABI[ \t]' 2>/dev/null | wc -l | tr -d ' ')
  echo "examined: $defs definition(s) of OB_ABI (include/osprey.h is the one source)"
  [ "$defs" -eq 1 ] || { echo "RED: OB_ABI is defined $defs times"; return 1; }
  cargo test --release -j "$JOBS" -p osprey_replay --features old-contract --test table >"$out" 2>&1; rc=$?; n=$(passed)
  echo "examined: the table test, $n passed (header define equals the library's entry table abi)"
  [ $rc -eq 0 ] || { grep -E "FAILED|panicked|^error" "$out" | head -3; echo "RED: table test"; return 1; }; need abi "$n"
}

# the binding and the libraries are built, then the three wheels are written from them and installed by name from the wheel folder into a fresh
# environment under a temporary folder (no network, no other package of ours); there, from outside the tree, the versions, the modules and the
# opened backends are checked. A second set of wheels one version newer is then installed over the first with plain pip: it must replace it.
# The newer set is built with OSPREY_TEST_DEV_NUMBER, which only this leg sets; dist/ is rebuilt at the real version afterwards.
leg_wheels() {
  build_env || return 1
  [ -n "$PYTHON" ] || { echo "RED: PYTHON is not set"; return 1; }
  build_and_test_binding "$PYTHON" "" || return 1
  rm -rf dist/wheels dist/wheels_next
  "$PYTHON" ci/wheels.py build >"$out" 2>&1 || { tail -3 "$out"; echo "RED: ci/wheels.py build"; return 1; }
  cat "$out"; n=$(ls dist/wheels/*.whl | wc -l | tr -d ' '); need wheels "$n" || return 1
  [ "$n" -eq 3 ] || { echo "RED: expected three wheels, got $n"; return 1; }
  version=$(sh ci/version.sh) || return 1
  newer=${version%%.dev*}.dev$((${version##*.dev} + 1))
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/wheels_venv.XXXXXX")
  "$PYTHON" -m venv "$scratch/venv" >"$out" 2>&1 || { tail -3 "$out"; echo "RED: venv"; return 1; }
  names="raptor-osprey raptor-osprey-cpu raptor-osprey-cuda"
  wheels_dir=$PWD/dist/wheels; next_dir=$PWD/dist/wheels_next
  # shellcheck disable=SC2086
  "$scratch/venv/bin/python" -m pip install --quiet --no-index --find-links "$wheels_dir" $names >"$out" 2>&1 \
    || { tail -3 "$out"; echo "RED: pip install of the wheels"; return 1; }
  (cd "$scratch" && PYTHONPATH= "$scratch/venv/bin/python" -I "$OLDPWD/ci/wheels.py" check "$version") >"$out" 2>&1 \
    || { tail -3 "$out"; echo "RED: wheels check"; return 1; }
  echo "examined: $(cat "$out")"
  OSPREY_TEST_DEV_NUMBER=${newer##*.dev} JOBS=$JOBS sh ci/dist.sh >"$out" 2>&1 || { tail -3 "$out"; echo "RED: ci/dist.sh for the newer version"; return 1; }
  OSPREY_TEST_DEV_NUMBER=${newer##*.dev} "$PYTHON" ci/wheels.py build "$next_dir" >"$out" 2>&1 || { tail -3 "$out"; echo "RED: newer wheels build"; return 1; }
  JOBS=$JOBS sh ci/dist.sh >"$out" 2>&1 || { tail -3 "$out"; echo "RED: ci/dist.sh back at the real version"; return 1; }
  # shellcheck disable=SC2086
  "$scratch/venv/bin/python" -m pip install --quiet --upgrade --no-index --find-links "$next_dir" $names >"$out" 2>&1 \
    || { tail -3 "$out"; echo "RED: pip install of the newer wheels"; return 1; }
  (cd "$scratch" && PYTHONPATH= "$scratch/venv/bin/python" -I "$OLDPWD/ci/wheels.py" check "$newer") >"$out" 2>&1 \
    || { tail -3 "$out"; echo "RED: the newer wheels did not replace the older ones"; return 1; }
  echo "examined: $version replaced by $newer: $(cat "$out")"
  rm -rf "$scratch" dist/wheels_next
}

leg_ruff() {
  command -v ruff >/dev/null 2>&1 || { echo "RED: ruff is not on PATH"; return 1; }
  n=$(git ls-files | grep -c '\.py$'); echo "examined: $n python files"; need ruff "$n" || return 1
  ruff check >"$out" 2>&1 || { tail -5 "$out"; echo "RED: ruff check"; return 1; }
  ruff format --check >"$out" 2>&1 || { tail -5 "$out"; echo "RED: ruff format"; return 1; }
}

# The C files are formatted by clang-format major version $CFORMAT_MAJOR (the formatter's output differs between majors): on another
# major this leg skips, saying so.
CFORMAT_MAJOR=23
leg_cformat() {
  command -v clang-format >/dev/null 2>&1 || { echo "RED: clang-format is not on PATH"; return 1; }
  major=$(clang-format --version | sed -E 's/^.*version ([0-9]+)\..*$/\1/')
  if [ "$major" != "$CFORMAT_MAJOR" ]; then echo "skipped: clang-format $major found, the C files are pinned to $CFORMAT_MAJOR"; leg_skips=$((leg_skips + 1)); return 0; fi
  n=$(echo "$CFILES" | grep -c .); echo "examined: $n C files in loader/ include/ binding/template/"; need cformat "$n" || return 1
  # shellcheck disable=SC2086
  clang-format --dry-run -Werror $CFILES >"$out" 2>&1 || { grep -E "error" "$out" | head -5; echo "RED: clang-format differs"; return 1; }
}

leg_tidy() {
  [ -n "$CLANG_TIDY" ] || { echo "RED: CLANG_TIDY is not set (examined: 0 files)"; return 1; }
  [ -n "$PYTHON" ] || { echo "RED: PYTHON is not set (the binding template includes Python.h)"; return 1; }
  door_root || return 1
  pyinc=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
  tfiles=$(git ls-files loader binding/template | grep '\.c$')
  n=$(echo "$tfiles" | grep -c .); echo "examined: $n C files of loader/ and binding/template/ with .clang-tidy"; need tidy "$n" || return 1
  # shellcheck disable=SC2086
  "$CLANG_TIDY" $tfiles -- -std=c11 -Iinclude -Ibinding/template -I"$pyinc" -I"$OSPREY_EAGLE_SRC" -I"$OSPREY_EAGLE_SRC/eagle/door/test/c_shim" -I../raptor/rs/ir/include >"$out" 2>&1 \
    || { grep -E "error|warning" "$out" | head -5; echo "RED: clang-tidy"; return 1; }
}

# the recorded small file replays through the writers end to end: every unit is written and none differs from its recording
leg_replay() {
  build_env || return 1
  cargo run --release -j "$JOBS" -p osprey_replay --features old-contract --bin treplay -- replay/tests/data/units_small.jsonl --target host >"$out" 2>&1 \
    || { tail -5 "$out"; echo "RED: treplay"; return 1; }
  line=$(grep -E '^text identical \(whole path\)' "$out" | head -1)
  echo "examined: $line"
  n=$(echo "$line" | sed -E 's/^[^0-9]*([0-9]+) \/ ([0-9]+).*written but different ([0-9]+).*$/\1 \2 \3/')
  set -- $n
  [ "$#" -eq 3 ] && [ "$1" -gt 0 ] && [ "$1" -eq "$2" ] && [ "$3" -eq 0 ] \
    && grep -q '^body written but different 0$' "$out" && grep -q '^render errors 0$' "$out" \
    || { grep -E 'different|errors|refused' "$out" | head -5; echo "RED: a unit was not written or differs from its recording"; return 1; }

  oracle
}

# the facts oracle: every recorded host unit's machine record equals the facts answer for the machine its recording states
# (it reads recordings of the old shape, so it needs the old-contract feature); skipped, with a line saying so, when no recording is found
oracle() {
  cargo test --release -j "$JOBS" -p osprey_replay --features old-contract --test facts_oracle -- --nocapture >"$out" 2>&1 \
    || { grep -E "FAILED|panicked|^error" "$out" | head -3; echo "RED: facts oracle"; return 1; }
  if grep -q "^skipped: .*no recording" "$out"; then grep "^skipped: .*no recording" "$out" | head -1; return 0; fi
  echo "examined: facts oracle, $(passed) test passed"; need oracle "$(passed)"
}

legs=${*:-fmt clippy test exports binding py39 abi wheels ruff cformat tidy replay}
green=; red=
for l in $legs; do
  echo "== $l"
  if "leg_$l"; then green="$green $l"; else red="$red $l"; fi
done
echo "skipped in total: $test_skips test(s), $leg_skips leg(s) (python3.9 / clang-format)"
echo "GREEN:$green"; [ -z "$red" ] || echo "RED:$red"
[ -z "$red" ]
