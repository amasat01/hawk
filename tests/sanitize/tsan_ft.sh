#!/usr/bin/env bash
# tsan-ft gate: hawk._core and hawk's host kernels built with ThreadSanitizer,
# the free-threading (`ft`) rows run under a TSan free-threaded CPython
# (build_tsan_python.sh).
#
#   tests/sanitize/tsan_ft.sh <tsan-python>        (run from the hawk root)
#
# Inputs from the environment: HAWK_AETHER_INCLUDE and HAWK_EAGLE_INCLUDE (the
# aether and eagle checkouts, as for any hawk build), RAPTOR_DIR (a raptor
# checkout; else raptor-core from PyPI), CC/CXX (default clang/clang++; TSan
# runtimes must match the interpreter's). eagle is built from
# $HAWK_EAGLE_INCLUDE without its CUDA plugin; no GPU is used.
#
# GREEN needs all of:
#   (i)   the interpreter is TSan-built with the GIL off, and _core carries TSan
#         instrumentation;
#   (ii)  the canary race IS reported without hawk's suppressions (the
#         detector is live and reaches _core);
#   (iii) the ft rows collect, pass, and produce ZERO TSan reports with
#         CPython's suppressions + tests/sanitize/tsan.supp. The host kernels
#         hawk compiles at run time are instrumented too ($HAWK_CXX wraps
#         $CXX with -fsanitize=thread).
set -euo pipefail

py="${1:?usage: tests/sanitize/tsan_ft.sh <tsan-python>}"
root="$(pwd)"
[ -f "$root/pyproject.toml" ] && [ -d "$root/hawk" ] || { echo "run from the hawk root"; exit 2; }
: "${HAWK_AETHER_INCLUDE:?set HAWK_AETHER_INCLUDE to an aether checkout}"
: "${HAWK_EAGLE_INCLUDE:?set HAWK_EAGLE_INCLUDE to an eagle checkout}"
work="$root/build/tsan"
prefix="$(cd "$(dirname "$py")/.." && pwd)"
cpython_supp="$prefix/share/tsan/suppressions_free_threading.txt"
[ -f "$cpython_supp" ] || { echo "RED: no CPython TSan suppressions at $cpython_supp"; exit 1; }
mkdir -p "$work"
cat "$cpython_supp" "$root/tests/sanitize/tsan.supp" > "$work/all.supp"
export CC="${CC:-clang}" CXX="${CXX:-clang++}" CUDA_VISIBLE_DEVICES=""

# (i) the interpreter
"$py" -c "
import sys, sysconfig
assert '--with-thread-sanitizer' in (sysconfig.get_config_var('CONFIG_ARGS') or ''), 'not a TSan build'
assert sys._is_gil_enabled() is False, 'GIL is enabled'
print('TSan CPython', sys.version.split()[0], 'GIL off')"

rm -rf "$work/venv"
"$py" -m venv --without-pip "$work/venv"
venv_py="$work/venv/bin/python"
# pip into the venv (the interpreter is built without ensurepip)
curl -fsSL -o "$work/get-pip.py" https://bootstrap.pypa.io/get-pip.py
"$venv_py" "$work/get-pip.py" --quiet
"$venv_py" -m pip install --quiet --only-binary=:all: pytest numpy scipy \
    "scikit-build-core>=0.10" "nanobind>=3.1,<4" "aether-dsc>=0.2,<0.3"
if [ -n "${RAPTOR_DIR:-}" ]; then
    "$venv_py" -m pip install --quiet --no-deps "$RAPTOR_DIR"
else
    "$venv_py" -m pip install --quiet --only-binary=:all: "raptor-core>=0.3"
fi

# eagle and hawk, instrumented, with debug info so reports name file:line
tsan_build() {
    CFLAGS="-fsanitize=thread -g -O2" CXXFLAGS="-fsanitize=thread -g -O2" LDFLAGS="-fsanitize=thread" \
        "$venv_py" -m pip install --quiet --no-deps --no-build-isolation \
        -Ccmake.build-type=RelWithDebInfo -Cinstall.strip=false -Cbuild-dir="build/tsan-{wheel_tag}" "$@"
}
tsan_build -Ccmake.define.EAGLE_PYTHON_CUDA_PLUGIN=OFF -Ccmake.define.EAGLE_PYTHON_MPI=OFF \
    -e "$HAWK_EAGLE_INCLUDE/python"
tsan_build -e "$root"
core="$("$venv_py" -c 'import hawk._core as m; print(m.__file__)')"
n_tsan="$(nm -D "$core" | grep -c __tsan_ || true)"
[ "$n_tsan" -gt 0 ] || { echo "RED: $core carries no TSan instrumentation"; exit 1; }
echo "_core instrumented ($n_tsan __tsan_ symbols)"

# run-time host kernels: the same compiler, instrumented, in a private cache
mkdir -p "$work/bin"
printf '#!/bin/sh\nexec %s -fsanitize=thread -g "$@"\n' "$(command -v "$CXX")" > "$work/bin/tsan-cxx"
chmod +x "$work/bin/tsan-cxx"
export HAWK_CXX="$work/bin/tsan-cxx" HAWK_CACHE_DIR="$work/kernel-cache"
rm -rf "$HAWK_CACHE_DIR"

# (ii) non-vacuity: the canary race must be reported with only CPython's file
rm -rf "$work/reports" && mkdir -p "$work/reports"
TSAN_OPTIONS="suppressions=$cpython_supp exitcode=0 log_path=$work/reports/canary" \
    "$venv_py" - <<'EOF'
import threading
import hawk._core as core
threads = [threading.Thread(target=lambda: [core._unsynchronised_bump() for _ in range(20000)])
           for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
EOF
if ! cat "$work"/reports/canary.* 2>/dev/null | grep -q "unsynchronised_bump"; then
    echo "RED: the canary race was not reported — TSan is not reaching _core"; exit 1
fi
echo "canary race reported (detector live)"

# (iii) the ft rows, with every suppression
n_ft="$(cd "$root" && "$venv_py" -m pytest tests -m ft --collect-only -q -p no:cacheprovider | grep -c '::' || true)"
[ "$n_ft" -gt 0 ] || { echo "RED: no ft rows collected"; exit 1; }
set +e
(cd "$root" && TSAN_OPTIONS="suppressions=$work/all.supp second_deadlock_stack=1 log_path=$work/reports/ft" \
    "$venv_py" -m pytest tests -m ft -q -rs -p no:cacheprovider --basetemp="$work/pytest")
rc=$?
set -e
n_reports="$(cat "$work"/reports/ft.* 2>/dev/null | grep -c 'WARNING: ThreadSanitizer' || true)"
# a TSan-fatal child (compiler trampoline, subprocess) must never read as success:
# the default exitcode (66) is kept for the ft run, and fatal lines are surfaced
if cat "$work"/reports/ft.* 2>/dev/null | grep -q -E 'FATAL: ThreadSanitizer|ERROR: ThreadSanitizer'; then
    echo "RED: ThreadSanitizer fatal error in a process of the ft run:"
    cat "$work"/reports/ft.* | grep -E -A12 'FATAL: ThreadSanitizer|ERROR: ThreadSanitizer' | head -60; exit 1
fi
if [ "$n_reports" -gt 0 ]; then
    echo "RED: $n_reports TSan report(s):"; cat "$work"/reports/ft.*; exit 1
fi
[ "$rc" -eq 0 ] || { echo "RED: ft rows failed (pytest rc=$rc)"; exit 1; }
echo "GREEN: ft rows under TSan ($n_ft collected), 0 reports"
