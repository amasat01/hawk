#!/bin/sh
# Each built library exports exactly one symbol, and no rlib defines it.
set -e
cd "$(dirname "$0")"
# the global symbols a library defines, one nm line each: GNU nm says --defined-only, the BSD nm of macOS says -U
if nm --version 2>/dev/null | grep -q GNU; then defined() { nm -g --defined-only "$1"; }; else defined() { nm -gU "$1"; }; fi
cargo build -j 2 --lib -p osprey_cpu_door -p osprey_cuda_door >/dev/null
for name in osprey_cpu osprey_cuda; do
  lib=$(ls target/debug/lib$name.dylib target/debug/lib$name.so 2>/dev/null | head -1)
  syms=$(defined "$lib" | awk 'NF >= 3 { print $3 }')
  echo "$syms"
  [ "$(echo "$syms" | wc -l | tr -d ' ')" = 1 ] || { echo "$name: expected exactly one exported symbol" >&2; exit 1; }
done
# the rlibs other crates link define no entry symbol: two libraries then link into one test binary
for name in osprey_cpu osprey_cuda; do
  for rlib in target/debug/deps/lib${name}_lib-*.rlib; do
    if nm -g "$rlib" 2>/dev/null | grep -q ' T _\{0,1\}osprey_v1$'; then echo "$name: the rlib defines osprey_v1" >&2; exit 1; fi
  done
done
