#!/bin/sh
# Each built library exports `osprey_v1` and, when its manifest states a runtime, exactly the functions of that runtime's door
# header (eagle's eg_door.h, under OSPREY_EAGLE_SRC); a library that states none exports no door function. No rlib defines the entry.
set -e
cd "$(dirname "$0")"
: "${OSPREY_EAGLE_SRC:?set OSPREY_EAGLE_SRC to the eagle source root: the door header names the functions a runtime exports}"
# the global symbols a library defines, one nm line each: GNU nm says --defined-only, the BSD nm of macOS says -U
if nm --version 2>/dev/null | grep -q GNU; then defined() { nm -g --defined-only "$1"; }; else defined() { nm -gU "$1"; }; fi
# the functions of the runtime door: `eg_status eg_<name>(` in its header
door_functions() { grep -oE '^eg_status +eg_[a-z0-9_]+\(' "$OSPREY_EAGLE_SRC/eagle/door/eg_door.h" | sed -E 's/.*(eg_[a-z0-9_]+)\(/\1/'; }
cargo build -j 2 --lib -p osprey_cpu_door -p osprey_cuda_door >/dev/null
for name in osprey_cpu osprey_cuda; do
  lib=$(ls target/debug/lib$name.dylib target/debug/lib$name.so 2>/dev/null | head -1)
  got=$(defined "$lib" | awk 'NF >= 3 { sub(/^_/, "", $3); print $3 }' | sort)
  want=osprey_v1
  if grep -q '"runtime": "eagle-door/1"' "$name/osprey.json"; then want=$(printf 'osprey_v1\n%s\n' "$(door_functions)" | sort); fi
  [ "$got" = "$(echo "$want" | sort)" ] || { echo "$name: the exported symbols differ from the manifest's runtime; extra or missing:" >&2; printf '%s\n%s\n' "$got" "$want" | sort | uniq -u >&2; exit 1; }
  echo "$name: $(echo "$got" | wc -l | tr -d ' ') exported symbols: osprey_v1$(echo "$got" | grep -q '^eg_' && echo " and the $(echo "$got" | grep -c '^eg_') functions of eagle-door/1")"
done
# the rlibs other crates link define no entry symbol: two libraries then link into one test binary
for name in osprey_cpu osprey_cuda; do
  for rlib in target/debug/deps/lib${name}_lib-*.rlib; do
    if nm -g "$rlib" 2>/dev/null | grep -q ' T _\{0,1\}osprey_v1$'; then echo "$name: the rlib defines osprey_v1" >&2; exit 1; fi
  done
done
