#!/bin/sh
# Each built library exports `osprey_v1` and, when it carries the runtime door, exactly the functions of that door's header (eagle's
# eg_door.h, under OSPREY_EAGLE_SRC); a library without the door exports no door function. The cpu library carries the door when its
# manifest states a runtime. The cuda library carries it when it was built with a CUDA toolkit (osprey_cuda/build.rs): the library
# itself says which mode it is in, by defining the door's first function, and the line printed for it names the mode that was
# checked, so a run on a machine with a toolkit that prints "without the device runtime" proves nothing about the device group.
# No library needs an OpenMP runtime, and no library exports or needs the CUDA runtime or driver API (the cuda library links its
# runtime statically and private, and loads cuBLAS and cuFFT at use). No rlib defines the entry. The rules are ci/export_rules.sh.
set -e
cd "$(dirname "$0")"
: "${OSPREY_EAGLE_SRC:?set OSPREY_EAGLE_SRC to the eagle source root: the door header names the functions a runtime exports}"
. ./ci/export_rules.sh
# the functions of the runtime door: `eg_status eg_<name>(` in its header
door_functions() { grep -oE '^eg_status +eg_[a-z0-9_]+\(' "$OSPREY_EAGLE_SRC/eagle/door/eg_door.h" | sed -E 's/.*(eg_[a-z0-9_]+)\(/\1/'; }
refuse() { echo "$1" >&2; [ -z "$2" ] || printf '%s\n' "$2" >&2; exit 1; }
cargo build -j 2 --lib -p osprey_cpu_door -p osprey_cuda_door >/dev/null
for name in osprey_cpu osprey_cuda; do
  lib=$(ls target/debug/lib$name.dylib target/debug/lib$name.so 2>/dev/null | head -1)
  got=$(defined_names "$lib")
  mode=$(echo "$got" | door_mode)
  case $name in
    osprey_cuda) with_door=$([ "$mode" = "with the device runtime" ] && echo yes || echo no) ;;
    *) with_door=$(grep -q '"runtime": "eagle-door/1"' "$name/osprey.json" && echo yes || echo no) ;;
  esac
  want=osprey_v1
  if [ "$with_door" = yes ]; then want=$(printf 'osprey_v1\n%s\n' "$(door_functions)"); fi
  differ=$(echo "$got" | export_differences "$want")
  [ -z "$differ" ] || refuse "$name: the exported symbols differ from the door the library carries; extra or missing:" "$differ"
  # a backend library brings no threading runtime into the process that loads it
  bad=$({ echo "$got"; undefined_names "$lib"; } | openmp_symbols)
  [ -z "$bad" ] || refuse "$name: the library needs an OpenMP runtime (the symbols below)" "$bad"
  bad=$(needed_names "$lib" | forbidden_needed)
  [ -z "$bad" ] || refuse "$name: the library needs a runtime library that must stay out (a shared OpenMP or CUDA library):" "$bad"
  bad=$(echo "$got" | cuda_symbols)
  [ -z "$bad" ] || refuse "$name: the library exports CUDA runtime or driver symbols (the static runtime must stay private):" "$bad"
  line="$name: $(echo "$got" | wc -l | tr -d ' ') exported symbols: osprey_v1"
  [ "$with_door" = no ] || line="$line and the $(echo "$got" | grep -c '^eg_') functions of eagle-door/1"
  [ "$name" != osprey_cuda ] || line="$line; mode checked: $mode"
  echo "$line"
done
# the rlibs other crates link define no entry symbol: two libraries then link into one test binary
for name in osprey_cpu osprey_cuda; do
  for rlib in target/debug/deps/lib${name}_lib-*.rlib; do
    if nm -g "$rlib" 2>/dev/null | grep -q ' T _\{0,1\}osprey_v1$'; then echo "$name: the rlib defines osprey_v1" >&2; exit 1; fi
  done
done
