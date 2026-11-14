#!/bin/sh
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# Builds the extension: generated C + template + libraptor_ir.a -> build/_osprey.abi3.so, then runs the gate.
# Environment (all optional):
#   RX_SRC    raptor checkout holding rs/ir/include/raptor_ir.h and target/release/libraptor_ir.a   (default: the carrier's parent folder)
#   RX_LIB    the static library (default: $RX_SRC/target/release/libraptor_ir.a)
#   PYTHON    interpreter whose headers are used (default: python3)
#   CC        C compiler (default: clang on macOS, cc elsewhere)
#   CFLAGS_EXTRA, RX_LDLIBS   extra compile flags; libraries the Rust library needs on Linux (default: -lpthread -ldl -lm)
#   OSPREY_BUILD   output folder (default: build next to this script)
set -e
here=$(cd "$(dirname "$0")" && pwd)
src=${RX_SRC:-$here/../..}
lib=${RX_LIB:-$src/target/release/libraptor_ir.a}
inc=$src/rs/ir/include
py=${PYTHON:-python3}
out=${OSPREY_BUILD:-$here/build}
os=$(uname -s)
[ "$os" = Darwin ] && cc=${CC:-clang} || cc=${CC:-cc}
pyinc=$("$py" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
mkdir -p "$out"
cargo build -j 2 --release --manifest-path "$here/gen/Cargo.toml" 2>&1 | tail -1
gen=$here/gen/target/release/gen
export RX_SRC="$src"
"$gen" --header "$inc/raptor_ir.h" --generated "$out/_osprey.c"
for f in "$out/_osprey.c" "$here/template/types.c"; do
  $cc -std=c11 -O2 -fPIC -fvisibility=hidden -Wall -Wextra -Werror $CFLAGS_EXTRA -I"$here/template" -I"$inc" -I"$pyinc" -c "$f" -o "$out/$(basename "$f" .c).o"
done
so=$out/_osprey.abi3.so
if [ "$os" = Darwin ]; then
  $cc -dynamiclib -o "$so" "$out/_osprey.o" "$out/types.o" "$lib" \
    -Wl,-exported_symbol,_PyInit__osprey -Wl,-dead_strip -Wl,-undefined,dynamic_lookup
  codesign -v "$so" && echo "codesign -v ok"
else
  printf '{ global: PyInit__osprey; local: *; };\n' > "$out/exports.map"
  $cc -shared -o "$so" "$out/_osprey.o" "$out/types.o" "$lib" \
    -Wl,--version-script="$out/exports.map" -Wl,--exclude-libs,ALL -Wl,--gc-sections ${RX_LDLIBS:--lpthread -ldl -lm}
fi
cd "$here"
"$gen" --check --header "$inc/raptor_ir.h" --generated "$out/_osprey.c" --template "$here/template" --ext "$so"
