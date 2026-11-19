#!/bin/sh
# generated C + template + loader + libraptor_ir.a -> build/_osprey.abi3.so ; then the gate.
# Needs PYTHON (an interpreter whose headers the extension is built against; PY_INCLUDE names another header directory, e.g. of an
# interpreter whose sysconfig points nowhere) and cargo on PATH. The extension is built for the stable ABI of Python 3.9 and later.
# RAPTOR is the raptor checkout (default: the sibling folder of this repository); RX_LIB and RX_INC override its library and header.
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
: "${PYTHON:?set PYTHON to the interpreter the extension is built for}"
# OSPREY_EAGLE_SRC is the eagle source root, the one input of the runtime: door.c is compiled against its door header
# (eagle/door/eg_door.h) and the DLPack header it vendors (plugin/dlpack.h), and the fixtures' sources are its door's own.
: "${OSPREY_EAGLE_SRC:?set OSPREY_EAGLE_SRC to the eagle source root}"
eagle=$OSPREY_EAGLE_SRC
# the vendored dlpack.h spells <cstdint>; the door repository's own C shim folder maps that name to <stdint.h>
shim=$eagle/eagle/door/test/c_shim
raptor=${RAPTOR:-$repo/../raptor}
lib=${RX_LIB:-$raptor/target/release/libraptor_wings.a}
inc=${RX_INC:-$raptor/rs/ir/include}
wings_inc=${RW_INC:-$raptor/rs/wings/include}
# the front end's surface: what it builds, what it reads back, and the analyses it asks (the query header holds the kind test, walks and constants)
hdrs="--header $inc/raptor_ir.h --header $inc/raptor_ir_front.h --header $inc/raptor_ir_query.h --header $wings_inc/raptor_wings_front.h"
py=${PY_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')}
cargo build -j 2 --release --manifest-path "$here/gen/Cargo.toml" 2>&1 | tail -1
gen=$repo/target/release/gen
# the manifest checks the loader answers three calls with (components/manifest)
cargo build -j 2 --release --manifest-path "$repo/Cargo.toml" -p osprey_manifest 2>&1 | tail -1
manifest_lib=$repo/target/release/libosprey_manifest.a
mkdir -p "$here/build"
"$gen" $hdrs --header "$repo/include/osprey.h" --generated "$here/build/_osprey.c"
for f in "$here/build/_osprey.c" "$here/template/types.c" "$here/template/door.c" "$repo/loader/loader.c"; do
  clang -std=c11 -O2 -fPIC -Wall -Wextra -Werror -I"$here/template" -I"$inc" -I"$wings_inc" -I"$repo/include" -I"$eagle" -I"$shim" -I"$py" -c "$f" -o "$here/build/$(basename "$f" .c).o"
done
# The link leaves unresolved symbols to the import (dynamic lookup, a version script), so a static library older than the headers the
# wrappers were generated from would only fail when the module loads. Every rx_ / rw_ function the objects call must be defined by it.
# The static library is link-time-optimised bitcode, which only LLVM's nm reads: NM, else llvm-nm on PATH, else the one beside
# CLANG_TIDY.
nm=${NM:-}
[ -n "$nm" ] || nm=$(command -v llvm-nm || true)
if [ -z "$nm" ] && [ -n "${CLANG_TIDY:-}" ] && [ -x "$(dirname "$CLANG_TIDY")/llvm-nm" ]; then
  nm=$(dirname "$CLANG_TIDY")/llvm-nm
fi
[ -n "$nm" ] || { echo "build.sh: no llvm-nm to read $lib; set NM to it" >&2; exit 1; }
"$nm" -u "$here/build/_osprey.o" "$here/build/types.o" "$here/build/door.o" "$here/build/loader.o" |
  sed -n 's/^ *_\{0,1\}\(r[xw]_[A-Za-z0-9_]*\)$/\1/p' | sort -u > "$here/build/symbols_needed.txt"
"$nm" -g --defined-only "$lib" | sed -n 's/^.* [TDS] _\{0,1\}\(r[xw]_[A-Za-z0-9_]*\)$/\1/p' | sort -u > "$here/build/symbols_defined.txt"
missing=$(comm -23 "$here/build/symbols_needed.txt" "$here/build/symbols_defined.txt")
if [ -n "$missing" ]; then
  echo "build.sh: $lib does not define: $(echo $missing | tr '\n' ' ')" >&2
  echo "build.sh: it is older than the headers the wrappers are generated from; rebuild it: (cd $raptor && cargo build --release)" >&2
  exit 1
fi
case $(uname -s) in
  Darwin) shared=-dynamiclib; ext=dylib ;;
  *) shared=-shared; ext=so ;;
esac
if [ "$ext" = dylib ]; then
  clang $shared -o "$here/build/_osprey.abi3.so" "$here/build/_osprey.o" "$here/build/types.o" "$here/build/door.o" "$here/build/loader.o" "$manifest_lib" "$lib" \
    -Wl,-exported_symbol,_PyInit__osprey -Wl,-dead_strip -Wl,-undefined,dynamic_lookup
  codesign -v "$here/build/_osprey.abi3.so" && echo "codesign -v ok"
else
  printf '{\n  global:\n    PyInit__osprey;\n  local:\n    *;\n};\n' > "$here/build/osprey.map"
  clang $shared -o "$here/build/_osprey.abi3.so" "$here/build/_osprey.o" "$here/build/types.o" "$here/build/door.o" "$here/build/loader.o" "$manifest_lib" "$lib" \
    -Wl,--version-script="$here/build/osprey.map" -Wl,--gc-sections -lm -ldl
fi
# a library with the wrong abi number, for the entry's refusal test
clang $shared -fPIC -o "$here/build/libabi1.$ext" "$here/fixture/abi1.c"
clang $shared -fPIC -I"$inc" -I"$repo/include" -o "$here/build/libcontract0.$ext" "$here/fixture/contract0.c"
# the unit libraries of the door's own fixtures (units; a recorded loop and its reverse; matrix-product units), for the binding's tests of the door calls
clang $shared -fPIC -std=c11 -I"$eagle" -o "$here/build/libfixture_unit.$ext" "$eagle/eagle/door/test/fixture_unit.c"
clang $shared -fPIC -std=c11 -I"$eagle" -o "$here/build/libfixture_reverse.$ext" "$eagle/eagle/door/test/fixture_reverse.c"
clang $shared -fPIC -std=c11 -I"$eagle" -o "$here/build/libfixture_gemm.$ext" "$eagle/eagle/door/test/fixture_gemm.c"
"$gen" --check $hdrs --header "$repo/include/osprey.h" --door-header "$eagle/eagle/door/eg_door.h" --door-source "$here/template/door.c" \
  --template "$here/template" --generated "$here/build/_osprey.c" --ext "$here/build/_osprey.abi3.so"
# the module must load: a link gap fails here, not in a later caller
PYTHONPATH="$here/build" "$PYTHON" -c "import _osprey"
