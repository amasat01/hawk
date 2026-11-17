#!/bin/sh
# generated C + template + loader + libraptor_ir.a -> build/_osprey.abi3.so ; then the gate.
# Needs PYTHON (an interpreter whose headers the extension is built against; PY_INCLUDE names another header directory, e.g. of an
# interpreter whose sysconfig points nowhere) and cargo on PATH. The extension is built for the stable ABI of Python 3.9 and later.
# RAPTOR is the raptor checkout (default: the sibling folder of this repository); RX_LIB and RX_INC override its library and header.
set -e
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
: "${PYTHON:?set PYTHON to the interpreter the extension is built for}"
# EAGLE_DOOR_DIR is the include root that holds eagle/door/eg_door.h (the runtime's C door), which door.c is compiled against.
: "${EAGLE_DOOR_DIR:?set EAGLE_DOOR_DIR to the include root of the door header}"
raptor=${RAPTOR:-$repo/../raptor}
lib=${RX_LIB:-$raptor/target/release/libraptor_wings.a}
inc=${RX_INC:-$raptor/rs/ir/include}
wings_inc=${RW_INC:-$raptor/rs/wings/include}
# the front end's surface: what it builds, what it reads back, and the analyses it asks (the query header holds the kind test, walks and constants)
hdrs="--header $inc/raptor_ir.h --header $inc/raptor_ir_front.h --header $inc/raptor_ir_query.h --header $wings_inc/raptor_wings_front.h"
py=${PY_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')}
cargo build -j 2 --release --manifest-path "$here/gen/Cargo.toml" 2>&1 | tail -1
gen=$here/gen/target/release/gen
mkdir -p "$here/build"
"$gen" $hdrs --header "$repo/include/osprey_host.h" --generated "$here/build/_osprey.c"
for f in "$here/build/_osprey.c" "$here/template/types.c" "$here/template/door.c" "$repo/loader/loader.c"; do
  clang -std=c11 -O2 -fPIC -Wall -Wextra -Werror -I"$here/template" -I"$inc" -I"$wings_inc" -I"$repo/include" -I"$EAGLE_DOOR_DIR" -I"$py" -c "$f" -o "$here/build/$(basename "$f" .c).o"
done
case $(uname -s) in
  Darwin) shared=-dynamiclib; ext=dylib ;;
  *) shared=-shared; ext=so ;;
esac
if [ "$ext" = dylib ]; then
  clang $shared -o "$here/build/_osprey.abi3.so" "$here/build/_osprey.o" "$here/build/types.o" "$here/build/door.o" "$here/build/loader.o" "$lib" \
    -Wl,-exported_symbol,_PyInit__osprey -Wl,-dead_strip -Wl,-undefined,dynamic_lookup
  codesign -v "$here/build/_osprey.abi3.so" && echo "codesign -v ok"
else
  printf '{\n  global:\n    PyInit__osprey;\n  local:\n    *;\n};\n' > "$here/build/osprey.map"
  clang $shared -o "$here/build/_osprey.abi3.so" "$here/build/_osprey.o" "$here/build/types.o" "$here/build/door.o" "$here/build/loader.o" "$lib" \
    -Wl,--version-script="$here/build/osprey.map" -Wl,--gc-sections -lm -ldl
fi
# a library with the wrong abi number, for the entry's refusal test
clang $shared -fPIC -o "$here/build/libabi1.$ext" "$here/fixture/abi1.c"
clang $shared -fPIC -I"$inc" -I"$repo/include" -o "$here/build/libcontract0.$ext" "$here/fixture/contract0.c"
# the unit library of the door's own fixture, for the binding's tests of the door calls
clang $shared -fPIC -std=c11 -I"$EAGLE_DOOR_DIR" -o "$here/build/libfixture_unit.$ext" "$EAGLE_DOOR_DIR/eagle/door/test/fixture_unit.c"
"$gen" --check $hdrs --header "$repo/include/osprey_host.h" --door-header "$EAGLE_DOOR_DIR/eagle/door/eg_door.h" --door-source "$here/template/door.c" \
  --template "$here/template" --generated "$here/build/_osprey.c" --ext "$here/build/_osprey.abi3.so"
