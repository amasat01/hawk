#!/bin/sh
# Builds each OSPREY and puts its library, manifest and data files in one folder: dist/osprey_cpu and dist/osprey_cuda.
# Usage: sh ci/dist.sh   (from the repository root; the build needs the OSPREY_*_HEADERS variables the gate exports)
# osprey.json is the manifest source of the library with the door number (include/osprey.h, the one source of it) and the version
# of ci/version.sh put in. A manifest source may hold a line `@RUNTIME@`: it becomes the runtime member when the built library
# carries the runtime door (it defines eg_ctx_new), else the line is dropped, so the manifest states what the library has.
set -e
cd "$(dirname "$0")/.."
. ./ci/export_rules.sh
JOBS=${JOBS:-2}
case $(uname -s) in Darwin) ext=dylib ;; *) ext=so ;; esac
door=$(sed -n 's/^#define OB_ABI \([0-9]*\).*/\1/p' include/osprey.h)
version=$(sh ci/version.sh)
for name in osprey_cpu osprey_cuda; do
  cargo build --release -j "$JOBS" -p ${name}_door
  folder=dist/$name
  rm -rf "$folder"; mkdir -p "$folder"
  cp target/release/lib$name.$ext "$folder/"
  cp -R $name/facts "$folder/facts"
  if [ "$(defined_names "$folder/lib$name.$ext" | door_mode)" = "with the device runtime" ]; then
    runtime='s/@RUNTIME@/"runtime": "eagle-door\/1",/'
  else
    runtime='/@RUNTIME@/d'
  fi
  sed -e "s/@DOOR@/$door/" -e "s/@VERSION@/$version/" -e "$runtime" $name/osprey.json > "$folder/osprey.json"
done
