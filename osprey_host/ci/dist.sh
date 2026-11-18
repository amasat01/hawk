#!/bin/sh
# Builds each OSPREY and puts its library, manifest and data files in one folder: dist/osprey_cpu and dist/osprey_cuda.
# Usage: sh ci/dist.sh   (from the repository root; the build needs the OSPREY_*_HEADERS variables the gate exports)
# osprey.json is the manifest source of the library with the door number (include/osprey.h, the one source of it) and the version
# of the workspace put in.
set -e
cd "$(dirname "$0")/.."
JOBS=${JOBS:-2}
case $(uname -s) in Darwin) ext=dylib ;; *) ext=so ;; esac
door=$(sed -n 's/^#define OB_ABI \([0-9]*\).*/\1/p' include/osprey.h)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
for name in osprey_cpu osprey_cuda; do
  cargo build --release -j "$JOBS" -p ${name}_door
  folder=dist/$name
  rm -rf "$folder"; mkdir -p "$folder"
  cp target/release/lib$name.$ext "$folder/"
  cp -R $name/facts "$folder/facts"
  sed -e "s/@DOOR@/$door/" -e "s/@VERSION@/$version/" $name/osprey.json > "$folder/osprey.json"
done
