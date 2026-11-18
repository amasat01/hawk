#!/bin/sh
# Prunes the header trees of both libraries into <out>/<cpu|cuda>/<aether|eagle|facts>, printing the files and bytes of each.
# Usage: sh ci/prepare_headers.sh <out> <aether checkout> <eagle checkout>   (run from the repository root)
set -e
out=$1; aether=$2; eagle=$3
writers=../aether/rs/writers/src:components/lowering/src; cards=../raptor/rs/wings/cards
for kind in cpu cuda; do
  echo "== $kind (files, bytes)"
  sh ci/headers.sh $kind "$writers" "$aether" "$eagle" "$cards/$kind" "$out/$kind"
done
# The runtime compiler has no system C++ headers: the cuda library serves it aether's shims of them under their bare standard names
# and eagle's layout header under the name the unit includes. The offline compiler never sees this tree (it would shadow the host's
# own standard headers), so it is no include root of any other tree.
echo "== cuda runtime shims (files, bytes)"
rtc=$out/cuda/rtc
rm -rf "$rtc"; mkdir -p "$rtc/plugin"
for shim in cstddef cstdint type_traits utility concepts cmath atomic; do
  [ -f "$aether/rtc/$shim" ] || { echo "prepare_headers.sh: $aether/rtc/$shim is missing" >&2; exit 1; }
  cp "$aether/rtc/$shim" "$rtc/$shim"
done
cp "$eagle/plugin/gref_layout.h" "$rtc/plugin/gref_abi.h"
echo "rtc $(find "$rtc" -type f | wc -l | tr -d ' ') $(find "$rtc" -type f -print0 | xargs -0 cat | wc -c | tr -d ' ')"
