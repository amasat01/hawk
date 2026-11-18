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
