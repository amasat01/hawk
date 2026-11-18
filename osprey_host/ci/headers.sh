#!/bin/sh
# Prunes the header trees a library embeds to the headers its compiles include.
# Usage: sh ci/headers.sh <cpu|cuda> <source folders, colon separated> <aether checkout> <eagle checkout> <raptor wings cards folder> <out folder>
# Writes <out>/aether and <out>/eagle (include roots holding aether/ and eagle/ + plugin/) and <out>/facts (the library's
# raptor_facts.h), and prints "<tree> <files> <bytes>" for each. The entry headers are the #include lines of the sources that write the texts (the writers' and the lowering's)
# (the device texts only for cuda); every header they reach by #include is copied, whatever condition guards the line.
set -e
kind=$1; sources=$2; aether=$3; eagle=$4; cards=$5; out=$6
[ -n "$out" ] || { echo "usage: headers.sh <cpu|cuda> <writers src> <aether> <eagle> <cards> <out>" >&2; exit 2; }
rm -rf "$out"; mkdir -p "$out/aether" "$out/eagle" "$out/facts"
queue=$(mktemp); seen=$(mktemp); next=$(mktemp)
trap 'rm -f "$queue" "$seen" "$next"' EXIT
if [ "$kind" = cpu ]; then
  echo "$sources" | tr ':' '\n' | xargs -I{} find {} -name '*.rs' -not -path '*/unit/device/*' -print0 | xargs -0 cat
else
  echo "$sources" | tr ':' '\n' | xargs -I{} find {} -name '*.rs' -print0 | xargs -0 cat
fi | sed 's/\\"/"/g' | grep -oE '#include [<"](aether|eagle|plugin)/[^>"]+[>"]' | sed -E 's/#include [<"]([^>"]+)[>"]/\1/' | sort -u > "$queue"
# the root that holds a header named relative to an include root
find_root() { for r in "$aether" "$eagle"; do [ -f "$r/$1" ] && { echo "$r"; return 0; }; done; return 1; }
while [ -s "$queue" ]; do
  : > "$next"
  while read -r rel; do
    grep -qxF "$rel" "$seen" && continue
    root=$(find_root "$rel") || { echo "headers.sh: $rel is in neither checkout" >&2; exit 1; }
    echo "$rel" >> "$seen"
    case $root in "$aether") dest=$out/aether ;; *) dest=$out/eagle ;; esac
    mkdir -p "$dest/$(dirname "$rel")"; cp "$root/$rel" "$dest/$rel"
    dir=$(dirname "$rel")
    grep -oE '#[ ]*include[ ]*[<"][^>"]+[>"]' "$root/$rel" | sed -E 's/#[ ]*include[ ]*[<"]([^>"]+)[>"]/\1/' | while read -r inc; do
      if find_root "$inc" >/dev/null; then echo "$inc"; elif find_root "$dir/$inc" >/dev/null; then echo "$dir/$inc"; fi
    done >> "$next"
  done < "$queue"
  sort -u "$next" > "$queue"
done
cp "$cards/raptor_facts.h" "$out/facts/"
for tree in aether eagle facts; do
  files=$(find "$out/$tree" -type f | wc -l | tr -d ' ')
  bytes=$(find "$out/$tree" -type f -print0 | xargs -0 cat | wc -c | tr -d ' ')
  echo "$tree $files $bytes"
done
