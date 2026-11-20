#!/bin/sh
# The rules an OSPREY library's exports must satisfy, as functions over plain lists (one name per line on standard input) so that
# ci/export_rules_test.sh can feed them synthetic lists. Sourced by check_exports.sh and ci/dist.sh; each function prints what it
# refuses and prints nothing, with status 0, for a clean list.

# The global symbols a library defines, one name per line, without the leading underscore of macOS.
defined_names() {
  if nm --version 2>/dev/null | grep -q GNU; then nm -g --defined-only "$1"; else nm -gU "$1"; fi |
    awk 'NF >= 3 { sub(/^_/, "", $3); print $3 }' | sort
}

# The shared libraries a library needs, one file name per line.
needed_names() {
  if [ "$(uname -s)" = Darwin ]; then
    otool -L "$1" | awk 'NR > 1 { n = split($1, part, "/"); print part[n] }'
  else
    readelf -d "$1" | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'
  fi
}

# The symbols a library leaves undefined, one name per line.
undefined_names() { nm -u "$1" | awk '{ sub(/^_/, "", $NF); print $NF }'; }

# Whether a library has the device runtime: it defines the door's first function.
door_mode() {
  if grep -qx 'eg_ctx_new'; then echo "with the device runtime"; else echo "without the device runtime"; fi
}

# The names of the list on standard input that are not in the wanted list $1, and the wanted names that are not in the list.
export_differences() {
  { sort -u; printf '%s\n' "$1" | sort -u; } | sort | uniq -u
}

# The OpenMP runtime's symbols, defined or needed: no backend library brings a threading runtime into the process that loads it.
openmp_symbols() { grep -E '^_*(GOMP_|omp_|kmpc_)' || true; }

# The needed libraries that must stay out of an OSPREY library: an OpenMP runtime, and the CUDA runtime and libraries, which are
# linked statically and kept private or loaded by the door at use.
forbidden_needed() { grep -E '^lib(gomp|omp|iomp|cudart|cufft|cublas|nvrtc)' || true; }

# The defined symbols that belong to the statically linked CUDA runtime or the driver API: they must stay private.
cuda_symbols() { grep -E '^(cuda|cu[A-Z])' || true; }
