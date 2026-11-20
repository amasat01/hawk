#!/bin/sh
# Table test of ci/export_rules.sh on synthetic lists: each forbidden item is caught and a clean list passes. Prints one line and
# exits 0 when every row holds.
cd "$(dirname "$0")" || exit 2
. ./export_rules.sh
failed=0; rows=0
# expect <what> <function> <input> <expected output>
expect() {
  rows=$((rows + 1))
  got=$(printf '%s' "$3" | "$2" | tr '\n' ' ' | sed 's/ $//')
  [ "$got" = "$4" ] || { echo "FAIL: $1: $2 gave [$got], wanted [$4]"; failed=1; }
}
expect "a clean needed list" forbidden_needed 'libstdc++.so.6
libm.so.6
libgcc_s.so.1
libc.so.6
ld-linux-x86-64.so.2' ''
for lib in libgomp.so.1 libomp.so.5 libiomp5.so libcudart.so.12 libcudart_static.so libcufft.so.11 libcublas.so.12 libcublasLt.so.12 libnvrtc.so.12; do
  expect "needed $lib" forbidden_needed "libc.so.6
$lib" "$lib"
done
expect "a clean defined list" cuda_symbols 'osprey_v1
eg_ctx_new
eg_status_text' ''
for sym in cudaMalloc cudaGetDeviceCount cuInit cuCtxCreate; do
  expect "defined $sym" cuda_symbols "eg_ctx_new
$sym" "$sym"
done
expect "a name that only contains cu" cuda_symbols 'eg_cufft_plan
secure' ''
expect "no OpenMP symbol in a clean list" openmp_symbols 'memcpy
pthread_create
comp_x
compute_62' ''
for sym in GOMP_parallel omp_get_num_threads __kmpc_fork_call _GOMP_barrier; do
  expect "OpenMP symbol $sym" openmp_symbols "memcpy
$sym" "$sym"
done
expect "mode with the door" door_mode 'osprey_v1
eg_ctx_new' 'with the device runtime'
expect "mode without the door" door_mode 'osprey_v1' 'without the device runtime'
expect "mode with a partial door" door_mode 'osprey_v1
eg_status_text' 'without the device runtime'
rows=$((rows + 1))
got=$(printf 'osprey_v1\neg_a\nstray\n' | export_differences 'osprey_v1
eg_a
eg_b' | tr '\n' ' ')
[ "$got" = "eg_b stray " ] || { echo "FAIL: export_differences gave [$got], wanted [eg_b stray ]"; failed=1; }
rows=$((rows + 1))
[ -z "$(printf 'osprey_v1\neg_a\n' | export_differences 'osprey_v1
eg_a')" ] || { echo "FAIL: export_differences of equal lists is not empty"; failed=1; }
[ "$failed" -eq 0 ] && echo "export rules: $rows rows hold"
exit "$failed"
