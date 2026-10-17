// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// Drives the emitted second-order drag kernel (tests/test_host_stale_stack.py) with zero tangent
// planes, after dirtying the stack with 1e200: the exact result is 0 for every sample, so any
// nonzero output is a stale stack read. usage: harness <kernel.so> <n> <shared 0|1>
#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct Plane { double* data; std::uint64_t samples, compStride, sampleStride; std::int32_t type, id; };
struct Mask { void* data; std::uint64_t samples, stride; std::int32_t type, id; };
using Kernel = void (*)(void* const*, std::int64_t, std::int64_t, std::int64_t);

static double* filled(int len, double val)
{
    double* p = static_cast<double*>(std::aligned_alloc(64, 4096));
    for (int i = 0; i < 512; ++i) p[i] = i < len ? val : -1e100;
    return p;
}

int main(int argc, char** argv)
{
    if (argc != 4) return 2;
    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) return 2;
    Kernel kernel = reinterpret_cast<Kernel>(dlsym(lib, "drag_jj_host"));
    if (!kernel) return 2;
    const int n = std::atoi(argv[2]);
    const bool shared = std::atoi(argv[3]) != 0;
    double* v = filled(3 * n, 0);
    for (int i = 0; i < 3 * n; ++i) v[i] = std::sin(1.3 * i + 0.4) * 1.7;
    double* out = filled(3 * n, 777.0);
    double* t0 = filled(3 * n, 0);
    double* t1 = shared ? t0 : filled(3 * n, 0);
    double* t2 = shared ? t0 : filled(3 * n, 0);
    unsigned char* terminated = static_cast<unsigned char*>(std::aligned_alloc(64, 4096));
    for (int i = 0; i < 4096; ++i) terminated[i] = 0;
    const std::uint64_t un = n;
    Plane go{out, un, un, 1, 1, 0}, p1{t0, un, un, 1, 1, 0}, p2{t1, un, un, 1, 1, 0},
        p3{t2, un, un, 1, 1, 0}, pv{v, un, un, 1, 1, 0};
    Mask term{terminated, un, 1, 1, 0};
    double cd = 0.3, dot2Cd = 1.0, dot2DotCd = 0.0, dotCd = 1.0;
    void* params[10] = { &go, &p1, &p2, &p3, &pv, &term, &cd, &dot2Cd, &dot2DotCd, &dotCd };
    {
        volatile double junk[4096];
        for (int i = 0; i < 4096; ++i) junk[i] = 1e200;
        (void)junk[7];
    }
    kernel(params, 0, n, n);
    double mx = 0;
    for (int i = 0; i < 3 * n; ++i) mx = std::fmax(mx, std::fabs(out[i]));
    std::printf("n=%d shared=%d max|out|=%.3g (exact 0)\n", n, shared, mx);
    return mx != 0;
}
