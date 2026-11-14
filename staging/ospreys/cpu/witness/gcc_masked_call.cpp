// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// Witness for the GCC if-conversion bug fixed by abbfe1e2616d ("middle-end: Fix ifcvt predicate generation for masked
// function calls"): GCC 13.1-13.3 and 14.1-14.2 mask a call under a condition with the wrong (un-inverted) predicate
// when the callee has masked (inbranch) SIMD clones. The loop is a lane-tile early-exit iteration with a select on the
// call's result, the shape hawk emits. Prints OK or MISMATCH; exit 1 on wrong results.
// Build: g++ -O3 -ffp-contract=fast -fno-trapping-math -march=native -mprefer-vector-width=512 hawk/compile/witness/gcc_masked_call.cpp
#include <cmath>
#include <cstdint>
#include <cstdio>

// GNU simd attribute: both masked and unmasked vector clones are generated from this definition
__attribute__((noinline, simd)) double fn(double t) { return std::tanh(t); }

constexpr long N = 64, TILE = 32;

__attribute__((noinline)) void run(const double* g, const double* x0, double lim, double* y)
{
    for (long lo = 0; lo < N; lo += TILE) {
        alignas(64) double ron[TILE], sL[TILE], tL[TILE], gL[TILE];
        for (long t = 0; t < TILE; ++t) { sL[t] = tL[t] = x0[lo + t]; gL[t] = g[lo + t]; ron[t] = 1.0; }
        for (int trip = 0; trip < 12; ++trip) {
            std::int64_t live = 0;
#pragma GCC ivdep
            for (long l = 0; l < TILE; ++l) {
                const bool on = ron[l] != 0.0;
                const double s = sL[l], t = tL[l];
                const double tb2 = s * 1.1 + gL[l] * t;
                const bool stop = tb2 > lim;
                const double th = fn(t);
                const double n1 = stop ? t : th + 0.1;
                sL[l] = on ? tb2 : sL[l];
                tL[l] = on ? n1 : tL[l];
                const bool more = on && !stop;
                ron[l] = more ? 1.0 : 0.0;
                live |= more ? 1 : 0;
            }
            if (!live) break;
        }
        for (long t = 0; t < TILE; ++t) y[lo + t] = sL[t] + tL[t];
    }
}

static double ref(double g, double x0, double lim)
{
    double s = x0, t = x0;
    for (int trip = 0; trip < 12; ++trip) {
        const double tb2 = s * 1.1 + g * t;
        s = tb2;
        if (tb2 > lim) break;
        t = std::tanh(t) + 0.1;
    }
    return s + t;
}

int main()
{
    double g[N], x0[N], y[N];
    for (long i = 0; i < N; ++i) { g[i] = 0.2 + 0.03 * (i % 11); x0[i] = 0.05 * (i % 7) - 0.1; }
    run(g, x0, 2.0, y);
    int bad = 0;
    for (long i = 0; i < N; ++i)
        if (!(std::fabs(y[i] - ref(g[i], x0[i], 2.0)) <= 1e-13 * (1.0 + std::fabs(y[i])))) ++bad;
    std::printf("%s\n", bad ? "MISMATCH" : "OK");
    return bad ? 1 : 0;
}
