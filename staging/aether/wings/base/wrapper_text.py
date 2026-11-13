# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The helper texts of an entry wrapper: the compensated own-column commit and the sample-sum row text."""

#: Compensated OWN-COLUMN commit: a Neumaier accumulate into the host's
#: prior value, one writer per sample, never atomic. The total is
#: ``target + companion``, so the host must zero both before the first term.
FINISH_COUNT = """
template <class ViewT>
{qualifier} void finish_count(const ViewT& counter)
{{
    // The finish epilogue's count (hawk.ir.nodes.FINISHED_PLANE): ONE
    // relaxed increment of a uint32 cell per NEWLY finished sample, at most N
    // over a whole run. Read only after the launch, in stream / team order.
    auto* cell = const_cast<std::uint32_t*>(
        reinterpret_cast<const std::uint32_t*>(counter.data()));
#if defined(__CUDA_ARCH__)
    atomicAdd(reinterpret_cast<unsigned int*>(cell), 1u);
#else
    std::atomic_ref<std::uint32_t>(*cell).fetch_add(1u, std::memory_order_relaxed);
#endif
}}
"""


#: The ``sample_sum`` row sum (:class:`raptor.ir.nodes.SampleSum`): the pair
#: combine, its host twin over one row of ``row`` slots (the variant's ``cells.row``; the template's holes ``{row}``, ``{half}``, ``{offsets}``), and the device's warp
#: butterfly over the same pairs. Needs ``twoSumErr_`` above.
SAMPLE_SUM = """
template <class E>
{qualifier} E ssExtreme_(int op, E a, E b)
{{
    // max (op 1) or min (op 2) that ignores a NaN operand, as hawk.math.max/min.
    if (a != a) return b;
    if (b != b) return a;
    return op == 1 ? (b > a ? b : a) : (b < a ? b : a);
}}

template <int OP, class E>
{qualifier} E ssIdentity_()
{{
    // What a slot past the last sample holds: 0, -inf, +inf.
    if constexpr (OP == 0) return E(0);
    else if constexpr (OP == 1) return -static_cast<E>(__builtin_inf());
    else return static_cast<E>(__builtin_inf());
}}

template <int OP = 0, class E>
{qualifier} void pairCombine_(E& s, E& c, E s2, E c2)
{{
    if constexpr (OP != 0) {{
        s = ssExtreme_(OP, s, s2);
        return;
    }}
    // Compensated combine of two partials (s, c) and (s2, c2), kept verbatim
    // with aether/accum/atomic.h. Symmetric: either side may hold either
    // partial. A non-finite sum drops its compensation, so Inf and NaN
    // follow IEEE through s.
    const E t = s + s2;
    const E e = aether::accum::twoSumErrFinite(s, s2, t);
    c = aether::accum::isFiniteBits(t) ? (c + c2) + e : E(0);
    s = t;
}}

template <int N, int OP = 0, class E, class Store>
{qualifier} void sampleSumRow_(E* terms, Store&& store)
{{
    // One row: {row} slots of N terms (row-major), slot l combined with slot
    // l + off for off = {offsets} -- the pairs of sampleSumWarp_ --
    // then store(column, sum, compensation). Leaves the slots zeroed.
    E comp[{row} * N] = {{}};
    for (int off = {half}; off >= 1; off >>= 1)
        for (int l = 0; l < off; ++l)
            for (int k = 0; k < N; ++k)
                pairCombine_<OP>(terms[l * N + k], comp[l * N + k],
                             terms[(l + off) * N + k], comp[(l + off) * N + k]);
    for (int k = 0; k < N; ++k) store(k, terms[k], comp[k]);
    for (int k = 0; k < {row} * N; ++k) terms[k] = ssIdentity_<OP, E>();
}}

#if defined(__CUDACC__)
template <class E>
__device__ __forceinline__ void ssPin_(E& x)
{{
    // A rounded term: a product in the body is never fused into a combine.
    if constexpr (sizeof(E) == 4) asm("" : "+f"(x));
    else asm("" : "+d"(x));
}}

template <int M, int OP, class E>
__device__ __forceinline__ void ssRound_(E* h, E* c, int off, bool up)
{{
    // One butterfly round over a lane's M columns: it keeps one contiguous
    // half (the upper one when `up`), adds its partner's (lane ^ off) copy
    // of that half, and ships the other half to the partner.
    constexpr int LO = M / 2, UP = M - LO;
#pragma unroll
    for (int j = 0; j < UP; ++j) {{
        const E sh = up ? (j < LO ? h[j] : ssIdentity_<OP, E>()) : h[LO + j];
        const E sc = up ? (j < LO ? c[j] : E(0)) : c[LO + j];
        const E rh = __shfl_xor_sync(0xffffffffu, sh, off);
        const E rc = __shfl_xor_sync(0xffffffffu, sc, off);
        if (up) pairCombine_<OP>(h[LO + j], c[LO + j], rh, rc);
        else if (j < LO) pairCombine_<OP>(h[j], c[j], rh, rc);
    }}
#pragma unroll
    for (int j = 0; j < UP; ++j) {{
        h[j] = up ? h[LO + j] : (j < LO ? h[j] : ssIdentity_<OP, E>());
        c[j] = up ? c[LO + j] : (j < LO ? c[j] : E(0));
    }}
}}

template <int N, int OP = 0, class E, class Store>
__device__ __forceinline__ void sampleSumWarp_(E (&h)[N], Store&& store)
{{
    // The warp's 32 lanes' terms summed column by column in five rounds
    // (lane ^ 16, 8, 4, 2, 1); each lane then stores the sums of the
    // contiguous run of columns it holds. Every lane of the warp calls it.
    E c[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {{
        ssPin_(h[k]);
        c[k] = E(0);
    }}
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    constexpr int M2 = N - N / 2, M3 = M2 - M2 / 2, M4 = M3 - M3 / 2;
    constexpr int M5 = M4 - M4 / 2, M6 = M5 - M5 / 2;
    ssRound_<N, OP>(h, c, 16, (lane & 16u) != 0u);
    ssRound_<M2, OP>(h, c, 8, (lane & 8u) != 0u);
    ssRound_<M3, OP>(h, c, 4, (lane & 4u) != 0u);
    ssRound_<M4, OP>(h, c, 2, (lane & 2u) != 0u);
    ssRound_<M5, OP>(h, c, 1, (lane & 1u) != 0u);
    // the columns a lane holds: the rounds' split points are M / 2 of the
    // padded counts above, so a lane may hold fewer than M (or none)
    const int halves[5] = {{N / 2, M2 / 2, M3 / 2, M4 / 2, M5 / 2}};
    int lo = 0, m = N;
#pragma unroll
    for (int r = 0; r < 5; ++r) {{
        if (lane & (16u >> r)) {{
            lo += halves[r];
            m -= halves[r];
        }} else if (m > halves[r]) {{
            m = halves[r];
        }}
    }}
#pragma unroll
    for (int j = 0; j < M6; ++j)
        if (j < m) store(lo + j, h[j], c[j]);
}}
#endif
"""
