# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The helper texts of an entry wrapper: the compensated own-column commit and the sample-sum row text."""

#: The host arm of each helper that has a device arm (:data:`aether.wings.simt.wrapper_text.ARMS`): the text a host unit carries.
HOST_ARMS = {
    "accum_add": "    std::atomic_ref<E>(slot).fetch_add(static_cast<E>(term), std::memory_order_relaxed);",
    "accum_add_compensated": "\n".join((
            "    const E old = std::atomic_ref<E>(slot).fetch_add(t, std::memory_order_relaxed);",
            "    const E s = old + t;",
            "    std::atomic_ref<E>(compSlot).fetch_add(aether::accum::twoSumErrAtomic(old, t, s),",
            "                                           std::memory_order_relaxed);",
        )),
    "accum_or": "    std::atomic_ref<E>(slot).fetch_or(static_cast<E>(bits), std::memory_order_relaxed);",
    "finish_count": "    std::atomic_ref<std::uint32_t>(*cell).fetch_add(1u, std::memory_order_relaxed);",
    "mix_count": "\n".join((
            "    (void)K;",
            "    std::atomic_ref<std::uint32_t>(cells[k]).fetch_add(1u, std::memory_order_relaxed);",
        )),
    "spacing": "    const T up = std::nextafter(a, a * T(2) + T(1)) - a;",
}

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
{arm}
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

"""
