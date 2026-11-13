# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ABI prelude of an emitted translation unit: the include block, the helper texts
(integer, compensated store, accumulate, spacing, mix, tiled, fold) and the functions that
assemble them into :func:`prelude`. The unit driver is :mod:`aether.wings.base.wrapper`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raptor.errors import RaptorError
from .wrapper_text import FINISH_COUNT, SAMPLE_SUM

if TYPE_CHECKING:
    from .wrapper import ScalarMode


#: The headers every emitted TU includes, in the mandated order: aether
#: first (defines ``AETHER_INDEX_T``), eagle's ABI header second, so
#: ``EAGLE_ABI_INDEX_T`` binds to the aether index width.
UNIT_INCLUDES = """#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <aether/expr/nodes/Geometric.h>
#include <aether/expr/nodes/Inverse.h>
#include <aether/expr/nodes/Product.h>
#include <aether/expr/nodes/Quaternion.h>
#include <aether/expr/nodes/Structural.h>
#include <aether/expr/Reduce.h>
#include <aether/math/math.h>

#include "plugin/gref_layout.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
"""

#: The one extra header a unit with a random op needs — the narrowest one
#: that compiles standalone, not the ``random.h`` umbrella, which drags in
#: a stateful API this op never calls. :func:`prelude` splices it in only
#: when ``needs_random`` is true, so a unit with none is unaffected.
_RANDOM_INCLUDE = '#include "aether/random/detail/Backend.h"\n'
_RANDOM_INCLUDE_MARKER = "#include <aether/math/math.h>\n"

#: The run commits' header, spliced after the same marker only for a body
#: that calls one (:func:`run_states`).
_RUN_INCLUDE = "#include <aether/accum/run.h>\n"
_TWO_SUM_INCLUDE = "#include <aether/accum/TwoSum.h>\n"

#: The linear-algebra header, spliced after the same marker only for a body
#: that calls it.
_LINALG_CALL = "aether::linalg::"
_LINALG_INCLUDE = "#include <aether/linalg/linalg.h>\n"
#: The variant's generated header (``variants/<name>/raptor_variant.h``, its directory on the compile line): linalg and the ray
#: primitives read ``RAPTOR_LINALG_MAX_DIM`` from it, so it comes first in a unit that includes either.
_VARIANT_INCLUDE = '#include "raptor_variant.h"\n'
_TEAM_INCLUDE = "#include <aether/linalg/team.h>\n"

#: The ray primitives' header, likewise.
_RAY_CALL = "aether::geometry::"
_RAY_INCLUDE = "#include <aether/geometry/geometry.h>\n"

#: The rolled-loop twins' header (:mod:`raptor.ir.fold`), likewise.
_RT_CALL = "aether::rt::"
_RT_INCLUDE = "#include <aether/expr/Rt.h>\n"
#: The two-sum residual (aether's) both compensated commits below share — pure
#: arithmetic (no atomics), so one definition serves the scattered and
#: own-column forms alike. Spliced whenever either needs it, never twice.
_TWO_SUM_ERR = """
template <class E>
{qualifier} E twoSumErr_(E a, E b, E s)
{{
    // aether's ONE two-sum residual: exact for the rounded a, b and s, NaN
    // for a non-finite s, +0 for a zero residual (aether/accum/TwoSum.h).
    return aether::accum::twoSumErr(a, b, s);
}}
"""

#: Compensated SCATTERED commit: a Knuth two-sum residual under the
#: target's atomic add, made atomic on both lanes on both targets, closing
#: the same race ``accum_add`` below closes for ``plain``/``atomic``.
_ACCUM_ADD_COMPENSATED = """
template <class ViewT, class T>
{qualifier} void accum_add_compensated(ViewT& target, ViewT& comp,
                                       const aether::SampleIndex& at, T term)
{{
    auto& slot = target(at.global());
    auto& compSlot = comp(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add_compensated: a scattered compensated "
                  "accumulate commits through the target's atomic add, and "
                  "only a floating accumulate plane has one");
    E t = static_cast<E>(term);
    AETHER_FP_BARRIER(t);  // rounded once: never fused into `old + t`
#if defined(__CUDA_ARCH__)
    const E old = atomicAdd(&slot, t);
    const E s = old + t;
    atomicAdd(&compSlot, aether::accum::twoSumErrAtomic(old, t, s));
#else
    const E old = std::atomic_ref<E>(slot).fetch_add(t, std::memory_order_relaxed);
    const E s = old + t;
    std::atomic_ref<E>(compSlot).fetch_add(aether::accum::twoSumErrAtomic(old, t, s),
                                           std::memory_order_relaxed);
#endif
}}
"""

#: The scattered ``contract="or"`` commit: an atomic OR of a 32-bit word into the
#: target's slot, the device's own ``atomicOr`` or an ``atomic_ref`` under the
#: OpenMP host team. Spliced only into a unit whose body calls it.
_ACCUM_OR = """
template <class ViewT, class T>
{qualifier} void accum_or(ViewT& target, const aether::SampleIndex& at, T bits)
{{
    auto& slot = target(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_integral_v<E> && sizeof(E) == 4,
                  "hawk_abi::accum_or: a scattered OR commits through the "
                  "target's atomic OR, on a 32-bit integer plane");
#if defined(__CUDA_ARCH__)
    atomicOr(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(bits));
#else
    std::atomic_ref<E>(slot).fetch_or(static_cast<E>(bits), std::memory_order_relaxed);
#endif
}}
"""

#: The fold of an owned companion into its target (:func:`fold_entries`):
#: the two-sum of the pair, the rounded sum into the target and its exact
#: residual into the companion, none for a non-finite sum. One lane per slot.
_FOLD_COMPANION = """
template <class ViewT>
{qualifier} void fold_companion(ViewT& target, ViewT& comp, const aether::SampleIndex& at)
{{
    auto& t = target(at.global());
    auto& c = comp(at.global());
    using E = std::remove_reference_t<decltype(t)>;
    const E s = t + c;
    const E e = twoSumErr_(t, c, s);
    t = s;
    c = aether::accum::isFiniteBits(s) ? e : E(0);
}}
"""









_STORE_COMPENSATED = """
template <class ViewT, class CompT, class T>
{qualifier} void store_compensated(ViewT& target, CompT& comp,
                                   const aether::SampleIndex& at, T term)
{{
    using E = typename ViewT::element_type;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::store_compensated: an own-column compensated "
                  "commit accumulates into a floating target plane only");
    if constexpr (ViewT::extents_type::Rank == 1) {{
        // a Scalar own-column target: ONE (component, sample) pair
        // collapses to a plain per-sample index.
        auto& slot = target(at.global());
        auto& compSlot = comp(at.global());
        const E old = slot;
        E t = static_cast<E>(term);
        AETHER_FP_BARRIER(t);  // rounded once: never fused into `old + t`
        const E s = old + t;
        slot = s;
        compSlot = compSlot + twoSumErr_(old, t, s);
    }} else {{
        static_assert(ViewT::extents_type::Rank == 2,
                      "hawk_abi::store_compensated: built for a Scalar or "
                      "Vector[N] own-column target only");
        // a Vector[N] own-column target: the SAME twoSumErr_ recurrence,
        // per component -- `target`/`comp` are (component, sample) views.
        // `term` is any same-width expression over per-sample values (a
        // bare `Item`, or arithmetic such as `(-k * norm(v)) * v`); it is
        // evaluated ONCE into an `aether::Item` so each component can be
        // read by index, and component `k` of the target accumulates
        // against component `k` of the term alone.
        constexpr std::size_t N = ViewT::extents_type::static_extent(0);
        const aether::Item<typename T::element_type, N> termItem(term);
        for (std::size_t k = 0; k < N; ++k) {{
            auto& slot = target(k, at.global());
            auto& compSlot = comp(k, at.global());
            const E old = slot;
            E t = static_cast<E>(termItem(k));
            AETHER_FP_BARRIER(t);
            const E s = old + t;
            slot = s;
            compSlot = compSlot + twoSumErr_(old, t, s);
        }}
    }}
}}
"""


# A runtime trip count as a 32-bit loop bound: counts past the int range
# clamp to its top, which no loop reaches in practice.
#: The ``spacing`` op's helper: one ulp of ``x``, signed like it
#: (``np.spacing``) — ``nextafter`` away from zero, minus ``x``. Emitted only
#: for a body that calls it, so every other unit's prelude is unchanged.
_SPACING_CALL = "hawk_abi::spacing("
_SPACING = """
template <class T>
{qualifier} T spacing(T x)
{{
    const T a = x < T(0) ? -x : x;
#if defined(__CUDA_ARCH__)
    const T up = ::nextafter(a, a * T(2) + T(1)) - a;
#else
    const T up = std::nextafter(a, a * T(2) + T(1)) - a;
#endif
    return x < T(0) ? -up : up;
}}
"""

#: An ``auto`` dispatch's branch-mix count: +1 in counter ``k`` per live
#: sample that picked branch ``k``. On the device one warp ballot per branch
#: and one atomic per warp (the leader adds the popcount); on the host one
#: relaxed atomic per sample. Integers only, so the counts are exact and
#: independent of scheduling. Emitted only for a body that counts.
_MIX_CALL = "hawk_abi::mix_count("
_MIX_COUNT = """
template <class ViewT>
{qualifier} void mix_count(const ViewT& mix, int K, int k)
{{
    auto* cells = const_cast<std::uint32_t*>(
        reinterpret_cast<const std::uint32_t*>(mix.data()));
#if defined(__CUDA_ARCH__)
    const unsigned active = __activemask();
    unsigned below;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(below));
    const bool leader = (active & below) == 0u;
    for (int j = 0; j < K; ++j) {{
        const unsigned picked = __ballot_sync(active, k == j);
        if (leader && picked != 0u)
            atomicAdd(reinterpret_cast<unsigned int*>(cells + j),
                      static_cast<unsigned int>(__popc(picked)));
    }}
#else
    (void)K;
    std::atomic_ref<std::uint32_t>(cells[k]).fetch_add(1u, std::memory_order_relaxed);
#endif
}}
"""

#: The integer helpers (``int32``/``uint32`` planes, and ``//``, ``%``, the
#: bit operators on an index ``Int``): numpy's results at every edge C++
#: leaves undefined or defines otherwise — wrapping ``+ - *`` and negation
#: (through the unsigned twin: signed overflow is undefined), floor ``//``
#: and divisor-signed ``%`` with a zero divisor giving 0, shifts by a count
#: outside ``[0, bits)`` giving 0 (or -1 for a negative value shifted
#: right), and float -> integer truncating toward zero, saturating at the
#: range and mapping NaN to 0. No ``<limits>``: the device compiler may not
#: carry it. Emitted only for a body that calls one.
_INT_CALL = "hawk_abi::int_"
_OR_CALL = "hawk_abi::accum_or("
_INT_HELPERS = """
template <class T> struct int_traits;
template <> struct int_traits<std::int32_t> {{
    using U = std::uint32_t;
    static constexpr bool is_signed = true;
    static constexpr std::int32_t lo = -2147483647 - 1, hi = 2147483647;
    static constexpr unsigned bits = 32u;
}};
template <> struct int_traits<std::uint32_t> {{
    using U = std::uint32_t;
    static constexpr bool is_signed = false;
    static constexpr std::uint32_t lo = 0u, hi = 4294967295u;
    static constexpr unsigned bits = 32u;
}};
template <> struct int_traits<long long> {{
    using U = unsigned long long;
    static constexpr bool is_signed = true;
    static constexpr long long lo = -9223372036854775807LL - 1, hi = 9223372036854775807LL;
    static constexpr unsigned bits = 64u;
}};

template <class T> {qualifier} T int_add(T a, T b)
{{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
}}
template <class T> {qualifier} T int_sub(T a, T b)
{{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
}}
template <class T> {qualifier} T int_mul(T a, T b)
{{
    using U = typename int_traits<T>::U;
    return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}}
template <class T> {qualifier} T int_neg(T a)
{{
    using U = typename int_traits<T>::U;
    return static_cast<T>(U(0) - static_cast<U>(a));
}}
template <class T> {qualifier} T int_abs(T a)
{{
    if constexpr (int_traits<T>::is_signed)
        return a < T(0) ? int_neg(a) : a;
    else
        return a;
}}
template <class T> {qualifier} T int_min(T a, T b) {{ return b < a ? b : a; }}
template <class T> {qualifier} T int_max(T a, T b) {{ return a < b ? b : a; }}
// One unsigned compare is the whole in-range test (a negative index wraps past n), so an in-range read pays one
// predictable branch: a min/max or nested-compare spelling costs more per read on the host gather loop.
template <class T> {qualifier} T int_clamp_index(T i, T n)
{{
    using U = typename int_traits<T>::U;
    return U(i) < U(n) ? i : (i < T(0) ? T(0) : n - T(1));
}}
template <class T> {qualifier} T int_floordiv(T a, T b)
{{
    if (b == T(0))
        return T(0);
    if constexpr (int_traits<T>::is_signed) {{
        if (b == T(-1))
            return int_neg(a);
        const T q = static_cast<T>(a / b);
        return (a % b != T(0) && ((a < T(0)) != (b < T(0)))) ? static_cast<T>(q - T(1)) : q;
    }} else {{
        return static_cast<T>(a / b);
    }}
}}
template <class T> {qualifier} T int_mod(T a, T b)
{{
    if (b == T(0))
        return T(0);
    if constexpr (int_traits<T>::is_signed) {{
        if (b == T(-1))
            return T(0);
        const T r = static_cast<T>(a % b);
        return (r != T(0) && ((r < T(0)) != (b < T(0)))) ? static_cast<T>(r + b) : r;
    }} else {{
        return static_cast<T>(a % b);
    }}
}}
template <class T> {qualifier} T int_and(T a, T b) {{ return static_cast<T>(a & b); }}
template <class T> {qualifier} T int_or(T a, T b) {{ return static_cast<T>(a | b); }}
template <class T> {qualifier} T int_xor(T a, T b) {{ return static_cast<T>(a ^ b); }}
template <class T> {qualifier} T int_invert(T a) {{ return static_cast<T>(~a); }}
template <class T> {qualifier} T int_shl(T a, T b)
{{
    using U = typename int_traits<T>::U;
    if (static_cast<U>(b) < U(int_traits<T>::bits))
        return static_cast<T>(static_cast<U>(a) << static_cast<U>(b));
    return T(0);
}}
template <class T> {qualifier} T int_shr(T a, T b)
{{
    using U = typename int_traits<T>::U;
    if (static_cast<U>(b) < U(int_traits<T>::bits))
        return static_cast<T>(a >> b);
    if constexpr (int_traits<T>::is_signed)
        return a < T(0) ? T(-1) : T(0);
    else
        return T(0);
}}
template <class T, class R> {qualifier} T int_from_real(R x)
{{
    if (aether::math::isnan(x))
        return T(0);
    if (x <= static_cast<R>(int_traits<T>::lo))
        return int_traits<T>::lo;
    if (x >= static_cast<R>(int_traits<T>::hi))
        return int_traits<T>::hi;
    return static_cast<T>(x);
}}
"""

_CLAMP_COUNT = """{qualifier} int hawk_clamp_count(long long n)
{{
    return n < 2147483647LL ? static_cast<int>(n) : 2147483647;
}}

"""


def abi_helpers(qualifier: str, *, device_text=None, compensated: bool = False,
                store_compensated: bool = False, finish: bool = False,
                spacing: bool = False, mix: bool = False,
                integers: bool = False, sample_sum: bool = False,
                fold: bool = False, scatter_or: bool = False, tiled: bool = False, warp_sum: bool = False,
                cells: bool = False, row: int | None = None) -> str:
    """The view-from-mirror reconstruction helpers — byte-for-byte the same
    on both targets, only the function qualifier differs; reads the
    mirror's own stride fields so a non-contiguous plane rebuilds
    correctly.

    ``compensated``/``store_compensated`` each append their own helper,
    emitted only for the unit whose resolved target needs it
    (:func:`hawk.ext.compensated_target`), so a PLAIN unit's prelude stays
    byte-identical. ``finish`` appends ``finish_count`` under the same
    rule, for a kernel that finishes its own sample, and ``spacing`` the
    ``spacing`` op's helper for a body that uses it, ``integers`` the
    integer helpers for a body that calls one, ``sample_sum`` the row sum
    for a kernel with a :class:`~raptor.ir.nodes.SampleSum` sink."""
    helpers = []
    if compensated or store_compensated or sample_sum or fold:
        helpers.append(_TWO_SUM_ERR.format(qualifier=qualifier))
    if fold:
        helpers.append(_FOLD_COMPANION.format(qualifier=qualifier))
    if sample_sum:
        if cells:
            helpers.append(device_text.BANDED_SUM.format(qualifier=qualifier))
        if row is None:
            raise RaptorError("the sample-sum row helper is emitted with the variant's cells.row, and none was given")
        halves = [row >> k for k in range(1, row.bit_length())]       # the pairs of one row: row/2, ..., 2, 1
        helpers.append((device_text.banded_sample_sum(SAMPLE_SUM) if cells else SAMPLE_SUM).format(
            qualifier=qualifier, row=row, half=row // 2, offsets=", ".join(map(str, halves))))
    if compensated:
        helpers.append(_ACCUM_ADD_COMPENSATED.format(qualifier=qualifier))
    if store_compensated:
        helpers.append(_STORE_COMPENSATED.format(qualifier=qualifier))
    if finish:
        helpers.append(FINISH_COUNT.format(qualifier=qualifier))
    if spacing:
        helpers.append(_SPACING.format(qualifier=qualifier))
    if mix:
        helpers.append(_MIX_COUNT.format(qualifier=qualifier))
    if integers:
        helpers.append(_INT_HELPERS.format(qualifier=qualifier))
    if scatter_or:
        helpers.append(_ACCUM_OR.format(qualifier=qualifier))
    if tiled:
        helpers.append(device_text.TILED.format(qualifier=qualifier))
    if warp_sum:
        helpers.append((device_text.WARP_SUM_BANDED if cells else device_text.WARP_SUM).format(qualifier=qualifier))
    return f"""namespace hawk_abi {{

template <class T, std::size_t N>
{qualifier} aether::View<T, aether::extents<N, aether::dyn>, aether::layout_right>
vec_view(const eagle::plugin::GRefMirror& m)
{{
    // A GRef plane is SoA and C-contiguous by eagle's packing rule (marshal.py):
    // unit sample stride, component pitch = m.compStride_ (the full plane's
    // sample count even for a partition slice). layout_right over that ONE
    // runtime pitch lets aether fold the per-component offsets.
    using Ext = aether::extents<N, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}}

template <class T, std::size_t R, std::size_t C>
{qualifier} aether::View<T, aether::extents<R, C, aether::dyn>, aether::layout_right>
mat_view(const eagle::plugin::GRefMirror& m)
{{
    using Ext = aether::extents<R, C, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}}

template <class T>
{qualifier} aether::View<T, aether::extents<aether::dyn>, aether::layout_right>
scalar_view(const eagle::plugin::ScalarHandle& m)
{{
    using Ext = aether::extents<aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data), Map(Ext(static_cast<std::size_t>(m.samples))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType), m.deviceId));
}}

template <class ViewT, class T>
{qualifier} void accum_add(ViewT& target, const aether::SampleIndex& at, T term)
{{
    // The scattered commit (hawk/ext): ONE body string spelled per
    // target here -- the device's own atomic add, or an atomic_ref under the
    // OpenMP host team -- because a scattered read-modify-write has no correct
    // use under any parallel launch. One subscript on the target.
    auto& slot = target(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add: a scattered accumulate commits through the "
                  "target's atomic add, and only a floating accumulate plane has one");
#if defined(__CUDA_ARCH__)
    atomicAdd(&slot, static_cast<E>(term));
#else
    std::atomic_ref<E>(slot).fetch_add(static_cast<E>(term), std::memory_order_relaxed);
#endif
}}
{"".join(helpers)}
}}  // namespace hawk_abi
"""


def prelude(mode: ScalarMode, qualifier: str, *, device_text=None, compensated: bool = False,
            store_compensated: bool = False, needs_random: bool = False,
            finish: bool = False, spacing: bool = False, mix: bool = False,
            integers: bool = False, sample_sum: bool = False,
            runs: bool = False, fold: bool = False, linalg: bool = False,
            rt: bool = False, scatter_or: bool = False, rays: bool = False,
            team: bool = False, tiled: bool = False, warp_sum: bool = False, narrow_int: bool = False,
            row: int | None = None) -> str:
    """Includes + the scalar-mode aliases + the reconstruction helpers.

    ``compensated``/``store_compensated`` reach :func:`abi_helpers`'s gate.
    ``needs_random`` applies the same discipline to :data:`UNIT_INCLUDES`: only
    a unit whose body emitted a random draw gets the extra header, and
    ``runs`` the run commits' header."""
    includes = UNIT_INCLUDES
    extra = ((_VARIANT_INCLUDE if linalg or rays or team else "") + mode.includes + (_RANDOM_INCLUDE if needs_random else "") + (_RUN_INCLUDE if runs else "")
             + (_TWO_SUM_INCLUDE if compensated or store_compensated or sample_sum or fold else "")
             + (_LINALG_INCLUDE if linalg else "") + (_RT_INCLUDE if rt else "")
             + (device_text.BANDED_SUM_INCLUDE if mode.cells and (sample_sum or team or warp_sum) else "") + (_RAY_INCLUDE if rays else "") + (_TEAM_INCLUDE if team else ""))
    if extra:
        includes = includes.replace(_RANDOM_INCLUDE_MARKER,
                                    _RANDOM_INCLUDE_MARKER + extra, 1)
    return (includes
            + f"\nusing Real = {mode.real_spelling};\nusing Int = {'int' if narrow_int else 'long long'};\n\n"
            + _CLAMP_COUNT.format(qualifier=qualifier)
            + abi_helpers(qualifier, device_text=device_text, compensated=compensated,
                         store_compensated=store_compensated, finish=finish,
                         spacing=spacing, mix=mix, integers=integers,
                         sample_sum=sample_sum, fold=fold, scatter_or=scatter_or, tiled=tiled, warp_sum=warp_sum,
                         cells=mode.cells, row=row))
