# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thread-team texts and refusals of an entry wrapper: the warp-sum, banded-sum and sample-sum helper texts a unit splices in when its
body names them, the refusals of a banded build of a team reduction, and the frame of a warp-form block."""

from __future__ import annotations

from raptor.errors import RaptorError
from raptor.ir.kinds import is_a
from aether.wings.base.spelling import binding_name

from raptor.wings.simt.lane import slice_min

from . import team_cells as team_cells      # the banded team stage forms ``team_body`` takes as its ``cells`` argument


#: The device arm of each helper the base prelude spells for both targets: what a CUDA unit carries; the host
#: arm is :data:`aether.wings.base.wrapper_text.HOST_ARMS`.
ARMS = {
    "accum_add": "    atomicAdd(&slot, static_cast<E>(term));",
    "accum_add_compensated": "\n".join((
            "    const E old = atomicAdd(&slot, t);",
            "    const E s = old + t;",
            "    atomicAdd(&compSlot, aether::accum::twoSumErrAtomic(old, t, s));",
        )),
    "accum_or": "    atomicOr(reinterpret_cast<unsigned int*>(&slot), static_cast<unsigned int>(bits));",
    "finish_count": "    atomicAdd(reinterpret_cast<unsigned int*>(cell), 1u);",
    "mix_count": "\n".join((
            "    const unsigned active = __activemask();",
            "    unsigned below;",
            "    asm(\"mov.u32 %0, %%lanemask_lt;\" : \"=r\"(below));",
            "    const bool leader = (active & below) == 0u;",
            "    for (int j = 0; j < K; ++j) {",
            "        const unsigned picked = __ballot_sync(active, k == j);",
            "        if (leader && picked != 0u)",
            "            atomicAdd(reinterpret_cast<unsigned int*>(cells + j),",
            "                      static_cast<unsigned int>(__popc(picked)));",
            "    }",
        )),
    "spacing": "    const T up = ::nextafter(a, a * T(2) + T(1)) - a;",
}


def arm(name: str) -> str:
    """The text of the helper ``name`` in a device unit: its device arm alone."""
    return ARMS[name]


#: The marker each side of an automatic kernel's branch opens with in a device unit (:func:`aether.wings.base.wrapper._path_marker`).
PATH_MARKER = '    asm volatile("// hawk: {}");'

#: The calls a unit's body spells that make the wrapper splice a device-only helper in (:data:`TILED`, :data:`WARP_SUM`).
TILED_CALL = "hawk_abi::tiled<"
WARP_SUM_CALL = "hawk_abi::warp_sum<"

#: One lane's elements of a scratch plane that tiles the lanes by warp (:mod:`raptor.wings.base.wide`): element ``k`` of lane
#: ``i`` of a plane of width ``W`` is at ``((i / 32) * W + k) * 32 + i % 32``, so the lane's proxy is its first element
#: and ``at(k)`` adds ``32 k`` elements. It has the shape of an aether per-sample proxy to the ``aether::rt`` twins.
#: Spliced only into a unit whose body names it.
TILED = """
template <class T, std::size_t W>
struct TiledLane {{
    using element_type = T;
    using element_extents = aether::extents<W>;
    T* base;
    {qualifier} T& at(std::size_t k) const {{ return base[k * 32]; }}
    /** the lane as an aether expression (a stride-0 sample mode: every sample reads this lane's elements) */
    {qualifier} auto expr() const
    {{
        using Ext = aether::extents<W, aether::dyn>;
        using V = aether::View<T, Ext, aether::layout_stride>;
        return V(base, typename V::mapping_type(Ext(1), {{std::size_t{{32}}, std::size_t{{0}}}}), aether::Device{{}});
    }}
}};

template <class T, std::size_t W, class ViewT>
{qualifier} TiledLane<T, W> tiled(const ViewT& plane, long long lane)
{{
    const std::size_t i = static_cast<std::size_t>(lane);
    return TiledLane<T, W>{{plane.data() + (i / 32 * W) * 32 + i % 32}};
}}
"""


#: The sum of one value per lane over the 32 lanes of a warp (:mod:`raptor.wings.simt.lane`): a fixed butterfly, so every lane ends
#: with the same bits and a replay gives them again. Spliced only into a unit whose body names it.
WARP_SUM = """
template <class T>
{qualifier} T warp_sum(T v)
{{
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}}
"""


#: The banded sum over the 32 lanes of a warp: each lane folds its value into a ``BandAccum``, the butterfly merges whole
#: accumulators through eagle's ``shflXor`` with ``combine`` (the lower lane's accumulator first, so every lane ends
#: with the same bits), and the total is encoded once.
WARP_SUM_BANDED = """
template <class T>
{qualifier} T warp_sum(T v)
{{
    aether::banded::BandAccum a;
    a.addTerm(aether::banded::Band(v));
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    for (int m = 16; m > 0; m >>= 1) {{
        aether::banded::BandAccum o = eagle::cuda::shflXor(0xffffffffu, a, m);
        if (lane & static_cast<unsigned>(m)) {{
            o.combine(a);
            a = o;
        }} else {{
            a.combine(o);
        }}
    }}
    return a.toBandedReal();
}}
"""



#: The banded row sum: a partial is aether's compensated ``BandAccum`` (the residue-lane fold) and two partials merge
#: with its ``combine``; the warp tree ships whole accumulators through eagle's ``shflXor``. These overloads are
#: declared BEFORE :data:`SAMPLE_SUM`, whose templates pick them for the accumulator type.
BANDED_SUM = """
template <int OP = 0>
{qualifier} void pairCombine_(aether::banded::BandAccum& s, aether::banded::BandAccum&,
                             aether::banded::BandAccum s2, aether::banded::BandAccum)
{{
    s.combine(s2);
}}

{qualifier} void ssPin_(aether::banded::BandAccum&) {{}}

// hawk.max / hawk.min: the fold is exact, so a partial is a bare Band and two partials merge with the certified fmax / fmin.
template <class E>
{qualifier} E ssInf_(float sign) {{ return E(sign * __builtin_inff(), 0.0f, 0.0f); }}

{qualifier} aether::banded::Band ssExtreme_(int op, aether::banded::Band a, aether::banded::Band b)
{{
    return op == 1 ? aether::banded::detail::fmax(a, b) : aether::banded::detail::fmin(a, b);
}}

template <int OP = 0>
{qualifier} void pairCombine_(aether::banded::Band& s, aether::banded::Band&, aether::banded::Band s2, aether::banded::Band)
{{
    s = ssExtreme_(OP, s, s2);
}}

{qualifier} void ssPin_(aether::banded::Band&) {{}}

{qualifier} Real ssOut_(aether::banded::Band s) {{ return Real(s); }}

{qualifier} Real ssOut_(aether::banded::BandAccum s) {{ return s.toBandedReal(); }}

{qualifier} Real ssComp_(aether::banded::BandAccum) {{ return Real{{}}; }}

__device__ __forceinline__ aether::banded::BandAccum ssShfl_(aether::banded::BandAccum v, int off)
{{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}}

__device__ __forceinline__ aether::banded::Band ssShfl_(aether::banded::Band v, int off)
{{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}}
"""


BANDED_SUM_INCLUDE = "#include <aether/banded/Accum.h>\n#include <eagle/cuda/WarpShuffle.h>\n"


BANDED_SS_HELPERS = '''template <class E>
{qualifier} E ssOut_(E s) {{ return s; }}

template <class E>
{qualifier} E ssComp_(E c) {{ return c; }}

template <class E>
__device__ __forceinline__ E ssShfl_(E v, int off)
{{
    return __shfl_xor_sync(0xffffffffu, v, off);
}}

template <class E>
__device__ __forceinline__ void ssPin_'''


#: The warp butterfly of the row sum, spliced after :data:`aether.wings.base.wrapper_text.SAMPLE_SUM` in a device unit: the same pairs, five
#: ``__shfl_xor_sync`` rounds. Needs ``twoSumErr_`` above it.
SAMPLE_SUM_WARP = """template <class E>
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
"""


def banded_sample_sum(text: str) -> str:
    """:data:`SAMPLE_SUM` for a ``cells`` mode: value-initialised zeros (an accumulator has no ``E(0)``), the warp
    shuffle through ``ssShfl_`` (eagle's, for an accumulator) and the generic ``ssOut_``/``ssComp_``/``ssShfl_``."""
    text = (text.replace("E(0)", "E{{}}")
                .replace("-static_cast<E>(__builtin_inf())", "ssInf_<E>(-1.0f)")
                .replace("static_cast<E>(__builtin_inf())", "ssInf_<E>(1.0f)")
                .replace("__shfl_xor_sync(0xffffffffu, sh, off)", "ssShfl_(sh, off)")
                .replace("__shfl_xor_sync(0xffffffffu, sc, off)", "ssShfl_(sc, off)"))
    return text.replace("""template <class E>
__device__ __forceinline__ void ssPin_""", BANDED_SS_HELPERS, 1)



def staged(wrapper, rows: int, sinks, mode, spec) -> tuple:
    """The frame of a block that serves ``rows`` lanes of a warp-form kernel (:meth:`CudaWrapper.stage_frame`): the vector
    outputs it stages through shared memory are those of its warp stores."""
    stages = tuple((binding_name(s.role, s.name), s.ttype.shape[0]) for s in sinks
                   if is_a(s, "warp_store") and s.ttype.shape and s.ttype.shape[0] >= slice_min(s.ttype.dtype, spec))
    return wrapper.stage_frame(rows, stages, "Real")


_ATOMIC_REMEDY = "a packed cell has no atomic add; use the default or compensated contract, which a banded build routes to the ordered stages"
_MATH_REMEDY = "until its banded row lands in aether; build this kernel in float64 or float32 meanwhile"

# The refusals below are reasoned, not pending:
#  * accum_write (hawk.scatter, and hawk.sum(x, over=...) with its default contract). The float64 emission opens an
#    aether::accum::run per lane and commits each run of equal keys with ONE atomicAdd into the output plane (and a second
#    one, of the two-sum residual read off the first's return value, into the companion plane). A banded cell is one
#    8-byte packed Band: there is no atomic add on it, and the compensated BandAccum (limbs plus residue lanes) does not
#    fit a cell, so the residual has nowhere to live between commits. A CAS loop (decode, plain add, encode) would be
#    built from the PLAIN fold, 94 ULP53 on the 2^-30 cancellation class against the 0.01 the banded sum promises, so it is
#    refused rather than shipped under that name. The design that meets the contract is a two-pass one: per-warp
#    BandAccum partials, shuffled with eagle's shflXor, in a BandAccum-shaped scratch plane (a new plane dtype in
#    team_planes, sized in _lane_bind and allocated by the runtime and eagle), then a deterministic combine stage with
#    BandAccum::combine into the cell.
#  * the ordered team stages (contract="ordered": block_part, block_sum, ordered_fold, the scatter_* steps) already ARE that
#    two-pass shape in float64, but their partials plane holds Real and their adds are a plain `+=` of cells, which
#    would be the plain fold under the sum's name; they stay refused until the partials plane can hold a BandAccum.
#  * warp_sum (a sum across the lanes of a team) needs the same BandAccum plane or a BandAccum shuffle in the team body.
#: What a banded build refuses BY NAME, with the remedy: ``{IR kind: (what the
#: author wrote, remedy)}``. Every reduction sink, and the math that has no
#: certified banded row yet; the rest of the certified facade is admitted.
_BANDED_UNCERTIFIED = {
    "accum_write": ("an atomic accumulate into a plane (hawk.scatter with contract 'plain', accum_write)", _ATOMIC_REMEDY),
    **{k: (f"hawk.math.{k}", _MATH_REMEDY)
       for k in ("erf", "erfc", "rint", "remainder", "fdim",
                 "isnan", "isinf", "isfinite", "random_uniform", "random_normal")},
}






def refuse_uncertified(sinks, walk, mode) -> None:
    """Refuse a ``cells`` build that reaches an op of :data:`_BANDED_UNCERTIFIED`, naming it and the remedy."""
    if not mode.cells:
        return
    for kind in (*walk.kinds, *(getattr(s, "kind", None) for s in sinks)):
        hit = _BANDED_UNCERTIFIED.get(kind)
        if hit is not None:
            raise RaptorError(f"{hit[0]} is not built in scalar mode {mode.id!r} yet: {hit[1]}")

# --------------------------------------------------------------------------- #
# Device arms of the team stage bodies (:mod:`aether.wings.base.team_emit` takes this module as its ``device_text``).
# --------------------------------------------------------------------------- #


def team_sh_decl(n: int) -> list:
    """A team's scratch of ``n`` partial sums, in shared memory."""
    return [f"__shared__ Real hawk_sh[{n}];"]


def fold_loop(FOLD_LANES: int, VIRTUAL_LANES: int, term: str) -> list:
    """The device arm of a block-part fold: each thread sums its rows into ``hawk_sh``."""
    return [
        f"for (std::size_t hawk_t = static_cast<std::size_t>(hawk_tm.rank); hawk_t < {FOLD_LANES}; "
        "hawk_t += static_cast<std::size_t>(hawk_tm.size)) {",
        "    Real hawk_s = Real(0);",
        f"    for (std::size_t hawk_r = hawk_grp * {FOLD_LANES} + hawk_t; hawk_r < hawk_rows; hawk_r += {VIRTUAL_LANES})",
        f"        hawk_s += {term};",
        "    hawk_sh[hawk_t] = hawk_s;",
        "}",
    ]


def scatter_bins(n: int, cells, plain: list) -> list:
    """The device arm of an ordered scatter's slot walk: up to 1024 cells a warp takes 32 rows at a time into a shared bin (``plain``, the serial
    walk, serves a larger target)."""
    walk = [
        "    if (hawk_tm.rank < 32) {",
        "        const unsigned hawk_ln = static_cast<unsigned>(hawk_tm.rank);",
        f"        const std::size_t hawk_end = hawk_hi * {n};",
        f"        for (std::size_t hawk_b = hawk_lo * {n}; hawk_b < hawk_end; hawk_b += 32) {{",
        "            const std::size_t hawk_r = hawk_b + hawk_ln;",
        "            const bool hawk_in = hawk_r < hawk_end;",
        "            const std::int32_t hawk_mk = hawk_in ? hawk_k[hawk_r] : -1;",
        "            const Real hawk_mv = hawk_in ? hawk_v[hawk_r] : Real{};",
        "            unsigned hawk_mine = 0u, hawk_bad = 0u;",
        "            for (unsigned hawk_i = 0; hawk_i < 32; ++hawk_i) {",
        "                const std::int32_t hawk_key = __shfl_sync(0xffffffffu, hawk_mk, hawk_i);",
        "                if (hawk_key == -1) continue;",
        "                if (hawk_key >= 0 && hawk_key < hawk_cells) {",
        "                    if ((static_cast<unsigned>(hawk_key) & 31u) == hawk_ln) hawk_mine |= 1u << hawk_i;",
        "                } else hawk_bad = 1u;",
        "            }",
        "            if (hawk_bad && hawk_ln == 0) hawk_m[1 + hawk_lane] = 1;",
        "            while (__any_sync(0xffffffffu, hawk_mine != 0u)) {",
        "                const unsigned hawk_src = hawk_mine != 0u ? static_cast<unsigned>(__ffs(hawk_mine) - 1) : 0u;",
        *(cells.SHUFFLED_VALUE if cells else
          ["                const Real hawk_x = __shfl_sync(0xffffffffu, hawk_mv, hawk_src);"]),
        "                const std::int32_t hawk_key = __shfl_sync(0xffffffffu, hawk_mk, hawk_src);",
        "                if (hawk_mine != 0u) {",
        (cells.BIN_ADD if cells else
         "                    hawk_bin[hawk_key] += hawk_x;"),
        "                    hawk_mine &= hawk_mine - 1u;",
        "                }",
        "            }",
        "            if (hawk_in && hawk_mk != -1) hawk_k[hawk_r] = -1;",
        "        }",
        "    }"]
    return [
        (cells.BIN_DECL if cells else "__shared__ Real hawk_bin[1024];"),
        "if (hawk_cells <= 1024) {",
        "    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); "
        "hawk_j += static_cast<std::size_t>(hawk_tm.size)) hawk_bin[hawk_j] = " + (cells.BIN_ZERO if cells else "Real(0);"),
        "    hawk_tm.sync();",
        *walk,
        "    hawk_tm.sync();",
        "    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); "
        "hawk_j += static_cast<std::size_t>(hawk_tm.size))",
        cells.BIN_COMBINE if cells else
        "        hawk_p[hawk_j * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_bin[hawk_j];",
        "} else {",
        *plain,
        "}",
    ]


#: The device arm of the warp eigenvalue step: one warp serves the lane, the rest of the team waits at the barrier.
EIGH_WARP = (
    "            __shared__ aether::linalg::warp::EighValuesScratch<Real> hawk_scratch;",
    "            if (hawk_tm.rank < 32) {",
    "                const auto hawk_r = aether::linalg::warp::eighValues<32>(hawk_scratch, static_cast<const Real*>(hawk_a));",
    "                hawk_w[hawk_tm.rank] = hawk_r.w;",
    "                if (hawk_tm.rank == 0) *hawk_info = hawk_r.info;",
    "            }",
)
