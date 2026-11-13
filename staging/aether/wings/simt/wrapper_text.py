# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thread-team texts and refusals of an entry wrapper: the warp-sum, banded-sum and sample-sum helper texts a unit splices in when its
body names them, the refusals of a banded build of a team reduction, and the frame of a warp-form block."""

from __future__ import annotations

from raptor.errors import RaptorError
from raptor.ir.kinds import is_a
from aether.wings.base.spelling import binding_name

from simt.lane import slice_min

from . import team_cells as team_cells      # the banded team stage forms ``team_body`` takes as its ``cells`` argument


#: The calls a unit's body spells that make the wrapper splice a device-only helper in (:data:`TILED`, :data:`WARP_SUM`).
TILED_CALL = "hawk_abi::tiled<"
WARP_SUM_CALL = "hawk_abi::warp_sum<"

#: One lane's elements of a scratch plane that tiles the lanes by warp (:mod:`lowering.wide`): element ``k`` of lane
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


#: The sum of one value per lane over the 32 lanes of a warp (:mod:`simt.lane`): a fixed butterfly, so every lane ends
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

#if defined(__CUDACC__)
__device__ __forceinline__ aether::banded::BandAccum ssShfl_(aether::banded::BandAccum v, int off)
{{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}}

__device__ __forceinline__ aether::banded::Band ssShfl_(aether::banded::Band v, int off)
{{
    return eagle::cuda::shflXor(0xffffffffu, v, off);
}}
#endif
"""


BANDED_SUM_INCLUDE = "#include <aether/banded/Accum.h>\n#if defined(__CUDACC__)\n#include <eagle/cuda/WarpShuffle.h>\n#endif\n"


BANDED_SS_HELPERS = '''template <class E>
{qualifier} E ssOut_(E s) {{ return s; }}

template <class E>
{qualifier} E ssComp_(E c) {{ return c; }}

#if defined(__CUDACC__)
template <class E>
__device__ __forceinline__ E ssShfl_(E v, int off)
{{
    return __shfl_xor_sync(0xffffffffu, v, off);
}}

template <class E>
__device__ __forceinline__ void ssPin_'''


def banded_sample_sum(text: str) -> str:
    """:data:`SAMPLE_SUM` for a ``cells`` mode: value-initialised zeros (an accumulator has no ``E(0)``), the warp
    shuffle through ``ssShfl_`` (eagle's, for an accumulator) and the generic ``ssOut_``/``ssComp_``/``ssShfl_``."""
    text = (text.replace("E(0)", "E{{}}")
                .replace("-static_cast<E>(__builtin_inf())", "ssInf_<E>(-1.0f)")
                .replace("static_cast<E>(__builtin_inf())", "ssInf_<E>(1.0f)")
                .replace("__shfl_xor_sync(0xffffffffu, sh, off)", "ssShfl_(sh, off)")
                .replace("__shfl_xor_sync(0xffffffffu, sc, off)", "ssShfl_(sc, off)"))
    return text.replace("""#if defined(__CUDACC__)
template <class E>
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
