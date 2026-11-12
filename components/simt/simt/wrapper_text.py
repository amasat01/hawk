# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thread-team texts and refusals of an entry wrapper: the warp-sum, banded-sum and sample-sum helper texts a unit splices in when its
body names them, the refusals of a banded build of a team reduction, and the frame of a warp-form block."""

from __future__ import annotations

from raptor.ir.kinds import is_a
from raptor.protocols.binding import binding_name

from .lane import slice_min


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
