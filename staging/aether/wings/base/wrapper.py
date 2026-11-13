# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wrapper seam and the scalar-mode seam.

Both seams are PROTOCOLS with their built implementations beside them,
never a class hierarchy: a wrapper supplies the entry around the
ONE body string :mod:`aether.wings.base.render` renders, and a scalar mode supplies
the ``Real`` the translation unit resolves to. Built today: the ``cuda``
and ``host`` wrappers, the ``float64``/``float32``/``banded`` modes; ``metal``
is the wrapper seam's NAMED empty slot.

This module also owns the shared translation-unit assembly, since it's the
same on both targets: the mandated include order (aether first, eagle's
ABI header second, so ``EAGLE_ABI_INDEX_T`` binds to the aether index
width), the scalar aliases, the reconstruction helpers, then ``unpack`` ->
``index_prologue`` -> the BODY -> ``index_epilogue``. Nothing here writes a
manifest, a sidecar or a layout export: those are :mod:`hawk.artifact`'s job.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from raptor.protocols.naming import fold_entry_name
from raptor.ir.linalg import LSTSQ_OPS
from raptor.ir import Walk
from raptor.errors import RaptorError
from raptor.ir.nodes import sample_sums
from raptor.ir.types import INT_PLANE_DTYPES, Slot
from lowering import box as _box
from .box import frame as _box_frame
from .render import Body, render_body
from .writer_aether import RUN_CALL, RUN_STATE
from .spelling import binding_name, element_spelling, finish_binding, param_cell_binding, view_type


@dataclass(frozen=True)
class SegmentSpec:
    """Per-unit geometry override for a ``segmented`` unit's index prologue
    (``hawk/ir/segment.py`` builds the unit and hands this to the wrapper
    so the prologue can read the run's own bounds).

    ``offsets_binding`` is the C++ identifier :func:`unpack_lines` already
    bound the offsets plane's view to; ``j`` is this unit's branch index, a
    compile-time Python int spelled as a literal, never a runtime value."""

    offsets_binding: str
    j: int


def segment_reads(segment: SegmentSpec) -> tuple:
    """``(seg_begin_expr, seg_end_expr)``: run ``j``'s two bounds off the
    offsets plane at compile-time indices ``j``/``j + 1``, shared by both
    wrappers so the two spellings cannot drift."""
    off = segment.offsets_binding
    begin_idx = f"aether::SampleIndex::make(static_cast<std::size_t>({segment.j}))"
    end_idx = f"aether::SampleIndex::make(static_cast<std::size_t>({segment.j + 1}))"
    return f"{off}[{begin_idx}].eval()", f"{off}[{end_idx}].eval()"

@dataclass(frozen=True)
class ActiveSpec:
    """The index prologue of a kernel whose guard reads an active-set
    index map: launch position ``t`` reads sample ``map[t]`` and exits at
    or past the device-read ``count``.

    ``map_binding``/``count_binding`` are bound by :func:`unpack_lines`
    exactly as :class:`SegmentSpec`'s offsets are. ``group`` is the
    compaction group, a lane count: at 1 the map holds sample indices; above
    1 it holds group indices and position ``t`` reads sample ``map[t / group]
    * group + t % group``."""

    map_binding: str | None     # None: the prefix form, lane ``t`` runs lane ``t``
    count_binding: str
    group: int = 1
    #: The map word, 32 or 64 bits (the count is always 32).
    map_bits: int = 32
    #: A lane set of 32-bit width (:class:`hawk.ext.IndexSet`): its lanes, and so every index the body makes of them,
    #: fit 32 bits, and the unit's ``Int`` is a 32-bit integer (:func:`prelude`).
    lane_set: bool = False


def active_reads(active: ActiveSpec, position: str) -> tuple:
    """``(count_expr, sample_expr)``: the live count and the sample at
    launch position ``position``, shared by both wrappers.

    The count is a 32-bit word and the map is 32-bit (eagle's compaction) or,
    for a lane set over an extent past ``2**31 - 1``, 64-bit; both read through
    their views' raw data pointers, unlike a HAWK integer plane's ``Int``
    spelling."""
    def word(binding: str, at: str, bits: int = 32) -> str:
        return f"reinterpret_cast<const std::int{bits}_t*>({binding}.data())[{at}]"

    if active.map_binding is None:
        return word(active.count_binding, "0"), f"({position})"
    if active.group > 1:
        g = active.group
        return (word(active.count_binding, "0"),
                f"static_cast<long long>({word(active.map_binding, f'({position}) / {g}', active.map_bits)})"
                f" * {g} + ({position}) % {g}")
    return word(active.count_binding, "0"), word(active.map_binding, position, active.map_bits)


def lane_loop_simd(wrapper, walk: Walk, active: ActiveSpec | None) -> bool:
    """Whether ``wrapper`` writes ``walk``'s loop over samples in its SIMD
    form: not behind a grouped map, whose positions are not contiguous."""
    simd_safe = getattr(wrapper, "simd_safe", None)
    return (simd_safe is not None and simd_safe(walk)
            and not (active is not None and (active.group > 1 or walk.reads_lane_position)))

#: Roles a body WRITES through. Their view binds non-const: a ``const``
#: ``View`` returns a read-only ``ConstSampleRef`` and cannot be assigned
#: through.
WRITABLE_ROLES = ("mutable", "out", "wide_out", "accum_out")


@dataclass(frozen=True)
class ScalarMode:
    """One compiled scalar mode (seam): its id, the storage ``Real`` a
    kernel resolves to and the headers that spell it. ``cells``: a ``Real``
    in storage is an 8-byte cell and a value in registers the three-limb
    ``Band``, so the body takes the writer's cell form
    (:class:`~aether.wings.base.writer_aether.AetherWriter`)."""

    id: str
    real_spelling: str
    includes: str = ""
    cells: bool = False


#: The BUILT modes.
FLOAT64 = ScalarMode("float64", "double")
FLOAT32 = ScalarMode("float32", "float")
BANDED = ScalarMode("banded", "aether::banded::BandedReal",
                    "#include <aether/banded/banded.h>\n#include <aether/banded/BandedRealOps.h>\n",
                    cells=True)

SCALAR_MODES = {m.id: m for m in (FLOAT64, FLOAT32, BANDED)}


def scalar_mode(name: str) -> ScalarMode:
    """Resolve a scalar mode by id."""
    if name in SCALAR_MODES:
        return SCALAR_MODES[name]
    raise RaptorError(
        f"unknown scalar mode {name!r}; the seam declares {tuple(SCALAR_MODES)}"
    )


_ATOMIC_REMEDY = "a packed cell has no atomic add; use the default or compensated policy, which a banded build routes to the ordered stages"
_MATH_REMEDY = "until its banded row lands in aether; build this kernel in float64 or float32 meanwhile"

# The refusals below are reasoned, not pending:
#  * accum_write (hawk.scatter, and hawk.sum(x, over=...) with its default policy). The float64 emission opens an
#    aether::accum::run per lane and commits each run of equal keys with ONE atomicAdd into the output plane (and a second
#    one, of the two-sum residual read off the first's return value, into the companion plane). A banded cell is one
#    8-byte packed Band: there is no atomic add on it, and the compensated BandAccum (limbs plus residue lanes) does not
#    fit a cell, so the residual has nowhere to live between commits. A CAS loop (decode, plain add, encode) would be
#    built from the PLAIN fold, 94 ULP53 on the 2^-30 cancellation class against the 0.01 the banded sum promises, so it is
#    refused rather than shipped under that name. The design that meets the contract is a two-pass one: per-warp
#    BandAccum partials, shuffled with eagle's shflXor, in a BandAccum-shaped scratch plane (a new plane dtype in
#    team_planes, sized in _lane_bind and allocated by the runtime and eagle), then a deterministic combine stage with
#    BandAccum::combine into the cell. Size L: the plane dtype across hawk/runtime/eagle, a combine stage, and
#    keyed-plane zeroing for banded cells.
#  * the ordered team stages (policy="ordered": block_part, block_sum, ordered_fold, the scatter_* steps) already ARE that
#    two-pass shape in float64, but their partials plane holds Real and their adds are a plain `+=` of cells, which
#    would be the plain fold under the sum's name; they stay refused until the partials plane can hold a BandAccum.
#  * warp_sum (a sum across the lanes of a team) needs the same BandAccum plane or a BandAccum shuffle in the team body.
#: What a banded build refuses BY NAME, with the remedy: ``{IR kind: (what the
#: author wrote, remedy)}``. Every reduction sink, and the math that has no
#: certified banded row yet; the rest of the certified facade is admitted.
_BANDED_UNCERTIFIED = {
    "accum_write": ("an atomic accumulate into a plane (hawk.scatter with policy 'plain', accum_write)", _ATOMIC_REMEDY),
    **{k: (f"hawk.math.{k}", _MATH_REMEDY)
       for k in ("erf", "erfc", "rint", "remainder", "fdim",
                 "isnan", "isinf", "isfinite", "random_uniform", "random_normal")},
}






def refuse_uncertified(sinks, walk: Walk, mode: ScalarMode) -> None:
    """Refuse a ``cells`` build that reaches an op of :data:`_BANDED_UNCERTIFIED`, naming it and the remedy."""
    if not mode.cells:
        return
    for kind in (*walk.kinds, *(getattr(s, "kind", None) for s in sinks)):
        hit = _BANDED_UNCERTIFIED.get(kind)
        if hit is not None:
            raise RaptorError(f"{hit[0]} is not built in scalar mode {mode.id!r} yet: {hit[1]}")


def _writer(target, mode: ScalarMode):
    """``target``'s writer in ``mode``: a ``cells`` mode takes the aether writer's cell form."""
    if not mode.cells:
        return target.writer
    from .writer_aether import AETHER_CELLS

    return AETHER_CELLS


class Wrapper(Protocol):
    """The wrapper seam: how a kernel is wrapped for one execution model.
    A new target with its own execution model implements exactly these members.

    ``storage`` takes the whole :class:`~hawk.types.Slot`, since the ABI
    shape a wire binds as is keyed on its role as well as its type. The one
    index seam splits into ``index_prologue``/``index_epilogue`` because a
    serial host loop's closing brace belongs to the same seam."""

    qualifier: str

    def vectorises(self, walk: Walk, spec, scalar_type: str) -> bool:
        """Whether the host loop over samples of ``walk`` is vectorised (asked only of a target that vectorises)."""

    def entry_signature(self, walk: Walk, name: str) -> str:
        """The entry's declaration: role args in ``arg_spec`` order, then the
        int64 partition triple."""

    def index_prologue(self, segment: SegmentSpec | None = None, *,
                       active: ActiveSpec | None = None, box=None) -> str:
        """THE one geometry seam — the only place a launch-geometry token
        may appear. Binds the body's ``i``.

        ``active`` (:class:`ActiveSpec`): launch position ``t = base +
        flat`` exits at or past the device-read live count, ``i`` is
        ``map[t]``; never combined with ``segment``. ``segment``: a
        ``segmented`` unit's column is ``offsets[j] + tid``, so the
        prologue reads its run's bounds off the offsets plane and exits a
        lane past the device-read run length. ``box``
        (:class:`simt.box.Box`): ``base + flat`` is a thread and ``i``
        its first lane. Omitted, this is the ordinary partition-relative
        prologue."""

    def index_epilogue(self) -> str:
        """The closing half of the index seam (empty where the prologue
        opens no scope). A wrapper declaring the optional ``lane_guards``
        hook (the host) is handed the prologue's own arguments here too,
        since what it closes depends on them."""

    def unpack(self, walk: Walk) -> str:
        """The prologue rebuilding every wire's aether view from its
        by-value mirror — ``params[]`` casts on the host, by-value binds
        on the device."""

    def fold_rows(self, width: int, spec) -> int:
        """Rows of 32 samples one warp reduces in a batched fold of ``width`` terms per sample (1: one row at a time)."""

    def sample_sum_frame(self, width: int, elem: str, store: str,
                         op: str = "sum", live: str | None = None, *, spec) -> tuple:
        """``(open, close)`` around the body of a kernel with ``sample_sum``
        sinks: the body writes ``width`` terms of type ``elem`` per sample
        into ``hawk_ss``; each row of 32 samples is reduced by ``op`` (sum,
        max or min) and handed to ``store`` (the entry-local
        ``hawk_ss_store``) at row ``hawk_row``. ``live`` (a C++ expression) is
        the device-read count of a prefix form: samples at or past it add the
        identity, and a row wholly past it is not stored."""

    def dispatch(self, kind_spelling: str, branch_spellings: Sequence[str],
                 policy: str, **kw: object) -> str:
        """The per-node ``predicated``/``switch`` text — identical C++ on
        both targets, so both built wrappers implement this member by
        calling :func:`aether.wings.base.render.render_dispatch_rank0` directly.
        ``segmented`` never reaches here."""


@dataclass(frozen=True)
class Source:
    """One emitted translation unit and the body STRING it wrapped."""

    text: str
    body: str
    entry: str
    target: str
    mode: str
    #: The host loop over samples is vectorised (see :func:`aether.wings.cpu.host_vector.vectorises`).
    vectorised: bool = False
    #: The thread box the entry was emitted with (:class:`simt.box.Box`), or ``None``.
    box: object = None


#: The headers every emitted TU includes, in the mandated order: aether
#: first (defines ``AETHER_INDEX_T``), eagle's ABI header second, so
#: ``EAGLE_ABI_INDEX_T`` binds to the aether index width.
_INCLUDES = """#include <aether/typedefs.h>
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
#: The macro that switches a host unit's interchanged loop on (the host wrapper defines it on by default).
INTERCHANGE_MACRO = "HAWK_HOST_INTERCHANGE"
#: The namespace of the collective calls of a team stage (its body is written by :mod:`aether.wings.base.team_emit`).
TEAM_CALL = "aether::linalg::team::"
from lowering import wrapper_text as _wt
from .wrapper_text import FINISH_COUNT, SAMPLE_SUM
from lowering import split as _split
_RAY_INCLUDE = "#include <aether/geometry/geometry.h>\n"

#: The rolled-loop twins' header (:mod:`raptor.ir.fold`), likewise.
_RT_CALL = "aether::rt::"
_TILED_CALL = "hawk_abi::tiled<"
_WARP_SUM_CALL = "hawk_abi::warp_sum<"
_RT_INCLUDE = "#include <aether/expr/Rt.h>\n"

_RUN_SITE = re.compile(rf"\b{RUN_STATE}(\w+)\);")


def run_states(text: str) -> list:
    """The targets a body's run commits name, in order, once each: each
    shares the entry-local state ``hawk_run_<target>``."""
    return list(dict.fromkeys(_RUN_SITE.findall(text)))


def _with_runs(loop: list, text: str) -> list:
    """``loop`` between the declarations of the run states ``text`` calls
    and their closes: the host commits each open run at the end of the
    entry's range; on the device both are no-ops. Unchanged for a body with
    no run commit."""
    runs = run_states(text)
    if not runs:
        return loop
    return ["\n".join(f"    auto {RUN_STATE}{t} = {RUN_CALL}open({t});" for t in runs),
            *loop,
            "\n".join(f"    {RUN_CALL}close({RUN_STATE}{t});" for t in runs)]

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

#: The scattered ``policy="or"`` commit: an atomic OR of a 32-bit word into the
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

#: One lane's elements of a scratch plane that tiles the lanes by warp (:mod:`simt.wide`): element ``k`` of lane
#: ``i`` of a plane of width ``W`` is at ``((i / 32) * W + k) * 32 + i % 32``, so the lane's proxy is its first element
#: and ``at(k)`` adds ``32 k`` elements. It has the shape of an aether per-sample proxy to the ``aether::rt`` twins.
#: Spliced only into a unit whose body names it.
_TILED = """
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
    for a kernel with a :class:`~hawk.ir.nodes.SampleSum` sink."""
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
        helpers.append(_TILED.format(qualifier=qualifier))
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
    ``needs_random`` applies the same discipline to :data:`_INCLUDES`: only
    a unit whose body emitted a random draw gets the extra header, and
    ``runs`` the run commits' header."""
    includes = _INCLUDES
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


def narrows_int(active: ActiveSpec | None, walk: Walk) -> bool:
    """Whether the unit's ``Int`` is 32 bits: a lane set of 32-bit width, which holds no integer the launch
    passes by value or binds as a plane (those keep the 64-bit word eagle packs). Every index the body makes of
    a lane of such a set, and every integer it computes, then takes a register, not a pair: a kernel that
    gathers through tables keeps a long chain of them live, and that chain is what its register count is."""
    return (active is not None and active.lane_set and active.map_bits == 32
            and not any(ttype.dtype in ("i32", "i64") or (role == "uniform" and ttype.dtype in INT_PLANE_DTYPES)
                        for (role, _name), ttype in walk.slot_types.items()))


def _for_build(wrapper, spec):
    """``wrapper`` as this build uses it: a wrapper that declares ``launch_bounds`` gets its own copy holding the build's
    (``spec.settings.launch_bounds``); the shared instance is never written."""
    if not hasattr(wrapper, "launch_bounds"):
        return wrapper
    import copy

    own = copy.copy(wrapper)
    own.launch_bounds = spec.settings.launch_bounds
    return own


def render_source(name: str, sinks, walk: Walk, target, *, spec, kind,
                  mode: ScalarMode = FLOAT64, exports: str = "",
                  body: Body | None = None,
                  segment: SegmentSpec | None = None,
                  active: ActiveSpec | None = None, one_step=None,
                  box=None, team: int | None = None, rows: int = 1) -> Source:
    """Assemble ONE translation unit around the ONE body string.

    ``target`` is the :class:`hawk.backends.Backend` row: its writer spells the
    body and its wrapper wraps it. Both wrappers are handed the SAME :class:`~aether.wings.base.render.Body`; the
    row that observes it compares :attr:`Source.body`, not the files.
    ``kind`` is the :class:`hawk.ext.Kind` the body's commits render under;
    ``exports`` is a per-TU export block spliced between the prelude and
    the entry, since it belongs to the artifact, never the body.

    ``body`` supplies an already-rendered body (how a fused lane group
    reaches this wrapper); omitted, it's rendered here from ``sinks``/
    ``walk``. ``segment`` threads to :meth:`Wrapper.index_prologue` (see
    :class:`SegmentSpec`). ``active`` (:class:`ActiveSpec`) is passed to
    the wrapper only when given, so a map-free kernel's text is unchanged.

    ``one_step`` is an automatic kernel's single-step kernel
    (``hawk.steps(kernel, "auto")``): the entry reads the ``fused_steps``
    word and branches — ``1`` runs the single step's own loop and body, any
    other word the fused loop. Each side opens with its own empty device
    ``asm`` marker so the compiler can't fold the two together. Omitted,
    the TU is unchanged.

    ``team`` (the threads per team of a team stage, :mod:`simt.team_emit`) makes the lane the
    block on the device; the host entry is the plain lane loop, a team of one.

    ``box`` (:class:`simt.box.Box`, from :func:`simt.box.thread_box`)
    emits the plain entry with a thread box: each thread computes ``box.T``
    lanes (:func:`simt.box.boxed_sinks`).

    A kernel with :class:`~hawk.ir.nodes.SampleSum` sinks runs the
    wrapper's ``sample_sum_frame`` instead of the index loop, and is refused
    under the segmented, active-set and automatic (``_persist``) entries."""
    if kind is None:
        raise RaptorError("the writer needs the kernel kind from its caller (hawk.ext.Kind); none was given")
    spec = spec.for_target(target)
    _wt.refuse_cells_off_device(name, mode, target.id, target)
    if walk.kinds.intersection(LSTSQ_OPS):     # the writer takes lowered sinks (lowering.lstsq.lowered), never lowers
        raise RaptorError(f"the writer was given an unlowered least-squares node ({', '.join(sorted(walk.kinds.intersection(LSTSQ_OPS)))}); "
                        "the caller lowers the kernel first (lowering.lstsq.lowered)")
    refuse_uncertified(sinks, walk, mode)
    writer = _writer(target, mode)
    sums = sample_sums(sinks)
    _wt.refuse_sample_sums(sums, segment=segment, active=active, one_step=one_step)
    if (one_step is not None and body is None and segment is None
            and getattr(getattr(kind, "sink", None), "id", None) != "compensated"):
        return _render_two_paths(name, sinks, walk, target, spec=spec, mode=mode,
                                 kind=kind, exports=exports, active=active,
                                 one_step=one_step, team=team, rows=rows)
    own_body = body is None
    if box is not None:
        # the box's body, and the box-free one for a launch with fewer rows than T
        plain = render_body(sinks, walk, indent=" " * 16, kind=kind, writer=writer, spec=spec)
        boxed = render_body(_box.boxed_sinks(sinks, box), walk, indent=" " * 12, kind=kind,
                            writer=writer, spec=spec)
        body = Body(_box_frame(box, plain.text, boxed.text))
    if body is None:
        plain_frame = (segment is None and active is None and box is None and team is None and one_step is None and not sums
                       and rows <= 1 and not any(role == "terminated" for role, _ in walk.arg_spec))
        body = render_body(sinks, walk, indent="        ", kind=kind, writer=writer, spec=spec.with_settings(block_fold=plain_frame))
    compensated = False
    wrapper = _for_build(target.wrapper, spec)
    store_compensated = False
    if getattr(kind.sink, "id", None) == "compensated":
        resolved = _split.compensated_target(kind, tuple(sinks))
        if resolved is not None:
            store_compensated = resolved[1]
            compensated = not resolved[1]
    # a policy="compensated" scatter that commits per lane
    compensated = compensated or "hawk_abi::accum_add_compensated(" in body.text
    # Optional wrapper hooks, asked only of a wrapper that declares them
    # (the host's SIMD loop, lane-guarded blocks, staging/aether/wings/cpu/host.py); every
    # other wrapper's call and text are unchanged. A wrapper with lane
    # guards closes its loop from the same arguments it opened with.
    prologue_kw: dict = {} if active is None else {"active": active}
    if team is not None and getattr(wrapper, "team_flat", None) is not None:
        prologue_kw["team"] = team      # the device lane is the block, the team its threads
        prologue_kw["warps"] = spec.settings.warps
    if getattr(wrapper, "simd_safe", None) is not None:
        prologue_kw["simd"] = team is None and lane_loop_simd(wrapper, walk, active)
    lane_guards = getattr(wrapper, "lane_guards", None)
    if lane_guards is not None:
        prologue_kw["guards"] = lane_guards(walk)
    lane_finish = getattr(wrapper, "lane_finish", None)
    if lane_finish is not None:
        prologue_kw["finish"] = lane_finish(walk)
    epilogue = (wrapper.index_epilogue(segment, **prologue_kw)
                if lane_guards is not None else wrapper.index_epilogue())
    tiled = (_interchange(wrapper, name, sinks, walk, kind, mode, spec)
             if own_body and segment is None and active is None and box is None
             and not compensated and not store_compensated and not sums else None)
    if run_states(body.text):
        tiled = None                    # the interchanged loop holds no run state
    extra_prelude = getattr(wrapper, "extra_prelude", None)
    #: what the headers are chosen by: the per-sample body, and the interchanged loop (which may name a library the body does not)
    lanes_text = body.text if tiled is None else body.text + tiled.text
    loop = [wrapper.index_prologue(segment, **prologue_kw), body.text, epilogue]
    if rows > 1:        # a block serves ``rows`` lanes and stores their vector outputs through a shared tile
        loop = [*wrapper.device_text.staged(wrapper, rows, sinks, mode, spec), ]
        loop = [loop[0], body.text, loop[1]]
    if box is not None:
        loop = [wrapper.index_prologue(box=box), body.text, wrapper.index_epilogue()]
    if sums:
        opened, closed = wrapper.sample_sum_frame(
            len(sums), ("aether::banded::BandAccum" if sums[0].op == "sum" else "aether::banded::Band") if mode.cells else element_spelling(sums[0].ttype.dtype),
            _sample_sum_store(sums, wrapper.fold_rows(len(sums), spec), mode.cells, spec),
            sums[0].op, None if active is None else active_reads(active, "0")[0], spec=spec)
        loop = [opened, body.text, closed]
    loop = _with_runs(loop, body.text)
    parts = [
        f"// HAWK-emitted {target.id} translation unit (scalar mode: {mode.id}).\n"
        f"// Body text is the ONE renderer's; this file is the entry wrapper.\n",
        prelude(mode, wrapper.qualifier, device_text=getattr(wrapper, "device_text", None), compensated=compensated,
               store_compensated=store_compensated, needs_random=body.needs_random,
               finish=getattr(walk, "finish", None) is not None,
               spacing=_SPACING_CALL in body.text, mix=_MIX_CALL in body.text,
               integers=_INT_CALL in body.text, sample_sum=bool(sums), row=spec.need("cells", "row") if sums else None,
               runs=RUN_CALL in body.text, fold=bool(walk.companions),
               linalg=_LINALG_CALL in lanes_text, rt=_RT_CALL in lanes_text,
               rays=_RAY_CALL in body.text, team=TEAM_CALL in body.text,
               scatter_or=_OR_CALL in body.text, tiled=_TILED_CALL in body.text, warp_sum=_WARP_SUM_CALL in body.text,
               narrow_int=narrows_int(active, walk)),
        _extra_prelude(extra_prelude, walk, tiled, body.text),
        _exports(exports, tiled),
        "",
        wrapper.entry_signature(walk, name),
        "{",
        wrapper.unpack(walk),
        *_tiled_or(tiled, loop),
        "}",
        "",
        *fold_entries(name, walk, wrapper),
    ]
    return Source(_filled(parts, mode), body.text, name,
                  target.id, mode.id,
                  target.vectorise and box is None
                  and wrapper.vectorises(walk, spec, mode.id), box)


def fold_entries(name: str, walk: Walk, wrapper) -> list:
    """One entry per owned companion (:attr:`hawk.ir.Walk.companions`): the
    kernel's own signature and argument block, one lane per slot of the
    target, ``hawk_abi::fold_companion``. The runtime launches it after the
    kernel, over the target's slots, so the target alone holds the
    compensated sum and the companion keeps the residual."""
    out = []
    for comp, target in walk.companions.items():
        out += [wrapper.entry_signature(walk, fold_entry_name(name, target)), "{",
                wrapper.unpack(walk), wrapper.index_prologue(),
                f"        hawk_abi::fold_companion({binding_name('accum_out', target)}, "
                f"{binding_name('accum_out', comp)}, i);",
                wrapper.index_epilogue(), "}", ""]
    return out




SAMPLE_SUM_OP_CODE = {"sum": 0, "max": 1, "min": 2}


def sample_sum_fill(op: str, width: int, array: str, elem: str) -> str:
    """The statement that sets ``array`` (``width`` slots) to the identity of
    ``op``: nothing a zero-initialised array lacks for a sum."""
    if op == "sum":
        return ""
    return (f"    for (int hawk_k = 0; hawk_k < {width}; ++hawk_k) "
            f"{array}[hawk_k] = hawk_abi::ssIdentity_<{SAMPLE_SUM_OP_CODE[op]}, {elem}>();\n")


def _sample_sum_store(sums, rows: int = 1, cells: bool = False, spec=None) -> str:
    """The entry-local ``hawk_ss_store(column, sum, compensation)`` the row
    sum calls: column ``k`` goes to the ``k``-th sink's two planes at row
    ``hawk_row``."""
    lines = ["    auto hawk_ss_store = "
             "[&](int hawk_col, auto hawk_s, auto hawk_c) {"]
    if rows > 1:
        w = len(sums)
        lines += [f"        const int hawk_rr = hawk_col / {w};",
                  f"        if (hawk_wbase + {spec.need('cells', 'row')} * hawk_rr >= count) return;",
                  "        const long long hawk_row = hawk_row0 + hawk_rr;",
                  f"        hawk_col %= {w};"]
    for k, sink in enumerate(sums):
        hi = binding_name(sink.role, sink.name)
        lines += [f"        {'else ' if k else ''}if (hawk_col == {k}) {{",
                  f"            {hi}.data()[hawk_row] = {'hawk_abi::ssOut_(hawk_s)' if cells else 'hawk_s'};"]
        if sink.op == "sum":
            comp = binding_name(sink.role, sink.companion.name)
            lines.append(f"            {comp}.data()[hawk_row] = {'hawk_abi::ssComp_(hawk_c)' if cells else 'hawk_c'};")
        lines.append("        }")
    return "\n".join(lines + ["    };"])


_REAL_LITERAL = re.compile(r"static_cast<Real>\((-?[0-9][0-9.eE+-]*)\)")
_LITERAL_TABLE = "hawk_f64_literals"


def literal_table(source: Source) -> Source:
    """``source`` with every ``double`` literal of its body read from one
    ``__constant__`` table instead of being an operand immediate.

    The table holds each distinct value once, in the literal's own
    round-tripping text, so every value is bit-identical to the literal it
    replaces. A ``float`` TU is returned unchanged (its literals are
    single-precision immediates, and a table of doubles would put 64-bit loads
    in a kernel that has none). The declaration goes after the last
    ``#include`` of the TU, so the table is visible to the whole body."""
    if source.mode != FLOAT64.id:
        return source
    slots: dict[str, int] = {}

    def slot(match):
        index = slots.setdefault(match.group(1), len(slots))
        return f"static_cast<Real>({_LITERAL_TABLE}[{index}])"

    head, sep, tail = source.text.rpartition("#include")
    first_line, nl, rest = tail.partition("\n")
    text = _REAL_LITERAL.sub(slot, rest)
    if not slots:
        return source
    table = (f"static __constant__ double {_LITERAL_TABLE}[{len(slots)}] = {{"
             + ", ".join(slots) + "};")
    return replace(source, text=f"{head}{sep}{first_line}\n{table}\n{text}")


def _interchange(wrapper: Wrapper, name: str, sinks, walk: Walk, kind,
                 mode: ScalarMode, spec):
    """The wrapper's interchanged fused-step loop for this TU (the host's
    optional ``interchange`` hook, :mod:`aether.wings.cpu.host`), or ``None``: a
    wrapper without the hook, or a kernel the hook declines, keeps its text
    unchanged."""
    hook = getattr(wrapper, "interchange", None)
    if hook is None:
        return None
    return hook(name, sinks, walk, kind, 4 if mode is FLOAT32 else 8, spec)


#: The host lane-guard ABI's lane type until the TU's scalar mode fills it in.
_LANE_TYPE = "HAWK_LANE_T"


def _filled(parts, mode: ScalarMode) -> str:
    """The TU text of ``parts``, its guard/tally lanes spelled as the mode's
    ``Real``, so a float32 loop has one width."""
    return "\n".join(p for p in parts if p is not None).replace(_LANE_TYPE, mode.real_spelling)


def _extra_prelude(hook, walk: Walk, tiled, body: str) -> str | None:
    """The wrapper's extra prelude, told about an interchange only when there
    is one, so every other TU's text is unchanged; ``body`` is the text it
    must supply helpers for."""
    if hook is None:
        return None
    return hook(walk, body=body) if tiled is None else hook(walk, tiled=tiled, body=body)


def _exports(exports: str, tiled) -> str:
    """The TU's export block plus an interchange's tile export."""
    return exports if tiled is None else f"{exports}\n{tiled.export}"


def _tiled_or(tiled, plain: list) -> list:
    """``plain`` (the index loop and body), or the interchanged loop under
    its macro with ``plain`` kept verbatim as the macro-off build."""
    if tiled is None:
        return plain
    return [f"#if {INTERCHANGE_MACRO}", tiled.text, "#else", *plain, "#endif"]




def _frame(walk: Walk, wrapper: Wrapper, active: ActiveSpec | None, team: int | None = None) -> tuple:
    """The index prologue and epilogue :func:`render_source` wraps a body of
    ``walk`` in (the host's SIMD/lane-guard hooks asked of ``walk``)."""
    kw: dict = {} if active is None else {"active": active}
    if team is not None and getattr(wrapper, "team_flat", None) is not None:
        kw["team"] = team                # the device lane is the block, the team its threads
    if getattr(wrapper, "simd_safe", None) is not None:
        kw["simd"] = lane_loop_simd(wrapper, walk, active)
    lane_guards = getattr(wrapper, "lane_guards", None)
    if lane_guards is not None:
        kw["guards"] = lane_guards(walk)
    lane_finish = getattr(wrapper, "lane_finish", None)
    if lane_finish is not None:
        kw["finish"] = lane_finish(walk)
    epilogue = (wrapper.index_epilogue(None, **kw)
                if lane_guards is not None else wrapper.index_epilogue())
    return wrapper.index_prologue(None, **kw), epilogue


#: The marker each side of an automatic kernel's branch opens with (device
#: only): two different side-effecting statements, so the compiler can't
#: hoist the sides' common code above the branch and fold them into one.
_PATH_MARKER = '#if defined(__CUDA_ARCH__)\n    asm volatile("// hawk: {}");\n#endif'


def _render_two_paths(name: str, sinks, walk: Walk, target, *, spec,
                      mode: ScalarMode, kind, exports: str,
                      active: ActiveSpec | None, one_step, team: int | None = None, rows: int = 1) -> Source:
    """The TU of an automatic kernel: :func:`render_source` with the entry's
    index loop and body emitted twice under one uniform branch on the
    ``fused_steps`` word — the single step's (``one_step``) when the word is
    ``1``, the fused loop's otherwise.

    On a wrapper with the ``persist`` hook (the device), the TU also carries
    the kernel's fast-path entries (:func:`_fast_entries`)."""
    wrapper = _for_build(target.wrapper, spec)
    writer = _writer(target, mode)
    fused = render_body(sinks, walk, indent="        ", kind=kind, writer=writer, spec=spec)
    single = render_body(one_step.sinks, one_step.walk, indent="        ", kind=kind,
                         writer=writer, spec=spec)
    tiled = None if active is not None else _interchange(wrapper, name, sinks,
                                                         walk, kind, mode, spec)
    extra_prelude = getattr(wrapper, "extra_prelude", None)
    parts = [
        f"// HAWK-emitted {target.id} translation unit (scalar mode: {mode.id}).\n"
        f"// Body text is the ONE renderer's; this file is the entry wrapper.\n",
        prelude(mode, wrapper.qualifier, device_text=getattr(wrapper, "device_text", None), needs_random=fused.needs_random
                or single.needs_random,
                finish=getattr(walk, "finish", None) is not None,
                spacing=_SPACING_CALL in fused.text + single.text,
                mix=_MIX_CALL in fused.text + single.text,
                integers=_INT_CALL in fused.text + single.text,
                linalg=_LINALG_CALL in fused.text + single.text,
                rt=_RT_CALL in fused.text + single.text,
                rays=_RAY_CALL in fused.text + single.text,
                scatter_or=_OR_CALL in fused.text + single.text,
                tiled=_TILED_CALL in fused.text + single.text,
                warp_sum=_WARP_SUM_CALL in fused.text + single.text),
        _extra_prelude(extra_prelude, walk, tiled, fused.text + single.text),
        _exports(exports, tiled),
        "",
        *_two_path_entry(name, walk, wrapper, fused, single, one_step, active,
                         tiled, team, wrapper.device_text.staged(wrapper, rows, sinks, mode, spec) if rows > 1 else None),
        "",
        *([] if team is not None else _fast_entries(wrapper, name, sinks, walk, kind, fused, single, one_step,
                                                    active, spec)),     # the fast entries are a lane per thread: a warp per lane has none
    ]
    return Source(_filled(parts, mode),
                  single.text + "\n" + fused.text, name, target.id, mode.id)


def _two_path_entry(entry: str, walk: Walk, wrapper: Wrapper, fused, single,
                    one_step, active: ActiveSpec | None, tiled, team: int | None = None, staged: tuple | None = None) -> list:
    """The two-path entry ``entry`` of an automatic kernel: its signature,
    unpack and the branch on the ``fused_steps`` word between the single
    step's index loop (``single``) and the fused loop's (``fused``), each
    under ``active``'s prologue (``None``: the contiguous range)."""
    from raptor.ir.nodes import FUSED_STEPS_PLANE

    word = (f"{binding_name('lookup', FUSED_STEPS_PLANE)}"
            "[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()")
    one_open, one_close = staged or _frame(one_step.walk, wrapper, active, team)
    many_open, many_close = staged or _frame(walk, wrapper, active, team)
    return [
        wrapper.entry_signature(walk, entry),
        "{",
        wrapper.unpack(walk),
        f"    if ({word} == 1) {{",
        _PATH_MARKER.format("one step"),
        one_open,
        single.text,
        one_close,
        "    } else {",
        _PATH_MARKER.format("fused steps"),
        *_tiled_or(tiled, [many_open, fused.text, many_close]),
        "    }",
        "}",
    ]


#: The suffix of an automatic kernel's contiguous-range entry:
#: ``<name>_range`` (device only, :func:`_fast_entries`).
RANGE_SUFFIX = "_range"

#: The macro an automatic kernel's device TU compiles its gated fast-path
#: entries under (:func:`_fast_entries`): ``0`` unless the build defines it
#: (``build_bundle(..., defines=("HAWK_FAST_ENTRIES=1",))``), so the default
#: build compiles only what it compiled before and a caller that routes to
#: the fast paths pays for them in its own build of the same source.
FAST_ENTRIES_MACRO = "HAWK_FAST_ENTRIES"


def _fast_entries(wrapper: Wrapper, name: str, sinks, walk: Walk, kind, fused,
                  single, one_step, active: ActiveSpec | None, spec) -> list:
    """The fast-path entries an automatic kernel's TU carries beside its own
    entry, on a wrapper with the ``persist`` hook (the device); ``[]`` on
    any other.

    ``<name>_persist`` (:func:`hawk.emit.cuda.persist_entry`) and
    ``<name>_range`` (:func:`hawk.emit.cuda.range_entry`, the kernel's own
    two-path entry over a contiguous range, ``base + flat``), both
    reporting the run's longest per-sample step count. Both take the
    kernel's own entry parameters (an active-set kernel's map and count
    among them, unread), so a caller binds every entry alike. The range
    entry, and an active-set kernel's persist entry, compile only under
    :data:`FAST_ENTRIES_MACRO`; a map-free kernel's persist entry always."""
    hook = getattr(wrapper, "persist", None)
    if hook is None:
        return []
    persist = hook(name, sinks, walk, kind, spec)
    ranged = wrapper.range(name, sinks, walk, kind, single.text, spec)
    always = [] if persist is None or active is not None else [persist, ""]
    gated = [] if ranged is None else [ranged, ""]
    if persist is not None and active is not None:
        gated += [persist, ""]
    if not always and not gated:
        return []
    out = [wrapper.fast_prelude(), "", *always]
    if gated:
        out += [f"#ifndef {FAST_ENTRIES_MACRO}\n#define {FAST_ENTRIES_MACRO} 0\n#endif",
                f"#if {FAST_ENTRIES_MACRO}", *gated, f"#endif  // {FAST_ENTRIES_MACRO}",
                ""]
    return out


def unpack_lines(walk: Walk, source_of) -> str:
    """The shared half of both wrappers' ``unpack``: one binding per
    ``arg_spec`` slot, in ``arg_spec`` order, each typed by ``storage`` and
    rebuilt from the by-value mirror ``source_of`` names on that target.

    The order is ``arg_spec``'s own, since it's also the entry's parameter
    order. A slot bound but never read still binds, and is voided so a real
    build does not warn about it."""
    lines = []
    finish = getattr(walk, "finish", None)
    finished_mask = None if finish is None else finish[0]
    for role, name in walk.arg_spec:
        slot = Slot(role, name, walk.slot_types[(role, name)])
        ident = binding_name(role, name)
        qual = "" if role in WRITABLE_ROLES else "const "
        lines.append(f"    {qual}{view_type(role, slot.ttype)} {ident} =")
        lines.append(f"        {source_of(slot)};")
        lines.append(f"    (void){ident};")
        if role == "lookup" and name in walk.param_planes:
            # a promoted uniform: ONE load, ahead of the sample index, so the
            # body reads it like the by-value immediate it replaces
            cell = param_cell_binding(name)
            lines.append(f"    const auto {cell} = {ident}[aether::SampleIndex::make("
                         "static_cast<std::size_t>(0))].eval();")
            lines.append(f"    (void){cell};")
        if role == "terminated" and name == finished_mask:
            # The finish epilogue's writable twin: same mirror, non-const
            # view, so the read binding is untouched.
            twin = finish_binding(name)
            lines.append(f"    {view_type(role, slot.ttype)} {twin} =")
            lines.append(f"        {source_of(slot)};")
            lines.append(f"    (void){twin};")
    return "\n".join(lines)
