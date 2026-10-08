# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The backend seam and the scalar-mode seam.

Both seams are PROTOCOLS with their built implementations beside them,
never a class hierarchy: a backend supplies the entry wrapper around the
ONE body string :mod:`hawk.emit.aether` renders, and a scalar mode supplies
the ``Real`` the translation unit resolves to. Built today: the ``cuda``
and ``host`` backends, the ``float64``/``float32`` modes; ``metal`` and
``banded`` are the seams' NAMED empty slots — asking for ``banded`` raises
and says who is expected to build it and when.

This module also owns the shared translation-unit assembly, since it's the
same on both targets: the mandated include order (aether first, eagle's
ABI header second, so ``EAGLE_ABI_INDEX_T`` binds to the aether index
width), the scalar aliases, the reconstruction helpers, then ``unpack`` ->
``index_prologue`` -> the BODY -> ``index_epilogue``. Nothing here writes a
manifest, a sidecar or a layout export: those are :mod:`hawk.artifact`'s job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..ir import HawkError, Walk
from ..types import Slot
from .aether import Body, render_body
from .spelling import binding_name, finish_binding, view_type


@dataclass(frozen=True)
class SegmentSpec:
    """Per-unit geometry override for a ``segmented`` unit's index prologue
    (``hawk/ir/segment.py`` builds the unit and hands this to the backend
    so the prologue can read the run's own bounds).

    ``offsets_binding`` is the C++ identifier :func:`unpack_lines` already
    bound the offsets plane's view to; ``j`` is this unit's branch index, a
    compile-time Python int spelled as a literal, never a runtime value."""

    offsets_binding: str
    j: int


def segment_reads(segment: SegmentSpec) -> tuple:
    """``(seg_begin_expr, seg_end_expr)``: run ``j``'s two bounds off the
    offsets plane at compile-time indices ``j``/``j + 1``, shared by both
    backends so the two spellings cannot drift."""
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
    exactly as :class:`SegmentSpec`'s offsets are."""

    map_binding: str
    count_binding: str


def active_reads(active: ActiveSpec, position: str) -> tuple:
    """``(count_expr, sample_expr)``: the live count and the sample at
    launch position ``position``, shared by both backends.

    The map and count are 32-bit words (eagle's compaction), unlike a HAWK
    integer plane's 64-bit ``Int`` spelling, so both read through their
    views' raw data pointers as ``std::int32_t``."""
    def word(binding: str, at: str) -> str:
        return f"reinterpret_cast<const std::int32_t*>({binding}.data())[{at}]"

    return word(active.count_binding, "0"), word(active.map_binding, position)

#: Roles a body WRITES through. Their view binds non-const: a ``const``
#: ``View`` returns a read-only ``ConstSampleRef`` and cannot be assigned
#: through.
WRITABLE_ROLES = ("mutable", "out", "wide_out", "accum_out")


@dataclass(frozen=True)
class ScalarMode:
    """One compiled scalar mode (seam): its id and the storage ``Real`` a
    kernel resolves to."""

    id: str
    real_spelling: str


#: The two BUILT modes.
FLOAT64 = ScalarMode("float64", "double")
FLOAT32 = ScalarMode("float32", "float")

SCALAR_MODES = {m.id: m for m in (FLOAT64, FLOAT32)}

#: The seam's NAMED third slot: present in the vocabulary, not built, and
#: refusing with what builds it and who needs it — no shim is built in the
#: meantime, since a shim would report a precision the artifact does not
#: carry.
_UNBUILT_MODES = {
    "banded": ("emulated double precision",
               "aether/banded/ (Band / BandCell8) and aether/accum/AccumPlane.h"),
}


def scalar_mode(name: str) -> ScalarMode:
    """Resolve a scalar mode by id. ``banded`` names what it is and where it lands."""
    if name in SCALAR_MODES:
        return SCALAR_MODES[name]
    if name in _UNBUILT_MODES:
        description, headers = _UNBUILT_MODES[name]
        raise HawkError(
            f"scalar mode {name!r} is declared but not built yet ({description}); "
            f"it will live in {headers} once built. HAWK ships no softdouble "
            f"shim in the "
            f"meantime — a shim would report a precision the artifact does not "
            f"carry. Built modes are {tuple(SCALAR_MODES)}"
        )
    declared = tuple(SCALAR_MODES) + tuple(_UNBUILT_MODES)
    raise HawkError(
        f"unknown scalar mode {name!r}; the seam declares {declared}"
    )


class Backend(Protocol):
    """The backend seam. ``metal`` will implement exactly these members.

    ``storage`` takes the whole :class:`~hawk.types.Slot`, since the ABI
    shape a wire binds as is keyed on its role as well as its type. The one
    index seam splits into ``index_prologue``/``index_epilogue`` because a
    serial host loop's closing brace belongs to the same seam."""

    id: str
    qualifier: str

    def entry_signature(self, walk: Walk, name: str) -> str:
        """The entry's declaration: role args in ``arg_spec`` order, then the
        int64 partition triple."""

    def index_prologue(self, segment: SegmentSpec | None = None, *,
                       active: ActiveSpec | None = None) -> str:
        """THE one geometry seam — the only place a launch-geometry token
        may appear. Binds the body's ``i``.

        ``active`` (:class:`ActiveSpec`): launch position ``t = base +
        flat`` exits at or past the device-read live count, ``i`` is
        ``map[t]``; never combined with ``segment``. ``segment``: a
        ``segmented`` unit's column is ``offsets[j] + tid``, so the
        prologue reads its run's bounds off the offsets plane and exits a
        lane past the device-read run length. Omitted, this is the
        ordinary partition-relative prologue."""

    def index_epilogue(self) -> str:
        """The closing half of the index seam (empty where the prologue
        opens no scope). A backend declaring the optional ``lane_guards``
        hook (the host) is handed the prologue's own arguments here too,
        since what it closes depends on them."""

    def unpack(self, walk: Walk) -> str:
        """The prologue rebuilding every wire's aether view from its
        by-value mirror — ``params[]`` casts on the host, by-value binds
        on the device."""

    def dispatch(self, kind_spelling: str, branch_spellings: Sequence[str],
                 policy: str, **kw: object) -> str:
        """The per-node ``predicated``/``switch`` text — identical C++ on
        both targets, so both built backends implement this member by
        calling :func:`hawk.emit.aether.render_dispatch_rank0` directly.
        ``segmented`` never reaches here."""


@dataclass(frozen=True)
class Source:
    """One emitted translation unit and the body STRING it wrapped."""

    text: str
    body: str
    entry: str
    backend: str
    mode: str
    reduce_op: str | None = None


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

#: The Knuth two-sum residual both compensated commits below share — pure
#: arithmetic (no atomics), so one definition serves the scattered and
#: own-column forms alike. Spliced whenever either needs it, never twice.
_TWO_SUM_ERR = """
#ifndef AETHER_FP_BARRIER
// aether/macros.h's rounding pin, for an aether header set that predates it
// (a sealed NVRTC payload): a no-op on the device, an empty asm on the host.
#if defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
#define AETHER_FP_BARRIER(x) ((void)0)
#elif defined(__x86_64__) || defined(__i386__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+x"(x))
#elif defined(__aarch64__)
#define AETHER_FP_BARRIER(x) __asm__("" : "+w"(x))
#else
#define AETHER_FP_BARRIER(x) __asm__("" : "+r"(x))
#endif
#endif
template <class E>
{qualifier} E twoSumErr_(E a, E b, E s)
{{
    // Knuth two-sum residual: (a + b) - s EXACTLY, any ordering/signs (no
    // fast2Sum precondition) -- verbatim aether::accum::atomic::detail::
    // twoSumErr_ (aether/accum/atomic.h). Exact only for the ROUNDED a, b
    // and s: the barriers stop a host build with FMA contraction on (the
    // fast host profile) from fusing a product into these subtractions.
    AETHER_FP_BARRIER(a);
    AETHER_FP_BARRIER(b);
    AETHER_FP_BARRIER(s);
    const E bb = s - a;
    return (a - (s - bb)) + (b - bb);
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
    atomicAdd(&compSlot, twoSumErr_(old, t, s));
#else
    const E old = std::atomic_ref<E>(slot).fetch_add(t, std::memory_order_relaxed);
    const E s = old + t;
    std::atomic_ref<E>(compSlot).fetch_add(twoSumErr_(old, t, s),
                                           std::memory_order_relaxed);
#endif
}}
"""

#: Compensated OWN-COLUMN commit: a Neumaier accumulate into the host's
#: prior value, one writer per sample, never atomic. The total is
#: ``target + companion``, so the host must zero both before the first term.
_FINISH_COUNT = """
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
_CLAMP_COUNT = """{qualifier} int hawk_clamp_count(long long n)
{{
    return n < 2147483647LL ? static_cast<int>(n) : 2147483647;
}}

"""


def abi_helpers(qualifier: str, *, compensated: bool = False,
                store_compensated: bool = False, finish: bool = False) -> str:
    """The view-from-mirror reconstruction helpers — byte-for-byte the same
    on both targets, only the function qualifier differs; reads the
    mirror's own stride fields so a non-contiguous plane rebuilds
    correctly.

    ``compensated``/``store_compensated`` each append their own helper,
    emitted only for the unit whose resolved target needs it
    (:func:`hawk.ext.compensated_target`), so a PLAIN unit's prelude stays
    byte-identical. ``finish`` appends ``finish_count`` under the same
    rule, for a kernel that finishes its own sample."""
    helpers = []
    if compensated or store_compensated:
        helpers.append(_TWO_SUM_ERR.format(qualifier=qualifier))
    if compensated:
        helpers.append(_ACCUM_ADD_COMPENSATED.format(qualifier=qualifier))
    if store_compensated:
        helpers.append(_STORE_COMPENSATED.format(qualifier=qualifier))
    if finish:
        helpers.append(_FINISH_COUNT.format(qualifier=qualifier))
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


def prelude(mode: ScalarMode, qualifier: str, *, compensated: bool = False,
            store_compensated: bool = False, needs_random: bool = False,
            finish: bool = False) -> str:
    """Includes + the scalar-mode aliases + the reconstruction helpers.

    ``compensated``/``store_compensated`` reach :func:`abi_helpers`'s gate.
    ``needs_random`` applies the same discipline to :data:`_INCLUDES`: only
    a unit whose body emitted a random draw gets the extra header."""
    includes = _INCLUDES
    if needs_random:
        includes = includes.replace(_RANDOM_INCLUDE_MARKER,
                                    _RANDOM_INCLUDE_MARKER + _RANDOM_INCLUDE, 1)
    return (includes
            + f"\nusing Real = {mode.real_spelling};\nusing Int = long long;\n\n"
            + _CLAMP_COUNT.format(qualifier=qualifier)
            + abi_helpers(qualifier, compensated=compensated,
                         store_compensated=store_compensated, finish=finish))


def render_source(name: str, sinks, walk: Walk, backend: Backend, *,
                  mode: ScalarMode = FLOAT64, kind=None, exports: str = "",
                  body: Body | None = None,
                  segment: SegmentSpec | None = None,
                  active: ActiveSpec | None = None, one_step=None) -> Source:
    """Assemble ONE translation unit around the ONE body string.

    Both backends are handed the SAME :class:`~hawk.emit.aether.Body`; the
    row that observes it compares :attr:`Source.body`, not the files.
    ``kind`` is the :class:`hawk.ext.Kind` the body's commits render under;
    ``exports`` is a per-TU export block spliced between the prelude and
    the entry, since it belongs to the artifact, never the body.

    ``body`` supplies an already-rendered body (how a fused lane group
    reaches this wrapper); omitted, it's rendered here from ``sinks``/
    ``walk``. ``segment`` threads to :meth:`Backend.index_prologue` (see
    :class:`SegmentSpec`). ``active`` (:class:`ActiveSpec`) is passed to
    the backend only when given, so a map-free kernel's text is unchanged.

    ``one_step`` is an automatic kernel's single-step kernel
    (``hawk.steps(kernel, "auto")``): the entry reads the ``fused_steps``
    word and branches — ``1`` runs the single step's own loop and body, any
    other word the fused loop. Each side opens with its own empty device
    ``asm`` marker so the compiler can't fold the two together. Omitted,
    the TU is unchanged."""
    if (one_step is not None and body is None and segment is None
            and getattr(getattr(kind, "sink", None), "id", None) != "compensated"):
        return _render_two_paths(name, sinks, walk, backend, mode=mode,
                                 kind=kind, exports=exports, active=active,
                                 one_step=one_step)
    own_body = body is None
    if body is None:
        body = render_body(sinks, walk, indent="        ", kind=kind)
    compensated = False
    store_compensated = False
    if kind is not None and getattr(kind.sink, "id", None) == "compensated":
        from ..ext import compensated_target

        resolved = compensated_target(kind, tuple(sinks))
        if resolved is not None:
            store_compensated = resolved[1]
            compensated = not resolved[1]
    # Optional backend hooks, asked only of a backend that declares them
    # (the host's SIMD loop, lane-guarded blocks, hawk/emit/host.py); every
    # other backend's call and text are unchanged. A backend with lane
    # guards closes its loop from the same arguments it opened with.
    prologue_kw: dict = {} if active is None else {"active": active}
    simd_safe = getattr(backend, "simd_safe", None)
    if simd_safe is not None:
        prologue_kw["simd"] = simd_safe(walk)
    lane_guards = getattr(backend, "lane_guards", None)
    if lane_guards is not None:
        prologue_kw["guards"] = lane_guards(walk)
    lane_finish = getattr(backend, "lane_finish", None)
    if lane_finish is not None:
        prologue_kw["finish"] = lane_finish(walk)
    epilogue = (backend.index_epilogue(segment, **prologue_kw)
                if lane_guards is not None else backend.index_epilogue())
    tiled = (_interchange(backend, name, sinks, walk, kind)
             if own_body and segment is None and active is None
             and not compensated and not store_compensated else None)
    extra_prelude = getattr(backend, "extra_prelude", None)
    loop = [backend.index_prologue(segment, **prologue_kw), body.text, epilogue]
    parts = [
        f"// HAWK-emitted {backend.id} translation unit (scalar mode: {mode.id}).\n"
        f"// Body text is the ONE renderer's; this file is the entry wrapper.\n",
        prelude(mode, backend.qualifier, compensated=compensated,
               store_compensated=store_compensated, needs_random=body.needs_random,
               finish=getattr(walk, "finish", None) is not None),
        _extra_prelude(extra_prelude, walk, tiled),
        _exports(exports, tiled),
        "",
        backend.entry_signature(walk, name),
        "{",
        backend.unpack(walk),
        *_tiled_or(tiled, loop),
        "}",
        "",
    ]
    return Source("\n".join(p for p in parts if p is not None), body.text, name,
                  backend.id, mode.id, body.reduce_op)


def _interchange(backend: Backend, name: str, sinks, walk: Walk, kind):
    """The backend's interchanged fused-step loop for this TU (the host's
    optional ``interchange`` hook, :mod:`hawk.emit.host`), or ``None``: a
    backend without the hook, or a kernel that is not a fused step, keeps
    its text unchanged."""
    hook = getattr(backend, "interchange", None)
    if hook is None or getattr(walk, "finish", None) is None:
        return None
    return hook(name, sinks, walk, kind)


def _extra_prelude(hook, walk: Walk, tiled) -> str | None:
    """The backend's extra prelude, told about an interchange only when there
    is one, so every other TU's text is unchanged."""
    if hook is None:
        return None
    return hook(walk) if tiled is None else hook(walk, tiled=tiled)


def _exports(exports: str, tiled) -> str:
    """The TU's export block plus an interchange's tile export."""
    return exports if tiled is None else f"{exports}\n{tiled.export}"


def _tiled_or(tiled, plain: list) -> list:
    """``plain`` (the index loop and body), or the interchanged loop under
    its macro with ``plain`` kept verbatim as the macro-off build."""
    if tiled is None:
        return plain
    from .host import INTERCHANGE_MACRO

    return [f"#if {INTERCHANGE_MACRO}", tiled.text, "#else", *plain, "#endif"]


def _frame(walk: Walk, backend: Backend, active: ActiveSpec | None) -> tuple:
    """The index prologue and epilogue :func:`render_source` wraps a body of
    ``walk`` in (the host's SIMD/lane-guard hooks asked of ``walk``)."""
    kw: dict = {} if active is None else {"active": active}
    simd_safe = getattr(backend, "simd_safe", None)
    if simd_safe is not None:
        kw["simd"] = simd_safe(walk)
    lane_guards = getattr(backend, "lane_guards", None)
    if lane_guards is not None:
        kw["guards"] = lane_guards(walk)
    lane_finish = getattr(backend, "lane_finish", None)
    if lane_finish is not None:
        kw["finish"] = lane_finish(walk)
    epilogue = (backend.index_epilogue(None, **kw)
                if lane_guards is not None else backend.index_epilogue())
    return backend.index_prologue(None, **kw), epilogue


#: The marker each side of an automatic kernel's branch opens with (device
#: only): two different side-effecting statements, so the compiler can't
#: hoist the sides' common code above the branch and fold them into one.
_PATH_MARKER = '#if defined(__CUDA_ARCH__)\n    asm volatile("// hawk: {}");\n#endif'


def _render_two_paths(name: str, sinks, walk: Walk, backend: Backend, *,
                      mode: ScalarMode, kind, exports: str,
                      active: ActiveSpec | None, one_step) -> Source:
    """The TU of an automatic kernel: :func:`render_source` with the entry's
    index loop and body emitted twice under one uniform branch on the
    ``fused_steps`` word — the single step's (``one_step``) when the word is
    ``1``, the fused loop's otherwise.

    On a backend with the ``persist`` hook (the device), the TU also carries
    the kernel's fast-path entries (:func:`_fast_entries`)."""
    fused = render_body(sinks, walk, indent="        ", kind=kind)
    single = render_body(one_step.sinks, one_step.walk, indent="        ", kind=kind)
    tiled = None if active is not None else _interchange(backend, name, sinks,
                                                         walk, kind)
    extra_prelude = getattr(backend, "extra_prelude", None)
    parts = [
        f"// HAWK-emitted {backend.id} translation unit (scalar mode: {mode.id}).\n"
        f"// Body text is the ONE renderer's; this file is the entry wrapper.\n",
        prelude(mode, backend.qualifier, needs_random=fused.needs_random
                or single.needs_random,
                finish=getattr(walk, "finish", None) is not None),
        _extra_prelude(extra_prelude, walk, tiled),
        _exports(exports, tiled),
        "",
        *_two_path_entry(name, walk, backend, fused, single, one_step, active,
                         tiled),
        "",
        *_fast_entries(backend, name, sinks, walk, kind, fused, single, one_step,
                       active),
    ]
    return Source("\n".join(p for p in parts if p is not None),
                  single.text + "\n" + fused.text, name, backend.id, mode.id,
                  fused.reduce_op)


def _two_path_entry(entry: str, walk: Walk, backend: Backend, fused, single,
                    one_step, active: ActiveSpec | None, tiled) -> list:
    """The two-path entry ``entry`` of an automatic kernel: its signature,
    unpack and the branch on the ``fused_steps`` word between the single
    step's index loop (``single``) and the fused loop's (``fused``), each
    under ``active``'s prologue (``None``: the contiguous range)."""
    from ..ir.nodes import FUSED_STEPS_PLANE

    word = (f"{binding_name('lookup', FUSED_STEPS_PLANE)}"
            "[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()")
    one_open, one_close = _frame(one_step.walk, backend, active)
    many_open, many_close = _frame(walk, backend, active)
    return [
        backend.entry_signature(walk, entry),
        "{",
        backend.unpack(walk),
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


def _fast_entries(backend: Backend, name: str, sinks, walk: Walk, kind, fused,
                  single, one_step, active: ActiveSpec | None) -> list:
    """The fast-path entries an automatic kernel's TU carries beside its own
    entry, on a backend with the ``persist`` hook (the device); ``[]`` on
    any other.

    ``<name>_persist`` (:func:`hawk.emit.cuda.persist_entry`) and
    ``<name>_range`` (:func:`hawk.emit.cuda.range_entry`, the kernel's own
    two-path entry over a contiguous range, ``base + flat``), both
    reporting the run's longest per-sample step count. Both take the
    kernel's own entry parameters (an active-set kernel's map and count
    among them, unread), so a caller binds every entry alike. The range
    entry, and an active-set kernel's persist entry, compile only under
    :data:`FAST_ENTRIES_MACRO`; a map-free kernel's persist entry always."""
    hook = getattr(backend, "persist", None)
    if hook is None:
        return []
    persist = hook(name, sinks, walk, kind)
    ranged = backend.range(name, sinks, walk, kind, single.text)
    always = [] if persist is None or active is not None else [persist, ""]
    gated = [] if ranged is None else [ranged, ""]
    if persist is not None and active is not None:
        gated += [persist, ""]
    if not always and not gated:
        return []
    out = [backend.fast_prelude(), "", *always]
    if gated:
        out += [f"#ifndef {FAST_ENTRIES_MACRO}\n#define {FAST_ENTRIES_MACRO} 0\n#endif",
                f"#if {FAST_ENTRIES_MACRO}", *gated, f"#endif  // {FAST_ENTRIES_MACRO}",
                ""]
    return out


def unpack_lines(walk: Walk, source_of) -> str:
    """The shared half of both backends' ``unpack``: one binding per
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
        if role == "terminated" and name == finished_mask:
            # The finish epilogue's writable twin: same mirror, non-const
            # view, so the read binding is untouched.
            twin = finish_binding(name)
            lines.append(f"    {view_type(role, slot.ttype)} {twin} =")
            lines.append(f"        {source_of(slot)};")
            lines.append(f"    (void){twin};")
    return "\n".join(lines)
