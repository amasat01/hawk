# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host wrapper: the SERIAL entry around the ONE body string.

Everything host-specific lives here: the ``extern "C"`` entry taking a
packed ``params[]`` block and the int64 partition triple, the
``params[k]`` casts the view reconstruction reads from, and a SERIAL loop
over ``[base, base + count)``. There is no ``#pragma omp`` anywhere and
there never will be: threading and tiling are eagle's ``HostTeam``, each
tile calling this entry with its own triple, which keeps wide/accum
columns disjoint.

SIMD within one thread. Serial is not scalar: for a ``sample_local``
kernel (:func:`simd_safe`) the loop is written so the auto-vectoriser can
run several samples per instruction. The loop's index is
``aether::offset_t`` (an int64 index narrowed at every access is an
address that may wrap, which the vectoriser treats as a gather); the loop
is annotated independent (``#pragma GCC ivdep`` / clang's
``vectorize(assume_safety)``, an assertion about memory only, since a
``sample_local`` kernel touches only its own sample of every plane).

``terminated`` is a 1-byte ``bool``, too narrow a type for the vectoriser
to widen into a lane mask, so the host binds it through
:data:`HOST_MASK_HELPERS` as ``unsigned char`` (``byte != 0``). That byte
guard would otherwise size GCC's vectorisation factor at 64 samples per
zmm, spilling every float64 value across eight registers; under AVX-512
with GCC (``HAWK_HOST_LANE_GUARDS``) the loop instead runs in blocks of
``kHostLaneBlock`` samples, each widening its guard bytes into a
lane-width array first (``HostMask::lanes``; the kernel's ``Real``, spelled
``HAWK_LANE_T`` until :func:`hawk.emit.wrapper.render_source` fills it in) so
the narrowest type in the per-sample loop is the state's own width. A kernel that finishes
its own sample would otherwise call and store a byte inside the loop,
which GCC refuses to vectorise; in a lane-guarded block that write lands
in the sample's own guard lane (``HostLaneFinish``) and tally
(``HostLaneTally``), committed to the real bytes and counter after the
loop (``finish_lanes``) — same bytes, same count.

The ACTIVE SET's loop runs ``[base, min(base + count, live))`` via a
generic per-sample lambda called from two loops: the contiguous
vectorisable loop when the tile's map is the identity (checked, never
assumed), else the still-independent but gather/scatter map loop.

A user ``Loop`` of a ``sample_local`` kernel runs INTERCHANGED
(:func:`loop_interchange`): its trips outside, a tile of samples inside,
the carried values in lane arrays, so the lane loop vectorises whatever the
trip count; a ``Loop`` nested in it stays each lane's own ``for``.

Any ``cross_sample_*``/``mapreduce`` kernel stays scalar (iterations
aren't independent). Results are bit-identical to the scalar build of the
same compile profile (:data:`hawk.compile.toolchain.HOST_PROFILES`) in
every case.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from ..ir import HawkError, Walk
from ..ir.nodes import FINISHED_PLANE
from ..types import Slot
from .render import (
    SAMPLE_SUM_TERMS,
    LaneSegments,
    LaneSplit,
    render_lane_segments,
    render_lane_split,
)
from .wrapper import (
    SAMPLE_SUM_OP_CODE,
    ActiveSpec,
    SegmentSpec,
    active_reads,
    sample_sum_fill,
    segment_reads,
    unpack_lines,
)
from .box import BOX_OWN, owned_lane
from .dispatch import render_dispatch_rank0
from .spelling import (
    binding_name,
    carry_type,
    element_spelling,
    finish_binding,
    mirror_of,
    reconstruct_call,
    view_type,
)

#: Host-only helpers spliced after the shared prelude: the ``terminated``
#: plane binding the host reads its masks through, same mirror and bytes
#: as the device's ``aether::View<bool>`` (module docstring, "the MASK"),
#: widened per block into ``HostLaneMask`` where the loop runs in blocks
#: ("the MASK'S WIDTH"). Same answer either way: ``byte != 0``.
HOST_MASK_HELPERS = """#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#define HAWK_HOST_LANE_GUARDS 1
#else
#define HAWK_HOST_LANE_GUARDS 0
#endif

namespace hawk_abi {

// Samples per block of the lane-guarded loop (one zmm of guard bytes).
inline constexpr aether::offset_t kHostLaneBlock = 64;

static inline aether::offset_t lane_block_end(aether::offset_t lo, aether::offset_t hi)
{
    return hi - lo < kHostLaneBlock ? hi : lo + kHostLaneBlock;
}

struct HostLaneMaskRef {
    HAWK_LANE_T lane;
    bool eval() const { return lane != 0.0; }
};

// A block's guards, one float64 lane per sample of [base, base + kHostLaneBlock).
struct HostLaneMask {
    const HAWK_LANE_T* lanes;
    aether::offset_t base;
    HostLaneMaskRef operator[](const aether::SampleIndex& at) const
    {
        return HostLaneMaskRef{ lanes[at.global() - base] };
    }
};

struct HostMaskRef {
    const unsigned char* at;
    bool eval() const { return *at != 0; }
};

struct HostMask {
    const unsigned char* data;
    HostMaskRef operator[](const aether::SampleIndex& at) const
    {
        return HostMaskRef{ data + at.global() };
    }
    // Widen the guards of samples [lo, hi) into out[0, hi - lo).
    HostLaneMask lanes(HAWK_LANE_T* out, aether::offset_t lo, aether::offset_t hi) const
    {
        for (aether::offset_t s = lo; s < hi; ++s)
            out[s - lo] = data[s] != 0 ? 1.0 : 0.0;
        return HostLaneMask{ out, lo };
    }
};

static inline HostMask host_mask(const eagle::plugin::ScalarHandle& m)
{
    return HostMask{ static_cast<const unsigned char*>(m.data) };
}

#if HAWK_HOST_LANE_GUARDS
struct HostLaneFinishRef {
    HAWK_LANE_T* lane;
    void operator=(bool v) const { *lane = v ? 1.0 : 0.0; }
};

// The finish epilogue's mask writes inside a lane-guarded block land in the
// sample's own guard lane, so the block's loop stores no byte and calls nothing.
struct HostLaneFinish {
    HAWK_LANE_T* lanes;
    aether::offset_t base;
    HostLaneFinishRef operator[](const aether::SampleIndex& at) const
    {
        return HostLaneFinishRef{ lanes + (at.global() - base) };
    }
};

// The finish epilogue's count inside a lane-guarded block: a plain increment
// of the block's own tally, which finish_lanes adds to the counter at once.
// (A float64 tally keeps the loop's narrowest type the float64 lane; it
// counts at most kHostLaneBlock, exactly.)
struct HostLaneTally {
    HAWK_LANE_T* finished;
};

static inline void finish_count(const HostLaneTally& tally) { *tally.finished += 1.0; }

// Commit a block's guard lanes back to the mask bytes of samples [lo, hi) (a
// set lane is a set byte), then its tally to the counter in ONE relaxed add:
// the same total as one increment per newly finished sample.
template <class ViewT>
static inline void finish_lanes(const HostMask& m, const HAWK_LANE_T* lanes,
                                aether::offset_t lo, aether::offset_t hi,
                                HAWK_LANE_T tally, const ViewT& counter)
{
    unsigned char* bytes = const_cast<unsigned char*>(m.data);
    for (aether::offset_t s = lo; s < hi; ++s) {
        if (lanes[s - lo] != 0.0)
            bytes[s] = 1;
    }
    const auto finished = static_cast<std::uint32_t>(tally);
    if (finished != 0) {
        auto* cell = const_cast<std::uint32_t*>(
            reinterpret_cast<const std::uint32_t*>(counter.data()));
        std::atomic_ref<std::uint32_t>(*cell).fetch_add(finished, std::memory_order_relaxed);
    }
}
#endif

}  // namespace hawk_abi
"""

#: The loop annotation a ``sample_local`` kernel's loop carries: iterations
#: carry no memory dependence (each touches only its own sample). Spelled for
#: both host compilers HAWK drives (``g++`` and the ``zig c++`` / clang tier).
_INDEPENDENT_ITERATIONS = """#if defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif"""


def simd_safe(walk: Walk) -> bool:
    """Whether ``walk``'s per-sample loop carries no dependence between
    iterations: exactly the ``sample_local`` access class. Assumes, as
    everywhere in HAWK, that two planes bound to one buffer coincide
    exactly or do not overlap."""
    return walk.access.cls == "sample_local"


def _host_masks(walk: Walk):
    """Each rank-0 ``bool`` ``terminated`` plane of a :func:`simd_safe`
    kernel, as ``(role, name, ttype)``."""
    if not simd_safe(walk):
        return
    for role, name in walk.arg_spec:
        ttype = walk.slot_types[(role, name)]
        if role == "terminated" and not ttype.shape and ttype.dtype == "bool":
            yield role, name, ttype


def lane_guards(walk: Walk) -> tuple:
    """The binding names of ``walk``'s ``HostMask`` planes; empty for a
    kernel that reads none, whose loop stays unblocked."""
    return tuple(binding_name(role, name) for role, name, _ in _host_masks(walk))


def lane_finish(walk: Walk) -> tuple | None:
    """The finish epilogue's ``(mask, twin, counter)`` bindings when
    ``walk`` finishes through a lane-guarded mask, else ``None``."""
    finish = getattr(walk, "finish", None)
    if finish is None:
        return None
    mask = binding_name("terminated", finish[0])
    if mask not in lane_guards(walk):
        return None
    return mask, finish_binding(finish[0]), binding_name("lookup", FINISHED_PLANE)


def _contiguous_loop(guards: Sequence[str], finish: tuple | None = None) -> str:
    """The independent-iterations loop over ``[hawk_lo, hawk_hi)``, binding
    ``hawk_j``. With ``guards`` it is also the lane-guarded form under
    ``HAWK_HOST_LANE_GUARDS``: blocks of ``kHostLaneBlock`` samples, each
    widening its guards into a float64 array and rebinding every guard's
    name, for the block's scope, to that array. Closed by
    :func:`_contiguous_close`.

    ``finish`` (:func:`lane_finish`) is the finish epilogue's ``(mask,
    twin, counter)`` bindings when the mask it sets is one of ``guards``:
    the twin rebinds to that guard's lane array and the counter to a
    block-local tally, committed by :func:`_contiguous_close`."""
    plain = (f"{_INDEPENDENT_ITERATIONS}\n"
             "    for (aether::offset_t hawk_j = hawk_lo; hawk_j < hawk_hi; ++hawk_j) {")
    if not guards:
        return plain
    lines = ["#if HAWK_HOST_LANE_GUARDS"]
    lines += [f"    alignas(64) HAWK_LANE_T hawk_lanes{k}[hawk_abi::kHostLaneBlock];"
              for k in range(len(guards))]
    lines += ["    for (aether::offset_t hawk_blo = hawk_lo; hawk_blo < hawk_hi;",
              "         hawk_blo += hawk_abi::kHostLaneBlock) {",
              "    const aether::offset_t hawk_bhi = hawk_abi::lane_block_end(hawk_blo, hawk_hi);"]
    lines += [f"    const hawk_abi::HostLaneMask hawk_lane{k} =\n"
              f"        {g}.lanes(hawk_lanes{k}, hawk_blo, hawk_bhi);"
              for k, g in enumerate(guards)]
    if finish is not None:
        mask, twin, counter = finish
        k = guards.index(mask)
        lines += [f"    const hawk_abi::HostMask hawk_finish_mask = {mask};",
                  f"    const auto& hawk_finish_count = {counter};",
                  "    HAWK_LANE_T hawk_finished = 0.0;"]
    lines += [f"    const hawk_abi::HostLaneMask {g} = hawk_lane{k};"
              for k, g in enumerate(guards)]
    if finish is not None:
        lines += [f"    const hawk_abi::HostLaneFinish {twin}{{hawk_lanes{k}, hawk_blo}};",
                  f"    const hawk_abi::HostLaneTally {counter}{{&hawk_finished}};",
                  f"    (void){twin};",
                  f"    (void){counter};"]
    lines += [_INDEPENDENT_ITERATIONS,
              "    for (aether::offset_t hawk_j = hawk_blo; hawk_j < hawk_bhi; ++hawk_j) {",
              "#else", plain, "#endif"]
    return "\n".join(lines)


def _finish_args(finish: tuple | None) -> tuple:
    """The finish twin/counter bindings an active-set per-sample lambda
    takes after the guards; empty without ``finish``."""
    return () if finish is None else tuple(finish[1:])


def _contiguous_close(guards: Sequence[str], finish: tuple | None = None) -> str:
    """Closes :func:`_contiguous_loop`, committing finished lanes first."""
    if not guards:
        return "    }"
    commit = ""
    if finish is not None:
        k = guards.index(finish[0])
        commit = (f"\n    hawk_abi::finish_lanes(hawk_finish_mask, hawk_lanes{k}, "
                  "hawk_blo, hawk_bhi, hawk_finished, hawk_finish_count);")
    return f"    }}\n#if HAWK_HOST_LANE_GUARDS{commit}\n    }}\n#endif"


# --------------------------------------------------------------------------- #
# The interchanged fused-step loop
# --------------------------------------------------------------------------- #
#: The macro an interchanged TU defines (to 1 unless the build already
#: defined it): ``-DHAWK_HOST_INTERCHANGE=0`` builds the SAME TU's plain
#: per-sample fused loop instead, which is how the old shape stays reachable
#: (``build_bundle(..., defines=("HAWK_HOST_INTERCHANGE=0",))``).
INTERCHANGE_MACRO = "HAWK_HOST_INTERCHANGE"

#: The lane-array helpers an interchanged TU adds after the mask helpers. A
#: value of rank >= 1 (an ``aether::Item``) is stored one component per row,
#: component ``c`` of lane ``t`` at ``lanes[c * Tile + t]``, so every row is a
#: unit-stride run the vectoriser loads whole.
INTERCHANGE_HELPERS = """namespace hawk_abi {

template <aether::offset_t Tile, class T>
static inline void lane_put(T* lanes, aether::offset_t t, const T& v)
{
    lanes[t] = v;
}

template <aether::offset_t Tile, class T, std::size_t... Es>
static inline void lane_put(T* lanes, aether::offset_t t,
                            const aether::Item<T, Es...>& v)
{
    for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c)
        lanes[static_cast<aether::offset_t>(c) * Tile + t] = v.data()[c];
}

// A finished lane keeps its old value: `on ? v : old`, one select per row.
template <aether::offset_t Tile, class T>
static inline void lane_keep(T* lanes, aether::offset_t t, bool on, const T& v)
{
    lanes[t] = on ? v : lanes[t];
}

template <aether::offset_t Tile, class T, std::size_t... Es>
static inline void lane_keep(T* lanes, aether::offset_t t, bool on,
                             const aether::Item<T, Es...>& v)
{
    for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c) {
        T& slot = lanes[static_cast<aether::offset_t>(c) * Tile + t];
        slot = on ? v.data()[c] : slot;
    }
}

template <class V>
struct LaneOf {
    using type = V;
    static V get(const V* lanes, aether::offset_t, aether::offset_t t)
    {
        return lanes[t];
    }
};

template <class T, std::size_t... Es>
struct LaneOf<aether::Item<T, Es...>> {
    using type = T;
    static aether::Item<T, Es...> get(const T* lanes, aether::offset_t tile,
                                      aether::offset_t t)
    {
        aether::Item<T, Es...> v;
        for (std::size_t c = 0; c < (std::size_t{1} * ... * Es); ++c)
            v.data()[c] = lanes[static_cast<aether::offset_t>(c) * tile + t];
        return v;
    }
};

template <class V, aether::offset_t Tile>
static inline V lane_get(const typename LaneOf<V>::type* lanes, aether::offset_t t)
{
    return LaneOf<V>::get(lanes, Tile, t);
}

// The lanes one chunk of the interchanged loop advances together: one vector
// register of float64 (8 under AVX-512, 4 under AVX, 2 otherwise), so a
// chunk's carried values stay in registers across its trips.
#if defined(__AVX512F__)
inline constexpr aether::offset_t kHostChunk = 8;
#elif defined(__AVX__)
inline constexpr aether::offset_t kHostChunk = 4;
#else
inline constexpr aether::offset_t kHostChunk = 2;
#endif

}  // namespace hawk_abi
"""

#: A tile's size bounds, in samples: at least one lane block
#: (``kHostLaneBlock``), at most what one launch's tile of eagle's team
#: plausibly holds.
TILE_MIN, TILE_MAX = 64, 1024
#: The samples per lane block, the tile's granule (``kHostLaneBlock``).
LANE_BLOCK = 64
#: The share of a cache level a tile's lane arrays may fill; the rest holds
#: the step's temporaries, the planes' lines in flight and the stack.
L1_SHARE, L2_SHARE = 0.5, 0.5


@dataclass(frozen=True)
class CacheSizes:
    """One core's data cache sizes in bytes, and where they were read."""

    l1d: int
    l2: int
    source: str


#: What :func:`machine_caches` answers when sysfs cannot be read: a common
#: x86-64 core (32 KiB L1d, 1 MiB L2).
FALLBACK_CACHES = CacheSizes(32 * 1024, 1024 * 1024, "fallback")


def _cache_bytes(text: str) -> int:
    """A sysfs ``size`` field (``32K``, ``1024K``, ``2M``) in bytes."""
    text = text.strip().upper()
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(text[-1:], 1)
    return int(text.rstrip("KMG")) * scale


@functools.cache
def machine_caches(root: str = "/sys/devices/system/cpu/cpu0/cache") -> CacheSizes:
    """CPU 0's L1d and L2 sizes off ``root`` (Linux sysfs), read once per
    process; :data:`FALLBACK_CACHES` for any size it cannot read."""
    found: dict = {}
    try:
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("index"):
                continue
            here = os.path.join(root, entry)

            def field(name, here=here):
                with open(os.path.join(here, name)) as fh:
                    return fh.read().strip()

            level, kind = int(field("level")), field("type")
            data = kind == "Data" if level == 1 else kind != "Instruction"
            if level in (1, 2) and data:
                found.setdefault(level, _cache_bytes(field("size")))
    except (OSError, ValueError):
        return FALLBACK_CACHES
    if not found.get(1) or not found.get(2):
        return FALLBACK_CACHES
    return CacheSizes(found[1], found[2], root)


@dataclass(frozen=True)
class TileRule:
    """What :func:`host_tile_rule` decided and every input it decided from.

    ``tile`` is the samples one tile advances together, ``k`` the steps one
    launch should ask for, ``level`` the cache the tile's lane arrays were
    sized for."""

    tile: int
    k: int
    level: str
    lane_bytes: int
    op_count: int
    finishes: bool
    stop: int
    l1d: int
    l2: int
    cache_source: str

    def comment(self) -> str:
        """The rule's record as one C++ comment line, spliced into the TU."""
        return (f"// hawk host tile rule: tile={self.tile} k={self.k}"
                f" level={self.level} (lane_bytes={self.lane_bytes}"
                f" op_count={self.op_count} finishes={int(self.finishes)}"
                f" stop={self.stop} l1d={self.l1d}"
                f" l2={self.l2} caches={self.cache_source})")


def host_tile_rule(lane_bytes: int, op_count: int, finishes: bool, stop: int,
                   caches: CacheSizes | None = None) -> TileRule:
    """The interchanged loop's tile size and steps per launch, from what
    HAWK knows at IR time and the machine's cache sizes.

    ``lane_bytes`` is one sample's share of the tile's lane arrays (its
    carried values, the step's inputs computed once before the loop, the
    trip count, the live flag and the guards), ``op_count`` the step's
    operation count, ``finishes`` whether the kernel finishes its own
    samples, ``stop`` the loop's static bound; ``caches`` defaults to
    :func:`machine_caches`.

    The rule: a tile's lane arrays fill at most :data:`L1_SHARE` of L1d,
    or :data:`L2_SHARE` of L2 when not even one lane block (``64``
    samples) fits that; the tile is the most samples that fit, rounded
    down to whole lane blocks and clamped to ``[TILE_MIN, TILE_MAX]``.
    ``k`` is ``stop`` when the kernel finishes its own samples (a tile stops
    once its last sample finished, so a larger ``k`` only saves launches),
    else 1. ``op_count`` is recorded for the cost model that will replace
    this rule; no measurement ties it to the tile yet."""
    caches = caches if caches is not None else machine_caches()
    lane_bytes = max(int(lane_bytes), 1)
    l1 = int(caches.l1d * L1_SHARE)
    if lane_bytes * LANE_BLOCK <= l1:
        budget, level = l1, "L1d"
    else:
        budget, level = int(caches.l2 * L2_SHARE), "L2"
    tile = budget // lane_bytes
    tile = min(max(tile - tile % LANE_BLOCK, TILE_MIN), TILE_MAX)
    return TileRule(tile, int(stop) if finishes else 1, level, lane_bytes,
                    int(op_count), bool(finishes), int(stop), caches.l1d, caches.l2,
                    caches.source)


@dataclass(frozen=True)
class Interchange:
    """An interchanged fused-step loop for one host TU: its tile ``rule``,
    the index loop and body ``text`` that replaces the per-sample fused loop
    under :data:`INTERCHANGE_MACRO`, and the ``export`` line that tells a
    runner the TU's tile (``0`` when built with the macro off)."""

    rule: TileRule | None
    text: str
    export: str
    #: The macro's default and the helpers' text of a loop interchange
    #: (:func:`loop_interchange`), which has no tile rule; ``None`` for the
    #: fused-step loop's.
    prelude: str | None = None


#: A step of at most this many operations is short: one dependent chain per
#: lane, too few independent operations to fill the FMA pipes from one
#: vector of lanes.
SHORT_STEP_OPS = 256


def chunk_vectors(op_count: int) -> int:
    """The vector registers of lanes one chunk of the interchanged loop
    advances together: two for a short step (:data:`SHORT_STEP_OPS`), so
    two independent chains hide each other's latency, else one (a long
    step has independent work of its own and needs the registers)."""
    return 2 if op_count <= SHORT_STEP_OPS else 1


_ELEM_BYTES = {"f64": 8, "f32": 4, "i32": 8, "i64": 8, "bool": 1, "int32": 4,
               "uint32": 4}


def _bytes_of(dtype: str, real_bytes: int) -> int:
    """The bytes one element of IR ``dtype`` stores as: a float is the
    compiled ``Real`` (``real_bytes``), whichever of ``f64``/``f32`` it is."""
    return real_bytes if dtype in ("f64", "f32") else _ELEM_BYTES[dtype]


def _width(ttype) -> int:
    """The components of one value of ``ttype``."""
    return prod(ttype.shape) if ttype.shape else 1


def _lane_type(ttype) -> str:
    """The value a lane array row holds back as: the carry's own spelling."""
    return carry_type(ttype)


_TILE_LOOP_HEAD = (
    "    for (aether::offset_t hawk_tlo = hawk_lo; hawk_tlo < hawk_hi;",
    "         hawk_tlo += hawk_tile) {",
    "    const aether::offset_t hawk_thi =",
    "        hawk_hi - hawk_tlo < hawk_tile ? hawk_hi : hawk_tlo + hawk_tile;",
    "    const aether::offset_t hawk_tn = hawk_thi - hawk_tlo;",
)


def interchange_text(split: LaneSplit, guards: Sequence[str], finish: tuple,
                     tile: int) -> str:
    """The interchanged loop over ``[base, base + count)``: per tile of
    ``tile`` samples, (A) each sample's head (:attr:`LaneSplit.pre`) fills
    the lane arrays, (B) per chunk of the tile's lanes (one or two vector
    registers of them, :func:`chunk_vectors`), the chunk's values are copied
    into its own small arrays once and every trip runs the step
    (:attr:`LaneSplit.step`) across the chunk's lanes, a finished lane
    keeping its old values, until no lane of the chunk is live or the bound
    is reached, then copied back once, then (C) each sample's tail
    (:attr:`LaneSplit.post`) commits from the lanes and finishes into the
    tile's guard lanes, which ``finish_lanes`` writes back. Per sample the
    same operations run in the same order as the plain fused loop: same
    bits."""
    mask, twin, counter = finish
    on_k = guards.index(mask)
    rows = []
    arrays = []
    for j, (name, ttype, _, _) in enumerate(split.carries):
        rows.append((f"hawk_lc{j}", name, ttype))
    for m, (name, ttype) in enumerate(split.live_ins):
        rows.append((f"hawk_li{m}", name, ttype))
    for arr, _, ttype in rows:
        arrays.append(f"    alignas(64) {element_spelling(ttype.dtype)} "
                      f"{arr}[{_width(ttype)} * hawk_tile] = {{}};")
    lines = [
        f"    constexpr aether::offset_t hawk_tile = {tile};",
        "    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);",
        "    const aether::offset_t hawk_hi =",
        "        static_cast<aether::offset_t>(base + count);",
        *[f"    alignas(64) HAWK_LANE_T hawk_tg{k}[hawk_tile];" for k in range(len(guards))],
        "    alignas(64) HAWK_LANE_T hawk_on[hawk_tile] = {};",
    ]
    if split.count is not None:
        lines.append("    alignas(64) Int hawk_cnt[hawk_tile] = {};")
    lines += arrays
    lines += [
        *_TILE_LOOP_HEAD,
        *[f"    const hawk_abi::HostLaneMask hawk_tlane{k} =\n"
          f"        {g}.lanes(hawk_tg{k}, hawk_tlo, hawk_thi);"
          for k, g in enumerate(guards)],
        f"    const hawk_abi::HostMask hawk_finish_mask = {mask};",
        f"    const auto& hawk_finish_count = {counter};",
        "    HAWK_LANE_T hawk_finished = 0.0;",
        *[f"    const hawk_abi::HostLaneMask {g} = hawk_tlane{k};"
          for k, g in enumerate(guards)],
        f"    const hawk_abi::HostLaneFinish {twin}{{hawk_tg{on_k}, hawk_tlo}};",
        f"    const hawk_abi::HostLaneTally {counter}{{&hawk_finished}};",
        f"    (void){twin};",
        f"    (void){counter};",
    ]
    index = "static_cast<std::size_t>(hawk_j)"
    sample = ["        const aether::offset_t hawk_t = hawk_j - hawk_tlo;",
              "        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);",
              "        const aether::SampleIndex i =",
              f"            aether::SampleIndex::make({index});",
              "        (void)hawk_t;",
              "        (void)hawk_i;",
              "        (void)nSamples;"]
    head = "    for (aether::offset_t hawk_j = hawk_tlo; hawk_j < hawk_thi; ++hawk_j) {"

    # (A) the head of every sample of the tile: lanes filled, live flag set
    puts = []
    for (arr, name, ttype), (_, _, init, _) in zip(rows, split.carries):
        value = (f"static_cast<{_lane_type(ttype)}>({init})" if not ttype.shape
                 else f"{_lane_type(ttype)}({init})")
        puts.append(f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {value});")
    for arr, name, ttype in rows[len(split.carries):]:
        if ttype.shape:
            puts.append(f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, "
                        f"{_lane_type(ttype)}({name}));")
        else:
            same = (f"std::is_same_v<std::remove_cv_t<decltype({name})>, "
                    f"{_lane_type(ttype)}>")
            puts.append(f'static_assert({same}, "hawk: a lane row holds its value\'s '
                        'own type");')
            puts.append(f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {name});")
    if split.count is not None:
        puts.append(f"hawk_cnt[hawk_t] = {split.count};")
        puts.append(f"hawk_go = {split.count} > 0 ? 1.0 : 0.0;")
    else:
        puts.append("hawk_go = 1.0;")
    lines += [_INDEPENDENT_ITERATIONS, head, *sample, "        HAWK_LANE_T hawk_go = 0.0;"]
    pad = "            "
    if split.guard is not None:
        lines.append(f"        if (!{split.guard}) {{")
    lines.append(split.pre) if split.pre else None
    lines += [pad + p for p in puts]
    if split.guard is not None:
        lines.append("        }")
    lines += ["        hawk_on[hawk_t] = hawk_go;", "    }"]

    # (B) the trips, per chunk of hawk_abi::kHostChunk lanes (one vector
    # register each): the chunk copies its lanes' values into its own small
    # arrays once, runs every trip on them with the lanes innermost, until
    # no lane is live or the bound is reached, then copies the carries back
    # once. A lane past the tile's end is never live; it reads the tile's
    # last sample, so no index leaves the tile.
    chunk = [(f"hawk_rc{j}", arr, ttype)
             for j, (arr, _, ttype) in enumerate(rows[:len(split.carries)])]
    chunk += [(f"hawk_ri{m}", arr, ttype)
              for m, (arr, _, ttype) in enumerate(rows[len(split.carries):])]
    loads = [f"{pad}const {_lane_type(t)} {name} = "
             f"hawk_abi::lane_get<{_lane_type(t)}, hawk_tile>({arr}, hawk_t);"
             for arr, name, t in rows]
    trip_loads = [f"        const {_lane_type(t)} {name} = "
                  f"hawk_abi::lane_get<{_lane_type(t)}, hawk_w>({mine}, hawk_l);"
                  for (mine, _, _), (_, name, t) in zip(chunk, rows)]
    more = f"!{split.exit}"
    if split.count is not None:
        more += " && hawk_trip + 1 < hawk_rcnt[hawk_l]"
    lines += [
        f"    constexpr aether::offset_t hawk_w = hawk_abi::kHostChunk * "
        f"{chunk_vectors(split.op_count)};",
        "    static_assert(hawk_tile % hawk_w == 0,",
        '                  "hawk: a tile holds whole chunks");',
        "    for (aether::offset_t hawk_cb = 0; hawk_cb < hawk_tn;",
        "         hawk_cb += hawk_w) {",
        *[f"    alignas(64) {element_spelling(t.dtype)} {mine}[{_width(t)} * hawk_w];"
          for mine, _, t in chunk],
        "    alignas(64) HAWK_LANE_T hawk_ron[hawk_w];",
    ]
    if split.count is not None:
        lines.append("    alignas(64) Int hawk_rcnt[hawk_w];")
    lines += [
        "    std::int64_t hawk_any = 0;",
        _INDEPENDENT_ITERATIONS,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {",
        "        const aether::offset_t hawk_t = hawk_cb + hawk_l;",
    ]
    for mine, arr, t in chunk:
        lines.append(f"        for (aether::offset_t hawk_c = 0; hawk_c < {_width(t)}; "
                     "++hawk_c)")
        lines.append(f"            {mine}[hawk_c * hawk_w + hawk_l] = "
                     f"{arr}[hawk_c * hawk_tile + hawk_t];")
    if split.count is not None:
        lines.append("        hawk_rcnt[hawk_l] = hawk_cnt[hawk_t];")
    lines += [
        "        const bool hawk_in = hawk_t < hawk_tn && hawk_on[hawk_t] != 0.0;",
        "        hawk_ron[hawk_l] = hawk_in ? 1.0 : 0.0;",
        "        hawk_any |= hawk_in ? std::int64_t{1} : std::int64_t{0};",
        "    }",
        "    if (hawk_any == 0) continue;",
        # a runtime trip count bounds each lane itself (``hawk_more``)
        (f"    for (Int hawk_trip = 0; hawk_trip < {split.stop}; ++hawk_trip) {{"
         if split.count is None else "    for (Int hawk_trip = 0;; ++hawk_trip) {"),
        "    std::int64_t hawk_live = 0;",
        _INDEPENDENT_ITERATIONS,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {",
        "        const aether::offset_t hawk_t =",
        "            hawk_cb + hawk_l < hawk_tn ? hawk_cb + hawk_l : hawk_tn - 1;",
        "        const aether::SampleIndex i =",
        "            aether::SampleIndex::make(",
        "                static_cast<std::size_t>(hawk_tlo + hawk_t));",
        "        (void)i;",
        "        const bool hawk_lane_on = hawk_ron[hawk_l] != 0.0;",
        f"        const Int {split.index} = hawk_trip;",
        f"        (void){split.index};",
    ]
    if split.speculative:
        lines += trip_loads
        lines.append(split.step)
        lines += [f"        hawk_abi::lane_keep<hawk_w>({mine}, hawk_l, "
                  f"hawk_lane_on, {nxt});"
                  for (mine, _, _), (_, _, _, nxt) in zip(chunk, split.carries)]
        lines.append(f"        const bool hawk_more = hawk_lane_on && {more};")
    else:
        lines += ["        bool hawk_more = false;", "        if (hawk_lane_on) {"]
        lines += ["    " + p for p in trip_loads]
        lines.append(split.step)
        lines += [f"{pad}hawk_abi::lane_put<hawk_w>({mine}, hawk_l, {nxt});"
                  for (mine, _, _), (_, _, _, nxt) in zip(chunk, split.carries)]
        lines += [f"{pad}hawk_more = {more};", "        }"]
    lines += [
        "        hawk_ron[hawk_l] = hawk_more ? 1.0 : 0.0;",
        "        hawk_live |= hawk_more ? std::int64_t{1} : std::int64_t{0};",
        "    }",
        "    if (hawk_live == 0) break;",
        "    }",
        _INDEPENDENT_ITERATIONS,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_w; ++hawk_l) {",
        "        const aether::offset_t hawk_t = hawk_cb + hawk_l;",
    ]
    for mine, arr, t in chunk[:len(split.carries)]:
        lines.append(f"        for (aether::offset_t hawk_c = 0; hawk_c < {_width(t)}; "
                     "++hawk_c)")
        lines.append(f"            {arr}[hawk_c * hawk_tile + hawk_t] = "
                     f"{mine}[hawk_c * hawk_w + hawk_l];")
    lines += ["    }", "    }"]

    # (C) the tail of every sample: commits and the finish, from the lanes
    lines += [_INDEPENDENT_ITERATIONS, head, *sample]
    if split.guard is not None:
        lines.append(f"        if (!{split.guard}) {{")
    lines += loads[:len(split.carries)]
    lines.append(split.post) if split.post else None
    if split.guard is not None:
        lines.append("        }")
    lines += [
        "    }",
        f"    hawk_abi::finish_lanes(hawk_finish_mask, hawk_tg{on_k}, hawk_tlo, "
        "hawk_thi, hawk_finished, hawk_finish_count);",
        "    }",
    ]
    return "\n".join(line for line in lines if line is not None)


def interchange(name: str, sinks, walk: Walk, kind=None,
                real_bytes: int = 8) -> Interchange | None:
    """The interchanged fused-step loop of ``walk``'s host TU, or ``None``
    where the TU keeps its plain per-sample fused loop.

    Interchanged as a fused step: a :func:`simd_safe` kernel that finishes
    its own sample through a lane-guarded mask (:func:`lane_finish`) and
    whose body :func:`~hawk.emit.render.render_lane_split` admits (one
    tail-break loop, no inner ``for``). Any other :func:`simd_safe` kernel
    with a ``Loop`` runs its loops interchanged by :func:`loop_interchange`.
    A kernel that is not ``sample_local`` has dependent iterations, so its
    lanes cannot run together: it keeps its per-sample loop."""
    if not simd_safe(walk):
        return None
    finish = lane_finish(walk)
    split = None if finish is None else render_lane_split(sinks, walk, kind=kind)
    if split is None:
        return loop_interchange(sinks, walk, kind)
    guards = lane_guards(walk)
    lane_bytes = real_bytes * (1 + len(guards)) + 8 * (split.count is not None)
    for _, ttype, _, _ in split.carries:
        lane_bytes += _width(ttype) * _bytes_of(ttype.dtype, real_bytes)
    for _, ttype in split.live_ins:
        lane_bytes += _width(ttype) * _bytes_of(ttype.dtype, real_bytes)
    rule = host_tile_rule(lane_bytes, split.op_count, True, split.stop)
    text = interchange_text(split, guards, finish, rule.tile)
    export = (f'extern "C" const std::int64_t {name}_host_tile = '
              f"{INTERCHANGE_MACRO} ? {rule.tile} : 0;")
    run = run_entry(name, walk, walk.finish[0], split.stop)
    if run is not None:
        export = f"{export}\n{run}"
    return Interchange(rule, text, export)


#: The text a loop-interchanged TU puts ahead of its helpers: the macro's
#: default (``-DHAWK_HOST_INTERCHANGE=0`` builds the per-sample loop).
LOOP_PRELUDE = (f"#ifndef {INTERCHANGE_MACRO}\n#define {INTERCHANGE_MACRO} 1\n#endif\n"
                "// hawk host loop interchange: tile = hawk_abi::kHostLoopTile\n")

#: The loop interchange's tile: the lanes of EIGHT vector registers of the
#: kernel's ``Real``, so eight independent chains of a dependent loop body are
#: in flight per trip (eight covers the latency of a floating-point add or
#: multiply times the ports that issue them): 64 float64 / 128 float32 lanes
#: under AVX-512, 32 / 64 under AVX, 16 / 32 on a 16-byte ISA such as SSE2 or NEON.
LOOP_HELPERS = """namespace hawk_abi {

#if defined(__AVX512F__)
inline constexpr aether::offset_t kHostVectorBytes = 64;
#elif defined(__AVX__)
inline constexpr aether::offset_t kHostVectorBytes = 32;
#else
inline constexpr aether::offset_t kHostVectorBytes = 16;
#endif
inline constexpr aether::offset_t kHostLoopRegisters = 8;
inline constexpr aether::offset_t kHostLoopTile =
    kHostLoopRegisters * kHostVectorBytes / static_cast<aether::offset_t>(sizeof(HAWK_LANE_T));

}  // namespace hawk_abi
"""

#: A declaration in a rendered segment (``const auto t5 = ...``, ``Real
#: hawk_v3_0_s = ...``, ``aether::Item<Real, 3> hz0 = ...``): group 1 is
#: the declared name.
_DECLARATION = re.compile(r"^\s*(?:const\s+)?[\w:]+(?:<[^;=]*?>)?\s+(\w+)\s*(?:=|\[|;)",
                          re.M)
_WORDS = re.compile(r"\b\w+\b")


def loop_interchange(sinks, walk: Walk, kind=None) -> Interchange | None:
    """The loop interchange of ``walk``'s host TU, or ``None`` where the TU
    keeps its per-sample loop.

    Every loop :func:`hawk.ir.vectorise.interchanged_loops` names runs as
    ``for trip { for lane in tile { body } }`` (:func:`loop_interchange_text`);
    ``-DHAWK_HOST_INTERCHANGE=0`` builds the same TU's per-sample loop. No
    trip-count threshold and no kernel name: the decision is the IR's."""
    from ..ir.vectorise import interchanged_loops

    cut = interchanged_loops(walk)
    if not cut:
        return None
    text = loop_interchange_text(render_lane_segments(sinks, walk, cut, kind=kind))
    if text is None:
        return None
    return Interchange(None, text, "", LOOP_PRELUDE)


def _row_put(arr: str, name: str, ttype, value: str | None = None) -> list:
    """Store ``name`` (or the text ``value``, a carry's initial value) into
    lane ``hawk_t`` of row ``arr``."""
    lane = _lane_type(ttype)
    if value is not None:
        cast = f"static_cast<{lane}>({value})" if not ttype.shape else f"{lane}({value})"
        return [f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {cast});"]
    if ttype.shape:
        return [f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {lane}({name}));"]
    same = f"std::is_same_v<std::remove_cv_t<decltype({name})>, {lane}>"
    return [f'static_assert({same}, "hawk: a lane row holds its value\'s own type");',
            f"hawk_abi::lane_put<hawk_tile>({arr}, hawk_t, {name});"]


def loop_interchange_text(segs: LaneSegments) -> str | None:
    """The interchanged loops over ``[base, base + count)``, per tile of
    ``hawk_abi::kHostLoopTile`` samples: each segment of the body
    (:class:`~hawk.emit.render.LaneSegments`) runs per sample of the tile,
    each loop between two segments runs its trips OUTSIDE and the tile's
    lanes inside. Every value a later text reads (a carried value included)
    lives in a lane row ``hawk_r_<name>[c * tile + lane]``. A lane is live
    from the start of a loop while its sample is unguarded; a loop with a
    ``break`` stops a lane at its break (``hawk_more``) and the trips stop
    once no lane of the tile is live. A stopped lane keeps its values by
    select (a speculative trip) or skips the trip (any other). Per sample the
    same operations run in the same order as the per-sample loop: under an
    exact profile the same bits.

    ``None`` when a value crossing a cut has no type the renderer recorded."""
    n = len(segs.loops)
    declared = [set(_DECLARATION.findall(t)) for t in segs.segments]
    seg_words = [set(_WORDS.findall(t)) for t in segs.segments]
    step_words = [set(_WORDS.findall(lp.step)) for lp in segs.loops]
    for k, lp in enumerate(segs.loops):
        # the carries' initial values and a runtime count's bounds are
        # evaluated at the end of segment k
        tail = [init for _, _, init, _ in lp.carries]
        if lp.bounds is not None:
            tail += [lp.bounds[1], lp.bounds[3]]
        seg_words[k] |= set(_WORDS.findall(" ".join(tail)))
    rows: dict = {}      # name -> (ttype, origin): ("seg", s) or ("loop", k)
    for s in range(n + 1):
        later = set().union(*seg_words[s + 1:], *step_words[s:])
        for name in sorted(declared[s] & later):
            if name not in segs.types:
                return None
            rows[name] = (segs.types[name], ("seg", s))
    for k, lp in enumerate(segs.loops):
        for name, ttype, _, _ in lp.carries:
            rows[name] = (ttype, ("loop", k))

    def arr(name: str) -> str:
        return f"hawk_r_{name}"

    def before(origin, stage: int) -> bool:
        # a row is readable in segment `stage` once written: by an earlier
        # segment, or by a loop that ran before it
        kind, at = origin
        return at < stage

    def loads(names, lane: str) -> list:
        return [f"const {_lane_type(rows[x][0])} {x} = "
                f"hawk_abi::lane_get<{_lane_type(rows[x][0])}, hawk_tile>({arr(x)}, {lane});"
                for x in sorted(names)]

    lines = [
        "    constexpr aether::offset_t hawk_tile = hawk_abi::kHostLoopTile;",
        "    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);",
        "    const aether::offset_t hawk_hi =",
        "        static_cast<aether::offset_t>(base + count);",
        "    alignas(64) HAWK_LANE_T hawk_on[hawk_tile] = {};",
        "    alignas(64) HAWK_LANE_T hawk_ron[hawk_tile] = {};",
        *[f"    alignas(64) {element_spelling(t.dtype)} {arr(x)}[{_width(t)} * hawk_tile] = {{}};"
          for x, (t, _) in sorted(rows.items())],
        *[f"    alignas(64) {lp.bounds[0]} hawk_b{k}[hawk_tile] = {{}};\n"
          f"    alignas(64) {lp.bounds[0]} hawk_e{k}[hawk_tile] = {{}};"
          for k, lp in enumerate(segs.loops) if lp.bounds is not None],
        *_TILE_LOOP_HEAD,
    ]
    pad = "            "
    for s in range(n + 1):
        # the segment, per sample of the tile
        lines += [
            _INDEPENDENT_ITERATIONS,
            "    for (aether::offset_t hawk_j = hawk_tlo; hawk_j < hawk_thi; ++hawk_j) {",
            "        const aether::offset_t hawk_t = hawk_j - hawk_tlo;",
            "        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);",
            "        const aether::SampleIndex i =",
            "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));",
            "        (void)hawk_t;",
            "        (void)hawk_i;",
            "        (void)nSamples;",
        ]
        if s == 0:
            lines.append("        HAWK_LANE_T hawk_go = 0.0;")
        if segs.guard is not None:
            lines.append(f"        if (!{segs.guard}) {{")
        lines += [pad + x for x in loads(
            [x for x, (_, o) in rows.items() if x in seg_words[s] and before(o, s)
             and not (o == ("seg", s))], "hawk_t")]
        if segs.segments[s]:
            lines.append(segs.segments[s])
        for x, (t, o) in sorted(rows.items()):
            if o == ("seg", s):
                lines += [pad + y for y in _row_put(arr(x), x, t)]
        if s < n:
            for x, t, init, _ in segs.loops[s].carries:
                lines += [pad + y for y in _row_put(arr(x), x, t, init)]
            bounds = segs.loops[s].bounds
            if bounds is not None:
                lines += [f"{pad}hawk_b{s}[hawk_t] = {bounds[1]};",
                          f"{pad}hawk_e{s}[hawk_t] = {bounds[3]};"]
        if s == 0:
            lines.append(pad + "hawk_go = 1.0;")
        if segs.guard is not None:
            lines.append("        }")
        if s == 0:
            lines.append("        hawk_on[hawk_t] = hawk_go;")
        lines.append("    }")
        if s == n:
            break
        lines += _trips(segs.loops[s], k=s, rows=rows, step_words=step_words[s],
                        loads=loads, arr=arr)
    lines.append("    }")
    return "\n".join(lines)


def _trips(lp, *, k: int, rows: dict, step_words: set, loads, arr) -> list:
    """The trips of one interchanged loop: trips outside, the tile's lanes
    inside (see :func:`loop_interchange_text`)."""
    pad = "            "
    counted = lp.bounds is not None
    stops = lp.exit != "false" or counted
    if counted:
        # the header's own index type, begin and end, per lane
        itype, _, cmp_, _ = lp.bounds
        index = f"hawk_b{k}[hawk_l] + static_cast<{itype}>(hawk_trip) * {lp.by}"
        runs = f" && hawk_b{k}[hawk_l] {cmp_} hawk_e{k}[hawk_l]"
        more = f" && {lp.index} + {lp.by} {cmp_} hawk_e{k}[hawk_l]"
    else:
        itype, runs, more = "Int", "", ""
        index = ("hawk_trip" if (lp.start, lp.by) == (0, 1)
                 else f"static_cast<Int>({lp.start}) + hawk_trip * static_cast<Int>({lp.by})")
    reads = [x for x, (_, o) in rows.items() if x in step_words
             and (o[0] == "seg" and o[1] <= k or o[0] == "loop" and o[1] <= k)]
    lines = [
        "    {",
        "    std::int64_t hawk_any = 0;",
        _INDEPENDENT_ITERATIONS,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_tile; ++hawk_l) {",
        f"        const bool hawk_in = hawk_l < hawk_tn && hawk_on[hawk_l] != 0.0{runs};",
        "        hawk_ron[hawk_l] = hawk_in ? 1.0 : 0.0;",
        "        hawk_any |= hawk_in ? std::int64_t{1} : std::int64_t{0};",
        "    }",
        (f"    for (Int hawk_trip = 0; hawk_any != 0 && hawk_trip < {lp.trips}; ++hawk_trip) {{"
         if not counted else "    for (Int hawk_trip = 0; hawk_any != 0; ++hawk_trip) {"),
    ]
    if stops:
        lines.append("    std::int64_t hawk_live = 0;")
    lines += [
        _INDEPENDENT_ITERATIONS,
        "    for (aether::offset_t hawk_l = 0; hawk_l < hawk_tile; ++hawk_l) {",
        "        const aether::offset_t hawk_t = hawk_l < hawk_tn ? hawk_l : hawk_tn - 1;",
        "        const aether::SampleIndex i =",
        "            aether::SampleIndex::make(static_cast<std::size_t>(hawk_tlo + hawk_t));",
        "        (void)i;",
        "        const bool hawk_lane_on = hawk_ron[hawk_l] != 0.0;",
        f"        const {itype} {lp.index} = {index};",
        f"        (void){lp.index};",
    ]
    nexts = [(x, nxt) for x, _, _, nxt in lp.carries]
    if lp.speculative:
        lines += ["        " + x for x in loads(reads, "hawk_l")]
        lines.append(lp.step)
        lines += [f"        hawk_abi::lane_keep<hawk_tile>({arr(x)}, hawk_l, hawk_lane_on, {nxt});"
                  for x, nxt in nexts]
        if stops:
            lines.append(f"        const bool hawk_more = hawk_lane_on && !{lp.exit}{more};")
    else:
        if stops:
            lines.append("        bool hawk_more = false;")
        lines.append("        if (hawk_lane_on) {")
        lines += [pad + x for x in loads(reads, "hawk_l")]
        lines.append(lp.step)
        lines += [f"{pad}hawk_abi::lane_put<hawk_tile>({arr(x)}, hawk_l, {nxt});"
                  for x, nxt in nexts]
        if stops:
            lines.append(f"{pad}hawk_more = !{lp.exit}{more};")
        lines.append("        }")
    if stops:
        lines += ["        hawk_ron[hawk_l] = hawk_more ? 1.0 : 0.0;",
                  "        hawk_live |= hawk_more ? std::int64_t{1} : std::int64_t{0};"]
    lines.append("    }")
    if stops:
        lines.append("    if (hawk_live == 0) break;")
    lines += ["    }", "    }"]
    return lines


def run_entry(name: str, walk: Walk, mask: str, stop: int) -> str | None:
    """The run-to-completion entry of an automatic kernel's host TU, or
    ``None`` for a TU without the ``fused_steps`` word.

    ``<name>_host_run`` has the entry's own signature and reads the word as
    the steps LEFT in the run. Over its ``[base, base + count)`` it calls
    ``<name>_host`` in rounds, each with a private word of ``min(stop,
    left)`` (the same words, in the same order, that one launch per round of
    the whole batch would pass), until every sample of the range is
    finished or no step is left. A caller's team then makes ONE pass over
    the batch per run instead of one per launch; each sample runs the same
    calls as before, so its results are the same bits. The rounds the
    busiest range ran are max-folded into ``<name>_host_run_rounds`` (the
    caller zeroes it before a run and reads it after; one run per artifact
    at a time), and ``<name>_host_run_k`` is ``stop``, the largest word a
    round passes. There is no threading here: the caller's team runs the
    ranges."""
    from ..ir.nodes import FUSED_STEPS_PLANE

    spec = list(walk.arg_spec)
    try:
        word = spec.index(("lookup", FUSED_STEPS_PLANE))
        done = spec.index(("terminated", mask))
    except ValueError:
        return None
    handle = "eagle::plugin::ScalarHandle"
    return f"""extern "C" const std::int64_t {name}_host_run_k = {stop};
extern "C" std::int64_t {name}_host_run_rounds = 0;
extern "C" void {name}_host(void* const* params,
    std::int64_t base,
    std::int64_t count,
    std::int64_t nSamples);
extern "C" void {name}_host_run(void* const* params,
    std::int64_t base,
    std::int64_t count,
    std::int64_t nSamples)
{{
    void* hawk_own[{len(spec)}];
    for (std::size_t hawk_s = 0; hawk_s < {len(spec)}; ++hawk_s)
        hawk_own[hawk_s] = params[hawk_s];
    {handle} hawk_word = *static_cast<const {handle}*>(params[{word}]);
    std::int64_t hawk_left = static_cast<std::int64_t>(
        static_cast<const Int*>(hawk_word.data)[0]);
    Int hawk_k = 0;
    hawk_word.data = &hawk_k;
    hawk_own[{word}] = &hawk_word;
    const unsigned char* hawk_done = static_cast<const unsigned char*>(
        static_cast<const {handle}*>(params[{done}])->data);
    std::int64_t hawk_rounds = 0;
    while (hawk_left > 0) {{
        bool hawk_live = false;
        for (std::int64_t hawk_s = base; hawk_s < base + count; ++hawk_s) {{
            if (hawk_done[hawk_s] == 0) {{
                hawk_live = true;
                break;
            }}
        }}
        if (!hawk_live) break;
        hawk_k = static_cast<Int>(hawk_left < {stop} ? hawk_left : {stop});
        {name}_host(hawk_own, base, count, nSamples);
        hawk_left -= static_cast<std::int64_t>(hawk_k);
        ++hawk_rounds;
    }}
    std::atomic_ref<std::int64_t> hawk_most({name}_host_run_rounds);
    std::int64_t hawk_seen = hawk_most.load(std::memory_order_relaxed);
    while (hawk_seen < hawk_rounds
           && !hawk_most.compare_exchange_weak(hawk_seen, hawk_rounds,
                                               std::memory_order_relaxed)) {{
    }}
}}"""


def interchange_prelude(rule: TileRule) -> str:
    """The text an interchanged TU puts ahead of the mask helpers: the
    macro's default and the tile rule's record."""
    if rule.op_count <= SHORT_STEP_OPS:
        default = f"#define {INTERCHANGE_MACRO} 1\n"
    else:
        # A long step vectorises across lanes only from GCC 14 on (measured:
        # GCC 11 leaves the chunk's lane loop scalar, and a scalar chunk pays
        # every finished lane's trips); below it the per-sample loop wins.
        default = (f"#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 14\n"
                   f"#define {INTERCHANGE_MACRO} 0\n#else\n"
                   f"#define {INTERCHANGE_MACRO} 1\n#endif\n")
    return f"#ifndef {INTERCHANGE_MACRO}\n{default}#endif\n{rule.comment()}\n"


class HostWrapper:
    """The host half of the seam."""

    #: Ordinary TU-local inlining; a host entry is not a launch hot path.
    qualifier = "static inline"

    def entry_signature(self, walk: Walk, name: str) -> str:
        """The entry. Role args arrive in the packed ``params[]`` block,
        which :meth:`unpack` reads in ``arg_spec`` order."""
        return (f'extern "C" void {name}_host(void* const* params,\n'
                f"    std::int64_t base,\n    std::int64_t count,\n"
                f"    std::int64_t nSamples)")

    def index_prologue(self, segment: SegmentSpec | None = None, *,
                       active: ActiveSpec | None = None, simd: bool = False,
                       guards: Sequence[str] = (),
                       finish: tuple | None = None, box=None) -> str:
        """The SERIAL range over ``[base, base + count)``. No OpenMP, no
        thread, no tile: those are eagle's.

        ``simd`` (:func:`simd_safe`) writes the vectorisable form; results
        match the plain loop. ``guards`` (:func:`lane_guards`) are the
        ``HostMask`` bindings a block widens (ignored without ``simd``).
        ``finish`` (:func:`lane_finish`) moves a finish epilogue on one of
        those guards into the block's lanes.

        ``active`` (:class:`ActiveSpec`) visits positions ``[base,
        min(base + count, live))`` off the map, the serial twin of the
        device's map prologue; with ``simd`` it opens the per-sample
        lambda :meth:`index_epilogue` calls.

        ``segment`` maps ``tid`` onto ``offsets[j] + tid`` instead of
        ``base + tid``, for ``tid`` in ``[base, base + count)``, stopping once
        ``tid`` passes the device-read run length, since the run is contiguous
        and sorted: a team may split a run across threads.

        ``box`` (:class:`hawk.emit.box.Box`) visits threads ``[base, base +
        count)``, each ending at its first owned lane (the device's thread
        box, serially; :func:`hawk.emit.box.frame` binds ``i``)."""
        if box is not None:
            return f"""    const std::int64_t hawk_end = base + count;
    for (std::int64_t hawk_t = base; hawk_t < hawk_end; ++hawk_t) {{
{owned_lane("long long", "hawk_t", box)}
        if ({BOX_OWN} >= nSamples) break;"""
        guards = tuple(guards) if simd else ()
        finish = finish if finish is not None and finish[0] in guards else None
        if active is not None:
            if segment is not None:
                raise HawkError("an active-set index map and a segmented unit "
                                "cannot share one prologue")
            count_expr, sample_expr = active_reads(active, "hawk_t")
            bound = f"""    const std::int64_t hawk_live = static_cast<std::int64_t>({count_expr});
    const std::int64_t hawk_end = base + count;
    const std::int64_t hawk_stop = hawk_end < hawk_live ? hawk_end : hawk_live;
"""
            if not simd:
                # a grouped map's last group may run past the batch
                past = ("        if (hawk_i >= nSamples) continue;\n"
                        if active.group > 1 else "")
                return bound + f"""    for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t) {{
        const std::int64_t hawk_i = static_cast<std::int64_t>({sample_expr});
{past}        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;"""
            # The finish twin is written through, so it's taken by
            # forwarding reference; guards and counter are read-only.
            fargs = _finish_args(finish)
            params = "".join(f", const auto& {g}" for g in guards)
            if fargs:
                params += f", auto&& {fargs[0]}, const auto& {fargs[1]}"
            return bound + f"""    auto hawk_sample = [&](const std::int64_t hawk_i{params})
        __attribute__((always_inline)) {{
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;"""
        if segment is None and simd:
            return f"""    const aether::offset_t hawk_lo =
        static_cast<aether::offset_t>(base);
    const aether::offset_t hawk_hi =
        static_cast<aether::offset_t>(base + count);
{_contiguous_loop(guards, finish)}
        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));
        (void)hawk_i;
        (void)nSamples;"""
        if segment is None:
            return """    const std::int64_t hawk_end = base + count;
    for (std::int64_t hawk_i = base; hawk_i < hawk_end; ++hawk_i) {
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;"""
        seg_begin, seg_end = segment_reads(segment)
        if simd:
            # The `break` becomes the loop bound: since the run is
            # contiguous, `[begin, min(begin + count, end))` is exactly
            # what the plain loop visits before breaking.
            return f"""    const std::int64_t hawk_seg_end = static_cast<std::int64_t>({seg_end});
    const std::int64_t hawk_seg_begin = static_cast<std::int64_t>({seg_begin}) + base;
    const std::int64_t hawk_seg_stop = hawk_seg_begin + count < hawk_seg_end
        ? hawk_seg_begin + count : hawk_seg_end;
    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(hawk_seg_begin);
    const aether::offset_t hawk_hi = hawk_seg_stop > hawk_seg_begin
        ? static_cast<aether::offset_t>(hawk_seg_stop) : hawk_lo;
{_contiguous_loop(guards, finish)}
        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));
        (void)hawk_i;
        (void)nSamples;"""
        return f"""    const std::int64_t hawk_seg_begin = static_cast<std::int64_t>({seg_begin});
    const std::int64_t hawk_seg_end = static_cast<std::int64_t>({seg_end});
    for (std::int64_t hawk_tid = base; hawk_tid < base + count; ++hawk_tid) {{
        const std::int64_t hawk_i = hawk_seg_begin + hawk_tid;
        if (hawk_i >= hawk_seg_end) break;
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;"""

    def sample_sum_frame(self, width: int, elem: str, store: str,
                         op: str = "sum", live: str | None = None) -> tuple:
        """The serial loop over the rows of 32 samples that START in
        ``[base, base + count)``, the last one cut at ``nSamples``: a row
        belongs to exactly one call however the team tiles the run. Each
        row's terms are summed by the device's pairs once its last sample
        ran."""
        opened = f"""    const std::int64_t hawk_lo = (base + 31) / 32 * 32;
    std::int64_t hawk_hi = (base + count + 31) / 32 * 32;
    if (hawk_hi > nSamples) hawk_hi = nSamples;
    {elem} hawk_ss_rows[32 * {width}] = {{}};
{sample_sum_fill(op, 32 * width, "hawk_ss_rows", elem)}    std::int64_t hawk_row = 0;
{store}
    for (std::int64_t hawk_i = hawk_lo; hawk_i < hawk_hi; ++hawk_i) {{
        {elem}* const {SAMPLE_SUM_TERMS} = hawk_ss_rows + (hawk_i & 31) * {width};
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));""" + (
            "" if live is None else f"""
        if (hawk_i < static_cast<std::int64_t>({live})) {{""")
        # a sample at or past the prefix count adds the identity (its row is still stored)
        ident = "0" if op == "sum" else f"hawk_abi::ssIdentity_<{SAMPLE_SUM_OP_CODE[op]}, {elem}>()"
        skip = "" if live is None else f"""
        }} else {{
            for (int hawk_k = 0; hawk_k < {width}; ++hawk_k) {SAMPLE_SUM_TERMS}[hawk_k] = {ident};
        }}
"""
        closed = skip + f"""        if ((hawk_i & 31) == 31 || hawk_i + 1 == hawk_hi) {{
            hawk_row = hawk_i >> 5;
            hawk_abi::sampleSumRow_<{width}, {SAMPLE_SUM_OP_CODE[op]}>(hawk_ss_rows, hawk_ss_store);
        }}
    }}"""
        return opened, closed

    def index_epilogue(self, segment: SegmentSpec | None = None, *,
                       active: ActiveSpec | None = None, simd: bool = False,
                       guards: Sequence[str] = (),
                       finish: tuple | None = None) -> str:
        """Closes what :meth:`index_prologue` opened for the same arguments.

        For a :func:`simd_safe` active-set kernel that is the per-sample
        lambda, then its two call loops: the contiguous loop when every
        position of the tile maps to itself (checked by OR-reducing
        ``map[t] ^ t``, ends tested first so a scattered map skips the
        scan), else the map loop. Both visit the same samples in order."""
        guards = tuple(guards) if simd else ()
        finish = finish if finish is not None and finish[0] in guards else None
        if active is not None and simd:
            def at(pos: str) -> str:
                return f"static_cast<std::int64_t>({active_reads(active, pos)[1]})"
            args = "".join(f", {g}" for g in (*guards, *_finish_args(finish)))
            return f"""    }};
    bool hawk_identity = base >= hawk_stop
        || ({at("base")} == base && {at("hawk_stop - 1")} == hawk_stop - 1);
    if (hawk_identity) {{
        std::int64_t hawk_strays = 0;
        for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t)
            hawk_strays |= {at("hawk_t")} ^ hawk_t;
        hawk_identity = hawk_strays == 0;
    }}
    if (hawk_identity) {{
    const aether::offset_t hawk_lo = static_cast<aether::offset_t>(base);
    const aether::offset_t hawk_hi = hawk_stop > base
        ? static_cast<aether::offset_t>(hawk_stop) : hawk_lo;
{_contiguous_loop(guards, finish)}
        hawk_sample(static_cast<std::int64_t>(hawk_j){args});
{_contiguous_close(guards, finish)}
    }} else {{
{_INDEPENDENT_ITERATIONS}
    for (std::int64_t hawk_t = base; hawk_t < hawk_stop; ++hawk_t) {{
        hawk_sample({at("hawk_t")}{args});
    }}
    }}"""
        if active is None and simd:
            return _contiguous_close(guards, finish)
        return "    }"

    def unpack(self, walk: Walk) -> str:
        """The prologue: each slot's mirror is ``params[k]``, ``k`` its
        ``arg_spec`` index.

        For a :func:`simd_safe` kernel a rank-0 ``terminated`` plane binds
        as :data:`HOST_MASK_HELPERS`' ``HostMask`` instead of
        ``aether::View<bool>`` (same mirror, bytes and ``[i].eval()``
        answer)."""
        def source(slot: Slot) -> str:
            k = walk.slot_of[(slot.role, slot.name)]
            return reconstruct_call(slot.role, slot.ttype, _deref(slot, k))
        text = unpack_lines(walk, source)
        for role, name, ttype in _host_masks(walk):
            slot = Slot(role, name, ttype)
            k = walk.slot_of[(role, name)]
            old = (f"    const {view_type(role, ttype)} {binding_name(role, name)} =\n"
                   f"        {reconstruct_call(role, ttype, _deref(slot, k))};")
            new = (f"    const hawk_abi::HostMask {binding_name(role, name)} =\n"
                   f"        hawk_abi::host_mask({_deref(slot, k)});")
            if text.count(old) != 1:                         # pragma: no cover
                raise HawkError(f"host unpack: cannot rebind mask {name!r}")
            text = text.replace(old, new)
        return text

    def simd_safe(self, walk: Walk) -> bool:
        """:func:`simd_safe`, as the hook
        :func:`~hawk.emit.wrapper.render_source` asks a wrapper for."""
        return simd_safe(walk)

    def lane_guards(self, walk: Walk) -> tuple:
        """:func:`lane_guards`, as the hook asks; reaches both halves of
        the index seam."""
        return lane_guards(walk)

    def lane_finish(self, walk: Walk) -> tuple | None:
        """:func:`lane_finish`, as the hook asks; reaches both halves of
        the index seam."""
        return lane_finish(walk)

    def extra_prelude(self, walk: Walk, *, tiled: Interchange | None = None) -> str:
        """Host-only namespace-scope text after the shared prelude: the mask
        helpers, when :meth:`unpack` binds through them. ``tiled`` (an
        :class:`Interchange`) adds the macro, the rule's record and the
        lane helpers, and builds the finish helpers whenever the macro is
        on, lane guards or not."""
        masks = simd_safe(walk) and any(
            role == "terminated" for role, _ in walk.arg_spec)
        if tiled is not None and tiled.prelude is not None:
            return (tiled.prelude + (HOST_MASK_HELPERS if masks else "") + "\n"
                    + INTERCHANGE_HELPERS + "\n" + LOOP_HELPERS)
        if masks:
            if tiled is None:
                return HOST_MASK_HELPERS
            helpers = HOST_MASK_HELPERS.replace(
                "#if HAWK_HOST_LANE_GUARDS\nstruct HostLaneFinishRef",
                f"#if HAWK_HOST_LANE_GUARDS || {INTERCHANGE_MACRO}\n"
                "struct HostLaneFinishRef", 1)
            return (interchange_prelude(tiled.rule) + helpers + "\n"
                    + INTERCHANGE_HELPERS)
        return ""

    def interchange(self, name: str, sinks, walk: Walk,
                    kind=None, real_bytes: int = 8) -> Interchange | None:
        """:func:`interchange`, as the hook
        :func:`~hawk.emit.wrapper.render_source` asks a wrapper for."""
        return interchange(name, sinks, walk, kind, real_bytes)

    def dispatch(self, kind_spelling: str, branch_spellings: Sequence[str],
                 policy: str, **kw: object) -> str:
        """The per-node text — identical C++ on both targets (the bodies
        are identical C++), so this is a one-line delegate."""
        return render_dispatch_rank0(kind_spelling, branch_spellings, policy, **kw)


def _deref(slot: Slot, k: int) -> str:
    """The ``params[k]`` cast one slot's by-value mirror is read through."""
    shape = mirror_of(slot.role, slot.ttype)
    if shape == "value":
        return f"*static_cast<const {view_type(slot.role, slot.ttype)}*>(params[{k}])"
    return f"*static_cast<const eagle::plugin::{shape}*>(params[{k}])"


#: The built host wrapper.
HOST = HostWrapper()
