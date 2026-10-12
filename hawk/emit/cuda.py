# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``cuda`` backend: the device entry wrapper around the ONE body string.

Everything device-specific lives here: the ``__global__`` entry taking
its role args as by-value mirrors followed by the int64 partition
triple, the one index prologue (flat-index, early-out on ``count``,
global index ``base + flat``), and the by-value binds the view
reconstruction reads from. The grid is eagle's, never HAWK's: this file
names no launch, no stream and no occupancy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..ir import HawkError, Walk
from .backend import ActiveSpec, SegmentSpec, active_reads, segment_reads, unpack_lines
from .dispatch import render_dispatch_rank0
from .spelling import param_name, reconstruct_call


class CudaBackend:
    """The device half of the seam."""

    id = "cuda"
    #: Force-inlined: a real call would add an ABI-shim frame for nothing.
    qualifier = "__device__ __forceinline__"

    def entry_signature(self, walk: Walk, name: str) -> str:
        """The ``__global__`` entry: role args, then the int64 triple."""
        params = [f"    {_param_type(role, walk.slot_types[(role, name_)])} "
                  f"{param_name(role, name_)}" for role, name_ in walk.arg_spec]
        params += ["    long long base", "    long long count",
                   "    long long nSamples"]
        return (f'extern "C" __global__ void {name}(\n' + ",\n".join(params) + ")")

    def index_prologue(self, segment: SegmentSpec | None = None, *,
                       active: ActiveSpec | None = None) -> str:
        """THE one geometry seam: the only text in a HAWK artifact that
        may name a launch-geometry builtin.

        ``segment`` replaces ``base + flat`` with ``offsets[j] + flat``,
        reading the run's own bounds off the offsets view and exiting a
        lane past the device-read run length rather than the launch's
        own cap; ``base`` is unused in that case."""
        # The launch-geometry builtins are spelled ONCE in this file; the
        # segmented prologue is DERIVED from it by swapping the
        # own-column line, never by restating them.
        plain = """    const long long hawk_flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (hawk_flat >= count) return;
    {
        const long long hawk_i = base + hawk_flat;
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));
        (void)hawk_i;
        (void)nSamples;"""
        if active is not None:
            if segment is not None:
                raise HawkError("an active-set index map and a segmented unit "
                                "cannot share one prologue")
            # DERIVED from `plain`: the own-column line becomes a map
            # read behind a device-read live-count exit.
            count_expr, sample_expr = active_reads(active, "hawk_t")
            own = "        const long long hawk_i = base + hawk_flat;"
            assert own in plain
            return plain.replace(own, (
                "        const long long hawk_t = base + hawk_flat;\n"
                f"        if (hawk_t >= static_cast<long long>({count_expr})) return;\n"
                "        const long long hawk_i = "
                f"static_cast<long long>({sample_expr});"))
        if segment is None:
            return plain
        seg_begin, seg_end = segment_reads(segment)
        own = "        const long long hawk_i = base + hawk_flat;"
        assert own in plain
        return plain.replace(own, f"""        (void)base;
        const long long hawk_seg_begin = static_cast<long long>({seg_begin});
        const long long hawk_seg_end = static_cast<long long>({seg_end});
        const long long hawk_i = hawk_seg_begin + hawk_flat;
        if (hawk_i >= hawk_seg_end) return;""")

    def index_epilogue(self) -> str:
        return "    }"

    def unpack(self, walk: Walk) -> str:
        """The prologue: every role arg arrives by value, as the source."""
        return unpack_lines(walk, lambda slot: reconstruct_call(
            slot.role, slot.ttype, param_name(slot.role, slot.name)))

    def persist(self, name: str, sinks, walk: Walk, kind=None) -> str | None:
        """The ``<name>_persist`` entry of an automatic kernel, or ``None``
        (see :func:`persist_entry`)."""
        return persist_entry(self, name, sinks, walk, kind)

    def range(self, name: str, sinks, walk: Walk, kind=None,
              single: str = "") -> str | None:
        """The ``<name>_range`` entry of an automatic kernel, or ``None``
        (see :func:`range_entry`)."""
        return range_entry(self, name, sinks, walk, kind, single)

    def fast_prelude(self) -> str:
        """The device helper the fast-path entries share (:data:`STEPS_MAX`)."""
        return STEPS_MAX + "\n\n" + STEPS_SUM + "\n\n" + FINISH_SUM + "\n\n" + PERSIST_REPORT

    def dispatch(self, kind_spelling: str, branch_spellings: Sequence[str],
                 policy: str, **kw: object) -> str:
        """The per-node text — identical C++ on both targets."""
        return render_dispatch_rank0(kind_spelling, branch_spellings, policy, **kw)


#: The extra parameters a ``_persist`` entry takes after the partition
#: triple: the sample counter its warps claim from, an optional slot-fill
#: pair (null: not counted), the run's step reports, and the chunk length.
PERSIST_PARAMS = ("    unsigned int* hawk_next",
                  "    unsigned long long* hawk_util",
                  "    unsigned int* hawk_steps",
                  "    unsigned long long* hawk_stepsum",
                  "    unsigned int hawk_chunk")

#: A short step's loop unrolls by 4: its own bookkeeping is a large share of
#: a light step, and a heavy step has none to amortise. Measured on a P2000,
#: FP32: unroll 4 is 5-8% faster on a 40-op step, unroll 2 changes nothing,
#: unroll 8 trades the uniform batch for the spread one.
SHORT_DEVICE_STEP_OPS = 128
PERSIST_UNROLL_LINE = "        #pragma unroll 4"

#: The parameter a ``_range`` entry takes after the partition triple: the
#: run's longest per-sample step count (null: not reported).
RANGE_PARAMS = ("    unsigned int* hawk_steps",
                "    unsigned long long* hawk_stepsum")

#: The fast-path entries' one step-count report: ``atomicMax`` of a lane's
#: steps into ``cell``, the converged lanes first reduced to one value (one
#: atomic per warp). The lane id is read off the PTX register, so no launch
#: geometry is named.
STEPS_MAX = """static __device__ __forceinline__ void hawk_steps_max(unsigned int* cell,
                                                      unsigned int v)
{
    const unsigned hawk_m = __activemask();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    v = __reduce_max_sync(hawk_m, v);
#else
    if (hawk_m != 0xffffffffu) {
        if (v != 0u) atomicMax(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(hawk_m, v, o));
#endif
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0u && hawk_lid == static_cast<unsigned>(__ffs(hawk_m) - 1))
        atomicMax(cell, v);
}"""

#: The fast-path entries' step-sum report: ``atomicAdd`` of a lane's steps
#: into ``cell``, a full warp first reduced to one value (one atomic per
#: warp); a split warp adds lane by lane. The sum over the batch, against
#: the batch times its longest sample, is how much of a one-lane-per-sample
#: launch would do work -- the launching runtime's choice of entry reads it.
STEPS_SUM = """static __device__ __forceinline__ void hawk_steps_sum(unsigned long long* cell,
                                                      unsigned long long v)
{
    const unsigned hawk_m = __activemask();
    if (hawk_m != 0xffffffffu) {
        if (v != 0ull) atomicAdd(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(hawk_m, v, o);
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0ull && hawk_lid == 0u) atomicAdd(cell, v);
}"""

#: The persistent entry's finish count: a lane's finishes (counted in a
#: register, never per sample) added once at the entry's exit into the
#: finish counter's ``uint32`` cell, the cell ``hawk_abi::finish_count``
#: adds to, a full warp first reduced to one value (one atomic per warp); a
#: split warp adds lane by lane.
FINISH_SUM = """template <class ViewT>
static __device__ __forceinline__ void hawk_finish_sum(const ViewT& counter, unsigned v)
{
    unsigned int* cell = const_cast<unsigned int*>(
        reinterpret_cast<const unsigned int*>(counter.data()));
    const unsigned hawk_m = __activemask();
    if (hawk_m != 0xffffffffu) {
        if (v != 0u) atomicAdd(cell, v);
        return;
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(hawk_m, v, o);
    unsigned hawk_lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (v != 0u && hawk_lid == 0u) atomicAdd(cell, v);
}"""


#: The persistent entry's step report into the slot-fill pair (null: not
#: counted): a lane's committed steps ``ran`` into the first cell. The second,
#: the slots, is added by each warp at each chunk head: 32 lanes times the
#: chunk length, the steps a full chunk could run.
PERSIST_REPORT = """static __device__ __forceinline__ void hawk_persist_util(unsigned long long* util,
                                                         unsigned ran)
{
    if (util != nullptr) atomicAdd(&util[0], static_cast<unsigned long long>(ran));
}

// Out of line: the flush runs on a lane's wrap only, and inlined its null
// tests would have the claim loop compiled twice, once per pointer state;
// a call keeps the rare path's instructions out of the chunk head.
static __device__ __noinline__ void hawk_persist_flush(unsigned long long* util,
                                                       unsigned long long* stepsum, unsigned ran)
{
    if (util != nullptr) atomicAdd(&util[0], static_cast<unsigned long long>(ran));
    if (stepsum != nullptr) atomicAdd(stepsum, static_cast<unsigned long long>(ran));
}

// Out of line for the same reason: the chunk head keeps one predicated call.
static __device__ __noinline__ void hawk_persist_slots(unsigned long long* util, unsigned chunk)
{
    unsigned hawk_lid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));
    if (hawk_lid == 0u) atomicAdd(&util[1], static_cast<unsigned long long>(chunk) * 32ull);
}"""


def entry_counts_finished(walk: Walk, split) -> bool:
    """Whether the fast entries count a sample finished ON ENTRY into the
    finish counter (as the step's own finish epilogue counts a sample that
    finishes during the run): a self-finishing kernel whose body is guarded
    by its mask. Then, the counter zeroed before the launch, it holds the
    batch's finished count after it, and the caller needs no separate count
    of the mask."""
    return getattr(walk, "finish", None) is not None and split is not None and split.guard is not None


def _finish_counter() -> str:
    """The finish counter's binding (``hawk.ir.nodes.FINISHED_PLANE``)."""
    from ..ir.nodes import FINISHED_PLANE
    from .spelling import binding_name

    return binding_name("lookup", FINISHED_PLANE)


def _finished_on_entry() -> str:
    """One count into the finish counter for a sample finished on entry."""
    return f"hawk_abi::finish_count({_finish_counter()});"


#: The persistent entry's own finish count: one more in the lane's register
#: (:data:`FINISH_SUM` adds the lane's total once, at the entry's exit).
_PERSIST_FINISH = "++hawk_fin;"


def _launch_budget(pre: str, count: str | None) -> tuple:
    """The declarations in ``pre`` of the step budget ``count`` when it reads
    the launch's fused_steps word alone (no per-sample value), so the
    persistent entry evaluates them once, at entry; ``()`` otherwise."""
    from ..ir.nodes import FUSED_STEPS_PLANE
    from .aether import _declarations
    from .spelling import binding_name

    if count is None or not re.fullmatch(r"\w+", count):
        return ()
    decls = _declarations(pre or "", [count])
    word = binding_name("lookup", FUSED_STEPS_PLANE)
    for d in decls:
        reads = re.findall(r"\b(\w+)\[", d)
        if re.search(r"\bi\b", d) or any(r != word for r in reads):
            return ()
    return decls


def persist_entry(backend: CudaBackend, name: str, sinks, walk: Walk,
                  kind=None) -> str | None:
    """The persistent entry of an automatic (``steps="auto"``) kernel, or
    ``None`` for any other kernel.

    ``<name>_persist`` takes the fused entry's parameters, then
    ``hawk_next`` (a ``uint32`` counter the caller zeroes),
    ``hawk_util`` (two ``uint64``: the lanes' steps and their slots --
    each lane's chunks times the chunk length -- or null),
    ``hawk_steps`` (a ``uint32`` the caller zeroes: the run's longest
    per-sample step count, or null), ``hawk_stepsum`` (a ``uint64`` the
    caller zeroes: every step run, or null) and ``hawk_chunk`` (the steps
    between chunk heads, a power of two; 0 runs each sample whole). It is
    launched in whole warps (a block size that is a multiple of 32).

    The loop is warp-converged by construction. At each chunk head the lanes
    of a warp without a sample claim the next ones together: one ballot, ONE
    ``atomicAdd(hawk_next, needy lanes)`` by the warp's leader, the base
    shuffled out and each needy lane taking base plus its rank; a claimed
    sample not finished on entry loads its carried values and per-sample
    inputs (the fused body's head; the ``fused_steps`` word, the step
    budget, is read once, at entry). The warp then runs up to ``hawk_chunk`` trips of the fused step,
    each lane predicated on holding a sample: a lane whose sample exits or
    spends its budget commits it (the fused body's tail: write-back and
    finish) and is masked until the next head. A warp leaves once none of
    its lanes holds a sample after a claim past ``count``. So each sample
    runs exactly what one launch of the fused entry with that word runs, the
    same bits, whatever the grid or the chunk length; only which lane runs
    it, and when, moves. The step body holds no warp collective and no
    atomic.

    A lane holding a sample at a head claimed it at a head, so it has run a
    multiple of ``hawk_chunk`` trips: every such lane of a warp ends the chunk
    at the same trip, with no chunk counter held across the step. Each warp
    adds its slots (32 lanes times ``hawk_chunk``) to ``hawk_util[1]`` at each
    head, so ``hawk_util[0] / hawk_util[1]`` is the run's slot fill.

    No atomic runs per sample beyond the claim: a lane keeps its committed
    samples' longest trip count, step count and finish count (the step's
    finish epilogue and the on-entry count) in registers and reports them
    once, at the entry's exit (:data:`STEPS_MAX`, :data:`STEPS_SUM`,
    :data:`FINISH_SUM`), into the same cells -- the counter still holds the
    batch's finished count after the launch.
    An active-set kernel's entry is the same text: it takes that kernel's
    parameters (the map and count planes among them, unread) and indexes
    samples ``base + fetched`` directly, as the map-free kernel's does.

    Admitted: a ``sample_local`` kernel with the ``fused_steps`` word whose
    fused body cuts into head, one trip and tail
    (:func:`~hawk.emit.aether.render_lane_split`)."""
    from ..ir.nodes import FUSED_STEPS_PLANE
    from .aether import render_lane_split
    from .spelling import carry_type

    if getattr(walk, "finish", None) is None or walk.access.cls != "sample_local":
        return None
    if not any(n == FUSED_STEPS_PLANE for _, n in walk.arg_spec):
        return None
    split = render_lane_split(sinks, walk, kind=kind)
    if split is None or split.count is None:
        return None
    # The finish epilogue's per-sample count becomes a register count here.
    per_sample = _finished_on_entry()
    post = split.post.replace(per_sample, _PERSIST_FINISH) if split.post else split.post
    assert per_sample not in (post or "") and per_sample not in split.step \
        and per_sample not in (split.pre or "")
    # A carry the step never reads (the fused loop's exit flag) is not held
    # across trips: the commit reads this trip's value of it directly.
    carries, fresh = [], []
    for j, (cname, t, init, nxt) in enumerate(split.carries):
        if re.search(rf"\b{cname}\b", split.step):
            carries.append((f"hawk_pc{j}", carry_type(t), cname, init, nxt, bool(t.shape)))
        else:
            fresh.append((carry_type(t), cname, nxt))
    # A value the step reads that no sample changes (a Param and what is
    # computed from it alone) is evaluated once, at entry, instead of carried.
    hoisted = set(split.sample_free)
    free = {re.match(r"const [^=;]*?\b(\w+) = ", d).group(1) for d in hoisted}
    pre = "\n".join(line for line in (split.pre or "").splitlines() if line.strip() not in hoisted)
    live = [(f"hawk_pl{m}", carry_type(t), lname)
            for m, (lname, t) in enumerate(split.live_ins) if lname not in free]

    def put(store, ctype, value, shaped):
        return f"{store} = {ctype}({value});" if shaped else \
            f"{store} = static_cast<{ctype}>({value});"

    # The step budget is the launch's one fused_steps word: read once, at
    # entry, instead of at every claim.
    budget = _launch_budget(pre, split.count)
    if budget:
        pre = "\n".join(line for line in pre.splitlines() if line.strip() not in budget)
    # A sample claimed past its guard is not finished, and only its own lane
    # writes it, so the commit need not read the guard again.
    if split.guard is not None and post:
        post = post.replace(f"!{split.guard} && ", "")

    head = backend.entry_signature(walk, name)
    assert head.endswith("    long long nSamples)")
    head = head[:-1] + ",\n" + ",\n".join(PERSIST_PARAMS) + ")"
    head = head.replace(f"void {name}(", f"void {name}_persist(", 1)
    # Registers decide this entry's residency, so the bookkeeping is lean: the
    # sample is ONE 32-bit offset from base (its index is rebuilt where read),
    # and a lane's steps are added once per sample, at its commit.
    index = "aether::SampleIndex::make(static_cast<std::size_t>(base + hawk_s))"
    lines = [head, "{", backend.unpack(walk)]
    lines += [f"    {ctype} {store}{{}};" for store, ctype, *_ in carries]
    lines += [f"    {ctype} {store}{{}};" for store, ctype, _ in live]
    lines += [f"    {d}" for d in split.sample_free]
    lines += [f"    {d}" for d in budget]
    lines.append(f"    const int hawk_budget = hawk_clamp_count({split.count});" if budget else
                 "    int hawk_budget = 0;")
    lines += [
        "    // a lane holds a sample while it has trips left to run (hawk_trip <",
        "    // hawk_budget), and needs one while it holds none and its last claim fell",
        "    // inside count (none yet: offset 0); a sample offset is 32-bit, as the",
        "    // claim counter is",
        "    const unsigned hawk_count = static_cast<unsigned>(count);",
        "    int hawk_trip = hawk_budget;",
        "    unsigned hawk_s = 0u;",
        "    unsigned hawk_maxtrip = 0u;",
        "    unsigned hawk_fin = 0u;",
        "    unsigned hawk_ran = 0u;",
        "    (void)nSamples;",
        "    while (true) {",
        "        // the chunk head: the warp's lanes without a sample claim the next ones",
        "        // together (the ballot is the warp's meeting point)",
        "        while (true) {",
        "            const bool hawk_needy = hawk_trip >= hawk_budget && hawk_s < hawk_count;",
        "            const unsigned hawk_need = __ballot_sync(0xffffffffu, hawk_needy);",
        "            if (hawk_need == 0u) break;",
        "            // read here, not hoisted: a value held from entry would cost the step a register",
        "            unsigned hawk_lid, hawk_below;",
        '            asm volatile("mov.u32 %0, %%laneid;" : "=r"(hawk_lid));',
        '            asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(hawk_below));',
        "            const int hawk_lead = __ffs(hawk_need) - 1;",
        "            unsigned hawk_base = 0u;",
        "            if (hawk_lid == static_cast<unsigned>(hawk_lead))",
        "                hawk_base = atomicAdd(hawk_next, static_cast<unsigned>(__popc(hawk_need)));",
        "            hawk_base = __shfl_sync(0xffffffffu, hawk_base, hawk_lead);",
        "            if (hawk_needy) {",
        "                // a lane commits at most one sample between two claims, so its",
        "                // step count is reported long before it could wrap",
        "                if (hawk_ran >= (1u << 31)) {",
        "                    hawk_persist_flush(hawk_util, hawk_stepsum, hawk_ran);",
        "                    hawk_ran = 0u;",
        "                }",
        "                hawk_s = hawk_base + static_cast<unsigned>(__popc(hawk_need & hawk_below));",
        "                if (hawk_s < hawk_count) {",
        f"                    const aether::SampleIndex i = {index};",
    ]
    inner = "                    "
    if split.guard is not None:
        lines.append(f"{inner}if (!{split.guard}) {{")
        inner += "    "
    if pre:
        lines.append(pre)
    lines += [inner + put(store, ctype, init, sh) for store, ctype, _, init, _, sh in carries]
    lines += [f"{inner}{store} = {lname};" for store, _, lname in live]
    if not budget:
        lines.append(f"{inner}hawk_budget = hawk_clamp_count({split.count});")
    lines.append(f"{inner}hawk_trip = 0;")
    if split.guard is not None:
        lines.append(f"                    }} else {{ {_PERSIST_FINISH} }}"
                     if entry_counts_finished(walk, split) else "                    }")
    lines += [
        "                }",
        "            }",
        "        }",
        "        if (!__any_sync(0xffffffffu, hawk_trip < hawk_budget)) break;",
        "        if (hawk_util != nullptr) hawk_persist_slots(hawk_util, hawk_chunk);",
        "        // the chunk: a lane whose sample commits is masked until the next head;",
        "        // a lane holding a sample at a head claimed it at a head, so every such",
        "        // lane reaches the chunk's end at the same trip count",
        "        if (hawk_trip < hawk_budget)",
        PERSIST_UNROLL_LINE if split.op_count <= SHORT_DEVICE_STEP_OPS else None,
        "        do {",
        f"            const aether::SampleIndex i = {index};",
        "            (void)i;",
        f"            const Int {split.index} = hawk_trip;",
        f"            (void){split.index};",
    ]
    lines += [f"            const {ctype} {cname} = {store};"
              for store, ctype, cname, *_ in carries]
    lines += [f"            const {ctype} {lname} = {store};"
              for store, ctype, lname in live]
    lines.append(split.step)
    lines += [f"            {store} = {nxt};" for store, _, _, _, nxt, _ in carries]
    lines += [
        "            ++hawk_trip;",
        f"            if ({split.exit} || hawk_trip >= hawk_budget) {{",
        "                hawk_maxtrip = max(hawk_maxtrip, static_cast<unsigned>(hawk_trip));",
        "                hawk_ran += static_cast<unsigned>(hawk_trip);",
        "                {",
    ]
    lines += [f"                const {ctype} {cname} = {store};"
              for store, ctype, cname, *_ in carries]
    lines += [f"                const {ctype} {cname} = {nxt};" for ctype, cname, nxt in fresh]
    lines.append(post) if post else None
    lines += ["                }",
              "                hawk_trip = hawk_budget;",
              "                break;",
              "            }",
              "        } while ((static_cast<unsigned>(hawk_trip) & (hawk_chunk - 1u)) != 0u);",
              "    }",
              "    hawk_persist_util(hawk_util, hawk_ran);",
              "    if (hawk_stepsum != nullptr) hawk_steps_sum(hawk_stepsum, hawk_ran);",
              "    if (hawk_steps != nullptr) hawk_steps_max(hawk_steps, hawk_maxtrip);",
              f"    hawk_finish_sum({_finish_counter()}, hawk_fin);",
              "}"]
    return "\n".join(line for line in lines if line is not None)


def range_entry(backend: CudaBackend, name: str, sinks, walk: Walk, kind=None,
                single: str = "") -> str | None:
    """The contiguous-range entry of an automatic kernel, or ``None`` where
    :func:`persist_entry` is.

    ``<name>_range`` takes the kernel's own entry parameters, then
    ``hawk_steps`` (a ``uint32`` the caller zeroes, or null). One lane per
    sample ``base + flat``, as the kernel's own entry over a contiguous
    range: the ``fused_steps`` word ``1`` runs ``single`` (the single
    step's body), any other word the fused body cut as
    :func:`persist_entry` cuts it (head, the trips up to the step's exit
    or the word, tail) -- the same texts, so the same bits. Each lane's
    steps this launch (``1`` or the trips for a sample not finished on
    entry, else ``0``) are maxed into ``hawk_steps`` after every lane is
    done, one atomic per warp."""
    from ..ir.nodes import FUSED_STEPS_PLANE
    from .aether import render_lane_split
    from .spelling import binding_name, carry_type

    if getattr(walk, "finish", None) is None or walk.access.cls != "sample_local":
        return None
    if not any(n == FUSED_STEPS_PLANE for _, n in walk.arg_spec):
        return None
    split = render_lane_split(sinks, walk, kind=kind)
    if split is None or split.count is None:
        return None
    # DERIVED from the one geometry seam: the lane past ``count`` skips its
    # sample instead of returning, so the warp meets at the report.
    plain = backend.index_prologue()
    exit_line = "    if (hawk_flat >= count) return;\n    {"
    assert exit_line in plain
    opened = plain.replace(exit_line, "    if (hawk_flat < count) {")
    word = (f"{binding_name('lookup', FUSED_STEPS_PLANE)}"
            "[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()")
    live = "true" if split.guard is None else f"!({split.guard})"
    carries = [(f"hawk_pc{j}", carry_type(t), cname, init, nxt, bool(t.shape))
               for j, (cname, t, init, nxt) in enumerate(split.carries)]
    head = backend.entry_signature(walk, name)
    assert head.endswith("    long long nSamples)")
    head = head[:-1] + ",\n" + ",\n".join(RANGE_PARAMS) + ")"
    head = head.replace(f"void {name}(", f"void {name}_range(", 1)
    pad = "            "
    lines = [head, "{", backend.unpack(walk), "    unsigned hawk_ran = 0u;",
             f"    if ({word} == 1) {{", opened,
             f"        const bool hawk_live = {live};", single,
             "        hawk_ran = hawk_live ? 1u : 0u;",
             f"        if (!hawk_live) {_finished_on_entry()}"
             if entry_counts_finished(walk, split) else None,
             "    }", "    } else {", opened]
    if split.guard is not None:
        lines.append(f"        if (!{split.guard}) {{")
    lines.append(split.pre) if split.pre else None
    lines += [(f"{pad}{ctype} {store} = {ctype}({init});" if shaped else
               f"{pad}{ctype} {store} = static_cast<{ctype}>({init});")
              for store, ctype, _, init, _, shaped in carries]
    lines += [f"{pad}const int hawk_cnt = hawk_clamp_count({split.count});",
              f"{pad}int hawk_trip = 0;",
              f"{pad}if (hawk_cnt > 0)",
              f"{pad}#pragma unroll 4" if split.op_count <= SHORT_DEVICE_STEP_OPS else None,
              f"{pad}while (true) {{",
              f"{pad}    const Int {split.index} = hawk_trip;",
              f"{pad}    (void){split.index};"]
    lines += [f"{pad}    const {ctype} {cname} = {store};"
              for store, ctype, cname, _, _, _ in carries]
    lines.append(split.step)
    lines += [f"{pad}    {store} = {nxt};" for store, _, _, _, nxt, _ in carries]
    lines += [f"{pad}    ++hawk_trip;",
              f"{pad}    if ({split.exit} || hawk_trip >= hawk_cnt) break;",
              f"{pad}}}",
              f"{pad}{{"]
    lines += [f"{pad}const {ctype} {cname} = {store};"
              for store, ctype, cname, _, _, _ in carries]
    lines.append(split.post) if split.post else None
    lines += [f"{pad}}}", f"{pad}hawk_ran = static_cast<unsigned>(hawk_trip);"]
    if split.guard is not None:
        lines.append(f"        }} else {{ {_finished_on_entry()} }}"
                     if entry_counts_finished(walk, split) else "        }")
    lines += ["    }", "    }",
              "    if (hawk_steps != nullptr) hawk_steps_max(hawk_steps, hawk_ran);",
              "    if (hawk_stepsum != nullptr) hawk_steps_sum(hawk_stepsum, hawk_ran);", "}"]
    return "\n".join(line for line in lines if line is not None)


def _param_type(role: str, ttype) -> str:
    """The by-value ABI shape one role crosses the launch boundary as."""
    from .spelling import mirror_of, view_type

    shape = mirror_of(role, ttype)
    if shape == "value":
        return view_type(role, ttype)
    return f"eagle::plugin::{shape}"


#: The built device backend.
CUDA = CudaBackend()
