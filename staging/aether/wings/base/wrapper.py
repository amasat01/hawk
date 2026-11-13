# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wrapper seam and the scalar-mode seam.

Both seams are PROTOCOLS with their built implementations beside them,
never a class hierarchy: a wrapper supplies the entry around the
ONE body string :mod:`aether.wings.base.render` renders, and a scalar mode supplies
the ``Real`` the translation unit resolves to. Built: the ``cuda``
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

from lowering.cells import SAMPLE_SUM_COMP
from lowering.exchange import fold_entry_name
from raptor.ir.linalg import LSTSQ_OPS
from raptor.ir import Walk
from raptor.errors import RaptorError
from raptor.ir.nodes import sample_sums
from raptor.ir.types import INT_PLANE_DTYPES, Slot
from lowering import box as _box
from lowering import split as _split
from lowering import wrapper_text as _wt
from .box import frame as _box_frame
from .render import Body, render_body
from .prelude import _INT_CALL, _LINALG_CALL, _MIX_CALL, _OR_CALL, _RAY_CALL, _RT_CALL, _SPACING_CALL, prelude
from .writer_aether import RUN_CALL, RUN_STATE
from .spelling import binding_name, element_spelling, finish_binding, param_cell_binding, view_type


@dataclass(frozen=True)
class SegmentSpec:
    """Per-unit geometry override for a ``segmented`` unit's index prologue
    (``hawk.artifact.bundle`` builds the unit and hands this to the wrapper
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
            and not (active is not None and (active.group > 1 or walk.reads_sample_position)))

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


def _device_calls(wrapper, text: str) -> dict:
    """The prelude flags of the device-only helpers ``text`` names (none for a wrapper without device text)."""
    device = getattr(wrapper, "device_text", None)
    return {} if device is None else {"tiled": device.TILED_CALL in text, "warp_sum": device.WARP_SUM_CALL in text}


def _writer(target, mode: ScalarMode):
    """``target``'s writer in ``mode``: a ``cells`` mode takes the row's cells writer (the device rows hold one)."""
    if not mode.cells:
        return target.writer
    if getattr(target, "cells_writer", None) is None:
        raise RaptorError(f"the target {target.id!r} has no writer for the scalar mode {mode.id!r}")
    return target.cells_writer


class Wrapper(Protocol):
    """The wrapper seam: how a kernel is wrapped for one execution model.
    A new target with its own execution model implements exactly these members.

    ``storage`` takes the whole :class:`~raptor.ir.types.Slot`, since the ABI
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
        (:class:`lowering.box.Box`): ``base + flat`` is a thread and ``i``
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
                 strategy: str, **kw: object) -> str:
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
    #: The thread box the entry was emitted with (:class:`lowering.box.Box`), or ``None``.
    box: object = None


#: The macro that switches a host unit's interchanged loop on (the host wrapper defines it on by default).
INTERCHANGE_MACRO = "HAWK_HOST_INTERCHANGE"
#: The namespace of the collective calls of a team stage (its body is written by :mod:`aether.wings.base.team_emit`).
TEAM_CALL = "aether::linalg::team::"

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

    ``team`` (the threads per team of a team stage, :mod:`aether.wings.base.team_emit`) makes the lane the
    block on the device; the host entry is the plain lane loop, a team of one.

    ``box`` (:class:`lowering.box.Box`, from :func:`lowering.box.thread_box`)
    emits the plain entry with a thread box: each thread computes ``box.T``
    lanes (:func:`lowering.box.boxed_sinks`).

    A kernel with :class:`~raptor.ir.nodes.SampleSum` sinks runs the
    wrapper's ``sample_sum_frame`` instead of the index loop, and is refused
    under the segmented, active-set and automatic (``_persist``) entries."""
    if kind is None:
        raise RaptorError("the writer needs the kernel kind from its caller (hawk.ext.Kind); none was given")
    spec = spec.for_target(target)
    _wt.refuse_cells_off_device(name, mode, target.id, target)
    if walk.kinds.intersection(LSTSQ_OPS):     # the writer takes lowered sinks (lowering.lstsq.lowered), never lowers
        raise RaptorError(f"the writer was given an unlowered least-squares node ({', '.join(sorted(walk.kinds.intersection(LSTSQ_OPS)))}); "
                        "the caller lowers the kernel first (lowering.lstsq.lowered)")
    if mode.cells:
        target.wrapper.device_text.refuse_uncertified(sinks, walk, mode)
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
    # a contract="compensated" scatter that commits per lane
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
    tiled = (_interchange(wrapper, name, sinks, walk, kind, mode, spec, writer)
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
               scatter_or=_OR_CALL in body.text, **_device_calls(wrapper, body.text),
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
    """One entry per owned companion (:attr:`raptor.ir.Walk.companions`): the
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
            comp = binding_name(sink.role, sink.name + SAMPLE_SUM_COMP)
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
                 mode: ScalarMode, spec, writer):
    """The wrapper's interchanged fused-step loop for this TU (the host's
    optional ``interchange`` hook, :mod:`aether.wings.cpu.host`), or ``None``: a
    wrapper without the hook, or a kernel the hook declines, keeps its text
    unchanged."""
    hook = getattr(wrapper, "interchange", None)
    if hook is None:
        return None
    return hook(name, sinks, walk, kind, 4 if mode is FLOAT32 else 8, spec, writer=writer)


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


def _path_marker(wrapper: Wrapper, label: str) -> list:
    """The marker each side of an automatic kernel's branch opens with: two different side-effecting statements, so the compiler can't
    hoist the sides' common code above the branch and fold them into one. A device wrapper's only (``[]`` on the host)."""
    return [] if wrapper.device_text is None else [wrapper.device_text.PATH_MARKER.format(label)]


def _render_two_paths(name: str, sinks, walk: Walk, target, *, spec,
                      mode: ScalarMode, kind, exports: str,
                      active: ActiveSpec | None, one_step, team: int | None = None, rows: int = 1) -> Source:
    """The TU of an automatic kernel: :func:`render_source` with the entry's
    index loop and body emitted twice under one uniform branch on the
    ``fused_steps`` word — the single step's (``one_step``) when the word is
    ``1``, the fused loop's otherwise.

    On a wrapper with the ``persist`` hook (the device), the TU also carries
    the kernel's fast-path entries (its ``fast_entries`` hook)."""
    wrapper = _for_build(target.wrapper, spec)
    writer = _writer(target, mode)
    fused = render_body(sinks, walk, indent="        ", kind=kind, writer=writer, spec=spec)
    single = render_body(one_step.sinks, one_step.walk, indent="        ", kind=kind,
                         writer=writer, spec=spec)
    tiled = None if active is not None else _interchange(wrapper, name, sinks,
                                                         walk, kind, mode, spec, writer)
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
                **_device_calls(wrapper, fused.text + single.text)),
        _extra_prelude(extra_prelude, walk, tiled, fused.text + single.text),
        _exports(exports, tiled),
        "",
        *_two_path_entry(name, walk, wrapper, fused, single, one_step, active,
                         tiled, team, wrapper.device_text.staged(wrapper, rows, sinks, mode, spec) if rows > 1 else None),
        "",
        *([] if team is not None or getattr(wrapper, "fast_entries", None) is None     # the fast entries are a lane per thread: a warp per lane has none
          else wrapper.fast_entries(name, sinks, walk, kind, fused, single, one_step, active, spec, writer)),
    ]
    return Source(_filled(parts, mode),
                  single.text + "\n" + fused.text, name, target.id, mode.id)


def _two_path_entry(entry: str, walk: Walk, wrapper: Wrapper, fused, single,
                    one_step, active: ActiveSpec | None, tiled, team: int | None = None, staged: tuple | None = None) -> list:
    """The two-path entry ``entry`` of an automatic kernel: its signature,
    unpack and the branch on the ``fused_steps`` word between the single
    step's index loop (``single``) and the fused loop's (``fused``), each
    under ``active``'s prologue (``None``: the contiguous range)."""
    from lowering.cells import FUSED_STEPS_PLANE

    word = (f"{binding_name('lookup', FUSED_STEPS_PLANE)}"
            "[aether::SampleIndex::make(static_cast<std::size_t>(0))].eval()")
    one_open, one_close = staged or _frame(one_step.walk, wrapper, active, team)
    many_open, many_close = staged or _frame(walk, wrapper, active, team)
    return [
        wrapper.entry_signature(walk, entry),
        "{",
        wrapper.unpack(walk),
        f"    if ({word} == 1) {{",
        *_path_marker(wrapper, "one step"),
        one_open,
        single.text,
        one_close,
        "    } else {",
        *_path_marker(wrapper, "fused steps"),
        *_tiled_or(tiled, [many_open, fused.text, many_close]),
        "    }",
        "}",
    ]


def unpack_lines(walk: Walk, source_of) -> str:
    """The shared half of both wrappers' ``unpack``: one binding per
    ``arg_spec`` slot, in ``arg_spec`` order, each typed by ``storage`` and
    rebuilt from the by-value mirror ``source_of`` names on that target.

    The order is ``arg_spec``'s own, since it's also the entry's parameter
    order. A slot bound but never read still binds, and is voided so a real
    build does not warn about it."""
    lines = []
    finish = getattr(walk, "finish", None)
    finished_mask = finish
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
