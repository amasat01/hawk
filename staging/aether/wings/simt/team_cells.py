# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The banded (``cells``) forms of the ordered team stages: every partial sum is aether's 16-byte ``BandPartial`` folded with
``BandAccum``, in the same order as the float64 stages (so the bits do not depend on the run); a stored cell is encoded once, at the end.
:func:`aether.wings.base.team_emit.team_body` takes this module as its ``cells`` argument (``None``: the float stages)."""

from __future__ import annotations

from aether.wings.base.spelling import binding_name

from raptor.wings.base.team import GROUPS

_ACC = "aether::banded::BandAccum"
_PART = "aether::banded::BandPartial"
_BAND = "aether::banded::Band"

#: A fold-only loop over a run of banded rows is unrolled by the accumulator's renormalise period, so its countdown
#: gate is a compile-time constant. Never on a kernel's sample loop or a loop whose accumulator changes per iteration.
_BAND_UNROLL = "#pragma unroll aether::banded::BandAccum::period"

#: The constant 1 an ordered plain sum multiplies by.
ONE = f"Real({_BAND}(1.0f, 0.0f, 0.0f))"

#: The scatter walk's pieces: one slot added into its chunk's partial, the 64-bit shuffle of a cell value, a shared bin's add / declaration
#: / zero / combine into the chunk's partial.
SLOT_ADD = ("            { " + f"{_PART}& hawk_q = hawk_p[static_cast<std::size_t>(hawk_key) * hawk_chunks + static_cast<std::size_t>(hawk_lane)]; "
            f"{_ACC} hawk_a = {_ACC}::fromPartial(hawk_q); hawk_a.addTerm({_BAND}(hawk_v[hawk_r])); hawk_q = hawk_a.partial(); }}")
SHUFFLED_VALUE = ("                unsigned long long hawk_xb;",
                  "                __builtin_memcpy(&hawk_xb, &hawk_mv, sizeof hawk_xb);",
                  "                hawk_xb = __shfl_sync(0xffffffffu, hawk_xb, hawk_src);",
                  "                Real hawk_x;",
                  "                __builtin_memcpy(&hawk_x, &hawk_xb, sizeof hawk_x);")
BIN_ADD = f"                    hawk_bin[hawk_key].addTerm({_BAND}(hawk_x));"
BIN_DECL = f"__shared__ {_ACC} hawk_bin[1024];"
BIN_ZERO = f"{_ACC}{{}};"
BIN_COMBINE = ("        { " + f"{_PART}& hawk_q = hawk_p[hawk_j * hawk_chunks + static_cast<std::size_t>(hawk_lane)]; "
               f"{_ACC} hawk_a = {_ACC}::fromPartial(hawk_q); hawk_a.combine(hawk_bin[hawk_j]); hawk_q = hawk_a.partial(); }}")


def _binding(name: str) -> str:
    return binding_name("wide_out", name)


def _part_ptr(name: str, offset: str) -> str:
    """A ``BandPartial`` pointer to ``offset`` elements into the scratch plane ``name``."""
    return f"(reinterpret_cast<{_PART}*>({_binding(name)}.data()) + {offset})"


def _combine(dst: str, src: str) -> str:
    """``dst`` (a BandPartial) becomes the combined partial of ``dst`` and ``src``."""
    return (f"{{ {_ACC} hawk_a = {_ACC}::fromPartial({dst}); hawk_a.combine({_ACC}::fromPartial({src})); "
            f"{dst} = hawk_a.partial(); }}")


def block_gp(roles) -> list:
    """The last stage of a plain ordered sum: one block folds the group partials of key ``hawk_key``."""
    return [
        f"__shared__ {_PART} hawk_sh[{GROUPS}];",
        f"for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < {GROUPS}; "
        "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
        f"    hawk_sh[hawk_i] = {_part_ptr(roles['gp'], f'hawk_key * {GROUPS}')}[hawk_i];",
        "hawk_tm.sync();",
        f"for (std::size_t hawk_w = {GROUPS // 2}; hawk_w > 0; hawk_w >>= 1) {{",
        "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; "
        "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
        "        " + _combine("hawk_sh[hawk_i]", "hawk_sh[hawk_i + hawk_w]"),
        "    hawk_tm.sync();",
        "}",
        f"if (hawk_tm.rank == 0) {_binding(roles['out'])}.data()[hawk_key] = {_ACC}::fromPartial(hawk_sh[0]).toBandedReal();",
        "hawk_tm.sync();"]


def block_part(roles, band_lanes: int, band_virtual_lanes: int) -> list:
    """The first stage of a plain ordered sum: block ``hawk_lane`` folds its ``band_lanes`` accumulators over its run of rows."""
    band = (f"{_BAND}(hawk_u[hawk_r]) * {_BAND}(hawk_v[hawk_r])" if "v" in roles else f"{_BAND}(hawk_u[hawk_r])")
    return [
        f"__shared__ {_PART} hawk_sh[{band_lanes}];",
        f"for (std::size_t hawk_t = static_cast<std::size_t>(hawk_tm.rank); hawk_t < {band_lanes}; "
        "hawk_t += static_cast<std::size_t>(hawk_tm.size)) {",
        f"    {_ACC} hawk_s;",
        f"    {_BAND_UNROLL}",
        f"    for (std::size_t hawk_r = hawk_grp * {band_lanes} + hawk_t; hawk_r < hawk_rows; hawk_r += {band_virtual_lanes})",
        f"        hawk_s.addTerm({_BAND}({band}));",
        "    hawk_sh[hawk_t] = hawk_s.partial();",
        "}",
        "hawk_tm.sync();",
        f"for (std::size_t hawk_w = {band_lanes // 2}; hawk_w > 0; hawk_w >>= 1) {{",
        "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; "
        "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
        "        " + _combine("hawk_sh[hawk_i]", "hawk_sh[hawk_i + hawk_w]"),
        "    hawk_tm.sync();",
        "}",
        f"if (hawk_tm.rank == 0) {_part_ptr(roles['gp'], f'hawk_key * {GROUPS}')}[hawk_grp] = hawk_sh[0];",
        "hawk_tm.sync();"]


def fold(step) -> list:
    """The call of an ordered fold in cells."""
    return ["const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
            "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / static_cast<std::size_t>(count);",
            f"const Real* const hawk_v = {_binding(step.plane('v'))}.data();",
            f"const Real* const hawk_c = {_binding(step.plane('i'))}.data();",
            f"Real* const hawk_out = {_binding(step.plane('out'))}.data();",
            f"for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {step.n}; "
            "hawk_e += static_cast<std::size_t>(hawk_tm.size)) {",
            f"    {_ACC} hawk_s;",
            f"    {_BAND_UNROLL}",
            "    for (std::size_t hawk_r = 0; hawk_r < hawk_rows; ++hawk_r)",
            f"        hawk_s.addTerm({_BAND}(hawk_v[(hawk_key * hawk_rows + hawk_r) * {step.n} + hawk_e]));",
            f"    hawk_out[static_cast<std::size_t>(hawk_c[hawk_key * {step.n} + hawk_e])] = hawk_s.toBandedReal();",
            "}",
            "hawk_tm.sync();"]


def scatter_zero(step) -> tuple:
    """``(part declaration, zero)`` of the first scatter stage."""
    return (f"{_PART}* const hawk_p = {_part_ptr(step.plane('part'), 'static_cast<std::size_t>(hawk_lane) * hawk_chunks')};", f"{_PART}{{}}")


def scatter_part(step) -> str:
    """The chunk stage's ``hawk_p`` declaration."""
    return f"{_PART}* const hawk_p = {_part_ptr(step.plane('part'), '0')};"


def scatter_cells(step, chunks: list) -> list:
    """The last scatter stage: one thread adds the chunk partials of its cell in chunk order (``chunks``: the chunk-count lines)."""
    return [*chunks,
            f"const {_PART}* const hawk_p = {_part_ptr(step.plane('part'), 'static_cast<std::size_t>(hawk_lane) * hawk_chunks')};",
            f"Real* const hawk_out = {_binding(step.plane('out'))}.data();",
            "if (hawk_tm.rank == 0) {",
            f"    {_ACC} hawk_s;",
            f"    hawk_s.addTerm({_BAND}(hawk_out[hawk_lane]));",
            f"    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_s.combine({_ACC}::fromPartial(hawk_p[hawk_b]));",
            "    hawk_out[hawk_lane] = hawk_s.toBandedReal();",
            "}",
            "hawk_tm.sync();"]
