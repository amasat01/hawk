# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The body of a team stage: collective calls into ``aether::linalg::team``.

A team stage is launched with one team per lane (device: ``blockIdx.x`` is the lane and the
block is the team; host: the lane loop of the entry, a team of one). Its body is the calls of
:class:`hawk.ir.team.Step`, in order, each on the lane's slice of the scratch planes the
per-lane stages stored the operands in. Every call is collective: all the threads of the team
make it with the same arguments, which the entry guarantees by leaving a lane past ``count``
for the whole block at once.
"""

from __future__ import annotations

from .._lane_bind import SCATTER_CHUNKS
from ..ir.team import GROUPS, TeamSpec
from .render import Body

#: Virtual lanes one block of an ordered plain sum adds; ``GROUPS`` blocks per key make the 65536 of the sum.
FOLD_LANES = 1024

VIRTUAL_LANES = FOLD_LANES * GROUPS

#: The text every team call carries; the wrapper includes the team headers for a body holding it.
TEAM_CALL = "aether::linalg::team::"

_NS = TEAM_CALL
_STRUCT = {"lu": "TeamLu", "chol": "TeamCholesky", "chol_from": "TeamCholesky", "eigh": "TeamEigh",
           "qr": "TeamQr"}
_FACTORISE = {"lu": "lu", "chol": "cholesky", "chol_from": "choleskyFrom", "eigh": "eigh", "qr": "qr"}
_SOLVE = {"lu_solve": "luSolve", "lu_solve_t": "luSolveT", "cho_solve": "choSolve",
          "tri_lower": "triLower", "tri_upper_t": "triUpperT"}
_PLANE = "hawk_lane"                 # the lane the entry bound: the sample index of the lane


def _binding(name: str) -> str:
    from ..emit.spelling import binding_name

    return binding_name("wide_out", name)


def _pointer(name: str, role: str, per_lane: int) -> str:
    ctype = "std::int32_t" if role in ("info", "piv") else "Real"
    return (f"{ctype}* const hawk_{role} = {_binding(name)}.data() + "
            f"static_cast<std::size_t>({_PLANE}) * {per_lane};")


def _struct(step, n: int) -> str:
    """The factor's view over the step's planes, as one expression."""
    kind = step.op if step.op in _STRUCT else {"lu_solve": "lu", "lu_solve_t": "lu", "lu_inverse": "lu",
                                                 "lu_det": "lu", "cho_solve": "chol", "qr_solve": "qr", "qr_gram_solve": "qr",
                                                 "tri_lower": "chol", "tri_upper_t": "chol"}[step.op]
    if kind == "lu":
        fields = "hawk_a, hawk_piv, hawk_info"
    elif kind == "eigh":
        fields = "hawk_w, hawk_v, hawk_work, hawk_info"
    elif kind == "qr":
        fields = "hawk_a, hawk_tau, hawk_work, hawk_info"
    else:
        fields = "hawk_a, hawk_info"
    return f"{_NS}{_STRUCT[kind]}<Real, {n}>{{{fields}}}"


def _call(step) -> str:
    n, op = step.n, step.op
    if op == "qr":
        return f"{_NS}qr<{n}>(hawk_tm, hawk_f, hawk_a, {step.m});"
    if op == "qr_solve":
        return f"{_NS}qrSolve<{n}>(hawk_tm, hawk_f, {step.m}, hawk_r, hawk_x, {step.k});"
    if op == "qr_gram_solve":           # (J^T J)^-1 B in place from R; zeros when the factor failed
        return f"{_NS}qrGramSolve<{n}>(hawk_tm, hawk_f, hawk_x, hawk_x, {step.k});"
    if op in _FACTORISE:
        return f"{_NS}{_FACTORISE[op]}<{n}>(hawk_tm, hawk_f, hawk_a);"
    if op == "lu_inverse":
        return f"{_NS}luInverse<{n}>(hawk_tm, hawk_f, hawk_x);"
    if op == "lu_det":
        return f"if (hawk_tm.rank == 0) *hawk_x = {_NS}luDet<{n}>(hawk_f);"
    return f"{_NS}{_SOLVE[op]}<{n}>(hawk_tm, hawk_f, hawk_x, hawk_x, {step.k});"


def _block(step, per_lane: dict, active) -> list:
    """The call of a block sum: the team of key ``hawk_lane`` sums that key's rows (``nSamples /
    count`` of them, contiguous: the reduced axes are the trailing ones) into its block ``hawk_g``,
    then copies it to the output plane when the sum is stored (``(*payload, *kept)``: entry ``e``
    of key ``k`` at ``k + e * count``). A lane holds ``step.m`` rows of a ``transpose(M) @ M``
    spelling, all of them rows of its key's sum, in lane order (the call's ``rowsPer``). Under an index map the rows of a
    key are the live samples in its range: the map is ascending, so they are the positions found by
    two binary searches."""
    p, q, per = step.n, step.k, max(step.m, 1)
    first, rows, index = "hawk_key * hawk_rows", "hawk_rows", ""
    lines = ["const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
             "const std::size_t hawk_keys = static_cast<std::size_t>(count);",
             "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / hawk_keys;"]
    if active is not None:
        from .wrapper import active_reads

        live, _ = active_reads(active, "0")
        bits = active.map_bits
        lines += [f"const std::int{bits}_t* const hawk_map = reinterpret_cast<const std::int{bits}_t*>("
                  f"{active.map_binding}.data());",
                  f"const std::size_t hawk_live = static_cast<std::size_t>({live});",
                  "std::size_t hawk_edge[2];",
                  "for (std::size_t hawk_s = 0; hawk_s < 2; ++hawk_s) {",
                  "    std::size_t hawk_lo = 0, hawk_hi = hawk_live;",
                  "    while (hawk_lo < hawk_hi) {",
                  "        const std::size_t hawk_mid = (hawk_lo + hawk_hi) / 2;",
                  "        if (static_cast<std::size_t>(hawk_map[hawk_mid]) < (hawk_key + hawk_s) * hawk_rows)"
                  " hawk_lo = hawk_mid + 1;",
                  "        else hawk_hi = hawk_mid;",
                  "    }",
                  "    hawk_edge[hawk_s] = hawk_lo;",
                  "}"]
        first, rows, index = "hawk_edge[0]", "hawk_edge[1] - hawk_edge[0]", ", hawk_map"
    roles = dict(step.planes)
    lines.append(f"const Real* const hawk_u = {_binding(step.plane('u'))}.data();")
    if "v" in roles:
        lines.append(f"const Real* const hawk_v = {_binding(step.plane('v'))}.data();")
        stride_v = str(q)
    else:                       # an ordered plain sum: v is the constant 1
        lines += ["const Real hawk_one = Real(1);", "const Real* const hawk_v = &hawk_one;"]
        stride_v = "0"
    if "gp" in roles and active is None:
        gp = _binding(roles["gp"])
        return lines + [
            "#if defined(__CUDA_ARCH__)",
            f"__shared__ Real hawk_sh[{GROUPS}];",
            "#else",
            f"Real hawk_sh[{GROUPS}];",
            "#endif",
            f"for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < {GROUPS}; "
            "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
            f"    hawk_sh[hawk_i] = {gp}.data()[hawk_key * {GROUPS} + hawk_i];",
            "hawk_tm.sync();",
            f"for (std::size_t hawk_w = {GROUPS // 2}; hawk_w > 0; hawk_w >>= 1) {{",
            "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; "
            "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
            "        hawk_sh[hawk_i] += hawk_sh[hawk_i + hawk_w];",
            "    hawk_tm.sync();",
            "}",
            f"if (hawk_tm.rank == 0) {_binding(roles['out'])}.data()[hawk_key] = hawk_sh[0];",
            "hawk_tm.sync();"]
    if "g" not in roles:        # stored only: the sum lands in the payload-first output plane, key fastest
        lines += [f"Real* const hawk_out = {_binding(roles['out'])}.data();",
                  f"{_NS}blockSum<{p}, {q}>(hawk_tm, hawk_out + hawk_key, hawk_u, hawk_v, {first}, {rows}, {p}, {stride_v}{index or ', nullptr'}, "
                  f"{q} * hawk_keys, hawk_keys, {per});"]
        return lines
    lines += [_pointer(step.plane("g"), "g", per_lane[step.plane("g")]),
              f"{_NS}blockSum<{p}, {q}>(hawk_tm, hawk_g, hawk_u, hawk_v, {first}, {rows}, {p}, {stride_v}{index or ', nullptr'}, 0, 0, {per});"]
    if "out" in roles:
        lines += [f"Real* const hawk_out = {_binding(step.plane('out'))}.data();",
                  f"for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {p * q}; "
                  "hawk_e += static_cast<std::size_t>(hawk_tm.size))",
                  "    hawk_out[hawk_key + hawk_e * hawk_keys] = hawk_g[hawk_e];"]
    return lines


def _block_part(step, active) -> list:
    """The first stage of a plain ordered sum: block ``hawk_lane`` is group ``hawk_lane % GROUPS`` of key ``hawk_lane / GROUPS``;
    virtual lane ``l`` of the 65536 adds rows ``l, l + 65536, ...`` of the key in order, each group's 1024 lanes are folded by the
    halving tree and the group's value goes to its cell of the partials plane. Constants and the row count only."""
    if active is not None:
        return []                       # an index-set build keeps the general block sum, in the last stage
    roles = dict(step.planes)
    lines = [f"const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane) / {GROUPS};",
             f"const std::size_t hawk_grp = static_cast<std::size_t>(hawk_lane) % {GROUPS};",
             f"const std::size_t hawk_keys = static_cast<std::size_t>(count) / {GROUPS};",
             "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / hawk_keys;",
             f"const Real* const hawk_u = {_binding(roles['u'])}.data() + hawk_key * hawk_rows;"]
    if "v" in roles:
        lines.append(f"const Real* const hawk_v = {_binding(roles['v'])}.data() + hawk_key * hawk_rows;")
        term = "hawk_u[hawk_r] * hawk_v[hawk_r]"
    else:
        term = "hawk_u[hawk_r]"
    return lines + [
        "#if defined(__CUDA_ARCH__)",
        f"__shared__ Real hawk_sh[{FOLD_LANES}];",
        "#else",
        f"Real hawk_sh[{FOLD_LANES}];",
        "#endif",
        f"for (std::size_t hawk_t = static_cast<std::size_t>(hawk_tm.rank); hawk_t < {FOLD_LANES}; "
        "hawk_t += static_cast<std::size_t>(hawk_tm.size)) {",
        "    Real hawk_s = Real(0);",
        f"    for (std::size_t hawk_r = hawk_grp * {FOLD_LANES} + hawk_t; hawk_r < hawk_rows; hawk_r += {VIRTUAL_LANES})",
        f"        hawk_s += {term};",
        "    hawk_sh[hawk_t] = hawk_s;",
        "}",
        "hawk_tm.sync();",
        f"for (std::size_t hawk_w = {FOLD_LANES // 2}; hawk_w > 0; hawk_w >>= 1) {{",
        "    for (std::size_t hawk_i = static_cast<std::size_t>(hawk_tm.rank); hawk_i < hawk_w; "
        "hawk_i += static_cast<std::size_t>(hawk_tm.size))",
        "        hawk_sh[hawk_i] += hawk_sh[hawk_i + hawk_w];",
        "    hawk_tm.sync();",
        "}",
        f"if (hawk_tm.rank == 0) {_binding(roles['gp'])}.data()[hawk_key * {GROUPS} + hawk_grp] = hawk_sh[0];",
        "hawk_tm.sync();"]


def _fold(step) -> list:
    """The call of an ordered fold: the team of key ``hawk_lane`` adds the key's ``nSamples / count``
    contiguous lanes' contributions to each of the ``step.n`` slots in lane order, into the cell the
    slot names."""
    return ["const std::size_t hawk_key = static_cast<std::size_t>(hawk_lane);",
            "const std::size_t hawk_rows = static_cast<std::size_t>(nSamples) / static_cast<std::size_t>(count);",
            f"const Real* const hawk_v = {_binding(step.plane('v'))}.data();",
            f"const Real* const hawk_c = {_binding(step.plane('i'))}.data();",
            f"Real* const hawk_out = {_binding(step.plane('out'))}.data();",
            f"for (std::size_t hawk_e = static_cast<std::size_t>(hawk_tm.rank); hawk_e < {step.n}; "
            "hawk_e += static_cast<std::size_t>(hawk_tm.size)) {",
            "    Real hawk_s = Real(0);",
            "    for (std::size_t hawk_r = 0; hawk_r < hawk_rows; ++hawk_r)",
            f"        hawk_s += hawk_v[(hawk_key * hawk_rows + hawk_r) * {step.n} + hawk_e];",
            f"    hawk_out[static_cast<std::size_t>(hawk_c[hawk_key * {step.n} + hawk_e])] = hawk_s;",
            "}",
            "hawk_tm.sync();"]


def _chunk_count() -> list:
    """The chunks of an ordered scatter, ``min(256, max(1, nSamples / cells))``: ``count`` is the cells."""
    return ["const std::size_t hawk_cells = static_cast<std::size_t>(count);",
            "const std::size_t hawk_fit = static_cast<std::size_t>(nSamples) / hawk_cells;",
            f"const std::size_t hawk_chunks = hawk_fit < 1 ? 1 : (hawk_fit > {SCATTER_CHUNKS} ? {SCATTER_CHUNKS} : hawk_fit);"]


def _scatter_zero(step) -> list:
    """An ordered scatter's first stage: block ``hawk_lane`` is cell ``t`` of the target (of ``count``
    cells); it zeroes the cell's partial sums, so a launch that stopped early leaves nothing behind.
    Block 0 also writes the cell count (the status plane is zero at the start of a run)."""
    return [*_chunk_count(),
            f"Real* const hawk_p = {_binding(step.plane('part'))}.data() + static_cast<std::size_t>(hawk_lane) * hawk_chunks;",
            f"std::int32_t* const hawk_m = const_cast<std::int32_t*>({_binding(step.plane('status'))}.data());",
            "if (hawk_tm.rank == 0) {",
            "    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_p[hawk_b] = Real(0);",
            "    if (hawk_lane == 0) hawk_m[0] = static_cast<std::int32_t>(hawk_cells);",
            "}",
            "hawk_tm.sync();"]


def _scatter_chunks(step) -> list:
    """An ordered scatter's middle stage: block ``hawk_lane`` is lane chunk ``b`` (of ``count`` chunks
    over ``nSamples`` lanes, ``step.n`` log slots each); one thread adds the chunk's slots in order into its
    own partial sum of each key, cell ``key * count + b`` of the parts plane. The key -1 adds nothing; a key
    outside the cells adds nothing either and sets the chunk's flag. Each slot read is put back to -1."""
    return ["const std::size_t hawk_chunks = static_cast<std::size_t>(count);",
            "const std::size_t hawk_span = (static_cast<std::size_t>(nSamples) + hawk_chunks - 1) / hawk_chunks;",
            "const std::size_t hawk_lo = static_cast<std::size_t>(hawk_lane) * hawk_span;",
            "const std::size_t hawk_hi = hawk_lo + hawk_span < static_cast<std::size_t>(nSamples) ? "
            "hawk_lo + hawk_span : static_cast<std::size_t>(nSamples);",
            f"std::int32_t* const hawk_k = const_cast<std::int32_t*>({_binding(step.plane('k'))}.data());",
            f"const Real* const hawk_v = {_binding(step.plane('v'))}.data();",
            f"Real* const hawk_p = {_binding(step.plane('part'))}.data();",
            f"std::int32_t* const hawk_m = const_cast<std::int32_t*>({_binding(step.plane('status'))}.data());",
            "const std::int64_t hawk_cells = hawk_m[0];",
            *_scatter_walk(step)]


def _scatter_walk(step) -> list:
    """The walk of the chunk's slots (see :func:`_scatter_chunks`). On the device (up to 1024 cells) a warp takes 32
    rows at a time, each lane owns the cells of its residue ``key % 32`` in a shared bin and adds that cell's rows in
    row order (a per-lane bitmask of the batch), so every cell's sum has the order of the serial walk: the same bits.
    The host and a larger target walk the slots serially."""
    n = step.n
    plain = [
        "if (hawk_tm.rank == 0)",
        f"    for (std::size_t hawk_r = hawk_lo * {n}; hawk_r < hawk_hi * {n}; ++hawk_r) {{",
        "        const std::int32_t hawk_key = hawk_k[hawk_r];",
        "        if (hawk_key == -1) continue;",
        "        if (hawk_key >= 0 && hawk_key < hawk_cells)",
        "            hawk_p[static_cast<std::size_t>(hawk_key) * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_v[hawk_r];",
        "        else",
        "            hawk_m[1 + hawk_lane] = 1;",
        "        hawk_k[hawk_r] = -1;",
        "    }",
        "hawk_tm.sync();"]
    walk = [
        "    if (hawk_tm.rank < 32) {",
        "        const unsigned hawk_ln = static_cast<unsigned>(hawk_tm.rank);",
        f"        const std::size_t hawk_end = hawk_hi * {n};",
        f"        for (std::size_t hawk_b = hawk_lo * {n}; hawk_b < hawk_end; hawk_b += 32) {{",
        "            const std::size_t hawk_r = hawk_b + hawk_ln;",
        "            const bool hawk_in = hawk_r < hawk_end;",
        "            const std::int32_t hawk_mk = hawk_in ? hawk_k[hawk_r] : -1;",
        "            const Real hawk_mv = hawk_in ? hawk_v[hawk_r] : Real(0);",
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
        "                const Real hawk_x = __shfl_sync(0xffffffffu, hawk_mv, hawk_src);",
        "                const std::int32_t hawk_key = __shfl_sync(0xffffffffu, hawk_mk, hawk_src);",
        "                if (hawk_mine != 0u) {",
        "                    hawk_bin[hawk_key] += hawk_x;",
        "                    hawk_mine &= hawk_mine - 1u;",
        "                }",
        "            }",
        "            if (hawk_in && hawk_mk != -1) hawk_k[hawk_r] = -1;",
        "        }",
        "    }"]
    return [
        "#if defined(__CUDA_ARCH__)",
        "__shared__ Real hawk_bin[1024];",
        "if (hawk_cells <= 1024) {",
        "    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); "
        "hawk_j += static_cast<std::size_t>(hawk_tm.size)) hawk_bin[hawk_j] = Real(0);",
        "    hawk_tm.sync();",
        *walk,
        "    hawk_tm.sync();",
        "    for (std::size_t hawk_j = static_cast<std::size_t>(hawk_tm.rank); hawk_j < static_cast<std::size_t>(hawk_cells); "
        "hawk_j += static_cast<std::size_t>(hawk_tm.size))",
        "        hawk_p[hawk_j * hawk_chunks + static_cast<std::size_t>(hawk_lane)] += hawk_bin[hawk_j];",
        "} else {",
        *plain,
        "}",
        "#else",
        *plain,
        "#endif"]


def _scatter_cells(step) -> list:
    """An ordered scatter's last stage: block ``hawk_lane`` is cell ``t`` of the target (of ``count``
    cells); one thread adds its partial sums in chunk order (the same number the middle stage cut the
    lanes into) and the total to the cell."""
    return [*_chunk_count(),
            f"const Real* const hawk_p = {_binding(step.plane('part'))}.data() + static_cast<std::size_t>(hawk_lane) * hawk_chunks;",
            f"Real* const hawk_out = {_binding(step.plane('out'))}.data();",
            "if (hawk_tm.rank == 0) {",
            "    Real hawk_s = Real(0);",
            "    for (std::size_t hawk_b = 0; hawk_b < hawk_chunks; ++hawk_b) hawk_s += hawk_p[hawk_b];",
            "    hawk_out[hawk_lane] += hawk_s;",
            "}",
            "hawk_tm.sync();"]


_SCATTER = {"scatter_zero": _scatter_zero, "scatter_chunks": _scatter_chunks, "scatter_cells": _scatter_cells}


def team_body(spec: TeamSpec, active=None) -> Body:
    """The body of the team stage ``spec``: its steps, each in a scope of its own. ``active``
    (:class:`~hawk.emit.wrapper.ActiveSpec`) is the index map of an index-set build, which a block
    sum walks."""
    per_lane = {p.name: p.per_lane for p in spec.planes}
    lines = [f"        const {_NS}Team hawk_tm = {_NS}Team::here();",
             "        const long long hawk_lane = hawk_i;"]
    for step in spec.steps:
        lines.append(f"        {{  // {step.op}, n = {step.n}")
        if step.op == "ordered_fold":
            lines += ["            " + line for line in _fold(step)]
            lines.append("        }")
            continue
        if step.op in _SCATTER:
            lines += ["            " + line for line in _SCATTER[step.op](step)]
            lines.append("        }")
            continue
        if step.op == "block_part":
            lines += ["            " + line for line in _block_part(step, active)]
            lines.append("        }")
            continue
        if step.op == "block_sum":
            lines += ["            " + line for line in _block(step, per_lane, active)]
            lines.append("        }")
            continue
        lines += ["            " + _pointer(name, role, per_lane[name]) for role, name in step.planes]
        if step.op == "eigh_warp":      # one warp serves the lane; the rest of the team (if any) waits at the barrier
            lines += ["#if defined(__CUDA_ARCH__)",
                      "            __shared__ aether::linalg::warp::EighValuesScratch<Real> hawk_scratch;",
                      "            if (hawk_tm.rank < 32) {",
                      "                const auto hawk_r = aether::linalg::warp::eighValues<32>(hawk_scratch, static_cast<const Real*>(hawk_a));",
                      "                hawk_w[hawk_tm.rank] = hawk_r.w;",
                      "                if (hawk_tm.rank == 0) *hawk_info = hawk_r.info;",
                      "            }",
                      "#else       // the host's team of one: the one-thread form, same bits",
                      "            aether::linalg::EighValues<Real, 32> hawk_r;",
                      "            aether::linalg::eighValues<32>(hawk_r, static_cast<const Real*>(hawk_a));",
                      "            for (int hawk_k = 0; hawk_k < 32; ++hawk_k) hawk_w[hawk_k] = hawk_r.w[hawk_k];",
                      "            *hawk_info = hawk_r.info;",
                      "#endif",
                      "            hawk_tm.sync();",
                      "        }"]
            continue
        lines.append(f"            const auto hawk_f = {_struct(step, step.n)};")
        lines.append(f"            {_call(step)}")
        lines.append("        }")
    return Body("\n".join(lines))
