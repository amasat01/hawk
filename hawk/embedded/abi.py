# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target's ABI: the ``<name>.h`` header and the
``<name>.c`` translation unit around a rendered body.

The header carries the traceability, budget and interface-contract blocks,
the ``HAWK_EMBEDDED_TYPES`` width guard (``hawk_real_t``, ``hawk_int_t``,
``hawk_status_t``), the args struct (one typed field per slot in
``arg_spec`` order, an ``int64_t <field>_len`` after every runtime-length
plane) and the status-returning prototype. The TU includes the header,
``<math.h>`` and ``<stdbool.h>``, checks the two type widths under a C11
guard, inlines the aether core closure and then hawk's helpers in dependency
order, and ends in the entry: it validates ``args``, every pointer and the
partition range (``HAWK_E_NULL`` / ``HAWK_E_RANGE``, one ``return``), then a
static run function binds every slot it reads to a ``const`` local and runs
the sample loop over ``[base, base + count)`` with the SoA pitch
``n_samples``.

One function per block; :func:`header_text` and :func:`tu_text` join them.
The ``nsamples`` slot is not a field: the entry's own ``n_samples`` argument
is the sample count, and the body reads it under that name.
"""

from __future__ import annotations

import re

from aether_dsc import c_core

from ..ir.nodes import FINISHED_PLANE
from .helpers import helper
from .spelling import WIDTH

_REAL = ("f64", "f32")
#: Element dtype -> its C type in a plane or a by-value field.
_ELEM = {"f64": "hawk_real_t", "f32": "hawk_real_t", "i32": "hawk_int_t",
         "i64": "hawk_int_t", "int32": "int32_t", "uint32": "uint32_t",
         "bool": "uint8_t"}
#: The width word in the header guard per scalar mode.
_BITS = {"float64": 64, "float32": 32}
_C_REAL = {"float64": "double", "float32": "float"}


def _guard_macro(name: str) -> str:
    return "HAWK_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_H"


# --------------------------------------------------------------------------
# the args struct


def _shape_word(shape) -> str:
    if not shape:
        return "Scalar"
    if len(shape) == 1:
        return f"Vector[{shape[0]}]"
    return f"Matrix[{shape[0]}, {shape[1]}]"


def mix_planes(walk) -> dict[str, int]:
    """``{plane name: K}`` for every ``auto`` dispatch group's branch-mix
    plane (:func:`hawk.ir.segment.mix_name`): K ``uint32`` counters."""
    from ..ir.segment import mix_name

    return {mix_name(info.name): int(info.K) for info in walk.dispatches
            if info.policy == "auto"}


def size_mix_rows(rows: list, walk) -> list:
    """Give each branch-mix plane's interface row its true shape ``[K]``
    (verify's rows describe every reserved counter as one cell)."""
    mixes = mix_planes(walk)
    for row in rows:
        if row["role"] == "lookup" and row["name"] in mixes:
            row["shape"] = [mixes[row["name"]]]
    return rows


def _layout(row: dict, mixes: dict) -> str:
    if row["layout"] == "value":
        return "by value"
    if row["layout"] == "partials":
        then = "zeroes it, then reduces" if row["init_by_caller"] else "reduces"
        return f"one partial per sample, the caller {then}"
    if row["role"] == "terminated":
        return "mask, one byte per sample"
    if row["role"] == "lookup" and row["name"] in mixes:
        return f"{mixes[row['name']]} cells, the caller zeroes them"
    if row["role"] == "lookup" and row["name"] == FINISHED_PLANE:
        return "counter, the caller zeroes it"
    if row["runtime_length"]:
        return f"length in {row['field']}_len"
    if row["role"] == "lookup":
        return "one cell, read once"
    return "SoA, pitch n_samples"


def args_struct(rows: list, walk, mode, name: str = "kernel") -> tuple[str, list[dict]]:
    """The args struct's text and its fields in ``arg_spec`` order: a by-value
    field per ``uniform``, a pointer per plane (``const`` when the interface
    direction is ``in``), an ``int64_t <field>_len`` after each runtime-length
    plane. ``rows`` are the report's interface rows (the verify pass's, with
    each branch-mix plane already sized by :func:`size_mix_rows`): the ONE
    interface source. Each field is ``{"field", "ctype", "kind", "row"}`` with
    ``kind`` one of ``value``, ``pointer``, ``length``."""
    mixes = mix_planes(walk)
    fields: list[dict] = []
    for row in rows:
        role, dtype = row["role"], row["dtype"]
        if role == "nsamples":
            continue
        if role == "uniform":
            # a Param is Scalar, Int32 or UInt32; an integer one travels as
            # the by-value 64-bit Int (hawk/trace/decl.py)
            elem = "hawk_real_t" if dtype in _REAL else "hawk_int_t"
            fields.append({"field": row["field"], "ctype": elem, "kind": "value",
                           "row": row})
            continue
        elem = "uint8_t" if role == "terminated" else _ELEM[dtype]
        const = "const " if row["direction"] == "in" else ""
        fields.append({"field": row["field"], "ctype": f"{const}{elem}*",
                       "kind": "pointer", "row": row})
        if row["runtime_length"]:
            fields.append({"field": f"{row['field']}_len", "ctype": "int64_t",
                           "kind": "length", "row": row})
    type_w = max((len(f["ctype"]) for f in fields), default=0) + 1
    name_w = max((len(f["field"]) for f in fields), default=0) + 3
    role_w = max((len(f["row"]["role"]) for f in fields), default=0)
    dir_w = max((len(f["row"]["direction"]) for f in fields), default=0) + 1
    shape_w = max((len(_shape_word(f["row"]["shape"])) for f in fields), default=0)
    lines = ["typedef struct {"]
    for f in fields:
        row = f["row"]
        if f["kind"] == "length":
            note = f"length of {row['field']} (>= 1)"
        else:
            note = (f"{row['role']:<{role_w}} {row['direction']:<{dir_w}} "
                    f"{_shape_word(row['shape']):<{shape_w}} {_layout(row, mixes)}")
        decl = f"{f['ctype']:<{type_w}}{f['field'] + ';':<{name_w}}"
        lines.append(f"    {decl}/* {note} */")
    lines.append(f"}} {name}_args_t;")
    return "\n".join(lines) + "\n", fields


# --------------------------------------------------------------------------
# the header


def _prototype(name: str) -> str:
    return (f"hawk_status_t {name}_c(const {name}_args_t* args, int64_t base, "
            "int64_t count, int64_t n_samples)")


def _traceability(name: str, mode, report: dict, profile) -> list[str]:
    from .. import __version__

    def held(key: str) -> str:
        value = report.get(key)
        return "unavailable" if value is None else str(value)

    return [
        " * TRACEABILITY:",
        f" *   kernel         {name}",
        f" *   hawk version   {report.get('hawk_version', __version__)}",
        f" *   walk digest    {held('walk_digest')}",
        f" *   kernel sha256  {held('source_digest')}",
        f" *   scalar mode    {mode.id}",
        " *   target         c, MISRA C:2012+AMD, C99 dialect, serial",
        f" *   profile        {profile!r}",
        f" *   aether core    {report.get('core_version', c_core.CORE_VERSION)}",
    ]


def _budgets(report: dict) -> list[str]:
    budgets = report["budgets"]
    ops = ", ".join(f"{k} {v:g}" for k, v in budgets["ops"].items())
    static = budgets["static_bytes"]

    def held(value) -> str:
        return "unavailable" if value is None else str(value)

    libm = ", ".join(report.get("libm", ())) or "none"
    return [
        " * BUDGETS (per sample, worst case):",
        f" *   loop trips     {budgets['max_trips']}",
        f" *   ops            {ops}",
        f" *   fp64 ops       {budgets['fp64_ops']:g}",
        f" *   static bytes   tapes {static['tapes']}, carries {static['carries']}, "
        f"temporaries {held(static['temporaries'])}, total {held(static['total'])}",
        f" *   libm           {libm}",
    ]


#: The element width a plane's caller allocates, by IR dtype (an index
#: ``i32`` is the 64-bit ``hawk_int_t``; a ``terminated`` mask is one byte).
_C_DTYPE = {"i32": "i64", "i64": "i64", "int32": "i32", "uint32": "u32", "bool": "u8"}


def _c_dtype(row: dict, mode) -> str:
    if row["role"] == "terminated":
        return "u8"
    if row["dtype"] in _REAL:
        return WIDTH[mode.id]
    if row["role"] == "uniform":
        return "i64"        # an integer Param is the by-value hawk_int_t
    return _C_DTYPE[row["dtype"]]


_CONTRACT_PREAMBLE = (
    " * INTERFACE CONTRACT: planes are SoA; rank-1 extent N stores component c "
    "of sample i at",
    " *   base[(c * n_samples) + i]; rank-2 (R,C) at base[(((r*C)+c) * n_samples) "
    "+ i]; rank-0 at",
    " *   base[i]. The pitch is ALWAYS n_samples, also for a partition "
    "[base, base+count).",
    " *   A runtime-length plane carries its length in the int64_t field after "
    "its pointer.",
)


#: Writable roles a mask guard leaves untouched at a masked sample (the
#: partials of an ``accum_out`` have their own line).
_OWN_COLUMN = ("mutable", "wide_out")


def _contract(name: str, report: dict, mode) -> list[str]:
    cols = [(r["field"], r["role"], r["direction"], _c_dtype(r, mode),
             str(tuple(r["shape"])),
             ("yes (by value)" if r["layout"] == "value"
              else "yes" if r["init_by_caller"] else "no"))
            for r in report["interface"] if r["role"] != "nsamples"]
    head = ("field", "role", "dir", "dtype", "shape", "init-by-caller")
    pads = (2, 2, 2, 1, 2)
    widths = [max(len(c[k]) for c in [head, *cols]) + pads[k] for k in range(5)]

    def line(c) -> str:
        return " *   " + "".join(f"{c[k]:<{widths[k]}}" for k in range(5)) + c[5]

    lengths = any(r["runtime_length"] for r in report["interface"])
    out = [
        *_CONTRACT_PREAMBLE,
        line(head),
        *(line(c) for c in cols),
    ]
    for r in report["interface"]:
        # the row's structural ``guard``: the bound masks that skip a sample
        guard = f"(guard: {', '.join(r['guard'])})"
        if r["layout"] == "partials":
            # under a guard, masked samples keep theirs: the caller zeroes first
            tail = (f"masked samples keep theirs {guard}: the caller zeroes it, "
                    "then reduces" if r["guard"] else "the caller reduces it")
            out.append(f" *   {r['field']} holds ONE PARTIAL PER SAMPLE (length "
                       f"n_samples); {tail}.")
        elif r["role"] in _OWN_COLUMN and r["guard"]:
            out.append(f" *   {r['field']}: masked samples keep the prior value "
                       f"{guard}; the caller initialises it.")
    out += [
        f" *   {name}_c() validates args and every pointer, 0 <= base <= n_samples and",
        " *   0 <= count <= n_samples - base"
        + (", every runtime length >= 1," if lengths else ",")
        + " returning HAWK_E_NULL / HAWK_E_RANGE untouched otherwise.",
        " *   Re-entrant, allocation-free. */",
    ]
    return out


def _types_guard(name: str, mode) -> list[str]:
    bits = _BITS[mode.id]
    real = _C_REAL[mode.id]
    return [
        "#ifndef HAWK_EMBEDDED_TYPES",
        f"#define HAWK_EMBEDDED_TYPES {bits}",
        f"typedef {real:<7} hawk_real_t;",
        "typedef int64_t hawk_int_t;",
        "typedef enum { HAWK_OK = 0, HAWK_E_NULL = 1, HAWK_E_RANGE = 2 } "
        "hawk_status_t;",
        f"#elif HAWK_EMBEDDED_TYPES != {bits}",
        f'#error "{name}.h: {mode.id} unit; another hawk header in this build defined '
        'hawk_real_t at a different width"',
        "#endif",
    ]


def header_text(name: str, walk, mode, report: dict, profile) -> str:
    """The ``<name>.h`` text (the spec's ``drag.h`` shape)."""
    from .. import __version__

    struct, _fields = args_struct(report["interface"], walk, mode, name)
    guard = _guard_macro(name)
    includes = ["#include <stdint.h>", "#include <stddef.h>"]
    lines = [
        f"/* {name}.h -- generated by hawk {__version__}; do not edit.",
        *_traceability(name, mode, report, profile),
        *_budgets(report),
        *_contract(name, report, mode),
        f"#ifndef {guard}",
        f"#define {guard}",
        *includes,
        *_types_guard(name, mode),
    ]
    return "\n".join(lines) + "\n" + struct + _prototype(name) + ";\n#endif\n"


# --------------------------------------------------------------------------
# the translation unit: closures


def core_closure(writer) -> tuple[str, list[str]]:
    """The aether core functions the body and the helpers call, with every
    struct typedef a declared local needs: the dependency-ordered text
    (:func:`aether_dsc.c_core.emit`) and the sorted function names."""
    names = set(writer.core) | set(writer.structs)
    functions = set(writer.core)
    for name in _helper_order(writer):
        for dep in _helper_fn(writer, name).deps:
            if dep.startswith("aether_c_"):
                names.add(dep)
                if not dep.endswith("_t"):
                    functions.add(dep)
    text = c_core.emit(sorted(names)) if names else ""
    return text, sorted(functions)


def _helper_fn(writer, name: str):
    base, shape = writer.helper_requests[name]
    return helper(base, mode=writer.mode, shape=shape)


#: Helper families in emission order: clamps, integers, reals, aggregates.
_FAMILY = {"clamp": 0, "i64": 1, "i32": 1, "u32": 1, "spacing": 2, "sign": 2}
_SHAPED_ORDER = ("load", "store", "splat", "make", "select", "at", "at_raw",
                 "set", "set_raw")


def _helper_order(writer) -> list[str]:
    """Every hawk helper the body references plus their ``hawk_*`` deps,
    each once, deps before users, in a fixed family order."""
    for name in list(writer.helper_requests):
        for dep in _helper_fn(writer, name).deps:
            if dep.startswith("hawk_") and dep not in writer.helper_requests:
                writer.helper_requests[dep] = (dep.removeprefix("hawk_"), None)

    def key(name: str):
        base, shape = writer.helper_requests[name]
        if shape is None:
            return (_FAMILY.get(base.split("_", 1)[0], 2), (), 0, name)
        return (3, (len(shape), shape), _SHAPED_ORDER.index(base), name)

    return sorted(writer.helper_requests, key=key)


def helper_closure(writer) -> tuple[str, list[str], list[str]]:
    """The hawk helpers' text in dependency order, their names, and the libm
    names they call."""
    order = _helper_order(writer)
    fns = [_helper_fn(writer, n) for n in order]
    libm = sorted({m for fn in fns for m in fn.libm})
    return "\n".join(fn.text for fn in fns), order, libm


# --------------------------------------------------------------------------
# the translation unit: the entry


def _includes(name: str) -> list[str]:
    return [f'#include "{name}.h"', "#include <math.h>", "#include <stdbool.h>"]


def _size_asserts(name: str, mode) -> list[str]:
    real = 8 if mode.id == "float64" else 4
    return [
        "#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)",
        f'_Static_assert(sizeof(hawk_real_t) == {real}U, "{name}.c: hawk_real_t is not '
        f'the {mode.id} width");',
        f'_Static_assert(sizeof(hawk_int_t) == 8U, "{name}.c: hawk_int_t is not '
        '64-bit");',
        "#endif",
    ]


def _validation(name: str, fields: list[dict]) -> list[str]:
    """The entry: ``args``, every pointer, every runtime length and the
    partition range checked before anything is read; one ``return``."""
    pointers = [f"(args->{f['field']} == NULL)" for f in fields
                if f["kind"] == "pointer"]
    ranges = ["(base < INT64_C(0))", "(base > n_samples)", "(count < INT64_C(0))",
              "(count > (n_samples - base))"]
    ranges += [f"(args->{f['field']} < INT64_C(1))" for f in fields
               if f["kind"] == "length"]
    lines = [f"{_prototype(name)}", "{", "    hawk_status_t status;",
             "    if (args == NULL) {", "        status = HAWK_E_NULL;"]
    if pointers:
        lines += [f"    }} else if ({_disjunction(pointers, 15)}) {{",
                  "        status = HAWK_E_NULL;"]
    lines += [f"    }} else if ({_disjunction(ranges, 15)}) {{",
              "        status = HAWK_E_RANGE;",
              "    } else {",
              f"        {name}_run(args, base, count, n_samples);",
              "        status = HAWK_OK;",
              "    }",
              "    return status;",
              "}"]
    return lines


def _disjunction(terms: list[str], indent: int) -> str:
    """``a || b || ...``, one term per line past the first two."""
    if len(terms) <= 2:
        return " || ".join(terms)
    pad = "\n" + " " * indent
    return (" ||" + pad).join(terms)


def _uses(body: str, ident: str) -> bool:
    return re.search(rf"\b{re.escape(ident)}\b", body) is not None


def _bindings(fields: list[dict], body: str, walk) -> list[str]:
    """Every slot the body reads, bound once to a ``const`` local (a pointer
    is ``T* const``); a parameter cell loaded once; the finish mask's
    writable twin."""
    from ..emit.spelling import finish_binding, param_cell_binding

    lines = []
    for f in fields:
        field, ct = f["field"], f["ctype"]
        row = f["row"]
        if f["kind"] == "pointer":
            if _uses(body, field):
                # the finish epilogue writes the mask through its twin only,
                # so the reading binding points to const (cppcheck
                # constVariablePointer)
                read = ct
                if (row["role"] == "terminated" and not ct.startswith("const ")
                        and _uses(body, finish_binding(row["name"]))):
                    read = f"const {ct}"
                lines.append(f"{read} const {field} = args->{field};")
            if row["role"] == "lookup":
                cell = param_cell_binding(row["name"])
                if _uses(body, cell):
                    elem = ct.removeprefix("const ").rstrip("*")
                    lines.append(f"const {elem} {cell} = args->{field}[0];")
            if row["role"] == "terminated":
                twin = finish_binding(row["name"])
                if _uses(body, twin):
                    lines.append(f"{ct.removeprefix('const ')} const {twin} = "
                                 f"args->{field};")
        elif _uses(body, field):
            prefix = "" if ct.startswith("const ") else "const "
            lines.append(f"{prefix}{ct} {field} = args->{field};")
    return lines


def _run(name: str, fields: list[dict], body: str, walk) -> list[str]:
    """The static run function: bindings, then the sample loop around the body."""
    bindings = _bindings(fields, body, walk)
    lines = [f"static void {name}_run(const {name}_args_t* args, int64_t base, "
             "int64_t count, int64_t n_samples)", "{"]
    if not bindings:
        lines.append("    (void)args;")
    if not any(_uses(text, "n_samples") for text in (body, *bindings)):
        lines.append("    (void)n_samples;")
    lines += [f"    {b}" for b in bindings]
    lines += ["    const int64_t hawk_end = base + count;",
              "    for (int64_t hawk_i = base; hawk_i < hawk_end; ++hawk_i) {",
              body,
              "    }",
              "}"]
    return lines


def tu_text(name: str, walk, mode, body: str, writer, core_closure: str,
            helper_closure: str, *, rows: list) -> str:
    """The ``<name>.c`` text: includes, the guarded size asserts, the core
    closure, the helper closure, the run function and the validating entry.
    ``rows`` are the report's interface rows (see :func:`args_struct`)."""
    from .. import __version__

    _struct, fields = args_struct(rows, walk, mode, name)
    blocks = [
        "\n".join([f"/* {name}.c -- generated by hawk {__version__}; do not edit. "
                   f"See {name}.h. */", *_includes(name), *_size_asserts(name, mode)]),
    ]
    if core_closure:
        blocks.append(core_closure.rstrip("\n"))
    if helper_closure:
        blocks.append(helper_closure.rstrip("\n"))
    blocks.append("\n".join(_run(name, fields, body, walk)))
    blocks.append("\n".join(_validation(name, fields)))
    return "\n\n".join(blocks) + "\n"


__all__ = ["args_struct", "core_closure", "header_text", "helper_closure", "tu_text"]

