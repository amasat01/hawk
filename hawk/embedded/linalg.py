# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C helpers of the per-lane linear-algebra kinds (:mod:`hawk.ir.linalg`).

Each helper is the looped C spelling of one ``aether::linalg`` function
(``aether/linalg/``), in the SAME order of floating-point operations, so a lane
computes the same bits as the host and device builds: column-by-column Crout
elimination with a pivot search that keeps the first of equals, forward-ordered
dot products in the triangular solves, the fixed-sweep Jacobi with the
Rutishauser rotation. A factorisation is a named struct (``hawk_lu_n<N>_<w>_t``
with the matrix, the pivots and ``info``; the Cholesky factor and the
eigendecomposition likewise) filled by loops; its consumers take it and the
right-hand side by ``const`` pointer and return the result by value. A
dimension of one is straight code. ``info`` is never written with a bitwise
operator (Rule 10.1): the Jacobi bits are added, the first before the second can
be set.

Names: ``hawk_lu_n<N>_<w>``, ``hawk_lu_solve_n<N>_<rhs>_<w>``,
``hawk_lu_solve_t_...``, ``hawk_lu_det_...``, ``hawk_chol_...``,
``hawk_chol_from_...``, ``hawk_chol_factor_...``, ``hawk_cho_solve_...``, ``hawk_tri_lower_...``,
``hawk_tri_upper_t_...``, ``hawk_eigh_...``; ``<rhs>`` is ``v`` for a vector
and ``k<K>`` for an ``N x K`` matrix, ``<w>`` the width word.
"""

from __future__ import annotations

from aether_dsc import c_core
from aether_dsc.c_core import CFunction

from ..ir import HawkError
from ..ir.linalg import MAX_DIM, sweeps
from .spelling import WIDTH, int_literal, libm_name, real_literal

#: The type helpers, by factor kind.
TYPES = ("lu_t", "chol_t", "eigh_t")
#: The function helpers over one dimension: shape ``(n,)``.
FACTOR_FUNCTIONS = ("lu", "lu_det", "chol", "chol_from", "chol_factor", "eigh")
#: The solves, over the right-hand side's shape: ``(n,)`` or ``(n, k)``.
SOLVES = ("lu_solve", "lu_solve_t", "cho_solve", "tri_lower", "tri_upper_t")
#: Every linear-algebra helper base, in emission order (types first).
BASES = TYPES + FACTOR_FUNCTIONS + SOLVES

_EPS = {"f64": 2.220446049250313e-16, "f32": 1.1920928955078125e-07}


class _Text:
    """Lines of C with an indentation level."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 1

    def add(self, *lines: str) -> None:
        self.lines += [f"{'    ' * self.depth}{line}" for line in lines]

    def block(self, header: str, body) -> None:
        """``header {``, ``body()`` one level deeper, ``}``."""
        self.add(f"{header} {{")
        self.depth += 1
        body()
        self.depth -= 1
        self.add("}")

    def loop(self, var: str, start: str, stop: str, body) -> None:
        self.block(f"for (int32_t {var} = {start}; {var} < {stop}; ++{var})", body)


def _i(value: int) -> str:
    return int_literal(value, "int32")


def _name(base: str, shape: tuple, w: str) -> str:
    n = shape[0]
    if base.endswith("_t") and base in TYPES:
        return f"hawk_{base[:-2]}_n{n}_{w}_t"
    if base in SOLVES:
        rhs = "v" if len(shape) == 1 else f"k{shape[1]}"
        return f"hawk_{base}_n{n}_{rhs}_{w}"
    return f"hawk_{base}_n{n}_{w}"


def _vec(n: int, w: str) -> str:
    return c_core.struct((n,), w).name


def _mat(n: int, k: int, w: str) -> str:
    return c_core.struct((n, k), w).name


def _check(base: str, shape: tuple) -> int:
    if base not in BASES:
        raise HawkError(f"unknown linear-algebra helper {base!r}")
    rank = (1, 2) if base in SOLVES else (1,)
    if len(shape) not in rank or not 1 <= shape[0] <= MAX_DIM:
        raise HawkError(f"helper {base!r}: bad shape {shape} (1 <= n <= {MAX_DIM})")
    return shape[0]


def _header(base: str, shape: tuple, mode, signature: str, t: _Text) -> str:
    lines = [f"/* hawk: la_{base} {tuple(shape)} {mode.id} */",
             f"static inline {signature}", "{", *t.lines, "}"]
    return "\n".join(lines) + "\n"


class _Spelling:
    """What the generators of one ``(width, mode)`` share."""

    def __init__(self, mode) -> None:
        self.w = WIDTH[mode.id]
        self.mode = mode
        self.fabs = libm_name("fabs", mode)
        self.sqrt = libm_name("sqrt", mode)
        self.libm = (self.fabs, self.sqrt)

    def r(self, value: float) -> str:
        return real_literal(value, self.mode)


# ---- the types --------------------------------------------------------------

def _type(base: str, n: int, s: _Spelling) -> CFunction:
    name = _name(base, (n,), s.w)
    m, v = _mat(n, n, s.w), _vec(n, s.w)
    members = {"lu_t": f"{m} a; int32_t piv[{n}]; int32_t info;",
               "chol_t": f"{m} l; int32_t info;",
               "eigh_t": f"{v} w; {m} v; int32_t info;"}[base]
    guard = name.upper()
    text = (f"#ifndef {guard}\n#define {guard}\n"
            f"typedef struct {{ {members} }} {name};\n#endif\n")
    return CFunction(name, text, (m, v) if base == "eigh_t" else (m,), ())


# ---- right-hand-side bookkeeping --------------------------------------------

def _cell(var: str, shape: tuple, row: str, col: str | int | None, arrow: bool = False) -> str:
    sep = "->" if arrow else "."
    if len(shape) == 1:
        return f"{var}{sep}v[{row}]"
    return f"{var}{sep}m[{row}][{col}]"


def _columns(t: _Text, shape: tuple, body) -> None:
    """``body(col)`` once per right-hand-side column: a loop, or the vector
    itself, or (at a dimension of one) the columns written out."""
    if len(shape) == 1:
        body(None)
    elif shape[0] == 1:
        for c in range(shape[1]):
            t.add("{")
            t.depth += 1
            body(str(c))
            t.depth -= 1
            t.add("}")
    else:
        t.loop("c", _i(0), _i(shape[1]), lambda: body("c"))


def _rows(t: _Text, n: int, body) -> None:
    """``body(row)`` for every row: a loop, or row ``0`` of a dimension of one."""
    if n == 1:
        body("0")
    else:
        t.loop("i", _i(0), _i(n), lambda: body("i"))


def _zero_rhs(t: _Text, s: _Spelling, shape: tuple, out: str) -> None:
    _rows(t, shape[0], lambda i: _columns(
        t, shape, lambda c: t.add(f"{_cell(out, shape, i, c)} = {s.r(0.0)};")))


def _copy_rhs(t: _Text, shape: tuple, src: str, out: str) -> None:
    _rows(t, shape[0], lambda i: _columns(
        t, shape, lambda c: t.add(
            f"{_cell(out, shape, i, c)} = {_cell(src, shape, i, c, True)};")))


def _swap_rows(t: _Text, shape: tuple, out: str, a: str, b: str) -> None:
    def swap(c) -> None:
        t.add(f"const hawk_real_t tmp = {_cell(out, shape, a, c)};",
              f"{_cell(out, shape, a, c)} = {_cell(out, shape, b, c)};",
              f"{_cell(out, shape, b, c)} = tmp;")
    _columns(t, shape, swap)


# ---- factorisations ---------------------------------------------------------

def _lu(n: int, s: _Spelling) -> CFunction:
    ft, m = _name("lu_t", (n,), s.w), _mat(n, n, s.w)
    t = _Text()
    t.add(f"{ft} f;")
    zero = s.r(0.0)
    if n == 1:
        t.add("f.a.m[0][0] = a->m[0][0];", "f.piv[0] = INT32_C(0);", "f.info = INT32_C(1);")
        t.block(f"if (f.a.m[0][0] != {zero})", lambda: t.add("f.info = INT32_C(0);"))
    else:
        t.loop("i", _i(0), _i(n), lambda: t.loop(
            "c", _i(0), _i(n), lambda: t.add("f.a.m[i][c] = a->m[i][c];")))
        t.add("f.info = INT32_C(0);")

        def column() -> None:
            def entry() -> None:
                t.add("const int32_t m = (i < j) ? i : j;", f"hawk_real_t acc = {zero};")
                t.loop("k", _i(0), "m", lambda: t.add(
                    "acc = acc + (f.a.m[i][k] * f.a.m[k][j]);"))
                t.add("f.a.m[i][j] = f.a.m[i][j] - acc;")
            t.loop("i", _i(0), _i(n), entry)
            t.add("int32_t p = j;", f"hawk_real_t big = {s.fabs}(f.a.m[j][j]);")

            def search() -> None:
                t.add(f"const hawk_real_t mag = {s.fabs}(f.a.m[i][j]);")
                t.block("if (mag > big)", lambda: t.add("big = mag;", "p = i;"))
            t.loop("i", f"j + {_i(1)}", _i(n), search)
            t.add("f.piv[j] = p;")

            def pivot() -> None:
                t.block("if (p != j)", lambda: t.loop("c", _i(0), _i(n), lambda: t.add(
                    "const hawk_real_t tmp = f.a.m[j][c];",
                    "f.a.m[j][c] = f.a.m[p][c];", "f.a.m[p][c] = tmp;")))
                t.add(f"const hawk_real_t rcp = {s.r(1.0)} / f.a.m[j][j];")
                t.loop("i", f"j + {_i(1)}", _i(n), lambda: t.add(
                    "f.a.m[i][j] = f.a.m[i][j] * rcp;"))
            t.block(f"if (f.a.m[p][j] != {zero})", pivot)
            t.add("else {", f"    if (f.info == {_i(0)}) {{", f"        f.info = j + {_i(1)};",
                  "    }", "}")
        t.loop("j", _i(0), _i(n), column)
    t.add("return f;")
    sig = f"{ft} {_name('lu', (n,), s.w)}(const {m}* a)"
    return CFunction(_name("lu", (n,), s.w), _header("lu", (n,), s.mode, sig, t),
                     (m,), (s.fabs,))


def _chol(n: int, s: _Spelling) -> CFunction:
    ft, m = _name("chol_t", (n,), s.w), _mat(n, n, s.w)
    zero, half = s.r(0.0), s.r(0.5)
    t = _Text()
    t.add(f"{ft} f;", "f.info = INT32_C(0);")

    def sym(i: str, j: str) -> str:
        return f"({half} * a->m[{i}][{j}]) + ({half} * a->m[{j}][{i}])"
    if n == 1:
        t.add(f"f.l.m[0][0] = {zero};", f"const hawk_real_t d = {sym('0', '0')};")
        t.block(f"if (d > {zero})", lambda: t.add(f"f.l.m[0][0] = {s.sqrt}(d);"))
        t.add("else {", "    f.info = INT32_C(1);", "}")
    else:
        t.loop("i", _i(0), _i(n), lambda: t.loop(
            "c", _i(0), _i(n), lambda: t.add(f"f.l.m[i][c] = {zero};")))

        def column() -> None:
            def live() -> None:
                t.add(f"hawk_real_t d = {sym('j', 'j')};")
                t.loop("k", _i(0), "j", lambda: t.add(
                    "d = d - (f.l.m[j][k] * f.l.m[j][k]);"))

                def positive() -> None:
                    t.add(f"const hawk_real_t ljj = {s.sqrt}(d);", "f.l.m[j][j] = ljj;")

                    def below() -> None:
                        t.add(f"hawk_real_t acc = {sym('i', 'j')};")
                        t.loop("k", _i(0), "j", lambda: t.add(
                            "acc = acc - (f.l.m[i][k] * f.l.m[j][k]);"))
                        t.add("f.l.m[i][j] = acc / ljj;")
                    t.loop("i", f"j + {_i(1)}", _i(n), below)
                t.block(f"if (d > {zero})", positive)
                t.add("else {", f"    f.info = j + {_i(1)};", "}")
            t.block(f"if (f.info == {_i(0)})", live)
        t.loop("j", _i(0), _i(n), column)
    t.add("return f;")
    sig = f"{ft} {_name('chol', (n,), s.w)}(const {m}* a)"
    return CFunction(_name("chol", (n,), s.w), _header("chol", (n,), s.mode, sig, t),
                     (m,), (s.sqrt,))


def _chol_from(n: int, s: _Spelling) -> CFunction:
    """``aether::linalg::choleskyFrom``: the lower triangle of ``a`` as the factor, ``info`` the
    first column whose diagonal entry is not positive and finite, the whole factor zero then."""
    ft, m = _name("chol_t", (n,), s.w), _mat(n, n, s.w)
    zero = s.r(0.0)
    t = _Text()
    t.add(f"{ft} f;", "f.info = INT32_C(0);")

    def unusable(d: str) -> str:
        return f"if (({d} <= {zero}) || (isfinite({d}) == 0))"
    if n == 1:
        t.add("const hawk_real_t d = a->m[0][0];", f"f.l.m[0][0] = {zero};")
        t.block(unusable("d"), lambda: t.add("f.info = INT32_C(1);"))
        t.add("else {", "    f.l.m[0][0] = d;", "}")
    else:
        def column() -> None:
            t.add("const hawk_real_t d = a->m[j][j];")
            t.block(f"if (f.info == {_i(0)})", lambda: (
                t.block(unusable("d"), lambda: t.add(f"f.info = j + {_i(1)};"))))
        t.loop("j", _i(0), _i(n), column)
        t.loop("i", _i(0), _i(n), lambda: t.loop("c", _i(0), _i(n), lambda: t.add(
            f"f.l.m[i][c] = ((f.info == {_i(0)}) && (c <= i)) ? a->m[i][c] : {zero};")))
    t.add("return f;")
    sig = f"{ft} {_name('chol_from', (n,), s.w)}(const {m}* a)"
    return CFunction(_name("chol_from", (n,), s.w), _header("chol_from", (n,), s.mode, sig, t),
                     (m,), ())


def _eigh(n: int, s: _Spelling) -> CFunction:
    ft, m, v = _name("eigh_t", (n,), s.w), _mat(n, n, s.w), _vec(n, s.w)
    zero, one, half = s.r(0.0), s.r(1.0), s.r(0.5)
    t = _Text()
    t.add(f"{ft} res;")
    if n == 1:
        t.add("res.w.v[0] = a->m[0][0];", f"res.v.m[0][0] = {one};", "res.info = INT32_C(0);",
              "return res;")
        sig = f"{ft} {_name('eigh', (n,), s.w)}(const {m}* a)"
        return CFunction(_name("eigh", (n,), s.w), _header("eigh", (n,), s.mode, sig, t),
                         (m,), ())
    eps = s.r(_EPS[s.w])
    t.add("res.info = INT32_C(0);", f"hawk_real_t s[{n}][{n}];", f"hawk_real_t fro2 = {zero};")
    t.loop("i", _i(0), _i(n), lambda: t.loop(
        "c", _i(0), _i(n), lambda: t.add(f"res.v.m[i][c] = {zero};")))

    def fill() -> None:
        def entry() -> None:
            t.block("if (j >= i)", lambda: t.add(
                f"s[i][j] = ({half} * a->m[i][j]) + ({half} * a->m[j][i]);"))
            t.add("else {", "    s[i][j] = s[j][i];", "}")
            t.add("fro2 = fro2 + (s[i][j] * s[i][j]);")
        t.loop("j", _i(0), _i(n), entry)
        t.add(f"res.v.m[i][i] = {one};")
    t.loop("i", _i(0), _i(n), fill)

    def rotate() -> None:
        t.add("const hawk_real_t app = s[p][p];", "const hawk_real_t aqq = s[q][q];",
              "const hawk_real_t apq = s[p][q];")

        def rotation() -> None:
            t.add(f"const hawk_real_t theta = (aqq - app) / ({s.r(2.0)} * apq);",
                  f"const hawk_real_t sg = (theta < {zero}) ? {s.r(-1.0)} : {one};",
                  f"const hawk_real_t tn = sg / ({s.fabs}(theta) + "
                  f"{s.sqrt}((theta * theta) + {one}));",
                  f"const hawk_real_t cs = {one} / {s.sqrt}((tn * tn) + {one});",
                  "const hawk_real_t sn = tn * cs;",
                  "s[p][p] = app - (tn * apq);", "s[q][q] = aqq + (tn * apq);",
                  f"s[p][q] = {zero};", f"s[q][p] = {zero};")

            def rows() -> None:
                def other() -> None:
                    t.add("const hawk_real_t akp = s[k][p];", "const hawk_real_t akq = s[k][q];",
                          "const hawk_real_t np = (cs * akp) - (sn * akq);",
                          "const hawk_real_t nq = (sn * akp) + (cs * akq);",
                          "s[k][p] = np;", "s[p][k] = np;", "s[k][q] = nq;", "s[q][k] = nq;")
                t.block("if ((k != p) && (k != q))", other)
                t.add("const hawk_real_t vp = res.v.m[k][p];",
                      "const hawk_real_t vq = res.v.m[k][q];",
                      "res.v.m[k][p] = (cs * vp) - (sn * vq);",
                      "res.v.m[k][q] = (sn * vp) + (cs * vq);")
            t.loop("k", _i(0), _i(n), rows)
        t.block(f"if (!({s.fabs}(apq) <= ({eps} * {s.sqrt}({s.fabs}(app * aqq)))))",
                rotation)
    t.loop("sweep", _i(0), _i(sweeps(n)), lambda: t.loop(
        "p", _i(0), _i(n - 1), lambda: t.loop("q", "p + INT32_C(1)", _i(n), rotate)))
    t.add(f"hawk_real_t off2 = {zero};")

    def diagonal() -> None:
        t.add("res.w.v[i] = s[i][i];")
        t.loop("j", _i(0), _i(n), lambda: t.block("if (j != i)", lambda: t.add(
            "off2 = off2 + (s[i][j] * s[i][j]);")))
    t.loop("i", _i(0), _i(n), diagonal)
    t.add(f"const hawk_real_t tol = {s.r(float(n))} * {eps};")
    t.block("if (!(off2 <= ((tol * tol) * fro2)))", lambda: t.add("res.info = INT32_C(1);"))

    def order() -> None:
        t.add("int32_t lo = i;")
        t.loop("j", f"i + {_i(1)}", _i(n), lambda: t.block(
            "if (res.w.v[j] < res.w.v[lo])", lambda: t.add("lo = j;")))

        def exchange() -> None:
            t.add("const hawk_real_t tw = res.w.v[i];", "res.w.v[i] = res.w.v[lo];",
                  "res.w.v[lo] = tw;")
            t.loop("k", _i(0), _i(n), lambda: t.add(
                "const hawk_real_t tv = res.v.m[k][i];", "res.v.m[k][i] = res.v.m[k][lo];",
                "res.v.m[k][lo] = tv;"))
        t.block("if (lo != i)", exchange)
    t.loop("i", _i(0), _i(n - 1), order)

    def sign() -> None:
        t.add("int32_t big = INT32_C(0);")
        t.loop("k", _i(1), _i(n), lambda: t.block(
            f"if ({s.fabs}(res.v.m[k][j]) > {s.fabs}(res.v.m[big][j]))",
            lambda: t.add("big = k;")))
        t.block(f"if (res.v.m[big][j] < {zero})", lambda: t.loop(
            "k", _i(0), _i(n), lambda: t.add("res.v.m[k][j] = -res.v.m[k][j];")))
    t.loop("j", _i(0), _i(n), sign)
    t.add(f"const hawk_real_t gap = tol * {s.sqrt}(fro2);", "bool tight = false;")
    t.loop("i", _i(0), _i(n - 1), lambda: t.block(
        "if (!((res.w.v[i + INT32_C(1)] - res.w.v[i]) > gap))", lambda: t.add("tight = true;")))
    t.block("if (tight)", lambda: t.add("res.info = res.info + INT32_C(2);"))
    t.add("return res;")
    sig = f"{ft} {_name('eigh', (n,), s.w)}(const {m}* a)"
    return CFunction(_name("eigh", (n,), s.w), _header("eigh", (n,), s.mode, sig, t),
                     (m, v), s.libm)


# ---- reads of a factor ------------------------------------------------------

def _lu_det(n: int, s: _Spelling) -> CFunction:
    ft = _name("lu_t", (n,), s.w)
    t = _Text()
    t.add(f"hawk_real_t d = {s.r(0.0)};")
    if n == 1:
        t.add("d = (f->info == INT32_C(0)) ? f->a.m[0][0] : d;")
    else:
        def live() -> None:
            t.add(f"d = {s.r(1.0)};")
            t.loop("k", _i(0), _i(n), lambda: (
                t.add("d = d * f->a.m[k][k];"),
                t.block("if (f->piv[k] != k)", lambda: t.add("d = -d;"))))
        t.block(f"if (f->info == {_i(0)})", live)
    t.add("return d;")
    sig = f"hawk_real_t {_name('lu_det', (n,), s.w)}(const {ft}* f)"
    return CFunction(_name("lu_det", (n,), s.w), _header("lu_det", (n,), s.mode, sig, t),
                     (), ())


def _chol_factor(n: int, s: _Spelling) -> CFunction:
    ft, m = _name("chol_t", (n,), s.w), _mat(n, n, s.w)
    t = _Text()
    t.add(f"{m} out;")
    cell = ("(f->info == INT32_C(0)) ? f->l.m[{i}][{j}] : " + s.r(0.0))
    if n == 1:
        t.add("out.m[0][0] = " + cell.format(i=0, j=0) + ";")
    else:
        t.loop("i", _i(0), _i(n), lambda: t.loop("j", _i(0), _i(n), lambda: t.add(
            "out.m[i][j] = " + cell.format(i="i", j="j") + ";")))
    t.add("return out;")
    sig = f"{m} {_name('chol_factor', (n,), s.w)}(const {ft}* f)"
    return CFunction(_name("chol_factor", (n,), s.w),
                     _header("chol_factor", (n,), s.mode, sig, t), (m,), ())


# ---- solves -----------------------------------------------------------------

def _rhs_struct(shape: tuple, w: str) -> str:
    return c_core.struct(tuple(shape), w).name


def _dot_rows(t: _Text, s: _Spelling, shape: tuple, out: str, lower: bool,
              entry_of, diag: str | None, src: str, trailing: bool) -> None:
    """The triangular sweep shared by ``tri_lower``, ``tri_upper_t`` and the LU
    solves: row ``i`` of ``out`` is ``src[i] - sum entry_of(i, j) out[j]`` over
    the already-solved rows ``j``, divided by ``diag`` when given. ``lower``
    runs ``i`` ascending over ``j < i``, otherwise ``i`` descending (``r`` is the
    count of rows done) over ``j > i``."""
    n = shape[0]

    def row(i: str) -> None:
        def column(c) -> None:
            t.add(f"hawk_real_t acc = {src.format(i=i, c=c)};")
            if n > 1:
                lo, hi = (_i(0), i) if lower else (f"{i} + {_i(1)}", _i(n))
                t.loop("j", lo, hi, lambda: t.add(
                    f"acc = acc - ({entry_of(i, 'j')} * {_cell(out, shape, 'j', c)});"))
            value = f"acc / {diag.format(i=i)}" if diag else "acc"
            t.add(f"{_cell(out, shape, i, c)} = {value};")
        _columns(t, shape, column)
    if n == 1:
        if diag is None:
            return                  # a unit triangle of one row changes nothing
        t.add("{")
        t.depth += 1
        row("0")
        t.depth -= 1
        t.add("}")
    elif lower:
        t.loop("i", _i(0), _i(n), lambda: row("i"))
    else:
        def down() -> None:
            t.add(f"const int32_t i = {_i(n - 1)} - r;")
            row("i")
        t.loop("r", _i(0), _i(n), down)


def _src(shape: tuple, var: str, arrow: bool) -> str:
    sep = "->" if arrow else "."
    return f"{var}{sep}v[{{i}}]" if len(shape) == 1 else f"{var}{sep}m[{{i}}][{{c}}]"


def _solve(base: str, shape: tuple, s: _Spelling) -> CFunction:
    n = shape[0]
    xt, m = _rhs_struct(shape, s.w), _mat(n, n, s.w)
    factor = "lu_t" if base.startswith("lu") else "chol_t"
    ft = _name(factor, (n,), s.w)
    t = _Text()
    t.add(f"{xt} x;")
    entry = "l" if factor == "chol_t" else "a"
    own = f"f->{entry}.m"

    if base in ("tri_lower", "tri_upper_t"):
        lower = base == "tri_lower"
        _dot_rows(t, s, shape, "x", lower, (lambda i, j: f"{own}[{i}][{j}]") if lower
                  else (lambda i, j: f"{own}[{j}][{i}]"), f"{own}[{{i}}][{{i}}]",
                  _src(shape, "b", True), True)
    elif base == "cho_solve":
        def live() -> None:
            _dot_rows(t, s, shape, "x", True, lambda i, j: f"{own}[{i}][{j}]",
                      f"{own}[{{i}}][{{i}}]", _src(shape, "b", True), True)
            _dot_rows(t, s, shape, "x", False, lambda i, j: f"{own}[{j}][{i}]",
                      f"{own}[{{i}}][{{i}}]", _src(shape, "x", False), True)
        t.block(f"if (f->info == {_i(0)})", live)
        t.add("else {")
        t.depth += 1
        _zero_rhs(t, s, shape, "x")
        t.depth -= 1
        t.add("}")
    else:
        def live() -> None:
            _copy_rhs(t, shape, "b", "x")
            transposed = base == "lu_solve_t"
            if not transposed:
                _lu_swaps(t, shape, n, ascending=True)
                _dot_rows(t, s, shape, "x", True, lambda i, j: f"{own}[{i}][{j}]", None,
                          _src(shape, "x", False), True)
                _dot_rows(t, s, shape, "x", False, lambda i, j: f"{own}[{i}][{j}]",
                          f"{own}[{{i}}][{{i}}]", _src(shape, "x", False), True)
            else:
                _dot_rows(t, s, shape, "x", True, lambda i, j: f"{own}[{j}][{i}]",
                          f"{own}[{{i}}][{{i}}]", _src(shape, "x", False), True)
                _dot_rows(t, s, shape, "x", False, lambda i, j: f"{own}[{j}][{i}]", None,
                          _src(shape, "x", False), True)
                _lu_swaps(t, shape, n, ascending=False)
        t.block(f"if (f->info == {_i(0)})", live)
        t.add("else {")
        t.depth += 1
        _zero_rhs(t, s, shape, "x")
        t.depth -= 1
        t.add("}")
    t.add("return x;")
    name = _name(base, shape, s.w)
    sig = f"{xt} {name}(const {ft}* f, const {xt}* b)"
    return CFunction(name, _header(base, shape, s.mode, sig, t), (xt, m), ())


def _lu_swaps(t: _Text, shape: tuple, n: int, ascending: bool) -> None:
    """The row exchanges of the pivot vector, in step order (``ascending``) or
    undone in reverse."""
    if n == 1:
        return

    def step(k: str) -> None:
        t.add(f"const int32_t p = f->piv[{k}];")
        t.block(f"if (p != {k})", lambda: _swap_rows(t, shape, "x", k, "p"))
    if ascending:
        t.loop("k", _i(0), _i(n), lambda: step("k"))
    else:
        def down() -> None:
            t.add(f"const int32_t k = {_i(n - 1)} - r;")
            step("k")
        t.loop("r", _i(0), _i(n), down)


# ---- the family ----------------------------------------------------------------

def helper(base: str, shape: tuple, *, mode) -> CFunction:
    """The helper ``base`` over ``shape`` under a scalar mode: ``(n,)`` for a
    type or a factorisation, the right-hand side's shape for a solve."""
    shape = tuple(shape)
    n = _check(base, shape)
    s = _Spelling(mode)
    if base in TYPES:
        return _type(base, n, s)
    if base in SOLVES:
        return _solve(base, shape, s)
    return {"lu": _lu, "lu_det": _lu_det, "chol": _chol, "chol_from": _chol_from,
            "chol_factor": _chol_factor, "eigh": _eigh}[base](n, s)
