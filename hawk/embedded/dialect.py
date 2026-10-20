# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The plain-C dialect: the standalone C target's spelling of every text the
body renderer produces, behind the :class:`hawk.emit.dialect.Dialect` seam.

Mirrors :mod:`hawk.emit.dialect_aether` member for member. ``r`` is the
renderer (``r._emit``, ``r._ref``, ``r.hoists``, ``r.indent``, ``r.loop_pos``).
Every rank>=1 value is a named C struct value (``aether_c_<w>_vecN_t``,
``aether_c_<w>_matRxC_t``); every fold, product and component-wise function
is an aether CORE function requested by ``(kind, shape, width)``
(:mod:`aether_dsc.c_core`); the plane access, clamps, aggregate glue and the
integer helpers are hawk's own (:mod:`hawk.embedded.helpers`). Rank-0 Real
math is libm at the mode's width (:func:`~hawk.embedded.spelling.libm_name`).

The dialect collects, as it spells, what the translation-unit assembler and
the report need: ``declared`` (every named local), ``helpers`` and
``helper_requests`` (hawk helpers by name, and how to regenerate each),
``core`` (core function names), ``libm``, ``fp_eq_sites``,
``literal_specials`` and ``shapes``. Index sites come from the verify pass's
ONE classification (``sites``: ``id(node) -> "proven" | "clamped"``), so a
proven index is emitted raw and every other one through a clamp. An op with
no C spelling raises :class:`~hawk.ir.HawkError` naming the kind and rank.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from aether_dsc import c_core

from ..emit.spelling import (
    _CWISE1,
    _INFIX,
    _MATH1,
    _MATH2,
    SAMPLE_INDEX_IDENT,
    binding_name,
    finish_binding,
    param_cell_binding,
)
from ..ir import HawkError
from ..ir.nodes import (
    FINISHED_PLANE,
    GATHER_ROLES,
    RAW_KIND,
    At,
    Const,
    Finish,
    Leaf,
    Op,
    Select,
    TapeWrite,
    is_param_cell,
)
from ..ir.ops import RANDOM_OPS
from ..types import INT_PLANE_DTYPES, TensorType
from .helpers import helper
from .spelling import (
    INT_HELPER,
    WIDTH,
    clamp_index,
    ctype,
    int_literal,
    libm_name,
    real_literal,
)

#: Real element dtypes (the scalar mode decides the width of both).
_REAL = ("f64", "f32")
#: Every integer element dtype; ``i32``/``i64`` are the 64-bit ``hawk_int_t``.
_INTS = ("i32", "i64") + INT_PLANE_DTYPES
#: Kinds an integer value always spells through the wrapping helpers.
_INT_KINDS = ("add", "sub", "mul", "neg", "abs", "min", "max", "div", "floordiv",
              "mod", "band", "bor", "bxor", "shl", "shr", "invert")
#: Rank-0 kinds that are aether CORE scalar functions, not libm.
_CORE_SCALARS = ("rsqrt", "clip", "remainder")
#: Rank-0 floating-point class tests: C99 macros, no width suffix.
_PREDICATES = ("isnan", "isinf", "isfinite")
#: Rank>=1 binary kinds a rank-0 operand splats into (the core takes two structs).
_SPLAT_BINARY = {"add": "add", "sub": "sub", "min": "min", "max": "max",
                 "pow": "pow"}
#: Rank>=1 quaternion and geometric kinds -> core kind (no shape: quaternions
#: are ``vec4``, ``(w, x, y, z)``).
_QUAT = {"quat_mul": "quat_mul", "quat_conj": "quat_conj", "quat_recip": "quat_recip",
         "quat_rotate": "quat_rotate", "as_pure": "quat_as_pure",
         "as_vec3": "quat_as_vec3"}
#: Bytes one element of a dtype occupies in the C build, by scalar mode.
_SIZEOF = {"f64": None, "f32": None, "i32": 8, "i64": 8, "int32": 4, "uint32": 4,
           "bool": 1}

_IDENT = r"[A-Za-z_]\w*"
#: Texts that are not composite expressions (MISRA C:2012 Rule 10.8): an
#: identifier, a plane read, a fixed-width literal, a cast real literal.
_ATOM = re.compile(
    rf"{_IDENT}|{_IDENT}\[[^\[\]]*\]|U?INT(?:32|64)_C\(-?\d+\)"
    r"|\(\(hawk_real_t\)-?[\w.+\-]+\)")


class CDialect:
    """The plain-C spelling (see the module docstring)."""

    id = "c"
    #: Every rank>=1 value is a C struct value and gets its own statement.
    names_every_tensor = True

    def __init__(self, mode, name: str, sites: dict, *, lengths: dict | None = None,
                 positions: dict | None = None) -> None:
        self.mode = mode
        self.width = WIDTH[mode.id]
        self.name = name
        #: ``id(node) -> "proven" | "clamped"``: verify's index-site table.
        self.sites = dict(sites)
        #: ``id(node) -> bound``: the site's length (binding name or int).
        self.lengths = dict(lengths or {})
        #: ``id(node) -> position`` (:func:`hawk.embedded.verify.positions`).
        self.positions = dict(positions or {})
        self.declared: list[tuple[str, TensorType]] = []
        self.helpers: set[str] = set()
        self.helper_requests: dict[str, tuple] = {}
        self.core: set[str] = set()
        self.core_functions: dict[str, c_core.CFunction] = {}
        #: Struct typedef names a declared local needs (always in the closure).
        self.structs: set[str] = set()
        self.libm: set[str] = set()
        self.fp_eq_sites: list = []
        self.literal_specials: list[str] = []
        self.shapes: set[tuple] = set()
        #: local name -> its type, for the members that get only text.
        self.types: dict[str, TensorType] = {}
        #: Bindings of planes read only through at() (``lookup``/``wide_in``).
        self._gathered: set[str] = set()

    # -- bookkeeping ---------------------------------------------------

    def _ctype(self, ttype: TensorType) -> str:
        if ttype.shape:
            self.shapes.add(tuple(ttype.shape))
            self.structs.add(c_core.struct(tuple(ttype.shape), self.width).name)
        return ctype(ttype, self.mode)

    def _helper(self, base: str, shape=None) -> str:
        shape = None if shape is None else tuple(shape)
        fn = helper(base, mode=self.mode, shape=shape)
        self.helpers.add(fn.name)
        self.helper_requests[fn.name] = (base, shape)
        if shape is not None:
            self.shapes.add(shape)
        return fn.name

    def _core(self, kind: str, shape=None, shape2=None) -> str:
        fn = c_core.function(kind, shape=shape, shape2=shape2, width=self.width)
        self.core.add(fn.name)
        self.core_functions[fn.name] = fn
        return fn.name

    def _libm(self, fn: str) -> str:
        name = libm_name(fn, self.mode)
        self.libm.add(name)
        return name

    def _record(self, name: str, ttype: TensorType) -> None:
        self.declared.append((name, ttype))
        self.types[name] = ttype

    def sizeof(self, ttype: TensorType) -> int:
        """Bytes one value of ``ttype`` occupies in this unit."""
        each = _SIZEOF[ttype.dtype] or (8 if self.mode.id == "float64" else 4)
        count = 1
        for extent in ttype.shape:
            count *= int(extent)
        return each * count

    def _atom(self, r, text: str, ttype: TensorType) -> str:
        """``text`` itself when it is not a composite expression, else a
        hoisted local (a cast or a member access never wraps an expression)."""
        if _ATOM.fullmatch(text):
            return text
        return self.hoist(r, text, ttype)

    def _named(self, r, text: str, ttype: TensorType) -> str:
        """``text`` as an identifier: a struct member is read off a name."""
        if re.fullmatch(_IDENT, text):
            return text
        return self.hoist(r, text, ttype)

    def _real(self, r, text: str, dtype: str) -> str:
        """A Real operand: an integer or bool value cast once, explicitly."""
        if dtype in _REAL:
            return text
        return f"((hawk_real_t){self._atom(r, text, TensorType((), dtype))})"

    def _coerce(self, r, text: str, source: str, target: str) -> str:
        """``text`` (typed ``source``) as a ``target`` value: only an
        integer-into-Real store needs a cast; the IR types everything else."""
        if target in _REAL and source not in _REAL:
            return self._real(r, text, source)
        return text

    def _status(self, node) -> str:
        return self.sites.get(id(node), "clamped")

    def _clamped(self, node, index: str, default) -> str:
        """``index`` raw when the site is proven, else clamped to its bound."""
        if self._status(node) == "proven":
            return index
        self._helper("clamp_index")
        length = self.lengths.get(id(node))
        length = default if length is None else length
        if isinstance(length, int):
            length = int_literal(length, "i64")
        return clamp_index(index, length)

    # -- declarations and loops ----------------------------------------

    def declare(self, name: str, ttype: TensorType, text: str) -> str:
        """A named local: ``const {ctype} {name} = {text};``."""
        if text in self._gathered:
            raise HawkError(
                f"renderer defect: a lookup/wide_in leaf reached declare "
                f"({name!r} = {text!r}); the renderer reads such a plane only "
                "through at() and never names its leaf")
        self._record(name, ttype)
        return f"const {self._ctype(ttype)} {name} = {text};"

    def carry_type(self, ttype: TensorType) -> str:
        """A loop-carried (or ``switch``-assigned) local's C type."""
        return self._ctype(ttype)

    def tape_decl(self, elem: str, name: str, trip: int) -> str:
        """A forward loop's per-iteration tape array."""
        return f"{elem} {name}[{trip}];"

    def counter_decl(self, counter: str) -> str:
        """A loop's per-sample iteration counter."""
        return f"hawk_int_t {counter} = {int_literal(0, 'i64')};"

    def for_header(self, r, index: str, loop, count: str | None) -> str:
        """The lowered loop's header. Static bounds are ``INT64_C`` literals;
        a RUNTIME count is first clamped into ``[0, trip]`` by its own
        statement, then the loop runs the first (head form) or the last
        (``count_tail``) ``cnt`` iterations of the static range, as
        :func:`hawk.emit.spelling._for_header` does, over an ``int32_t``."""
        cmp_ = "<" if loop.step > 0 else ">"
        if count is None:
            lit = lambda v: int_literal(v, "i64")  # noqa: E731
            incr = (f"++{index}" if loop.step == 1 else f"--{index}" if loop.step == -1
                    else f"{index} += {lit(loop.step)}")
            return (f"for (hawk_int_t {index} = {lit(loop.start)}; {index} {cmp_} "
                    f"{lit(loop.stop)}; {incr}) {{")
        lit = lambda v: int_literal(v, "int32")  # noqa: E731
        incr = (f"++{index}" if loop.step == 1 else f"--{index}" if loop.step == -1
                else f"{index} += {lit(loop.step)}")
        cnt = f"hawk_cnt{r.loop_pos[id(loop)]}"
        trips = self._helper("clamp_trips")
        r._emit(f"const int32_t {cnt} = {trips}({count}, {lit(loop.trip)});")
        if loop.count_tail:
            sign = "+" if loop.step > 0 else "-"
            # no `* 1`: an operation with no effect is dead code (Rule 2.2)
            span = f"({lit(loop.trip)} - {cnt})"
            if abs(loop.step) != 1:
                span = f"({span} * {lit(abs(loop.step))})"
            begin = f"{lit(loop.start)} {sign} {span}"
            end = lit(loop.stop)
        else:
            begin = lit(loop.start)
            scaled = cnt if loop.step == 1 else f"({cnt} * {lit(loop.step)})"
            end = (cnt if (loop.start, loop.step) == (0, 1)
                   else f"({lit(loop.start)} + {scaled})")
        return f"for (int32_t {index} = {begin}; {index} {cmp_} {end}; {incr}) {{"

    def tail_break(self, flag: str) -> str:
        """The tail-form exit, braced."""
        return f"if ({flag}) {{ break; }}"

    def exit_if(self, cond: str) -> str:
        """The ``if`` opening a head/middle-form ``break``. A condition that is
        already one parenthesised expression is not wrapped again (clang's
        ``-Wparentheses-equality`` rejects ``if ((a == b))``)."""
        return f"if {_paren(cond)} {{"

    def sample_index_value(self) -> str:
        """A ``SampleIndex`` node: the loop's own ``int64_t`` sample index."""
        return SAMPLE_INDEX_IDENT

    def hoist(self, r, text: str, ttype: TensorType | None = None) -> str:
        """Bind ``text`` to a fresh typed local ahead of the current line."""
        if ttype is None:
            raise HawkError("the C dialect hoists only a typed value "
                            f"(no type for {text!r})")
        name = f"hz{r.hoists}"
        r.hoists += 1
        self._record(name, ttype)
        r._emit(f"const {self._ctype(ttype)} {name} = {text};")
        return name

    # -- dispatch -------------------------------------------------------

    def selector_decl(self, r, kname: str, kind_expr: str, k: int, *,
                      in_range: bool = False) -> str:
        """The selector clamped into ``[0, K)`` as an ``int32_t``, over a named
        ``hawk_int_t`` operand. Always clamped, ``in_range`` or not: the verify
        pass records every selector as a clamped site, and the clamp is a no-op
        on an in-range value."""
        if not re.fullmatch(_IDENT, kind_expr):
            kind_expr = self.hoist(r, kind_expr, TensorType((), "i64"))
        x = kind_expr
        return (f"const int32_t {kname} = (({x} < {_i64(0)}) ? {_i32(0)} : "
                f"(({x} > {_i64(k - 1)}) ? {_i32(k - 1)} : (int32_t){x}));")

    def mix_count(self, mix: str, k: int, kname: str) -> str:
        """An ``auto`` dispatch's branch-mix count, one ``uint32_t`` per branch."""
        return f"{mix}[{kname}] = {mix}[{kname}] + 1U;"

    def predicated_rank0(self, kname: str, texts: Sequence[str]) -> str:
        """A right-nested ternary chain over every branch."""
        expr = texts[-1]
        for j in range(len(texts) - 2, -1, -1):
            expr = f"(({kname} == {_i32(j)}) ? {texts[j]} : {expr})"
        return expr

    def predicated_tensor(self, r, kname: str, texts: Sequence[str]) -> str:
        """Nested ``hawk_{S}_select`` calls, the branches named struct values."""
        ttype = self.types.get(texts[-1])
        if ttype is None or not ttype.shape:
            raise HawkError(
                f"a rank>=1 predicated dispatch in {self.name!r} reached the C "
                f"dialect with an unnamed branch ({texts[-1]!r}); use policy='switch'")
        select = self._helper("select", ttype.shape)
        expr = texts[-1]
        for j in range(len(texts) - 2, -1, -1):
            expr = f"{select}(({kname} == {_i32(j)}), {texts[j]}, {expr})"
        return expr

    def switch_default_tail(self) -> str:
        """Every ``default`` clause ends in a ``break`` (MISRA C:2012 16.3)."""
        return "break;"

    # -- sinks and the guard --------------------------------------------

    def guard(self, bound: Sequence[str]) -> str | None:
        """The guard's condition: any bound ``uint8_t`` mask set at this sample."""
        if not bound:
            return None
        reads = [self._mask(b) for b in bound]
        return reads[0] if len(reads) == 1 else "(" + " || ".join(reads) + ")"

    @staticmethod
    def _mask(binding: str) -> str:
        return f"({binding}[{SAMPLE_INDEX_IDENT}] != (uint8_t)0)"

    def store(self, r, sink, target: str, value: str) -> None:
        """The own column: a rank-0 element, or a SoA store with pitch ``n_samples``."""
        value = self._coerce(r, value, sink.value.ttype.dtype, sink.ttype.dtype)
        if sink.ttype.shape:
            store = self._helper("store", sink.ttype.shape)
            r._emit(f"{store}({target}, {SAMPLE_INDEX_IDENT}, n_samples, {value});")
            return
        r._emit(f"{target}[{SAMPLE_INDEX_IDENT}] = {value};")

    def scatter_store(self, r, sink, target: str, index: str, value: str) -> None:
        """A scattered wide write: a store at the clamped computed row."""
        value = self._coerce(r, value, sink.value.ttype.dtype, sink.ttype.dtype)
        row = self._clamped(sink, index, f"{target}_len")
        r._emit(f"{target}[{row}] = {value};")

    def accum_scattered(self, r, sink, target: str, index: str, value: str) -> None:
        """A scattered accumulate: a serial ``+=`` at the clamped computed row."""
        value = self._coerce(r, value, sink.value.ttype.dtype, sink.ttype.dtype)
        row = self._clamped(sink, index, f"{target}_len")
        r._emit(f"{target}[{row}] += {value};")

    def store_compensated(self, r, sink, target: str, comp: str, value: str) -> None:
        """Refused: the compensated sink policy is not lowered (verify refuses
        it first; this is the belt and braces)."""
        raise HawkError(f"{self.name!r}: the C target does not lower the "
                        "compensated sink policy; use the plain sink policy")

    def sample_sum(self, r, sink, column: int) -> None:
        """Refused (belt and braces; the verify pass refuses it first): a sum
        over samples needs the wrapper's cross-sample reduction."""
        raise HawkError(f"{self.name!r}: the C target does not lower a sum over "
                        "samples; commit the own column and reduce in the caller")

    # -- expressions ----------------------------------------------------

    def leaf(self, r, leaf: Leaf, materialised: bool) -> str:
        name = binding_name(leaf.role, leaf.name)
        t = leaf.ttype
        if leaf.role == "nsamples":
            return "n_samples"
        if leaf.role == "uniform":
            if t.dtype in INT_PLANE_DTYPES:
                return f"(({self._ctype(t)}){name})"
            return name
        if leaf.role == "terminated":
            return self._mask(name)
        if leaf.role in GATHER_ROLES:
            # Read only through at(): the renderer still visits the plane leaf,
            # whose text no expression uses, and never names it (:meth:`declare`
            # keeps a belt-and-braces check).
            self._gathered.add(name)
            return name
        if t.shape:
            if t.dtype not in _REAL:
                raise HawkError(f"{self.name!r}: a rank-{len(t.shape)} {t.dtype} plane "
                                f"({leaf.name!r}) has no C load")
            load = self._helper("load", t.shape)
            return f"{load}({name}, {SAMPLE_INDEX_IDENT}, n_samples)"
        return f"{name}[{SAMPLE_INDEX_IDENT}]"

    def const(self, node: Const) -> str:
        t = node.ttype
        if t.shape:
            raise HawkError(
                f"a rank-{len(t.shape)} literal typed {t!r} has no C spelling; "
                "a broadcast literal is a rank-0 constant under a 'splat'")
        if t.dtype == "bool" or t.dtype in _INTS:
            return _int_zero_like(t.dtype, int(node.literal))
        text = real_literal(float(node.literal), self.mode)
        if "NAN" in text or "INFINITY" in text:
            self.literal_specials.append(text[len("((hawk_real_t)"):-1])
        return text

    def at(self, r, node: At) -> str:
        if is_param_cell(node):
            return param_cell_binding(node.plane.name)
        if node.ttype.shape:
            raise HawkError(f"{self.name!r}: a rank-{len(node.ttype.shape)} at() read "
                            "has no C spelling")
        plane = binding_name(node.plane.role, node.plane.name)
        index = self._clamped(node, r._ref(node.index), f"{plane}_len")
        return f"{plane}[{index}]"

    def select(self, r, node: Select, c: str, a: str, b: str) -> str:
        """A rank-0 ternary, or ``hawk_{S}_select`` on a rank-0 condition."""
        t = node.ttype
        if not t.shape:
            a = self._coerce(r, a, node.operands[1].ttype.dtype, t.dtype)
            b = self._coerce(r, b, node.operands[2].ttype.dtype, t.dtype)
            return f"({c} ? {a} : {b})"
        if node.operands[0].ttype.shape:
            raise HawkError(f"{self.name!r}: a select on a rank>=1 condition is not "
                            "lowered; select per component with a rank-0 condition")
        return f"{self._helper('select', t.shape)}({c}, {a}, {b})"

    def op(self, r, node: Op, args: Sequence[str], ranks: Sequence[int]) -> str:
        kind, rank = node.kind, len(node.ttype.shape)
        if kind == RAW_KIND or kind in RANDOM_OPS:
            raise HawkError(f"{self.name!r}: op {kind!r} has no C spelling "
                            "(the C target refuses it)")
        if rank == 0 and all(rk == 0 for rk in ranks):
            return self._scalar_op(r, kind, args, node)
        return self._tensor_op(r, kind, args, ranks, node)

    def _scalar_op(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        dtype = node.ttype.dtype
        srcs = [o.ttype.dtype for o in node.operands]
        if kind == "cast":
            return self._cast_op(r, args[0], node)
        if dtype in _INTS and kind in _INT_KINDS:
            # Int and int-plane arithmetic: the wrapping, numpy-defined helpers.
            fn = self._helper(INT_HELPER[(kind, dtype)].removeprefix("hawk_"))
            return f"{fn}({', '.join(args)})"
        if kind in _PREDICATES:
            return f"({kind}({args[0]}) != 0)"
        if kind in ("eq", "ne") and any(s in _REAL for s in srcs):
            self.fp_eq_sites.append(self.positions.get(id(node)))
        reals = ([self._real(r, a, s) for a, s in zip(args, srcs)]
                 if dtype in _REAL or kind in ("lt", "le", "gt", "ge", "eq", "ne")
                 and any(s in _REAL for s in srcs) else list(args))
        if kind == "spacing":
            return f"{self._helper('spacing')}({reals[0]})"
        if kind == "sign":
            return f"{self._helper('sign_nan')}({reals[0]})"
        if kind in _CORE_SCALARS:
            return f"{self._core(kind)}({', '.join(reals)})"
        if kind == "abs":
            return f"{self._libm('fabs')}({reals[0]})"
        if kind in _MATH1:
            return f"{self._libm(_MATH1[kind])}({reals[0]})"
        if kind in _MATH2:
            return f"{self._libm(_MATH2[kind])}({reals[0]}, {reals[1]})"
        if kind == "fma":
            return f"{self._libm('fma')}({', '.join(reals)})"
        if kind in _INFIX and len(args) == 2:
            return f"({reals[0]} {_INFIX[kind]} {reals[1]})"
        if kind == "neg":
            return f"(-{reals[0]})"
        if kind == "lnot":
            return f"(!{args[0]})"
        if kind in ("dot", "norm", "sum", "component"):
            return self._tensor_op(r, kind, args, [0] * len(args), node)
        raise HawkError(_no_spelling(kind, 0))

    def _cast_op(self, r, arg: str, node: Op) -> str:
        """``cast``: Real -> integer through the saturating ``from_real``
        helper, integer -> Real and integer -> integer as an explicit C cast
        of a non-composite operand (Rule 10.8)."""
        target, source = node.ttype.dtype, node.operands[0].ttype.dtype
        if target in _REAL and source in _REAL:
            return arg
        if target in _INTS and source in _REAL:
            fn = self._helper(INT_HELPER[("from_real", target)].removeprefix("hawk_"))
            return f"{fn}({arg})"
        to = self._ctype(TensorType((), target))
        if to == self._ctype(TensorType((), source)):
            return arg
        return f"(({to}){self._atom(r, arg, node.operands[0].ttype)})"

    def _reduction(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        """The rank>=1 -> rank-0 forms: a core fold, or a struct member."""
        src = node.operands[0].ttype
        if kind == "component":
            index = node.literal if isinstance(node.literal, tuple) else (node.literal,)
            x = self._named(r, args[0], src)
            if len(src.shape) == 1:
                return f"{x}.v[{int(index[0])}]"
            if len(index) == 2:
                return f"{x}.m[{int(index[0])}][{int(index[1])}]"
            row = int(index[0])
            cells = ", ".join(f"{x}.m[{row}][{c}]" for c in range(src.shape[1]))
            return f"{self._helper('make', (src.shape[1],))}({cells})"
        if kind == "sum":
            word = "vec" if len(src.shape) == 1 else "mat"
            return f"{self._core(f'{word}_sum', src.shape)}({args[0]})"
        if len(src.shape) != 1:
            raise HawkError(_no_spelling(kind, len(src.shape)))
        if kind == "dot":
            return f"{self._core('vec_dot', src.shape)}({args[0]}, {args[1]})"
        return f"{self._core('vec_norm', src.shape)}({args[0]})"

    def _runtime_component(self, r, node: Op, args: Sequence[str]) -> str:
        """``v[k]`` at a runtime index: ``hawk_{S}_at`` (clamped) or ``_at_raw``."""
        src = node.operands[0].ttype
        base = "at_raw" if self._status(node) == "proven" else "at"
        return f"{self._helper(base, src.shape)}({args[0]}, {args[1]})"

    def _runtime_component_store(self, r, node: Op, args: Sequence[str]) -> str:
        """``v[k] = e`` as a value: ``hawk_{S}_set`` (clamped) or ``_set_raw``."""
        t = node.ttype
        base = "set_raw" if self._status(node) == "proven" else "set"
        e = self._real(r, args[2], node.operands[2].ttype.dtype)
        return f"{self._helper(base, t.shape)}({args[0]}, {args[1]}, {e})"

    def _tensor_op(self, r, kind: str, args: Sequence[str], ranks: Sequence[int],
                   node: Op) -> str:
        t = node.ttype
        rank = len(t.shape)
        if kind == "component_at":
            return self._runtime_component(r, node, args)
        if kind == "set_component_at":
            return self._runtime_component_store(r, node, args)
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(r, kind, args, node)
        if rank == 0:
            raise HawkError(_no_spelling(kind, max(ranks)))
        word = "vec" if rank == 1 else "mat"
        srcs = [o.ttype.dtype for o in node.operands]
        if kind == "splat":
            return self._splat(r, args[0], srcs[0], t.shape)
        if kind == "vec":
            if not args:
                raise HawkError("vec: a vector literal needs at least one component")
            parts = ", ".join(self._real(r, a, s) for a, s in zip(args, srcs))
            return f"{self._helper('make', t.shape)}({parts})"
        if kind == "mul" and 0 in ranks:
            # s * v == v * s: one core scale (its signature is ``(a, s)``;
            # each component is ``s * a_k``, multiplication commutes).
            s, v = (0, 1) if ranks[0] == 0 else (1, 0)
            return (f"{self._core(f'{word}_scale', t.shape)}("
                    f"{args[v]}, {self._real(r, args[s], srcs[s])})")
        if kind == "div" and ranks[1] == 0:
            # the reciprocal once, then a scale (never a cwise_div by a splat).
            return (f"{self._core(f'{word}_div_s', t.shape)}("
                    f"{args[0]}, {self._real(r, args[1], srcs[1])})")
        if kind in _SPLAT_BINARY or kind in ("mul", "div"):
            core = {"mul": "mul", "div": "cwise_div"}.get(kind, _SPLAT_BINARY.get(kind))
            a, b = (self._splat(r, x, s, t.shape) if rk == 0 else x
                    for x, s, rk in zip(args, srcs, ranks))
            return f"{self._core(f'{word}_{core}', t.shape)}({a}, {b})"
        if kind in ("neg", "abs") or kind in _CWISE1:
            return f"{self._core(f'{word}_{kind}', t.shape)}({args[0]})"
        src = node.operands[0].ttype
        if kind == "mv":
            return f"{self._core('mat_mv', src.shape)}({args[0]}, {args[1]})"
        if kind == "mm":
            fn = self._core("mat_mm", src.shape, node.operands[1].ttype.shape)
            return f"{fn}({args[0]}, {args[1]})"
        if kind == "outer":
            fn = self._core("vec_outer", src.shape, node.operands[1].ttype.shape)
            return f"{fn}({args[0]}, {args[1]})"
        if kind == "transpose":
            return f"{self._core('mat_transpose', src.shape)}({args[0]})"
        if kind == "cross":
            return f"{self._core('vec_cross', src.shape)}({args[0]}, {args[1]})"
        if kind in _QUAT:
            return f"{self._core(_QUAT[kind])}({', '.join(args)})"
        raise HawkError(_no_spelling(kind, rank))

    def _splat(self, r, text: str, dtype: str, shape) -> str:
        """A rank-0 operand of a two-struct core function, broadcast."""
        return f"{self._helper('splat', shape)}({self._real(r, text, dtype)})"

    def tape_write(self, r, sink: TapeWrite, target: str) -> None:
        """One tape ROW on the own column, under ``if (on)``: component
        ``row*w + c`` of the sample's ``(W*w,)`` SoA plane gets the ``c``-th
        flattened component of the stored values, in order. The row is
        clamped to the plane's row count unless verify proved it."""
        rows = sink.ttype.shape[0] // sink.width
        row = self._clamped(sink, r._ref(sink.row), rows)
        lines = []
        offset = 0
        for value in sink.values:
            text = r._ref(value)
            shape = value.ttype.shape
            if not shape:
                comps = [self._real(r, text, value.ttype.dtype)]
            else:
                x = self._named(r, text, value.ttype)
                comps = [f"{x}.{cell}" for cell in _cells(shape)]
            for comp in comps:
                cell = (f"((hawk_tape_row * {_i64(sink.width)}) + {_i64(offset)})"
                        f" * n_samples")
                lines.append(f"{target}[({cell}) + {SAMPLE_INDEX_IDENT}] = {comp};")
                offset += 1
        on, on_dtype = r._ref(sink.on), sink.on.ttype.dtype
        cond = on if on_dtype == "bool" else f"{on} != {_int_zero_like(on_dtype)}"
        r._emit(f"if {_paren(cond)} {{")
        r.indent += "    "
        r._emit(f"const int64_t hawk_tape_row = {row};")
        for line in lines:
            r._emit(line)
        r.indent = r.indent[:-4]
        r._emit("}")

    def finish(self, r, sink: Finish) -> None:
        """The finish epilogue — the last text of the guarded scope: an
        unfinished sample whose condition holds sets its mask through the
        writable twin and adds one to the reserved counter."""
        mask = binding_name("terminated", sink.name)
        cond = r._ref(sink.value)
        counter = binding_name("lookup", FINISHED_PLANE)
        r._emit(f"if (({mask}[{SAMPLE_INDEX_IDENT}] == (uint8_t)0) && {cond}) {{")
        r._emit(f"    {finish_binding(sink.name)}[{SAMPLE_INDEX_IDENT}] = (uint8_t)1;")
        r._emit(f"    {counter}[0] = {counter}[0] + 1U;")
        r._emit("}")

    def typed_zero(self, ttype: TensorType) -> str:
        """The typed zero a ``switch`` ``default:`` writes when a branch was elided."""
        if ttype.shape:
            zero = real_literal(0.0, self.mode)
            return f"{self._helper('splat', ttype.shape)}({zero})"
        if ttype.dtype in _REAL:
            return real_literal(0.0, self.mode)
        return _int_zero_like(ttype.dtype)


def _paren(cond: str) -> str:
    """``cond`` as an ``if`` condition: itself when it is ONE parenthesised
    expression (the outer pair matches), else wrapped once."""
    if cond.startswith("(") and cond.endswith(")"):
        depth = 0
        for k, ch in enumerate(cond):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 0 and k < len(cond) - 1:
                break
        else:
            return cond
    return f"({cond})"


def _i64(value: int) -> str:
    return int_literal(value, "i64")


def _i32(value: int) -> str:
    return int_literal(value, "int32")


def _int_zero_like(dtype: str, value: int = 0) -> str:
    """An integer or bool literal of ``dtype`` (an index ``i32`` is 64-bit)."""
    return int_literal(value, "i64" if dtype == "i32" else dtype)


def _cells(shape) -> list[str]:
    """Row-major struct member accessors (``v[k]`` / ``m[r][c]``)."""
    if len(shape) == 1:
        return [f"v[{k}]" for k in range(shape[0])]
    return [f"m[{i}][{j}]" for i in range(shape[0]) for j in range(shape[1])]


def _no_spelling(kind: str, rank: int) -> str:
    return f"op {kind!r} has no C spelling at rank {rank}"

