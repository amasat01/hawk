# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether C++ dialect: today's spelling of every text the body renderer
produces, behind the :class:`hawk.emit.dialect.Dialect` seam.

Every method here is the text :class:`hawk.emit.aether._Renderer` emitted
before the seam existed, moved verbatim; the renderer keeps the traversal
and hands this dialect already-rendered operand text. ``r`` is the renderer
(``r._emit``, ``r._ref``, ``r.items``, ``r.hoists``, ``r.indent``,
``r.uses_random``). Rank is the dispatch axis: rank-0 spells through
``aether::math``, rank>=1 through ``Expression``'s methods and
``aether/expr`` operators, quaternions through aether's method spellings. An
op with no aether spelling at its rank raises :class:`~hawk.ir.HawkError`
naming the kind, the rank and what aether would need.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..ir import HawkError
from ..ir.nodes import (
    FINISHED_PLANE,
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
from .dispatch import _clamp_expr, render_dispatch_rank0
from .spelling import (
    _CWISE,
    _CWISE1,
    _INFIX,
    _MATH1,
    _MATH2,
    _MATH3,
    _METHOD0,
    _METHOD1,
    _RANDOM_FN,
    SAMPLE_INDEX_IDENT,
    _for_header,
    _no_spelling,
    _real_literal,
    binding_name,
    carry_type,
    element_spelling,
    finish_binding,
    param_cell_binding,
)


class AetherDialect:
    """The aether C++ spelling (see the module docstring)."""

    id = "aether"
    #: aether's lazy expressions inline until the renderer's fan-out rule
    #: names them; only leaf values (``Item``/``Constant``) are forced.
    names_every_tensor = False
    commits_per_component = True

    # -- declarations and loops ----------------------------------------

    def declare(self, name: str, ttype: TensorType, text: str) -> str:
        """A named local: ``const auto`` lets aether's expression type through."""
        return f"const auto {name} = {text};"

    def carry_type(self, ttype: TensorType) -> str:
        """:func:`hawk.emit.spelling.carry_type`."""
        return carry_type(ttype)

    def tape_decl(self, elem: str, name: str, trip: int) -> str:
        """A forward loop's per-iteration tape array."""
        return f"{elem} {name}[{trip}];"

    def counter_decl(self, counter: str) -> str:
        """A loop's per-sample iteration counter."""
        return f"Int {counter} = 0;"

    def for_header(self, r, index: str, loop, count: str | None) -> str:
        """:func:`hawk.emit.spelling._for_header`."""
        return _for_header(index, loop, count)

    def tail_break(self, flag: str) -> str:
        """The tail-form exit, tested after the carries moved."""
        return f"if ({flag}) break;"

    def exit_if(self, cond: str) -> str:
        """The ``if`` opening a head/middle-form ``break``, before the body's rest."""
        return f"if ({cond}) {{"

    def sample_index_value(self) -> str:
        """A ``SampleIndex`` node: the lane's own integer index as an ``Int``."""
        return f"static_cast<Int>({SAMPLE_INDEX_IDENT})"

    def hoist(self, r, text: str, ttype: TensorType | None = None) -> str:
        """Bind ``text`` to a fresh named local ahead of the current line — the
        lifetime-safe home for a value that is an aether LEAF but not a node of
        its own (a broadcast ``Constant``, a raw block's operand)."""
        name = f"hz{r.hoists}"
        r.hoists += 1
        r._emit(f"const auto {name} = {text};")
        return name

    def _hoist(self, r, text: str) -> str:
        return self.hoist(r, text, None)

    # -- dispatch -------------------------------------------------------

    def selector_decl(self, r, kname: str, kind_expr: str, k: int, *,
                      in_range: bool = False) -> str:
        """The selector, materialised once: clamped (:func:`_clamp_expr`), or a
        plain cast when it is a select between two in-range literals."""
        spelled = (f"static_cast<std::int32_t>({kind_expr})" if in_range
                   else _clamp_expr(kind_expr, k))
        return f"const std::int32_t {kname} = {spelled};"

    def mix_count(self, mix: str, k: int, kname: str) -> str:
        """An ``auto`` dispatch's branch-mix count (``hawk_abi::mix_count``)."""
        return f"hawk_abi::mix_count({mix}, {k}, {kname});"

    def predicated_rank0(self, kname: str, texts: Sequence[str]) -> str:
        """:func:`hawk.emit.dispatch.render_dispatch_rank0`'s ternary chain."""
        return render_dispatch_rank0(kname, texts, "predicated")

    def predicated_tensor(self, r, kname: str, texts: Sequence[str]) -> str:
        """Nested ``aether::select`` calls, each condition broadcast into a
        hoisted rank-0 ``Constant`` (the shape ``render_dispatch_rank0`` can't
        build: it has no scope to hoist into)."""
        expr = texts[-1]
        for j in range(len(texts) - 2, -1, -1):
            cond = self._hoist(r, f"aether::constant<aether::extents<>>"
                                  f"({kname} == {j})")
            expr = f"aether::select({cond}, {texts[j]}, {expr})"
        return expr

    def switch_default_tail(self) -> str:
        """Nothing follows a C++ ``default`` clause."""
        return ""

    # -- sinks and the guard --------------------------------------------

    def guard(self, bound: Sequence[str]) -> str | None:
        """The guard's CONDITION over the bound mask bindings: each read at
        the body's own sample, the commit skipped when any is set."""
        if not bound:
            return None
        if len(bound) == 1:
            return f"{bound[0]}[i].eval()"
        return "(" + " || ".join(f"{b}[i].eval()" for b in bound) + ")"

    def store(self, r, sink, target: str, value: str) -> None:
        """The own column: one elementwise contribution, one store."""
        r._emit(f"{target}[i] = {value};")

    def scatter_store(self, r, sink, target: str, index: str, value: str) -> None:
        """A scattered wide_write targets a computed ROW: a store, exactly as
        the reference evaluator reads it."""
        r._emit(f"{target}[{self._sample_index(index)}] = {value};")

    def accum_scattered(self, r, sink, target: str, index: str, value: str) -> None:
        """The scattered ``plain``/``atomic`` commit: the target's atomic add."""
        r._emit(f"hawk_abi::accum_add({target}, {self._sample_index(index)}, {value});")

    def store_compensated(self, r, sink, target: str, comp: str, value: str) -> None:
        """The ``compensated`` kind's commit: a Neumaier accumulate into the
        own column (``i`` is already the bound ``SampleIndex``), or Knuth
        two-sum under the target's atomic add when scattered."""
        if sink.index is None:
            r._emit(f"hawk_abi::store_compensated({target}, {comp}, i, {value});")
            return
        index = self._sample_index(r._ref(sink.index))
        r._emit(f"hawk_abi::accum_add_compensated({target}, {comp}, {index}, {value});")

    def component_commit(self, r, target: str, at: str, op: str, value: str) -> str:
        return f"{target}({at}, i.global()) {op} {value};"

    def in_place_accumulate(self, r, target: str, k: str, value: str) -> str:
        return f"{target}[i].at({k}) += {value};"

    def zero_row(self, r, target: str, width: int) -> str:
        return (f"for (Int hk = 0; hk < {width}; ++hk) "
                f"{target}[i].at(hk) = {self.typed_zero(TensorType((), 'f64'))};")

    def sample_sum(self, r, sink, column: int) -> None:
        """The sample's term into its column of the row the wrapper sums
        (:data:`hawk.emit.aether.SAMPLE_SUM_TERMS`)."""
        from .aether import SAMPLE_SUM_TERMS

        r._emit(f"{SAMPLE_SUM_TERMS}[{column}] = {r._ref(sink.value)};")

    # -- expressions ----------------------------------------------------

    def leaf(self, r, leaf: Leaf, materialised: bool) -> str:
        name = binding_name(leaf.role, leaf.name)
        if leaf.role == "nsamples":
            return f"static_cast<Int>({name})"
        if leaf.role == "uniform":
            if leaf.ttype.dtype in INT_PLANE_DTYPES:
                return f"static_cast<{element_spelling(leaf.ttype.dtype)}>({name})"
            return name
        if leaf.role == "mutable":
            # `.prior` — the only way a `mutable`-role LEAF (as opposed to its
            # SINK) reaches here. The plane's local is a WRITABLE view, so `[i]`
            # calls the non-const `View::operator[]` and returns a `SampleRef`,
            # which offers only `.get()` (no bare `.eval()` — that's
            # `ConstSampleRef`'s overload, reached only through a `const` view).
            # Rank>=1 stops at the materialised `Item`; rank 0 additionally
            # calls its zero-argument `operator()()` to recover the raw scalar
            # `aether::math::fmax`/`+`/... require.
            if leaf.ttype.shape:
                return f"{name}[i].get()"
            return f"{name}[i].get()()"
        if leaf.ttype.shape and r.position.get(id(leaf)) in r.lazy:
            return f"{name}[i]"     # proxy, element reads only (LAZY_LEAF_WIDTH)
        if leaf.ttype.shape:
            # Rank >= 1: always the body's own sample, never the whole plane.
            # `view[i].get()` materialises the row into an `Item`, which is what
            # makes the value sample-bound — a bare view carries no sample, and
            # every aether fold reached through one silently answers at
            # `SampleIndex::make(0)`. `_materialised` names every rank>=1 leaf,
            # so this branch has no unmaterialised arm to take.
            return f"{name}[i].get()"
        return f"{name}[i].eval()"

    def const(self, node: Const) -> str:
        t = node.ttype
        if t.shape:
            raise HawkError(
                f"a rank-{len(t.shape)} literal typed {t!r} has no aether spelling; "
                "a broadcast literal is a rank-0 constant under a 'splat'"
            )
        if t.dtype == "bool":
            return "true" if node.literal else "false"
        if t.dtype in ("i32", "i64"):
            return f"static_cast<Int>({int(node.literal)})"
        if t.dtype in INT_PLANE_DTYPES:
            return f"static_cast<{element_spelling(t.dtype)}>({int(node.literal)}LL)"
        return f"static_cast<Real>({_real_literal(float(node.literal))})"

    def at(self, r, node: At) -> str:
        if is_param_cell(node):
            return param_cell_binding(node.plane.name)
        plane = binding_name(node.plane.role, node.plane.name)
        idx = self._sample_index(r._ref(node.index))
        return f"{plane}[{idx}]" + (".get()" if node.ttype.shape else ".eval()")

    def select(self, r, node: Select, c: str, a: str, b: str) -> str:
        """The ``cond ? a : b``. A rank-0 select is a plain C++ ternary; a
        rank>=1 one is ``aether::select``, which reads both branches through
        the shared working-type helper — a raw ternary can't name them, since
        two aether expressions of different C++ types have no common type."""
        if not node.ttype.shape:
            return f"({c} ? {a} : {b})"
        cond_rank = len(node.operands[0].ttype.shape)
        if cond_rank == 0:
            # A per-sample mask: aether::select needs an EXPRESSION, and a
            # rank-0 C++ bool isn't one, so it broadcasts through a rank-0
            # Constant (Select's `Cond::element_extents::Rank == 0` arm).
            c = self._hoist(r, f"aether::constant<aether::extents<>>({c})")
        return f"aether::select({c}, {a}, {b})"

    def op(self, r, node: Op, args: Sequence[str], ranks: Sequence[int]) -> str:
        kind, rank = node.kind, len(node.ttype.shape)
        if kind == RAW_KIND:
            return self._raw(r, node, args)
        if rank == 0 and all(rk == 0 for rk in ranks):
            return self._scalar_op(r, kind, args, node)
        return self._tensor_op(r, kind, args, ranks, node)

    def _raw(self, r, node: Op, args: Sequence[str]) -> str:
        block = node.literal
        for k, arg in enumerate(args):
            r._emit(f"const auto hawk_raw_arg{k} = {arg};")
        return f"({block.text})"

    def _scalar_op(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        if kind in _INT_CALLS or (kind in _INT_WRAPPING
                                  and node.ttype.dtype in INT_PLANE_DTYPES):
            # integer planes (and `//`, `%`, the bit operators on an index):
            # the wrapping, numpy-defined helpers (hawk/emit/backend.py
            # INT_HELPERS), since C++'s own `+ - * / % << >>` are undefined
            # or differently defined at the edges numpy pins.
            name = _INT_CALLS.get(kind) or _INT_WRAPPING[kind]
            return f"hawk_abi::{name}({', '.join(args)})"
        if kind == "cast":
            return self._cast_op(args[0], node)
        if kind in RANDOM_OPS:
            return self._random_draw(r, kind, args)
        if kind == "spacing":
            return f"hawk_abi::spacing({args[0]})"
        if kind == "sign":
            # aether's sign maps NaN to 0; numpy keeps the NaN.
            x = args[0] if args[0].isidentifier() else self._hoist(r, args[0])
            return f"(aether::math::isnan({x}) ? {x} : aether::math::sign({x}))"
        if kind in _MATH1:
            return f"aether::math::{_MATH1[kind]}({args[0]})"
        if kind in _MATH2:
            return f"aether::math::{_MATH2[kind]}({args[0]}, {args[1]})"
        if kind in _MATH3:
            return f"aether::math::{_MATH3[kind]}({', '.join(args)})"
        if kind in _INFIX and len(args) == 2:
            return f"({args[0]} {_INFIX[kind]} {args[1]})"
        if kind == "neg":
            return f"(-{args[0]})"
        if kind == "lnot":
            return f"(!{args[0]})"
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(r, kind, args, node)
        raise HawkError(_no_spelling(kind, 0))

    @staticmethod
    def _cast_op(arg: str, node: Op) -> str:
        """``cast``: float -> integer plane through the saturating helper,
        every other pair a plain ``static_cast`` (modular between integers,
        exact into ``Real`` up to its precision)."""
        target, source = node.ttype.dtype, node.operands[0].ttype.dtype
        elem = element_spelling(target)
        if target in INT_PLANE_DTYPES and source in ("f64", "f32"):
            return f"hawk_abi::int_from_real<{elem}>({arg})"
        return f"static_cast<{elem}>({arg})"

    def _random_draw(self, r, kind: str, args: Sequence[str]) -> str:
        """HAWK's ``(seed, counter)`` draw: aether's stateless
        ``detail::uniform01``/``detail::standardNormal<Real>(seed, global,
        counter)`` (:data:`_RANDOM_FN`), with ``global`` the kernel's own
        flattened sample index spelled here as the bound :data:`SAMPLE_INDEX_IDENT`
        directly. Every lane is its own stream by construction (an author names
        only ``seed``/``counter``), which is also what lets a captured graph
        re-draw by writing one of those wires rather than re-capturing. Sets
        :attr:`uses_random` so :class:`Body` carries the fact
        :func:`hawk.emit.backend.prelude` needs to add the header."""
        r.uses_random = True
        seed, counter = args
        return (f"aether::random::detail::{_RANDOM_FN[kind]}<Real>("
                f"static_cast<std::uint64_t>({seed}), "
                f"static_cast<aether::offset_t>({SAMPLE_INDEX_IDENT}), "
                f"static_cast<std::uint64_t>({counter}))")

    def _reduction(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        """The rank>=1 -> rank-0 forms: they READ a tensor and RETURN a scalar."""
        if kind == "component":
            from .aether import _through

            index = node.literal if isinstance(node.literal, tuple) else (node.literal,)
            ks = ', '.join(str(int(k)) for k in index)
            if r.position.get(id(_through(node.operands[0]))) in r.lazy:
                return f"{args[0]}.eval<{ks}>()"
            return f"{args[0]}.eval<{ks}>(i)"
        if kind == "sum" and len(node.operands[0].ttype.shape) == 2:
            return self._matrix_sum(r, args[0], node.operands[0].ttype)
        if kind in _METHOD1:
            return f"{args[0]}.{_METHOD1[kind]}({args[1]})"
        return f"{args[0]}.{_METHOD0[kind]}()"

    def _runtime_component(self, r, node: Op, args: Sequence[str]) -> str:
        """``v[k]`` at a RUNTIME index, in aether's own spelling.

        aether answers this through ``Item::operator()``, which casts each
        index to ``std::size_t`` and folds them row-major over the static
        extents — no local array copy is needed. The catch: ``operator()`` is
        ``Item``'s own, not the plain ``Expression`` CRTP's, so the operand must
        first be MATERIALISED into an ``aether::Item`` local. That
        materialisation is named once per value per SCOPE (:attr:`items`), so
        two runtime reads of one vector inside a loop body share one Item, and
        an Item built inside a loop isn't referenced after the scope closes."""
        from .aether import _reads_in_place

        source = node.operands[0]
        if _reads_in_place(source):
            # a read-only input plane: the element at the runtime index straight
            # from the view (`SampleRef::at`), no `Item` of the whole row
            return f"{binding_name(source.role, source.name)}[i].at({args[1]})"
        held = r.items.get(id(source))
        if held is None:
            t = source.ttype
            extents = ", ".join(str(e) for e in t.shape)
            held = f"hawk_it{r.hoists}"
            r.hoists += 1
            r._emit(f"const aether::Item<{element_spelling(t.dtype)}, "
                       f"{extents}> {held} = {args[0]};")
            r.items[id(source)] = held
        return f"{held}({args[1]})"

    def _runtime_component_store(self, r, node: Op, args: Sequence[str],
                                 accumulate: bool = False) -> str:
        """``v[k] = e`` at a RUNTIME index, as a VALUE.

        The IR node is functional — ``v`` with one component replaced — and C++
        has no such expression, so this is the one place the emitter turns a
        value into two statements: a non-const ``aether::Item`` copy of the
        incoming vector, then a write through its non-const ``operator()``.
        The copy preserves the IR's functional meaning (a later read of ``v``
        sees the old value); it's register-resident, so a compiler that can
        see the original elides it. Emitting statements mid-expression is safe
        because :meth:`_expr` runs once per node in traversal order, so the
        lines land above their use."""
        t = node.ttype
        name = f"hawk_sv{r.hoists}"
        r.hoists += 1
        r._emit(f"{carry_type(t)} {name} = {args[0]};")
        op = "+=" if accumulate else "="
        r._emit(f"{name}({args[1]}) {op} {self._cast(args[2], t.dtype)};")
        return name

    def _matrix_sum(self, r, arg: str, ttype: TensorType) -> str:
        """The FULL reduction of a rank-2 expression, in aether's own spelling.

        ``hawk.ir.ops`` admits ``sum`` over a rank-2 operand, and the reverse
        rule mints one for the adjoint of a rank-0 operand broadcast against a
        matrix. aether's ``Expression::sum()`` asserts rank 1, so this uses
        ``Expression::matDot()`` instead — the Frobenius inner product with a
        matrix of ones gives the same total, one aether node, no scalarised
        per-element chain. The ones matrix is HOISTED like any other broadcast
        constant: ``aether::constant`` is a leaf and ``matDot`` binds it by
        reference."""
        ext = ", ".join(str(e) for e in ttype.shape)
        ones = self._hoist(r, f"aether::constant<aether::extents<{ext}>>("
                           f"{self._cast('1', ttype.dtype)})")
        return f"{arg}.matDot({ones})"

    def _tensor_op(self, r, kind: str, args: Sequence[str], ranks: Sequence[int],
                   node: Op) -> str:
        rank = len(node.ttype.shape)
        if kind == "component_at":
            return self._runtime_component(r, node, args)
        if kind == "set_component_at":
            return self._runtime_component_store(r, node, args)
        if kind == "add_at":
            return self._runtime_component_store(r, node, args, accumulate=True)
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(r, kind, args, node)
        if kind == "splat":
            ext = ", ".join(str(e) for e in node.ttype.shape)
            return (f"aether::constant<aether::extents<{ext}>>("
                    f"{self._cast(args[0], node.ttype.dtype)})")
        if kind == "vec":
            if not args:
                raise HawkError("vec: a vector literal needs at least one component")
            elem = element_spelling(node.ttype.dtype)
            # aether added the 1-component `Item<T,1>{v}` ctor beside the
            # N-scalar one, so the arity floor is 1, not 2.
            ext = ", ".join(str(e) for e in node.ttype.shape)
            return f"aether::Item<{elem}, {ext}>{{{', '.join(args)}}}"
        if kind in ("add", "sub"):
            return self._additive(r, kind, args, ranks, node)
        if kind == "mul":
            return self._multiplicative(r, args, ranks, "*", "cwiseMul", node)
        if kind == "div" and ranks[1] == 0:
            return self._multiplicative(r, args, ranks, "/", None, node)
        if kind in _CWISE:
            # min/max/div/pow: one spelling for both operand shapes —
            # `f(expr, bound)` when the right operand is rank-0, `f(expr, expr)`
            # otherwise. Only the LEFT operand must be an expression in every
            # overload, so a rank-0 left one broadcasts through a Constant first.
            left = args[0] if ranks[0] else self._broadcast(r, args[0], node.ttype)
            return f"{_CWISE[kind]}({left}, {args[1]})"
        if kind == "neg":
            return f"({args[0]} * static_cast<Real>(-1))"
        if kind == "abs":
            return f"aether::cwiseAbs({args[0]})"
        if kind in _CWISE1:
            return f"aether::{_CWISE1[kind]}({args[0]})"
        if kind in ("mv", "mm"):
            return f"({args[0]} * {args[1]})"
        if kind == "outer":
            return f"aether::outer({args[0]}, {args[1]})"
        if kind in _METHOD1:
            return f"{args[0]}.{_METHOD1[kind]}({args[1]})"
        if kind in _METHOD0:
            return f"{args[0]}.{_METHOD0[kind]}()"
        raise HawkError(_no_spelling(kind, rank))

    def _additive(self, r, kind: str, args: Sequence[str], ranks: Sequence[int],
                  node: Op) -> str:
        sym = "+" if kind == "add" else "-"
        left, right = args
        if ranks[0] == 0:
            left = self._broadcast(r, left, node.ttype)
        if ranks[1] == 0:
            right = self._broadcast(r, right, node.ttype)
        return f"({left} {sym} {right})"

    def _multiplicative(self, r, args: Sequence[str], ranks: Sequence[int], sym: str,
                        cwise: str | None, node: Op) -> str:
        a, b = args
        if ranks[0] == 0 and sym == "*":
            return f"({self._cast(a, node.ttype.dtype)} * {b})"
        if ranks[1] == 0:
            return f"({a} {sym} {self._cast(b, node.ttype.dtype)})"
        if cwise is not None and ranks[0] == ranks[1]:
            return f"{a}.{cwise}({b})"
        kind = "div" if sym == "/" else "mul"
        raise HawkError(
            _no_spelling(kind, len(node.ttype.shape))
            + " for two rank>=1 operands: aether carries a component-wise PRODUCT "
            "(Expression::cwiseMul) and no component-wise quotient"
        )

    def _broadcast(self, r, scalar: str, ttype: TensorType) -> str:
        """A rank-0 operand added to a tensor becomes an aether broadcast leaf —
        HOISTED, since a ``Constant`` is a leaf and would be stored by reference
        by the enclosing node."""
        ext = ", ".join(str(e) for e in ttype.shape)
        return self._hoist(r, f"aether::constant<aether::extents<{ext}>>("
                           f"{self._cast(scalar, ttype.dtype)})")

    @staticmethod
    def _cast(text: str, dtype: str) -> str:
        elem = element_spelling(dtype)
        if text.startswith(f"static_cast<{elem}>"):
            return text
        return f"static_cast<{elem}>({text})"

    def tape_write(self, r, sink: TapeWrite, target: str) -> None:
        """One tape ROW on the own column, under ``if (on)``: component
        ``row*w + c`` of the sample's ``(W*w,)`` plane gets the ``c``-th
        flattened component of the stored values, in order. Rendered where the
        sink stands — a twin puts it FIRST, so it reads the pre-step state
        before any commit overwrites it."""
        row = r._ref(sink.row)
        lines = []
        offset = 0
        for value in sink.values:
            text = r._ref(value)
            shape = value.ttype.shape
            comps = ([((), text)] if not shape
                     else [(idx, f"{text}.eval<{', '.join(map(str, idx))}>(i)")
                           for idx in _flat_indices(shape)])
            for _idx, comp in comps:
                lines.append(f"{target}(static_cast<std::size_t>(hawk_tape_row * "
                             f"{sink.width} + {offset}), i.global()) = {comp};")
                offset += 1
        r._emit(f"if ({r._ref(sink.on)} != 0) {{")
        r.indent += "    "
        r._emit(f"const std::int64_t hawk_tape_row = {row};")
        for line in lines:
            r._emit(line)
        r.indent = r.indent[:-4]
        r._emit("}")

    @staticmethod
    def _sample_index(expr: str) -> str:
        return f"aether::SampleIndex::make(static_cast<std::size_t>({expr}))"

    def finish(self, r, sink: Finish) -> None:
        """The finish epilogue — the last text of the guarded scope, so the step's
        own commits land before the sample is marked: an unfinished sample whose
        condition holds sets its mask (via :func:`finish_binding`) and adds one to
        the reserved counter (``hawk_abi::finish_count``)."""
        mask = binding_name("terminated", sink.name)
        cond = r._ref(sink.value)
        counter = binding_name("lookup", FINISHED_PLANE)
        r._emit(f"if (!{mask}[i].eval() && {cond}) {{ "
                   f"{finish_binding(sink.name)}[i] = true; "
                   f"hawk_abi::finish_count({counter}); }}")

    def typed_zero(self, ttype: TensorType) -> str:
        """The typed zero the ``switch`` ``default:`` writes when at
        least one branch was elided — the same spelling :meth:`_broadcast`
        builds, inlined rather than hoisted, since it's assigned straight
        into the declared local, which copies it out immediately."""
        elem = element_spelling(ttype.dtype)
        if not ttype.shape:
            return f"static_cast<{elem}>(0)"
        ext = ", ".join(str(e) for e in ttype.shape)
        return f"aether::constant<aether::extents<{ext}>>(static_cast<{elem}>(0))"


#: Integer-only kinds -> their ``hawk_abi`` helper (any integer type).
_INT_CALLS = {"floordiv": "int_floordiv", "mod": "int_mod", "band": "int_and",
              "bor": "int_or", "bxor": "int_xor", "shl": "int_shl", "shr": "int_shr",
              "invert": "int_invert"}
#: Kinds an integer PLANE value routes through a wrapping helper, where the
#: infix spelling would overflow a signed int (undefined) or promote.
_INT_WRAPPING = {"add": "int_add", "sub": "int_sub", "mul": "int_mul",
                 "neg": "int_neg", "abs": "int_abs", "min": "int_min",
                 "max": "int_max"}


def _flat_indices(shape: tuple) -> list:
    """Every index tuple of ``shape`` in row-major (C) order."""
    out: list = [()]
    for extent in shape:
        out = [idx + (k,) for idx in out for k in range(int(extent))]
    return out


#: The built aether dialect: what every backend wraps today.
AETHER = AetherDialect()
