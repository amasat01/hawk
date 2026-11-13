# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether C++ writer: today's spelling of every text the body renderer
produces, behind the :class:`aether.wings.base.writer.Writer` seam.

Every method here is the text :class:`aether.wings.base.render._Renderer` emitted
before the seam existed, moved verbatim; the renderer keeps the traversal
and hands this writer already-rendered operand text. ``r`` is the renderer
(``r._emit``, ``r._ref``, ``r.items``, ``r.hoists``, ``r.indent``,
``r.uses_random``). Rank is the dispatch axis: rank-0 spells through
``aether::math``, rank>=1 through ``Expression``'s methods and
``aether/expr`` operators, quaternions through aether's method spellings. An
op with no aether spelling at its rank raises :class:`~hawk.ir.RaptorError`
naming the kind, the rank and what aether would need.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from fractions import Fraction
from math import isfinite, prod

from raptor.errors import RaptorError
from raptor.ir.nodes import Loop, TapeRead
from raptor.ir.nodes import (
    _through,
    FINISHED_PLANE,
    GATHER_ROLES,
    RAW_KIND,
    RAW_LANGUAGE,
    SWITCH_RENDERED,
    At,
    Dispatch,
    Const,
    Finish,
    Leaf,
    Node,
    Op,
    ReduceSamples,
    Select,
    TapeWrite,
    is_param_cell,
)
from raptor.ir.linalg import CONSUMERS, FACTORISATIONS, LINALG_OPS, QR_OPS, op_bytes
from lowering import kinds as _lane, wide as _wide
from raptor.ir.rays import RAY_OPS
from raptor.ir.ops import RANDOM_OPS
from raptor.ir.types import INT_PLANE_DTYPES, TensorType
from . import rtstore
from raptor.ir.fold import exceeds, rolled
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
    _flat_indices,
    _for_header,
    _no_spelling,
    _real_literal,
    binding_name,
    carry_type,
    element_spelling,
    finish_binding,
    param_cell_binding,
)

#: aether's per-distribution salts (`UniformLeaf::salt`, `NormalLeaf::salt` in
#: `aether/random/detail/DistributionLeaf.h`): the seed handed to the backend
#: is `seed ^ salt`, so each kind keys its own stream.
_RANDOM_SALT = {"random_uniform": "0x243F6A8885A308D3ULL", "random_normal": "0x13198A2E03707344ULL"}


#: The namespace of aether's run commits (``aether/accum/run.h``) and the
#: prefix of the entry-local run state each target's calls share.
RUN_CALL = "aether::accum::run::"
RUN_STATE = "hawk_run_"

#: The dtypes a host select lowers to a bitwise ``hawk_abi::blend`` (a ``?:`` on a float is a branch the vectoriser refuses).
_BLEND_DTYPES = ("f64", "f32")


#: Op kinds that index a value at a run-time position (out of range is a fault) or reduce across lanes.
_GUARDED_KINDS = ("component_at", "set_component_at", "add_at")


def _evaluates_freely(node: Select) -> bool:
    """Whether both arms of ``node`` may run whatever the condition says. A blend evaluates both arms where ``?:`` evaluates
    one, so an arm that can fault or must not run stays a ``?:``: a plane read at a computed index (``At``, a team-stage read:
    a stencil's edge guard), a tape read or write, a raw block, a run-time component index, a ``switch``, a ``Loop``
    (its trips) or a collective over the lanes."""
    stack, seen = list(node.operands[1:]), set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, (At, TapeRead, TapeWrite, Loop, ReduceSamples)):
            return False
        if isinstance(n, Op) and (n.kind == RAW_KIND or n.kind in _GUARDED_KINDS or n.kind == _lane.WARP_SUM_KIND):
            return False
        if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
            return False
        stack.extend(n.operands)
    return True


#: The tensor ops that read a placed value's proxy element by element (``.at``), not as an expression.
_READS_PROXY = frozenset({"component", "component_at", "set_component_at", "add_at"})


class AetherWriter:
    """The aether C++ spelling (see the module docstring).

    ``cells`` is the form a ``cells`` scalar mode takes (``banded``): a Real
    literal is the three-limb register value ``Band(h, l, t)``, a store into a
    plane packs (``Real(value)``), and a named rank-0 Real local is bound as a
    ``Band``, normalised once where it is named."""

    def __init__(self, cells: bool = False) -> None:
        self.cells = cells

    id = "aether"
    #: sites over the fold threshold spell as ``aether::rt`` twins
    #: (:mod:`raptor.ir.fold`), which the renderer prepares for through
    #: :meth:`unrendered` and :meth:`fresh_value`
    rolled_forms = True
    #: aether's lazy expressions inline until the renderer's fan-out rule
    #: names them; only leaf values (``Item``/``Constant``) are forced.
    names_every_tensor = False
    #: stores at a runtime index are written in place (:mod:`aether.wings.base.rtstore`)
    in_place_stores = True
    commits_per_component = True

    # -- declarations and loops ----------------------------------------

    def declare(self, name: str, ttype: TensorType, text: str, ordered: bool = True, held: bool = False) -> str:
        """A named local: ``const auto`` lets aether's expression type through. Under ``cells`` a rank-0 Real is a
        ``Band`` when a reader needs its limbs in order, else the raw chain flows on unnormalised.

        A vector ``held`` by more than one reader is an ``aether::Item``, evaluated once: as a lazy expression
        every reader would nest and re-evaluate it, so the text (and the compile) would grow with each reuse."""
        if held and not self.cells and ttype is not None and len(ttype.shape) == 1:
            return f"const aether::Item<{element_spelling(ttype.dtype)}, {ttype.shape[0]}> {name} = {text};"
        return f"const {self._local_type(ttype) if ordered else 'auto'} {name} = {text};"

    @staticmethod
    def reads_raw(kind: str) -> bool:
        """Does the op ``kind`` take an unnormalised chain operand: arithmetic and the facade doors that normalise
        at entry. The limb-order readers (``abs``, ``min``/``max``, ``copysign``, compares, ``select``) do not."""
        return kind in _RAW_READERS

    def _local_type(self, ttype: TensorType | None) -> str:
        """``auto``, or under ``cells`` a rank-0 Real's ``Band``: a later op that reads the limbs in order (``abs``,
        a compare, a division, a root, a transcendental, a store) needs a normalised value, and naming the local
        normalises it once, here (a ``Band`` already normalised copies)."""
        if self.cells and ttype is not None and not ttype.shape and ttype.dtype in ("f64", "f32"):
            return "aether::banded::Band"
        return "auto"

    def carry_type(self, ttype: TensorType) -> str:
        """:func:`aether.wings.base.spelling.carry_type`; under ``cells`` a rank-0 Real is carried as a ``Band``, so it
        stays normalised across iterations and is encoded to a cell once, where it is stored."""
        banded = self._local_type(ttype)
        return carry_type(ttype) if banded == "auto" else banded

    def tape_decl(self, elem: str, name: str, trip: int) -> str:
        """A forward loop's per-iteration tape array."""
        return f"{elem} {name}[{trip}];"

    def counter_decl(self, counter: str) -> str:
        """A loop's per-sample iteration counter."""
        return f"Int {counter} = 0;"

    def for_header(self, r, index: str, loop, count: str | None) -> str:
        """:func:`aether.wings.base.spelling._for_header`."""
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

    def plane_extent_value(self, node) -> str:
        """A ``PlaneExtent`` node: the element count of the plane's view."""
        return f"static_cast<Int>({binding_name(node.role, node.name)}.samples())"

    def lane_position_value(self, mapped: bool) -> str:
        """A ``LanePosition`` node: ``hawk_t`` under an index map, else the
        lane's own index (the launch position is the sample index there)."""
        return f"static_cast<Int>({'hawk_t' if mapped else SAMPLE_INDEX_IDENT})"

    def hoist(self, r, text: str, ttype: TensorType | None = None) -> str:
        """Bind ``text`` to a fresh named local ahead of the current line — the
        lifetime-safe home for a value that is an aether LEAF but not a node of
        its own (a broadcast ``Constant``, a raw block's operand)."""
        name = f"hz{r.hoists}"
        r.hoists += 1
        r._emit(f"const {self._local_type(ttype)} {name} = {text};")
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
        """:func:`aether.wings.base.dispatch.render_dispatch_rank0`'s ternary chain."""
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
        """The own column: one elementwise contribution, one store. A fresh ``Item`` of
        more than the variant's ``rolled_fold_min`` elements is copied by a loop
        (``aether::rt::store``): the unrolled assignment holds a register per element."""
        if value == f"{target}[i]":
            return                  # a library result already written into the plane
        if value in r.proxies:      # a value that lives in another plane: copied element by element
            r._emit(f"aether::rt::store({target}[i], {value});")
            return
        if value in r.fresh_items and exceeds(prod(sink.value.ttype.shape), r.rolled_min):
            r._emit(f"aether::rt::store({target}[i], {value});")
        else:
            r._emit(f"{target}[i] = {self._packed(sink.value.ttype, value)};")

    def _packed(self, ttype: TensorType | None, value: str) -> str:
        """``value`` as a plane stores it: under ``cells`` a rank-0 Real packs into its cell (``Real(value)``)."""
        if self.cells and (ttype is None or (not ttype.shape and ttype.dtype in ("f64", "f32"))):
            return f"Real({value})"
        return value

    def scatter_store(self, r, sink, target: str, index: str, value: str) -> None:
        """A scattered wide_write targets a computed ROW: a store, exactly as
        the reference evaluator reads it."""
        r._emit(f"{target}[{self._sample_index(index)}] = {self._packed(sink.value.ttype, value)};")

    def accum_scattered(self, r, sink, target: str, index: str, value: str) -> None:
        """The scattered ``plain``/``atomic`` commit: the target's atomic add."""
        r._emit(f"hawk_abi::accum_add({target}, {self._sample_index(index)}, {value});")

    def or_scattered(self, r, sink, target: str, index: str, value: str) -> None:
        """The scattered ``policy="or"`` commit: the target's atomic OR."""
        r._emit(f"hawk_abi::accum_or({target}, {self._sample_index(index)}, {value});")

    def store_compensated(self, r, sink, target: str, comp: str, value: str) -> None:
        """The ``compensated`` kind's commit: a Neumaier accumulate into the
        own column (``i`` is already the bound ``SampleIndex``), or Knuth
        two-sum under the target's atomic add when scattered."""
        if sink.index is None:
            r._emit(f"hawk_abi::store_compensated({target}, {comp}, i, {value});")
            return
        index = self._sample_index(r._ref(sink.index))
        r._emit(f"hawk_abi::accum_add_compensated({target}, {comp}, {index}, {value});")

    def accum_run(self, r, sink, target: str, comp: str | None, index: str, value: str,
                  run) -> None:
        """A scattered commit whose key moves in runs of lanes
        (:func:`simt.lanes.key_run`): ``aether::accum::run``'s call for the
        run's form, compensated when ``comp`` names the companion. ``i`` is
        the lane's own sample (the alignment vote reads it); the run state
        ``hawk_run_<target>`` is declared and closed by the entry
        (:func:`aether.wings.base.wrapper.run_states`)."""
        c = "Compensated" if comp else ""
        name = {"proven": f"add{c}<{run.width}>", "tested": f"addTested{c}",
                "sorted": f"addSorted{c}"}[run.form]
        full = []
        if comp and run.block and run.form == "proven" and r._scope_depth == 0 and r.spec.settings.block_fold:
            name, full = "addCompensatedBlock", [f"{RUN_CALL}blockFull(base, count)"]   # one commit per block
        args = [target, *([comp] if comp else []), self._sample_index(index), value,
                *(["i"] if run.form == "proven" else []), *full, f"{RUN_STATE}{target}"]
        r._emit(f"{RUN_CALL}{name}({', '.join(args)});")

    def component_commit(self, r, target: str, at: str, op: str, value: str) -> str:
        return f"{target}({at}, i.global()) {op} {value};"

    def in_place_accumulate(self, r, target: str, k: str, value: str) -> str:
        return f"{target}[i].at({k}) += {value};"

    def zero_row(self, r, target: str, width: int) -> str:
        return (f"for (Int hk = 0; hk < {width}; ++hk) "
                f"{target}[i].at(hk) = {self.typed_zero(TensorType((), 'f64'))};")

    def sample_sum(self, r, sink, column: int) -> None:
        """The sample's term into its column of the row the wrapper sums
        (:data:`aether.wings.base.render.SAMPLE_SUM_TERMS`)."""
        from .render import SAMPLE_SUM_TERMS

        if self.cells:
            term = f"aether::banded::Band({r._ref(sink.value)})"
            r._emit(f"{SAMPLE_SUM_TERMS}[{column}].addTerm({term});" if sink.op == "sum" else f"{SAMPLE_SUM_TERMS}[{column}] = {term};")
            return
        r._emit(f"{SAMPLE_SUM_TERMS}[{column}] = {r._ref(sink.value)};")

    # -- expressions ----------------------------------------------------

    def leaf(self, r, leaf: Leaf, materialised: bool) -> str:
        name = binding_name(leaf.role, leaf.name)
        if _wide.is_scratch(leaf):
            # the plane a wide value lives in (:mod:`simt.wide`): its text is the lane's own proxy of it
            if not r.placed:
                return ""
            if _wide.is_tiled(leaf) and r.lane_rows is not None:
                # an interchanged host lane: the plane is a row of the tile, declared by the text around the segments
                proxy = f"{name}_i"
                r.lane_rows[proxy] = (name, r.plane_widths[leaf.name])
            elif _wide.is_tiled(leaf):
                proxy = f"{name}_i"
                r._emit(f"const auto {proxy} = hawk_abi::tiled<Real, {r.plane_widths[leaf.name]}>({name}, "
                        f"{SAMPLE_INDEX_IDENT});")
            else:
                proxy = f"{name}[i]"
            r.proxies.add(proxy)
            return proxy
        if leaf.role == "nsamples":
            return f"static_cast<Int>({name})"
        if leaf.role == "uniform":
            if leaf.ttype.dtype in INT_PLANE_DTYPES:
                return f"static_cast<{element_spelling(leaf.ttype.dtype)}>({name})"
            return name
        if leaf.ttype.shape and r.position.get(id(leaf)) in r.lazy:
            return f"{name}[i]"     # proxy, element reads only (LAZY_LEAF_WIDTH, or a mutable plane read in place)
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
            raise RaptorError(
                f"a rank-{len(t.shape)} literal typed {t!r} has no aether spelling; "
                "a broadcast literal is a rank-0 constant under a 'splat'"
            )
        if t.dtype == "bool":
            return "true" if node.literal else "false"
        if t.dtype in ("i32", "i64"):
            return f"static_cast<Int>({int(node.literal)})"
        if t.dtype in INT_PLANE_DTYPES:
            return f"static_cast<{element_spelling(t.dtype)}>({int(node.literal)}LL)"
        if self.cells:
            return band_literal(float(node.literal))
        return f"static_cast<Real>({_real_literal(float(node.literal))})"

    def at(self, r, node: At) -> str:
        if is_param_cell(node):
            return param_cell_binding(node.plane.name)
        plane = binding_name(node.plane.role, node.plane.name)
        idx = self._sample_index(r._ref(node.index))
        if r.position.get(id(node)) in r.lazy:
            return f"{plane}[{idx}]"        # proxy, handed to a library call as it stands
        return f"{plane}[{idx}]" + (".get()" if node.ttype.shape else ".eval()")

    def select(self, r, node: Select, c: str, a: str, b: str) -> str:
        """The ``cond ? a : b``. A rank-0 select is a plain C++ ternary; a
        rank>=1 one is ``aether::select``, which reads both branches through
        the shared working-type helper — a raw ternary can't name them, since
        two aether expressions of different C++ types have no common type."""
        host = r.spec.settings.blend and _evaluates_freely(node)
        if not node.ttype.shape:
            if host and node.ttype.dtype in _BLEND_DTYPES:
                return f"hawk_abi::blend({c}, {a}, {b})"
            return f"({c} ? {a} : {b})"
        cond_rank = len(node.operands[0].ttype.shape)
        if cond_rank == 0 and host:
            ext = ", ".join(str(e) for e in node.ttype.shape)
            # An Item is a leaf, held by const& in whatever expression takes it: held once in a named local, or it dangles.
            return self._hoist(r, f"hawk_abi::select_item<{element_spelling(node.ttype.dtype)}, {ext}>({c}, {a}, {b})")
        if cond_rank == 0:
            # A per-sample mask: aether::select needs an EXPRESSION, and a
            # rank-0 C++ bool isn't one, so it broadcasts through a rank-0
            # Constant (Select's `Cond::element_extents::Rank == 0` arm).
            c = self._hoist(r, f"aether::constant<aether::extents<>>({c})")
        return f"aether::select({c}, {a}, {b})"

    def op(self, r, node: Op, args: Sequence[str], ranks: Sequence[int]) -> str:
        kind, rank = node.kind, len(node.ttype.shape)
        if kind == _wide.RESIDENT_KIND:
            return self._resident(r, node, args)
        if kind == _lane.WARP_SUM_KIND:
            partial = f"{args[0]}.sum()" if node.operands[0].ttype.shape else args[0]
            return f"hawk_abi::warp_sum<Real>(static_cast<Real>({partial}))"
        if kind == RAW_KIND:
            return self._raw(r, node, args)
        if kind in LINALG_OPS:
            return self._linalg(r, node, args)
        if kind in RAY_OPS:
            return self._rays(r, node, args)
        if rank == 0 and all(rk == 0 for rk in ranks):
            return self._scalar_op(r, kind, args, node)
        return self._tensor_op(r, kind, args, ranks, node)

    def _rays(self, r, node: Op, args: Sequence[str]) -> str:
        """A ray primitive (``aether::geometry``): the library returns its four values
        by value, held once, and the node is the ``(4,)`` item of them. The vector
        operands go in as per-sample operands the library reads by element; the
        rank-0 ones (``r``, ``tmax``) as they are."""
        elem = element_spelling(node.ttype.dtype)
        shapes = [len(o.ttype.shape) for o in node.operands]
        spelled = [self._operand(r, o, a) if rank else a
                   for o, a, rank in zip(node.operands, args, shapes)]
        held = self._hoist(r, f"aether::geometry::{_RAY[node.kind]}<{elem}>({', '.join(spelled)})")
        return f"aether::Item<{elem}, 4>{{{', '.join(f'{held}.v[{k}]' for k in range(4))}}}"

    def _linalg(self, r, node: Op, args: Sequence[str]) -> str:
        """A factorisation, a solve against one, or a read of a field of one,
        through ``aether::linalg`` (``aether/linalg/``).

        A factor is the library's struct, a named local the call fills in place
        (a struct of ``n x n`` returned by value is copied, and the copies are
        what spill). Matrices go in and
        out as per-sample operands: a plane leaf as its ``[i]`` proxy, so the
        library reads the plane element by element and the lane holds only the
        factor and one column; any other value as a named ``Item``. A solve or
        a matrix field is written into a fresh ``Item`` by statements ahead of
        the line that uses it, as the runtime-index stores do; ``info``,
        ``det`` and the factor itself are plain expressions."""
        kind = node.kind
        n = node.operands[0].ttype.shape[0]
        if r.max_dim is None:
            raise RaptorError(f"op {kind!r}: the variant of the target being built has no linalg.max_dim: a backend whose manifest names no variant cannot factor per lane")
        if n > r.max_dim or kind in QR_OPS:
            raise RaptorError(
                f"op {kind!r} at n={n} reached the per-lane writer, which holds the matrix in one "
                f"lane (n <= {r.max_dim}): above that a team of threads works on it, which is a stage "
                "of its own the build splits out (a team stage inside a loop body, a thread box or "
                "the embedded c target has none)")
        if kind in FACTORISATIONS:
            name = f"hawk_lf{r.hoists}"
            r.hoists += 1
            vectors = any(m.kind == "eigh_v" and _through(m.operands[0]) is node for m in r.nodes if isinstance(m, Op))
            fn, struct = _ENTRY[kind, ("v",) if vectors else ()]
            r._emit(f"aether::linalg::{struct}<{element_spelling(node.ttype.dtype)}, {n}> {name};")
            self._linalg_call(r, fn, (n,), name,
                              self._operand(r, node.operands[0], args[0]), emit=True)
            return name
        if kind.endswith("_info"):
            return f"{args[0]}.info"
        if kind == "lu_det":
            return self._linalg_call(r, "luDet", (), args[0])
        out = r.direct.get(r.position.get(id(node))) or self._fresh_item(r, node.ttype)
        if kind in CONSUMERS:
            rhs = node.operands[1]
            k = rhs.ttype.shape[1] if len(rhs.ttype.shape) == 2 else 1
            if kind == "lu_solve" and _is_identity(rhs):
                self._linalg_call(r, "luInverse", (), args[0], out, emit=True)
                return out
            held = self._operand(r, rhs, args[1])
            if kind in _TRIANGULAR:
                self._linalg_call(r, _TRIANGULAR[kind], (n, k), f"{args[0]}.l", held, out, emit=True)
            else:
                self._linalg_call(r, _SOLVE[kind], (k,), args[0], held, out, emit=True)
        elif kind == "chol_factor":
            self._linalg_call(r, "choleskyFactor", (), args[0], out, emit=True)
        else:
            field = "w" if kind == "eigh_w" else "v"
            count = 1
            for extent in node.ttype.shape:
                count *= extent
            slot = f"{out}.data()[hawk_lk]" if out in r.fresh_items else (
                f"{out}.at(hawk_lk)" if len(node.ttype.shape) == 1
                else f"{out}.at(hawk_lk / {node.ttype.shape[1]}, hawk_lk % {node.ttype.shape[1]})")
            r._emit(f"for (Int hawk_lk = 0; hawk_lk < {count}; ++hawk_lk) "
                    f"{slot} = {args[0]}.{field}[hawk_lk];")
        return out

    @staticmethod
    def _linalg_call(r, fn: str, targs: Sequence[int], *args: str, emit: bool = False) -> str:
        """The one place a call into ``aether::linalg`` is spelled: ``fn<targs>(args)``,
        as a statement when ``emit`` (the call writes its last operand), else
        as the expression it is."""
        spelled = f"aether::linalg::{fn}" + (f"<{', '.join(map(str, targs))}>" if targs else "")
        call = f"{spelled}({', '.join(args)})"
        if emit:
            r._emit(call + ";")
        return call

    def _operand(self, r, source: Node, arg: str) -> str:
        """``source`` as a per-sample operand a library twin reads by element
        (``aether::rt`` and ``aether::linalg``): a read-only plane is its proxy
        (the leaf's own local when it is bound as one, else ``plane[i]``), a
        value that is already a named ``Item`` itself, anything else an ``Item``
        held for the scope (:meth:`_held_item`)."""
        source = _through(source)
        pos = r.position.get(id(source))
        if arg in r.proxies:
            return arg          # a value that lives in a plane: its proxy
        if _is_plane(source):
            return arg if pos in r.lazy else f"{binding_name(source.role, source.name)}[i]"
        if isinstance(source, At) and pos in r.lazy:
            return arg          # a plane read at another lane: its proxy
        if arg in r.fresh_items or (isinstance(source, Op) and source.kind == "vec"):
            return arg          # already a named Item: a library result, an assembled matrix
        if isinstance(source, Leaf) and source.ttype.shape and source.role != "mutable" \
                and r._materialised(source, pos):
            return arg
        return self._held_item(r, source, arg)

    def _product_operand(self, r, source: Node) -> str:
        """An operand of a rolled product: a transpose is read through its operand
        (``aether::rt::transposed``), so the transposed value is never built."""
        source = _through(source)
        if isinstance(source, Op) and source.kind == "transpose":
            return f"aether::rt::transposed({self._product_operand(r, source.operands[0])})"
        return self._operand(r, source, r._ref(source))

    def _assemble(self, r, node: Op, args: Sequence[str]) -> str:
        """A matrix built from entries, past the threshold: a fresh ``Item`` its entries
        are stored into one statement each. The brace initialiser of the same entries
        is one expression whose every entry is live at once."""
        out = self._fresh_item(r, node.ttype)
        cols = node.ttype.shape[1]
        for k, arg in enumerate(args):
            r._emit(f"{out}({k // cols}, {k % cols}) = {arg};")
        return out

    def _rolled_product(self, r, node: Op) -> str:
        """A product over the fold threshold (:func:`raptor.ir.fold.rolled`) as its
        ``aether::rt`` twin, written into a fresh ``Item`` ahead of the line that uses it.
        The static text multiplies register ``Item``s, so the twin is the one that
        contracts as they do (``matVecItems``)."""
        out = self._fresh_item(r, node.ttype)
        left, right = (self._product_operand(r, o) for o in node.operands)
        r._emit(f"aether::rt::{'matVecItems' if node.kind == 'mv' else 'matMat'}({out}, {left}, {right});")
        return out

    def unrendered(self, r) -> frozenset:
        """The positions no statement renders: a value every one of whose users reads
        it through its own operand (an identity right-hand side ``luInverse``
        replaces, a transpose a rolled product reads through), and, in the device text, the
        elementwise values a placement writes into its plane element by element."""
        reads: dict = {}
        for node in r.nodes:
            if not isinstance(node, Op) or r.stores.skips(node):
                continue
            for k, child in enumerate(node.operands):
                child = _through(child)
                if _absorbs(node, k, child, r.rolled_min, r.spec.settings.form):
                    p = r.position[id(child)]
                    reads[p] = reads.get(p, 0) + 1
        out = {p for p, n in reads.items() if n == r.fanout.get(p, 0)}
        if r.placed:
            written = self._written(r)
            r.absorbed = frozenset(written)
            out |= {r.position[i] for i in written if i in r.position}
        return frozenset(out)

    @staticmethod
    def _written(r) -> set:
        """The ids of the elementwise values (and vectors of parts), in the body or inside a loop, that a placement writes
        element by element: its value, and the values only those read, down to a leaf or another placement."""
        from .render import _flat_body

        scanned = list(r.nodes)
        for node in r.nodes:
            if isinstance(node, Loop):
                scanned.extend(_flat_body(node))
        def key(n):             # equal values share one position: a use of either is a use of both
            return r.position.get(id(n), id(n))

        users: dict = {}
        for node in scanned:
            if isinstance(node, (Op, Select, Loop)):        # a loop reads its initial values too
                for o in node.operands:
                    users.setdefault(key(_through(o)), []).append(node)
        seen: set = set()
        reads, keep = [], set()
        stack = [_through(n.operands[0]) for n in scanned if _wide.is_resident(n)]
        while stack:
            node = stack.pop()
            if id(node) in seen or _wide.element_kind(node) not in ("parts", "splat", "elementwise"):
                continue
            if not all((_wide.is_resident(u) and key(_through(u.operands[0])) == key(node))
                       or (id(u) in seen and _wide.element_kind(u) == "elementwise") for u in users.get(key(node), ())):
                continue
            seen.add(id(node))
            run = _wide.gather(node)
            if run is not None:                 # the reads are one expression of the element index: none is rendered
                reads += [part for part in node.operands if len(users.get(key(part), ())) == 1]
                keep.add(id(run.base))
            stack.extend(_through(o) for o in node.operands if len(o.ttype.shape) == 1)
        seen.update(id(part) for part in reads)
        for part in reads:                      # nor are the constant indices only those reads made
            stack = [part.operands[1]]
            while stack:
                node = stack.pop()
                if (id(node) in seen or id(node) in keep or not isinstance(node, (Op, Const))
                        or not all(id(u) in seen for u in users.get(id(node), ()))):
                    continue
                seen.add(id(node))
                stack.extend(node.operands)
        return seen

    # -- wide values in planes (:mod:`simt.wide`) -----------------------------------------

    def _resident(self, r, node: Op, args: Sequence[str]) -> str:
        """A value placed in its plane (``args[1]`` is the lane's proxy of it): the statements that write the
        value there, then the proxy as the text of the value. The host text has no planes: the value itself."""
        if not r.placed:
            return args[0]
        value, proxy = _through(node.operands[0]), args[1]
        if id(value) in r.absorbed:
            self._fill(r, proxy, value)
        elif args[0] != proxy and _wide.is_tiled(node.operands[1]):       # a scratch lane is written element by element
            r._emit("AETHER_NO_UNROLL()")
            held = self._element_of(r, value, "hawk_lk", None) if _wide.element_kind(value) in ("input", "elementwise", "parts", "splat") \
                else f"{args[0]}(hawk_lk)"
            r._emit(f"for (Int hawk_lk = 0; hawk_lk < {node.ttype.shape[0]}; ++hawk_lk) {proxy}.at(hawk_lk) = {held};")
        elif args[0] != proxy:
            r._emit(f"{proxy} = {args[0]};")
        return proxy

    def _fill(self, r, proxy: str, value: Node) -> None:
        """Write ``value``, an elementwise value or a vector of parts, into ``proxy`` one element at a time: a rolled
        loop over the elements when each is the same expression of its index, else one statement per element."""
        width = value.ttype.shape[0]
        flat = self._rolled_elements(value)
        if flat:
            r._emit("AETHER_NO_UNROLL()")
            r._emit(f"for (Int hawk_lk = 0; hawk_lk < {width}; ++hawk_lk) "
                    f"{proxy}.at(hawk_lk) = {self._cast(self._element_of(r, value, 'hawk_lk', None), value.ttype.dtype)};")
            return
        for k in range(width):
            r._emit(f"{proxy}.at({k}) = {self._cast(self._element_of(r, value, str(k), k), value.ttype.dtype)};")

    def _rolled_elements(self, node: Node) -> bool:
        """Whether every element of ``node`` is one expression of the element index: no vector of distinct parts is under it."""
        stack, seen = [node], set()
        while stack:
            node = _through(stack.pop())
            if id(node) in seen or len(node.ttype.shape) != 1:
                continue
            seen.add(id(node))
            kind = _wide.element_kind(node)
            if kind == "parts":
                first = node.operands[0]
                if _wide.gather(node) is None and not all(
                        isinstance(o, Const) and isinstance(first, Const) and o.literal == first.literal for o in node.operands):
                    return False
            elif kind == "elementwise":
                stack.extend(node.operands)
        return True

    def _element_of(self, r, node: Node, k: str, static: int | None) -> str:
        """The text of element ``k`` of ``node`` (``static`` its value when it is a compile-time index): a plane read
        at it, a part, or the operation on the elements of its operands."""
        work, vals = [(node, False)], []             # an explicit stack: a chain of any depth costs one frame
        while work:
            node, expanded = work.pop()
            node = _through(node)
            text = None if expanded else self._element_text(r, node, k, static)
            if text is not None:
                vals.append(text)
            elif not expanded:
                work.append((node, True))
                work.extend((o, False) for o in reversed(node.operands))
            else:
                args = vals[len(vals) - len(node.operands):]
                del vals[len(vals) - len(node.operands):]
                vals.append(self._operation(node.kind, args))
        return vals[0]

    def _element_text(self, r, node: Node, k: str, static: int | None) -> str | None:
        """The text of element ``k`` of ``node`` where it is not an operation on its operands' elements, else ``None``."""
        if len(node.ttype.shape) == 0:
            return r._ref(node)
        if _wide.is_prior(node):                         # the Mutable's own plane, read at its element before it is written
            return f"{binding_name(node.role, node.name)}[i].at({k})"
        if id(node) not in r.absorbed:
            text = r._ref(node)
            if text in r.proxies or r.position.get(id(node)) in r.lazy:
                return f"{text}.at({k})"
        kind = _wide.element_kind(node)
        if kind == "parts":
            run = _wide.gather(node)
            if run is not None:
                return self._gathered(r, run, k, static)
            if static is not None:
                return r._ref(node.operands[static])
            return r._ref(node.operands[0])         # every part the same literal
        if kind == "splat":
            return r._ref(node.operands[0])
        if kind == "elementwise":
            return None
        if kind == "input":                        # a read-only vector plane of the kernel: its element, straight from the view
            return f"{binding_name(node.role, node.name)}[i].at({k})"
        raise RaptorError(f"a value of kind {getattr(node, 'kind', type(node).__name__)!r} has no element-by-element spelling")

    def _gathered(self, r, run, k: str, static: int | None) -> str:
        """Element ``k`` of a run of reads of one plane (:class:`simt.wide.Gather`)."""
        at = f"static_cast<Int>({run.start + run.step * static})" if static is not None else (
            f"(static_cast<Int>({run.start}) + {k} * static_cast<Int>({run.step}))")
        index = at if run.base is None else f"({r._ref(run.base)} + {at})"
        plane = binding_name(run.plane.role, run.plane.name)
        if run.clamped:     # each read of the run is held in the plane, as the reads it replaces were
            index = f"hawk_abi::int_clamp_index<Int>({index}, static_cast<Int>({plane}.samples()))"
        return f"{plane}[{self._sample_index(index)}].eval()"

    @staticmethod
    def _operation(kind: str, a: Sequence[str]) -> str:
        """One element of an elementwise ``kind`` over the element texts ``a``."""
        if kind in ("add", "sub", "mul", "div"):
            return f"({a[0]} {_INFIX[kind]} {a[1]})"
        if kind == "neg":
            return f"(-{a[0]})"
        if kind == "select":
            return f"({a[0]} ? {a[1]} : {a[2]})"
        if kind in ("min", "max"):
            return f"aether::math::{_MATH2[kind]}({a[0]}, {a[1]})"
        return f"aether::math::{_MATH1[kind]}({a[0]})"

    @staticmethod
    def direct_kind(node: Op) -> bool:
        """Whether ``node`` is a library result that can be written into the plane it is stored to."""
        return node.kind in CONSUMERS or node.kind in ("chol_factor", "eigh_w", "eigh_v")

    @staticmethod
    def own_shape(node: Op) -> tuple:
        return node.ttype.shape

    def fresh_value(self, r, node: Node) -> bool:
        """Whether ``node`` renders as the name of a fresh local its statements
        fill (a factor, a library result, a rolled product, a team result read back): it is never
        copied."""
        if getattr(node, "fresh_read", False):
            return True
        return isinstance(node, Op) and bool(node.ttype.shape) and (
            node.kind in LINALG_OPS or rolled(node, r.rolled_min, r.spec.settings.form) or _assembled(node, r.rolled_min))

    @staticmethod
    def _fresh_item(r, ttype: TensorType) -> str:
        """A declared, unset ``aether::Item`` of ``ttype`` for statements to fill."""
        name = f"hawk_la{r.hoists}"
        r.hoists += 1
        r.fresh_items.add(name)
        extents = ", ".join(str(e) for e in ttype.shape)
        r._emit(f"aether::Item<{element_spelling(ttype.dtype)}, {extents}> {name};")
        return name

    def _raw(self, r, node: Op, args: Sequence[str]) -> str:
        block = node.literal
        if block.language != RAW_LANGUAGE:
            raise RaptorError(f"a raw block written in {block.language!r} cannot be "
                            f"spliced into aether text; write it in {RAW_LANGUAGE!r}")
        decls = [f"const {self._local_type(source.ttype)} hawk_raw_arg{k} = {arg};"
                 for k, (arg, source) in enumerate(zip(args, node.operands))]
        seen = getattr(r, "raw_blocks_with_operands", 0)
        if not decls:
            return f"({block.text})"
        r.raw_blocks_with_operands = seen + 1
        if seen == 0:
            # the first block with operands declares the names the block's text reads, in the kernel's own scope
            for d in decls:
                r._emit(d)
            return f"({block.text})"
        # a later block reads the same names: its own scope, so two blocks never redefine them
        return f"[&] {{ {' '.join(decls)} return ({block.text}); }}()"

    def _scalar_op(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        if kind in _INT_CALLS or (kind in _INT_WRAPPING
                                  and node.ttype.dtype in INT_PLANE_DTYPES):
            # integer planes (and `//`, `%`, the bit operators on an index):
            # the wrapping, numpy-defined helpers (staging/aether/wings/base/wrapper.py
            # INT_HELPERS), since C++'s own `+ - * / % << >>` are undefined
            # or differently defined at the edges numpy pins.
            name = _INT_CALLS.get(kind) or _INT_WRAPPING[kind]
            return f"hawk_abi::{name}({', '.join(args)})"
        if kind in ("min", "max") and node.ttype.dtype in ("i32", "i64"):
            # an index: integer throughout, never through the floating-point fmin/fmax
            return f"hawk_abi::int_{kind}<Int>({', '.join(args)})"
        if kind == "clamp_index":
            clamp = f"hawk_abi::int_clamp_index<Int>({', '.join(args)})"
            span = getattr(r, "clamp_spans", {}).get(id(node))
            if span is None:
                return clamp
            base, lo, hi = span         # one range test for every read of the group: in range, the read as it is
            b = f"static_cast<Int>({r._ref(base)})"
            held = clamp
            return (f"(__builtin_expect(({b} >= static_cast<Int>({-lo})) & ({b} < {args[1]} - static_cast<Int>({hi})), 1) "
                    f"? {args[0]} : {held})")
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
        if self.cells and kind in ("sin", "cos"):
            shared = self._sincos(r, kind, args[0], node)
            if shared is not None:
                return shared
        if self.cells and kind in _RAW_TWIN and r.readers.get(id(node)) and r.readers[id(node)] <= _TWIN_READERS:
            return f"{_DETAIL}{_RAW_TWIN[kind]}(aether::banded::Band({args[0]}))"
        if kind in _MATH1:
            return f"aether::math::{_MATH1[kind]}({args[0]})"
        if kind in _MATH2:
            return f"aether::math::{_MATH2[kind]}({args[0]}, {args[1]})"
        if kind in _MATH3:
            return f"aether::math::{_MATH3[kind]}({', '.join(args)})"
        if kind in _INFIX and len(args) == 2:
            if self.cells and kind in _LITERAL_KINDS:
                folded = _banded_literal_op(kind, args, node)
                if folded is not None:
                    return folded
                if kind == "add":
                    fused = self._fused_add(r, node)
                    if fused is not None:
                        return fused
            return f"({args[0]} {_INFIX[kind]} {args[1]})"
        if kind == "neg":
            return f"(-{args[0]})"
        if kind == "lnot":
            return f"(!{args[0]})"
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(r, kind, args, node)
        raise RaptorError(_no_spelling(kind, 0))

    def _fused_add(self, r, node: Op) -> str | None:
        """``a*b + c`` (either order) as the fused raw intake ``fma(a, b, c)`` when ``c`` is a normalised ``Band``
        (a plane element or the result of a door), the product is read by this add alone and neither factor is a
        literal (those take their own raw forms). ``None``: left to the raw chain, which accepts a raw addend."""
        for j in (0, 1):
            prod_node, addend = (_through(o) for o in (node.operands[j], node.operands[1 - j]))
            pos = r.position.get(id(prod_node))
            if (not (isinstance(prod_node, Op) and prod_node.kind == "mul" and len(prod_node.operands) == 2)
                    or r.fanout.get(pos, 0) != 1 or r._materialised(prod_node, pos)
                    or any(_real_const(o) is not None for o in prod_node.operands)
                    or not (isinstance(addend, (Leaf, At)) or (isinstance(addend, Op) and addend.kind in _DOORS))
                    or addend.ttype.shape or addend.ttype.dtype not in ("f64", "f32")):
                continue
            a, b = (r._ref(o) for o in prod_node.operands)
            return f"{_DETAIL}fma({a}, {b}, aether::banded::Band({r._ref(addend)}))"
        return None

    def _sincos(self, r, kind: str, arg: str, node: Op) -> str | None:
        """Under ``cells``, the ``sin`` and ``cos`` of ONE operand node, both at the top scope, share one reduction:
        the first reached binds ``sincos`` once, the other reads its result. ``None``: not such a pair."""
        operand = node.operands[0]
        partner = "cos" if kind == "sin" else "sin"
        if r._scope_depth or not any(isinstance(o, Op) and o.kind == partner and o.operands[0] is operand
                                     and r.position.get(id(o)) in r._outside for o in r.nodes):
            return None
        if r.position.get(id(node)) not in r._outside:
            return None
        names = r.sincos.get(id(operand))
        if names is None:
            k = len(r.sincos)
            names = r.sincos[id(operand)] = (f"hawk_sin{k}", f"hawk_cos{k}")
            r._emit(f"aether::banded::Band {names[0]}, {names[1]};")
            r._emit(f"aether::math::sincos(aether::banded::Band({arg}), &{names[0]}, &{names[1]});")
        return names[0] if kind == "sin" else names[1]

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
        counter)`` (:data:`_RANDOM_FN`) under the seed XOR-ed with the kind's own
        salt (:data:`_RANDOM_SALT`, aether's per-leaf constants), so a uniform
        and a normal drawn on the same ``(seed, counter)`` are unrelated
        streams. ``global`` is the kernel's own flattened sample index, spelled
        here as the bound :data:`SAMPLE_INDEX_IDENT` directly. Every lane is its own stream by construction (an author names
        only ``seed``/``counter``), which is also what lets a captured graph
        re-draw by writing one of those wires rather than re-capturing. Sets
        :attr:`uses_random` so :class:`Body` carries the fact
        :func:`aether.wings.base.wrapper.prelude` needs to add the header."""
        r.uses_random = True
        seed, counter = args
        return (f"aether::random::detail::{_RANDOM_FN[kind]}<Real>("
                f"static_cast<std::uint64_t>({seed}) ^ {_RANDOM_SALT[kind]}, "
                f"static_cast<aether::offset_t>({SAMPLE_INDEX_IDENT}), "
                f"static_cast<std::uint64_t>({counter}))")

    def _reduction(self, r, kind: str, args: Sequence[str], node: Op) -> str:
        """The rank>=1 -> rank-0 forms: they READ a tensor and RETURN a scalar."""
        if kind != "component" and rolled(node, r.rolled_min, r.spec.settings.form):
            operands = ", ".join(self._operand(r, o, a) for o, a in zip(node.operands, args))
            return f"aether::rt::{kind}({operands})"
        if kind == "component":
            index = node.literal if isinstance(node.literal, tuple) else (node.literal,)
            ks = ', '.join(str(int(k)) for k in index)
            if args[0] in r.proxies:
                return f"{args[0]}.at({ks})"
            source = _through(node.operands[0])
            if r.position.get(id(source)) in r.lazy:
                return f"{args[0]}.at({ks})" if source.role == "mutable" else f"{args[0]}.eval<{ks}>()"
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
        from .render import _reads_in_place

        source = node.operands[0]
        if r.stores.skips(node):
            return ""               # a store of the same slot adds in place
        if args[0] in r.proxies:
            return f"{args[0]}.at({args[1]})"       # a value in a plane: the element, straight from it
        if _reads_in_place(source, r.inplace) or node.literal == _lane.WARP_READ:
            # a read-only input plane: the element at the runtime index straight
            # from the view (`SampleRef::at`), no `Item` of the whole row
            return f"{binding_name(source.role, source.name)}[i].at({args[1]})"
        held = self._held_item(r, source, args[0])
        return f"{held}({args[1]})"

    @staticmethod
    def _held_item(r, source: Node, arg: str) -> str:
        """The name of ``source``'s value materialised once per scope as a
        const ``aether::Item`` (``arg`` is its expression)."""
        held = r.items.get(id(source))
        if held is None and arg in r.item_names and not r.writer.cells:
            return arg                  # already a named Item: read it in place
        if held is None:
            t = source.ttype
            extents = ", ".join(str(e) for e in t.shape)
            held = f"hawk_it{r.hoists}"
            r.hoists += 1
            r._emit(f"const aether::Item<{element_spelling(t.dtype)}, "
                       f"{extents}> {held} = {arg};")
            r.items[id(source)] = held
        return held

    def _runtime_component_store(self, r, node: Op, args: Sequence[str]) -> str:
        """``v[k] = e`` at a RUNTIME index, as a VALUE.

        The IR node is functional — ``v`` with one component replaced — and C++
        has no such expression, so this is the one place the emitter turns a
        value into statements: :func:`aether.wings.base.rtstore.emit` writes the whole
        run of stores on one vector into one mutable local, at the run's last
        store, and the stores before it render nothing. Emitting statements
        mid-expression is safe because :meth:`_expr` runs once per node in
        traversal order, so the lines land above their use."""
        return rtstore.emit(self, r, node, args)

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
        if kind not in _READS_PROXY and not rolled(node, r.rolled_min, r.spec.settings.form):
            args = [f"{a}.expr()" if a in r.proxies else a for a in args]     # a placed value is an aether expression to the whole-value ops
        if kind == "component_at":
            return self._runtime_component(r, node, args)
        if kind == "set_component_at":
            return self._runtime_component_store(r, node, args)
        if kind == "add_at":
            return self._runtime_component_store(r, node, args)
        if kind in ("dot", "norm", "sum", "component"):
            return self._reduction(r, kind, args, node)
        if kind == "splat":
            ext = ", ".join(str(e) for e in node.ttype.shape)
            return (f"aether::constant<aether::extents<{ext}>>("
                    f"{self._cast(args[0], node.ttype.dtype)})")
        if kind == "vec" and _assembled(node, r.rolled_min):
            return self._assemble(r, node, args)
        if kind == "vec":
            if not args:
                raise RaptorError("vec: a vector literal needs at least one component")
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
            if rolled(node, r.rolled_min, r.spec.settings.form):
                return self._rolled_product(r, node)
            return f"({args[0]} * {args[1]})"
        if kind == "outer":
            return f"aether::outer({args[0]}, {args[1]})"
        if kind in _METHOD1:
            return f"{args[0]}.{_METHOD1[kind]}({args[1]})"
        if kind in _METHOD0:
            return f"{args[0]}.{_METHOD0[kind]}()"
        raise RaptorError(_no_spelling(kind, rank))

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
        raise RaptorError(
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
            return band_literal(0.0) if self.cells and ttype.dtype in ("f64", "f32") else f"static_cast<{elem}>(0)"
        ext = ", ".join(str(e) for e in ttype.shape)
        return f"aether::constant<aether::extents<{ext}>>(static_cast<{elem}>(0))"


def _assembled(node: Node, rolled_min: int) -> bool:
    """A matrix built from more entries than the static threshold (:mod:`raptor.ir.fold`)."""
    return (isinstance(node, Op) and node.kind == "vec" and len(node.ttype.shape) == 2
            and exceeds(prod(node.ttype.shape), rolled_min))


def _is_identity(node: Node) -> bool:
    """Whether ``node`` is the literal identity matrix (``hawk.trace.linalg.eye``)."""
    node = _through(node)
    return (isinstance(node, Op) and node.kind == "vec" and len(node.ttype.shape) == 2
            and node.ttype.shape[0] == node.ttype.shape[1]
            and all(isinstance(c, Const) and c.literal == (1.0 if k // node.ttype.shape[0] == k % node.ttype.shape[0] else 0.0)
                    for k, c in enumerate(node.operands)))


def _is_plane(node: Node) -> bool:
    """A read-only rank>=1 plane leaf: ``plane[i]`` reads it element by element, and
    nothing in the kernel writes it."""
    return (isinstance(node, Leaf) and bool(node.ttype.shape)
            and node.role not in ("mutable", "uniform", "nsamples") and node.role not in GATHER_ROLES
            and not is_param_cell(node))


def _absorbs(node: Op, k: int, child: Node, rolled_min: int, form: str = "auto") -> bool:
    """Whether ``node`` reads operand ``k`` through ``child``'s own operand instead of
    its text: an identity right-hand side of ``lu_solve``, a transpose under a rolled product."""
    if node.kind == "lu_solve":
        return k == 1 and _is_identity(child)
    return rolled(node, rolled_min, form) and isinstance(child, Op) and child.kind == "transpose"


#: The ``aether::geometry`` function each ray kind calls.
_RAY = {"ray_tri": "rayTriangle", "ray_aabb": "rayAabb", "ray_sphere": "raySphere"}

#: The ``aether::linalg`` (function, struct) of each factorisation kind by the outputs read beyond the base (eigh: ``"v"``, the
#: eigenvectors; without them V is never accumulated), and the function of each solve.
_ENTRY = {("lu", ()): ("lu", "Lu"), ("chol", ()): ("cholesky", "Cholesky"), ("chol_from", ()): ("choleskyFrom", "Cholesky"),
          ("eigh", ()): ("eighValues", "EighValues"), ("eigh", ("v",)): ("eigh", "Eigh")}
_SOLVE = {"lu_solve": "luSolve", "lu_solve_t": "luSolveT", "cho_solve": "choSolve"}
#: The solves against the factor's ``L`` alone (they read the struct's ``l``).
_TRIANGULAR = {"tri_lower": "triLower", "tri_upper_t": "triUpperT"}

#: Integer-only kinds -> their ``hawk_abi`` helper (any integer type).
_INT_CALLS = {"floordiv": "int_floordiv", "mod": "int_mod", "band": "int_and",
              "bor": "int_or", "bxor": "int_xor", "shl": "int_shl", "shr": "int_shr",
              "invert": "int_invert"}
#: Kinds an integer PLANE value routes through a wrapping helper, where the
#: infix spelling would overflow a signed int (undefined) or promote.
_INT_WRAPPING = {"add": "int_add", "sub": "int_sub", "mul": "int_mul",
                 "neg": "int_neg", "abs": "int_abs", "min": "int_min",
                 "max": "int_max"}


#: Op kinds whose banded result is a normalised ``Band`` (a facade door or a division).
_DOORS = frozenset(_MATH1) | frozenset(_MATH2) | frozenset(_MATH3) | {"div"}
#: The certified ops that have a raw-exit twin in ``aether::banded::detail`` (no exit normalise), and the readers
#: that take the twin: an operator of the chain. Any other reader gets the facade entry, which normalises.
_RAW_TWIN = {"sin": "sinRaw", "cos": "cosRaw", "exp": "expRaw"}
_TWIN_READERS = frozenset(("add", "sub", "mul", "neg", "div"))
_RAW_READERS = frozenset(("add", "sub", "mul", "div", "neg", "exp", "log", "sin", "cos", "sqrt"))
_LITERAL_KINDS = ("add", "sub", "mul", "div")
_DETAIL = "aether::banded::detail::"


def _real_const(operand) -> Fraction | None:
    """The exact value of a finite rank-0 Real literal operand, else ``None``."""
    if isinstance(operand, Const) and not operand.ttype.shape and operand.ttype.dtype in ("f64", "f32"):
        value = float(operand.literal)
        return Fraction(value) if isfinite(value) else None
    return None


def _float_exact(value: Fraction) -> float | None:
    """The float32 equal to ``value`` exactly, else ``None``."""
    try:
        f = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return None
    return f if Fraction(f) == value else None


def _is_pow2(value: Fraction) -> bool:
    n, d = abs(value).numerator, abs(value).denominator
    return n > 0 and (n & (n - 1)) == 0 and (d & (d - 1)) == 0


def _banded_literal_op(kind: str, args: Sequence[str], node: Op) -> str | None:
    """``x <op> c`` (either side for ``+``/``*``) with a literal ``c``, spelled by its exact class: the identities
    (``x*1``, ``x+0``, ``x-0``, ``x/1``) are ``x``, ``x*-1`` is ``-x``, a power of two scales exactly
    (``scalePow2f``), a float-exact ``c`` takes the one-limb ``mulRawS``/``addRawS``, and ``x / c`` is ``x`` times
    the three-limb reciprocal of ``c`` split from the exact quotient. ``None``: not a literal case, spelled as an
    operator on a three-limb ``Band``."""
    lc, rc = (_real_const(o) for o in node.operands)
    if (lc is None) == (rc is None):
        return None                      # no literal, or two: nothing to classify
    left_lit = lc is not None
    c = lc if left_lit else rc
    x = args[1] if left_lit else args[0]
    if kind in ("sub", "div") and left_lit:
        return None
    if kind in ("add", "sub"):
        if c == 0:
            return x
        f = _float_exact(-c if kind == "sub" else c)
        return f"{_DETAIL}addRawS({x}, {_real_literal(f)}f)" if f is not None else None
    if kind == "div":
        if c == 0:
            return None
        c = 1 / c
    if c == 1:
        return x
    if c == -1:
        return f"(-{x})"
    f = _float_exact(c)
    if f is not None:
        helper = "scalePow2f" if _is_pow2(c) else "mulRawS"
        return f"{_DETAIL}{helper}({x}, {_real_literal(f)}f)"
    if kind == "div":
        limbs = _split3(c)
        if limbs is not None:
            return f"({x} * aether::banded::Band({', '.join(_real_literal(h) + 'f' for h in limbs)}))"
    return None


def _split3(value: Fraction):
    """``value`` as three floats, each the float nearest the rest; ``None`` out of the FP32 range."""
    limbs, rest = [], value
    try:
        for _ in range(3):
            limbs.append(struct.unpack("<f", struct.pack("<f", float(rest)))[0])
            rest -= Fraction(limbs[-1])
    except OverflowError:
        return None
    if abs(limbs[0]) < 2.0 ** -94 or abs(limbs[0]) >= 2.0 ** 126:
        return None
    return limbs


def band_literal(value: float) -> str:
    """``value`` as the three-limb ``Band(h, l, t)``: ``h`` the float nearest it, ``l`` the float nearest the rest,
    ``t`` the rest after that, from the exact value (:class:`~fractions.Fraction`). Exact for every double in the
    banded range; a literal the three limbs cannot hold exactly is refused by name."""
    if not isfinite(value):
        return f"aether::banded::Band({_real_literal(value)}, 0.0f, 0.0f)"
    if value == 0:
        return f"aether::banded::Band({_real_literal(value)}f, 0.0f, 0.0f)"     # keeps the sign of -0.0
    limbs, rest = [], Fraction(value)
    try:
        for _ in range(3):
            limbs.append(struct.unpack("<f", struct.pack("<f", float(rest)))[0])
            rest -= Fraction(limbs[-1])
    except OverflowError:
        rest = 1
    if rest != 0:
        raise RaptorError(f"the literal {value!r} is outside the banded range: a banded value lives in the FP32 "
                        "exponent envelope [2^-94, 2^126); scale the quantity, or build in float64")
    return "aether::banded::Band(" + ", ".join(_real_literal(h) + "f" for h in limbs) + ")"


#: The built aether writer: what every wrapper wraps today.
AETHER = AetherWriter()
#: Its form under a ``cells`` scalar mode (``banded``).
AETHER_CELLS = AetherWriter(cells=True)


def working_set_bytes(sinks, rolled_fold_min: int, form: str = "auto") -> int:
    """Bytes of per-lane storage the emitted text declares for its library results: every
    linear-algebra node at :func:`hawk.ir.linalg.op_bytes`, and each product or built matrix
    past the fold threshold ``rolled_fold_min`` as the ``Item`` it is written into.
    Values the kernel may keep in registers are not counted. A figure to read a kernel's local
    memory against: a kernel whose frame is much larger is copying a matrix it need not."""
    from raptor.ir.walk import canonical_nodes

    width = {"f64": 8, "f32": 4}
    total = 0
    for node in canonical_nodes(tuple(sinks))[0]:
        if not isinstance(node, Op):
            continue
        if node.kind in LINALG_OPS and node.ttype.shape:
            total += op_bytes(node)
        elif node.ttype.shape and (rolled(node, rolled_fold_min, form) or (
                node.kind == "vec" and len(node.ttype.shape) == 2
                and exceeds(prod(node.ttype.shape), rolled_fold_min))):
            total += prod(node.ttype.shape) * width.get(node.ttype.dtype, 8)
    return total
