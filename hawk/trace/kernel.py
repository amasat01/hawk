# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@kernel`` — the decorator that traces a Python function body into IR.

Tracing is one pass: resolve each parameter's declaration
(:mod:`hawk.trace.decl`), mint its leaf or write-side ref, run the body's
source through the ``ast`` pass (:mod:`hawk.trace.astpass`), then EXECUTE it
so the DAG builds through :class:`~hawk.trace.value.Value`'s operator
overloads. The sinks a body commits, in PARAMETER order, are handed to
``canonical()``, whose products (``arg_spec``, ``slot_of``, ``access``,
``digest``) are the only thing any consumer reads. A ``Quantity`` parameter
binds through :meth:`~hawk.ir.compound.Quantity.read_with`, binding every
wire of that quantity at once.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from collections.abc import Callable
from typing import Any

from ..ir import Const, HawkError, Leaf, Node, Quantity, Walk, canonical, make
from ..ir.nodes import Assign
from ..types import Slot, TensorType
from . import astpass
from .decl import Plane, resolve, role_of
from .value import (
    AccumRef,
    FinishRefusal,
    MutableRef,
    ReduceRef,
    TableRef,
    TerminatedRef,
    Value,
    WideOutRef,
)

#: How an input plane's ROLE picks the leaf KIND it mints.
_LEAF_KIND = {"per_sample": "vocab_read", "vec_in": "vocab_read",
              "mat_in": "vocab_read", "uniform": "uniform",
              "terminated": "terminated"}
_SINK_FORMS = ("mutable", "accum", "reduce", "wide_out")

#: per-process AST-transform memo: the ``ast`` pass is a PURE function of
#: the source and the declared sink/mutable sets, so retracing the SAME
#: definition twice is pure waste. Keyed on ``id(fn.__code__)``, with a
#: stored check (``co_code``/``co_firstlineno``/filename/sink-mutable sets)
#: guarding against a LATER, unrelated code object reusing that id once the
#: first is collected. A check miss is treated like an absent entry.
_TRANSFORM_MEMO: dict[int, tuple] = {}
_TRANSFORM_MEMO_LIMIT = 512


class Kernel:
    """A traced kernel: its sink set, its canonical walk and its declarations."""

    __slots__ = ("name", "sinks", "walk", "planes", "kind", "step", "one_step")

    def __init__(self, name: str, sinks: tuple[Node, ...],
                 planes: dict[str, Any] | None = None, *, kind: Any = None,
                 step: Kernel | None = None,
                 one_step: Kernel | None = None) -> None:
        from ..ext import DEFAULT_KIND

        self.name = name
        self.sinks = sinks
        self.planes = {} if planes is None else planes
        self.kind = kind if kind is not None else DEFAULT_KIND
        self.walk: Walk = canonical(sinks, _declared(self.planes))
        #: The single step a default automatic kernel was derived from (see
        #: :func:`kernel`); ``None`` otherwise, used by ``vjp``/``jvp``.
        self.step = step
        #: The single step an automatic kernel runs when ``fused_steps`` is
        #: ``1`` (entry branches once per launch between this body and the
        #: fused loop); ``None`` otherwise. Emission only.
        self.one_step = one_step

    @property
    def arg_spec(self) -> tuple[tuple[str, str], ...]:
        """The walk's ``(role, name)`` slot order — every consumer's source."""
        return self.walk.arg_spec

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise HawkError(
            f"kernel {self.name!r} cannot be launched yet: HAWK's execution path is "
            "the C++ host entry and the device artifact eagle deploys "
            " — HAWK delivers tracing and autodiff only"
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Kernel {self.name} slots={len(self.arg_spec)}>"


def kernel(fn: Callable[..., None] | None = None, *, kind: Any = None,
           steps: Any = None) -> Any:
    """Trace ``fn`` into a :class:`Kernel`. Refusals fire here.

    ``kind`` supplies the kernel's :class:`hawk.ext.Kind` (vocabulary,
    output set, seams); omitted, it gets :data:`hawk.ext.DEFAULT_KIND`.

    ``steps`` is how many steps one launch advances a kernel that finishes
    its own sample (``terminated = cond``)::

        @hawk.kernel(steps=16)
        def oscillator_step(...): ...

    is ``hawk.steps(<kernel>, 16)``, named ``oscillator_step_x16`` with state
    kept in registers across the steps. ``steps="auto"`` makes ONE kernel
    whose launch reads the step count from the reserved ``fused_steps`` word
    (at most ``hawk.trace.steps.AUTO_K_MAX``); an invalid ``steps`` refuses
    at decoration.

    Without ``steps=``, a kernel :func:`hawk.steps` admits for ``"auto"`` is
    built as the automatic kernel instead, paced by :func:`eagle.until_done`.
    ``steps=1`` is the explicit opt-out: the plain kernel, always."""
    if fn is None:
        def decorate(f: Callable[..., None]) -> Any:
            return kernel(f, kind=kind, steps=steps)
        return decorate
    from .steps import check_steps
    from .steps import steps as derive_steps

    if steps is None:
        from .steps import default_steps

        return default_steps(_trace(fn, kind))
    check_steps(fn.__name__, steps)
    traced = _trace(fn, kind)
    return traced if steps == 1 else derive_steps(traced, steps)


def _trace(fn: Callable[..., None], kind: Any) -> Kernel:
    """The trace :func:`kernel` runs: ``fn`` under ``kind`` (or the default
    kind) into one :class:`Kernel`."""
    from ..ext import DEFAULT_KIND

    resolved_kind = kind if kind is not None else DEFAULT_KIND
    if resolved_kind.output.returned_ttype is not None:
        return _returned_kernel(fn, resolved_kind)
    vocab, scope, own, params = _declarations(fn, resolved_kind)
    planes, args, sink_names, mutables, finishes = _bound_planes(
        fn, params, own, vocab, resolved_kind)
    traced = _retrace(fn, scope, sink_names, mutables, finishes=finishes)
    _call_finishing(traced, fn, args)
    finished = _finished(fn.__name__, params, args, finishes, resolved_kind)
    sinks = _collect(fn.__name__, params, args, sink_names,
                     finishing=bool(finished)) + finished
    extra = _compensated_companion(resolved_kind, planes, vocab, sinks)
    if extra is not None:
        sinks = sinks + (extra,)
    return Kernel(fn.__name__, sinks, planes, kind=resolved_kind)


def _declarations(fn: Callable[..., Any], kind: Any) -> tuple:
    """``(vocabulary, scope, own declarations, parameter names)`` for ``fn``
    traced under ``kind``."""
    vocab = dict(kind.vocabulary)
    defining = _defining_frame(fn)
    scope = _scope(fn, defining.f_locals if defining is not None else {})
    own, params = _own_declarations(fn, scope)
    return vocab, scope, own, params


def _bound_planes(fn: Callable[..., Any], params: list[str], own: dict[str, Any],
                  vocab: dict[str, Any], kind: Any) -> tuple:
    """``(planes, args, sink names, mutable names, finish names)``: every
    parameter's resolved plane (plus the kind's ``terminated`` vocabulary
    planes) and the traced value bound to each parameter.

    ``mutables`` names every Mutable AND WideOut OWN-COLUMN parameter: both
    forms get the SAME plain-variable ast-front-end treatment
    (:mod:`hawk.trace.astpass`) — a WideOut's SCATTER write
    (``buf.write(v, at=lane)``) is a method call, untouched either way."""
    planes = _resolve_planes(fn, params, own, vocab, kind)
    for name, plane in vocab.items():
        if (name not in planes and isinstance(plane, Plane)
                and plane.form == "terminated"):
            planes[name] = plane
    args = {name: _bind(name, planes[name]) for name in params}
    sink_names = frozenset(n for n in params if _is_sink(planes[n]))
    mutables = frozenset(n for n in sink_names
                         if planes[n].form in ("mutable", "wide_out"))
    finishes = _finish_names(params, planes)
    return planes, args, sink_names, mutables, finishes


def _returned_kernel(fn: Callable[..., Any], kind: Any) -> Kernel:
    """The ``Output.returned`` path: trace ``fn``'s body as a VALUE — the
    same ``returns_value`` retrace :func:`trace_value` itself uses — and
    commit it into the ``Mutable`` slot the KIND synthesises. Reached only
    from :func:`kernel` when ``kind.output.returned_ttype`` is set."""
    ttype, slot = kind.output.returned_ttype, kind.output.slot
    vocab, scope, own, params = _declarations(fn, kind)
    if slot in params:
        raise HawkError(
            f"kernel {fn.__name__!r}: Output.returned's synthesised slot "
            f"{slot!r} may not also be a parameter name"
        )
    if slot in vocab:
        raise HawkError(
            f"kernel {fn.__name__!r}: Output.returned's synthesised slot "
            f"{slot!r} collides with kind {kind.slug!r}'s own vocabulary"
        )
    planes, args, sink_names, mutables, finishes = _bound_planes(
        fn, params, own, vocab, kind)
    traced = _retrace(fn, scope, sink_names, mutables, returns_value=True,
                      finishes=finishes)
    out = _call_finishing(traced, fn, args)
    if not isinstance(out, Value):
        raise HawkError(
            f"kernel {fn.__name__!r}: the body returned {type(out).__name__} "
            f"{out!r}, not a traced value; the returned expression must depend "
            "on at least one parameter"
        )
    ref = MutableRef(slot, ttype)
    ref.write(out)
    named = _collect(fn.__name__, params, args, sink_names) if sink_names else ()
    sinks = (tuple(named) + tuple(ref.sinks)
             + _finished(fn.__name__, params, args, finishes, kind))
    extra = _compensated_companion(kind, planes, vocab, sinks)
    if extra is not None:
        sinks = sinks + (extra,)
    return Kernel(fn.__name__, sinks, planes, kind=kind)


def _own_declarations(fn: Callable[..., None], scope: dict[str, Any]
                      ) -> tuple[dict[str, Any], list[str]]:
    """A body's own annotated declarations and its parameter names, in
    signature order."""
    own = {name: resolve(name, _annotation(fn, name, ann, scope))
           for name, ann in _annotations(fn).items()}
    return own, list(inspect.signature(fn).parameters)


def _resolve_planes(fn: Callable[..., None], params: list[str],
                    own: dict[str, Any], vocab: dict[str, Any],
                    kind: Any) -> dict[str, Any]:
    """A parameter's declaration: its OWN annotation, or the kind's
    vocabulary entry — precedence is NOT "own wins". A vocabulary parameter
    carries no annotation, or one EQUAL to the vocabulary's; a parameter
    outside it must carry its own."""
    mismatched = [n for n in params
                  if n in own and n in vocab and own[n] != vocab[n]]
    if mismatched:
        raise HawkError(
            f"kernel {fn.__name__!r}: parameter(s) {mismatched} carry an "
            f"annotation that disagrees with kind {kind.slug!r}'s vocabulary "
            "— a vocabulary parameter carries no annotation of its own, or one "
            "equal to the vocabulary's"
        )
    planes: dict[str, Any] = {}
    for name in params:
        if name in own:
            planes[name] = own[name]
        elif name in vocab:
            planes[name] = vocab[name]
    missing = [p for p in params if p not in planes]
    if missing:
        raise HawkError(
            f"kernel {fn.__name__!r}: parameter(s) {missing} carry no declaration — "
            "every parameter IS its vocabulary declaration"
            + (f" or a member of kind {kind.slug!r}'s own vocabulary "
               f"{sorted(vocab)}" if vocab else "")
        )
    return planes


def _compensated_companion(kind: Any, planes: dict[str, Any],
                           vocab: dict[str, Any], sinks: tuple[Node, ...]
                          ) -> tuple[Slot, ...]:
    """The ``Slot`` a compensated own-column target's companion is declared
    under (:func:`hawk.ext.compensated_target`'s own-column case), or ``()``
    for a non-compensated kind or the SCATTERED form.

    Refuses if the companion name collides with a parameter or vocabulary
    name — it belongs to the KIND, never the author."""
    from ..ext import compensated_target

    resolved = compensated_target(kind, sinks)
    if resolved is None:
        return None
    name, own_column = resolved
    if not own_column:
        return None
    comp = kind.sink.compensation
    if comp in planes or comp in vocab:
        raise HawkError(
            f"Kind({kind.slug!r}): the compensated companion {comp!r} collides "
            "with a parameter or vocabulary name — the companion is synthesised "
            "by the KIND for the own-column form and may not be declared by the "
            "author"
        )
    target_sink = next(s for s in sinks if s.name == name)
    # A real Assign sink, shaped like an author's own Mutable commit (never
    # rendered: `_sink` in hawk/emit/aether.py skips `kind.sink.compensation`).
    # The dummy value is a structural zero of the target's rank — a bare
    # rank>=1 `Const` has no aether spelling, so it is splatted instead.
    ttype = target_sink.ttype
    zero = Const(0.0, TensorType((), ttype.dtype))
    value = zero if not ttype.shape else make("splat", (zero,), ttype.shape)
    return Assign(comp, value, ttype)


def trace_value(fn: Callable[..., Any], bindings: dict[str, Any]) -> Value:
    """Trace ``fn`` — a body that RETURNS one value and commits nothing — and
    return that value: how a host builds a kernel from an authored EXPRESSION
    (an event predicate, a loss term). ``bindings`` maps every parameter of
    ``fn`` to its declaration, so ``fn`` needs no annotations. The body must
    end with ``return <expr>``.
    """
    params = list(inspect.signature(fn).parameters)
    missing = [p for p in params if p not in bindings]
    extra = [n for n in bindings if n not in params]
    if missing or extra:
        raise HawkError(
            f"trace_value({fn.__name__!r}): bindings must name every parameter "
            f"exactly; missing {missing}, not a parameter {extra}")
    planes = {name: resolve(name, decl) for name, decl in bindings.items()}
    sinks = [n for n in params if _is_sink(planes[n])]
    if sinks:
        raise HawkError(
            f"trace_value({fn.__name__!r}): {sinks} declare an output plane; a traced "
            "value commits nothing — return the value and commit it in the kernel "
            "that uses it")
    args = {name: _bind(name, planes[name]) for name in params}
    traced = _retrace(fn, _scope(fn, {}), frozenset(), frozenset(),
                      returns_value=True)
    out = _call(traced, fn, args)
    if not isinstance(out, Value):
        raise HawkError(
            f"trace_value({fn.__name__!r}): the body returned {type(out).__name__} "
            f"{out!r}, not a traced value; the returned expression must depend on "
            "at least one parameter")
    return out


def _call(traced: Callable[..., Any], fn: Callable[..., Any],
          args: dict[str, Any]) -> Any:
    """Call the re-traced body by parameter KIND: positional parameters
    positionally, keyword-only ones by name. The slot order is the traced walk's,
    so how a parameter is passed never changes the artifact."""
    positional, keyword = [], {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise HawkError(
                f"{fn.__name__!r}: *{name} / **{name} cannot be traced — every "
                "parameter is one declared plane")
        if p.kind is p.KEYWORD_ONLY:
            keyword[name] = args[name]
        else:
            positional.append(args[name])
    return traced(*positional, **keyword)


def _call_finishing(traced: Callable[..., Any], fn: Callable[..., Any],
                    args: dict[str, Any]) -> Any:
    """:func:`_call`, with a finish refusal raised mid-body re-raised NAMING
    the kernel (the body runs before the refusal knows which kernel it is in)."""
    try:
        return _call(traced, fn, args)
    except FinishRefusal as exc:
        raise HawkError(f"kernel {fn.__name__!r}: {exc}") from None


def _finish_names(params: list[str], planes: dict[str, Any]) -> frozenset[str]:
    """The parameters a body may FINISH: every ``Terminated`` it declares."""
    return frozenset(n for n in params if isinstance(planes[n], Plane)
                     and planes[n].form == "terminated")


def _finished(name: str, params: list[str], args: dict[str, Any],
              finishes: frozenset[str], kind: Any) -> tuple[Node, ...]:
    """The kernel's :class:`~hawk.ir.nodes.Finish` sinks, refusing a mask its
    kind's guard does not gate on (a finished sample the guard never skips
    would run again on the next launch and count twice) and a second finished
    mask (the counter and the sidecar's ``finish`` name ONE)."""
    out: list[Node] = []
    for p in params:
        if p not in finishes:
            continue
        committed = args[p].sinks
        if not committed:
            continue
        if p not in kind.guard.names:
            raise HawkError(
                f"kernel {name!r}: finishes the mask {p!r}, which kind "
                f"{kind.slug!r}'s guard does not gate on "
                f"({list(kind.guard.names)}); a kernel finishes only a mask its "
                "guard skips")
        out.extend(committed)
    if len(out) > 1:
        raise HawkError(
            f"kernel {name!r}: finishes {len(out)} masks "
            f"({[s.name for s in out]}); a kernel finishes its sample through ONE")
    return tuple(out)


def _declared(planes: dict[str, Any]) -> tuple:
    """The slots a DECLARATION binds whether or not the body reads them.

    Today exactly one form: the reserved ``terminated`` MASK. The guard seam
    gates on it, so it must be a slot of the kernel's ``arg_spec`` even in a
    body that never names it."""
    return tuple(Slot(role_of(p), name, p.ttype)
                 for name, p in planes.items()
                 if isinstance(p, Plane) and p.form == "terminated")


def _is_sink(plane: Plane | Quantity) -> bool:
    return isinstance(plane, Plane) and plane.form in _SINK_FORMS


def _annotations(fn: Callable[..., None]) -> dict[str, Any]:
    return {k: v for k, v in getattr(fn, "__annotations__", {}).items()
            if k != "return"}


def _annotation(fn: Callable[..., None], name: str, ann: Any,
                scope: dict[str, Any]) -> Any:
    """A string annotation (``from __future__ import annotations``) is resolved
    in the function's own scope; anything else is already the declaration."""
    if not isinstance(ann, str):
        return ann
    try:
        return eval(ann, scope)  # the author's own annotation
    except Exception as exc:                      # pragma: no cover - author error
        raise HawkError(
            f"kernel {fn.__name__!r}: parameter {name!r}'s annotation {ann!r} does "
            f"not resolve in the function's scope ({exc})"
        ) from exc


def _defining_frame(fn: Callable[..., None]) -> Any:
    """The frame that executed the ``def`` (or ``lambda``) producing ``fn``:
    walk OUTWARD from this call, stopping at the first frame whose OWN code
    embeds ``fn.__code__`` as a literal constant (what a nested ``def`` leaves
    in its enclosing scope's ``co_consts``), skipping decorator-wrapper
    frames. ``None`` if no such frame is reachable — the caller falls back to
    globals + closure only.

    THE ONE PLACE hawk touches the call stack."""
    frame = sys._getframe(1)
    while frame is not None:
        if fn.__code__ in frame.f_code.co_consts:
            return frame
        frame = frame.f_back
    return None


def _scope(fn: Callable[..., None], defining: dict[str, Any]) -> dict[str, Any]:
    """The function's globals, the locals of the scope it is DEFINED in and its
    closure cells — so a kernel defined inside another function sees the names
    its body reads, and a declaration held in a local still resolves when
    ``from __future__ import annotations`` makes the annotation a string."""
    scope = dict(fn.__globals__)
    scope.update(defining)
    cells = fn.__closure__ or ()
    for name, cell in zip(fn.__code__.co_freevars, cells):
        try:
            scope[name] = cell.cell_contents
        except ValueError:                        # pragma: no cover - unbound cell
            pass
    return scope


def _bind(name: str, plane: Plane | Quantity) -> Any:
    """The traced argument one declaration binds to."""
    if isinstance(plane, Quantity):
        return plane.read_with(Value)
    role = role_of(plane)
    if plane.form == "mutable":
        return MutableRef(name, plane.ttype)
    if plane.form == "accum":
        return AccumRef(name, plane.ttype)
    if plane.form == "reduce":
        return ReduceRef(name, plane.ttype, plane.op)
    if plane.form == "wide_out":
        return WideOutRef(name, plane.ttype)
    if plane.form == "terminated":
        return TerminatedRef(Leaf("terminated", role, name, plane.ttype))
    if plane.form in ("table", "wide"):
        # one READ form for both roles: pins `lookup` and `wide_in` to the
        # same 32-byte ScalarHandle, so the access is the at(expr) either way.
        return TableRef(name, plane.ttype, role=role, dims=plane.dims)
    return Value(Leaf(_LEAF_KIND[role], role, name, plane.ttype))


def _retrace(fn: Callable[..., None], scope: dict[str, Any],
             sink_names: frozenset[str],
             mutables: frozenset[str],
             returns_value: bool = False,
             finishes: frozenset[str] = frozenset()) -> Callable[..., None]:
    """Parse, rewrite and re-materialise the body (the ``ast`` pass) — the
    transform runs once PER DEFINITION; the trace itself, below
    in :func:`kernel`, still runs on every call, since the values it produces
    differ per call."""
    filename = inspect.getsourcefile(fn) or "<kernel>"
    code = fn.__code__
    check = (code.co_code, code.co_firstlineno, filename, sink_names, mutables,
             returns_value, finishes)
    cached = _TRANSFORM_MEMO.get(id(code))
    if cached is not None and cached[0] == check:
        compiled = cached[1]
    else:
        source, first = inspect.getsourcelines(fn)
        tree = ast.parse(textwrap.dedent("".join(source)), filename=filename)
        ast.increment_lineno(tree, first - 1)
        fn_def = tree.body[0]
        if not isinstance(fn_def, ast.FunctionDef):
            raise HawkError(
                f"kernel {fn.__name__!r}: only a `def` can be traced")
        rewritten = astpass.transform(fn_def, filename, sink_names, mutables,
                                      returns_value, finishes=finishes)
        module = ast.Module(body=[rewritten], type_ignores=[])
        compiled = compile(module, filename, "exec")  # the author's body
        if len(_TRANSFORM_MEMO) >= _TRANSFORM_MEMO_LIMIT:
            _TRANSFORM_MEMO.clear()
        _TRANSFORM_MEMO[id(code)] = (check, compiled)
    scope[astpass.SELECT_HELPER] = astpass.merge
    scope[astpass.LOOP_HELPER] = astpass.open_loop
    scope[astpass.SETITEM_HELPER] = astpass.set_component
    exec(compiled, scope)  # the author's rewritten body
    return scope[fn.__name__]


def _collect(name: str, params: list[str], args: dict[str, Any],
             sink_names: frozenset[str], *,
             finishing: bool = False) -> tuple[Node, ...]:
    """The sink set, in PARAMETER order — the walk's declaration order."""
    sinks: list[Node] = []
    for p in params:
        if p not in sink_names:
            continue
        committed = args[p].sinks
        if not committed:
            raise HawkError(
                f"kernel {name!r}: the output {p!r} is declared but never committed "
                "— every declared sink must be written (a plane this kernel only "
                "reads is an input — declare it Scalar/Vector[W])"
            )
        sinks.extend(committed)
    if not sinks and not finishing:
        raise HawkError(
            f"kernel {name!r}: declares no output plane; a kernel commits through a "
            "Mutable / Accum / Reduce parameter"
        )
    return tuple(sinks)
