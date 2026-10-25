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
import hashlib
import inspect
import sys
import textwrap
import threading
import typing
from collections.abc import Callable
from typing import Any

from ..ir import Const, HawkError, Leaf, Node, Quantity, Walk, canonical, make
from ..ir.nodes import COMPANION_SUFFIX, AccumWrite, Assign, compensates
from ..types import Slot, TensorType
from . import astpass, reduce
from .decl import Plane, resolve, role_of, spell
from .lane_axes import LaneAxes, check_group, lane_space, tracing
from .value import (
    FinishRefusal,
    MutableRef,
    TableRef,
    TerminatedRef,
    Value,
    WideOutRef,
)

#: How an input plane's ROLE picks the leaf KIND it mints.
_LEAF_KIND = {"per_sample": "vocab_read", "vec_in": "vocab_read",
              "mat_in": "vocab_read", "uniform": "uniform",
              "terminated": "terminated"}
_SINK_FORMS = ("mutable", "wide_out")

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

    __slots__ = ("name", "sinks", "walk", "planes", "kind", "step", "one_step",
                 "time", "time_step", "source_digest", "jump", "returns",
                 "lane_space", "group", "_plan", "_lock")

    def __init__(self, name: str, sinks: tuple[Node, ...],
                 planes: dict[str, Any] | None = None, *, kind: Any = None,
                 step: Kernel | None = None,
                 one_step: Kernel | None = None,
                 source_digest: str | None = None,
                 returns: Any = None) -> None:
        from ..ext import DEFAULT_KIND

        self.name = name
        self.kind = kind if kind is not None else DEFAULT_KIND
        sinks = _scattered_companion(self.kind, sinks)
        self.sinks = sinks
        self.planes = {} if planes is None else planes
        self.walk: Walk = canonical(sinks, _declared(self.planes))
        #: The launch shape, ``((label|None, extent|None), ...)`` row-major with the
        #: fastest axis last: the ``Grid[...]`` declared, else one unlabelled
        #: runtime axis. Read by the sidecar and the bind doors.
        self.lane_space = lane_space(self.planes)
        #: The compaction group, a lane count (``@hawk.kernel(group=...)``): 1.
        self.group = 1
        #: The single step a default automatic kernel was derived from (see
        #: :func:`kernel`); ``None`` otherwise, used by ``vjp``/``jvp``.
        self.step = step
        #: The single step an automatic kernel runs when ``fused_steps`` is
        #: ``1`` (entry branches once per launch between this body and the
        #: fused loop); ``None`` otherwise. Emission only.
        self.one_step = one_step
        #: The time plane and step-size parameter the author named
        #: (``@hawk.kernel(time="t", time_step="dt")``), else ``None``: read
        #: by the taping twin, never by the plain kernel's emission.
        self.time: str | None = None
        self.time_step: str | None = None
        #: The sha256 of the traced function's dedented source (see
        #: :func:`source_digest`); ``None`` for a hand-built kernel or one
        #: whose source cannot be read. Traceability only.
        self.source_digest = source_digest
        #: The width of the derivative through the kernel's switches
        #: (``@hawk.kernel(jump=...)``): a number, ``"auto"`` or a Param
        #: name; ``None`` (a switch has derivative zero) otherwise. Read by
        #: :mod:`hawk.diff`, never by the kernel's own emission.
        self.jump: Any = None
        #: What the body ``return``s, as the same tree of tuples / lists /
        #: dicts with each leaf replaced by its ``Mutable`` slot name (a bare
        #: name for one array); ``None`` for a kernel that commits through
        #: its parameters and for a hand-built one.
        self.returns = returns
        self._plan = None
        self._lock = threading.Lock()

    @property
    def arg_spec(self) -> tuple[tuple[str, str], ...]:
        """The walk's ``(role, name)`` slot order — every consumer's source."""
        return self.walk.arg_spec

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run one launch where the arrays live, through the plan eagle
        deploys for this kernel (built on the first call, cached on disk)."""
        plan = self._plan
        if plan is None:
            plan = self._deploy()
        return plan(*args, **kwargs)

    def _deploy(self) -> Any:
        try:
            import eagle
        except ModuleNotFoundError as exc:
            if exc.name != "eagle":
                raise
            raise ImportError(f"{self.name}(...) runs through eagle: "
                              "pip install raptor-eagle") from None
        with self._lock:
            if self._plan is None:
                self._plan = eagle.deploy(self)
        return self._plan

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<Kernel {self.name} slots={len(self.arg_spec)}>"


def kernel(fn: Callable[..., None] | None = None, *, kind: Any = None,
           steps: Any = None, time: str | None = None,
           time_step: str | None = None, jump: Any = None,
           group: int | str = 1) -> Any:
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
    ``steps=1`` is the explicit opt-out: the plain kernel, always.

    ``time_step`` names the step-size parameter (a declared scalar input)
    and ``time`` the time plane (default ``"t"`` once ``time_step`` is
    given). The plain kernel is unchanged; :func:`hawk.diff.tape` reads
    them to promote the step size to a per-sample plane and to write the
    time plane in index form. A time plane the body updates must update as
    ``t = t + dt`` (checked here, at decoration); one the body does not
    declare is minted by the twin.

    ``jump`` makes :func:`hawk.diff.vjp`/:func:`~hawk.diff.jvp` differentiate
    the kernel's parameter-dependent ``if``/``where`` switches as the
    smoothed loss: a comparison ``l < r`` passes on the jump between its
    branches, spread over ``|l - r| < sqrt(5) * jump`` (Epanechnikov kernel
    of standard deviation ``jump``). ``jump=`` is the width in the units of
    the compared values (about 2% of their span), or the name of a
    ``Param[Scalar]`` holding it; ``"auto"`` picks it per batch from the
    spread of the compared values. The primal kernel is unchanged.

    ``group`` is the compaction unit: a lane count (``group=32``) or the label of
    a ``Grid`` axis (``group="sample"``: that axis and every axis after it). A
    run that drops finished lanes keeps whole groups together. The default, 1,
    compacts lane by lane."""
    if fn is None:
        def decorate(f: Callable[..., None]) -> Any:
            return kernel(f, kind=kind, steps=steps, time=time, time_step=time_step,
                          jump=jump, group=group)
        return decorate
    if time is not None and time_step is None:
        raise HawkError(
            f"kernel {fn.__name__!r}: time={time!r} needs time_step=, the "
            "parameter the time plane advances by")
    built = _kernel(fn, kind, steps)
    count = check_group(fn.__name__, group, built.planes)
    for k in (built, built.step, built.one_step):
        if k is not None:
            k.group = count
    if jump is not None:
        from ..diff.jump import check_jump

        planes = (built.step or built).planes
        checked = check_jump(fn.__name__, jump, planes)
        for k in (built, built.step, built.one_step):
            if k is not None:
                k.jump = checked
    if time_step is not None:
        from ..diff.tape import check_time

        check_time(built, time or "t", time_step)
        for k in (built, built.step, built.one_step):
            if k is not None:
                k.time, k.time_step = time or "t", time_step
    return built


def resolve_dispatch(kernel: Kernel, policies: dict[str, str]) -> Kernel:
    """``kernel`` with each ``"auto"`` dispatch group lowered as
    ``policies[group]`` (``"switch"``, ``"segmented"`` or ``"predicated"``):
    how a launching runtime applies the policy it chose. A group the author
    declared a policy for is refused — the author's choice is never
    overridden — as is an unknown group or policy. An automatic kernel is
    resolved on its single step and re-derived, under the same name — except
    to ``"segmented"``, which splits the step into one kernel per branch and
    so returns the single step itself: its units run one step per launch."""
    from ..ir.nodes import DISPATCH_POLICIES
    from ..ir.segment import with_policies
    from ..ir.walk import dispatch_groups, loop_body_groups
    from .steps import default_steps

    step = kernel.step if kernel.step is not None else kernel
    groups = dispatch_groups(step.walk)
    in_loops = loop_body_groups(step.walk)
    chosen = {}
    for group, policy in policies.items():
        node = groups.get(group)
        if node is None:
            raise HawkError(
                f"resolve_dispatch({kernel.name!r}): {group!r} is not a dispatch "
                f"group of this kernel; its groups are {sorted(groups)}")
        if node.policy != "auto":
            raise HawkError(
                f"resolve_dispatch({kernel.name!r}): {group!r} was declared "
                f"policy={node.policy!r}; only an 'auto' group is resolved — the "
                "author's policy is kept")
        if group in in_loops:
            raise HawkError(
                f"resolve_dispatch({kernel.name!r}): {group!r} sits inside a loop "
                "body, where a policy cannot be swapped yet; declare its policy "
                "in the kernel instead")
        if policy == "auto" or policy not in DISPATCH_POLICIES:
            raise HawkError(
                f"resolve_dispatch({kernel.name!r}): {group!r} resolves to one of "
                f"'switch', 'segmented', 'predicated', got {policy!r}")
        chosen[id(node)] = policy
    resolved = Kernel(step.name, with_policies(step.sinks, chosen), step.planes,
                      kind=step.kind, source_digest=step.source_digest,
                      returns=step.returns)
    resolved.time, resolved.time_step = step.time, step.time_step
    resolved.jump = step.jump
    if kernel.step is None or "segmented" in chosen.values():
        return resolved
    fused = default_steps(resolved)
    fused.time, fused.time_step = step.time, step.time_step
    fused.jump = step.jump
    return fused


def _kernel(fn: Callable[..., None], kind: Any, steps: Any) -> Any:
    """:func:`kernel`'s build, before the time declaration is recorded."""
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
    kind) into one :class:`Kernel`. A body ending ``return <expr>`` is traced
    as a VALUE tree whose leaves commit into synthesised ``Mutable`` slots."""
    from ..ext import DEFAULT_KIND

    resolved_kind = kind if kind is not None else DEFAULT_KIND
    forced = resolved_kind.output.returned_ttype
    vocab, scope, own, params = _declarations(fn, resolved_kind)
    if forced is not None:
        slot = resolved_kind.output.slot
        if slot in params:
            raise HawkError(
                f"kernel {fn.__name__!r}: Output.returned's synthesised slot "
                f"{slot!r} may not also be a parameter name"
            )
        if slot in vocab:
            raise HawkError(
                f"kernel {fn.__name__!r}: Output.returned's synthesised slot "
                f"{slot!r} collides with kind {resolved_kind.slug!r}'s own vocabulary"
            )
    planes, args, sink_names, mutables, finishes = _bound_planes(
        fn, params, own, vocab, resolved_kind)
    scope[astpass.REFS] = args
    traced, returning, shape = _retrace(
        fn, scope, sink_names, mutables, finishes=finishes,
        returns_value=True if forced is not None else None)
    with tracing(lane_space(planes)):
        out = _call_finishing(traced, fn, args)
    _retag_by_key(planes, args)
    _check_axis_sinks(fn.__name__, planes, args)
    finished = _finished(fn.__name__, params, args, finishes, resolved_kind)
    if returning:
        refs, returns = _commit_returns(fn, out, shape, forced, resolved_kind,
                                        planes, vocab, scope)
        named = _collect(fn.__name__, params, args, sink_names) if sink_names else ()
        sinks = named + tuple(s for r in refs for s in r.sinks) + finished
    else:
        returns = None
        sinks = _collect(fn.__name__, params, args, sink_names,
                         finishing=bool(finished)) + finished
    extra = _compensated_companion(resolved_kind, planes, vocab, sinks)
    if extra is not None:
        sinks = sinks + (extra,)
    sinks = sinks + _policy_companions(fn.__name__, planes, sinks)
    return Kernel(fn.__name__, sinks, planes, kind=resolved_kind,
                  source_digest=source_digest(fn), returns=returns)


def source_digest(fn: Callable[..., Any]) -> str | None:
    """The sha256 hex digest of ``fn``'s dedented source, as
    :func:`inspect.getsource` returns it (decorators included), or ``None``
    when the source cannot be read."""
    try:
        text = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    return hashlib.sha256(text.encode()).hexdigest()


def _retag_by_key(planes: dict[str, Any], args: dict[str, Any]) -> None:
    """A plane a body wrote by key (``hawk.scatter(y)[key] (+)= v``, or
    ``y = hawk.sum(x, over=...)``) is shared, so its declaration becomes the
    matching accumulate / store plane."""
    for name, ref in args.items():
        kind = getattr(ref, "by_key_kind", None)
        if kind is not None:
            planes[name] = Plane(kind, planes[name].ttype, dims=planes[name].dims)


def _check_axis_sinks(name: str, planes: dict[str, Any], args: dict[str, Any]) -> None:
    """A sink with named axes is bound ``(*payload, *axes)``, not one value per
    lane: a body that assigns it on its own column is refused."""
    for p, plane in planes.items():
        if (isinstance(plane, Plane) and plane.form == "mutable" and plane.dims
                and p in args and any(isinstance(s, Assign) for s in args[p].sinks)):
            axes = ", ".join(f'"{label}":{size}' for label, size, _stride in plane.dims)
            raise HawkError(
                f"kernel {name!r}: {p!r} is declared with the axes [{axes}], so it is "
                f"written by key (hawk.scatter({p})[key] += v, or {p} = hawk.sum(x, "
                "over=...)), not assigned per lane")


def _declarations(fn: Callable[..., Any], kind: Any) -> tuple:
    """``(vocabulary, scope, own declarations, parameter names)`` for ``fn``
    traced under ``kind``."""
    vocab = dict(kind.vocabulary)
    defining = _defining_frame(fn)
    scope = _scope(fn, defining.f_locals if defining is not None else {})
    own, params = _own_declarations(fn, scope)
    from ..compile.toolchain import libc_symbols

    if fn.__name__ in libc_symbols():
        raise HawkError(
            f"kernel {fn.__name__!r}: the name is a C library function, and the kernel's "
            "entry would redeclare it; rename the kernel")
    if "buffer" in params:
        raise HawkError(
            f"kernel {fn.__name__!r}: `buffer` is a reserved parameter name — "
            "`buffer=` is the keyword a kernel call takes inside an eagle graph "
            "builder for the array an output is written to; rename the parameter")
    return vocab, scope, own, params


def _bound_planes(fn: Callable[..., Any], params: list[str], own: dict[str, Any],
                  vocab: dict[str, Any], kind: Any) -> tuple:
    """``(planes, args, sink names, mutable names, finish names)``: every
    parameter's resolved plane (plus the kind's ``terminated`` vocabulary
    planes) and the traced value bound to each parameter.

    ``mutables`` names every Mutable AND WideOut OWN-COLUMN parameter: both
    forms get the SAME plain-variable ast-front-end treatment
    (:mod:`hawk.trace.astpass`) — a WideOut's SCATTER write
    (``hawk.scatter(buf)[k] = v``) is a subscript store, untouched either way."""
    planes = _resolve_planes(fn, params, own, vocab, kind)
    for name, plane in vocab.items():
        if (name not in planes and isinstance(plane, Plane)
                and plane.form == "terminated"):
            planes[name] = plane
    _check_grid(fn.__name__, params, planes)
    args = {name: _bind(name, planes[name]) for name in params}
    sink_names = frozenset(n for n in params if _is_sink(planes[n]))
    mutables = frozenset(n for n in sink_names
                         if planes[n].form in ("mutable", "wide_out"))
    finishes = _finish_names(params, planes)
    return planes, args, sink_names, mutables, finishes


def _check_grid(name: str, params: list[str], planes: dict[str, Any]) -> None:
    """Refuse a second ``Grid[...]`` and a Grid label that is also a parameter name (a bind keyword
    would name two things)."""
    grids = [p for p in params if isinstance(planes[p], Plane) and planes[p].grid]
    if not grids:
        return
    if len(grids) > 1:
        raise HawkError(
            f"kernel {name!r}: {grids} are all Grid[...] declarations; a kernel has "
            "ONE launch shape")
    axes = planes[grids[0]].grid
    clash = [label for label, _ in axes if label in params]
    if clash:
        raise HawkError(
            f"kernel {name!r}: the Grid axis label(s) {clash} are also parameter "
            "names, so a launch could not tell which one a keyword binds; rename the label")


class _Leaf:
    """One returned array: its slot name (``None`` until named) and its value."""

    __slots__ = ("name", "value")

    def __init__(self, name: str | None, value: Value) -> None:
        self.name, self.value = name, value


def _commit_returns(fn: Callable[..., Any], out: Any, shape: Any, forced: Any,
                    kind: Any, planes: dict[str, Any], vocab: dict[str, Any],
                    scope: dict[str, Any]) -> tuple[list[MutableRef], Any]:
    """Commit what the body returned into one ``Mutable`` slot per leaf.
    Returns the refs (leaf order) and the ``Kernel.returns`` tree. ``forced``
    is the ttype of the slot a kind synthesises (``Output.returned``): a
    single value, written to that slot."""
    where = f"kernel {fn.__name__!r}"
    leaves: list[_Leaf] = []
    if forced is not None:
        tree = _leaves(out, None, None, where, leaves)
        if not isinstance(tree, _Leaf):
            raise HawkError(f"{where}: kind {kind.slug!r} returns one value into "
                            f"{kind.output.slot!r}, not a tuple/list/dict")
        tree.name = kind.output.slot
        ttypes = [forced]
    else:
        tree = _leaves(out, shape, None, where, leaves)
        taken = set(planes) | set(vocab)
        for k, leaf in enumerate(leaves):
            if leaf.name is None:
                leaf.name = "out" if len(leaves) == 1 else f"out{k}"
            if leaf.name in taken:
                raise HawkError(
                    f"{where}: the return names an output {leaf.name!r} after a "
                    f"parameter or another output; use another name "
                    f"({leaf.name}_next)")
            taken.add(leaf.name)
        ann = _return_annotation(fn, scope)
        if ann is not inspect.Signature.empty:
            _check_annotation(where, ann, tree, "")
        ttypes = [leaf.value.ttype for leaf in leaves]
    refs = []
    for leaf, ttype in zip(leaves, ttypes):
        if isinstance(leaf.value, reduce.Reduced):
            dims = reduce.axes_plane(leaf.value.kept)
            ref = MutableRef(leaf.name, ttype, dims)
            ref.write(leaf.value)
            planes[leaf.name] = Plane("accum", ttype, dims=dims)
        else:
            ref = MutableRef(leaf.name, ttype)
            ref.write(leaf.value)
            planes[leaf.name] = Plane("mutable", ttype)
        refs.append(ref)
    return refs, _names(tree)


def _leaves(out: Any, shape: Any, hint: str | None, where: str,
            leaves: list[_Leaf]) -> Any:
    """``out`` with every traced value replaced by a :class:`_Leaf`, named by
    the dict key or variable the static ``shape`` (``astpass.return_shape``)
    gives it."""
    kind = type(out)
    if kind is tuple or kind is list:
        kids = (shape[1] if isinstance(shape, tuple) and shape[0] == kind.__name__
                and len(shape[1]) == len(out) else (None,) * len(out))
        return kind(_leaves(o, k, None, where, leaves) for o, k in zip(out, kids))
    if kind is dict:
        if not (isinstance(shape, tuple) and shape[0] == "dict"
                and [k for k, _ in shape[1]] == list(out)):
            raise HawkError(f"{where}: a returned dict is written in the return "
                            'statement with string-literal keys (`return {"pos": p}`)')
        return {k: _leaves(out[k], s, k, where, leaves) for k, s in shape[1]}
    if not isinstance(out, (Value, reduce.Reduced)):
        what = ("a class instance is not supported — return a value, a tuple/list, "
                "or a dict of them" if hasattr(out, "__dict__") else
                "the returned expression must depend on at least one parameter")
        raise HawkError(f"{where}: the body returned {type(out).__name__} {out!r}, "
                        f"not a traced value; {what}")
    leaf = _Leaf(hint or (shape if isinstance(shape, str) else None), out)
    leaves.append(leaf)
    return leaf


def _names(tree: Any) -> Any:
    """``tree`` with each leaf replaced by its slot name."""
    if isinstance(tree, _Leaf):
        return tree.name
    if isinstance(tree, dict):
        return {k: _names(v) for k, v in tree.items()}
    return type(tree)(_names(v) for v in tree)


def _return_annotation(fn: Callable[..., Any], scope: dict[str, Any]) -> Any:
    ann = getattr(fn, "__annotations__", {}).get("return", inspect.Signature.empty)
    if isinstance(ann, str):
        try:
            return eval(ann, scope)  # the author's own annotation
        except Exception as exc:                  # pragma: no cover - author error
            raise HawkError(f"kernel {fn.__name__!r}: the return annotation {ann!r} "
                            f"does not resolve in the function's scope ({exc})") from exc
    return ann


def _spelled(ann: Any) -> str:
    """A return annotation as the author wrote it: payload types by name
    (``Scalar``, ``Vector[3]``), containers around them."""
    if isinstance(ann, TensorType):
        return spell(ann)
    org, args = typing.get_origin(ann), typing.get_args(ann)
    if org in (tuple, list, dict) and args:
        return f"{org.__name__}[{', '.join('...' if a is Ellipsis else _spelled(a) for a in args)}]"
    return getattr(ann, "__name__", repr(ann))


def _check_annotation(where: str, ann: Any, tree: Any, at: str) -> None:
    """Refuse a ``-> T`` the traced ``tree`` does not match: container kind,
    item count, then each leaf's type."""
    org = typing.get_origin(ann) or (ann if ann in (tuple, list, dict) else None)
    args = typing.get_args(ann)
    got = (spell(tree.value.ttype) if isinstance(tree, _Leaf)
           else f"a {type(tree).__name__}")
    if org is None:
        if not isinstance(ann, TensorType):
            raise HawkError(f"{where}: `-> {_spelled(ann)}` is not a payload type "
                            "(Scalar, Vector[W], ...) or a tuple/list/dict of them")
        if not isinstance(tree, _Leaf) or tree.value.ttype != ann:
            raise HawkError(f"{where}: `-> {spell(ann)}`, but the return{at} is {got}")
        return
    if type(tree) is not org:
        raise HawkError(f"{where}: `-> {_spelled(ann)}` declares a {org.__name__}, but the "
                        f"return{at} is {got}")
    if org is dict:
        items = [(f"{at}[{k!r}]", v, args[1] if args else None)
                 for k, v in tree.items()]
    else:
        many = org is list or (len(args) == 2 and args[1] is Ellipsis)
        if args and not many and len(args) != len(tree):
            raise HawkError(f"{where}: `-> {_spelled(ann)}` declares {len(args)} items, but the "
                            f"return{at} has {len(tree)}")
        items = [(f"{at}[{i}]", v, (args[0] if many else args[i]) if args else None)
                 for i, v in enumerate(tree)]
    for loc, v, a in items:
        if a is not None:
            _check_annotation(where, a, v, loc)


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
    # rendered: `_sink` in hawk/emit/render.py skips `kind.sink.compensation`).
    # The dummy value is a structural zero of the target's rank — a bare
    # rank>=1 `Const` has no aether spelling, so it is splatted instead.
    ttype = target_sink.ttype
    zero = Const(0.0, TensorType((), ttype.dtype))
    value = zero if not ttype.shape else make("splat", (zero,), ttype.shape)
    return Assign(comp, value, ttype)


def _scattered_companion(kind: Any, sinks: tuple[Node, ...]) -> tuple[Node, ...]:
    """``sinks`` plus the companion of a compensated SCATTERED target when no
    sink commits it yet: ``kind.sink.compensation`` accumulated at the target's
    own key, a zero never rendered (`_sink` in hawk/emit/render.py skips it), so
    the kernel declares and binds the plane. How a ``vjp``'s ``bar_<p>`` scatter
    gets its ``bar_<p>_c``. A tagged sink tuple (``Derived``) keeps its tags."""
    from ..ext import compensated_target
    from ..ir.nodes import AccumWrite

    resolved = compensated_target(kind, sinks)
    if resolved is None or resolved[1] or any(
            s.name == kind.sink.compensation for s in sinks):
        return sinks
    target = next(s for s in sinks if s.name == resolved[0])
    extra = AccumWrite(kind.sink.compensation, Const(0.0, TensorType((), target.ttype.dtype)),
                       target.index, target.ttype)
    out = (*sinks, extra)
    if hasattr(sinks, "__dict__"):
        out = tuple.__new__(type(sinks), out)
        out.__dict__.update(vars(sinks))
    return out


def _policy_companions(name: str, planes: dict[str, Any], sinks: tuple[Node, ...]) -> tuple:
    """The companion plane ``<plane>_c`` of each ``policy="compensated"``
    accumulate, bound by a zero commit at the same key (the commit's correction
    lands there; the target writes it, not this sink). Declared like its plane."""
    out: dict = {}
    for s in sinks:
        if isinstance(s, AccumWrite) and compensates(s.policy):
            comp = s.name + COMPANION_SUFFIX
            if comp in planes and comp not in out and not any(
                    isinstance(c, AccumWrite) and c.name == comp for c in sinks):
                raise HawkError(
                    f"kernel {name!r}: {s.name!r} is added under policy='compensated', "
                    f"whose correction lands in {comp!r}, which names a parameter; "
                    "rename the parameter")
            zero = Const(0.0, TensorType((), s.ttype.dtype))
            out.setdefault(comp, []).append(AccumWrite(comp, zero, s.index, s.ttype))
            planes[comp] = Plane("accum", planes[s.name].ttype, dims=planes[s.name].dims)
    return tuple(c for group in out.values() for c in group)


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
                      returns_value=True)[0]
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
    if plane.grid:
        return LaneAxes(plane.grid)
    if plane.form == "mutable":
        return MutableRef(name, plane.ttype, plane.dims)
    if plane.form == "wide_out":
        return WideOutRef(name, plane.ttype, plane.dims)
    if plane.form == "terminated":
        return TerminatedRef(Leaf("terminated", role, name, plane.ttype))
    if plane.form in ("table", "wide"):
        # one READ form for both roles: pins `lookup` and `wide_in` to the
        # same 32-byte ScalarHandle, so the access is the at(expr) either way.
        return TableRef(name, plane.ttype, role=role, dims=plane.dims)
    return Value(Leaf(_LEAF_KIND[role], role, name, plane.ttype, sorted=plane.sorted))


def _retrace(fn: Callable[..., None], scope: dict[str, Any],
             sink_names: frozenset[str],
             mutables: frozenset[str],
             returns_value: bool | None = False,
             finishes: frozenset[str] = frozenset()) -> tuple[Callable[..., None], bool, Any]:
    """Parse, rewrite and re-materialise the body (the ``ast`` pass) — the
    transform runs once PER DEFINITION; the trace itself, below
    in :func:`kernel`, still runs on every call, since the values it produces
    differ per call. ``returns_value=None`` lets the body decide: it returns
    when its last statement is ``return <expr>``. Gives ``(function, whether
    it returns, the static shape of its return)``."""
    filename = inspect.getsourcefile(fn) or "<kernel>"
    code = fn.__code__
    check = (code.co_code, code.co_firstlineno, filename, sink_names, mutables,
             returns_value, finishes)
    cached = _TRANSFORM_MEMO.get(id(code))
    if cached is not None and cached[0] == check:
        compiled, returning, shape = cached[1:]
    else:
        source, first = _source_lines(fn)
        tree = ast.parse(textwrap.dedent("".join(source)), filename=filename)
        ast.increment_lineno(tree, first - 1)
        fn_def = tree.body[0]
        if not isinstance(fn_def, ast.FunctionDef):
            raise HawkError(
                f"kernel {fn.__name__!r}: only a `def` can be traced")
        last = fn_def.body[-1]
        returning = (returns_value if returns_value is not None else
                     isinstance(last, ast.Return) and last.value is not None)
        shape = (astpass.return_shape(last.value, filename)
                 if returning and isinstance(last, ast.Return) and last.value else None)
        rewritten = astpass.transform(fn_def, filename, sink_names, mutables,
                                      returning, finishes=finishes)
        module = ast.Module(body=[rewritten], type_ignores=[])
        compiled = compile(module, filename, "exec")  # the author's body
        if len(_TRANSFORM_MEMO) >= _TRANSFORM_MEMO_LIMIT:
            _TRANSFORM_MEMO.clear()
        _TRANSFORM_MEMO[id(code)] = (check, compiled, returning, shape)
    scope[astpass.SELECT_HELPER] = astpass.merge
    scope[astpass.LOOP_HELPER] = astpass.open_loop
    scope[astpass.SETITEM_HELPER] = astpass.set_component
    scope[astpass.MAXMIN_HELPER] = reduce.maxmin
    scope[astpass.SCATTER_HELPER] = reduce.scatter_store
    scope[astpass.RANGE_HELPER] = astpass.static_range
    exec(compiled, scope)  # the author's rewritten body
    return scope[fn.__name__], returning, shape


def _source_lines(fn: Callable[..., None]) -> tuple[list[str], int]:
    """``inspect.getsourcelines(fn)``; the text of a kernel defined under
    ``python -c`` is lifted through :mod:`raptor.flow`, and any other source
    that is not on disk is refused with the way out."""
    try:
        return inspect.getsourcelines(fn)
    except (OSError, TypeError):
        from raptor import flow

        if flow.lift_dash_c(fn):
            return inspect.getsourcelines(fn)
        raise HawkError(
            f"kernel {fn.__name__!r}: its source is not on disk (python -c with "
            "other code, exec, the REPL before Python 3.13) — hawk traces the "
            "source text. Put the kernel in a .py file, or use IPython / Jupyter, "
            "which keep cell sources; to define it from a string, run it with "
            "`raptor.flow.exec_source(text, namespace)`, or register the text "
            "yourself: `linecache.cache[filename] = (len(text), None, "
            "text.splitlines(True), filename)` and compile under that filename"
        ) from None


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
            "Mutable / WideOut parameter"
        )
    return tuple(sinks)
