# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The extension protocol: the seams a kind DECLARES.

Each seam is a declaration consumed by the walk and the emitter, never a
post-hoc edit of generated text. This module lands two seams:

* **Sink policy** — how an accumulate sink commits. Built: ``plain``,
  ``atomic`` and ``compensated`` (Neumaier, correction committed to a
  companion plane the declaration names). A scattered accumulate commits
  through the target's atomic add under ``plain``/``atomic`` alike
  (``hawk_abi::accum_add``) — a scattered read-modify-write has no correct
  use under any parallel launch; ``atomic`` just names that commit
  explicitly. An own-column write is a store under ``plain``/``atomic``,
  but an accumulate into the plane's prior value under ``compensated``
  (see :func:`compensated` for the host-side zeroing contract). ``banded``
  is a named, unbuilt slot that refuses with its own message.
* **Guard policy** — which mask gates the sample, and whether one is
  present at all. A set mask skips the sample's whole body. The default
  gates on a bound ``terminated`` slot; ``Guard(mask=None)`` is the
  data-only seam, for a kind that must run every sample.
  ``Guard(active_set=True)`` additionally reads an active-set index map
  (eagle's ``ActiveSet``): the index prologue maps launch position ``t``
  to sample ``active_map[t]`` and stops at ``active_count``, so live
  samples share warps; the body is unchanged.

The custom-primitive seam (:mod:`hawk.ext.primitive`) adds a forward the
author writes as an ordinary Python function of traced values, inlined at
every call, plus the VJP/JVP rules that replace the table's rule for that
subgraph.

A :class:`Kind` carries all of these plus its compile hook. A vocabulary is
extended by INHERITANCE: a :class:`KernelKind` subclass, or :meth:`Kind.extend`
for the instance form, carries a base's vocabulary and seam fields forward,
merging rather than shadowing — see :class:`KernelKind` for the rules.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..ir import HawkError
from ..types import TensorType
from .primitive import PrimitiveDef, primitive, primitives

__all__ = ["Guard", "Kind", "Output", "PLAIN", "ATOMIC", "DEFAULT_GUARD",
           "sink_policy", "compensated", "PrimitiveDef", "primitive", "primitives"]


@dataclass(frozen=True)
class SinkPolicy:
    """How a sink commits, plus the accumulate target's companion role.
    ``id`` is the declared vocabulary value; ``compensation`` names the
    ``accum_out`` slot a ``compensated`` commit writes its correction to."""

    id: str
    compensation: str | None = None


#: The default. An own-column accumulate is a store; a scattered one
#: commits through the target's atomic add (``hawk_abi::accum_add``, see
#: the module docstring).
PLAIN = SinkPolicy("plain")
#: The scattered commit ``plain`` already makes, named out loud.
ATOMIC = SinkPolicy("atomic")

SINK_POLICIES = {"plain": PLAIN, "atomic": ATOMIC}

#: The seam's NAMED unbuilt slot (the shape, applied to).
_UNBUILT_SINKS = {
    "banded": ("emulated double precision",
               "aether/banded/ (Band / BandCell8) and aether/accum/AccumPlane.h"),
}


def compensated(into: str) -> SinkPolicy:
    """A Neumaier-compensated accumulate whose correction lands in ``into``.
    The total is always ``target + into``, but what's committed depends
    on the kernel's output: a scattered write commits through the
    target's atomic add, with ``into`` an ``accum_out`` plane the kernel
    declares beside it. An own-column write stops being a store and
    becomes an accumulate into the plane's prior value — each launch adds
    one more term on top of whatever it already held. The host must zero
    both the target and ``into`` before the first launch, or a leftover
    value silently adds onto this run's answer. ``into`` is a ``Mutable``
    the kind synthesises; never declare it yourself."""
    if not into:
        raise HawkError(
            "compensated(): a compensated commit needs the NAME of the companion "
            "accum plane its correction term is committed to; declare it beside "
            "the target"
        )
    return SinkPolicy("compensated", into)


def sink_policy(name: str, **kw) -> SinkPolicy:
    """Resolve an value by name. ``banded`` names what builds it."""
    if name == "compensated":
        return compensated(kw.get("into", ""))
    if name in SINK_POLICIES:
        return SINK_POLICIES[name]
    if name in _UNBUILT_SINKS:
        description, why = _UNBUILT_SINKS[name]
        raise HawkError(
            f"sink policy {name!r} is a reserved sink policy name and is not built "
            f"yet ({description}); it will live in {why}. "
            f"Built policies are {('plain', 'atomic', 'compensated')}"
        )
    raise HawkError(
        f"unknown sink policy {name!r}; the known policies are "
        f"{tuple(SINK_POLICIES) + ('compensated',) + tuple(_UNBUILT_SINKS)}"
    )


@dataclass(frozen=True)
class Guard:
    """Which masks gate the sample. ``mask`` is one bound slot name, or
    ``None`` for the data-only seam. ``masks`` names several (e.g.
    ``("terminated", "rejected")``): a sample runs only when none of them
    is set. Each name is a slot the kernel declares with the
    ``Terminated`` form.

    ``active_set=True`` makes the kernel read an active-set index map: the
    build binds ``active_map`` (``i32``, ascending live indices) and
    ``active_count`` (``i32``), and the index prologue maps launch
    position ``t`` to sample ``active_map[t]``, exiting past
    ``active_count[0]`` (:class:`eagle.ActiveSet` fills both). Only the
    prologue changes, so a stale map is safe. It needs a mask, so
    ``Guard(mask=None, active_set=True)`` is refused; a kernel that
    accumulates or reduces across samples, and a segmented unit, are
    refused too (the map would reorder the sum, or the unit already owns
    the prologue)."""

    mask: str | None = "terminated"
    masks: tuple[str, ...] = ()
    active_set: bool = False

    def __repr__(self) -> str:
        # The pre-active-set spelling for every guard without it, so a kind's
        # repr (part of a unit's key) is unchanged for every existing kernel.
        text = f"Guard(mask={self.mask!r}, masks={self.masks!r}"
        if self.active_set:
            text += ", active_set=True"
        return text + ")"

    def __post_init__(self):
        if not isinstance(self.active_set, bool):
            raise HawkError(f"Guard(active_set=...) is a bool, got {self.active_set!r}")
        if self.active_set and not self.names:
            raise HawkError(
                "Guard(mask=None, active_set=True): an active-set index map is "
                "derived from a mask, and the data-only seam (no mask) must run "
                "every sample — name the mask the map follows"
            )
        if isinstance(self.masks, str):
            raise HawkError(f"Guard(masks=...) takes a tuple of slot names, got the "
                            f"string {self.masks!r}; write masks=({self.masks!r},)")
        if len(set(self.masks)) != len(self.masks) or not all(self.masks):
            raise HawkError(f"Guard(masks=...) names each mask once, non-empty: "
                            f"{self.masks!r}")
        if self.masks and self.mask not in (None, "terminated", *self.masks):
            raise HawkError(f"Guard: mask={self.mask!r} is not among masks="
                            f"{self.masks!r}; name every mask in `masks`")

    @property
    def names(self) -> tuple[str, ...]:
        """Every mask this guard gates on, in declaration order."""
        if self.masks:
            return self.masks
        return () if self.mask is None else (self.mask,)


#: What a kind declaring no guard gets: gate on a bound ``terminated``
#: slot. A kernel binding none is unguarded, trivially.
DEFAULT_GUARD = Guard("terminated")
#: The mask-free (data-only) seam value.
DATA_ONLY = Guard(None)


@dataclass(frozen=True)
class Output:
    """Which sink parameter(s) a kind's kernel treats as its output set —
    the set a ``compensated`` sink's target is resolved from
    (:func:`compensated_target`).

    ``Output.named()`` (no names) is today's behaviour: every sink
    parameter the body commits. ``Output.named(*names)`` restricts the set
    to the named parameters. ``Output.returned(ttype, slot=...)`` has the
    body end ``return <expr>`` instead of committing a declared sink, into
    a synthesised ``Mutable`` slot named ``slot``. Ordinary sink
    parameters may still be declared beside the ``return``, committing in
    parameter order ahead of the synthesised slot::

        ACCEL = Kind("drag", output=Output.returned(Vector[3], slot="acc"))

        @ACCEL
        def drag(velocity: Vector[3], speed: Mutable[Scalar]):
            speed = norm(velocity)          # a named sink, commits first
            return (-k * norm(velocity)) * velocity  # then the returned slot

    A ``compensated`` sink's target then resolves from the ``returned``
    slot alone."""

    names: tuple[str, ...] = ()
    #: The synthesised slot's ttype, or ``None`` for the ordinary
    #: ``Output.named`` form. Named ``returned_ttype`` rather than
    #: ``returned`` — the FIELD may not share the classmethod's own name, or
    #: the method definition below would overwrite the field's default.
    returned_ttype: TensorType | None = None
    slot: str | None = None

    @classmethod
    def named(cls, *names: str) -> Output:
        """Every name given is a Mutable/Accum/Reduce/WideOut parameter the
        body commits; no names (the default) means every sink parameter."""
        return cls(names=tuple(names))

    @classmethod
    def returned(cls, ttype: TensorType, *, slot: str) -> Output:
        """The body ends ``return <expr>``; ``slot`` names the synthesised
        Mutable the returned value commits to."""
        if not slot:
            raise HawkError(
                "Output.returned(): `slot` names the synthesised Mutable the "
                "returned value commits to — it cannot be empty"
            )
        return cls(returned_ttype=ttype, slot=slot)


@dataclass(frozen=True)
class Kind:
    """A kind's class-level vocabulary, its output set, its declared seams
    and its compile hook.

    ``vocabulary`` maps a parameter name to a declaration (anything
    :func:`hawk.trace.decl.resolve` accepts) — a declaration binds a slot
    whether or not a kernel's own signature names it. Every value is
    resolved here, at construction, so a bad one refuses before the first
    kernel ever sees it. Calling a ``Kind`` instance is the decorator
    form: ``@Kind(...)`` == ``hawk.kernel(fn, kind=that Kind)``.

    ``compile_hook(options) -> options`` returns the
    :class:`hawk.compile.CompileOptions` to use."""

    slug: str
    vocabulary: Mapping[str, Any] | tuple = field(default_factory=dict)
    output: Output = field(default_factory=Output.named)
    sink: SinkPolicy = PLAIN
    guard: Guard = DEFAULT_GUARD
    quantities: Sequence[object] = field(default=())
    compile_hook: Callable | None = None

    def __post_init__(self) -> None:
        from ..trace.decl import Plane, resolve

        resolved: dict[str, Any] = {}
        for name, decl in dict(self.vocabulary).items():
            try:
                resolved[name] = resolve(name, decl)
            except HawkError as exc:
                raise HawkError(
                    f"Kind({self.slug!r}): vocabulary entry {name!r} is not a "
                    f"valid declaration ({exc})"
                ) from exc
        object.__setattr__(self, "vocabulary", tuple(resolved.items()))
        if resolved:
            bad = [m for m in self.guard.names
                   if not (isinstance(resolved.get(m), Plane)
                           and resolved[m].form == "terminated")]
            if bad:
                raise HawkError(
                    f"Kind({self.slug!r}): guard mask(s) {bad} are not a "
                    f"`terminated`-form entry of this kind's own vocabulary "
                    f"({sorted(resolved)}) — a guard may only gate on a mask "
                    "the vocabulary itself declares"
                )

    def __call__(self, fn):
        """The decorator form: ``@Kind(...)`` == ``kernel(fn, kind=self)``."""
        from ..trace.kernel import kernel

        return kernel(fn, kind=self)

    def apply(self, options):
        """Run the compile hook, if this kind declares one."""
        return options if self.compile_hook is None else self.compile_hook(options)

    def extend(self, slug: str, *, vocabulary: Mapping[str, Any] | None = None,
              **seams: Any) -> Kind:
        """The instance-form spelling of :class:`KernelKind` inheritance: a new
        ``Kind`` that carries THIS kind's vocabulary and seam fields forward.

        ``vocabulary``'s entries are merged into this kind's own: an equal
        RESOLVED re-declaration of an entry this kind already carries is fine
        (idempotent), a differing one refuses naming the entry, this kind's
        slug, and both declarations. ``seams`` overrides this kind's own seam
        fields (``output``/``sink``/``guard``/``quantities``/``compile_hook``);
        one left out keeps this kind's own value, never the bare ``Kind``
        default. This is the SAME merge :meth:`KernelKind.__init_subclass__`
        runs for the class form — ``base_kind.extend(...)`` and
        ``class X(SomeClassFormOfBaseKind, ...): ...`` produce the same
        ``Kind`` for the same inputs, one object, two spellings."""
        merged_vocabulary = _merge_vocabulary(
            [(self.slug, dict(self.vocabulary)), (slug, dict(vocabulary or {}))],
            owner=f"Kind({self.slug!r}).extend({slug!r})")
        merged_seams = {name: getattr(self, name) for name in _KIND_SEAM_FIELDS}
        merged_seams.update(seams)
        return Kind(slug, vocabulary=merged_vocabulary, **merged_seams)


#: The kind every kernel emitted without one gets: plain commits, the
#: default mask guard, no compile hook, no vocabulary.
DEFAULT_KIND = Kind("default")
__all__.append("DEFAULT_KIND")


def _merge_vocabulary(layers: Sequence[tuple[str, Mapping[str, Any]]],
                      *, owner: str) -> dict[str, Any]:
    """Merge vocabulary layers in priority order (most-base first): a name
    repeated across layers must resolve to an EQUAL declaration, or the merge
    refuses naming the entry, the layer that first declared it, and both
    (differing) resolved declarations. Returns resolved (``Plane``/``Quantity``)
    values, ready for :class:`Kind` (whose own ``__post_init__`` resolves them
    again — :func:`~hawk.trace.decl.resolve` is idempotent on an already-
    resolved value, see there)."""
    from ..trace.decl import resolve as _resolve_decl

    merged: dict[str, Any] = {}
    declared_by: dict[str, str] = {}
    for label, raw in layers:
        for name, value in dict(raw).items():
            decl = _resolve_decl(name, value)
            if name in merged:
                if merged[name] != decl:
                    raise HawkError(
                        f"{owner}: vocabulary entry {name!r} is declared "
                        f"differently by {declared_by[name]!r} "
                        f"({merged[name]!r}) and {label!r} ({decl!r}) — an "
                        "inherited entry may only be re-declared with an "
                        "EQUAL resolved declaration"
                    )
                continue
            merged[name] = decl
            declared_by[name] = label
    return merged


#: Every :class:`Kind` field the class form's body may set directly
#: (``output = ...``), by plain assignment — every field except ``slug``
#: (the class-statement keyword) and ``vocabulary`` (built from the class
#: body's own annotations instead).
_KIND_SEAM_FIELDS = ("output", "sink", "guard", "quantities", "compile_hook")


def _class_scope() -> dict[str, Any]:
    """The globals + locals of the frame that ran the ``class X(KernelKind,
    ...):`` statement — the scope a string annotation resolves in. A class
    statement is never wrapped the way a decorated function can be, so the
    caller's immediate frame IS that scope."""
    frame = sys._getframe(2)  # this function's caller's caller: __init_subclass__
    return {**frame.f_globals, **frame.f_locals}


def _resolve_annotation(cls: type, name: str, ann: Any, scope: dict[str, Any]) -> Any:
    """A class-body annotation (a real declaration, or a string one under
    ``from __future__ import annotations``) resolved to a vocabulary value."""
    if not isinstance(ann, str):
        return ann
    try:
        return eval(ann, scope)  # noqa: - the author's own annotation
    except Exception as exc:                      # pragma: no cover - author error
        raise HawkError(
            f"class {cls.__name__!r}(KernelKind, ...): annotation {name!r}={ann!r} "
            f"does not resolve in the class statement's own scope ({exc})"
        ) from exc


class KernelKind:
    """The class-form spelling of :class:`Kind`: sugar, not a second kind.

    ``class X(KernelKind, slug=...): ...`` builds the same ``Kind`` object
    an equivalent instance-form call would, and ``@X`` on a kernel
    function is exactly ``hawk.kernel(fn, kind=X.kind)``: one object, two
    spellings, no second code path downstream.

    A ``KernelKind`` subclass INHERITS the vocabulary entries and seam
    fields (``output``/``sink``/``guard``/``quantities``/``compile_hook``)
    of every ``KernelKind`` base in its MRO, most-base first, so a
    vocabulary is extended the ordinary Python way — subclass it — and
    the extended kind works end to end like any other: trace, compile,
    run on host and device, derivatives. Re-declaring an inherited
    vocabulary entry is fine with an EQUAL resolved declaration; a
    differing one refuses, naming the entry, the base, and both
    declarations — a diamond where two bases disagree on the same entry
    refuses the same way. A class body's own seam-field assignment
    overrides every base's; one left unset keeps whatever the nearest
    base that DID set it carries, never silently reverting to the bare
    ``Kind`` default. The resulting ``cls.kind`` is a plain ``Kind`` — one
    object, no second code path, whatever the depth of the hierarchy.

    The class body is read at class-creation time
    (``__init_subclass__``), so a bad one refuses before the first kernel
    ever sees it here too. Every class attribute must be either an
    annotation carrying no value (a vocabulary entry) or a plain
    assignment naming one of :class:`Kind`'s own remaining fields
    (:data:`_KIND_SEAM_FIELDS`); anything else refuses by name — the
    class form is sugar over :class:`Kind`, never a second, arbitrary
    namespace an author can add code to."""

    def __init_subclass__(cls, *, slug: str | None = None, **kwargs) -> None:
        if kwargs:
            raise HawkError(
                f"class {cls.__name__!r}(KernelKind, ...): unknown class-"
                f"statement keyword(s) {sorted(kwargs)} — only slug=... is "
                "accepted here"
            )
        super().__init_subclass__()
        if not slug:
            raise HawkError(
                f"class {cls.__name__!r}(KernelKind, slug=...): the class "
                "form needs slug=..., the same NAME the instance form's "
                "Kind(slug=...) takes"
            )
        own_annotations = cls.__dict__.get("__annotations__", {})
        scope = _class_scope()
        own_vocabulary = {name: _resolve_annotation(cls, name, ann, scope)
                         for name, ann in own_annotations.items()
                         if not name.startswith("_")}
        own_seams: dict[str, Any] = {}
        for name, value in cls.__dict__.items():
            if name.startswith("_") or name in own_vocabulary:
                continue
            if name not in _KIND_SEAM_FIELDS:
                raise HawkError(
                    f"class {cls.__name__!r}(KernelKind, ...): class attribute "
                    f"{name!r} is neither an ANNOTATION (a vocabulary entry) "
                    f"nor one of Kind's own fields ({', '.join(_KIND_SEAM_FIELDS)}) "
                    "— the class form is sugar over Kind, never a second "
                    "code path an author can add to"
                )
            own_seams[name] = value

        # Every proper KernelKind ancestor's OWN declarations (never an
        # already-inherited copy — `__dict__` holds only what THAT class's
        # own body wrote), most-base first. Python's own C3 MRO already
        # resolves a diamond to one deterministic order, each ancestor
        # appearing exactly once.
        bases = [c for c in reversed(cls.__mro__[1:]) if "_own_vocabulary" in c.__dict__]
        layers = [(base.__name__, base.__dict__["_own_vocabulary"]) for base in bases]
        layers.append((cls.__name__, own_vocabulary))
        merged_vocabulary = _merge_vocabulary(
            layers, owner=f"class {cls.__name__!r}(KernelKind, ...)")
        merged_seams: dict[str, Any] = {}
        for base in bases:
            merged_seams.update(base.__dict__["_own_seams"])
        merged_seams.update(own_seams)

        cls._own_vocabulary = own_vocabulary
        cls._own_seams = own_seams
        cls.kind = Kind(slug, vocabulary=merged_vocabulary, **merged_seams)

    def __new__(cls, fn):
        """``@X`` / ``X(fn)`` == ``kernel(fn, kind=X.kind)`` — the SAME
        decorator :meth:`Kind.__call__` gives the instance form."""
        return cls.kind(fn)


__all__.append("KernelKind")


def compensated_target(kind: Kind, sinks: Sequence[Any]) -> tuple[str, bool] | None:
    """The compensated sink's resolved target, out of ``kind.output`` and
    the kernel's own committed ``sinks`` — shared by the tracer and the
    renderer so the two never resolve a different target for the same
    kernel.

    ``None`` when this kind's sink policy is not ``compensated``.
    Otherwise ``(name, own_column)``: ``own_column`` is ``True`` for an
    ``Assign`` or an ``AccumWrite`` with ``index=None``, ``False`` for a
    scattered ``AccumWrite``. The companion (``kind.sink.compensation``)
    is excluded from the candidate pool before counting."""
    if getattr(kind.sink, "id", None) != "compensated":
        return None
    from ..ir.nodes import AccumWrite, Assign, MapreducePartial, WideWrite

    comp = kind.sink.compensation
    all_names = tuple(dict.fromkeys(s.name for s in sinks))
    pool = (
        (kind.output.slot,) if kind.output.returned_ttype is not None
        else kind.output.names if kind.output.names else all_names
    )
    candidates = [n for n in pool if n != comp]
    if len(candidates) != 1:
        source = ("Output.named" + repr(tuple(pool)) if kind.output.names
                 else "the committed sinks " + repr(pool))
        raise HawkError(
            f"Kind({kind.slug!r}): a compensated sink needs exactly ONE output "
            f"target (excluding the companion {comp!r}); resolved {candidates} "
            f"from {source}"
        )
    name = candidates[0]
    matches = [s for s in sinks if s.name == name]
    bad = [s for s in matches if isinstance(s, (MapreducePartial, WideWrite))]
    if bad:
        kind_name = "Reduce" if isinstance(bad[0], MapreducePartial) else "WideOut"
        raise HawkError(
            f"Kind({kind.slug!r}): compensated target {name!r} is a {kind_name} "
            "sink; a compensated commit targets a Mutable or an Accum only"
        )
    own_column = any(isinstance(s, Assign) for s in matches) or any(
        isinstance(s, AccumWrite) and s.index is None for s in matches)
    return name, own_column
