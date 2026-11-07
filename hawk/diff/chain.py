# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative of a chain (:func:`hawk.chain`) is the chain of its members' derivatives.

``jvp(chain)``: each member's tangent kernel, the tangent of a value an earlier member made read under its
``dot_`` name. ``vjp(chain)``: the members before the last, then the members' adjoint kernels in reverse order; a
primal value an adjoint needs is a chain value, and the adjoints several readers send to one name add. The
result keeps its ``members``, so the fusion rule cuts the derivative chain as it cuts any chain.
"""

from __future__ import annotations

from ..ir import canonical_nodes
from ..ir import make as _mk
from ..ir.chain import _walk_sinks, bind
from ..ir.nodes import Assign, Leaf
from .transform import _DIFFERENTIABLE, Derived, _jvp, _sinks_of, _wrt, vjp


def _inputs(kernel, names) -> list:
    """The differentiable inputs of ``kernel`` (in order) that are among ``names``."""
    leaves = _walk_sinks(kernel.sinks)[0]
    return list(dict.fromkeys(lf.name for lf in leaves if lf.ttype.dtype in _DIFFERENTIABLE and lf.name in names))


def _member(kernel, sinks, label):
    from ..trace.kernel import Kernel

    out = Kernel(f"{kernel.name}_{label}", tuple(sinks), kind=kernel.kind)
    out.lane_space, out.group, out.jump = kernel.lane_space, kernel.group, kernel.jump
    return out


def _prune(kernel, own, needed: set, label, derived=()):
    """The member of ``kernel`` that keeps its ``own`` sinks read in ``needed``, and ``derived`` in full."""
    kept = [s for s in own if s.name in needed] + list(derived)
    needed.update(lf.name for lf in _walk_sinks(kept)[0])
    return _member(kernel, kept, label) if kept else None


def _derived(chain, members, kind, wrt, primal_unit, jump=None) -> Derived:
    out = Derived(bind(members).sinks, primal=chain.name, kind=kind, wrt=wrt, primal_unit=primal_unit, jump=jump)
    out.members = tuple(members)
    return out


def _resolve(chain, wrt) -> tuple:
    return tuple(dict.fromkeys(lf.name for lf in _wrt(canonical_nodes(chain.sinks)[0], wrt)))


def jvp_chain(chain, wrt, primal_unit) -> Derived:
    names = _resolve(chain, wrt)
    tangent_of, tangents, jump = set(names), [], None
    for k in chain.members:
        w = _inputs(k, tangent_of)
        d = _jvp(k, wrt=w, primal_unit=primal_unit) if w else ()
        jump = jump or getattr(d, "jump", None)
        tangents.append(tuple(d))
        tangent_of.update(s.name for s in _sinks_of(k) if s.ttype.dtype in _DIFFERENTIABLE and d)
    needed, members = set(), []
    for k, d in zip(reversed(chain.members), reversed(tangents)):
        m = _prune(k, _sinks_of(k), needed, "jvp", d)
        if m is not None:
            members.append(m)
    return _derived(chain, members[::-1], "jvp", names, primal_unit, jump)


def vjp_chain(chain, wrt, primal_unit, policy) -> Derived:
    names = _resolve(chain, wrt)
    made, reach = [], set(names)               # the differentiable inputs of each member that carry an adjoint
    for k in chain.members:
        made.append(_inputs(k, reach))
        reach.update(s.name for s in _sinks_of(k))
    written, adjoints, jump = set(), [], None
    for k, w in zip(reversed(chain.members), reversed(made)):
        if not w:
            continue
        d = vjp(k, wrt=w, primal_unit=primal_unit, policy=policy)
        jump = jump or d.jump
        sinks = []
        for s in d:
            if isinstance(s, Assign) and s.name in written:         # a second reader of the value: the adjoints add
                held = Leaf("prior_read", "mutable", s.name, s.ttype)
                s = Assign(s.name, _mk("add", (s.value, held)), s.ttype)
            sinks.append(s)
        written.update(s.name for s in sinks)
        adjoints.append(_member(k, sinks, "vjp"))
    needed = {lf.name for m in adjoints for lf in _walk_sinks(m.sinks)[0]}
    forward = []
    for k in reversed(chain.members[:-1]):
        m = _prune(k, _sinks_of(k), needed, "fwd")
        if m is not None:
            forward.append(m)
    return _derived(chain, [*forward[::-1], *adjoints], "vjp", names, primal_unit, jump)
