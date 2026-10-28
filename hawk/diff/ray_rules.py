# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivative rules of the ray kinds (:mod:`hawk.ir.rays`).

Closed forms built from hawk ops over the node's own operands, with the SAME
guarded denominators as the primitive (``aether/geometry/Ray.h``): every division
and square root sees a non-zero operand before the select, so a degenerate input
(a ray parallel to the triangle, a zero-area triangle, a zero direction, a
negative discriminant) gives a derivative of exactly zero, never ``0 * inf``. The
primitive's body is never differentiated.

``hit`` (entry 0) has zero derivative and the cotangent ``g[0]`` is ignored. The
values are the computed ones also on a miss, and the derivative follows them
(through the live arm of every ``min``/``max`` of the slab and the live root of the
sphere); there is no silhouette term, so the jump of ``t`` at a triangle edge or a
sphere's rim is not modelled. A slab with a zero direction component has a constant
interval end, so its derivative with respect to the origin and the box is zero.
``tmax`` only enters the hit test, so it has no derivative either.
"""

from __future__ import annotations

from collections.abc import Callable

from ..ir import Const, Node, Select
from ..ir import make as _mk
from ..types import TensorType

Tangent = Callable[[Node], "Node | None"]

_F64 = TensorType((), "f64")
_BIG = 1e18


def _o(kind: str, *args: Node, literal=None) -> Node:
    return _mk(kind, args, literal)


def _c(x: float) -> Node:
    return Const(float(x), _F64)


def _add(*xs: Node) -> Node:
    out = xs[0]
    for x in xs[1:]:
        out = _o("add", out, x)
    return out


def _sub(a: Node, b: Node) -> Node:
    return _o("sub", a, b)


def _mul(*xs: Node) -> Node:
    out = xs[0]
    for x in xs[1:]:
        out = _o("mul", out, x)
    return out


def _neg(a: Node) -> Node:
    return _o("neg", a)


def _dot(a: Node, b: Node) -> Node:
    return _o("dot", a, b)


def _cross(a: Node, b: Node) -> Node:
    return _o("cross", a, b)


def _k(v: Node, k: int) -> Node:
    return _o("component", v, literal=k)


def _vec(*xs: Node) -> Node:
    return _o("vec", *xs)


def _where(c: Node, a: Node, b: Node) -> Node:
    return Select(c, a, b, a.ttype)


def _zero(t: TensorType) -> Node:
    z = _c(0.0)
    return z if not t.shape else _o("splat", z, literal=t.shape)


def _live(t: Tangent, x: Node) -> Node:
    """``x``'s tangent, a structural zero when it has none."""
    d = t(x)
    return _zero(x.ttype) if d is None else d


def _guard(x: Node, ok: Node, fill: float = 1.0) -> Node:
    """``x`` where ``ok``, else ``fill``: the denominator the division sees."""
    return _where(ok, x, _c(fill))


def _inverse(x: Node, ok: Node) -> Node:
    """``1 / x`` where ``ok``, else 0."""
    return _where(ok, _o("div", _c(1.0), _guard(x, ok)), _c(0.0))


def _result(hit: Node, a: Node, b: Node, c: Node) -> Node:
    return _vec(hit, a, b, c)


def _from_cotangent(g: Node) -> tuple:
    return _k(g, 1), _k(g, 2), _k(g, 3)


# --------------------------------------------------------------------------
# triangle

def _tri_parts(n: Node) -> dict:
    o, d, a, b, c = n.operands[:5]
    e1, e2 = _sub(b, a), _sub(c, a)
    p = _cross(d, e2)
    det = _dot(e1, p)
    ok = _o("ne", det, _c(0.0))
    inv = _inverse(det, ok)
    s = _sub(o, a)
    q = _cross(s, e1)
    return dict(o=o, d=d, a=a, b=b, c=c, e1=e1, e2=e2, p=p, inv=inv, s=s, q=q,
                nu=_dot(s, p), nv=_dot(d, q), nt=_dot(e2, q))


def _ray_tri_jvp(n: Node, t: Tangent) -> Node | None:
    x = _tri_parts(n)
    d_o, d_d, d_a, d_b, d_c = (_live(t, x[k]) for k in "odabc")
    de1, de2, ds = _sub(d_b, d_a), _sub(d_c, d_a), _sub(d_o, d_a)
    dp = _add(_cross(d_d, x["e2"]), _cross(x["d"], de2))
    ddet = _add(_dot(de1, x["p"]), _dot(x["e1"], dp))
    dq = _add(_cross(ds, x["e1"]), _cross(x["s"], de1))
    dnu = _add(_dot(ds, x["p"]), _dot(x["s"], dp))
    dnv = _add(_dot(d_d, x["q"]), _dot(x["d"], dq))
    dnt = _add(_dot(de2, x["q"]), _dot(x["e2"], dq))
    inv = x["inv"]
    dinv = _neg(_mul(inv, inv, ddet))
    out = [_add(_mul(dinv, x[nn]), _mul(inv, dn)) for nn, dn in
           (("nt", dnt), ("nu", dnu), ("nv", dnv))]
    return _result(_c(0.0), *out)


def _ray_tri_vjp(n: Node, g: Node) -> tuple:
    x = _tri_parts(n)
    gt, gu, gv = _from_cotangent(g)
    inv = x["inv"]
    gnt, gnu, gnv = _mul(gt, inv), _mul(gu, inv), _mul(gv, inv)
    ginv = _add(_mul(gt, x["nt"]), _mul(gu, x["nu"]), _mul(gv, x["nv"]))
    gdet = _neg(_mul(inv, inv, ginv))
    # nu = s.p   nv = d.q   nt = e2.q   det = e1.p   q = s x e1   p = d x e2
    gq = _add(_mul(gnv, x["d"]), _mul(gnt, x["e2"]))
    gp = _add(_mul(gnu, x["s"]), _mul(gdet, x["e1"]))
    gs = _add(_mul(gnu, x["p"]), _cross(x["e1"], gq))
    ge1 = _add(_mul(gdet, x["p"]), _cross(gq, x["s"]))
    gd = _add(_mul(gnv, x["q"]), _cross(x["e2"], gp))
    ge2 = _add(_mul(gnt, x["q"]), _cross(gp, x["d"]))
    return ((x["o"], gs), (x["d"], gd), (x["a"], _neg(_add(gs, ge1, ge2))),
            (x["b"], ge1), (x["c"], ge2))


# --------------------------------------------------------------------------
# box

def _aabb_axes(n: Node) -> dict:
    o, d, lo, hi = n.operands[:4]
    axes = []
    for k in range(3):
        dk = _k(d, k)
        ok = _o("ne", dk, _c(0.0))
        inv = _where(ok, _o("div", _c(1.0), _guard(dk, ok)),
                     _o("copysign", _c(_BIG), dk))
        gap_lo, gap_hi = _sub(_k(lo, k), _k(o, k)), _sub(_k(hi, k), _k(o, k))
        a, b = _mul(gap_lo, inv), _mul(gap_hi, inv)
        lt = _o("lt", a, b)
        axes.append(dict(ok=ok, inv=inv, gap_lo=gap_lo, gap_hi=gap_hi, lt=lt,
                         n=_where(lt, a, b), f=_where(lt, b, a)))
    c_in = [_o("gt", axes[1]["n"], axes[0]["n"])]
    tin1 = _where(c_in[0], axes[1]["n"], axes[0]["n"])
    c_in.append(_o("gt", axes[2]["n"], tin1))
    c_out = [_o("lt", axes[1]["f"], axes[0]["f"])]
    tout1 = _where(c_out[0], axes[1]["f"], axes[0]["f"])
    c_out.append(_o("lt", axes[2]["f"], tout1))
    return dict(o=o, d=d, lo=lo, hi=hi, axes=axes, c_in=c_in, c_out=c_out)


def _ray_aabb_jvp(n: Node, t: Tangent) -> Node | None:
    x = _aabb_axes(n)
    d_o, d_d, d_lo, d_hi = (_live(t, x[k]) for k in ("o", "d", "lo", "hi"))
    dn, df = [], []
    for k, ax in enumerate(x["axes"]):
        inv_live = _where(ax["ok"], ax["inv"], _c(0.0))
        dinv = _where(ax["ok"], _neg(_mul(ax["inv"], ax["inv"], _k(d_d, k))), _c(0.0))
        da = _add(_mul(_sub(_k(d_lo, k), _k(d_o, k)), inv_live), _mul(ax["gap_lo"], dinv))
        db = _add(_mul(_sub(_k(d_hi, k), _k(d_o, k)), inv_live), _mul(ax["gap_hi"], dinv))
        dn.append(_where(ax["lt"], da, db))
        df.append(_where(ax["lt"], db, da))
    dtin1 = _where(x["c_in"][0], dn[1], dn[0])
    dtout1 = _where(x["c_out"][0], df[1], df[0])
    return _result(_c(0.0), _where(x["c_in"][1], dn[2], dtin1),
                   _where(x["c_out"][1], df[2], dtout1), _c(0.0))


def _ray_aabb_vjp(n: Node, g: Node) -> tuple:
    x = _aabb_axes(n)
    zero = _c(0.0)
    g_in, g_out = _k(g, 1), _k(g, 2)
    c_in, c_out = x["c_in"], x["c_out"]
    gn2, gtin1 = _where(c_in[1], g_in, zero), _where(c_in[1], zero, g_in)
    gn1, gn0 = _where(c_in[0], gtin1, zero), _where(c_in[0], zero, gtin1)
    gf2, gtout1 = _where(c_out[1], g_out, zero), _where(c_out[1], zero, g_out)
    gf1, gf0 = _where(c_out[0], gtout1, zero), _where(c_out[0], zero, gtout1)
    g_o, g_d, g_lo, g_hi = [], [], [], []
    for k, (ax, gn, gf) in enumerate(zip(x["axes"], (gn0, gn1, gn2), (gf0, gf1, gf2))):
        ga = _add(_where(ax["lt"], gn, zero), _where(ax["lt"], zero, gf))
        gb = _add(_where(ax["lt"], zero, gn), _where(ax["lt"], gf, zero))
        inv = _where(ax["ok"], ax["inv"], zero)
        g_lo.append(_mul(ga, inv))
        g_hi.append(_mul(gb, inv))
        g_o.append(_neg(_mul(_add(ga, gb), inv)))
        ginv = _add(_mul(ga, ax["gap_lo"]), _mul(gb, ax["gap_hi"]))
        g_d.append(_where(ax["ok"], _neg(_mul(ax["inv"], ax["inv"], ginv)), zero))
    return ((x["o"], _vec(*g_o)), (x["d"], _vec(*g_d)),
            (x["lo"], _vec(*g_lo)), (x["hi"], _vec(*g_hi)))


# --------------------------------------------------------------------------
# sphere

def _sphere_parts(n: Node) -> dict:
    o, d, c, r = n.operands[:4]
    oc = _sub(o, c)
    qa, hb = _dot(d, d), _dot(oc, d)
    qc = _sub(_dot(oc, oc), _mul(r, r))
    disc = _sub(_mul(hb, hb), _mul(qa, qc))
    pos = _o("gt", qa, _c(0.0))
    ok = _o("land", pos, _o("ge", disc, _c(0.0)))
    sq = _o("sqrt", _guard(disc, ok))
    sq_pos = _o("gt", sq, _c(0.0))
    return dict(o=o, d=d, c=c, r=r, oc=oc, qa=qa, hb=hb, qc=qc, ok=ok, sq=sq,
                sq_pos=sq_pos, sq2=_guard(_mul(_c(2.0), sq), sq_pos),
                inv=_o("div", _c(1.0), _guard(qa, pos)))


def _ray_sphere_jvp(n: Node, t: Tangent) -> Node | None:
    x = _sphere_parts(n)
    d_o, d_d, d_c, d_r = (_live(t, x[k]) for k in ("o", "d", "c", "r"))
    d_oc = _sub(d_o, d_c)
    dqa = _mul(_c(2.0), _dot(x["d"], d_d))
    dhb = _add(_dot(d_oc, x["d"]), _dot(x["oc"], d_d))
    dqc = _sub(_mul(_c(2.0), _dot(x["oc"], d_oc)), _mul(_c(2.0), x["r"], d_r))
    ddisc = _sub(_sub(_mul(_c(2.0), x["hb"], dhb), _mul(dqa, x["qc"])), _mul(x["qa"], dqc))
    dsq = _where(x["sq_pos"], _o("div", ddisc, x["sq2"]), _c(0.0))
    inv = x["inv"]
    dinv = _neg(_mul(inv, inv, dqa))
    zero = _c(0.0)
    roots = []
    for sign in (-1.0, 1.0):
        lead = _add(_neg(x["hb"]), _mul(_c(sign), x["sq"]))
        slope = _add(_neg(dhb), _mul(_c(sign), dsq))
        roots.append(_where(x["ok"], _add(_mul(slope, inv), _mul(lead, dinv)), zero))
    return _result(zero, roots[0], roots[1], zero)


def _ray_sphere_vjp(n: Node, g: Node) -> tuple:
    x = _sphere_parts(n)
    zero = _c(0.0)
    gt0, gt1 = (_where(x["ok"], gi, zero) for gi in (_k(g, 1), _k(g, 2)))
    inv = x["inv"]
    ginv = _add(_mul(gt0, _sub(_neg(x["hb"]), x["sq"])), _mul(gt1, _add(_neg(x["hb"]), x["sq"])))
    ghb = _neg(_mul(_add(gt0, gt1), inv))
    gsq = _mul(_sub(gt1, gt0), inv)
    gqa = _neg(_mul(inv, inv, ginv))
    gdisc = _where(x["sq_pos"], _o("div", gsq, x["sq2"]), zero)
    ghb = _add(ghb, _mul(_c(2.0), x["hb"], gdisc))
    gqa = _sub(gqa, _mul(x["qc"], gdisc))
    gqc = _neg(_mul(x["qa"], gdisc))
    goc = _add(_mul(_c(2.0), gqc, x["oc"]), _mul(ghb, x["d"]))
    gr = _neg(_mul(_c(2.0), x["r"], gqc))
    gd = _add(_mul(ghb, x["oc"]), _mul(_c(2.0), gqa, x["d"]))
    return ((x["o"], goc), (x["d"], gd), (x["c"], _neg(goc)), (x["r"], gr))


#: ``kind -> (vjp, jvp)``.
RAY_RULES: dict[str, tuple[Callable, Callable]] = {
    "ray_tri": (_ray_tri_vjp, _ray_tri_jvp),
    "ray_aabb": (_ray_aabb_vjp, _ray_aabb_jvp),
    "ray_sphere": (_ray_sphere_vjp, _ray_sphere_jvp),
}
