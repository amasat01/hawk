# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.math`` — the free-function namespace a kernel body writes math through.

A namespace, not a layer: every name here either is one of
:mod:`hawk.trace`'s free functions, re-exported unchanged, or a
two-line composition of them, so ``hawk.math`` and ``hawk.ir.ops``
declare the same vocabulary by construction
(``tests/test_math_namespace.py`` traces and emits every name in
:data:`__all__`).

It exists because ``exp``, ``log``, ``tanh``, ``vec`` and ``max`` want
spelling without importing forty names one at a time, and because
three spellings have no free function elsewhere in HAWK yet:

* :func:`sample_index` — the lane's own index as a value;
* :func:`split_index` — the ``(major, minor)`` decomposition of a
  flattened two-dimensional launch domain;
* :func:`take` — the free-function spelling of a plane's ``at()`` read.

The names match the previous code generator's ``vmath`` where it has
one, so a re-authored body moves across with its diffs readable;
semantics do not move (``hawk.math.max`` is HAWK's ``maximum``).

Importing this module imports the tracer, so ``import hawk`` never
reaches it: it is asked for by name, ``from hawk import math as m``.

Element-wise math, on host and device alike, with numpy's names:

* exp / log: ``exp`` ``exp2`` ``expm1`` ``log`` ``log2`` ``log10``
  ``log1p`` ``power`` (``pow``, ``**``) ``sqrt`` ``rsqrt`` ``cbrt`` ``hypot``
* trigonometric: ``sin`` ``cos`` ``tan`` ``asin`` ``acos`` ``atan`` ``atan2``
* hyperbolic: ``sinh`` ``cosh`` ``tanh`` ``asinh`` ``acosh`` ``atanh``
* rounding: ``floor`` ``ceil`` ``trunc`` ``round`` ``rint``
* misc: ``absolute`` (``abs``) ``sign`` ``copysign`` ``fmod`` ``remainder``
  ``fdim`` ``fma`` ``clip`` ``minimum`` (``min``) ``maximum`` (``max``)
  ``isnan`` ``isinf`` ``isfinite``
* special: ``erf`` ``erfc``

Semantics differing from numpy, by design: ``round`` is C's (halves away
from zero; ``rint`` is numpy's half-to-even ``np.round``), and
``minimum``/``maximum`` ignore a NaN operand (``np.fmin``/``np.fmax``).
``remainder`` is numpy's floor-mod (sign of the divisor), ``fmod`` C's
(sign of the dividend). The class tests return ``bool``.

Shapes: every element-wise function takes vector and matrix operands as
the arithmetic operators do — equal shapes, or one operand rank-0
broadcast — and refuses any other combination in the operators' words.
``exp`` ``log`` ``sqrt`` ``rsqrt``, the trigonometric functions but
``atan2``, ``tanh``, ``abs``, ``power``, ``minimum`` and ``maximum`` lower to
one aether expression at any rank; the rest, with the comparisons and
``land``/``lor``/``lnot``, are mapped entry by entry at trace time and
re-assembled — aether spells them at rank 0 only — so a class test
or a comparison on a matrix is a bool matrix, the mask
:func:`where`/:func:`select` takes per entry (a rank-0 branch broadcasts).

Derivatives: the rounding family, ``sign`` and the class tests carry a
zero derivative. ``abs`` takes ``+1`` at zero (as JAX); ``copysign`` is
``|a|`` times ``b``'s sign, constant in ``b``; ``fmod``/``remainder``
differentiate as ``a - trunc(a/b)*b`` / ``a - floor(a/b)*b``; ``fdim`` is
zero at ``a == b``; ``hypot`` is zero at the origin; ``clip`` passes the
gradient to ``x`` on the closed interval ``[lo, hi]`` (torch's ``clamp``;
JAX halves it at a tie), to ``hi`` above it or whenever ``lo > hi``, and
to ``lo`` below it.

Not yet: ``lgamma``/``tgamma``/``digamma``, Bessel functions, and
integer-specific operations beyond the existing ones.
"""

from __future__ import annotations

import math as _stdmath
from types import SimpleNamespace
from typing import Any

from .ir import HawkError
from .trace.value import (
    Value,
    absolute,
    acos,
    acosh,
    as_pure,
    as_vec3,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cbrt,
    ceil,
    clip,
    copysign,
    cos,
    cosh,
    cross,
    dispatch,
    dot,
    erf,
    erfc,
    exp,
    exp2,
    expm1,
    fdim,
    floor,
    fma,
    fmod,
    hypot,
    index_node,
    isfinite,
    isinf,
    isnan,
    land,
    lnot,
    log,
    log1p,
    log2,
    log10,
    lor,
    maximum,
    minimum,
    n_samples,
    node_of,
    norm,
    outer,
    power,
    quat_conj,
    quat_mul,
    quat_recip,
    quat_rotate,
    remainder,
    rint,
    rsqrt,
    sample_index,
    select,
    sign,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    transpose,
    trunc,
    vsum,
)
from .trace.value import random_normal as _normal01
from .trace.value import random_uniform as _uniform01
from .trace.value import round as round  # noqa: A004 - numpy's spelling
from .trace.value import vec as _vec

#: Familiar spellings for the two-argument, NaN-ignoring element-wise
#: min/max (``np.fmin``/``np.fmax``, never the array reductions) —
#: aliases of :func:`hawk.trace.minimum`/:func:`~hawk.trace.maximum`.
max = maximum   # noqa: A001 - a namespace module's names are its API (cf. numpy)
min = minimum   # noqa: A001
#: ``np.abs``/``np.pow``: aliases of :func:`absolute`/:func:`power`.
abs = absolute  # noqa: A001
pow = power     # noqa: A001

#: Knuth's product method, truncated: ``P[X >= 16] < 1e-13`` at ``lambda = 1``.
_POISSON1_DRAWS = 16
_POISSON1_L = _stdmath.exp(-1.0)


def random_poisson1(seed: Any, counter: Any) -> Value:
    """A Poisson(1)-distributed count per lane, as a real value: Knuth's
    product method, multiplying uniform draws until the running product
    falls below ``e**-1`` — the number of products still above it is the
    draw. Written straight-line over 16 draws with sub-counters
    ``counter*16 + k``, no loop node: the count is monotone in ``k``, so
    the sum of indicators is the stopping index."""
    base = counter * _POISSON1_DRAWS
    product = None
    count = None
    for k in range(_POISSON1_DRAWS):
        u = _uniform01(seed, base + k)
        product = u if product is None else product * u
        hit = select(product > _POISSON1_L, 1.0, 0.0)
        count = hit if count is None else count + hit
    return count


def _is_default(x: Any, value: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and float(x) == value


def random_uniform(seed: Any, counter: Any, lo: Any = 0.0, hi: Any = 1.0) -> Value:
    """A uniform draw on ``[lo, hi)`` — at the defaults exactly the
    primitive op, otherwise the composition ``lo + (hi - lo) * u``,
    aether's own formula. ``lo``/``hi`` may be traced values, giving the
    reparameterisation gradient for free."""
    u = _uniform01(seed, counter)
    if _is_default(lo, 0.0) and _is_default(hi, 1.0):
        return u
    return u * (hi - lo) + lo


def random_normal(seed: Any, counter: Any, mean: Any = 0.0, sd: Any = 1.0) -> Value:
    """A normal draw ``N(mean, sd^2)`` — at the defaults exactly the primitive
    op (:func:`hawk.trace.random_normal`); otherwise ``mean + sd * z``."""
    z = _normal01(seed, counter)
    if _is_default(mean, 0.0) and _is_default(sd, 1.0):
        return z
    return z * sd + mean


def random_lognormal(seed: Any, counter: Any, mu: Any = 0.0, sigma: Any = 1.0) -> Value:
    """``exp(mu + sigma * z)`` — aether's ``Generator::lognormal(mu, sigma)``."""
    return exp(_normal01(seed, counter) * sigma + mu)


def random_exponential(seed: Any, counter: Any, rate: Any = 1.0) -> Value:
    """An exponential draw with ``rate`` (mean ``1/rate``): ``-log(1 - u)
    / rate``, not aether's own ``-log(u) / rate``, since ``u`` in
    ``[0, 1)`` keeps the argument in ``(0, 1]`` so no lane draws ``+inf``."""
    u = _uniform01(seed, counter)
    return log(u * (-1.0) + 1.0) * (-1.0) / rate


def random_bernoulli(seed: Any, counter: Any, p: Any = 0.5) -> Value:
    """A Bernoulli draw as a real ``1.0`` / ``0.0``: ``u < p`` (aether's rule)."""
    return select(_uniform01(seed, counter) < p, 1.0, 0.0)


def random_uniform_int(seed: Any, counter: Any, lo: Any, hi: Any) -> Value:
    """An integer draw uniform over the closed range ``[lo, hi]``,
    carried as a real value: ``floor(lo + (hi - lo + 1) * u)`` — ``u < 1``
    strictly, so ``hi`` is reached and never exceeded (aether's
    ``uniformInt`` convention)."""
    return floor(_uniform01(seed, counter) * (hi - lo + 1) + lo)


def random_multivariate_normal(seed: Any, counter: Any, mean: Any,
                               cholesky: Any) -> Value:
    """A correlated normal vector ``mean + L z`` with ``L`` the (lower)
    Cholesky factor of the covariance and ``z`` D independent standard
    draws on sub-counters ``counter*D + k`` — aether's
    ``LowerTriangular`` composition, through the ``mv`` op. Like
    :func:`random_poisson1`'s stride 16: two compositions sharing a seed
    and overlapping sub-counter windows would share uniforms."""
    n_dim = int(node_of(cholesky).ttype.shape[0])
    base = counter * n_dim
    z = _vec(*[_normal01(seed, base + k) for k in range(n_dim)])
    return cholesky @ z + mean


#: The spec's own spelling, ``hawk.math.random.uniform``/``.normal`` — a
#: tiny attribute namespace over the same functions ``__all__`` exports
#: flat, never a second implementation (``m.random.uniform is
#: m.random_uniform``).
random = SimpleNamespace(
    uniform=random_uniform, normal=random_normal, lognormal=random_lognormal,
    exponential=random_exponential, bernoulli=random_bernoulli,
    uniform_int=random_uniform_int, poisson1=random_poisson1,
    multivariate_normal=random_multivariate_normal)

__all__ = [
    "abs", "absolute", "acos", "acosh", "argmax", "as_pure", "as_vec3", "asin",
    "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "clip", "copysign", "cos",
    "cosh", "cross", "dispatch", "dot", "erf", "erfc",
    "exp", "exp2", "expm1", "fdim", "floor", "fma", "fmod", "hypot", "isfinite",
    "isinf", "isnan", "land", "lnot", "log", "log10", "log1p", "log2", "lor",
    "max", "maximum", "min", "minimum",
    "n_samples", "norm", "outer", "pow", "power", "quat_conj", "quat_mul",
    "quat_recip", "quat_rotate", "random_bernoulli", "random_exponential",
    "random_lognormal", "random_multivariate_normal", "random_normal",
    "random_poisson1", "random_uniform", "random_uniform_int", "remainder",
    "rint", "round", "rsqrt", "sample_index", "select", "sign", "sin", "sinh",
    "split_index", "sqrt", "take", "tan", "tanh", "transpose", "trunc", "vec",
    "vsum", "where",
]


def vec(*components: Any) -> Value:
    """Build a rank-1 value from scalar components: both ``vec(a, b, c)``
    and ``vec([a, b, c])`` work, the second being what a static
    comprehension evaluates to. A single list argument is the
    components; anything else is taken positionally, so ``vec(x)`` is
    still the width-1 vector, not a mis-read singleton."""
    if len(components) == 1 and isinstance(components[0], (list, tuple)):
        components = tuple(components[0])
    return _vec(*components)


def where(cond: Any, a: Any, b: Any) -> Value:
    """Branchless ``cond ? a: b`` — the ``np.where`` spelling of :func:`select`."""
    return select(cond, a, b)


def argmax(v: Any, /, *components: Any) -> Value:
    """The index (rank-0 ``i32``) of the first maximum: of a rank-1 value
    (``argmax(logits)``), of a Python list of rank-0 values, or of
    rank-0 components given positionally (``argmax(a, b, c)``).

    A select chain over the components — no new op kind. Ties go to the
    lower index, as ``np.argmax``; a NaN component never wins. The index
    literals are minted through :func:`~hawk.trace.value.index_node`,
    the ``i32`` spelling, since a bare ``int`` in a body is the real
    number."""
    if components:
        comps = (v,) + tuple(components)
    elif isinstance(v, (list, tuple)):
        comps = tuple(v)
    else:
        node = node_of(v)
        if len(node.ttype.shape) != 1:
            raise HawkError(
                f"argmax takes a rank-1 value or rank-0 components, got {node.ttype!r}")
        comps = tuple(Value(node)[j] for j in range(node.ttype.shape[0]))
    if not comps:
        raise HawkError("argmax of no components")
    best = Value(index_node(0))
    best_value = comps[0]
    for j in range(1, len(comps)):
        wins = comps[j] > best_value
        best = select(wins, Value(index_node(j)), best)
        best_value = select(wins, comps[j], best_value)
    return best


def split_index(minor: int) -> tuple:
    """Split this lane's index into its ``(major, minor)`` components.

    A kernel whose natural domain is two-dimensional — rows of ``minor``
    entries each — is launched over the flattened product, one lane per
    pair. This returns the two coordinates that flattening folded
    together, so a body reads ``row, entry = split_index(64)`` instead
    of writing the division and remainder by hand. The minor component
    varies between neighbouring lanes; declare it as the last axis of
    any plane read through it, so consecutive lanes touch consecutive
    addresses.

    ``minor`` must be a compile-time positive ``int``: it is the divisor
    baked into the emitted arithmetic, so a traced value there would
    make the split unknowable at trace time. The remainder is spelled
    ``lane - major * minor`` rather than through a modulo op, since
    HAWK's op set carries none and the identity is exact for a
    non-negative lane index."""
    if not isinstance(minor, int) or isinstance(minor, bool):
        raise HawkError(
            "split_index(minor): minor must be a compile-time positive int (got "
            f"{type(minor).__name__}); a traced value cannot size the split")
    if minor <= 0:
        raise HawkError(f"split_index(minor): minor must be positive, got {minor}")
    lane = sample_index()
    divisor = Value(index_node(minor))
    major = lane / divisor
    return major, lane - major * divisor


def take(buf: Any, /, *positional: Any, **named: Any) -> Value:
    """Read ``buf`` at an absolute index — the free-function spelling of
    ``at``. ``take(t, i)`` for a flat plane and ``take(t, row=r, col=c)``
    for a named-axis one; both delegate to the plane's own
    :meth:`hawk.trace.value.TableRef.at`.

    The plane is positional-only on purpose: an axis may legitimately be
    labelled ``plane`` (``Staged`` declares exactly that one), and a
    keyword parameter here would collide with it silently."""
    reader = getattr(buf, "at", None)
    if reader is None:
        raise HawkError(
            f"take(): {buf!r} is not a readable plane — take() reads a "
            "Table[...] / Staged[...] / Wide[W] parameter, which is the only "
            "declaration the at(expr) form applies to")
    return reader(*positional, **named)
