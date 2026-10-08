# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The authoring surface spells EVERY op the IR carries.

WHY THIS ROW EXISTS. ``hawk.ir.ops.OP_KINDS`` is the vocabulary, the rule
table is keyed on it and ``hawk/emit/aether.py`` renders it — three consumers
that a row already checks against each other. Nobody was checking the FOURTH:
the surface an author writes a kernel through. ``outer`` was in ``MATRIX_OPS``,
carried a derivative rule, and rendered as ``aether::outer(...)``, but
``hawk.trace`` exported no name for it, so a body could not reach it. That is
not a missing convenience — it is a hole in the language, and it cost the 
card a workload deviation: one additive-loop row had to drop ``+ outer(x, x)`` on BOTH arms,
which is a benchmark reduced to fit a gap rather than a gap fixed.

THE GATE IS THE ENUMERATION. :data:`SPELLINGS` maps every op kind to a callable
that MINTS it using nothing but names re-exported from ``hawk.trace``, and each
one is executed and its result's ``kind`` checked — so an entry that spells the
wrong op fails as loudly as a missing one. The covered set must then equal
``OP_KINDS`` minus the single documented exception (``splat``), and a kind added
to ``ops.py`` later lands HERE until the surface can say it.

THE NUMERIC HALF is a real workload — ``y = A@x`` and
``W = k*(A^T A) + outer(x, x)`` — traced, emitted, compiled and RUN on the host,
primal and VJP, against the ``tests/_eval.py`` interpreter sample by sample.
That body is chosen deliberately: it is the workload the card had to
deform, and it needs BOTH of the closed findings at once — ``outer`` reachable
from the surface, and a rank-2 reduction aether will accept (``bar_k``'s adjoint
is ``sum(bar_W o (A^T A))``, a rank-2 ``sum``).

This test previously failed via two plants against an earlier hawk tree, both
observed and both removed.

(a) ``outer`` un-exported again (the state this row was written against —
``hawk/trace/value.py``'s ``outer = _binary("outer")`` and its re-export
commented out). The module did not even COLLECT, because the kernel below is
traced at import::

    tests/test_trace_op_exports.py:215: in matvocab
        W[...] = k * (T.transpose(A) @ A) + T.outer(x, x)
    E AttributeError: module 'hawk.trace' has no attribute 'outer'

and with that one term removed so the module could collect, the two enumeration
rows failed on their own::

    AssertionError: these op kinds are in hawk.ir.ops.OP_KINDS but no name
    re-exported by hawk.trace mints them: ['outer']. An op the IR types, the
    rule table differentiates and the emitter renders, that an author cannot
    write, is a hole in the language
    AssertionError: the surface spelling recorded for 'outer' does not run:
    AttributeError("module 'hawk.trace' has no attribute 'outer'")

That second message is why :func:`_mint` returns its failure instead of raising:
a table with a KEY for every kind would otherwise report full coverage while the
name behind the key did not exist.

(b) the rank-2 reduction spelled ``.sum()`` again (``_Renderer._matrix_sum``
bypassed). The VJP half stopped compiling, on ``bar_k`` exactly::

    hawk.ir.nodes.HawkError: host compile of 'matvocab_vjp' failed (rc=1).
    matvocab_vjp.cpp:126:64:
      126 | mut_bar_k[i] = mtx_bar_W_i.cwiseMul((t1 * mtx_A_i)).sum();
    aether/expr/Expression.h:246:54: error: static assertion failed:
      sum: only rank-1 expressions are supported
"""

from __future__ import annotations

from types import SimpleNamespace

import _oracle as O
import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as hm
from hawk.artifact import build
from hawk.diff import vjp
from hawk.ir import Leaf, canonical
from hawk.ir.ops import OP_KINDS
from hawk.trace.value import TableRef, node_of
from hawk.types import TensorType

S = TensorType((), "f64")
B = TensorType((), "bool")
I32 = TensorType((), "i32")
V3 = TensorType((3,), "f64")
Q = TensorType((4,), "f64", "quaternion")
M33 = TensorType((3, 3), "f64")

#: Samples the end-to-end half runs. Every column differs, so a body that read
#: a fixed sample would be visible.
N = 5


def _v(t: TensorType, name: str = "a"):
    """A traced value over a fresh leaf of type ``t``, built through the surface.

    A bool payload binds the reserved ``terminated`` mask role, which is the one
    boolean leaf HAWK's vocabulary has; everything else binds by rank."""
    if t.dtype == "bool":
        return hawk.Value(Leaf("terminated", "terminated", name, t))
    role = {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(t.shape)]
    return hawk.Value(Leaf("vocab_read", role, name, t))


def _spell_at():
    """The ``at(expr)``: the ONE form that mints an ``At`` (a Table read)."""
    return TableRef("table", S).at(_v(S, "where"))


def _spell_set_component_at():
    """``v[k] = e``. The author writes a STATEMENT, and a statement
    has no value to return here, so this row mints the node through the very
    function the rewritten body calls — ``hawk.trace.astpass.set_component``,
    which is what ``__hawk_setitem__`` is bound to in a traced kernel's scope.
    The spelling being exercised is therefore the real one, one link down."""
    from hawk.trace.astpass import set_component

    return set_component(_v(V3), hm.sample_index(), _v(S, "e"), "v")


#: ``op kind -> a callable that mints it through hawk.trace's own exports``.
#: Operators count: an author writes ``a + b``, and ``Value.__add__`` is as much
#: of the surface as ``hm.norm`` is.
SPELLINGS = {
    "add": lambda: _v(V3) + _v(V3, "b"),
    "sub": lambda: _v(V3) - _v(V3, "b"),
    "mul": lambda: _v(V3) * _v(S, "b"),
    "div": lambda: _v(V3) / _v(S, "b"),
    "pow": lambda: _v(S) ** _v(S, "b"),
    "min": lambda: hm.minimum(_v(V3), _v(V3, "b")),
    "max": lambda: hm.maximum(_v(V3), _v(V3, "b")),
    "atan2": lambda: hm.atan2(_v(S), _v(S, "b")),
    "neg": lambda: -_v(V3),
    "abs": lambda: abs(_v(V3)),
    "sqrt": lambda: hm.sqrt(_v(S)),
    "rsqrt": lambda: hm.rsqrt(_v(S)),
    "exp": lambda: hm.exp(_v(S)),
    "log": lambda: hm.log(_v(S)),
    "sin": lambda: hm.sin(_v(S)),
    "cos": lambda: hm.cos(_v(S)),
    "tan": lambda: hm.tan(_v(S)),
    "tanh": lambda: hm.tanh(_v(S)),
    "asin": lambda: hm.asin(_v(S)),
    "acos": lambda: hm.acos(_v(S)),
    "atan": lambda: hm.atan(_v(S)),
    "floor": lambda: hm.floor(_v(S)),
    "exp2": lambda: hm.exp2(_v(S)),
    "expm1": lambda: hm.expm1(_v(S)),
    "log2": lambda: hm.log2(_v(S)),
    "log10": lambda: hm.log10(_v(S)),
    "log1p": lambda: hm.log1p(_v(S)),
    "cbrt": lambda: hm.cbrt(_v(S)),
    "sinh": lambda: hm.sinh(_v(S)),
    "cosh": lambda: hm.cosh(_v(S)),
    "asinh": lambda: hm.asinh(_v(S)),
    "acosh": lambda: hm.acosh(_v(S)),
    "atanh": lambda: hm.atanh(_v(S)),
    "ceil": lambda: hm.ceil(_v(S)),
    "trunc": lambda: hm.trunc(_v(S)),
    "round": lambda: hm.round(_v(S)),
    "rint": lambda: hm.rint(_v(S)),
    "sign": lambda: hm.sign(_v(S)),
    "erf": lambda: hm.erf(_v(S)),
    "erfc": lambda: hm.erfc(_v(S)),
    "hypot": lambda: hm.hypot(_v(S), _v(S, "b")),
    "copysign": lambda: hm.copysign(_v(S), _v(S, "b")),
    "fmod": lambda: hm.fmod(_v(S), _v(S, "b")),
    "remainder": lambda: hm.remainder(_v(S), _v(S, "b")),
    "fdim": lambda: hm.fdim(_v(S), _v(S, "b")),
    "fma": lambda: hm.fma(_v(S), _v(S, "b"), _v(S, "c")),
    "clip": lambda: hm.clip(_v(S), _v(S, "b"), _v(S, "c")),
    "isnan": lambda: hm.isnan(_v(S)),
    "isinf": lambda: hm.isinf(_v(S)),
    "isfinite": lambda: hm.isfinite(_v(S)),
    # the two RUNTIME-indexed component ops: `sample_index()` is an
    # i32 VALUE, which is the shape a lowered loop's own index has.
    "component_at": lambda: _v(V3)[hm.sample_index()],
    "set_component_at": _spell_set_component_at,
    "lt": lambda: _v(S) < _v(S, "b"),
    "le": lambda: _v(S) <= _v(S, "b"),
    "gt": lambda: _v(S) > _v(S, "b"),
    "ge": lambda: _v(S) >= _v(S, "b"),
    "eq": lambda: _v(S) == _v(S, "b"),
    "ne": lambda: _v(S) != _v(S, "b"),
    "land": lambda: hm.land(_v(B), _v(B, "b")),
    "lor": lambda: hm.lor(_v(B), _v(B, "b")),
    "lnot": lambda: hm.lnot(_v(B)),
    "dot": lambda: hm.dot(_v(V3), _v(V3, "b")),
    "cross": lambda: hm.cross(_v(V3), _v(V3, "b")),
    "norm": lambda: hm.norm(_v(V3)),
    "sum": lambda: hm.vsum(_v(V3)),
    "component": lambda: _v(V3)[1],
    "vec": lambda: hm.vec(_v(S), _v(S, "b"), _v(S, "c")),
    "mv": lambda: _v(M33) @ _v(V3, "b"),
    "mm": lambda: _v(M33) @ _v(M33, "b"),
    "transpose": lambda: hm.transpose(_v(M33)),
    "outer": lambda: hm.outer(_v(V3), _v(V3, "b")),
    "quat_mul": lambda: hm.quat_mul(_v(Q), _v(Q, "b")),
    "quat_conj": lambda: hm.quat_conj(_v(Q)),
    "quat_recip": lambda: hm.quat_recip(_v(Q)),
    "quat_rotate": lambda: hm.quat_rotate(_v(Q), _v(V3, "b")),
    "as_pure": lambda: hm.as_pure(_v(V3)),
    "as_vec3": lambda: hm.as_vec3(_v(Q)),
    "select": lambda: hm.select(_v(B), _v(S, "b"), _v(S, "c")),
    "dispatch": lambda: hm.dispatch(_v(I32, "k"), [_v(S, "b"), _v(S, "c")]),
    "at": _spell_at,
    "random_uniform": lambda: hm.random_uniform(_v(I32, "seed"), _v(I32, "counter")),
    "random_normal": lambda: hm.random_normal(_v(I32, "seed"), _v(I32, "counter")),
}

#: The ONE kind with no authoring spelling, and why. ``splat`` is a BROADCAST
#: node: an author never writes one, because writing ``a + 1.0`` against a
#: rank-1 value already broadcasts through the elementwise type rule.
#: The kind exists so a DERIVATIVE can name the transpose of that broadcast —
#: ``hawk/diff/rules.zero_like`` and the ``sum``/``splat`` pair — which is IR
#: the transform mints, never text an author types.
UNSPELLED = {"splat": "the broadcast a derivative mints; an author's rank-0 "
                      "operand broadcasts through the elementwise type rule"}


def _mint(spell):
    """Run one recorded spelling; a failure is RETURNED, never raised.

    Returning it is what keeps the coverage row honest: a kind whose spelling
    raises (``AttributeError: module 'hawk.trace' has no attribute 'outer'``)
    must count as UNCOVERED, so the row that asks "does the surface spell every
    op" fails naming the kind, rather than passing because the table has a key
    for it."""
    try:
        return node_of(spell())
    except Exception as exc:                            # noqa: BLE001 - reported
        return exc


#: Every spelling, RUN once at collection: ``kind -> the node it minted`` or the
#: exception it raised.
MINTED = {kind: _mint(spell) for kind, spell in SPELLINGS.items()}


def test_every_op_kind_has_an_authoring_spelling():
    covered = {kind for kind, got in MINTED.items()
               if not isinstance(got, Exception) and got.kind == kind}
    missing = sorted(set(OP_KINDS) - covered - set(UNSPELLED))
    assert not missing, (
        "these op kinds are in hawk.ir.ops.OP_KINDS but no name re-exported by "
        f"hawk.trace mints them: {missing}. An op the IR types, the rule table "
        "differentiates and the emitter renders, that an author cannot write, is "
        "a hole in the language")
    stale = sorted(set(SPELLINGS) - set(OP_KINDS))
    assert not stale, f"this row spells {stale}, which ops.py no longer declares"


@pytest.mark.parametrize("kind", sorted(SPELLINGS), ids=sorted(SPELLINGS))
def test_each_spelling_mints_the_kind_it_claims(kind):
    """An entry that mints the WRONG op fails as loudly as a missing one."""
    node = MINTED[kind]
    assert not isinstance(node, Exception), (
        f"the surface spelling recorded for {kind!r} does not run: {node!r}")
    assert node.kind == kind, (
        f"the surface spelling recorded for {kind!r} mints {node.kind!r} instead")


def test_the_unspelled_kinds_are_named_with_a_reason():
    """An exception list is only honest while it is short and explained."""
    assert set(UNSPELLED) <= set(OP_KINDS)
    for kind, why in UNSPELLED.items():
        assert why, f"{kind!r} is excused with no reason"


# --------------------------------------------------------------------------- #
# The end-to-end half: the quadratic-form autodiff case, authored through hawk.trace.
# --------------------------------------------------------------------------- #
@hawk.kernel
def matvocab(A: hawk.Matrix[3, 3], x: hawk.Vector[3], k: hawk.Param,
             y: hawk.Mutable[hawk.Vector[3]], W: hawk.Mutable[hawk.Matrix[3, 3]]):
    """``y = A@x``; ``W = k*(A^T A) + outer(x, x)``.

    Every term is here for a reason the card names: ``outer`` was unreachable
    from this surface, and the scalar ``k`` on the MATRIX product is what makes
    ``bar_k``'s adjoint a rank-2 reduction."""
    y = A @ x
    W = k * (hm.transpose(A) @ A) + hm.outer(x, x)


def _planes(rng) -> dict:
    """Input planes whose every column differs."""
    return {"A": rng.uniform(0.5, 1.8, size=(9, N)),
            "x": rng.uniform(0.4, 2.0, size=(3, N)),
            "k": 1.3,
            "bar_y": rng.uniform(0.3, 1.7, size=(3, N)),
            "bar_W": rng.uniform(0.2, 1.4, size=(9, N))}


def _env(planes: dict, i: int) -> dict:
    """Sample ``i`` in the interpreter's shapes (a ``mat_in`` binds FLAT, (R*C, n))."""
    return {"A": planes["A"][:, i].reshape(3, 3), "x": planes["x"][:, i],
            "k": planes["k"], "bar_y": planes["bar_y"][:, i],
            "bar_W": planes["bar_W"][:, i].reshape(3, 3)}


def _run(kernel, directory, cache, planes: dict) -> dict:
    """Emit, compile and RUN one kernel on the host; outputs by name."""
    build(kernel, directory, targets=("host",), cache_dir=cache)
    loaded = O.load(directory, kernel.name)
    wanted = {name: planes[name] for role, name in loaded.arg_spec
              if role in O.INPUT_ROLES or role == "uniform"}
    got = O.run_kernel(loaded, N, **wanted)
    names = [nm for role, nm in loaded.arg_spec if role in O.OUTPUT_ROLES]
    return dict(zip(names, got if isinstance(got, tuple) else (got,)))


@pytest.fixture(scope="module")
def matvocab_run(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("hawk_op_exports")
    cache = str(tmp_path_factory.mktemp("hawk_op_exports_cache"))
    planes = _planes(np.random.default_rng(20260905))
    sinks = vjp(matvocab)
    derived = SimpleNamespace(name="matvocab_vjp", sinks=sinks,
                              walk=canonical(sinks))
    return {"planes": planes,
            "primal": (matvocab.sinks,
                       _run(matvocab, root / "primal", cache, planes)),
            "vjp": (sinks, _run(derived, root / "vjp", cache, planes))}


@pytest.mark.parametrize("half", ("primal", "vjp"))
def test_the_matrix_vocabulary_agrees_with_the_interpreter(matvocab_run, half):
    """Sample by sample, on the compiled host path."""
    planes = matvocab_run["planes"]
    sinks, got = matvocab_run[half]
    for i in range(N):
        want = evaluate(sinks, _env(planes, i))
        for name, ref in want.items():
            ref = np.asarray(ref, dtype=float)
            column = np.asarray(got[name])
            mine = column[i] if column.ndim == 1 else column[:, i].reshape(ref.shape)
            assert np.allclose(mine, ref, rtol=1e-12, atol=1e-12), (
                f"matvocab {half}, plane {name!r}, sample {i}: the host path gave "
                f"{mine!r}, the interpreter gives {ref!r}")


def test_the_vjp_of_the_scalar_takes_the_full_matrix_contraction(matvocab_run):
    """``bar_k`` is a rank-2 reduction, and it must be the WHOLE contraction.

    A rank-2 ``sum`` reaching aether as ``matDot`` against a ones matrix is only
    right if every entry is counted once; a spelling that folded a row or a
    diagonal would still compile and would still be a number."""
    planes = matvocab_run["planes"]
    _sinks, got = matvocab_run["vjp"]
    for i in range(N):
        env = _env(planes, i)
        expect = float(np.sum(env["bar_W"] * (env["A"].T @ env["A"])))
        assert np.allclose(np.asarray(got["bar_k"])[i], expect,
                           rtol=1e-12, atol=1e-12), (
            f"bar_k sample {i}: {got['bar_k'][i]!r} != the full contraction "
            f"{expect!r}")
