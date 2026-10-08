# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every rank>=1 operand of a RANK-COLLAPSING op is read at the body's own sample.

WHAT THIS ROW EXISTS FOR. aether's reductions are *sample-free* by contract:
``Expression::dot()``, ``norm()``, ``sum()`` and ``matDot()`` all fold at
``SampleIndex::make(0)`` and say so in their own header ("the intended usage is
on an already-materialized (Item-level) expression, batched access goes through
``view[i].get()`` first" — ``aether/expr/Expression.h``). A whole-plane ``View``
handed to one of them therefore returns SAMPLE 0's answer for every sample, and
it does so silently: the body compiles, the artifact loads, the plan runs, and
every column of the output carries the first column's reduction.

HAWK's renderer used to produce exactly that. A rank>=1 leaf was materialised
(``vin_v[i].get()``, an ``Item``, which IS sample-bound) only when its fan-out
was >= 2; at fan-out 1 it rendered as the bare view, so ``y = norm(v)``
emitted ``mut_y[i] = vin_v.norm();``. The whole fixture corpus missed it
because every rank>=1 leaf in that corpus is read at least twice (``energy``
spells ``dot(v, v)``, ``vocab`` reads ``v`` three times), and a leaf read twice
is a leaf that was already being hoisted for a completely different reason
( address-computation argument).

THE CORPUS ENUMERATES, IT DOES NOT SAMPLE. The set of rank-collapsing kinds is
COMPUTED from ``hawk.ir.ops`` itself (:func:`collapsing_kinds` probes every kind
in ``OP_KINDS`` over every operand-type combination it accepts and keeps the
ones whose RESULT rank is below their widest operand's), and
:func:`test_the_corpus_covers_every_rank_collapsing_kind` fails if the table
below stops covering that set — so a kind added to ``ops.py`` later cannot
quietly escape this row. Across that set the corpus is the full cross product of

  * operand FORM: the bare leaf, a unary of the leaf, a binary of a uniform and
    the leaf, and a binary of two distinct leaves — the four shapes a
    collapsing op's rank>=1 operand can have, only the first of which is a leaf
    the renderer sees directly;
  * FAN-OUT 1 and 2: fan-out 2 is produced by a second sink that echoes the
    same leaf, which is the ONLY thing that used to separate a right answer
    from a wrong one;
  * RANK 1 (width 3) and RANK 2 (2x2 for ``sum``, 3x3 for ``mv``), wherever the
    kind's own type rule admits that rank.

and every case's answer is checked, per sample, against the ``tests/_eval.py``
interpreter on inputs that DIFFER IN EVERY COLUMN. That last property is not
decoration: a plane that is constant across samples makes sample 0's reduction
the right answer everywhere, which is precisely how this defect survived.

THE SECOND HALF — the rank-2 reduction's SPELLING. ``hawk.ir.ops`` admits
``sum`` over a rank-2 operand and the reverse rule of a scalar uniform times a
rank-2 expression MINTS one (``hawk/diff/transform._reduce_to`` un-broadcasts
the rank-0 operand's adjoint with a ``sum``), but aether's ``Expression::sum()``
``static_assert``s rank 1. The emitted TU did not compile at all, so this half's
failure was loud rather than silent — a refusal in the middle of a derivative
nobody could emit. It is checked here beside the first half because it is the
same question (how does a rank>=1 value reach a rank-0 aether result) and the
same fix site.

This test previously failed: restoring ``hawk/emit/aether.py`` to an
earlier state under this row gave **77 failed, 42 passed**.

  * the NUMERIC row failed on 28 cases — every fan-out-1 case of ``norm``,
    ``dot`` and ``sum`` at rank 1, every ``sum`` case at rank 2 and both VJP
    rows, plus the fan-out-2 cases whose OTHER operand was still an
    unsubscripted plane (``dot``'s second vector, ``binary_leaf_leaf``'s second
    leaf)::

        AssertionError: norm/rank1/leaf/fanout1 sample 1: the host path gave
        np.float64(1.5528165144102362), the interpreter gives
        array(1.78863966). A rank>=1 operand folded at aether's sample-free
        SampleIndex::make(0) answers with sample 0's value in every column.
            mut_out_norm1_leaf_1[i] = vin_norm1_leaf_1_a.norm();
            mut_out_norm1_leaf_2[i] = vin_norm1_leaf_2_a_i.norm();

    The two lines side by side are the whole finding: the SAME body, one case
    apart, reading the plane and reading the sample.
  * the rank-2 arms did not get as far as a wrong answer — they did not
    COMPILE::

        hawk.ir.nodes.HawkError: host compile of 'vjp_scalar_rank2' failed
        (rc=1). aether/expr/Expression.h:246:54: error: static assertion
        failed: sum: only rank-1 expressions are supported

  * the STRUCTURAL row failed on 49 cases, a strict superset: it also fires on
    ``component`` and ``mv``, whose NUMERIC answers were right before the fix
    (``component`` renders ``.eval<K>(i)``, naming the sample itself; ``mv`` is
    a lazy ``MatVec`` the SINK's subscript evaluates) but whose rank>=1 operands
    were still whole planes one refactor away from reaching a fold. They are in
    the corpus because the corpus enumerates the vocabulary — and the structural
    half is why their being right by accident does not count as being right.
"""

from __future__ import annotations

import itertools
import zlib

import _oracle as O
import numpy as np
import pytest
from _eval import evaluate

from hawk.artifact import build
from hawk.diff import vjp
from hawk.emit import render_body
from hawk.emit.aether import mirror_of
from hawk.ir import Assign, HawkError, Leaf, canonical
from hawk.ir import make as mk
from hawk.ir.ops import OP_KINDS
from hawk.types import TensorType

#: Samples per case. Small because every case is also evaluated sample by sample
#: by the scratch interpreter; seven is enough that a body reading a fixed column
#: cannot coincide with the right answer anywhere but that column.
N = 7

S = TensorType((), "f64")
V3 = TensorType((3,), "f64")
M22 = TensorType((2, 2), "f64")
M33 = TensorType((3, 3), "f64")

_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}

#: The four operand shapes a rank-collapsing op's rank>=1 operand can take.
FORMS = ("leaf", "unary_of_leaf", "binary_uniform_leaf", "binary_leaf_leaf")
#: Fan-out of the leaf under test. 1 is the arm the defect lived in.
FANOUTS = (1, 2)


# --------------------------------------------------------------------------- #
# Which kinds are rank-collapsing — computed from ops.py, never listed by hand.
# --------------------------------------------------------------------------- #
_PROBE_TYPES = (S, V3, TensorType((4,), "f64", "quaternion"), M22, M33,
                TensorType((3, 2), "f64"), TensorType((), "bool"))
_PROBE_LITERALS = (None, 0, (3,))

#: ``at`` is excluded from the probe's answer, and the exclusion is CHECKED
#: rather than asserted: its second operand is an absolute sample INDEX, not a
#: value, so a rank-2 index makes the probe read a rank-0 gather of a rank-0
#: plane as "collapsing". The real reason it can carry no rank>=1 operand at all
#: is — a ``lookup`` plane is pinned to the 32-byte ``ScalarHandle`` and
#: ``mirror_of`` REFUSES a rank>=1 one — which
#: :func:`test_a_lookup_plane_cannot_carry_a_rank1_operand_at_all` re-derives
#: from the emitter every time this row runs.
_INDEX_TAKING = frozenset({"at"})


def collapsing_kinds() -> frozenset:
    """Every op kind whose RESULT rank is below its widest operand's rank.

    Derived by probing :func:`hawk.ir.ops.make` over a small closed set of
    operand types rather than by transcribing a list: the vocabulary lives in
    ``ops.py`` and a kind added there must not be able to slip past this row
    just because nobody remembered to extend a constant here."""
    found = set()
    for kind in OP_KINDS:
        for arity in (1, 2, 3):
            for operands in itertools.product(_PROBE_TYPES, repeat=arity):
                for literal in _PROBE_LITERALS:
                    leaves = [Leaf("vocab_read", _ROLE[len(t.shape)], f"p{i}", t)
                              for i, t in enumerate(operands)]
                    try:
                        node = mk(kind, leaves, literal)
                    except Exception:
                        continue
                    if len(node.ttype.shape) < max(len(t.shape) for t in operands):
                        found.add(kind)
    return frozenset(found) - _INDEX_TAKING


# --------------------------------------------------------------------------- #
# The corpus: (kind, rank of the operand under test) -> a builder per case.
# --------------------------------------------------------------------------- #
#: ``(kind, rank)`` groups, one emitted kernel each. The rank axis carries the
#: ranks the kind's own type rule admits for the operand under test.
GROUPS = (("norm", 1), ("dot", 1), ("sum", 1), ("sum", 2),
          ("component", 1), ("mv", 1), ("mv", 2))


def _ttype(kind: str, rank: int) -> TensorType:
    """The type of the operand under test. ``mv``'s rank-1 arm must contract
    with a 3x3, and ``sum``'s rank-2 arm deliberately uses the OTHER admitted
    square extent so both 2x2 and 3x3 are emitted somewhere in the corpus."""
    if rank == 1:
        return V3
    return M33 if kind == "mv" else M22


def _operand(form: str, tag: str, t: TensorType) -> tuple:
    """One operand shape, plus every leaf it binds and the leaf UNDER TEST."""
    role = _ROLE[len(t.shape)]
    a = Leaf("vocab_read", role, f"{tag}_a", t)
    if form == "leaf":
        return a, a, (a,)
    if form == "unary_of_leaf":
        return mk("tanh", (a,)), a, (a,)
    if form == "binary_uniform_leaf":
        u = Leaf("uniform", "uniform", f"{tag}_u", S)
        return mk("mul", (u, a)), a, (a, u)
    b = Leaf("vocab_read", role, f"{tag}_b", t)
    return mk("add", (a, b)), a, (a, b)


def _collapse(kind: str, operand, tag: str) -> tuple:
    """Apply the rank-collapsing op to ``operand``; returns ``(node, extra leaves)``."""
    if kind == "norm":
        return mk("norm", (operand,)), ()
    if kind == "sum":
        return mk("sum", (operand,)), ()
    if kind == "component":
        return mk("component", (operand,), 1), ()
    if kind == "dot":
        other = Leaf("vocab_read", "vec_in", f"{tag}_c", V3)
        return mk("dot", (operand, other)), (other,)
    # mv: the operand under test is the MATRIX on the rank-2 arm and the VECTOR
    # on the rank-1 arm, so both of a rank-changing binary's sides are covered.
    if len(operand.ttype.shape) == 2:
        other = Leaf("vocab_read", "vec_in", f"{tag}_c", V3)
        return mk("mv", (operand, other)), (other,)
    other = Leaf("vocab_read", "mat_in", f"{tag}_c", M33)
    return mk("mv", (other, operand)), (other,)


def _group_sinks(kind: str, rank: int) -> tuple:
    """Every case of one ``(kind, rank)`` group, as ONE kernel's sink set.

    The cases share a kernel (and therefore a compile) but no NODE: each case's
    leaves carry its own names, so the canonical walk cannot merge two cases and
    the fan-out under test is the case's own."""
    sinks, cases = [], []
    for form, fanout in itertools.product(FORMS, FANOUTS):
        tag = f"{kind}{rank}_{form}_{fanout}"
        t = _ttype(kind, rank)
        operand, under_test, leaves = _operand(form, tag, t)
        value, extra = _collapse(kind, operand, tag)
        out = f"out_{tag}"
        sinks.append(Assign(out, value, value.ttype))
        echo = None
        if fanout == 2:
            # the ONLY thing separating the two arms: a second sink that reads
            # the same leaf, which is what used to buy it a materialisation.
            echo = f"echo_{tag}"
            sinks.append(Assign(echo, under_test, under_test.ttype))
        cases.append({"id": f"{kind}/rank{rank}/{form}/fanout{fanout}",
                      "out": out, "echo": echo,
                      "leaves": tuple(leaves) + tuple(extra)})
    return tuple(sinks), tuple(cases)


def _vjp_group() -> tuple:
    """The scalar-times-rank-2 VJP: ``W = k * M``, differentiated.

    ``bar_k``'s contribution is a rank-2 product un-broadcast by
    ``hawk/diff/transform._reduce_to`` into ``sum(bar_W * M)`` — a rank-2 ``sum``
    node, which is the shape aether's ``Expression::sum()`` refuses. It is built
    through ``hawk.diff.vjp`` rather than by hand so the row observes the rule
    table's own output."""
    k = Leaf("uniform", "uniform", "vjpk_k", S)
    m = Leaf("vocab_read", "mat_in", "vjpk_m", M22)
    primal = (Assign("vjpk_W", mk("mul", (k, m)), M22),)
    return vjp(primal), (
        {"id": "vjp/scalar_times_rank2", "out": "bar_vjpk_k", "echo": None,
         "leaves": (k, m, Leaf("vocab_read", "mat_in", "bar_vjpk_W", M22))},
        {"id": "vjp/scalar_times_rank2/matrix_adjoint", "out": "bar_vjpk_m",
         "echo": None, "leaves": ()},
    )


#: ``group name -> (sinks, cases)``: one emitted, compiled host kernel each.
def _corpus() -> dict:
    out = {f"{kind}{rank}": _group_sinks(kind, rank) for kind, rank in GROUPS}
    out["vjp_scalar_rank2"] = _vjp_group()
    return out


CORPUS = _corpus()


# --------------------------------------------------------------------------- #
# Inputs that differ in every column, and the interpreter's answer for each.
# --------------------------------------------------------------------------- #
def _plane(name: str, t: TensorType, n: int) -> np.ndarray:
    """A plane whose every entry differs, seeded on the LEAF NAME.

    Not ``arange``: a monotone plane makes several of these bodies monotone too,
    and a comparison that would also pass on the wrong column is no comparison.
    A per-name seed keeps the arrays reproducible without making two leaves of
    one case carry the same numbers."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    width = int(np.prod(t.shape)) if t.shape else 1
    values = rng.uniform(0.6, 2.4, size=(width, n))
    return values[0] if not t.shape else values


def _inputs(cases) -> dict:
    """Every input plane one group binds, keyed by leaf name."""
    bound: dict = {}
    for case in cases:
        for leaf in case["leaves"]:
            if leaf.name in bound:
                continue
            bound[leaf.name] = (float(_plane(leaf.name, leaf.ttype, 1)[0]) + 0.37
                                if leaf.role == "uniform"
                                else _plane(leaf.name, leaf.ttype, N))
    return bound


def _column(value, ttype: TensorType, i: int):
    """Sample ``i`` of one bound plane, in the interpreter's own shape."""
    if not isinstance(value, np.ndarray):
        return value
    if not ttype.shape:
        return float(value[i])
    if len(ttype.shape) == 1:
        return np.asarray(value)[:, i]
    return np.asarray(value)[:, i].reshape(ttype.shape)


def _expected(sinks, cases, bound: dict) -> list:
    """The interpreter's answer for every sample, one dict per sample."""
    out = []
    for i in range(N):
        env = {}
        for case in cases:
            for leaf in case["leaves"]:
                env[leaf.name] = _column(bound[leaf.name], leaf.ttype, i)
        out.append(evaluate(sinks, env))
    return out


# --------------------------------------------------------------------------- #
# Build + run the host path.
# --------------------------------------------------------------------------- #
class _Built:
    """One group's compiled host kernel, its outputs and the oracle's answer."""

    def __init__(self, name, sinks, cases, directory, cache_dir):
        from types import SimpleNamespace

        self.cases = cases
        self.sinks = sinks
        kernel = SimpleNamespace(name=name, sinks=sinks, walk=canonical(sinks))
        build(kernel, directory, targets=("host",), cache_dir=cache_dir)
        loaded = O.load(directory, name)
        self.bound = _inputs(cases)
        got = O.run_kernel(loaded, N, **self.bound)
        names = [nm for role, nm in loaded.arg_spec if role in O.OUTPUT_ROLES]
        self.got = dict(zip(names, got if isinstance(got, tuple) else (got,)))
        self.expected = _expected(sinks, cases, self.bound)
        #: the BODY string, not the whole TU: the wrapper's ``params[]`` unpack
        #: names every binding too, and a failure that pasted it would bury the
        #: one line that is wrong.
        self.body = render_body(sinks, kernel.walk).text


class _Builder:
    """Builds each group ON DEMAND and remembers it.

    Lazy on purpose: a group whose emitted TU does not COMPILE (the rank-2
    ``sum`` arm, before the renderer learned aether's spelling for it) must fail
    its own cases and leave the other groups' verdicts standing — a module-wide
    fixture that raised would erase the very rows that show which half of the
    corpus is affected."""

    def __init__(self, root, cache) -> None:
        self.root, self.cache, self._made = root, cache, {}

    def __getitem__(self, group: str) -> _Built:
        if group not in self._made:
            sinks, cases = CORPUS[group]
            self._made[group] = _Built(group, sinks, cases,
                                       self.root / group, self.cache)
        return self._made[group]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> _Builder:
    """Every group, emitted, compiled and run once — on first use."""
    return _Builder(tmp_path_factory.mktemp("hawk_subscript"),
                    str(tmp_path_factory.mktemp("hawk_subscript_cache")))


def _ids() -> list:
    return [c["id"] for _s, cases in CORPUS.values() for c in cases]


def _flat() -> list:
    return [(name, c) for name, (_s, cases) in CORPUS.items() for c in cases]


@pytest.mark.parametrize("group,case", _flat(), ids=_ids())
def test_every_sample_gets_its_own_reduction(built, group, case):
    """The whole point: sample ``i``'s output is sample ``i``'s answer.

    A body that folds a whole-plane expression answers with SAMPLE 0's value in
    every column, so the comparison is made per sample and the failure names the
    sample as well as the case."""
    b = built[group]
    got = np.asarray(b.got[case["out"]])
    for i in range(N):
        want = np.asarray(b.expected[i][case["out"]], dtype=float)
        mine = got[i] if got.ndim == 1 else got[:, i].reshape(want.shape)
        assert np.allclose(mine, want, rtol=1e-12, atol=1e-12), (
            f"{case['id']} sample {i}: the host path gave {mine!r}, the "
            f"interpreter gives {want!r}. A rank>=1 operand folded at aether's "
            "sample-free SampleIndex::make(0) answers with sample 0's value in "
            f"every column.\n{b.body}"
        )


@pytest.mark.parametrize("group,case", _flat(), ids=_ids())
def test_no_rank1_leaf_reaches_a_fold_unsubscripted(built, group, case):
    """The STRUCTURAL half, so the row still fails if the answer coincides.

    Every rank>=1 binding a body reads is subscripted exactly once — the
    materialisation into an ``Item`` — and no bare view identifier is ever the
    receiver of a fold. Checked on the emitted text because a numeric row alone
    would pass on a plane whose columns happened to agree."""
    import re

    b = built[group]
    for leaf in case["leaves"]:
        if not leaf.ttype.shape:
            continue
        ident = f"{'vin' if len(leaf.ttype.shape) == 1 else 'mtx'}_{leaf.name}"
        bad = re.findall(rf"\b{ident}\.(norm|sum|dot|matDot)\(", b.body)
        assert not bad, (
            f"{case['id']}: the emitted body folds the WHOLE PLANE {ident!r} "
            f"({bad}) — aether's reductions evaluate at SampleIndex::make(0), so "
            f"this is sample 0's answer for every sample.\n{b.body}")
        assert len(re.findall(rf"\b{ident}\[", b.body)) == 1, (
            f"{case['id']}: {ident!r} is not subscripted exactly once — a rank>=1 "
            f"leaf is read at the body's own sample, exactly once.\n{b.body}")


def test_the_corpus_covers_every_rank_collapsing_kind():
    """The gate half: the table above must cover the whole collapsing vocabulary.

    Computed from ``ops.py``, so a rank-collapsing kind added later fails HERE
    rather than shipping unexercised."""
    covered = {kind for kind, _rank in GROUPS}
    missing = sorted(collapsing_kinds() - covered)
    assert not missing, (
        "these op kinds collapse rank and no case in this row exercises them: "
        f"{missing}. Every rank>=1 operand of a collapsing op must be read at "
        "the body's own sample, so each such kind needs a group above")


def test_a_lookup_plane_cannot_carry_a_rank1_operand_at_all():
    """Why ``at`` is the one collapsing-looking kind with no group.

    The probe reads ``at`` as rank-collapsing because its SECOND operand is an
    absolute sample index rather than a value. The reason it needs no group is
    stronger and mechanical: pins the ``lookup`` role to eagle's 32-byte
    ``ScalarHandle``, so a rank>=1 gathered plane cannot be emitted at all — and
    that is re-derived here from the emitter's own map rather than trusted to
    the comment beside ``_INDEX_TAKING``."""
    with pytest.raises(HawkError, match="ScalarHandle"):
        mirror_of("lookup", V3)


def test_the_corpus_enumerates_both_fanouts_and_all_four_forms():
    """The corpus cannot quietly shrink: the cross product is asserted whole."""
    for kind, rank in GROUPS:
        _sinks, cases = CORPUS[f"{kind}{rank}"]
        want = {f"{kind}/rank{rank}/{f}/fanout{n}"
                for f in FORMS for n in FANOUTS}
        assert {c["id"] for c in cases} == want, (
            f"{kind}/rank{rank} no longer enumerates every (form, fan-out) pair")
