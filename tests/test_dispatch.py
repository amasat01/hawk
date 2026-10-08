# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests the `dispatch` node and its `predicated`/`switch`/`segmented`
emission: K in {2, 5, 16}, a kind plane INCLUDING out-of-range values,
rank-0/rank-1 branches, both backends. The built policies agree with the
serial oracle (EXACT for an integer-typed branch set), and each policy's
units carry DISTINCT unit digests.

`_eval.py`'s scratch interpreter carries its own `Dispatch` arm alongside
`Select`'s (the same clamp every emitted policy shares) — this file's first
block is that arm's own unit check, single-sample, no compile; the numeric
rows that follow build real artifacts and compare against a plain numpy
reference on both the compiled HOST oracle (``hawk._core``) and the compiled
DEVICE kernel (eagle's `DeviceKernel`).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from _deploy import device_plugin
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as M
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.diff.rules import zero_like
from hawk.emit import render_body
from hawk.ir import AccumWrite, Assign, At, Dispatch, HawkError, Leaf, canonical
from hawk.ir import make as mk
from hawk.ir.nodes import DISPATCH_POLICIES
from hawk.ir.ops import OP_KINDS
from hawk.ir.segment import split_segmented
from hawk.types import TensorType

S = TensorType((), "f64")
I32 = TensorType((), "i32")
V3 = TensorType((3,), "f64")

_EPS = np.finfo(np.float64).eps


def _band(s: int, a, b) -> float:
    """The ONE anchor, restated (test_serial_oracle.band): S x 2 x eps,
    applied relatively — derived, never fitted."""
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return s * 2.0 * _EPS * scale


def _leaf(name: str, t: TensorType, role: str | None = None) -> Leaf:
    role = role or {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(t.shape)]
    return Leaf("vocab_read", role, name, t)


def _dispatch_sinks(policy: str, K: int = 3, rank: int = 0):
    k = _leaf("k", I32)
    t = S if rank == 0 else V3
    branches = [_leaf(f"b{i}", t) for i in range(K)]
    d = Dispatch(k, branches, policy, t)
    return (Assign("out", d, t),)


# --------------------------------------------------------------------------- #
# 1. The node — pure IR, no emission, no compile.
# --------------------------------------------------------------------------- #
def test_dispatch_is_in_the_op_vocabulary_beside_select():
    assert "dispatch" in OP_KINDS


def test_the_selector_must_be_a_rank0_i32_value():
    b0, b1 = _leaf("b0", S), _leaf("b1", S)
    bad = _leaf("bad", S)                      # f64, not i32
    with pytest.raises(HawkError, match="rank-0 i32"):
        Dispatch(bad, [b0, b1], "switch", S)
    bad_rank = _leaf("bad2", TensorType((3,), "i32"))
    with pytest.raises(HawkError, match="rank-0 i32"):
        Dispatch(bad_rank, [b0, b1], "switch", S)


def test_at_least_two_branches_are_required():
    k = _leaf("k", I32)
    with pytest.raises(HawkError, match="at least 2 branches"):
        Dispatch(k, [_leaf("b0", S)], "switch", S)


def test_every_branch_must_share_one_tensor_type():
    k = _leaf("k", I32)
    b0, b1 = _leaf("b0", S), _leaf("b1", V3)
    with pytest.raises(HawkError, match="disagree"):
        Dispatch(k, [b0, b1], "switch", S)


def test_an_unknown_policy_refuses_by_name():
    k = _leaf("k", I32)
    b0, b1 = _leaf("b0", S), _leaf("b1", S)
    with pytest.raises(HawkError, match="unknown policy"):
        Dispatch(k, [b0, b1], "auto", S)


def test_K_and_policy_are_readable_attributes():
    k = _leaf("k", I32)
    b0, b1, b2 = (_leaf(f"b{i}", S) for i in range(3))
    d = Dispatch(k, [b0, b1, b2], "predicated", S)
    assert d.K == 3
    assert d.policy == "predicated"
    assert d.selector is k
    assert d.branches == (b0, b1, b2)


def test_dispatch_defaults_to_switch_and_mirrors_selects_trace_entry():
    from hawk.trace.value import node_of
    k = hawk.Value(_leaf("k", I32))
    b0, b1 = hawk.Value(_leaf("b0", S)), hawk.Value(_leaf("b1", S))
    node = node_of(M.dispatch(k, [b0, b1]))
    assert node.kind == "dispatch"
    assert node.policy == "switch"


def test_the_three_policies_carry_distinct_unit_digests():
    """Policy is not cosmetic — a silent lowering-as-another-policy is
    exactly the defect this pins."""
    digests = {p: canonical(_dispatch_sinks(p)).digest for p in DISPATCH_POLICIES}
    assert len(set(digests.values())) == 3, digests


def test_sidecar_lists_every_dispatch_node():
    walk = canonical(_dispatch_sinks("switch", K=4))
    assert len(walk.dispatches) == 1
    info = walk.dispatches[0]
    assert (info.name, info.K, info.policy) == ("dispatch0", 4, "switch")


# --------------------------------------------------------------------------- #
# 1b. `_eval.py`'s own Dispatch arm — the serial-oracle harness asks for.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind_value", [-2, -1, 0, 1, 2, 3, 5])
def test_eval_harness_clamps_exactly_like_H97(kind_value):
    K = 3
    sinks = _dispatch_sinks("switch", K=K)      # policy is irrelevant here
    env = {"k": kind_value, "b0": 10.0, "b1": 20.0, "b2": 30.0}
    got = evaluate(sinks, env)["out"]
    want = [10.0, 20.0, 30.0][min(max(kind_value, 0), K - 1)]
    assert got == want


def test_eval_harness_is_exact_for_an_integer_branch_set():
    from hawk.ir.nodes import Dispatch as _D
    ik = _leaf("k", I32)
    ib0, ib1 = _leaf("b0", I32), _leaf("b1", I32)
    d = _D(ik, [ib0, ib1], "switch", I32)
    sinks = (Assign("out", d, I32),)
    for kind_value, want in ((-1, 7), (0, 7), (1, 9), (9, 9)):
        got = evaluate(sinks, {"k": kind_value, "b0": 7, "b1": 9})["out"]
        assert got == want


# --------------------------------------------------------------------------- #
# 2. Emission — string-level, no compile.
# --------------------------------------------------------------------------- #
def test_every_dispatch_materialises_the_shared_clamp():
    sinks = _dispatch_sinks("switch", K=3)
    text = render_body(sinks, canonical(sinks)).text
    assert "std::int32_t" in text
    assert "< 0 ?" in text and "2" in text       # K - 1 == 2 appears in the clamp


def test_switch_emits_a_declared_local_and_a_switch_statement():
    sinks = _dispatch_sinks("switch", K=3)
    text = render_body(sinks, canonical(sinks)).text
    assert "switch (" in text
    assert "case 0:" in text and "case 1:" in text
    assert "default:" in text


def test_predicated_rank0_emits_a_ternary_chain_and_no_switch():
    sinks = _dispatch_sinks("predicated", K=3)
    text = render_body(sinks, canonical(sinks)).text
    assert " ? " in text
    assert "switch (" not in text


def test_predicated_rank1_reaches_aether_select():
    sinks = _dispatch_sinks("predicated", K=3, rank=1)
    text = render_body(sinks, canonical(sinks)).text
    assert "aether::select(" in text


def test_a_literal_zero_branch_is_elided_under_switch():
    """The elision, exercised at the EMISSION level ("no
    case, no commit"): branch 1 is the structural zero `zero_like` mints, so
    its `case 1:` label must not appear and `default:` must carry the typed
    zero rather than the last branch's own expression."""
    k = _leaf("k", I32)
    b0, b2 = _leaf("b0", S), _leaf("b2", S)
    z = zero_like(S)
    d = Dispatch(k, [b0, z, b2], "switch", S)
    sinks = (Assign("out", d, S),)
    text = render_body(sinks, canonical(sinks)).text
    assert "case 1:" not in text
    assert "case 0:" in text
    assert "default:" in text
    assert "static_cast<Real>(0)" in text


def test_segmented_has_no_per_node_emission_yet():
    """A `segmented` dispatch is a per-UNIT split, done ahead of emission — by the
    time a renderer sees a body, the node must already be gone."""
    sinks = _dispatch_sinks("segmented", K=2)
    with pytest.raises(HawkError, match="segmented"):
        render_body(sinks, canonical(sinks))


def test_both_backends_share_one_dispatch_helper():
    from hawk.emit import BACKENDS
    text = {b.id: b.dispatch("hawk_disp0_k", ["b0", "b1"], "predicated")
            for b in BACKENDS.values()}
    assert text["cuda"] == text["host"]


# --------------------------------------------------------------------------- #
# 2a. Numeric: real artifacts, both backends, both built policies.
# --------------------------------------------------------------------------- #
def _branches(bs, k):
    """A PLAIN-PYTHON helper (never traced by the `ast` pass): the K static
    component reads a `Vector[K]` input unpacks into a branch list."""
    return [bs[i] for i in range(k)]


@hawk.kernel
def disp_r0_k2_pred(k: hawk.Index, bs: hawk.Vector[2], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 2), policy="predicated")


@hawk.kernel
def disp_r0_k2_switch(k: hawk.Index, bs: hawk.Vector[2], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 2), policy="switch")


@hawk.kernel
def disp_r0_k5_pred(k: hawk.Index, bs: hawk.Vector[5], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 5), policy="predicated")


@hawk.kernel
def disp_r0_k5_switch(k: hawk.Index, bs: hawk.Vector[5], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 5), policy="switch")


@hawk.kernel
def disp_r0_k16_pred(k: hawk.Index, bs: hawk.Vector[16], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 16), policy="predicated")


@hawk.kernel
def disp_r0_k16_switch(k: hawk.Index, bs: hawk.Vector[16], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 16), policy="switch")


@hawk.kernel
def disp_r1_k3_pred(k: hawk.Index, b0: hawk.Vector[3], b1: hawk.Vector[3], b2: hawk.Vector[3],
                    y: hawk.Mutable[hawk.Vector[3]]):
    y = M.dispatch(k, [b0, b1, b2], policy="predicated")


@hawk.kernel
def disp_r1_k3_switch(k: hawk.Index, b0: hawk.Vector[3], b1: hawk.Vector[3], b2: hawk.Vector[3],
                      y: hawk.Mutable[hawk.Vector[3]]):
    y = M.dispatch(k, [b0, b1, b2], policy="switch")


@hawk.kernel
def disp_int_pred(k: hawk.Index, b0: hawk.Index, b1: hawk.Index, y: hawk.Mutable[hawk.Index]):
    y = M.dispatch(k, [b0, b1], policy="predicated")


@hawk.kernel
def disp_int_switch(k: hawk.Index, b0: hawk.Index, b1: hawk.Index, y: hawk.Mutable[hawk.Index]):
    y = M.dispatch(k, [b0, b1], policy="switch")


@hawk.kernel
def disp_train_k4(k: hawk.Index, bs: hawk.Vector[4], y: hawk.Mutable[hawk.Scalar]):
    """The training-switch fixture (row): K = 4, `switch` — the
    DEFAULT policy, and the one this row cares about (a `switch`
    kernel is what a captured training step launches)."""
    y = M.dispatch(k, _branches(bs, 4), policy="switch")


@hawk.kernel
def disp_r0_k2_seg(k: hawk.Index, bs: hawk.Vector[2], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 2), policy="segmented")


@hawk.kernel
def disp_r0_k5_seg(k: hawk.Index, bs: hawk.Vector[5], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 5), policy="segmented")


@hawk.kernel
def disp_r0_k16_seg(k: hawk.Index, bs: hawk.Vector[16], y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(k, _branches(bs, 16), policy="segmented")


_KERNELS = [disp_r0_k2_pred, disp_r0_k2_switch, disp_r0_k5_pred, disp_r0_k5_switch,
            disp_r0_k16_pred, disp_r0_k16_switch, disp_r1_k3_pred, disp_r1_k3_switch,
            disp_int_pred, disp_int_switch, disp_train_k4]

#: The `segmented` units are `cross_sample_write` while
#: every kernel above is `sample_local` -- a bundle declares ONE execution
#: axis, so the segmented kernels build into their OWN bundle
#: (`segmented_bundle`) rather than joining `_KERNELS`.
_SEG_KERNELS = [disp_r0_k2_seg, disp_r0_k5_seg, disp_r0_k16_seg]

N = 40


@pytest.fixture(scope="module")
def dispatch_bundle(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("hawk_dispatch")
    return build_bundle(_KERNELS, root, targets=("cuda", "host"), cache_dir=cache_dir)


@pytest.fixture(scope="module")
def segmented_bundle(tmp_path_factory, cache_dir):
    """`build_bundle([kern])` expands EVERY segmented member into its K units
    (`hawk/ir/segment.py`) before publishing — this bundle's own artifacts are
    `disp_r0_k2_seg__k0/__k1`, `disp_r0_k5_seg__k0..__k4`,
    `disp_r0_k16_seg__k0..__k15`: 2 + 5 + 16 = 23 units, no
    `disp_r0_k*_seg` artifact of its own."""
    root = tmp_path_factory.mktemp("hawk_dispatch_seg")
    return build_bundle(_SEG_KERNELS, root, targets=("cuda", "host"),
                        cache_dir=cache_dir)


def _kind_plane(K: int, n: int = N) -> np.ndarray:
    """A kind plane that INCLUDES out-of-range values (negative and >= K),
    deterministic."""
    rng = np.random.default_rng(20260908)
    return rng.integers(-3, K + 3, size=n).astype(np.int64)


_CASES = [
    ("disp_r0_k2_pred", 2, 0, "f"), ("disp_r0_k2_switch", 2, 0, "f"),
    ("disp_r0_k5_pred", 5, 0, "f"), ("disp_r0_k5_switch", 5, 0, "f"),
    ("disp_r0_k16_pred", 16, 0, "f"), ("disp_r0_k16_switch", 16, 0, "f"),
    ("disp_r1_k3_pred", 3, 1, "f"), ("disp_r1_k3_switch", 3, 1, "f"),
    ("disp_int_pred", 2, 0, "i"), ("disp_int_switch", 2, 0, "i"),
]


def _case_kw_and_want(K: int, rank: int, dtype: str):
    kind_vals = _kind_plane(K)
    clamped = np.clip(kind_vals, 0, K - 1)
    if dtype == "i":
        b0 = np.arange(N, dtype=np.int64) + 100
        b1 = np.arange(N, dtype=np.int64) + 900
        want = np.where(clamped == 0, b0, b1)
        # a caller-supplied output plane is written IN PLACE on both targets
        # (hawk.runtime's and eagle's own rule, `_oracle.py`'s module note) —
        # supplied explicitly here at its TRUE int64 wire width rather
        # than relying on either allocator's own dtype inference for a
        # Mutable this fixture set is the first to declare integer-typed.
        return {"k": kind_vals, "b0": b0, "b1": b1,
                "y": np.zeros(N, dtype=np.int64)}, want
    if rank == 0:
        bs = (np.arange(K * N, dtype=np.float64).reshape(K, N) * 0.1
              + np.arange(K)[:, None] * 10.0)
        want = bs[clamped, np.arange(N)]
        return {"k": kind_vals, "bs": bs}, want
    branches = [np.arange(3 * N, dtype=np.float64).reshape(3, N) * 0.1 + j * 5.0
                for j in range(K)]
    want = np.stack(branches, axis=0)[clamped, :, np.arange(N)].T
    kw = {"k": kind_vals, **{f"b{j}": branches[j] for j in range(K)}}
    return kw, want


@pytest.mark.parametrize("name,K,rank,dtype", _CASES, ids=[c[0] for c in _CASES])
def test_forward_dispatch_matches_the_serial_oracle_on_the_host(
        dispatch_bundle, name, K, rank, dtype):
    import _oracle as O

    kw, want = _case_kw_and_want(K, rank, dtype)
    got = O.run(dispatch_bundle.directory, name, N, sidecar_of(dispatch_bundle, name),
               **kw)
    got = np.asarray(got)
    if dtype == "i":
        np.testing.assert_array_equal(got, want)
        return
    limit = _band(got.size, got, want)
    worst = float(np.max(np.abs(got.astype(float) - np.asarray(want, dtype=float))))
    assert worst <= limit, f"{name}: host dispatch differs from the reference by "\
        f"{worst}, band {limit}"


@pytest.mark.parametrize("name,K,rank,dtype", _CASES, ids=[c[0] for c in _CASES])
@pytest.mark.gpu
def test_forward_dispatch_matches_the_serial_oracle_on_the_device(
        dispatch_bundle, name, K, rank, dtype):
    import eagle.exec as eexec
    from eagle import plan as eplan

    kw, want = _case_kw_and_want(K, rank, dtype)
    plugin = device_plugin(dispatch_bundle.directory, name,
                           sidecar_of(dispatch_bundle, name))
    got = np.asarray(eplan.plan(plugin, structure=eexec.DeviceKernel,
                                npartitions=1).run(**kw))
    if dtype == "i":
        np.testing.assert_array_equal(got, want)
        return
    limit = _band(got.size, got, want)
    worst = float(np.max(np.abs(got.astype(float) - np.asarray(want, dtype=float))))
    assert worst <= limit, f"{name}: device dispatch differs from the reference by "\
        f"{worst}, band {limit}"


def test_the_dispatch_sidecar_entries_land_on_the_deployed_artifact(dispatch_bundle):
    """The deployed sidecar lists ``{name, K, policy}`` per kernel, which a
    downstream consumer reads without re-walking the IR."""
    sidecar = sidecar_of(dispatch_bundle, "disp_r0_k5_switch")
    assert sidecar["dispatch"] == [{"name": "dispatch0", "K": 5, "policy": "switch"}]
    sidecar = sidecar_of(dispatch_bundle, "disp_r1_k3_pred")
    assert sidecar["dispatch"] == [{"name": "dispatch0", "K": 3, "policy": "predicated"}]


# --------------------------------------------------------------------------- #
# 2b. Segmented: build([kern]) returns K units; each is run PER-UNIT (its
# own `count = cap_j`) against the SAME serial oracle predicated/switch match.
# --------------------------------------------------------------------------- #
_SEG_BASE = {2: "disp_r0_k2_seg", 5: "disp_r0_k5_seg", 16: "disp_r0_k16_seg"}

#: How far a unit's launch cap is padded PAST its true run length — the
#: idle tail ( asks for a `cap_j > run length` case explicitly).
_SEG_PAD = 3


def _sorted_kind_with_empty_run(K: int, n: int) -> tuple:
    """A SORTED kind plane over ``n`` samples with bucket ``K // 2``
    deliberately EMPTY (that bucket's run is included, and it is empty).
    Returns ``(kinds, counts)``."""
    empty = K // 2
    others = [j for j in range(K) if j != empty]
    base, extra = divmod(n, len(others))
    counts = [0] * K
    for idx, j in enumerate(others):
        counts[j] = base + (1 if idx < extra else 0)
    kinds = np.concatenate(
        [np.full(counts[j], j, dtype=np.int64) for j in range(K)])
    assert kinds.size == n
    return kinds, counts


def _seg_offsets(counts) -> np.ndarray:
    """The exclusive-prefix ``offsets`` plane (K+1 entries), built on the
    host in this test."""
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


def _seg_bs(K: int, n: int = N) -> np.ndarray:
    return (np.arange(K * n, dtype=np.float64).reshape(K, n) * 0.1
            + np.arange(K)[:, None] * 10.0)


def _seg_offsets_name(bundle, base_name: str) -> str:
    sidecar = sidecar_of(bundle, f"{base_name}__k0")
    return next(nm for role, nm in sidecar["arg_spec"] if role == "lookup")


@pytest.mark.parametrize("K", [2, 5, 16])
def test_segmented_forward_matches_the_serial_oracle_on_the_host(segmented_bundle, K):
    """A `segmented` dispatch's K units, launched per-unit with the offsets BUILT ON
    THE HOST here, reproduce the SAME serial oracle predicated/switch match —
    one bucket EMPTY, and every unit's `cap_j` PADDED past its true run
    length (the idle tail)."""
    import _oracle as O

    base_name = _SEG_BASE[K]
    kinds, counts = _sorted_kind_with_empty_run(K, N)
    offsets = _seg_offsets(counts)
    bs = _seg_bs(K)
    want = bs[kinds, np.arange(N)]
    offs_name = _seg_offsets_name(segmented_bundle, base_name)
    y = np.zeros(N, dtype=np.float64)
    for j in range(K):
        uname = f"{base_name}__k{j}"
        sidecar = sidecar_of(segmented_bundle, uname)
        kernel = O.load(segmented_bundle.directory, uname, sidecar)
        cap = counts[j] + _SEG_PAD               # cap_j > run length
        O.run_kernel(kernel, N, base=0, count=cap, bs=bs, y=y,
                    **{offs_name: offsets})
    limit = _band(N, y, want)
    worst = float(np.max(np.abs(y - want)))
    assert worst <= limit, (
        f"K={K}: segmented host run differs from the reference by {worst}, "
        f"band {limit}"
    )


@pytest.mark.parametrize("K", [2, 5, 16])
@pytest.mark.gpu
def test_segmented_forward_matches_the_serial_oracle_on_the_device(
        segmented_bundle, K):
    """The same K units, launched on the DEVICE — ``eagle.plan``'s own
    ``npartitions=1`` whole-view geometry (``count = N``, the SAME shape the
    predicated/switch device row above already uses), which is itself a
    ``cap_j > run length`` launch for every bucket here (``N=40`` against
    bucket sizes of a handful each): ``eagle.plan``'s partition model has no
    door for an explicit ``count`` SMALLER than the bound planes' own sample
    count (:meth:`eagle.plan.Plan.bind`'s own words: "the sample count comes
    off the bound planes' shape"), so the launch cap this row exercises is
    the WHOLE view and the unit's own OFFSETS-bounded early exit is what
    stops it past the true run length — the SAME mechanism, at a larger
    cap than the host row above uses."""
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    base_name = _SEG_BASE[K]
    kinds, counts = _sorted_kind_with_empty_run(K, N)
    offsets = _seg_offsets(counts)
    bs = _seg_bs(K)
    want = bs[kinds, np.arange(N)]
    offs_name = _seg_offsets_name(segmented_bundle, base_name)

    y_dev = cp.zeros(N, dtype=np.float64)
    bs_dev = cp.asarray(bs)
    off_dev = cp.asarray(offsets)
    for j in range(K):
        uname = f"{base_name}__k{j}"
        sidecar = sidecar_of(segmented_bundle, uname)
        plugin = device_plugin(segmented_bundle.directory, uname, sidecar)
        p = eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=1)
        p.bind(bs=bs_dev, y=y_dev, **{offs_name: off_dev}).launch()
    cp.cuda.get_current_stream().synchronize()
    got = cp.asnumpy(y_dev)
    limit = _band(N, got, want)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= limit, (
        f"K={K}: segmented device run differs from the reference by {worst}, "
        f"band {limit}"
    )


def test_segmented_units_carry_distinct_unit_digests(segmented_bundle):
    """The K units of ONE segmented kernel are K DIFFERENT compiled
    bodies (branch j substituted) — silently emitting the same body K times
    would be exactly the "silent lowering" defect pins for the three
    policies, one level down."""
    for K, base_name in _SEG_BASE.items():
        digests = {sidecar_of(segmented_bundle, f"{base_name}__k{j}")["digest"]
                  for j in range(K)}
        assert len(digests) == K, (base_name, digests)


def test_segmented_sidecar_carries_the_group_and_its_offsets_plane(segmented_bundle):
    """Each unit's own sidecar lists ``{name, K, policy, units}`` for the
    GROUP it belongs to (its own walk carries no Dispatch node any more —
     already replaced it), and its ``offsets`` plane like any other
    ``lookup`` plane (the ordinary ``arg_spec``/``buffers`` machinery, no
    special case)."""
    sidecar = sidecar_of(segmented_bundle, "disp_r0_k5_seg__k2")
    assert sidecar["dispatch"] == [{
        "name": "dispatch0", "K": 5, "policy": "segmented",
        "units": [f"disp_r0_k5_seg__k{j}" for j in range(5)],
    }]
    assert ("lookup", "dispatch0_offsets") in [tuple(p) for p in sidecar["arg_spec"]]


# --------------------------------------------------------------------------- #
# 3. Differentiation — Row .
# --------------------------------------------------------------------------- #
_H = 1e-5


def _dispatch_diff_primal(policy: str, K: int = 3):
    """A K-branch dispatch whose branch j reads ONLY leaf ``xj`` (a
    branch-PRIVATE parameter each, this fixture's own shape), through a
    nonlinear ``x**3`` so a swapped derivative is visible."""
    xs = [_leaf(f"x{i}", S) for i in range(K)]
    k = _leaf("k", I32)
    branches = [mk("mul", (mk("mul", (xs[i], xs[i])), xs[i])) for i in range(K)]
    d = Dispatch(k, branches, policy, S)
    return xs, k, (Assign("out", d, S),)


@pytest.mark.parametrize("policy", ["predicated", "switch", "segmented"])
def test_vjp_matches_finite_differences_and_is_exactly_zero_off_branch(policy):
    K = 3
    xs, _k, sinks = _dispatch_diff_primal(policy, K)
    rng = np.random.default_rng(20260908)
    derived = vjp(sinks)
    for kind_value in range(K):
        env = {"k": kind_value,
              **{f"x{i}": float(rng.uniform(0.5, 2.0)) for i in range(K)}}
        seed = float(rng.normal())
        got = evaluate(derived, {**env, "bar_out": seed})
        for i in range(K):
            actual = got[f"bar_x{i}"]
            if i != kind_value:
                # A branch-private parameter's gradient is EXACTLY zero
                # on an element of another kind, never merely small.
                assert actual == 0.0, (
                    f"{policy}: bar_x{i} is {actual} at kind={kind_value}, "
                    "branch-private but nonzero off its own branch")
                continue
            plus = dict(env)
            plus[f"x{i}"] = env[f"x{i}"] + _H
            minus = dict(env)
            minus[f"x{i}"] = env[f"x{i}"] - _H
            fplus = evaluate(sinks, plus)["out"]
            fminus = evaluate(sinks, minus)["out"]
            expect = (fplus - fminus) / (2 * _H) * seed
            assert actual == pytest.approx(expect, rel=1e-5, abs=1e-7), (
                f"{policy} kind={kind_value}: bar_x{i} = {actual} vs FD {expect}")


@pytest.mark.parametrize("policy", ["predicated", "switch", "segmented"])
def test_jvp_matches_finite_differences(policy):
    K = 3
    xs, _k, sinks = _dispatch_diff_primal(policy, K)
    rng = np.random.default_rng(20260908 + 1)
    derived = jvp(sinks)
    for kind_value in range(K):
        env = {"k": kind_value,
              **{f"x{i}": float(rng.uniform(0.5, 2.0)) for i in range(K)}}
        tangents = {f"x{i}": float(rng.normal()) for i in range(K)}
        got = evaluate(derived, {**env, **{f"dot_x{i}": v
                                           for i, v in enumerate(tangents.values())}})
        plus = {k2: v + _H * tangents[k2] for k2, v in env.items() if k2 != "k"}
        minus = {k2: v - _H * tangents[k2] for k2, v in env.items() if k2 != "k"}
        plus["k"] = minus["k"] = kind_value
        fplus = evaluate(sinks, plus)["out"]
        fminus = evaluate(sinks, minus)["out"]
        expect = (fplus - fminus) / (2 * _H)
        assert got["dot_out"] == pytest.approx(expect, rel=1e-5, abs=1e-7), (
            f"{policy} kind={kind_value}: JVP {got['dot_out']} vs FD {expect}")


_SWITCH_BLOCK_RE = re.compile(
    r"(?P<type>.+?) (?P<var>hawk_disp\d+);\n\s*switch \([^)]*\) \{\n"
    r"(?P<body>(?:.*\n)*?)\s*\}", re.M)


def _switch_blocks(text: str) -> dict:
    """``{declared local -> the set of case labels its own switch carries}``,
    parsed straight off the emitted text (no re-implementation of the
    emitter's own logic — this is an AUDIT of the string the row below asks
    for)."""
    return {m.group("var"): set(re.findall(r"case (\d+):", m.group("body")))
            for m in _SWITCH_BLOCK_RE.finditer(text)}


def test_no_accum_add_for_an_elided_branch_in_the_derived_source():
    """A `Table` read inside ONE branch reverses into a SCATTER
    whose value must be MASKED to that branch alone — the switch
    feeding `hawk_abi::accum_add` must carry a case for the branch that reads
    the table and NO case for the branch that never does, and there is
    exactly ONE `accum_add` call in the whole derived source (never a masked
    commit per branch, the shape names WRONG)."""
    k = _leaf("k", I32)
    x0 = _leaf("x0", S)
    tab = Leaf("table_read", "lookup", "tab", S)
    idx = _leaf("idx", I32)
    lane = _leaf("lane", I32)
    branch0 = mk("mul", (x0, x0))          # branch 0: reads x0 only
    branch1 = At(tab, idx, S)              # branch 1: reads the table only
    d = Dispatch(k, [branch0, branch1], "switch", S)
    sinks = (AccumWrite("acc", d, lane, S),)
    derived = vjp(sinks)
    text = render_body(derived, canonical(derived)).text

    assert text.count("hawk_abi::accum_add(") == 1, (
        "the scattered commit must fire ONCE, with a masked value -- not once "
        f"per branch:\n{text}"
    )
    call = next(line for line in text.splitlines() if "hawk_abi::accum_add(" in line)
    value_arg = call.strip().removeprefix("hawk_abi::accum_add(").split(",")[2].strip()
    value_arg = value_arg.rstrip(");")
    blocks = _switch_blocks(text)
    assert value_arg in blocks, (
        f"accum_add's value {value_arg!r} is not a declared switch local; "
        f"blocks found: {sorted(blocks)}\n{text}"
    )
    assert blocks[value_arg] == {"1"}, (
        f"the scatter's masked value must carry a case for branch 1 (the table "
        f"reader) and NONE for branch 0 (elided) — got {blocks[value_arg]}"
    )


# --------------------------------------------------------------------------- #
# 4. The training switch — Row (device).
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_training_switch_rewrites_the_kind_plane_without_rebuild_or_retrace(
        dispatch_bundle):
    """ONE captured graph, launched twice — between the launches
    only the kind plane changes (a plane WRITE, cupy, no rebuild): the second
    output is the NEW dispatch, and NOTHING about the artifact moved (the
    unit digest, the artifact directory's own files, and the graph object are
    all literally the SAME object/bytes across both launches), and
    ``hawk.artifact.bundle.build_bundle`` is called ZERO times between them
    (verified with a spy rather than by asserting the test never
    called it)."""
    import sys

    import cupy
    import eagle.exec as eexec
    from eagle import plan as eplan

    # the publisher module
    _build = sys.modules["hawk.artifact.bundle"]

    name = "disp_train_k4"
    K = 4
    sidecar = sidecar_of(dispatch_bundle, name)
    plugin = device_plugin(dispatch_bundle.directory, name, sidecar)
    p = eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=1)

    n = 32
    rng = np.random.default_rng(20260908)
    kind_first = rng.integers(0, K, size=n).astype(np.int64)
    bs_host = (np.arange(K * n, dtype=np.float64).reshape(K, n) * 0.1
              + np.arange(K)[:, None] * 10.0)

    kind_dev = cupy.asarray(kind_first)
    bs_dev = cupy.asarray(bs_host)
    y_dev = cupy.zeros(n, dtype=np.float64)

    bound = p.bind(k=kind_dev, bs=bs_dev, y=y_dev)

    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        bound.launch()
        graph = stream.end_capture()
    graph_id = id(graph)

    digest_before = dispatch_bundle.digest
    files_before = sorted(
        (str(q.relative_to(dispatch_bundle.directory)), q.read_bytes())
        for q in dispatch_bundle.directory.rglob("*") if q.is_file())

    calls = {"n": 0}
    real_build_bundle = _build.build_bundle

    def _spy(*a, **kw):
        calls["n"] += 1
        return real_build_bundle(*a, **kw)

    _build.build_bundle = _spy
    try:
        graph.launch(stream=stream)
        stream.synchronize()
        out1 = cupy.asnumpy(y_dev)

        # REWRITE the kind plane IN PLACE — a plane WRITE, cupy, no rebuild —
        # and launch the SAME graph object again.
        kind_second = (kind_first + 1) % K
        kind_dev[...] = cupy.asarray(kind_second)
        assert id(graph) == graph_id

        graph.launch(stream=stream)
        stream.synchronize()
        out2 = cupy.asnumpy(y_dev)
    finally:
        _build.build_bundle = real_build_bundle

    assert calls["n"] == 0, (
        f"no rebuild/re-trace between the two launches — build_bundle "
        f"was called {calls['n']} time(s)"
    )
    assert dispatch_bundle.digest == digest_before, (
        "the unit digest must not move between the two launches")
    files_after = sorted(
        (str(q.relative_to(dispatch_bundle.directory)), q.read_bytes())
        for q in dispatch_bundle.directory.rglob("*") if q.is_file())
    assert files_after == files_before, (
        "the artifact directory's own files must not move")

    want1 = bs_host[kind_first, np.arange(n)]
    want2 = bs_host[kind_second, np.arange(n)]
    np.testing.assert_allclose(out1, want1, rtol=0, atol=_band(n, out1, want1))
    np.testing.assert_allclose(out2, want2, rtol=0, atol=_band(n, out2, want2))
    assert not np.allclose(out1, out2), (
        "the rewritten kind plane must change the dispatch's own output — "
        "identical out1/out2 would mean the replay never re-read the plane")


@pytest.mark.gpu
def test_segmented_training_switch_rewrites_offsets_without_rebuild_or_retrace(
        segmented_bundle):
    """For `segmented`: ONE captured graph over ALL K units' launches,
    launched twice — between the launches the SORTED kind plane AND the
    offsets plane it implies are rewritten ON THE DEVICE, and the graph is
    replayed unchanged (no rebuild, no re-trace, no re-bind). Every unit's
    launch cap is fixed at N (a bind-time upper bound) so
    the SAME graph stays valid however the buckets' true run lengths move
    between the two arrangements — only each unit's OWN offsets-bounded early
    exit sees a different run."""
    import sys

    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    K = 5
    base_name = _SEG_BASE[K]
    n = 32
    offs_name = _seg_offsets_name(segmented_bundle, base_name)
    bs_host = (np.arange(K * n, dtype=np.float64).reshape(K, n) * 0.1
              + np.arange(K)[:, None] * 10.0)

    # A deliberately ASYMMETRIC histogram (not its own reverse) — the 
    # rows above already cover the empty-run case; this row's own point is
    # that the SAME graph survives a genuinely DIFFERENT arrangement.
    counts_first = [10, 8, 6, 4, 4]
    assert sum(counts_first) == n
    kind_first = np.concatenate(
        [np.full(counts_first[j], j, dtype=np.int64) for j in range(K)])
    offsets_first = _seg_offsets(counts_first)

    _build = sys.modules["hawk.artifact.bundle"]

    bs_dev = cp.asarray(bs_host)
    y_dev = cp.zeros(n, dtype=np.float64)
    off_dev = cp.asarray(offsets_first)

    bounds = []
    for j in range(K):
        uname = f"{base_name}__k{j}"
        plugin = device_plugin(segmented_bundle.directory, uname,
                               sidecar_of(segmented_bundle, uname))
        p = eplan.plan(plugin, structure=eexec.DeviceKernel, npartitions=1)
        bounds.append(p.bind(bs=bs_dev, y=y_dev, **{offs_name: off_dev}))

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        for bound in bounds:
            bound.launch(stream=stream)
        graph = stream.end_capture()
    graph_id = id(graph)

    digest_before = segmented_bundle.digest
    files_before = sorted(
        (str(q.relative_to(segmented_bundle.directory)), q.read_bytes())
        for q in segmented_bundle.directory.rglob("*") if q.is_file())

    calls = {"n": 0}
    real_build_bundle = _build.build_bundle

    def _spy(*a, **kw):
        calls["n"] += 1
        return real_build_bundle(*a, **kw)

    _build.build_bundle = _spy
    try:
        graph.launch(stream=stream)
        stream.synchronize()
        out1 = cp.asnumpy(y_dev)

        # A DIFFERENT sorted arrangement — reversed bucket order, still SORTED
        # and still an exclusive-prefix histogram over the same n=32 — written
        # on the device: both the kind plane's OWN meaning (not read by the
        # unit any more) and the offsets it implies move.
        counts_second = list(reversed(counts_first))
        kind_second = np.concatenate(
            [np.full(counts_second[j], j, dtype=np.int64) for j in range(K)])
        offsets_second = _seg_offsets(counts_second)
        off_dev[...] = cp.asarray(offsets_second)
        assert id(graph) == graph_id

        graph.launch(stream=stream)
        stream.synchronize()
        out2 = cp.asnumpy(y_dev)
    finally:
        _build.build_bundle = real_build_bundle

    assert calls["n"] == 0, (
        f"no rebuild/re-trace between the two launches — build_bundle "
        f"was called {calls['n']} time(s)"
    )
    assert segmented_bundle.digest == digest_before, (
        "the unit digest must not move between the two launches")
    files_after = sorted(
        (str(q.relative_to(segmented_bundle.directory)), q.read_bytes())
        for q in segmented_bundle.directory.rglob("*") if q.is_file())
    assert files_after == files_before, (
        "the artifact directory's own files must not move")

    want1 = bs_host[kind_first, np.arange(n)]
    want2 = bs_host[kind_second, np.arange(n)]
    np.testing.assert_allclose(out1, want1, rtol=0, atol=_band(n, out1, want1))
    np.testing.assert_allclose(out2, want2, rtol=0, atol=_band(n, out2, want2))
    assert not np.allclose(out1, out2), (
        "the rewritten offsets plane must change the segmented run's own "
        "output — identical out1/out2 would mean the replay never re-read it")


def test_segmented_units_are_divergence_free_by_construction():
    """The STATIC form of (ii):
    a segmented unit's body carries NO dispatch combinator (no `switch`, no
    select chain on a kind) and never reads the kind plane — the "divergence-
    free by construction" claim asserted EXACTLY, because at few-percent effect
    sizes the timing form flipped run-to-run on identical code."""
    for K in (2, 5, 16):
        units = split_segmented("seg", _dispatch_sinks("segmented", K=K))
        assert units is not None and len(units) == K
        for j, unit in enumerate(units):
            assert not any(isinstance(n, Dispatch) for n in unit.walk.order), (
                f"unit {unit.name} (K={K}, j={j}) still carries a Dispatch node")
            text = render_body(unit.sinks, unit.walk).text
            assert "switch (" not in text and "switch(" not in text, (
                f"unit {unit.name} emits a switch:\n{text}")
            assert "hawk_dispatch_k" not in text and " k;" not in text and "(k)" not in text, (
                f"unit {unit.name} reads the kind plane:\n{text}")
            assert f"b{j}" in text, (
                f"unit {unit.name} must evaluate branch {j} alone:\n{text}")


# --------------------------------------------------------------------------- #
# The kind plane's WRITER may be a KERNEL inside the
# same captured graph -- the router's arg-max. Two kernels, ONE graph.
# --------------------------------------------------------------------------- #
@hawk.kernel
def argmax_router(logits: hawk.Vector[4], kind: hawk.Mutable[hawk.Index]):
    kind = M.argmax(logits)


def _argmax_consumer_branches(x):
    return [x * 1.0, x * 2.0, x * 3.0, x * 4.0]


@hawk.kernel
def argmax_consumer(x: hawk.Scalar, kind: hawk.Index, y: hawk.Mutable[hawk.Scalar]):
    y = M.dispatch(kind, _argmax_consumer_branches(x), policy="switch")


@pytest.mark.gpu
def test_a_kernel_written_kind_plane_is_consumed_in_the_same_captured_graph(
        tmp_path):
    """The kind plane's writer is a KERNEL --
    ``kind = argmax(logits)`` -- and a dispatch kernel reads that plane
    in the SAME captured graph, so a switch involves no host at all. Launch
    once; rewrite ONLY the evidence (the logits plane, a cupy write) and
    launch the same graph object again: the kind plane and the dispatch both
    follow, ``build_bundle`` is never called, and the host never touches the
    kind plane between the two launches. The host oracle runs the same
    router first (the Index plane comes back INTEGER-valued).

    Needs ``kind = M.argmax(logits)`` to actually route by argmax, or
    the consumer returns ``x * 1`` everywhere."""
    import sys

    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    import hawk.runtime as RT

    bundle = build_bundle([argmax_router, argmax_consumer],
                          tmp_path / "h100_writer", targets=("cuda", "host"))
    n = 256
    rng = np.random.default_rng(20260909)
    logits = rng.standard_normal((4, n))
    want = logits.argmax(axis=0)

    sc_r = sidecar_of(bundle, "argmax_router")
    sc_c = sidecar_of(bundle, "argmax_consumer")
    kind_host = np.zeros(n, dtype=np.int64)
    RT.run(RT.load(bundle.directory, "argmax_router", sc_r), base=0, count=n,
           n_samples=n, logits=logits, kind=kind_host)
    assert np.array_equal(kind_host, want)
    assert np.issubdtype(kind_host.dtype, np.integer)

    router = eplan.plan(device_plugin(bundle.directory, "argmax_router", sc_r),
                        structure=eexec.DeviceKernel, npartitions=1)
    consumer = eplan.plan(
        device_plugin(bundle.directory, "argmax_consumer", sc_c),
        structure=eexec.DeviceKernel, npartitions=1)
    logits_dev = cp.asarray(logits)
    kind_dev = cp.zeros(n, dtype=np.int64)
    x = np.arange(n, dtype=np.float64)
    x_dev = cp.asarray(x)
    y_dev = cp.zeros(n, dtype=np.float64)
    bound_router = router.bind(logits=logits_dev, kind=kind_dev)
    bound_consumer = consumer.bind(x=x_dev, kind=kind_dev, y=y_dev)

    # The seeds above (cp.asarray / cp.zeros) ran on the legacy stream, which
    # a non-blocking stream is NOT ordered after: under GPU load the zero
    # fill of kind_dev can land after the graph's own write.
    cp.cuda.Stream.null.synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        bound_router.launch()
        bound_consumer.launch()
        graph = stream.end_capture()

    _build = sys.modules["hawk.artifact.bundle"]
    calls = {"n": 0}
    real_build_bundle = _build.build_bundle

    def _spy(*a, **kw):
        calls["n"] += 1
        return real_build_bundle(*a, **kw)

    _build.build_bundle = _spy
    try:
        graph.launch(stream=stream)
        stream.synchronize()
        kind_first = cp.asnumpy(kind_dev)
        y_first = cp.asnumpy(y_dev)
        # EVIDENCE rewrite only -- the kind plane is the graph's own to write.
        logits_dev[...] = cp.asarray(-logits)
        cp.cuda.Stream.null.synchronize()  # same legacy-stream seed, same order
        graph.launch(stream=stream)
        stream.synchronize()
        kind_second = cp.asnumpy(kind_dev)
        y_second = cp.asnumpy(y_dev)
    finally:
        _build.build_bundle = real_build_bundle

    want_second = (-logits).argmax(axis=0)
    assert np.array_equal(kind_first, want)
    assert np.allclose(y_first, x * (want + 1))
    assert np.array_equal(kind_second, want_second)
    assert np.allclose(y_second, x * (want_second + 1))
    assert not np.array_equal(kind_first, kind_second)
    assert calls["n"] == 0, (
        f"build_bundle was called {calls['n']} time(s) between launches")
