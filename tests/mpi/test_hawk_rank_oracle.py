# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's own half: the RANK-partitioned plane against ``hawk._core``'s
serial oracle, bit for bit.

TWO HALVES, ONE GATE. ``tests/mpi/check_hawk_rank_bed.sh`` builds a HAWK
artifact and then does two things with it: it hands it to EAGLE's own rank bed
(``eagle/python/tests/mpi/check_rank_bed.sh``, whose
``$EAGLE_RANK_BED_ARTIFACT`` is documented as the door), which certifies the
STRUCTURE — the contiguous rank cut, the replicated-input ruling, the F-e
refusal, the rank-ordered fold; and it runs THIS file, which certifies the
ANSWER against HAWK's own reference arm. eagle's bed compares a rank run against
eagle's serial ``HostTeam.run_serial``; this file compares it against
``hawk._core``, which is a different arm entirely and the one name as
the oracle. Neither half subsumes the other.

EVERY ROW RUNS ON EVERY RANK. The collectives inside ``RankPartition`` are
blocking, so a row that returned early on one rank would hang its peers; there
is deliberately no ``if rank == 0`` here and no ``pytest.skip`` — a rank that
skips a row its peers enter is the same deadlock wearing a green hat. The gate
compares the ranks' JUnit verdicts independently, so a row that quietly asserted
nothing on rank 1 would still be caught.

WHAT MAKES THE IDENTITY ROWS NON-VACUOUS. A distributed identity row passes
trivially if every rank simply ran the WHOLE view and the gather then overwrote
everything with the same numbers. So the identity rows also run a PRE-GATHER arm
(``gather=False``) and assert that outside this rank's own sub-partition the
plane still holds its sentinel — and this rank's expected slice is stated from
the CONTRACT (contiguous, first ``rem`` ranks take one extra), never read back
from ``RankPartition.local``, because asking the function under test what to
expect is how a row ends up comparing a defect to itself.

An earlier version of this test failed: the two ``gather=`` flags were SWAPPED — the gathered
arm run with ``gather=False`` and the pre-gather arm run with the gather on — so
each row was asked about the other row's state. Four of the ten rows went red on
both ranks::

    AssertionError: the rank-partitioned sample_local plane differs from the
    hawk._core serial oracle: 499 of 997 samples differ
    AssertionError: the rank-partitioned cross_sample_read plane differs from the
    hawk._core serial oracle: 499 of 997 samples differ

(and 498 of 997 on the other rank, which is the uneven split showing through —
each rank had written only its own share). The two pre-gather rows failed
symmetrically on ``this rank wrote samples belonging to another rank``. The plant
was then removed.

Worth recording what the numbers say: ~half the plane differing is exactly one
rank's share of a 997-sample prime split across two ranks, which is what makes
these rows a check on the GATHER and not merely on the arithmetic.
"""

from __future__ import annotations

import os
import pathlib

import _bed_artifact as B
import _oracle as O
import eagle.exec as eexec
import numpy as np
import pytest
from eagle import plan as eplan

#: A value no body in this artifact writes, so "still the sentinel" is
#: unambiguous evidence that nothing touched that sample on this rank.
SENTINEL = -12345.0

#: Prime, so the rank split is uneven and an off-by-one cannot hide in it.
N = 997

_EPS = np.finfo(np.float64).eps


@pytest.fixture(scope="module")
def units():
    """Every unit of the HAWK artifact under ``$EAGLE_RANK_BED_ARTIFACT``.

    RAISES, never skips, when the variable is unset: this bed exists to certify
    a real HAWK emission, and a run that quietly certified nothing is the exact
    defect the gate is built around."""
    root = os.environ.get("EAGLE_RANK_BED_ARTIFACT")
    if not root:
        raise RuntimeError(
            "$EAGLE_RANK_BED_ARTIFACT is unset: this bed drives a deployed HAWK "
            "artifact root. Run it through tests/mpi/check_hawk_rank_bed.sh, which "
            "builds the artifact, names its digest and exports the variable."
        )
    return B.units_by_access(pathlib.Path(root))


def _world():
    return eexec.RankPartition.world()


def _expected_slice(n: int):
    """THIS rank's ``(base, count)``, stated from the CONTRACT rather than read
    back from :meth:`eagle.exec.RankPartition.local`."""
    rank, size = _world()
    q, rem = divmod(n, size)
    count = q + (1 if rank < rem else 0)
    base = rank * q + (rank if rank < rem else rem)
    return base, count


def _band(s: int, a, b) -> float:
    """The ONE anchor, ``S x 2 x eps``, applied relatively — the same rule
    ``tests/test_serial_oracle.band`` states, restated (not imported) so this
    bed stands alone under ``mpirun``. Derived, never fitted."""
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return s * 2.0 * _EPS * scale


def _oracle(directory, sidecar, **kw):
    """HAWK's reference arm: ONE serial call over the whole range through
    ``hawk._core``. Runs identically on every rank — it is local, it
    touches no collective, and it is what the distributed answer is judged
    against."""
    return O.run(directory, sidecar["kernel"], N, sidecar, **kw)


def _x(n: int) -> np.ndarray:
    """No two samples alike, so a row comparing the wrong samples cannot pass by
    coincidence."""
    return 1.0 / (np.arange(n, dtype=np.float64) + 1.0)


def _plugin(directory, sidecar):
    """The plan-able view of a unit's HOST entry, self-checked by ``hawk._core`` exactly as any consumer's would be."""
    import _deploy as L

    return L.host_plugin(directory, sidecar["kernel"], sidecar)


def _device_plugin(directory, sidecar):
    import _deploy as L

    return L.device_plugin(directory, sidecar["kernel"], sidecar)


def _uniforms(sidecar) -> dict:
    return {name: 3.25 - 0.75 * i
            for i, (role, name) in enumerate(sidecar["arg_spec"]) if role == "uniform"}


def _input_name(sidecar) -> str:
    names = [n for r, n in sidecar["arg_spec"] if r == "per_sample"]
    assert len(names) == 1, f"this unit declares {names} per_sample inputs"
    return names[0]


def _output_name(sidecar) -> str:
    names = [n for r, n in sidecar["arg_spec"]
             if r in ("out", "mutable", "wide_out", "accum_out")]
    assert len(names) == 1, f"this unit declares {names} output planes"
    return names[0]


def _all_ranks(values) -> np.ndarray:
    """Every rank's copy of ``values`` as a ``(size, len)`` plane, built out of
    the gather already under test: over a world of ``size``,
    ``Partition.whole(size)`` gives each rank exactly one sample."""
    rank, size = _world()
    flat = np.ascontiguousarray(np.atleast_1d(np.asarray(values, dtype=np.float64)))
    plane = np.zeros((size, flat.size), dtype=np.float64)
    plane[rank, :] = flat
    eexec.RankPartition.allgather_plane(plane.ctypes.data,
                                        eexec.Partition.whole(size), int(flat.size))
    return plane


# --------------------------------------------------------------------------- #
# The world, and the artifact this bed was pointed at.
# --------------------------------------------------------------------------- #
def test_the_world_is_a_multi_rank_world():
    rank, size = _world()
    expected = int(os.environ.get("HAWK_MPI_BED_RANKS", "2"))
    assert size == expected, f"expected a world of {expected}, got {size}"
    assert size >= 2, "a one-rank world exercises nothing this bed exists to check"
    assert 0 <= rank < size


def test_the_artifact_under_test_is_a_hawk_emission(units):
    """The bed's INPUT is named, not assumed. A HAWK artifact carries the keys
    only HAWK writes — its ``Walk.digest`` and the two RESOLVED header roots — so a run against eagle's own hand-written fixture would be
    caught here rather than reported as a HAWK certification."""
    missing = [a for a in B.REQUIRED_ACCESS_CLASSES if a not in units]
    assert not missing, (
        f"the artifact declares {sorted(units)} and is missing {missing}; this bed "
        f"certifies every class in {list(B.REQUIRED_ACCESS_CLASSES)}"
    )
    for access, (_directory, sidecar) in units.items():
        assert sidecar["aether_abi"] == "aether-abi/2", access
        assert sidecar["schema_version"] == 2, access
        for key in ("digest", "aether_include", "eagle_include", "host_entry",
                    "host_artifact"):
            assert sidecar.get(key), f"{access}: sidecar carries no {key!r}"


# --------------------------------------------------------------------------- #
# The oracle rows.
# --------------------------------------------------------------------------- #
def test_the_core_serial_oracle_agrees_on_every_rank(units):
    """Before the oracle can judge anything it must be the SAME oracle on every
    rank — bit for bit, not merely plausible. It is a local, collective-free
    call, so a disagreement here would mean the ranks are running different
    binaries or different bytes."""
    directory, sidecar = units["sample_local"]
    kw = {_input_name(sidecar): _x(N), **_uniforms(sidecar)}
    mine = _oracle(directory, sidecar, **kw, **{_output_name(sidecar):
                                                np.full(N, SENTINEL)})
    plane = _all_ranks(mine)
    for r in range(1, plane.shape[0]):
        assert plane[r].tobytes() == plane[0].tobytes(), (
            f"rank {r}'s serial oracle differs BITWISE from rank 0's"
        )


@pytest.mark.parametrize("access", ["sample_local", "cross_sample_read"])
def test_the_gathered_plane_is_bit_identical_to_the_core_serial_oracle(units, access):
    """The claim, for the two classes that are BIT-invariant across ranks
    (``sample_local`` elementwise; ``cross_sample_read`` because inputs are
    REPLICATED, so every rank holds the whole plane — which is the
    condition for BIT-invariance, met by construction)."""
    directory, sidecar = units[access]
    xname, yname = _input_name(sidecar), _output_name(sidecar)
    kw = {xname: _x(N), **_uniforms(sidecar)}

    got = eplan.plan(_plugin(directory, sidecar), structure=eexec.RankPartition,
                     inner=eexec.HostTeam).run(**kw,
                                               **{yname: np.full(N, SENTINEL)})
    oracle = _oracle(directory, sidecar, **kw, **{yname: np.full(N, SENTINEL)})
    differing = int(np.count_nonzero(np.asarray(got) != np.asarray(oracle)))
    assert got.tobytes() == oracle.tobytes(), (
        f"the rank-partitioned {access} plane differs from the hawk._core serial "
        f"oracle: {differing} of {N} samples differ"
    )
    plane = _all_ranks(got)
    for r in range(1, plane.shape[0]):
        assert plane[r].tobytes() == plane[0].tobytes(), (
            f"rank {r}'s gathered plane differs from rank 0's"
        )


@pytest.mark.parametrize("access", ["sample_local", "cross_sample_read"])
def test_the_pre_gather_plane_holds_only_this_ranks_own_slice(units, access):
    """NON-VACUITY for the row above. Before the gather, this rank's plane must
    carry its OWN answers inside its sub-partition and the untouched sentinel
    everywhere else — otherwise every rank could have run the whole view and the
    gather would have hidden it."""
    directory, sidecar = units[access]
    xname, yname = _input_name(sidecar), _output_name(sidecar)
    kw = {xname: _x(N), **_uniforms(sidecar)}
    base, count = _expected_slice(N)

    pre = eplan.plan(_plugin(directory, sidecar), structure=eexec.RankPartition,
                     inner=eexec.HostTeam,
                     gather=False).run(**kw, **{yname: np.full(N, SENTINEL)})
    mine = np.zeros(N, dtype=bool)
    mine[base:base + count] = True
    assert not np.any(pre[mine] == SENTINEL), (
        "this rank left samples of its OWN sub-partition unwritten"
    )
    assert np.all(pre[~mine] == SENTINEL), (
        "this rank wrote samples belonging to another rank — the ranks are not "
        "running disjoint sub-partitions, and the gather would hide it"
    )
    oracle = _oracle(directory, sidecar, **kw, **{yname: np.full(N, SENTINEL)})
    assert pre[mine].tobytes() == np.asarray(oracle)[mine].tobytes(), (
        "this rank's OWN slice already disagrees with the serial oracle, before "
        "any gather"
    )


def test_the_device_inner_is_within_the_ruled_band_of_the_core_serial_oracle(units):
    """The device arm, deliberately UNMARKED (a ``gpu`` marker would turn it into
    a SKIP on a GPU-less box, and a skipped row inside a pinned manifest is green
    having certified nothing). Each rank launches over its own sub-partition on
    GPU-0 in its own context; the gather is host-staged ( ruled out
    CUDA-aware MPI). Banded, not bit-gated: host vs device is the third row."""
    directory, sidecar = units["sample_local"]
    xname, yname = _input_name(sidecar), _output_name(sidecar)
    kw = {xname: _x(N), **_uniforms(sidecar)}

    got = eplan.plan(_device_plugin(directory, sidecar),
                     structure=eexec.RankPartition,
                     inner=eexec.DeviceKernel).run(**kw,
                                                   **{yname: np.full(N, SENTINEL)})
    oracle = np.asarray(_oracle(directory, sidecar, **kw,
                                **{yname: np.full(N, SENTINEL)}))
    worst = float(np.max(np.abs(np.asarray(got) - oracle)))
    limit = _band(N, got, oracle)
    assert worst <= limit, (
        f"the rank-partitioned DEVICE plane differs from the hawk._core serial "
        f"oracle by {worst}, band S*2*eps*scale = {limit}"
    )
    plane = _all_ranks(got)
    for r in range(1, plane.shape[0]):
        assert plane[r].tobytes() == plane[0].tobytes()


def test_the_mapreduce_fold_is_within_the_band_of_the_core_serial_oracle(units):
    """/F-d. Each rank folds its own partials, the partials cross as ONE
    allgather, and every rank folds THEM in a fixed RANK ORDER. Band-gated
    because the combine order differs from the whole run by construction — and
    the band is proven not vacuous by requiring a result MISSING one rank's
    partial to fail it."""
    directory, sidecar = units["mapreduce"]
    xname = _input_name(sidecar)
    op = sidecar["exec_op"]
    assert op, "a mapreduce unit must declare exec_op (L8)"
    kw = {xname: _x(N), **_uniforms(sidecar)}

    partials = eplan.plan(_plugin(directory, sidecar), structure=eexec.RankPartition,
                          inner=eexec.HostTeam).run(**kw)
    whole = eexec.fold(op, np.asarray(_oracle(directory, sidecar, **kw)).tolist())

    base, count = _expected_slice(N)
    mine = eexec.fold(op, partials[base:base + count].tolist())
    split = eexec.RankPartition.allgather_fold(op, mine)

    limit = _band(N, np.asarray([whole]), np.asarray([split]))
    assert abs(whole - split) <= limit, (
        f"|{whole} - {split}| exceeds the ruled band {limit}"
    )
    assert abs(whole - mine) > limit, (
        "the band accepts a result missing an entire rank's partial — it is so "
        "wide that it certifies nothing"
    )


def test_a_cross_sample_write_body_is_refused_under_the_rank_structure(units):
    """F-e, on a HAWK-emitted body: ranks accumulating into one shared target
    would need a partial-accum combine that is not specified, so the placement is
    refused at PLAN time — at EVERY world size, naming the rule."""
    directory, sidecar = units["cross_sample_write"]
    with pytest.raises(ValueError) as excinfo:
        eplan.plan(_plugin(directory, sidecar), structure=eexec.RankPartition,
                   inner=eexec.HostTeam)
    message = str(excinfo.value)
    for fragment in ("cross_sample_write", "rank_partition", "F-e"):
        assert fragment in message, f"the refusal does not name {fragment!r}: {message}"
