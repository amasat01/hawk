# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Session fixtures for the artifact rows.

ONE built bundle per (kernel, kind, scalar mode) shape, in a session tmpdir,
under ONE session cache directory — so the content-closure cache is exercised
across the whole session and no row pays for another's compile. Every row that
LOADS or RUNS an artifact takes ``built``; rows that only read text take the
piece they need.
"""

from __future__ import annotations

import pathlib

import _deployable as D
import pytest

from hawk.artifact import build_bundle
from hawk.emit import BACKENDS

BOTH = ("cuda", "host")

#: ``(bundle name, kernels, kwargs)`` — the fixture set rows ///
#: / build between them.
_UNITS = (
    ("axpb", [D.axpb], {}),
    ("vec3", [D.vec3_scale], {}),
    ("vec3_f32", [D.vec3_scale], {"mode": "float32"}),
    ("mat", [D.mat_apply], {}),
    ("multi", [D.two_outputs], {}),
    ("gather", [D.gather], {}),
    ("energy", [D.energy], {}),
    ("speed", [D.speed], {}),
    ("scatter", [D.scatter], {}),
    ("fraction", [D.fraction], {}),
    ("spin", [D.spin], {}),
    ("two_wire", [D.two_wire], {}),
    ("vocab", [D.vocab], {}),
    ("scatter_c_plain", [D.scatter_c], {}),
    ("scatter_c_comp", [D.scatter_c], {"kind": D.COMPENSATED}),
    ("diag_guarded", [D.diagnostic], {}),
    ("diag_free", [D.diagnostic], {"kind": D.MASK_FREE}),
    ("diag_one_mask", [D.diagnostic_two], {}),
    ("diag_two_masks", [D.diagnostic_two], {"kind": D.TWO_MASKS}),
)


#: The MPI bed is EXCLUDED from the default collection, exactly as eagle's is:
#: its rows only mean anything under `mpirun -np 2`, they raise (never skip) when
#: their artifact is unset, and initialising MPI inside a session that is also
#: driving CUDA is how an interpreter dies at finalize. `tests/mpi/
#: check_hawk_rank_bed.sh` is the ONE way to run them, and it names the directory
#: directly — pytest's own rule for initial arguments, which this hook does not
#: apply to.
collect_ignore = ["mpi"]


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("hawk_compile_cache"))


@pytest.fixture(scope="session")
def built(tmp_path_factory, cache_dir) -> dict:
    """Every fixture bundle, built once. ``{name: Bundle}``.

    Host side under the EXACT ``native`` profile: the rows that run these
    bundles compare the host results with NumPy references and the serial
    oracle bit for bit, which only the exact profiles promise (the default
    fast profile contracts ``a*b + c`` into FMAs)."""
    root = tmp_path_factory.mktemp("hawk_artifacts")
    out = {}
    for name, kernels, kw in _UNITS:
        out[name] = build_bundle(kernels, pathlib.Path(root) / name,
                                 targets=BOTH, cache_dir=cache_dir,
                                 host_profile="native", **kw)
    return out


def sidecar_of(bundle, name: str) -> dict:
    """The named artifact's sidecar out of a built bundle."""
    return next(a.sidecar for a in bundle.artifacts if a.name == name)


__all__ = ["BOTH", "sidecar_of"]
assert BACKENDS  # the emitter's backend table is what `targets` names
