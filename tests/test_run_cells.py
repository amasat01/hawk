# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.run`` binds cells, not rows, and finishes them on the host.

* **twin** — the cell is BITWISE the one eagle's ordered finish gives (sum, max,
  min, mean; scalar and three-column cells; ragged and aligned ``n``; both
  precisions); skipped without eagle;
* **numpy** — the cell is within rounding of numpy (exact for max and min);
* **scratch** — the rows are made once per kernel and sample count;
* **refusal** — binding rows instead of the cell names the cell.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math
from hawk import HawkError, Kernel, Mutable, Scalar, Vector, kernel
from hawk import _cells, runtime
from hawk.artifact import build_bundle

SIZES = (1, 31, 32, 33, 1000, 100003, 300007)
OPS = ("sum", "mean", "max", "min")


def _scalar(op):
    @kernel
    def cell(x: Scalar, t: Mutable[Scalar]):
        t = getattr(hawk, op)(x * x)
    return cell


@kernel
def wide(x: Scalar, t: Mutable[Vector[3]]):
    t = hawk.sum(hawk.math.vec(x, x * x, 1.5 * x))


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    # one bundle per kernel: a bundle declares one execution axis, and the
    # cells' ops differ
    kernels = {f"cell_{op}": _scalar(op).sinks for op in OPS} | {"wide": wide.sinks}
    bundles = {name: build_bundle([Kernel(name, sinks)],
                                  tmp_path_factory.mktemp(f"run_{name}_{request.param}"),
                                  mode=request.param, targets=("host",),
                                  cache_dir=cache_dir, host_profile="native")
               for name, sinks in kernels.items()}
    return bundles, request.param


def _x(n, mode):
    return np.random.default_rng(n).uniform(-2.0, 2.0, n).astype(mode)


def _hawk_cell(bundles, name, n, mode, columns=1):
    kernel_ = runtime.load(bundles[name].directory, name)
    t = np.full(columns, np.nan, dtype=mode)
    runtime.run(kernel_, x=_x(n, mode), t=t)
    return kernel_, t


def _eagle_cell(bundles, name, n, mode):
    bundle = bundles[name]
    eexec = pytest.importorskip("eagle.exec")
    from eagle import plan as eplan

    import _deploy as L
    from conftest import sidecar_of

    sidecar = sidecar_of(bundle, name)
    plugin = L.host_plugin(bundle.directory, name, sidecar)
    plugin._meta = sidecar      # the plan-able view drops ``cells``
    got = eplan.plan(plugin, structure=eexec.HostTeam).run(x=_x(n, mode))
    # one output comes back as the plane itself, several as a dict by name
    return got if isinstance(got, dict) else {"t": got}


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", OPS)
def test_a_cell_is_bitwise_eagles_ordered_cell(built, op, n):
    bundles, mode = built
    _, got = _hawk_cell(bundles, f"cell_{op}", n, mode)
    want = np.asarray(_eagle_cell(bundles, f"cell_{op}", n, mode)["t"])
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("n", SIZES)
def test_a_three_column_cell_is_bitwise_eagles_ordered_cell(built, n):
    bundles, mode = built
    _, got = _hawk_cell(bundles, "wide", n, mode, columns=3)
    out = _eagle_cell(bundles, "wide", n, mode)
    want = np.concatenate([np.asarray(out[f"t_{k}"]) for k in range(3)])
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", OPS)
def test_a_cell_equals_numpy(built, op, n):
    bundles, mode = built
    _, got = _hawk_cell(bundles, f"cell_{op}", n, mode)
    # the term is formed in the plane's precision, then reduced in float64
    x = _x(n, mode)
    t = (x * x).astype(np.float64)
    want = {"sum": t.sum(), "mean": t.sum() / n, "max": t.max(), "min": t.min()}[op]
    if op in ("max", "min"):
        assert float(got[0]) == want
    else:
        tol = 1e-14 if mode == "float64" else 1e-5
        np.testing.assert_allclose(float(got[0]), want, rtol=tol, atol=tol * np.abs(t).sum() / max(n, 1))


def test_the_rows_are_made_once_per_kernel_and_sample_count(built):
    bundles, mode = built
    kernel_ = runtime.load(bundles["cell_sum"].directory, "cell_sum")
    t = np.zeros(1, dtype=mode)
    before = _cells.made
    for _ in range(3):
        runtime.run(kernel_, x=_x(100, mode), t=t)
    assert _cells.made == before + 1
    first = kernel_._cell_scratches[100]
    runtime.run(kernel_, x=_x(100, mode), t=t)
    assert kernel_._cell_scratches[100] is first
    runtime.run(kernel_, x=_x(101, mode), t=t)
    assert _cells.made == before + 2


def test_binding_the_rows_instead_of_the_cell_names_the_cell(built):
    bundles, mode = built
    kernel_ = runtime.load(bundles["cell_sum"].directory, "cell_sum")
    with pytest.raises(HawkError, match=r"bind \['t'\]"):
        runtime.run(kernel_, x=_x(64, mode))
