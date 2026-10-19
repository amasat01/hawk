# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.param_planes(kernel)`` — uniforms read from one-cell planes.

Rows (host; the ``cuda`` arm of a compiled row is marked ``gpu``):

* **plain untouched** — the plain kernel's digest and emitted text do not
  change by promoting;
* **bitwise** — promoted-form results equal the by-value results exactly;
* **rebind** — changing the cell between two launches of ONE built artifact
  changes the result, no recompile;
* **vjp** — the gradient with respect to a promoted uniform is one sum per
  32 samples (plus its ``_comp`` term) whose total equals the
  finite-difference ``dJ/dk``; the plain kernel's per-sample gradient,
  summed the same way, gives the same numbers;
* **sidecar** — ``param_planes`` lists the promoted planes (absent on the
  plain kernel); the access class is the plain kernel's;
* **refusals** — a name clash with an existing plane, an unknown name.
"""

from __future__ import annotations

import os

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk.artifact import build_bundle
from hawk.diff import param_planes, vjp
from hawk.ir import HawkError

N = 29


@kernel
def law(x: Scalar, k: Param, b: Param, y: Mutable[Scalar]):
    y = k * x * x + b * hawk.math.sin(x)


@kernel
def clash(x: Scalar, k: Param, y: Mutable[Scalar]):
    y = k * x


@kernel
def clash_plane(x: Scalar, k: Scalar, y: Mutable[Scalar]):
    y = k * x


PROMOTED = param_planes(law)
VJP = vjp(PROMOTED, wrt=("x", "k", "b"))
VJP_PLAIN = vjp(law, wrt=("x", "k", "b"))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    from hawk import Kernel

    root = tmp_path_factory.mktemp("param_planes")
    return build_bundle([law, PROMOTED, Kernel("law_pvjp", VJP),
                         Kernel("law_vjp", VJP_PLAIN)], root, targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native")


def _launch(bundle, name, target, **planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = (L.host_plugin(bundle.directory, name, sidecar),
                             eexec.HostTeam)
    else:
        plugin, structure = (L.device_plugin(bundle.directory, name, sidecar),
                             eexec.DeviceKernel)
    bound = eplan.plan(plugin, structure=structure).bind(**planes)
    bound.launch()
    return bound


def _x():
    return np.random.default_rng(5).uniform(-1.0, 1.0, N)


def test_plain_kernel_is_unchanged_by_promoting(bundle, tmp_path, cache_dir):
    digest = law.walk.digest
    param_planes(law)
    assert law.walk.digest == digest
    assert sidecar_of(bundle, "law")["digest"] == digest
    alone = build_bundle([law], tmp_path, targets=("host",), cache_dir=cache_dir,
                         host_profile="native")
    assert sidecar_of(alone, "law")["digest"] == digest
    mixed = sorted(p for p in bundle.directory.glob("law.*") if p.suffix not in (".so",))
    pure = sorted(p for p in alone.directory.glob("law.*") if p.suffix not in (".so",))
    texts = lambda ps: {p.suffix: p.read_bytes() for p in ps if p.suffix in (".cpp", ".cu")}
    host = texts(pure)
    assert host and all(texts(mixed)[k] == v for k, v in host.items())


def test_promoted_form_names_and_roles():
    spec = dict((n, r) for r, n in PROMOTED.walk.arg_spec)
    assert spec["k"] == "lookup" and spec["b"] == "lookup"
    assert PROMOTED.name == "law_params"


def test_sidecar_marks_promoted_planes_and_keeps_the_access_class(bundle):
    plain, promoted = sidecar_of(bundle, "law"), sidecar_of(bundle, "law_params")
    assert "param_planes" not in plain
    assert promoted["param_planes"] == ["k", "b"] or sorted(
        promoted["param_planes"]) == ["b", "k"]
    assert promoted["exec_access"] == plain["exec_access"] == "sample_local"
    assert sidecar_of(bundle, "law_pvjp")["param_planes"]
    assert promoted["params"] == []


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_promoted_equals_by_value_bitwise(bundle, target):
    xp = np
    if target == "cuda":
        import cupy as xp
    x = xp.asarray(_x())
    y_val, y_cell = xp.zeros(N), xp.zeros(N)
    _launch(bundle, "law", target, x=x, k=1.7, b=-0.4, y=y_val)
    _launch(bundle, "law_params", target, x=x, k=xp.asarray([1.7]),
            b=xp.asarray([-0.4]), y=y_cell)
    np.testing.assert_array_equal(np.asarray(y_cell.get() if target == "cuda"
                                             else y_cell),
                                  np.asarray(y_val.get() if target == "cuda"
                                             else y_val))


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_changing_the_cell_between_launches_changes_the_result(bundle, target):
    xp = np
    if target == "cuda":
        import cupy as xp
    x, k, b, y = xp.asarray(_x()), xp.asarray([1.0]), xp.asarray([0.0]), xp.zeros(N)
    bound = _launch(bundle, "law_params", target, x=x, k=k, b=b, y=y)
    first = np.array(y.get() if target == "cuda" else y)
    k[0] = 3.0
    bound.launch()
    second = np.array(y.get() if target == "cuda" else y)
    np.testing.assert_array_equal(first, np.asarray(_x() ** 2))
    np.testing.assert_array_equal(second, (3.0 * _x()) * _x())


def _loss_gradients(bundle, name, k, b):
    x = _x()
    bar_y = np.ones(N)
    sc = sidecar_of(bundle, name)
    planes = {"x": x, "bar_y": bar_y, "bar_k": np.zeros(N), "bar_b": np.zeros(N),
              "bar_x": np.zeros(N)}
    if name == "law_pvjp":
        rows = -(-N // 32)
        planes.update(k=np.array([k]), b=np.array([b]))
        planes.update({n: np.full(rows, np.nan) for n in
                       ("bar_k", "bar_k_comp", "bar_b", "bar_b_comp")})
    else:
        planes.update(k=k, b=b)
    assert {"bar_k", "bar_b"} <= {n for _, n in sc["arg_spec"]}
    _launch(bundle, name, "host", **planes)
    if name == "law_pvjp":
        return (planes["bar_k"] + planes["bar_k_comp"],
                planes["bar_b"] + planes["bar_b_comp"])
    return planes["bar_k"], planes["bar_b"]


def _per_32(plane):
    return np.add.reduceat(plane, np.arange(0, plane.size, 32))


def test_vjp_gradient_is_summed_per_32_samples_and_matches_finite_difference(bundle):
    k, b, h = 1.3, 0.7, 1e-6
    gk, gb = _loss_gradients(bundle, "law_pvjp", k, b)
    assert gk.shape == (-(-N // 32),) and gb.shape == (-(-N // 32),)
    x = _x()

    def loss(kk, bb):
        return float(np.sum(kk * x * x + bb * np.sin(x)))

    assert abs(gk.sum() - (loss(k + h, b) - loss(k - h, b)) / (2 * h)) < 1e-6
    assert abs(gb.sum() - (loss(k, b + h) - loss(k, b - h)) / (2 * h)) < 1e-6
    pk, pb = _loss_gradients(bundle, "law_vjp", k, b)
    np.testing.assert_allclose(gk, _per_32(pk), rtol=1e-14)
    np.testing.assert_allclose(gb, _per_32(pb), rtol=1e-14)


def test_a_name_clash_and_an_unknown_name_are_refused():
    with pytest.raises(HawkError, match="unknown|not uniform"):
        param_planes(law, ["nope"])
    with pytest.raises(HawkError, match="reads no uniform"):
        param_planes(clash_plane)
    assert param_planes(clash, ["k"]).name == "clash_params"


def test_a_plane_already_named_like_the_uniform_is_refused():
    from hawk import Kernel
    from hawk.ir import Assign, At, Const, Leaf
    from hawk.ir import make as mk
    from hawk.types import TensorType

    s, i = TensorType((), "f64"), TensorType((), "i32")
    uniform = Leaf("uniform", "uniform", "k", s)
    table = At(Leaf("table_read", "lookup", "k", s), Const(0, i), s)
    clashing = Kernel("clashing", (Assign("y", mk("add", (uniform, table)), s),))
    with pytest.raises(HawkError, match="already exists"):
        param_planes(clashing)


def test_an_automatic_kernel_is_promoted_through_its_step_and_tapes():
    from hawk.diff import tape

    promoted = param_planes(AUTO)
    assert promoted.step is not None and promoted.name == "auto_step_params"
    assert ("lookup", "dt") in promoted.walk.arg_spec
    assert ("lookup", "dt") in tape(promoted, 4).walk.arg_spec


@kernel
def auto_step(dt: Param, t_final: Scalar, terminated: Terminated,
              x: Mutable[Scalar], t: Mutable[Scalar]):
    tn = t + dt
    x = x + dt * x
    t = tn
    terminated = tn >= t_final


AUTO = auto_step


def test_the_cell_is_loaded_once_before_the_sample_loop(bundle):
    plain = (bundle.directory / "law.cpp").read_text()
    cell = (bundle.directory / "law_params.cpp").read_text()
    loop = cell[cell.index("for (aether::offset_t hawk_j"):]
    assert "lut_k_cell = lut_k[" in cell and "lut_b_cell = lut_b[" in cell
    assert "lut_k[" not in loop and "lut_b[" not in loop      # no per-sample cell read
    assert "lut_k_cell" in loop
    # past the hoisted loads, the body is the by-value body with the names swapped
    body = lambda t: t[t.index("const auto psc_x_i"):].split("}\n}")[0]
    assert (body(plain).replace("uni_k", "lut_k_cell").replace("uni_b", "lut_b_cell")
            == body(cell))


def _cell_and_value_arms(bundle):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 1 << 18
    x, y1, y2 = np.random.default_rng(1).uniform(-1, 1, n), np.zeros(n), np.zeros(n)

    def bind(name, **planes):
        plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
        return eplan.plan(plugin, structure=eexec.HostTeam).bind(x=x, **planes)

    by_value = bind("law", k=1.7, b=-0.4, y=y1)
    cell = bind("law_params", k=np.array([1.7]), b=np.array([-0.4]), y=y2)
    return by_value, cell, y1, y2


def test_a_cell_gives_the_bits_the_immediate_gives(bundle):
    by_value, cell, y1, y2 = _cell_and_value_arms(bundle)
    by_value.launch()
    cell.launch()
    np.testing.assert_array_equal(y1, y2)


@pytest.mark.wall_clock
def test_a_cell_costs_what_the_immediate_costs(bundle):
    """Paired host timing, alternating arms, a large launch: the loop reads
    the hoisted value, so it vectorises like the by-value kernel."""
    import time

    by_value, cell, y1, y2 = _cell_and_value_arms(bundle)
    for arm in (by_value, cell):
        arm.launch()
    best = {"value": np.inf, "cell": np.inf}
    for _ in range(60):
        for key, arm in (("value", by_value), ("cell", cell)):
            t0 = time.perf_counter()
            arm.launch()
            best[key] = min(best[key], time.perf_counter() - t0)
    ratio = best["cell"] / best["value"]
    print(f"cell {best['cell'] * 1e6:.0f} us, value {best['value'] * 1e6:.0f} us, "
          f"ratio {ratio:.3f}, load {os.getloadavg()[0]:.1f}")
    assert ratio < 1.10, ratio
