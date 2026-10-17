# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``param_planes`` through a loop in the step body.

A loop reads its parameter as a loop-invariant boundary value, so promoting the
uniform to a one-cell plane read changes that boundary value and nothing else:
the loop is rebuilt over the new boundary and the cell is read once, before the
loop. Rows, for an accumulator, a recurrence, a nested loop and a loop with a
``break``, in float64 and float32, on the host and the device:

* the promoted kernel's values equal the ``Param`` kernel's (bit for bit on the
  host, to a few ulps on the device, whose compiler contracts the spellings apart);
* its ``vjp`` gives the same per-sample gradient plane, which sums to a central
  difference (float64);
* its ``jvp``, and the second-order routes ``jvp(vjp)`` and ``vjp(jvp)``, equal
  the ``Param`` kernel's.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Kernel, Mutable, Param, Scalar
from hawk.artifact import build_bundle
from hawk.diff import jvp, param_planes, vjp

N = 11
CAP = 6


@hawk.kernel
def accumulate(x: Scalar, g: Param, y: Mutable[Scalar]):
    acc = 0.0
    for k in range(8):
        acc = acc + g * m.sin(x + k)
    y = acc


@hawk.kernel
def recurrence(x: Scalar, g: Param, y: Mutable[Scalar]):
    acc = 0.0
    for _k in range(CAP):
        acc = m.tanh(g * acc + x)
    y = acc


@hawk.kernel
def nested(x: Scalar, g: Param, y: Mutable[Scalar]):
    total = 0.0
    for _i in range(3):
        inner = x
        for j in range(4):
            inner = inner + g * m.cos(inner + j)
        total = total + inner
    y = total


@hawk.kernel
def early_exit(x: Scalar, g: Param, y: Mutable[Scalar]):
    s = x
    for _k in range(CAP):
        if s > 1.5:
            break
        s = m.sin(g * s) + s * 1.2
    y = s


SUBJECTS = {"accumulate": accumulate, "recurrence": recurrence, "nested": nested,
            "early_exit": early_exit}


def _family(kernel):
    """``{label: Kernel}`` of the Param kernel and the promoted one, each with the
    derivatives the rows run (names tagged ``p`` for the promoted)."""
    out = {}
    for tag, base in (("", kernel), ("p_", param_planes(kernel))):
        stem = tag + kernel.name
        out[tag + "f"] = base
        out[tag + "v"] = Kernel(stem + "_v", vjp(base, wrt=("x", "g")))
        out[tag + "j"] = Kernel(stem + "_j", jvp(base, wrt=("x", "g")))
        out[tag + "jv"] = Kernel(stem + "_jv", jvp(out[tag + "v"]))
        out[tag + "vj"] = Kernel(stem + "_vj", vjp(out[tag + "j"]))
    return out


FAMILIES = {name: _family(k) for name, k in SUBJECTS.items()}
MODES = {"float64": np.float64, "float32": np.float32}


@pytest.fixture(scope="module", params=sorted(MODES))
def bundle(request, tmp_path_factory, cache_dir):
    kernels = [k for fam in FAMILIES.values() for k in fam.values()]
    return request.param, build_bundle(
        kernels, tmp_path_factory.mktemp("loops_" + request.param), mode=request.param,
        targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")


TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


def _run(built, kernel, target, values, dtype):
    import eagle.exec as eexec
    from eagle import plan as eplan

    xp = np
    if target == "cuda":
        import cupy as xp
    bound, outs = {}, {}
    for role, name in kernel.arg_spec:
        if role == "mutable":
            outs[name] = bound[name] = xp.zeros(N, dtype=dtype)
        elif role == "uniform":
            bound[name] = float(values[name])
        elif role == "lookup":
            bound[name] = xp.asarray(np.array([float(values[name])], dtype=dtype))
        else:
            bound[name] = xp.asarray(np.asarray(values[name], dtype=dtype))
    sidecar = sidecar_of(built, kernel.name)
    if target == "host":
        import eagle.exec as eexec
        plugin, structure = L.host_plugin(built.directory, kernel.name, sidecar), eexec.HostTeam
    else:
        plugin, structure = (L.device_plugin(built.directory, kernel.name, sidecar),
                             eexec.DeviceKernel)
    eplan.plan(plugin, structure=structure).bind(**bound).launch()
    return {n: np.asarray(a.get() if target == "cuda" else a, dtype=np.float64)
            for n, a in outs.items()}


def _point(kernel, dtype):
    rng = np.random.default_rng(len(kernel.name) + 3)
    out = {}
    for role, name in kernel.arg_spec:
        if role != "mutable":
            out[name] = (float(np.float32(0.8)) if role in ("uniform", "lookup")
                         else rng.uniform(0.2, 1.3, N).astype(dtype))
    return out


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("subject", sorted(SUBJECTS))
@pytest.mark.parametrize("level", ["f", "v", "j", "jv", "vj"])
def test_a_promoted_loop_kernel_equals_the_param_kernel_at_every_level(
        bundle, subject, level, target):
    mode, built = bundle
    plain, promoted = FAMILIES[subject][level], FAMILIES[subject]["p_" + level]
    assert [r for r, n in promoted.arg_spec if n == "g"] in (["lookup"], [])
    point = _point(plain, MODES[mode])
    point.update({n: v for n, v in _point(promoted, MODES[mode]).items() if n not in point})
    got = _run(built, promoted, target, point, MODES[mode])
    want = _run(built, plain, target, point, MODES[mode])
    assert got.keys() == want.keys()
    for n in want:
        if target == "host":
            np.testing.assert_array_equal(got[n], want[n])
        else:   # the device compiler contracts the two spellings differently: a few ulps
            eps = np.finfo(MODES[mode]).eps
            np.testing.assert_allclose(got[n], want[n], rtol=0,
                                       atol=16 * eps * max(1.0, float(np.abs(want[n]).max())))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("subject", sorted(SUBJECTS))
def test_the_promoted_gradient_is_per_sample_and_sums_to_a_difference(bundle, subject, target):
    mode, built = bundle
    if mode != "float64":
        pytest.skip("the float64 reference row")
    fam, h = FAMILIES[subject], 1e-6
    f, v = fam["p_f"], fam["p_v"]
    point = _point(v, np.float64)
    point["bar_y"] = np.ones(N)
    bar = _run(built, v, target, point, np.float64)["bar_g"]
    assert bar.shape == (N,)

    def total(g):
        return _run(built, f, target, {**point, "g": g}, np.float64)["y"].sum()

    fd = (total(point["g"] + h) - total(point["g"] - h)) / (2 * h)
    assert bar.sum() == pytest.approx(fd, rel=1e-6, abs=1e-6)
