# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.artifact.plugins``: one bundle -> ``{name: plugin}``, the objects
``eagle.plan`` takes, assembled in one place.

What each row proves:

* SHAPE: a host-only bundle's plugins carry the kernel's ``plan_view``
  declaration (minus ``host_entry``, replaced by the entry's ADDRESS), its
  name and a keepalive, and no ``device_function``; the device loader is never
  called for a bundle with no ``cuda`` target.
* EQUALITY: a ``plugins`` plugin runs under eagle's ``HostTeam`` bit-identical
  to the hand-assembled one (``_deploy.host_plugin``), for every kernel of a
  two-kernel bundle.
* DEVICE: with ``eagle.registry.load_manifest`` as the loader, the loader is
  called ONCE with the manifest path and every plugin carries a
  ``device_function`` that runs under ``DeviceKernel`` within one ULP of the
  host entry per multiply-add the device may contract.
* STAGES: a plugin carries its sidecar (read-only) and the plugins of its
  bundle, so a kernel built in stages runs through ``plugins`` and
  ``eagle.plan.auto`` as it does through a deploy: on the host equal bit for
  bit to the same chain built whole, on the device (a team stage) equal to NumPy.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest

import hawk
from hawk import Mutable, Param, Scalar, Vector
from hawk.artifact import build_bundle, plan_view, plugins
from hawk.math import dot

N = 33


@hawk.kernel
def pl_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def pl_norm2(v: Vector[3], s: Scalar, e: Mutable[Scalar]):
    e = dot(v, v) * s


#: The multiply-adds the device may contract in each kernel, each moving a
#: result by at most one ULP (positive terms, no cancellation): the bound of
#: the device-vs-host row.
_CONTRACTIONS = {"pl_affine": 1, "pl_norm2": 3}


def _inputs():
    rng = np.random.default_rng(5)
    # positive terms: no cancellation, so a contracted multiply-add on the
    # device moves a result by at most one ULP of it
    return {"pl_affine": {"x": rng.uniform(0.5, 2.0, N), "a": 1.5, "b": 0.25},
            "pl_norm2": {"v": rng.normal(size=(3, N)), "s": rng.uniform(0.5, 2, N)}}


@pytest.fixture(scope="module")
def host_bundle(tmp_path_factory):
    return build_bundle([pl_affine, pl_norm2], tmp_path_factory.mktemp("pl_host"),
                        targets=("host",))


def test_a_host_bundle_gives_host_entries_and_the_plan_view(host_bundle):
    calls = []
    got = plugins(host_bundle, device_loader=lambda path: calls.append(path))
    assert calls == []                       # no cuda target: never loaded
    assert list(got) == ["pl_affine", "pl_norm2"]
    for art in host_bundle.artifacts:
        p = got[art.name]
        view = plan_view(art.sidecar)
        view.pop("host_entry")
        assert p.name == art.name
        assert isinstance(p.host_entry, int) and p.host_entry != 0
        assert not hasattr(p, "device_function")
        assert p._keepalive
        for key, value in view.items():
            assert getattr(p, key) == value, key


def test_plugins_run_bit_identical_to_the_hand_assembled_plugin(host_bundle):
    eexec = pytest.importorskip("eagle.exec")
    from eagle import plan as eplan

    got = plugins(host_bundle)
    for art in host_bundle.artifacts:
        inputs = _inputs()[art.name]
        hand = L.host_plugin(host_bundle.directory, art.name, art.sidecar)
        want = eplan.plan(hand, structure=eexec.HostTeam).run(**inputs)
        have = eplan.plan(got[art.name], structure=eexec.HostTeam).run(**inputs)
        assert np.asarray(have).tobytes() == np.asarray(want).tobytes()
        assert len(set(np.asarray(have).tolist())) == N   # distinct rows


@pytest.mark.gpu
def test_the_device_loader_is_called_once_and_gives_device_functions(
        tmp_path_factory):
    cp = pytest.importorskip("cupy")
    eexec = pytest.importorskip("eagle.exec")
    from eagle import plan as eplan
    from eagle.registry import load_manifest

    bundle = build_bundle([pl_affine, pl_norm2], tmp_path_factory.mktemp("pl_dev"),
                          targets=("host", "cuda"))
    calls = []

    def loader(path):
        calls.append(path)
        return load_manifest(path)

    got = plugins(bundle, device_loader=loader)
    assert calls == [bundle.manifest_path]
    for art in bundle.artifacts:
        inputs = _inputs()[art.name]
        p = got[art.name]
        assert p.device_function and p.host_entry
        host = np.asarray(eplan.plan(p, structure=eexec.HostTeam).run(**inputs))
        dev = eplan.plan(p, structure=eexec.DeviceKernel).run(
            **{k: (cp.asarray(v) if isinstance(v, np.ndarray) else v)
               for k, v in inputs.items()})
        ulps = np.abs(cp.asnumpy(dev).view(np.int64) - host.view(np.int64))
        assert int(ulps.max()) <= _CONTRACTIONS[art.name], art.name


def test_a_derivative_plugin_carries_its_tag_and_a_primal_carries_none(tmp_path):
    """A forward-only door (a run loop, a simulation, an automatic plan)
    refuses a derivative kernel by reading ``plugin.derivative``: a PREBUILT
    plugin must carry the same tag the sidecar does, and ``None`` for a primal."""
    from hawk.diff import vjp

    grad = hawk.Kernel("pl_affine_vjp", vjp(pl_affine, wrt=("a", "b")))
    bundle = build_bundle([pl_affine, grad], tmp_path, targets=("host",))
    got = plugins(bundle)
    assert got["pl_affine"].derivative is None
    tag = got["pl_affine_vjp"].derivative
    assert tag == {"kind": "vjp", "wrt": ("a", "b"), "primal": "pl_affine",
                   "primal_unit": None}
    sidecar = next(a.sidecar for a in bundle.artifacts if a.name == "pl_affine_vjp")
    assert plan_view(sidecar)["derivative"]["primal"] == sidecar["derivative"]["primal"]


# ---- a kernel built in stages, through plugins() and eagle's plan ------------------------------------------------------

def _fan(tmp_path, monkeypatch, *, cut: bool):
    """The ten-member fan of :mod:`test_chain` on the host, cut in two stages (an AVX-512 register file) or whole."""
    from test_chain import fan_in

    from hawk.ir import fusion
    from hawk.targets.cards import fusion as fusion_card

    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fusion, "host_isa", lambda: "avx512")
    if not cut:
        monkeypatch.setattr(fusion, "row", lambda st, host=False: {**fusion_card(st, "avx512"), "R_file": 4096})
    fan = hawk.chain(*fan_in(10), name="fan")
    bundle = build_bundle([fan], tmp_path / ("cut" if cut else "whole"), targets=("host",), mode="float64",
                          host_profile="native")
    return bundle, next(a.sidecar for a in bundle.artifacts if a.name == "fan")


def test_a_plugin_carries_its_sidecar_and_the_plugins_of_its_bundle(tmp_path, monkeypatch):
    bundle, side = _fan(tmp_path, monkeypatch, cut=True)
    got = plugins(bundle)
    assert [st["kernel"] for st in side["stages"]] == ["fan", "fan_2"]
    for art in bundle.artifacts:
        assert got[art.name].sidecar == art.sidecar and got[art.name].bundle_plugins is got
    with pytest.raises(TypeError):
        got["fan"].sidecar["stages"] = []          # read-only: the plan reads it, never writes it


def test_a_kernel_in_stages_runs_through_plugins_and_the_automatic_plan_as_the_whole_kernel(tmp_path, monkeypatch):
    """The documented prebuilt path, ``plugins(bundle)`` then ``eagle.plan.auto``, runs every stage: equal, bit for
    bit, to the same chain built as one kernel (0.7.7 and before ran the first stage alone, with no error)."""
    eplan = pytest.importorskip("eagle.plan")
    x = np.linspace(-1.5, 2.0, 2048)
    cut, side = _fan(tmp_path / "a", monkeypatch, cut=True)
    assert len(side["stages"]) == 2
    have = np.asarray(eplan.auto(plugins(cut)["fan"]).run(x=x))
    whole, side = _fan(tmp_path / "b", monkeypatch, cut=False)
    assert "stages" not in side
    want = np.asarray(eplan.auto(plugins(whole)["fan"]).run(x=x))
    assert have.shape == want.shape and have.tobytes() == want.tobytes()
    assert len(np.unique(have[0])) == x.size        # every sample written


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["matvec_in", "matvec_in_vjp"])
@pytest.mark.parametrize("n", [2048, 2045])
def test_a_team_stage_runs_through_plugins_and_the_automatic_plan_on_the_device(name, n, tmp_path, cache_dir):
    """The warp form of a wide value is a team stage: through ``plugins(bundle)`` and ``eagle.plan.auto`` every sample
    equals NumPy (0.7.7 and before launched it on the per-sample grid: 16 of 2048 samples written, no error)."""
    import _wide_kernels as W
    from eagle import arrays
    from eagle._alloc import synchronize
    from eagle.registry import load_manifest
    eplan = pytest.importorskip("eagle.plan")

    entry = W.CORPUS[name]
    planes = entry.planes(np.random.default_rng(7), np.float64, n, True)
    bundle = build_bundle([entry.kernel], tmp_path, targets=("cuda",), mode="float64", cache_dir=cache_dir,
                          wide_form="team")
    assert next(a.sidecar for a in bundle.artifacts if a.name == entry.kernel.name)["stages"][0]["team"]
    plugin = plugins(bundle, device_loader=load_manifest)[entry.kernel.name]
    bound_args = {k: arrays.asarray(np.ascontiguousarray(v), device="cuda:0") for k, v in planes.items()}
    out = arrays.asarray(np.zeros(entry.shape(n), np.float64), device="cuda:0")
    eplan.auto(plugin).bind(**bound_args, **{entry.out: out}).launch()
    synchronize()
    assert np.array_equal(np.asarray(out.get()), entry.want(np.float64, planes, n))
