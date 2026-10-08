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
