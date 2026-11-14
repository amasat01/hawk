# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""hawk with the GPU OSPREY (osprey-cuda) off the path is a working host-only hawk: it imports, builds
with default targets, runs, and refuses the device target by name."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

SCRIPT = """
import importlib.metadata as _md
_orig = _md.distribution


class _NoCuda:
    '''hawk's own distribution as installed without the GPU OSPREY: its metadata does not declare osprey-cuda.'''

    def __init__(self, dist):
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def entry_points(self):
        return [e for e in self._dist.entry_points if e.name != "cuda"]


_md.distribution = lambda name: _NoCuda(_orig(name)) if name == "raptor-hawk" else _orig(name)
import json, sys
import numpy as np
import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import plugins
from hawk.targets import DEFAULT_TARGETS, TARGETS

@hawk.kernel
def ho_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b

out = {"targets": list(TARGETS), "default": list(DEFAULT_TARGETS)}
bundle = hawk.build([ho_affine], sys.argv[1], cache_dir=sys.argv[2])
(art,) = bundle.artifacts
out["entries"] = sorted(art.entries)
x = np.linspace(0.5, 2.0, 33)
from eagle import exec as eexec, plan as eplan
got = np.asarray(eplan.plan(plugins(bundle)[art.name], structure=eexec.HostTeam).run(x=x, a=1.5, b=0.25))
out["matches"] = bool(np.allclose(got, 1.5 * x + 0.25))
for call in (lambda: hawk.build([ho_affine], sys.argv[1] + "_d", targets=("cuda",), cache_dir=sys.argv[2]),
             lambda: hawk.artifact.arch()):
    try:
        call()
        out.setdefault("refusals", []).append(None)
    except hawk.HawkError as exc:
        out.setdefault("refusals", []).append(str(exc))
print(json.dumps(out))
"""


#: The state without the extension packages is built from the source tree's layout; a wheel carries them beside hawk.
pytestmark = pytest.mark.repo_local


@pytest.fixture(scope="module")
def host_only(tmp_path_factory):
    pytest.importorskip("eagle.exec")
    root = tmp_path_factory.mktemp("host_only")
    (root / "probe.py").write_text(SCRIPT)
    path = os.pathsep.join(p for p in sys.path if p)    # an explicit path; the cuda OSPREY is hidden by the metadata above
    done = subprocess.run([sys.executable, str(root / "probe.py"), str(root / "out"), str(root / "cache")],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": path}, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_without_the_gpu_osprey_no_device_backend_is_registered(host_only):
    assert "cuda" not in host_only["targets"] and "host" in host_only["targets"]


def test_default_targets_build_the_host_alone(host_only):
    assert host_only["default"] == ["host"] and host_only["entries"] == ["host"]


def test_the_host_only_build_runs_through_eagle_and_matches_numpy(host_only):
    assert host_only["matches"] is True


def test_the_device_target_and_the_arch_refuse_by_name(host_only):
    for said in host_only["refusals"]:
        assert said is not None and "install the distribution osprey-cuda" in said
