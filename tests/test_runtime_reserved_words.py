"""A raw host launch fills the reserved words a runner would provide.

A kernel that finishes its own samples counts them into a one-cell
``finished_count`` word, and an automatic-step kernel reads a ``fused_steps``
word. Outside a runner the caller should not have to know either exists:
leaving them out of ``hawk.runtime.run`` binds a fresh counter and a step count
of one."""
from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.runtime as R
from hawk import Mutable, Param, Scalar, Terminated
from hawk.artifact import build_bundle


@hawk.kernel
def finishing_step(dt: Param, t_end: Param, terminated: Terminated,
                   t: Mutable[Scalar]):
    t += dt
    terminated = t >= t_end


@pytest.fixture(scope="module")
def finishing_unit(tmp_path_factory, cache_dir):
    directory = tmp_path_factory.mktemp("finishing_unit")
    build_bundle([finishing_step], directory, targets=("host",), cache_dir=cache_dir)
    return directory


def _unit_kernel(directory):
    (so,) = [p for p in directory.rglob("*.so") if "finishing_step" in p.name]
    return R.load(so.parent, so.stem)


def test_a_finishing_kernel_runs_without_binding_its_counter(finishing_unit):
    k = _unit_kernel(finishing_unit)
    t = np.array([0.0, 0.5, 0.95])
    terminated = np.zeros(3, dtype=bool)
    R.run(k, dt=0.1, t_end=1.0, t=t, terminated=terminated)
    assert np.allclose(t, [0.1, 0.6, 1.05])
    assert terminated.tolist() == [False, False, True]


def test_a_bound_counter_is_used_as_given(finishing_unit):
    k = _unit_kernel(finishing_unit)
    t = np.array([0.95, 0.99])
    terminated = np.zeros(2, dtype=bool)
    counter = np.zeros(1, dtype=np.int64)
    R.run(k, dt=0.1, t_end=1.0, t=t, terminated=terminated, finished_count=counter)
    assert counter[0] == 2
