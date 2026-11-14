# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""One hawk-absent test per entry module, on IR built through raptor only (a stub stands for the build spec)."""
from types import SimpleNamespace

import pytest
from raptor.ir.nodes import AccumWrite, Assign, At, Const, Leaf, SampleIndex
from raptor.errors import RaptorError
from raptor.ir.types import TensorType
from raptor.ir import make

F64, I32 = TensorType((), "f64"), TensorType((), "i32")
X = Leaf("vocab_read", "per_sample", "x", F64)
KEY = Leaf("vocab_read", "per_sample", "key", I32)


def spec(**settings):
    s = SimpleNamespace(route_to_ordered=False, team_threads=0, linalg_form="auto", wide_form="auto", scalar="", device_only=True,
                        staged_rows=1)
    s.__dict__.update(settings)
    return SimpleNamespace(settings=s, primary={}, device=None, need=lambda *k: 32, wide=lambda dtype: {"SLICE_MIN": 4})


def test_team_names_and_stages():
    from raptor.wings.base import team

    assert len(team.scatter_names("t")) == 4 and team.threads(10) == 128 and team.threads(10, 7) == 7
    assert isinstance(team.PARTIAL, str)
    stages = team.scatter_specs("t", 2, 256, 32)
    assert len(stages) == 3 and all(isinstance(s, team.TeamSpec) for s in stages)
    assert team.Plane("p", "f64", 1).name == "p"


def test_split_turns_an_ordered_scatter_into_stages():
    from raptor.wings.base import split

    sinks = (AccumWrite("acc", X, KEY, F64, "ordered"),)
    found = split.stages("k", sinks, spec=spec())
    assert found is not None and all(isinstance(s, split.Stage) for s in found) and len(found) >= 3
    assert split.stages("k", (Assign("m", X),), spec=spec()) is None
    assert callable(split.internal)


def test_wrapper_text_helpers_and_refusals():
    from raptor.wings.base import wrapper_text as w

    mode = SimpleNamespace(cells=True, id="banded")
    with pytest.raises(RaptorError, match="banded runs on device"):
        w.refuse_cells_off_device("k", mode, "host", SimpleNamespace(device=False))
    team = SimpleNamespace(steps=[SimpleNamespace(op="block_sum", planes=())])
    with pytest.raises(RaptorError, match="block sum over matrices"):
        w.refuse_team_reductions(team, mode)
    w.refuse_team_reductions(None, mode)


def test_wide_queries_on_plain_nodes():
    from raptor.wings.base import wide

    assert not wide.is_resident(X) and not wide.is_scratch(X) and not wide.is_tiled(X) and not wide.is_prior(X)
    assert "dot" in wide.REDUCTIONS and isinstance(wide.RESIDENT_KIND, str)
    assert wide.record(()) == {} and wide.gather(X) is None
    sinks = (Assign("m", make("add", (X, X))),)
    assert len(wide.simplified(sinks)) == 1 and len(wide.unstored(sinks)) == 1
    assert wide.element_kind(X) in (None, "input", "elementwise", "parts", "splat")




def test_lanes_queries():
    from raptor.wings.base import lanes

    assert lanes.CONTRACTIONS == ("gemm", "direct")
    uniform = lanes.warp_uniform((Assign("m", X),), 32)
    assert uniform(Const(1, I32)) and not uniform(KEY)
    assert lanes.key_run((Assign("m", X),), Assign("m", X), None) is None


def test_box_text_and_declines():
    from raptor.wings.base import box

    b = box.Box(0, 4, 1)
    assert box.threads(None, 10) >= 1
    assert box.thread_box(SimpleNamespace(sinks=(), walk=SimpleNamespace(arg_spec=())), "float64", spec=spec(), card=None, enabled=False) is None
    assert callable(box.boxed_sinks)





def test_team_max_dim_is_the_facts_and_refused_by_name():
    from raptor.wings.base import team

    spec = SimpleNamespace(need=lambda section, key=None: {"linalg": {"team_max_dim": 4096}}[section][key])
    assert team.team_max_dim_of(spec) == 4096

    def need(section, key=None):
        raise RaptorError(f"the wing of the target being built has no {section}.{key}")

    with pytest.raises(RaptorError, match="linalg.team_max_dim"):
        team.team_max_dim_of(SimpleNamespace(need=need))


def test_a_warp_entry_is_the_facts_data_and_a_host_build_has_none():
    from raptor.wings.base import team

    host = spec(linalg_form="auto", scalar="float32")
    host.device = None
    assert team.warp_factors((), (), host) == frozenset()
