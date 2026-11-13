# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lowered-unit frame (raptor.protocols.unit) filled from real kernels on the host target: raptor objects and plain data only,
no backend imported to build it, and the emitted text unchanged by it."""

import contextlib
import contextvars
import sys
from dataclasses import fields, is_dataclass

import _deployable as D
from _specs import HOST
import pytest

from hawk.artifact import bundle
from hawk.artifact.frame import frame_of
from raptor.ir import IR_VERSION
from raptor.protocols.unit import LoweredUnit

_CASES = [(k, None) for k in (D.axpb, D.vec3_scale, D.mat_apply, D.two_outputs, D.gather, D.energy, D.speed, D.scatter,
                              D.fraction, D.spin, D.two_wire, D.vocab, D.diagnostic, D.diagnostic_two)]
_CASES += [(D.scatter_c, D.COMPENSATED), (D.diagnostic, D.MASK_FREE), (D.diagnostic_two, D.TWO_MASKS)]
_IDS = [f"{k.name}-{getattr(kind, 'slug', 'default')}" for k, kind in _CASES]
_OWN = ("hawk", "eagle", "aether", "hawk_c", "hawk_cuda")


def _lowered(kernel, kind):
    options = bundle._user_options(None, False, None, "", "")
    try:
        return bundle._emitted(kernel, kernel.name, "float64", kind, ("host",), "", "t", "t", HOST, options=options)
    except bundle.HawkError as why:
        pytest.skip(f"refused by the single-launch emit path: {why}")


def _reachable_hawk(root):
    """Paths in ``root`` that hold an instance of a class defined in hawk (or another product package)."""
    seen, bad, stack = set(), [], [("unit", root)]
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type(None), bool, int, float, str, bytes)):
            continue
        seen.add(id(obj))
        if type(obj).__module__.split(".")[0] in _OWN:
            bad.append(f"{path}: {type(obj).__module__}.{type(obj).__name__}")
            continue
        if isinstance(obj, dict):
            stack += [(f"{path}[{k!r}]", v) for k, v in obj.items()]
        elif isinstance(obj, (tuple, list, set, frozenset)):
            stack += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        else:
            names = [f.name for f in fields(obj)] if is_dataclass(obj) and not isinstance(obj, type) else []
            names += [s for c in type(obj).__mro__ for s in getattr(c, "__slots__", ())]
            stack += [(f"{path}.{n}", getattr(obj, n)) for n in dict.fromkeys(names) if hasattr(obj, n)]
    return bad


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_the_frame_holds_raptor_objects_and_plain_data_only(case):
    _, lowered, sources, _ = _lowered(*case)
    with contextlib.nullcontext():
        unit = frame_of(lowered)
        assert isinstance(unit, LoweredUnit) and unit.ir_version == IR_VERSION and unit.name == lowered.name
        assert unit.walk_digest == lowered.kernel.walk.digest
        assert not _reachable_hawk(unit)
        assert set(unit.cost) == {"ops", "fp64_ops", "working_set_bytes"}
        assert not _reachable_hawk(unit) and isinstance(unit.cost, dict)      # nothing is skipped: the cost is plain data


@pytest.mark.parametrize("case", _CASES[:12], ids=_IDS[:12])
def test_building_the_frame_imports_no_backend_and_leaves_the_text_alone(case):
    _, lowered, sources, options = _lowered(*case)
    before = {m for m in sys.modules}
    with contextlib.nullcontext():
        unit = frame_of(lowered)
        unit.cost
    new = {m for m in sys.modules if m not in before}
    assert not [m for m in new if m.startswith(("hawk.backends", "hawk_c", "hawk_cuda", "hawk.compile"))], new
    again = bundle._emit(lowered, ("host",), options)
    assert again["host"].text == sources["host"].text


def test_cost_is_a_plain_dict_worked_out_at_frame_build():
    _, lowered, _, _ = _lowered(D.axpb, None)
    with contextlib.nullcontext():
        unit = frame_of(lowered)
        assert type(unit.cost) is dict and unit.cost is unit.cost


def test_cost_read_outside_the_settings_block_gives_the_same_value():
    _, lowered, _, _ = _lowered(D.energy, None)
    with contextlib.nullcontext():
        unit = frame_of(lowered)
        inside = frame_of(lowered).cost
    assert contextvars.Context().run(lambda: unit.cost) == inside        # an empty context: no ambient machine
