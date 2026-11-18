# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The manifest's one execution axis: the units of ONE kernel may differ in access (the kernel states the strictest); two kernels of a
bundle must agree on the whole axis."""

from types import SimpleNamespace

import pytest

from hawk.errors import HawkError
from hawk.artifact.bundle import _manifest_of


def _unit(name, *, access="sample_local", targets=("host",), op=None):
    sidecar = {"exec_targets": list(targets), "exec_access": access, **({} if op is None else {"exec_op": op})}
    return SimpleNamespace(name=name, sidecar=sidecar, sidecar_name=name + ".json", entries={"host": (None, name + ".so", None, "host")})


def test_the_units_of_one_kernel_differing_only_in_access_state_the_strictest():
    doc = _manifest_of([[_unit("a", access="sample_local"), _unit("a__k0", access="cross_sample_write"),
                         _unit("a__sweep", access="cross_sample_read")]])
    assert doc["exec_access"] == "cross_sample_write" and doc["exec_targets"] == ["host"]


def test_two_kernels_differing_in_access_are_refused():
    with pytest.raises(HawkError, match="ONE execution axis"):
        _manifest_of([[_unit("a", access="sample_local")], [_unit("b", access="cross_sample_write")]])


def test_a_kernel_whose_strictest_unit_matches_the_other_kernel_shares_the_bundle():
    doc = _manifest_of([[_unit("a__k0", access="cross_sample_write"), _unit("a__sweep")], [_unit("b", access="cross_sample_write")]])
    assert doc["exec_access"] == "cross_sample_write"


def test_units_differing_in_targets_are_refused_even_in_one_kernel():
    with pytest.raises(HawkError, match="ONE execution axis"):
        _manifest_of([[_unit("a"), _unit("a_2", targets=("device",))]])


def test_a_mapreduce_unit_must_agree_exactly():
    with pytest.raises(HawkError, match="ONE execution axis"):
        _manifest_of([[_unit("a"), _unit("a_2", access="mapreduce", op="sum")]])
