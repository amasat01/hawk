# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The manifest/sidecar half of the gap table (the eight rows, …).

Read as DOCUMENTS, before anything is loaded: the schema version and the MAX
twin, the ``aether-abi/2`` tag, the FLAT execution keys in the contractual
top-level ORDER, the ``exec_op`` iff-mapreduce rule, and — the reason the order
matters at all — that raptor's own shape validator accepts what HAWK writes.
``Walk.digest`` is asserted to live in the SIDECAR and NOT in the manifest
(the manifest's key order is contractual and gains no new key).
"""

from __future__ import annotations

import json

import pytest
from conftest import sidecar_of

from hawk import _contracts
from hawk.artifact import TOP_LEVEL_KEY_ORDER_V2


def test_the_manifest_passes_raptors_own_shape_validator(built):
    from raptor.schema import validate_manifest

    for name, bundle in built.items():
        validate_manifest(json.loads(bundle.manifest_path.read_text())), name


def test_top_level_key_order_is_the_contractual_one(built):
    from raptor.schema.manifest import TOP_LEVEL_KEY_ORDER_V2 as RAPTOR_ORDER

    assert TOP_LEVEL_KEY_ORDER_V2 == RAPTOR_ORDER
    for bundle in built.values():
        doc = json.loads(bundle.manifest_path.read_text())
        assert list(doc) == [k for k in RAPTOR_ORDER if k in doc]
        assert "execution" not in doc, "the v2 axis is FLAT keys, not a nested block"


def test_the_schema_and_abi_twins_are_the_max_ones(built):
    for bundle in built.values():
        doc = bundle.manifest
        assert doc["schema_version"] == _contracts.MAX_SCHEMA_VERSION == 2
        assert doc["aether_abi"] == _contracts.AETHER_ABI_VERSION == "aether-abi/2"


def test_exec_op_is_present_iff_mapreduce(built):
    assert built["energy"].manifest["exec_access"] == "mapreduce"
    assert built["energy"].manifest["exec_op"] == "sum"
    assert "exec_op" not in built["vec3"].manifest
    assert built["gather"].manifest["exec_access"] == "cross_sample_read"
    assert built["scatter"].manifest["exec_access"] == "cross_sample_write"


def test_the_digest_lives_in_the_sidecar_not_the_manifest(built):
    import _deployable as D

    bundle = built["vec3"]
    assert "digest" not in bundle.manifest
    assert sidecar_of(bundle, "vec3_scale")["digest"] == D.vec3_scale.walk.digest


def test_a_bundle_whose_members_disagree_on_the_axis_is_refused(tmp_path, cache_dir):
    import _deployable as D

    from hawk.artifact import build_bundle
    from hawk.ir import HawkError

    with pytest.raises(HawkError, match="ONE execution axis"):
        build_bundle([D.axpb, D.energy], tmp_path / "mixed", targets=("host",),
                     cache_dir=cache_dir)
