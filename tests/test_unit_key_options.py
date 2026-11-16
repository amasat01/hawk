# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every user option of a BUILT back end moves the unit key, and the same options in another order give the same key.

``options={"<backend id>.<key>": value}`` enters the key by rule (:func:`hawk.options.canon` through
``hawk.artifact.unit_cache._row_terms``), without the driver naming any option. Two observations:

* the public build: a build with another host option is a MISS of the per-process memo (``memo_hits`` does not move) and
  publishes its own unit, while the same options in another order are a HIT;
* the key itself, for a host and a cuda option (the cuda build needs a device compiler, so its key is read directly
  from ``_unit_key``, which is what the build computes before it compiles anything).

The options are those the shipped back ends declare: ``host.profile`` and ``cuda.literal_table`` / ``cuda.arch``."""

from __future__ import annotations

import pytest

import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import build_bundle, reset_unit_memo, unit_stats
from hawk.artifact.unit_cache import _unit_key


@hawk.kernel
def affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


def _key(targets, options=None, **kw):
    return _unit_key(["request"], ["declaration"], targets=targets, defines=(), cache_dir="c", device_arch="sm_61",
                     derivative_blocks=[{}], options=options, **kw)


@pytest.mark.parametrize("targets,first,second", [
    (("host",), {"host.profile": "native"}, {"host.profile": "portable"}),
    (("cuda",), {"cuda.literal_table": False}, {"cuda.literal_table": True}),
    (("cuda",), {"cuda.arch": "sm_61"}, {"cuda.arch": "sm_89"}),
], ids=["host.profile", "cuda.literal_table", "cuda.arch"])
def test_an_option_of_a_built_backend_moves_the_key(targets, first, second):
    base = _key(targets)
    assert _key(targets, first) != base and _key(targets, second) != base      # given at all: hashed, even at a default
    assert _key(targets, first) != _key(targets, second)
    assert _key(targets, first) == _key(targets, dict(first))


def test_options_in_another_order_give_the_same_key():
    one = {"cuda.literal_table": True, "cuda.arch": "sm_61"}
    other = dict(reversed(list(one.items())))
    assert list(one) != list(other)
    assert _key(("cuda",), one) == _key(("cuda",), other)


def test_an_option_of_a_backend_this_build_does_not_build_is_not_hashed():
    assert _key(("host",), {"cuda.literal_table": True}) == _key(("host",))


def test_a_host_option_is_a_miss_of_the_memo_and_the_same_options_in_another_order_a_hit(tmp_path, cache_dir):
    reset_unit_memo()
    before = unit_stats()["memo_hits"]
    build_bundle([affine], tmp_path / "a", targets=("host",), cache_dir=cache_dir, options={"host.profile": "native"})
    build_bundle([affine], tmp_path / "b", targets=("host",), cache_dir=cache_dir, options={"host.profile": "portable"})
    assert unit_stats()["memo_hits"] == before, "another host profile was served from the first build's memo"
    build_bundle([affine], tmp_path / "c", targets=("host",), cache_dir=cache_dir, options={"host.profile": "portable"})
    assert unit_stats()["memo_hits"] == before + 1, "the same options did not hit the memo"
