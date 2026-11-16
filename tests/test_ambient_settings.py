# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``build_bundle`` takes its build settings as ONE explicit value (``settings=``, :class:`hawk.build_settings.BuildSettings`), never from
an ambient block: an explicit keyword wins over the settings, an unset one is the settings's value, and the build's spec carries every
setting forward as plain values. A key no target declares in its manifest is refused by name."""

from __future__ import annotations

import pytest

import hawk
from hawk import Mutable, Scalar
from hawk.artifact import build_bundle
from hawk.build_settings import BuildSettings
from hawk import HawkError


def _kernel():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = x * x + 1.0
    return hawk.kernel(body)


def _cu(tmp_path, cache_dir, **kw):
    b = build_bundle([k := _kernel()], tmp_path, targets=("cuda",), cache_dir=cache_dir, **kw)
    return (b.directory / f"{k.name}.cu").read_text(), b


def test_settings_launch_bounds_reach_the_build(tmp_path_factory, cache_dir):
    plain, b0 = _cu(tmp_path_factory.mktemp("a"), cache_dir)
    bounded, b1 = _cu(tmp_path_factory.mktemp("b"), cache_dir, settings=BuildSettings(launch_bounds=(128, 2)))
    assert "__launch_bounds__" not in plain
    assert "__launch_bounds__(128, 2)" in bounded and b1.digest != b0.digest




def test_a_key_no_target_declares_is_refused_by_name(tmp_path, cache_dir):
    with pytest.raises(HawkError, match="fold_rows"):
        build_bundle([_kernel()], tmp_path, targets=("host",), cache_dir=cache_dir, settings=BuildSettings(fold_rows=2))


def test_the_settings_validates_its_values():
    with pytest.raises(ValueError, match="emit_form"):
        BuildSettings(emit_form="loud")
