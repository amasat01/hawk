# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``build_bundle`` takes its build policy as ONE explicit value (``build_policy=``, :class:`hawk.build_policy.BuildPolicy`), never from
an ambient block: an explicit keyword wins over the policy, an unset one is the policy's value, and the build's spec carries every
setting forward as plain values. A key no target declares in its manifest is refused by name."""

from __future__ import annotations

import pytest

import hawk
from hawk import Mutable, Scalar
from hawk.artifact import build_bundle
from hawk.build_policy import BuildPolicy
from hawk import HawkError


def _kernel():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = x * x + 1.0
    return hawk.kernel(body)


def _cu(tmp_path, cache_dir, **kw):
    b = build_bundle([k := _kernel()], tmp_path, targets=("cuda",), cache_dir=cache_dir, **kw)
    return (b.directory / f"{k.name}.cu").read_text(), b


def test_policy_launch_bounds_reach_the_build(tmp_path_factory, cache_dir):
    plain, b0 = _cu(tmp_path_factory.mktemp("a"), cache_dir)
    bounded, b1 = _cu(tmp_path_factory.mktemp("b"), cache_dir, build_policy=BuildPolicy(launch_bounds=(128, 2)))
    assert "__launch_bounds__" not in plain
    assert "__launch_bounds__(128, 2)" in bounded and b1.digest != b0.digest


def test_the_build_carries_every_policy_field_forward(tmp_path_factory, cache_dir):
    seen = {}
    import hawk.artifact.bundle as bundle_module
    real = bundle_module.lower

    def spy(kernel, name, smode, kind, targets, spec):
        cfg = spec.settings
        seen.update(form=cfg.form, linalg_form=cfg.linalg_form, wide_form=cfg.wide_form, contraction=cfg.contraction)
        return real(kernel, name, smode, kind, targets, spec)

    bundle_module.lower = spy
    try:
        _cu(tmp_path_factory.mktemp("d"), cache_dir, build_policy=BuildPolicy(
            emit_form="rolled", linalg_form="thread", wide_form="thread", contraction="gemm"))
    finally:
        bundle_module.lower = real
    assert seen["form"] == "rolled" and seen["linalg_form"] == "thread"
    assert seen["wide_form"] == "thread" and seen["contraction"] == "gemm"


def test_a_key_no_target_declares_is_refused_by_name(tmp_path, cache_dir):
    with pytest.raises(HawkError, match="fold_rows"):
        build_bundle([_kernel()], tmp_path, targets=("host",), cache_dir=cache_dir, build_policy=BuildPolicy(fold_rows=2))


def test_the_policy_validates_its_values():
    with pytest.raises(ValueError, match="emit_form"):
        BuildPolicy(emit_form="loud")
