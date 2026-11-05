# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``build_bundle`` reads the caller's ``policy.using`` block for what it is not given: an explicit argument wins,
an unset one is the ambient value, and the build's own ``using`` carries every field forward."""

from __future__ import annotations

import hawk
from hawk import Mutable, Scalar
from hawk.artifact import build_bundle
from hawk.emit import policy


def _kernel():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = x * x + 1.0
    return hawk.kernel(body)


def _cu(tmp_path, cache_dir, **kw):
    b = build_bundle([k := _kernel()], tmp_path, targets=("cuda",), cache_dir=cache_dir, **kw)
    return (b.directory / f"{k.name}.cu").read_text(), b


def test_ambient_launch_bounds_reach_the_build(tmp_path_factory, cache_dir):
    plain, b0 = _cu(tmp_path_factory.mktemp("a"), cache_dir)
    with policy.using(launch_bounds=(128, 2)):
        ambient, b1 = _cu(tmp_path_factory.mktemp("b"), cache_dir)
        explicit, _ = _cu(tmp_path_factory.mktemp("c"), cache_dir, launch_bounds=(64, 4))
    assert "__launch_bounds__" not in plain
    assert "__launch_bounds__(128, 2)" in ambient and b1.digest != b0.digest
    assert "__launch_bounds__(64, 4)" in explicit and "(128, 2)" not in explicit


def test_the_build_carries_every_ambient_field_forward(tmp_path_factory, cache_dir):
    seen = {}
    real = policy.using

    with policy.using(form="rolled", linalg_form="thread", wide_form="thread", contraction="gemm"):
        import hawk.artifact.bundle as bundle_module

        class Spy:
            def __getattr__(self, name):
                return getattr(policy, name)

            def using(self, *a, **kw):
                seen.update(kw, form=a[0] if a else kw.get("form"))
                return real(*a, **kw)
        bundle_module.policy = Spy()
        try:
            _cu(tmp_path_factory.mktemp("d"), cache_dir)
        finally:
            bundle_module.policy = policy
    assert seen["form"] == "rolled" and seen["linalg_form"] == "thread"
    assert seen["wide_form"] == "thread" and seen["contraction"] == "gemm"
