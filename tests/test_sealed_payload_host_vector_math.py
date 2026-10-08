# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regression: a HOST compile against the SHIPPED aether-dsc sealed payload,
with `$HAWK_AETHER_INCLUDE`/`$HAWK_EAGLE_INCLUDE` BOTH unset -- the exact
shape a `pip install raptor-hawk` consumer is in, with no dev-tree override
to bypass the sealed payload (`_host_uses_sealed_payload()` only ever takes
the real-root branch when `$HAWK_AETHER_INCLUDE` is set; a dev env sources
one, so this path is otherwise never exercised).

Found failing before the fix: the default host profile
(`native-vector-math` on x86-64, `hawk.compile.toolchain.host_codegen_flags`)
always adds `-DAETHER_HOST_VECTOR_MATH` to every host compile, regardless of
whether the kernel itself calls a transcendental function -- the macro makes
`aether/math/detail/HostVectorMathRoute.h` (already payload-manifested)
`#include "aether/math/detail/HostVectorMath.h"`, which pulls in the
`aether/backend/cpu/simd/math/Packet*.h` headers. None of those eight files
were in `aether_dsc`'s NVRTC-clean manifest
(`aether_dsc/payload_headers.txt`), so the payload's served `-I` root had no
such files on disk and `g++` failed outright on ANY emitted host kernel --
`pip install raptor-hawk` users could never compile one. Confirmed by hand
before this test existed: `g++ ... -I<served_root> ...` on a real emitted
`osc_step.cpp` stopped with
``fatal error: aether/math/detail/HostVectorMath.h: No such file or
directory``.

Fixed by sealing those eight files into a second, host-only section of the
SAME payload (`aether_dsc/payload_headers_host.txt` +
`Payload.host_only_names`) -- served to a host `-I` root
(`Payload.serve()`) exactly like every other payload header, but never
handed to NVRTC (`Payload.device_headers` is what both
`hawk.compile.drivers._compile_nvrtc_source` and the public
`hawk.compile.nvrtc.device`/`cubin` now serve from) and never claimed
NVRTC-clean (`aether_dsc._seal_impl.seal_and_write` runs the gate over the
NVRTC-clean subset only).
"""
from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.compile as hc
from hawk import Mutable, Param, Scalar, Terminated


@hawk.kernel
def _sealed_payload_probe(omega: Scalar, dt: Param, terminated: Terminated,
                          x: Mutable[Scalar], v: Mutable[Scalar]):
    """Deliberately the same shape as `test_host_simd.py`'s `osc_step`: no
    transcendental call in the kernel BODY at all -- the point of this row
    is that `-DAETHER_HOST_VECTOR_MATH` pulls `HostVectorMath.h` into every
    host TU unconditionally (through `aether/device.h`'s own include chain),
    not only a TU whose kernel happens to call `sin`/`exp`/etc."""
    x0 = x
    v0 = v
    w2 = omega * omega
    x = x0 + dt * v0
    v = v0 - dt * w2 * x0


def test_host_compile_and_run_against_the_shipped_sealed_payload(
        monkeypatch, tmp_path):
    """No `$HAWK_AETHER_INCLUDE`/`$HAWK_EAGLE_INCLUDE` override: a host
    kernel must compile through `aether_dsc.payload()` alone (the shipped
    blob) and RUN, producing the right numbers -- not merely "a .so exists".
    """
    for var in ("HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE"):
        monkeypatch.delenv(var, raising=False)
    # Forget any memoised LIVE-sealed payload an earlier test built while
    # $HAWK_AETHER_INCLUDE was still set -- this process must re-derive
    # `current_payload()` under the no-override condition this test sets up,
    # never reuse a dev-tree answer computed before it ran.
    hc._reset_artifact_memo()
    try:
        from hawk.artifact import build_bundle

        bundle = build_bundle(
            [_sealed_payload_probe], tmp_path / "bundle", targets=("host",),
            cache_dir=str(tmp_path / "cache"))

        try:
            import eagle.exec as eexec
            from eagle import plan as eplan
        except ImportError:
            pytest.skip("eagle is not importable in this environment")

        n = 16
        dt = 0.01
        rng = np.random.default_rng(20261004)
        omega = rng.uniform(1.0, 5.0, n)
        terminated = np.zeros(n, dtype=bool)
        x = rng.uniform(-1.0, 1.0, n)
        v = rng.uniform(-1.0, 1.0, n)
        x0, v0 = x.copy(), v.copy()

        plugin = L.host_plugin(bundle.directory, "_sealed_payload_probe",
                               sidecar_of(bundle, "_sealed_payload_probe"))
        step = eplan.plan(plugin, structure=eexec.HostTeam).bind(
            omega=omega, dt=dt, terminated=terminated, x=x, v=v)
        step.launch()

        want_x = x0 + dt * v0
        want_v = v0 - dt * omega * omega * x0
        np.testing.assert_allclose(x, want_x)
        np.testing.assert_allclose(v, want_v)
    finally:
        # Never leak the shipped-payload memo into a later test that expects
        # the dev-tree root once $HAWK_AETHER_INCLUDE is restored.
        hc._reset_artifact_memo()
