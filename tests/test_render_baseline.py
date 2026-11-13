# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel without linear algebra renders the same text it did before.

Per-lane linear algebra added kinds, helpers and a target feature to code every
kernel passes through. This row pins, per kernel, one sha256 over the host and
CUDA translation units (both scalar modes) and the standalone C unit (both
modes, ``.c`` and ``.h``), for the kernel and its ``vjp`` and ``jvp``. The values
were taken from the v0.7.0 tree before the change; the hawk version stamp in the C
text is replaced by ``VERSION`` first. A deliberate change to the emitted text
re-pins them in the same commit; an accidental one fails here.

Re-pinned for 0.7.2: ``gather``, ``scatter`` and ``scatter_c`` read an ``Index`` plane, which binds
``int32`` and is widened where the body reads it; no other kernel's text changed.

Re-pinned for 0.7.5: a vector value read more than once is declared as an ``aether::Item``, and an output is
stored right after its value is complete instead of at the end of the body; the values are unchanged. Then
``gather``: a ``Table.at`` index is clamped to the table's length. Then ``two_fin``: a lane loop tests whether
any lane is still live after the loop, not inside it, and the host flags drop math errno.

Re-pinned for the digest domain ``raptor.ir.walk/1``: the C header and the manifest carry the walk digest, which changed with it, and a
refusal is named by its class, ``RaptorError``; the generator is :func:`digests` below, the values are its output.
"""

from __future__ import annotations

from _specs import default_kind
from _specs import spec_for
import hashlib

import _deployable as D
import _kernels as K

import hawk
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from hawk_c import render
from aether.wings.base.wrapper import render_source
from aether.wings.base.wrapper import scalar_mode
from hawk.targets import TARGETS
from hawk.trace.kernel import Kernel

MODES = ("float64", "float32")

#: kernel name -> sha256 of its renders (see the module docstring)
PINNED = {
    "as_body": "9b8938d9b7c0a40b6d79da4cc28fe7895f696c115f8b015017cbcab4268ad3a9",
    "axpb": "629f377d9abe48719b973b1c32b12f7f9582ecc6940418b153f7a393532e096b",
    "chain6": "6b348aab62741e1b9eb195742f28c3b812c37e22e61824a7804501801f93a767",
    "diagnostic": "9da78309a43469340324e39b0f0ad6aa8555fde511ea0ed7e15fefdf8fbd9214",
    "diagnostic_two": "9313491642b2f863c8d2370eb028f974838401217e243215d8d99e44405240c6",
    "drag": "780b3085739cc49c3e866a1697c1fd500dcbec240d8146795038dd7e8cd7a180",
    "energy": "87fd780baacd5f6a779f3de000575df2f7e6604fe394f5eb589676d009a52cd9",
    "fraction": "41b1137bcefca8863b5084bd96312533bc7d134c7bdcd193efb0afc21904e7c6",
    "gather": "89fbf4a3d70b194d4688ced1c5597484901386d400fee6c399d9482b7dea2755",
    "mat_apply": "a02f7dd35e83ed8413f608a519ae0b3f8bc89dd6e7f77d15dfbe80731b070a41",
    "rotate_quat": "eee541fe7701d8634497078e234dd4211568a69e122f12583fdafab14c46b9b0",
    "scale": "92e29055be8d7ad98ada24bdda6e6e5795f67939f2b8871b623c110911f3f681",
    "scatter": "d470991a1140eaadce46913cf11327609c46702a30e20c1bb7dd5beb005b828b",
    "scatter_c": "34d12d50cc6669524af1d4a061a1e3c013438a84a093c16e9bffebacfc36e019",
    "speed": "5e2cd9118073ecbf84a2a72c40a384ad8499c7f847025f639c42313a71830e16",
    "spin": "1690627137889b8b8d652d7832da8454f3b714488928664426577cd7b3da4f00",
    "two_fin": "9ebff54786c51d5f382dc2b29e206bfd2762040eb68724e0ead21faead1d618b",
    "two_outputs": "8c41b1c9783c3167a0de1eded1ad1dc4f0ab4fb1548b30f78b27a069bcfbccd0",
    "two_wire": "dc815f652a353e93c39147cf4fa4cf486f4cc6ff732b902b02087dcd224db517",
    "vec3_scale": "c5453555825cb9efba9cb1fd90807001ffc76908cc758cbd89202f754b848485",
    "vocab": "a47b0b4915aedaa718d5d3bef8329672851dde69ff6f4930a43bc1f4916e6d9d",
}


def _text(kernel) -> bytes:
    out = []
    for target in ("host", "cuda"):
        for mode in MODES:
            try:
                text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                                     mode=scalar_mode(mode), exports=exports(target), spec=spec_for(TARGETS[target]), kind=default_kind()).text
            except Exception as exc:                        # noqa: BLE001 - a refusal is pinned too
                text = f"refused {type(exc).__name__}"
            out.append(f"{target}|{mode}|{text}")
    for mode in MODES:
        try:
            unit = render(kernel, mode=mode)
            text = (unit.text + unit.header).replace(hawk.__version__, "VERSION")
        except Exception as exc:                            # noqa: BLE001
            text = f"refused {type(exc).__name__}"
        out.append(f"c|{mode}|{text}")
    return "\n".join(out).encode()


def digests() -> dict:
    """``kernel name -> sha256`` over the kernel, its vjp and its jvp."""
    found = {}
    for module in (D, K):
        for name in sorted(vars(module)):
            kernel = getattr(module, name)
            if not isinstance(kernel, Kernel) or name in found:
                continue
            parts = [kernel]
            for tag, derive in (("vjp", vjp), ("jvp", jvp)):
                try:
                    parts.append(Kernel(f"{kernel.name}_{tag}", derive(kernel)))
                except Exception:                           # noqa: BLE001 - not every kernel derives
                    pass
            found[name] = hashlib.sha256(b"\0".join(_text(p) for p in parts)).hexdigest()
    return found


def test_kernels_without_linear_algebra_render_the_text_they_always_did(monkeypatch):
    import aether.wings.cpu.host as host_emit

    # The host tile rule is sized from (and names) the machine's caches: render as on the pinning machine.
    pinning_machine = host_emit.CacheSizes(32 * 1024, 1024 * 1024, "/sys/devices/system/cpu/cpu0/cache")
    monkeypatch.setattr(host_emit, "machine_caches", lambda *a, **k: pinning_machine)
    got = digests()
    assert got.keys() == PINNED.keys()
    changed = sorted(name for name in got if got[name] != PINNED[name])
    assert not changed, f"the rendered text of these kernels changed: {changed}"
