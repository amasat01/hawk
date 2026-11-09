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
"""

from __future__ import annotations

import hashlib

import _deployable as D
import _kernels as K

import hawk
from hawk.artifact.layout import exports
from hawk.diff import jvp, vjp
from hawk.embedded import render
from hawk.emit import render_source
from hawk.emit.wrapper import scalar_mode
from hawk.targets import TARGETS
from hawk.trace.kernel import Kernel

MODES = ("float64", "float32")

#: kernel name -> sha256 of its renders (see the module docstring)
PINNED = {
    "as_body": "490e0c894ca78e61f5a3604cc54d0726126d6044ffd23791a4e62615b36fb49f",
    "axpb": "d22f70834151488b7d9061b8c2ec427bcad3b62c87d02a9032f26e38e90e98b9",
    "chain6": "6281f86ec8a83f366aaa675fce4542d6a42e21a575236f1aab5e28cd1da651cd",
    "diagnostic": "409ac8e6d3aed3c21fa6b7821c15ef18bf08e01a2d2728299a3e258534b3d7e8",
    "diagnostic_two": "dde31bf42cab6abc90af1becbffc04b541fd98bed6bacee58ce4b8425d7171fb",
    "drag": "78878d712b6ffba3f86110302da5dd2a921f6e83c7360ecb4849416c298871ae",
    "energy": "9a0ae2abf5056e9cb2fe1103c9037854b4482e3a9d76fb42722c168129ecfd1d",
    "fraction": "cb6e48abaec0f8d6fab4c956f0258db0e53eaea4910d614a3d0ba79610dbdd11",
    "gather": "ee4e06a4ff6ec822a1f745fd6b87719ad1260bf44a9feea5049ad887e2bae6de",
    "mat_apply": "bdee5c50cc7733067602760a0c4a0b6a55d3d6965a141caa8969a93f8f882007",
    "rotate_quat": "a4a01777b7ec5e36a7e78637bf62841f1ba5c668a1e9502df3e3125c65bc442a",
    "scale": "558b0831a64a7c6ba792610694c7e92eedec6da5a54ae2ea115b4ae20f970389",
    "scatter": "d182385cd8a35be4db6941c1e0570368c3ccb76b85c23f9d29b7f58d27d2fa6f",
    "scatter_c": "fd8ff1fdec21b3596de808ff3f2c088e9a3d5c127f333d4384c318f8c667e6ba",
    "speed": "2f52cb7fd79d54f4db17c357278f2b099dd1cd3c219f19d44beb81a579f72e87",
    "spin": "62b0d8722d8e1298b687c6397a13887785b37f1c9c645aeb4f588c2c752e81fc",
    "two_fin": "da73bdf49d4fc3341a037cd44111367344b1a8a82573ced69628c307668e3437",
    "two_outputs": "3b9fdbeb5478dcf534cae87ee3321cf864ff5f98492958706ace6810a5a72419",
    "two_wire": "e20d62075ff489262a1a814fd7eacd78d3d3961cb1d4bc624a44627ebbf0749f",
    "vec3_scale": "0d204277907cbf3c9f261060f84f1e8ff79fc115751a0018002c295fa3769ee8",
    "vocab": "ec7cca12f511950aab2bad22baf2c525edd2c7170581443b47f00081658a8935",
}


def _text(kernel) -> bytes:
    out = []
    for target in ("host", "cuda"):
        for mode in MODES:
            try:
                text = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                                     mode=scalar_mode(mode), exports=exports(target)).text
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
    # The host tile rule is sized from (and names) the machine's caches: render as on the pinning machine.
    pinning_machine = hawk.emit.host.CacheSizes(32 * 1024, 1024 * 1024, "/sys/devices/system/cpu/cpu0/cache")
    monkeypatch.setattr(hawk.emit.host, "machine_caches", lambda *a, **k: pinning_machine)
    got = digests()
    assert got.keys() == PINNED.keys()
    changed = sorted(name for name in got if got[name] != PINNED[name])
    assert not changed, f"the rendered text of these kernels changed: {changed}"
