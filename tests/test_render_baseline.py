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
    "axpb": "c347f543c5a46a79b5ba814ff2378959b51f0840cfe3437693484c2d492906f3",
    "chain6": "fa70e82fb8a3ff8d4a512b27bf979ace9fae81f01340b0b61076c7563a33878e",
    "diagnostic": "297ee73dbc5fd6a887cba135146c87a3f041fc24d53d170c5dfbf1899010feec",
    "diagnostic_two": "1622a681c35a32e9a18e77f254e92973b1e1e05c48cb403b23064b0ccb7c4e3b",
    "drag": "3e75ec20e504e3a7a3608e54392815d83f7088949556da984c50a2d4de9c5a8c",
    "energy": "4b3ba2dc2c6781fc664ba2eb4cfdfb8ab21f31120ba77c66c296be0d0d041b2f",
    "fraction": "cb6e48abaec0f8d6fab4c956f0258db0e53eaea4910d614a3d0ba79610dbdd11",
    "gather": "c4431ef79117cda7f7b7f7100d6a2fc15e4dd72092ad6e098dcae6d07a25f90a",
    "mat_apply": "0a1e353c357258c49d7f55cb36153b9061b4baff13db92952b8db39dc499c009",
    "rotate_quat": "c660fe5309dce802136572e5b85217971eb9481e8244b0a73d96f69a4cdd2d6e",
    "scale": "6320908a72c41a0e5ca27e68dcb38dcd2794607099168e10162059ef25089af0",
    "scatter": "8562be3b605b1d70d6f2cb8dabd25c8ab258adc9ff03b24f1ee2ea00237e71f2",
    "scatter_c": "9946fd5ed17b1e7d3563fe4db2a9a022609a9a9220edd61b130ec7fe7192af48",
    "speed": "2f52cb7fd79d54f4db17c357278f2b099dd1cd3c219f19d44beb81a579f72e87",
    "spin": "5cfd6e47650778a097d0854031a7695ccf1132d984923c43006b5daed764165c",
    "two_fin": "1ee79388146d5d07543f08ff4b129156a73868db1f988e55267c9e9326654979",
    "two_outputs": "1f600d289c272bb4dc75714683f203c7b1d034471c3cc736d00e0e281983e0e8",
    "two_wire": "e20d62075ff489262a1a814fd7eacd78d3d3961cb1d4bc624a44627ebbf0749f",
    "vec3_scale": "64bac3a402be5718444bc5b30a5e19460eaab2c280b3344037ecaf3d4c6ebe4d",
    "vocab": "6ef8afd43ced81755f90d9e15aea0254d0d7305e0ad33c1b96356cd21d024d76",
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
