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

Re-pinned for the host-only arms: a host unit no longer carries the device arms of its helpers (``#if defined(__CUDA_ARCH__)`` blocks and the
warp row sum); all 21 values moved with the host text, and the CUDA text is unchanged.

Re-pinned for the device-only arms: a device unit no longer carries the host arms of its helpers (the ``#if defined(__CUDA_ARCH__)`` and
``__CUDACC__`` guards and their host halves); all 21 values moved with the CUDA text, the host text is unchanged.

Re-pinned for the keyword rename: ``scatter`` and ``scatter_c`` spell ``contract=`` in their source, and the C header carries the digest of that source.
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
    "as_body": "887e691ce7b236a74c3f4d74b583542489be3f06f2e7119e57abcf7bed3b5f75",
    "axpb": "787d1e2c1bd9a44e8d3d9133c12d3c1336ee6b62950751fac06b60d307c90e2c",
    "chain6": "21f757f9886c34e4c6f901441231ac6b86561a9ba5ea20ed2960a63e1b9b43b5",
    "diagnostic": "5037aa87f00513dce36b61a8b7edd6f0a7d407f13190d37a17bc36d1044aa0ee",
    "diagnostic_two": "0e11070b7ab9de2a07e56c92bcd13ef8249c35fbd9bc009e327a3b57766263cc",
    "drag": "a033948b11df46ca81f38c8ea8b3ede7d148ffc059913d45b7c9b0715b149037",
    "energy": "d0fe4498ecbab95dcd2e0b80ac8a29ed2431e1e4fb3c2ae210f28d31c1167454",
    "fraction": "5237c1c345f86dfbb8c6b29212a172e7f0e685e004a3769940b1e699d4f25d70",
    "gather": "500018226bca128da905bab1ba5503132db566d1667f66dd18c95b981242eea1",
    "mat_apply": "0a2a36e3b3b40e757f29aa40c0a63e477221c5683ecc0ee0b40a2da42e94e185",
    "rotate_quat": "44f441134140afd0ac5ea49a2bba642c5615f77e00bdeb9593e06affc6f6985a",
    "scale": "44d595b4c2cd66e47b7bf6ee163a650354de889a77900c59256cf15b27e0b4a6",
    "scatter": "e270f1f59b59ceaa821f58eee56dd0a093ac75803b7497e9ca2b4f4ab974aa25",
    "scatter_c": "f054b2e777c47ac38c6c6051bdc3571e1382373b23f19fe06c7d9fcf4155efd7",
    "speed": "5f38284fcde6faa45b815a270ce288027987a64780bf04dba6e47e06141a4e92",
    "spin": "938518fcb32ab9b7bec857dd30e6bd87cbdb73af40ad5540ab0c909e37454e5c",
    "two_fin": "b610d57445e75cf4eefef1a722de118a66ecc3221e700b51356e40029a7c7ce9",
    "two_outputs": "9fc732f193da0aacf658e5c237732b30781ae7df5c2dfe8d58c6d87ace72de71",
    "two_wire": "ab37200d6f3475b4f2ee9c9d02b3009cf61c082253fa3f74758b150dbc79e146",
    "vec3_scale": "52d0535bb992eff2e137943e3fcad861fb7daf88e4f76a6d5279ea5ae952170c",
    "vocab": "9c2a8873261b8109df004bff578d68bfdb1645b1c49bf59bbb91f4b459ac5797",
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
    from _specs import pin_caches

    # The host TU's tile is sized to (and names) the machine's caches: pin the ones the digests were frozen with.
    pin_caches(monkeypatch)
    got = digests()
    assert got.keys() == PINNED.keys()
    changed = sorted(name for name in got if got[name] != PINNED[name])
    assert not changed, f"the rendered text of these kernels changed: {changed}"
