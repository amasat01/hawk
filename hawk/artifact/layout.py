# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-TU ABI exports: ``eagle_abi_tag`` + ``eagle_layout_sizes``.

Both are derived from the build, never hand-written: the sizes come out
of ``sizeof`` over eagle's own PODs, included rather than re-declared,
and the fourth field is ``sizeof(EAGLE_ABI_INDEX_T)``, which tracks
``AETHER_INDEX_T`` — so a ``-DAETHER_INDEX_T=std::uint64_t`` build
exports 8 there without a line of HAWK changing.

``layout_sizes_override=`` is a test-only door that replaces the five
``sizeof`` expressions with literal numbers, so a deliberately-wrong
artifact can exist whose refusal a row observes.
"""

from __future__ import annotations

from .. import _contracts
from ..ir import HawkError

#: The array's positional field order, fixed by the mirror PODs'
#: ``static_assert`` layout pins; authoritative over any prose list.
LAYOUT_FIELDS = ("sizeof(GRefMirror)", "sizeof(ScalarHandle)", "sizeof(IntHandle)",
                 "sizeof(aether::idx_t)", "sizeof(PartitionTriple)")

_DERIVED = ("sizeof(eagle::plugin::GRefMirror)",
            "sizeof(eagle::plugin::ScalarHandle)",
            "sizeof(eagle::plugin::IntHandle)",
            "sizeof(EAGLE_ABI_INDEX_T)",
            "sizeof(eagle::plugin::PartitionTriple)")


def exports(backend: str, *, layout_sizes_override=None) -> str:
    """The export block for one target's TU.

    The host object exports plain externs a loader reaches with ``dlsym``; the
    device TU exports ``__device__`` twins, which is what survives into PTX and
    what ``cuModuleGetGlobal`` resolves."""
    values = _DERIVED
    if layout_sizes_override is not None:
        got = tuple(layout_sizes_override)
        if len(got) != len(_DERIVED):
            raise HawkError(
                f"layout_sizes_override= takes {len(_DERIVED)} sizes in "
                f"{LAYOUT_FIELDS} order, got {len(got)}"
            )
        values = tuple(f"{int(v)}ull" for v in got)
    tag = _contracts.AETHER_ABI_VERSION
    if backend == "cuda":
        head = ('extern "C" __device__ const char eagle_abi_tag[] = '
                f'"{tag}";\n'
                'extern "C" __device__ unsigned long long eagle_layout_sizes[5] = {')
    else:
        head = ('extern "C" const char eagle_abi_tag[] = '
                f'"{tag}";\n'
                'extern "C" const std::uint64_t eagle_layout_sizes[5] = {')
    body = ",\n    ".join(values)
    return (f"\n// The self-check exports, DERIVED from this build"
            f".\n{head}\n    {body},\n}};\n")
