# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's OWN copies of the schema / aether-ABI / mirror-layout constants
(the severance law this module satisfies, described below).

HAWK's authoring/codegen/compile/emit path needs these values — they are stamped into
every manifest/sidecar it writes and
size every by-value mirror that crosses the nanobind boundary — but must
import ZERO raptor/eagle at runtime. So this
module holds conformance-gated, VERBATIM copies of the values raptor's schema
module and eagle's Python + C++ sources define.

Nothing here is a live import of raptor or eagle: the values are plain
literals, kept in sync by raptor's TEST-TIME conformance suite
(``raptor/tests/test_hawk_contracts_conformance.py``), which reads
this file by AST (never imports ``hawk``) and cross-checks it against
raptor's own schema module and eagle's Python + C++ sources. If any of these
values ever changes upstream, bump it here too and let the conformance gate
catch a miss.

UNLIKE the previous code generator's contracts module (which pinned only
:data:`SCHEMA_VERSION`, the
four-way cross-repo floor), HAWK carries the MAX twin as well
(:data:`MAX_SCHEMA_VERSION`) because HAWK writes ONLY schema-v2 documents — the number a
v2 document actually declares is the ceiling, not the
floor.
"""

from __future__ import annotations

#: Plugin-schema FLOOR — the four-way cross-repo pin (``SCHEMA_VERSION``,
#: unmoved since). Copy of ``raptor.schema.manifest.SCHEMA_VERSION``, itself
#: 1:1 with ``eagle.roles.SCHEMA_VERSION`` (an import of the same value) and
#: the C++ ``kPluginSchemaVersion`` (``eagle/plugin/roles.h``). HAWK does
#: not write schema-v1 documents, but the floor is still the number of
#: record every schema-version comparison anchors on.
SCHEMA_VERSION = 1

#: The HIGHEST ``schema_version`` this era's loaders accept, and the value
#: HAWK actually writes into every manifest/sidecar (this is the MAX
#: twin). Copy of ``raptor.schema.manifest.MAX_SCHEMA_VERSION`` /
#: ``eagle.roles.MAX_SCHEMA_VERSION`` (both derive from the same number) and
#: the C++ ``kPluginMaxSchemaVersion`` (``eagle/plugin/roles.h``).
#: Emitting ``SCHEMA_VERSION`` (1) into a v2 document is refused by
#: ``raptor.schema.manifest.check_execution_axis``; emitting this value
#: without the exec keys is refused by the same function's v2 arm.
MAX_SCHEMA_VERSION = 2

#: The by-value POD layout contract tag HAWK stamps on every artifact.
#: Copy of ``eagle.abi.ABI_TAG_V2`` (``eagle/python/eagle/abi.py``),
#: itself 1:1 with the C++ ``EAGLE_AETHER_ABI_V2`` macro
#: (``eagle/plugin/gref_abi.h``) and with
#: ``raptor.schema.manifest.AETHER_ABI_V2``. HAWK never emits the v1 tag
#: (``aether-abi/1``) — every HAWK artifact is schema-v2 / ABI-v2.
AETHER_ABI_VERSION = "aether-abi/2"

#: Byte size of the by-value ``GRefMirror`` POD: the aether
#: ``View<Real, extents<N,dyn>, layout_stride>`` mirror a ``vec_in``/``out``/
#: matrix-mutable role crosses the nanobind boundary as. Copy of the layout
#: ``eagle.host_launch.GRefMirror`` ctypes struct asserts, and of the C++
#: ``static_assert(sizeof(GRefMirror) == 40, ...)``
#: (``eagle/plugin/gref_abi.h``).
GREF_MIRROR_SIZE = 40

#: Byte size of the by-value ``ScalarHandle`` POD: the aether
#: ``View<Real, extents<dyn>, layout_stride>`` mirror a ``per_sample``/
#: ``lookup``/``terminated``/``wide_in``/``wide_out``/``accum_out``/
#: scalar-mutable role crosses as. Copy of
#: ``eagle.host_launch.ScalarHandle`` and of the C++
#: ``static_assert(sizeof(ScalarHandle) == 32, ...)``.
SCALAR_HANDLE_SIZE = 32

#: Byte size of the by-value ``IntHandle`` POD: the int-valued twin of
#: ``ScalarHandle`` (an ``int``-typed ``Mutable``/lookup role). No Python
#: ctypes twin exists yet on eagle's side (the citation is the C++ struct
#: only); copy of the C++ ``static_assert(sizeof(IntHandle) == 32, ...)``
#: (``eagle/plugin/gref_abi.h``).
INT_HANDLE_SIZE = 32

#: Byte size of the by-value ``PartitionTriple`` POD — the ``{base, count,
#: nSamples}`` int64 triple every ``aether-abi/2`` entry takes after its role
#: args, and the FIFTH field of the layout self-check
#: array. Copy of the C++ ``static_assert(sizeof(PartitionTriple) == 24, ...)``
#: (``eagle/plugin/gref_abi.h``). Fixed by construction (three ``int64``s),
#: unlike the array's FOURTH field ``sizeof(EAGLE_ABI_INDEX_T)``, which is a
#: BUILD AXIS (``-DAETHER_INDEX_T``) and therefore has no constant twin here:
#: its agreement is checked where it can actually disagree — artifact against
#: host, at ``hawk._core.HostLibrary`` load.
PARTITION_TRIPLE_SIZE = 24
