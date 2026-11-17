# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host backend: C++ source, compiled for the CPU."""

from __future__ import annotations

import platform
from pathlib import Path

from hawk.backends import Backend
from . import toolchain as tc

#: The headers a unit includes: the names they declare are the C library names a kernel may not take.
UNIT_INCLUDES = """#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <aether/expr/nodes/Geometric.h>
#include <aether/expr/nodes/Inverse.h>
#include <aether/expr/nodes/Product.h>
#include <aether/expr/nodes/Quaternion.h>
#include <aether/expr/nodes/Structural.h>
#include <aether/expr/Reduce.h>
#include <aether/math/math.h>

#include "plugin/gref_layout.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
"""


def _vector_isa() -> str:
    """The vector ISA of this host, the key of the fusion card's host row."""
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "neon"
    try:
        flags = next((line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")), "")
    except OSError:
        flags = ""
    words = set(flags.split())
    return "avx512" if "avx512f" in words else "avx2" if "avx2" in words else "sse2"


class Host(Backend):
    id = "host"
    about = "for the CPU"
    entry = "host"
    format = "host"
    artifact = ".so"
    source = ".cpp"
    vectorise = True
    host_compiled = True
    host_entry = True

    #: The program the masked-call witness builds and runs (:func:`hawk.compile.host.masked_call_verdict`).
    masked_call_witness_source = tc.MASKED_CALL_WITNESS

    def compiler(self) -> str:
        return tc.host_compiler()

    def target_identity(self) -> str:
        return tc.host_target_identity()

    def machine_facts(self):
        caches = tc.host_caches()
        in_effect = tc.asked_host_profile()     # unchecked: the library's answer to the facts refuses a profile this compiler cannot build
        cache_facts = {} if caches is None else {"l1d_cache_bytes": caches[0], "l2_cache_bytes": caches[1], "cache_source": caches[2]}
        return {**cache_facts, "isa": "x86_64" if tc._is_x86_64() else "arm64", "arm64_gcc": bool(tc._arm64_gcc()),
                "exact_host_profiles": tuple(tc.EXACT_HOST_PROFILES), "host_profile": in_effect,
                "vector_isa": _vector_isa(), "reserved_names": {name: "the name is a C library function, and the kernel's entry would redeclare it"
                                   for name in tc.libc_symbols(UNIT_INCLUDES)}}

    def key_terms(self, source_digests, own):
        # the profile in effect (asked, else $HAWK_HOST_PROFILE, else this machine's): the unit's text is a function of it
        return ("host_profile:" + tc.host_profile(own.get("profile") or None),)


BACKEND = Host()
