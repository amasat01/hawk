# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host backend: C++ source, compiled for the CPU."""

from __future__ import annotations

from ..compile import drivers
from ..compile import toolchain as tc
from ..emit.host import HOST
from ..emit.writer_aether import AETHER
from . import Backend


class Host(Backend):
    id = "host"
    about = "for the CPU"
    writer = AETHER
    wrapper = HOST
    entry = "host"
    format = "host"
    artifact = ".so"
    source = ".cpp"
    vectorise = True
    host_compiled = True
    host_entry = True

    def compile(self, source, name, opts):
        if drivers._host_uses_sealed_payload():
            return drivers._compile_host_sealed(source, name, opts, self)
        return drivers.compile_cached(
            source, name, opts, self, compiler=tc.host_compiler(),
            argv_flags=lambda o, target: tc.host_flags(extra=(*drivers.define_args(o), *o.extra), profile=o.host_profile or None,
                                                  opt=o.opt_level or None),
            pick_target=lambda compiler: "ptx",
            terms=lambda flags, target: drivers._host_key_terms(flags, opts),
            parse=lambda stderr: {})


BACKEND = Host()
