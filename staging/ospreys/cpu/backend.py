# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host backend: C++ source, compiled for the CPU."""

from __future__ import annotations

import platform
from pathlib import Path

from raptor.errors import RaptorError
from hawk.backends import Backend
from hawk.compile import toolchain as _shared
from aether.wings.cpu.host import HostWrapper
from aether.wings.base.prelude import UNIT_INCLUDES
from aether.wings.base.writer_aether import AetherWriter
from . import toolchain as tc


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
    writer = AetherWriter()
    wrapper = HostWrapper()
    entry = "host"
    format = "host"
    artifact = ".so"
    source = ".cpp"
    vectorise = True
    host_compiled = True
    host_entry = True

    def machine_facts(self):
        caches = tc.host_caches()
        try:
            in_effect = tc.host_profile()
        except RaptorError:                  # a refusal (unknown name, no GCC): key_terms and the compile raise it by name, later
            in_effect = tc.default_host_profile()
        cache_facts = {} if caches is None else {"l1d_cache_bytes": caches[0], "l2_cache_bytes": caches[1], "cache_source": caches[2]}
        return {**cache_facts, "isa": "x86_64" if tc._is_x86_64() else "arm64", "arm64_gcc": bool(tc._arm64_gcc()),
                "exact_host_profiles": tuple(tc.EXACT_HOST_PROFILES), "host_profile": in_effect,
                "vector_isa": _vector_isa(), "reserved_names": {name: "the name is a C library function, and the kernel's entry would redeclare it"
                                   for name in tc.libc_symbols(UNIT_INCLUDES)}}

    def key_terms(self, source_digests, own):
        # the profile in effect (asked, else $HAWK_HOST_PROFILE, else this machine's): the unit's text is a function of it
        return ("host_profile:" + tc.host_profile(own.get("profile") or None),)

    def compile_plan(self, opts, inputs):
        compiler = tc.host_compiler()
        defines = [f"-D{d}" for d in opts.defines]
        head = (*tc.host_codegen_flags(opts.host_profile or None, opts.opt_level or None, (*defines, *opts.extra)),
                f"-std={tc.HOST_STD}", "-shared", "-fPIC", "-DAETHER_CPP_MODE")
        plan = {"compiler": compiler, "identity": _shared.compiler_identity(compiler), "deps": True}
        if inputs.sealed:
            # the payload is served into a private directory per compile: the key names it by digest
            digest = inputs.payload.digest
            flags = (*head, *defines, *opts.extra)
            return {**plan, "flags": flags, "key_terms": (*self._terms(flags, opts), f"payload:{digest}"), "payload": digest,
                    "serve": True, "hit_argv": flags, "tail": " against the sealed payload", "meta": {"flags": list(flags)}}
        flags = (*head, *inputs.include_flags(), *defines, *opts.extra)
        keyed = (*flags, "-MD", "-MF")
        return {**plan, "flags": flags, "key_terms": self._terms(keyed, opts), "payload": None, "serve": False,
                "hit_argv": (compiler, *keyed, Path("<dep>"), Path("<src>"), "-o", Path("<out>")), "tail": "",
                "meta": {"flags": list(keyed), "device_target": "ptx"}}

    def compile_hook(self, plan, job, run):
        served = () if job.served is None else (f"-I{job.served}",)
        argv = [plan["compiler"], *plan["flags"], *served, "-MD", "-MF", str(job.dep), str(job.src), "-o", str(job.out)]
        run(argv)
        return {"argv": tuple(argv), "figures": {}}

    @staticmethod
    def _terms(flags, opts) -> tuple:
        """The key's flag terms (a flag carrying a ``key_term`` stands for it), the profile, the resolved opt level and
        the C library (a host ``.so`` links against the libc it was built on); for a ``-march=native`` profile also the
        CPU it targets (:func:`~ospreys.cpu.toolchain.host_target_identity`), since that flag reads the same on every
        machine and a shared cache directory would otherwise serve one CPU's binary to another's."""
        profile = tc.host_profile(opts.host_profile or None)
        level = _shared.opt_level(opts.opt_level or None)
        libc = "-".join(part for part in platform.libc_ver() if part) or "unknown"
        terms = (*(getattr(f, "key_term", "") or f for f in flags), f"host_profile:{profile}", f"opt_level:{level}", f"libc:{libc}")
        if profile != "portable":
            terms = (*terms, f"host_target:{tc.host_target_identity()}")
        return terms


BACKEND = Host()
