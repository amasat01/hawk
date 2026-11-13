# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C backend: a standalone MISRA C unit for an embedded build, exported whole, never compiled."""

from __future__ import annotations

import hashlib

import hawk_c as embedded
from .writer import CWriter
from hawk.backends import Backend


class C(Backend):
    id = "c"
    about = "a standalone MISRA C unit for an embedded build"
    writer = CWriter
    format = "c"
    artifact = ".h"
    source = ".c"
    builds = False
    unit_note = "a standalone C unit (<name>.c, <name>.h, <name>.embedded.json, see hawk_c) for an embedded build"
    api = embedded

    def render(self, lowered, options):
        profile = self.own_options(options).get("profile")
        profile = profile if profile is not None else self.default_profile()
        return embedded.render(lowered.kernel, name=lowered.name, mode=lowered.smode.id, kind=lowered.kind, profile=profile,
                              spec=lowered.spec.for_target(self))

    def report_file(self, name):
        return f"{name}.embedded.json"

    def machine_facts(self):
        import platform

        # the isa its manifest's variant map is keyed by, and the names a kernel may not take: the unit writes <name>.h, so
        # the stem of a standard header (ISO C11 7.1.2) would name a header of its own after one; no compiler is asked
        stems = ("assert complex ctype errno fenv float inttypes iso646 limits locale math setjmp signal stdalign stdarg stdatomic "
                 "stdbool stddef stdint stdio stdlib stdnoreturn string tgmath threads time uchar wchar wctype")
        return {"isa": "x86_64" if platform.machine().lower() in ("x86_64", "amd64") else "arm64",
                "reserved_names": {stem: "the generated header <name>.h would take the name of a standard C header"
                                   for stem in stems.split()}}

    def key_terms(self, source_digests, own):
        from aether_dsc import c_core

        return ("source_digests:" + "\x00".join(str(d) for d in source_digests),
                "c_core:" + c_core.CORE_VERSION)

    def refuse_kernels(self, names, kernels):
        from raptor.errors import RaptorError

        if any(getattr(k, "team", None) is not None for k in kernels):
            raise RaptorError(
                f"{names[0]!r} has linear algebra worked by a team of threads (a matrix above 32, or "
                "linalg_form='team'): the standalone C target has no team form, and its helpers hold "
                "n <= 32; build the C unit from a kernel whose matrices fit one lane")

    def default_profile(self):
        return embedded.EmbeddedProfile()

    def package(self, source, result, *, name, smode, profile):
        """The audit and the toolchain checks (any finding refuses, before anything is written), then the report
        :func:`hawk_c.export` writes, built by the same code."""
        from hawk.artifact.bundle import _ExportResult, _File, _blob, _digest_bytes

        profile = profile if profile is not None else self.default_profile()
        embedded.refuse_audit_hits(source)
        checks = embedded.check(source, profile=profile)
        report = embedded.finish_report(source, checks)
        h = hashlib.sha256()
        for text in (source.text, source.header):
            h.update(_blob(text))
            h.update(b"\0")
        files = []
        for fname, text in ((f"{name}{self.source}", source.text),
                            (f"{name}{self.artifact}", source.header),
                            (self.report_file(name), embedded.report_text(report))):
            blob = _blob(text)
            files.append(_File(fname, _digest_bytes(blob), blob=blob))
        entry = (_ExportResult(h.hexdigest(), False, float(checks.get("seconds", 0.0))),
                 f"{name}{self.artifact}", f"{name}{self.source}", self.format)
        return entry, files, report


BACKEND = C()
