# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The targets table: one row per thing ``hawk.build(targets=...)`` can build.

A row pairs a writer (how the language spells each construct) with a wrapper
(how a kernel is wrapped for one execution model) and carries what the build
and the package step need: how it compiles, what eagle enters it as, its
file suffixes and the features it builds. The build and the compile drivers
read the row; none of them compares a target's name.

A row without a wrapper is written whole by its writer; one without a
compiler is exported, not built. Adding a target is adding a row to
:data:`TARGETS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import embedded
from ..embedded.writer import CWriter
from ..emit.cuda import CUDA
from ..emit.host import HOST
from ..emit.writer_aether import AETHER
from ..emit.wrapper import Wrapper
from .features import FEATURES as FEATURES

#: The toolchains a row compiles with: the device compiler or the host one.
GPU = "gpu"
CPU = "cpu"
#: How eagle enters a built kernel (the manifest's ``exec_targets`` words).
ENTRY_KINDS = ("device", "host")
ENTRY_DEVICE, ENTRY_HOST = ENTRY_KINDS


@dataclass(frozen=True)
class Target:
    """One row of :data:`TARGETS`.

    ``wrapper`` is ``None`` when the writer produces the whole file;
    ``compile`` is ``None`` when the row is exported rather than built, and
    then ``export`` is the module that renders, audits and reports the unit
    (:mod:`hawk.embedded`); ``entry`` is ``None`` when eagle never enters it.
    ``artifact`` and ``source`` are the file suffixes, ``format`` the
    manifest format, ``about`` the words that say what the row builds, and
    ``features`` the names from :data:`FEATURES` it builds (plus ``vectorise``
    when its loop over samples may be vectorised). Both scalar modes build
    on every row."""

    id: str
    about: str
    writer: object
    wrapper: Wrapper | None
    compile: str | None
    entry: str | None
    format: str
    artifact: str
    source: str
    features: frozenset
    export: object = None


TARGETS: dict[str, Target] = {t.id: t for t in (
    Target("cuda", "builds for the GPU", AETHER, CUDA, GPU, ENTRY_DEVICE, "ptx",
           ".ptx", ".cu", frozenset(FEATURES)),
    Target("host", "for the CPU", AETHER, HOST, CPU, ENTRY_HOST, "host",
           ".so", ".cpp", frozenset(FEATURES) | {"vectorise"}),
    Target("c", "a standalone MISRA C unit for an embedded build", CWriter, None,
           None, None, "c", ".h", ".c", frozenset(), export=embedded),
)}

#: What ``build`` builds when ``targets`` is not given: every row with a compiler.
DEFAULT_TARGETS = tuple(t for t, row in TARGETS.items() if row.compile is not None)
