# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Deliberately-broken artifacts a gate BUILDS FOR ITSELF, to prove it can fail.

Rows and assert that a HAWK body gives the same answer whole as it does
over k partitions, and that the serial oracle agrees with eagle's structures.
Green on those comparisons means nothing unless something could have made them
red — and the only honest something is an artifact that really is NOT
partition-invariant, compiled by the same toolchain, loaded by the same loader
and run through the same plan.

So this module builds one. It takes the ``fraction`` kernel — whose body divides
by the ``nsamples`` ROLE, the TRUE total every partition is handed — and
compiles a second host object in which that read is replaced by the triple's
``count``, the partition's OWN width. Whole-view the two agree exactly; split in
two, the planted one divides by half the count and its answer moves. That is
the failure mode verbatim: "read ``nSamples``, not ``count``, wherever the
sample count is baked", the mistake that produces a plausible number rather than
a crash.

It is a source SURGERY on the emitted TU and not a HAWK feature: HAWK has no way
to emit this body, which is exactly why it has to be built here (the same shape
as the ``layout_sizes_override=`` door, which exists only so row has a
wrong-layout artifact to be refused). Nothing outside the tests imports it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hawk.artifact.layout import exports
from hawk.compile import CompileOptions, compile_source
from hawk.emit import BACKENDS, FLOAT64, render_source

#: What the emitted ``fraction`` body reads, and what the plant makes it read.
TRUE_TOTAL = "static_cast<Int>(nsm_n_samples)"
PARTITION_WIDTH = "static_cast<Int>(count)"


def count_reading_bundle(kernel, directory, *, name: str = "fraction",
                         cache_dir: str | None = None) -> Path:
    """Publish a host-only artifact of ``kernel`` whose body reads ``count``.

    Returns the directory; it carries ``<name>.so`` and ``<name>.json`` and so
    loads through ``hawk.runtime.load`` and ``_deploy.host_plugin`` exactly like
    a real one. Device is deliberately NOT built: the property under test is
    partition invariance on the HOST arm, and one target is enough to make the
    comparison fail."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = render_source(name, kernel.sinks, kernel.walk, BACKENDS["host"],
                           mode=FLOAT64, exports=exports("host"))
    if TRUE_TOTAL not in source.text:
        raise AssertionError(
            f"the planted surgery cannot find {TRUE_TOTAL!r} in the emitted "
            "body, so it would publish an artifact identical to the real one and "
            "the non-vacuity arm would certify nothing:\n" + source.text
        )
    planted = source.text.replace(TRUE_TOTAL, PARTITION_WIDTH)
    result = compile_source(planted, name,
                            CompileOptions(backend="host", mode="float64",
                                           cache_dir=cache_dir))
    shutil.copyfile(result.artifact, directory / f"{name}.so")
    (directory / f"{name}.cpp").write_text(planted)
    return directory


def with_sidecar(directory, sidecar: dict, name: str = "fraction") -> Path:
    """Write ``sidecar`` beside the planted object, so the same loader path
    reads it. The sidecar is the REAL one: the artifact lies in its BODY, which
    is the whole point — no declaration says it is not partition-invariant."""
    directory = Path(directory)
    (directory / f"{name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return directory
