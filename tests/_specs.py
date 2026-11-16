# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build specs for tests that call a pass, a renderer or a cost outside a build: facts passed in, never a module table patched."""

import dataclasses
from types import MappingProxyType

from hawk.targets import build_spec

#: A host-only build on this machine.
HOST = build_spec(("host",))
#: A host and device build on this machine, the device arch ``sm_61``.
DEVICE = build_spec(("host", "cuda"), "sm_61")


def device(arch: str):
    """A host and device build on this machine for device arch ``arch`` (``""``: a host-only build)."""
    return build_spec(("host", "cuda"), arch)


def for_targets(targets, arch: str = ""):
    """The spec of a build of ``targets`` (ids) on this machine."""
    return build_spec(tuple(targets), arch)


def spec_for(target, settings=None, spec=None):
    """The spec a render of ``target`` (a backend row) is given: ``spec`` (default: a build of that target alone) with ``settings`` laid over it."""
    from hawk.build_settings import apply

    return apply(build_spec((target.id,)) if spec is None else spec, settings, (target.id,))


def validation_spec():
    """The spec of a ``compose`` call: a build of every host row."""
    from hawk import registry
    from hawk.targets import TARGETS

    return build_spec(tuple(t for t in TARGETS if not registry.traits(t).get("device", False)))


def default_kind():
    """The kind a build resolves when neither the kernel nor the caller names one."""
    from hawk.ext import DEFAULT_KIND

    return DEFAULT_KIND


#: The caches a host row states on the machine the text digests were frozen with.
PINNED_CACHES = (32 * 1024, 1024 * 1024, "/sys/devices/system/cpu/cpu0/cache")


def with_caches(spec, l1d: int, l2: int, source: str = "test"):
    """``spec`` with the machine's cache-size facts set to these (the facts a cpu row states)."""
    facts = {**spec.machine.facts, "l1d_cache_bytes": l1d, "l2_cache_bytes": l2, "cache_source": source}
    return dataclasses.replace(spec, machine=dataclasses.replace(spec.machine, facts=MappingProxyType(facts)))


def pin_caches(monkeypatch):
    """Builds made after this call state the pinned caches, whatever machine runs the test (the cpu row's probe answers them)."""
    import ospreys.cpu.toolchain as toolchain

    monkeypatch.setattr(toolchain, "host_caches", lambda *a, **k: PINNED_CACHES)


def aether_writer(cells: bool = False):
    """The aether C++ writer a render outside a build is given (the building backend row holds its own)."""
    from aether.wings.base.writer_aether import AetherWriter
    from aether.wings.simt.writer_cells import CellsWriter

    return CellsWriter() if cells else AetherWriter()
