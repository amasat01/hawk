# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build specs for tests that call a pass, a renderer or a cost outside a build: variant data passed in, never a module table patched."""

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
