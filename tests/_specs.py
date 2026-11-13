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


def spec_for(target, build_policy=None, spec=None):
    """The spec a render of ``target`` (a backend row) is given: ``spec`` (default: a build of that target alone) with ``build_policy`` laid over it."""
    from hawk.build_policy import apply

    return apply(build_spec((target.id,)) if spec is None else spec, build_policy, (target.id,))


def validation_spec():
    """The spec of a ``compose`` call: a build of every host row."""
    from hawk import registry
    from hawk.targets import TARGETS

    return build_spec(tuple(t for t in TARGETS if not registry.traits(t).get("device", False)))


def default_kind():
    """The kind a build resolves when neither the kernel nor the caller names one."""
    from hawk.ext import DEFAULT_KIND

    return DEFAULT_KIND
