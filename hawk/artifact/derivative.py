# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel ``derivative`` sidecar blocks for a bundle, including a primal
published in a different unit."""

from __future__ import annotations

from collections.abc import Mapping

from ..diff import Derived
from ..ir import HawkError


# -- Per-kernel ``derivative`` resolution, including cross-unit primals. --- #
def _split_override(override) -> tuple:
    """One mapping VALUE as ``(primal_name, primal_unit_digest_or_None)``.

    A bare string names a primal publishing in this bundle
    (``primal_unit=None``); a ``(name, unit_digest)`` pair names one
    already published, or publishing, in a different unit."""
    if override is None:
        return None, None
    if isinstance(override, str):
        return override, None
    if isinstance(override, tuple) and len(override) == 2:
        return override
    raise HawkError(
        f"a derivative mapping value names a primal as a string ({'<name>'!r}) "
        "or, since HW7a4, a (name, unit_digest) pair for a primal published in "
        f"a DIFFERENT unit (S21-O2) — got {override!r}"
    )


def _per_kernel_derivative(kernels, names, derivative) -> list:
    """One ``derivative`` block PER KERNEL — never one value broadcast to
    every bundle member, which breaks the moment a bundle carries a
    primal alongside its own vjp/jvp.

    Two sources feed each kernel's block, and the mapping overrides one
    field rather than replacing the whole block: (1) the kernel's own
    recording — built from :func:`hawk.diff.vjp`/:func:`~hawk.diff.jvp`,
    ``.sinks`` is a :class:`hawk.diff.transform.Derived` tagged with the
    direction, the resolved ``wrt`` names and the primal's unit digest
    when already known; (2) the mapping (:func:`_split_override`), which
    names the primal and, for a cross-unit reference, its unit, since a
    kernel's own sinks carry no back-reference to which bundle member it
    was taken against. A kernel neither ``Derived``-tagged nor named in
    the mapping gets block ``None`` (an absent sidecar key).

    The same execution-axis rule that keeps one axis per manifest means a
    ``Table``/``Staged``-reading primal (``cross_sample_read``) and its
    VJP (a scatter, ``cross_sample_write``) can never share a bundle. A
    resolved primal outside this bundle is accepted provided a
    ``primal_unit`` digest names where it lives; the sidecar then carries
    that digest instead of the implicit ``None`` a same-bundle primal gets.

    Refuses a mapping key naming no bundle member (a silently-ignored
    typo would ship an untagged kernel), and a resolved primal that is
    neither a bundle member nor accompanied by a ``primal_unit`` digest —
    HAWK will not guess which other unit "not here" means."""
    if derivative is not None and not isinstance(derivative, Mapping):
        raise HawkError(
            "build_bundle's derivative= is per-kernel (HW7a3): a mapping "
            "{kernel_name: primal_name} (or, since HW7a4, {kernel_name: "
            "(primal_name, primal_unit_digest)}), or None for a bundle with "
            f"no derivative back-references — got {type(derivative).__name__}"
        )
    unknown_keys = [n for n in (derivative or {}) if n not in names]
    if unknown_keys:
        raise HawkError(
            f"derivative names {unknown_keys}, which "
            f"{'is' if len(unknown_keys) == 1 else 'are'} not a member of this "
            f"bundle; built are {names}"
        )
    out = []
    for k, n in zip(kernels, names):
        auto = k.sinks if isinstance(k.sinks, Derived) else None
        override_name, override_unit = _split_override(
            derivative.get(n) if derivative else None)
        if override_name is None and auto is None:
            out.append(None)
            continue
        if auto is None:
            raise HawkError(
                f"derivative names a primal for {n!r}, but {n!r} carries no "
                "recorded direction — it was not built by hawk.diff.vjp/jvp, "
                "so HAWK has no kind/wrt to stamp for it even with a primal "
                "name in hand"
            )
        primal_name = override_name if override_name is not None else auto.primal
        if primal_name is None:
            raise HawkError(
                f"{n!r} is a {auto.kind} derivative but its primal was taken "
                f"from an unnamed sink set (hawk.diff.{auto.kind} was not "
                f"handed a named Kernel) — pass derivative={{{n!r}: <primal "
                "name>}} to name it explicitly"
            )
        primal_unit = (override_unit if override_unit is not None
                       else getattr(auto, "primal_unit", None))
        if primal_name not in names:
            if primal_unit is None:
                raise HawkError(
                    f"{n!r}'s derivative names primal {primal_name!r}, which is "
                    f"not a member of this bundle; built are {names}, and no "
                    "primal_unit names the unit it lives in either ("
                    "S21-O2, HW7a4): a cross-unit primal must be named by "
                    f"digest — pass derivative={{{n!r}: ({primal_name!r}, "
                    "<unit digest>)}}, or tag hawk.diff.vjp/jvp's primal_unit= "
                    "when the primal was already published"
                )
        else:
            # A same-bundle primal is always spelled ``None``, even if a
            # digest was supplied: this unit's own digest isn't known yet
            # at this point in the build, and "in this bundle" already
            # says everything a digest would.
            primal_unit = None
        out.append({"kind": auto.kind, "wrt": list(auto.wrt), "primal": primal_name,
                    "primal_unit": primal_unit})
    return out
