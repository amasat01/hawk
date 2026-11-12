# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""BUILD POLICIES: the knobs that steer code generation and compilation, as ONE explicit value (never ambient).

A :class:`BuildPolicy` is passed as ``build_policy=`` to a public build entry (``build``, ``emit_source``, ``build_bundle``,
``render_source``, ``hawk_c.render``). The entry validates it here, checks that every key it sets is DECLARED by a backend of
the build (a backend's manifest lists the policy keys it accepts under ``"policy"``; an undeclared key is refused by name), and
turns it into the plain named values of the build spec's :class:`hawk.passes.policy.Settings`. This type, its validation and its
defaults live in hawk only: below hawk only the plain values travel."""

from __future__ import annotations

from types import MappingProxyType

from dataclasses import dataclass, fields, replace

from ._errors import HawkError
from .passes import policy as _settings


@dataclass(frozen=True)
class BuildPolicy:
    """Every field left ``None`` is unset (the build's default, or the entry's own keyword).

    ``emit_form`` how a product, dot, sum or norm is spelled (``auto``, ``static``, ``rolled``); ``linalg_form`` how per-lane linear
    algebra is worked (``auto``, ``thread``, ``team``); ``wide_form`` where a per-lane value wider than a thread's registers
    lives (``auto``, ``thread``, ``team``); ``contraction`` how a dense contraction is run (``direct``, ``gemm``); ``launch_bounds``
    ``(threads, blocks_per_sm)`` for a device kernel; ``fold_rows`` the rows per warp forced in every cell-sum kernel."""

    emit_form: str | None = None
    linalg_form: str | None = None
    wide_form: str | None = None
    contraction: str | None = None
    launch_bounds: tuple | None = None
    fold_rows: int | None = None

    def __post_init__(self):
        for key in _settings.CHOICES:
            value = getattr(self, key)
            if value is not None:
                _settings.refuse_choice(key, value)

    def keys(self) -> tuple:
        """The names of the fields this policy sets."""
        return tuple(f.name for f in fields(self) if getattr(self, f.name) is not None)


#: field -> the :class:`~hawk.passes.policy.Settings` field it becomes.
_SETTING = MappingProxyType({"emit_form": "form", "linalg_form": "linalg_form", "wide_form": "wide_form", "contraction": "contraction",
                             "launch_bounds": "launch_bounds", "fold_rows": "fold_k"})


def settings_of(build_policy: BuildPolicy | None, base):
    """``base`` (a :class:`~hawk.passes.policy.Settings`) with the fields ``build_policy`` sets laid over it: plain values only."""
    if build_policy is None:
        return base
    if not isinstance(build_policy, BuildPolicy):
        raise HawkError(f"build_policy must be a hawk.build_policy.BuildPolicy, got {type(build_policy).__name__}")
    return replace(base, **{_SETTING[k]: getattr(build_policy, k) for k in build_policy.keys()})


def refuse_undeclared(build_policy: BuildPolicy | None, targets) -> None:
    """Refuse by name a key ``build_policy`` sets that no target of the build declares in its manifest (``"policy"``)."""
    if build_policy is None:
        return
    from . import registry

    declared = set()
    for target in targets:
        if registry.has("backends", target):
            declared |= set(registry.traits(target).get("policy", ()))
    for key in build_policy.keys():
        if key not in declared:
            raise HawkError(f"build policy key {key!r} is declared by none of the targets {tuple(targets)}: a backend's manifest "
                            f"names the policy keys it accepts")


def apply(spec, build_policy: BuildPolicy | None, targets):
    """``spec`` with ``build_policy`` laid over its settings, after the keys it sets are checked against the targets' manifests."""
    if build_policy is None:
        return spec
    refuse_undeclared(build_policy, targets)
    return replace(spec, settings=settings_of(build_policy, spec.settings))
