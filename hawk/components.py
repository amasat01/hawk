# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Components hawk loads: installed distributions that publish a component package under the ``raptor.components`` entry-point
group. A component declares in its manifest (``MANIFEST["entries"]``, module -> names) what hawk calls; :func:`view` serves
exactly those names, so hawk reaches a component through its declared entry list and nothing else."""

from __future__ import annotations

import importlib
import importlib.metadata
from functools import cache
from types import SimpleNamespace

from ._errors import HawkError
from .variants import discover as _discover, entry_points  # noqa: F401  (the one door: variants.py holds it, stdlib only)

GROUP = "raptor.components"


def discover(group: str, name: str, kind: str, error=HawkError):
    """The one entry point named ``name`` of ``group`` (:func:`hawk.variants.discover`), refused with ``error``."""
    return _discover(group, name, kind, error)


@cache
def component(name: str):
    """The component package ``name`` of the installed distribution that declares it; its kinds join the front end's registry.

    One load per name (the function's own memo; a refusal is not remembered, so a later install is seen)."""
    found = [discover(GROUP, name, "component")]
    mod = found[0].load()
    from .dialects import KINDS

    KINDS.kinds.update(mod.KINDS.kinds)
    KINDS.classes.update(mod.KINDS.classes)
    return mod


def require(name: str, backend: str) -> None:
    """Refuse by name a build of ``backend`` when the component ``name`` it needs is not installed or cannot be imported."""
    try:
        component(name)
    except (HawkError, ImportError) as e:
        raise HawkError(f"the {backend!r} backend requires the {name!r} component, which is not available ({e}); "
                        f"install the distribution raptor-{name}") from e


class _View:
    """The declared entries of module ``part`` of component ``name``; the component loads at the first entry read."""

    def __init__(self, name: str, part: str) -> None:
        self._name, self._part, self._ns = name, part, None

    def __getattr__(self, entry: str):
        if self._ns is None:
            mod = component(self._name)
            declared = mod.MANIFEST["entries"][self._part]
            sub = importlib.import_module(f"{mod.__name__}.{self._part}")
            self._ns = SimpleNamespace(**{n: getattr(sub, n) for n in declared})
        try:
            return getattr(self._ns, entry)
        except AttributeError:
            raise HawkError(f"{entry!r} is not a declared entry of the {self._name!r} component's {self._part!r} module") from None


@cache
def view(name: str, part: str) -> _View:
    """The declared entries of module ``part`` of component ``name``, as a namespace (loaded on first use; one view per pair)."""
    return _View(name, part)


def device_view(part: str) -> _View:
    """The declared entries of module ``part`` of the component a device backend names beyond the ones the host backends name (the
    form only a device lowering leaves behind); refused by name when no installed default backend names one."""
    from . import registry

    rows = [registry.traits(b) for b in registry.backend_ids(default=True)]
    shared = {c for t in rows if not t.get("device") for c in t.get("components", ())}
    names = sorted({c for t in rows if t.get("device") for c in t.get("components", ())} - shared)
    if len(names) != 1:
        raise HawkError("no device backend is installed that names a component of its own"
                        if not names else f"the device backends name several components of their own: {names}")
    return view(names[0], part)
