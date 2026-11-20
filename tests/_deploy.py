# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The consumer side of a HAWK artifact: LOAD it through hawk, then run it.

Deliberately in ``hawk/tests/`` and not in ``hawk/hawk/``: what this module does is what any consumer must, through
hawk's own verbs (:func:`hawk.load`, :func:`hawk.plan`):

1. :func:`load_bundle` loads each kernel of a bundle's manifest, which runs the load-time gates of the library (the
   layout sizes, the ABI tag) and refuses a mismatch NAMING THE FIELD;
2. the SELF-CHECK: read the artifact's OWN exported ``eagle_abi_tag`` and ``eagle_layout_sizes`` and refuse a mismatch
   NAMING THE FIELD. The HOST half is :class:`hawk._core.HostLibrary`'s constructor (dlopen, the tag, the layout, all
   in C++), so the check a consumer gets is the one HAWK's own execution path performs and not a second Python
   transcription of it. The DEVICE half stays here in cupy, because there is no dlsym on a PTX module and HAWK's host
   path never loads one; it compares against the sizes and the tag of this build (:mod:`hawk._core`,
   :mod:`hawk._contracts`);
3. project the sidecar into the plan-able declaration with ``hawk.artifact.plan_view`` and attach the loaded entry
   point.

A fourth, narrower check: :func:`check_declared_planes` is HAWK's OWN bind rule for the width and the dtype of a declared
plane, asked of the v2 sidecar's ``arg_shapes``/``arg_dtypes`` fields. A plane bound at the wrong INTEGER width has no
gate in the float planes' dtype check (an integer or bool array is passed at its own dtype); since the v2 sidecar knows
every slot's wire dtype, a consumer can close that gap on its own side.
"""

from __future__ import annotations

import ctypes
import pathlib
from types import MappingProxyType, SimpleNamespace

import json

from conftest import refuse_device_row

import hawk
from hawk import _contracts, _core
from hawk.artifact import fold_entries, plan_view

#: How many entries the layout-sizes array carries (``gref_abi.h``'s own constant).
LAYOUT_FIELDS = 5


def load_bundle(directory):
    """The kernels of a HAWK bundle's manifest, loaded: ``{id: entry}`` with ``entry.loaded`` the kernel :func:`hawk.load` made and
    ``entry.derivative`` the derivative block its sidecar states (``None`` for a primal)."""
    directory = pathlib.Path(directory)
    plugins = json.loads((directory / "manifest.json").read_text())["plugins"]
    entries = {}
    for plugin in plugins:
        loaded = hawk.load(directory, plugin["id"])
        entries[plugin["id"]] = SimpleNamespace(loaded=loaded, derivative=loaded.sidecar.get("derivative"))
    return entries


def host_selfcheck(so_path) -> tuple:
    """Read the host object's OWN exports and refuse a mismatch.

    THE CHECK IS ``hawk._core``'s. Constructing a
    :class:`hawk._core.HostLibrary` dlopens the artifact ``RTLD_LOCAL|RTLD_NOW``,
    reads its exported ``eagle_abi_tag`` and ``eagle_layout_sizes``, and refuses a
    mismatch, so the refusal names the field in the one spelling every other door
    uses, and a consumer gets the check HAWK's execution path actually performs
    rather than a Python re-statement of it.

    Returns ``(tag, sizes, cdll)``. The CDLL is opened because a plugin view takes
    the entry as a raw ADDRESS, and the object that address points into has to be
    kept alive by something the caller holds; the ``HostLibrary`` is attached to it
    as its keepalive too."""
    lib = _core.HostLibrary(str(so_path))                # refuses, naming the field
    tag_value, sizes = lib.abi_tag, list(lib.layout_sizes)
    cdll = ctypes.CDLL(str(so_path))
    cdll._hawk_core_library = lib
    return tag_value, sizes, cdll


def device_selfcheck(module) -> tuple:
    """The PTX twin of :func:`host_selfcheck` — ``cuModuleGetGlobal`` through
    cupy's ``RawModule.get_global`` (there is no dlsym on a
    PTX module)."""
    import cupy as cp

    sizes = [int(v) for v in cp.ndarray(
        (LAYOUT_FIELDS,), dtype=cp.uint64,
        memptr=module.get_global("eagle_layout_sizes")).get()]
    raw = bytes(cp.ndarray((len(_contracts.AETHER_ABI_VERSION) + 1,), dtype=cp.uint8,
                           memptr=module.get_global("eagle_abi_tag")).get())
    tag_value = raw.split(b"\0", 1)[0].decode()
    if tag_value != _contracts.AETHER_ABI_VERSION:
        raise ValueError(f"the device module: exported eagle_abi_tag is {tag_value!r}, this build speaks "
                         f"{_contracts.AETHER_ABI_VERSION!r}")
    for field, (got, want) in enumerate(zip(sizes, _core.layout_sizes())):
        if got != want:
            raise ValueError(f"the device module: layout field {_core.layout_field_name(field)} is {got} bytes, "
                             f"this build expects {want}")
    return tag_value, sizes


def device_plugin(bundle_dir, name, sidecar=None):
    """The loaded kernel of a bundle's DEVICE entry, for ``hawk.plan(loaded, hawk.DeviceKernel)``: the load runs the library's
    own self-check, and a row that needs a device is skipped under ``--host-only``."""
    refuse_device_row()
    return hawk.load(bundle_dir, name, sidecar)


def host_plugin(bundle_dir, name, sidecar):
    """The plan-able view of a bundle's HOST entry, self-checked.

    The manifest names the DEVICE artifact (a manifest ``format`` is one of
    ``ptx``/``cubin``/``fatbin``); the host object is its sibling ``<id>.so``
    and the sidecar's ``host_entry`` names the symbol — the previous code
    generator's ``compile_to_ptx(also_host=True)`` convention, so nothing new is invented."""
    so = pathlib.Path(bundle_dir) / f"{name}.so"
    _tag, _sizes, lib = host_selfcheck(so)
    view = plan_view(sidecar)
    entry = getattr(lib, view.pop("host_entry"))
    library = _core.HostLibrary(str(so))
    return SimpleNamespace(host_entry=ctypes.cast(entry, ctypes.c_void_p).value, sidecar=MappingProxyType(sidecar),
                           _keepalive=(lib, library), **fold_entries(library, sidecar), **view)


#: The sidecar's ``arg_dtypes`` string -> the numpy dtype a caller-bound array
#: must present. Built from the SAME four-value vocabulary
#: ``hawk/artifact/sidecar.py``'s ``_wire_dtype`` writes, so this door and the
#: sidecar cannot silently drift about what a string like ``"int64"`` means.
_NUMPY_OF_WIRE = {"float64": "float64", "float32": "float32", "int64": "int64",
                  "int32": "int32", "uint32": "uint32", "bool": "bool"}


def check_declared_planes(sidecar: dict, arrays: dict, n_samples: int) -> None:
    """HAWK's OWN bind-time width/dtype guard — the previous code generator's
    ``vec_widths`` bind rule, re-owned on HAWK's v2 sidecar (see this module's
    docstring).

    Every declared plane in ``arrays`` is checked against the sidecar's
    ``arg_shapes``/``arg_dtypes`` and a mismatch is refused NAMING THE SLOT,
    before anything is packed or launched, including the ONE case the float
    planes' dtype check does not reach (an integer- or bool-typed plane). Read from ``arg_shapes`` rather than
    ``arg_widths``: the two agree on every statically-shaped role, but only
    ``arg_shapes`` is COMPLETE enough to say, for a runtime-length role, that
    there IS no static width to check (``arg_shapes[name] is None``) rather
    than leaving a reader to guess whether an absent key means that or a bug.
    Either way such a slot is skipped: guessing a width for it is exactly
    what this avoids."""
    import numpy as np

    widths = sidecar.get("arg_shapes", {})
    dtypes = sidecar.get("arg_dtypes", {})
    for role, name in (tuple(pair) for pair in sidecar["arg_spec"]):
        if name not in arrays:
            continue
        arr = np.asarray(arrays[name])
        want_dtype = dtypes.get(name)
        if want_dtype is not None:
            got_dtype = np.dtype(_NUMPY_OF_WIRE[want_dtype])
            if arr.dtype != got_dtype:
                raise ValueError(
                    f"hawk consumer bind check: {role} {name!r} is declared "
                    f"dtype {want_dtype!r}, but the bound plane is {arr.dtype} "
                    "(HAWK's own vec_widths behaviour)"
                )
        want_width = widths.get(name)
        if want_width is not None:
            want_shape = ((n_samples,) if want_width <= 1
                          else (want_width, n_samples))
            if tuple(arr.shape) != want_shape:
                raise ValueError(
                    f"hawk consumer bind check: {role} {name!r} is declared "
                    f"{want_width} component(s) wide, so at n={n_samples} its "
                    f"plane is {want_shape}; got {tuple(arr.shape)} "
                    "(HAWK's own vec_widths behaviour)"
                )


def resolve_primal(sidecar: dict, own_bundle, other_bundles: dict | None = None) -> dict:
    """Resolve a derivative sidecar's ``{kind, wrt, primal, primal_unit}``
    back-reference to the PRIMAL's own sidecar — HAWK's producer-side
    contract (``hawk/artifact/bundle.py``'s ``_per_kernel_derivative``) made
    concrete on the CONSUMER's side, which is where the reference is actually
    walked: a producer names a unit by digest because it may not have that
    unit's ``Bundle`` object in hand any more (a build in a different process,
    or a different session entirely) — a consumer that DOES load a set of
    units is the one that can turn the digest back into a sidecar.

    ``primal_unit`` absent/``None`` means "the same bundle as this
    derivative" (the original, single-manifest case) and is looked
    up in ``own_bundle`` directly; any other value is looked up in
    ``other_bundles``, a ``{unit_digest: Bundle}`` map of every OTHER unit
    this consumer has loaded — the shape a real consumer builds once, from
    every manifest it opened, rather than per reference. Refuses NAMING what
    it could not resolve: an unrecognised digest (the caller loaded the wrong
    set, or the reference is stale/wrong) or a primal name absent from the
    unit its digest does resolve to (a corrupt or hand-edited sidecar) are two
    different mistakes and get two different messages."""
    derivative = sidecar.get("derivative")
    if derivative is None:
        raise ValueError(
            f"{sidecar.get('kernel')!r} carries no derivative block to resolve "
            "a primal reference from"
        )
    primal_name = derivative["primal"]
    unit_digest = derivative.get("primal_unit")
    if unit_digest is None:
        bundle = own_bundle
    else:
        bundle = (other_bundles or {}).get(unit_digest)
        if bundle is None:
            raise ValueError(
                f"{sidecar.get('kernel')!r}'s derivative names primal_unit "
                f"{unit_digest!r}, which is none of the units this consumer "
                f"has loaded ({sorted(other_bundles or {})})"
            )
    for artifact in bundle.artifacts:
        if artifact.name == primal_name:
            return artifact.sidecar
    raise ValueError(
        f"{sidecar.get('kernel')!r}'s derivative names primal {primal_name!r}, "
        f"which is not a member of the resolved unit (has: "
        f"{[a.name for a in bundle.artifacts]})"
    )
