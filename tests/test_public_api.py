# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""ONE canonical import path per public name, and the public manifest.

The authoring vocabulary (``kernel``, ``Param``, ``Mutable``, ``Table``,
``Vector``, ``Scalar``, ``Accum``, ``Reduce``, ``Quantity``, ``Terminated``,
``HawkError``, …) lives at the top level; math functions live at
``hawk.math``; the dependency-free types (``TensorType``, ``Wire``,
``PER_SAMPLE_ROLES``, …) at ``hawk.types``. ``hawk.trace`` exports nothing,
and ``hawk.ir`` / ``hawk.emit`` are internal: their ``__all__`` is the small
"IR access" surface a downstream package reads.

TWO rows pin this:

1. :func:`test_the_public_manifest_is_exactly_this` — the per-surface
   ``__all__`` set, sorted, pinned verbatim. A name silently added or
   removed from a public surface fails here first, by name.
2. :func:`test_no_public_name_answers_to_two_paths` — no public name is
   reachable from more than one of the surfaces below.
"""

from __future__ import annotations

from collections import defaultdict

import hawk
import hawk.artifact
import hawk.compile
import hawk.diff
import hawk.emit
import hawk.ext
import hawk.ir
import hawk.math
import hawk.runtime
import hawk.trace
import hawk.types

#: Every public SURFACE this row accounts for, by its import spelling.
#: ``hawk.types`` is the one surface with no ``__all__`` (its own module
#: docstring: "importable at near-zero cost" — adding one costs nothing at
#: runtime, but this row does not require it to pin the light tier's
#: public set; :data:`_TYPES_PUBLIC_NAMES` below is pinned by hand instead).
_SURFACES = {
    "hawk": hawk,
    "hawk.trace": hawk.trace,
    "hawk.math": hawk.math,
    "hawk.runtime": hawk.runtime,
    "hawk.ir": hawk.ir,
    "hawk.emit": hawk.emit,
    "hawk.ext": hawk.ext,
    "hawk.compile": hawk.compile,
    "hawk.diff": hawk.diff,
    "hawk.artifact": hawk.artifact,
}

#: ``hawk.types`` carries no ``__all__``; its public names are pinned here.
_TYPES_PUBLIC_NAMES = frozenset({"DTYPES", "LaneMeta", "PER_SAMPLE_ROLES", "Slot",
                                 "TensorType", "Wire"})

#: The exact, sorted public name set per surface — the MANIFEST. Any name
#: added to or removed from a surface's ``__all__`` moves this row off its
#: pin; update BOTH here and in the surface's own ``__all__`` together, in
#: the SAME commit, same as any other committed manifest in this repo.
_EXPECTED_MANIFEST = {
    "hawk": [
        "Accum", "HawkError", "Index", "Kernel", "Matrix", "Mutable",
        "Param", "Quantity", "Quat", "RawBlock", "Reduce", "Scalar",
        "Staged", "Table", "Terminated", "Value", "Vector", "Wide",
        "WideOut", "__version__", "build", "kernel", "load", "raw_device",
        "run", "samples_first", "samples_last", "steps",
    ],
    "hawk.trace": [],
    "hawk.math": [
        "abs", "absolute", "acos", "acosh", "argmax", "as_pure",
        "as_vec3", "asin", "asinh", "atan", "atan2", "atanh", "cbrt",
        "ceil", "clip", "copysign", "cos", "cosh", "cross",
        "dispatch", "dot", "erf", "erfc", "exp", "exp2", "expm1",
        "fdim", "floor", "fma", "fmod", "hypot", "isfinite", "isinf",
        "isnan", "land", "lnot", "log", "log10", "log1p", "log2",
        "lor", "max", "maximum", "min", "minimum", "n_samples",
        "norm", "outer", "pow", "power", "quat_conj", "quat_mul",
        "quat_recip", "quat_rotate", "random_bernoulli",
        "random_exponential", "random_lognormal",
        "random_multivariate_normal", "random_normal",
        "random_poisson1", "random_uniform", "random_uniform_int",
        "remainder", "rint", "round", "rsqrt", "sample_index",
        "select", "sign", "sin", "sinh", "split_index", "sqrt",
        "take", "tan", "tanh", "transpose", "trunc", "vec", "vsum",
        "where",
    ],
    "hawk.runtime": [
        "ArgBlock", "DEVICE_CPU", "HostKernel",
        "descriptor_for_sidecar", "kind_for", "load", "run",
    ],
    "hawk.ir": [
        "AccumWrite", "Assign", "At", "Const", "Op", "canonical",
        "infer", "make", "recognize",
    ],
    "hawk.emit": [
        "BACKENDS", "compose", "render_body", "render_source",
        "scalar_mode",
    ],
    "hawk.ext": [
        "ATOMIC", "DEFAULT_GUARD", "DEFAULT_KIND", "Guard",
        "KernelKind", "Kind", "Output", "PLAIN", "PrimitiveDef",
        "compensated", "primitive", "primitives", "sink_policy",
    ],
    "hawk.compile": [
        "Cache", "CompileOptions", "DEVICE", "DeviceImage", "HOST",
        "HOST_PROFILES", "OPT_LEVELS", "aether_include", "cache_stats",
        "compile_source", "compiler_identity", "cubin",
        "cubin_available", "current_payload", "default_cache_dir",
        "device", "device_compiler", "device_compiler_kind",
        "digest_file", "eagle_include", "host_codegen_flags",
        "host_compiler", "host_flags", "host_profile", "lookup_key",
        "nvrtc", "opt_level", "publish", "reset_cache_stats",
    ],
    "hawk.diff": [
        "ADJOINT_PREFIX", "Derived", "TANGENT_PREFIX", "jvp", "vjp",
    ],
    "hawk.artifact": [
        "Artifact", "Bundle", "EXEC_ACCESS_CLASSES", "EXEC_OPS",
        "EXEC_TARGETS", "Entry", "LAYOUT_FIELDS", "PARAMS_SCHEMA",
        "STAMP_NAME", "TOP_LEVEL_KEY_ORDER_V2", "arch", "build",
        "build_bundle", "exports", "plan_view", "plugins",
        "reset_unit_memo", "unit_stats", "write_manifest",
    ],
}


def test_the_public_manifest_is_exactly_this():
    """Every surface's ``__all__``, sorted, must be EXACTLY the pinned set —
    a name quietly added or removed fails here, by name."""
    for surface, mod in _SURFACES.items():
        got = sorted(mod.__all__)
        want = sorted(_EXPECTED_MANIFEST[surface])
        assert got == want, (
            f"{surface}.__all__ drifted from the pinned manifest:\n"
            f"  missing (pinned, not exported): {sorted(set(want) - set(got))}\n"
            f"  extra (exported, not pinned):    {sorted(set(got) - set(want))}"
        )
    got_types = sorted(n for n in dir(hawk.types) if not n.startswith("_")
                       and n not in ("dataclass", "annotations"))
    assert got_types == sorted(_TYPES_PUBLIC_NAMES), (
        f"hawk.types's public names drifted: got {got_types}, "
        f"pinned {sorted(_TYPES_PUBLIC_NAMES)}"
    )


#: The top-level convenience doors (``hawk.__init__``'s ``_TOP_LEVEL_DOORS``)
#: that answer at ``hawk`` ON PURPOSE beside a submodule's own export of the
#: same name -- ``load``/``run`` are the identical object as
#: ``hawk.runtime``'s, reached by a second spelling; ``build`` is a
#: DIFFERENT object than ``hawk.artifact.build`` (the deployment-unit door
#: vs. the single-kernel primitive), overloaded across the two surfaces by
#: name alone. Every other duplicate below still fails: this is the one
#: named exception, not a loophole.
_TOP_LEVEL_DOOR_NAMES = frozenset({"build", "load", "run"})


def test_no_public_name_answers_to_two_paths():
    """No public name is reachable from more than one surface, except the
    top-level convenience doors in :data:`_TOP_LEVEL_DOOR_NAMES`, which
    answer at ``hawk`` by design beside their submodule's own export."""
    owners: dict[str, set] = defaultdict(set)
    for surface, mod in _SURFACES.items():
        for name in mod.__all__:
            owners[name].add(surface)
    for name in _TYPES_PUBLIC_NAMES:
        owners[name].add("hawk.types")
    duplicates = {name: paths for name, paths in owners.items() if len(paths) > 1}
    unexpected = {name: paths for name, paths in duplicates.items()
                 if not (name in _TOP_LEVEL_DOOR_NAMES and "hawk" in paths
                        and len(paths) == 2)}
    assert not unexpected, (
        f"these public names answer to MORE THAN ONE canonical path: {unexpected}"
    )
    for name in _TOP_LEVEL_DOOR_NAMES:
        assert duplicates.get(name) and "hawk" in duplicates[name], (
            f"{name!r} was expected to answer at both 'hawk' and its own "
            f"submodule; got {duplicates.get(name)} -- update "
            "_TOP_LEVEL_DOOR_NAMES if the door moved or was removed"
        )
