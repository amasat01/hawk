# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two facts about the index widths — the TRIPLE is int64, the ``nsamples`` ROLE
is not, and the mandated include order is what keeps the second true.

**The role stays narrow.** The role and the triple's ``nSamples`` are different wire
objects. Every consumer packs the ROLE as 4 bytes today (``ctypes.c_uint32``,
``counts.push_back(unsigned(n))``, the device registry's "``aether::idx_t`` ->
uint32, passed by value"), so emitting it as ``int64`` would shift every
parameter after it by 4 bytes — deterministic garbage, no crash, the failure
mode exactly. HAWK therefore emits the role at ``EAGLE_ABI_INDEX_T``, and this
row pins that spelling AND the width it resolves to, at the default and under
``-DAETHER_INDEX_T=std::uint64_t``.

**The include order is what keeps that true.** ``EAGLE_ABI_INDEX_T`` binds to ``AETHER_INDEX_T`` only if that
macro is visible when ``plugin/gref_layout.h`` is preprocessed, so the include
order is contractual: aether FIRST, eagle's ABI header second. Getting it
backwards silently NARROWS the index width. The row asserts the order in the
emitted text and then compiles the emitted TU under the 64-bit define and reads
the artifact's own ``eagle_layout_sizes[3]`` — the field that IS
``sizeof(EAGLE_ABI_INDEX_T)`` — back out of it.

This test previously failed. Two plants: the ``nsamples`` role emitted ``long
long`` --

    AssertionError: assert 'EAGLE_ABI_INDEX_T p_nsm_n_samples' in
      '... eagle::plugin::ScalarHandle p_psc_x,\n long long p_nsm_n_samples,
       \n long long base, ...'

-- and the include order reversed (``plugin/gref_layout.h`` first) --
``AssertionError: (b): aether must be included FIRST -- EAGLE_ABI_INDEX_T
binds to AETHER_INDEX_T only if that macro is already visible``. Both plants
were then removed.
"""

from __future__ import annotations

import ctypes

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from _device_file import device_artifact
from conftest import sidecar_of

from hawk.artifact import build_bundle
from hawk.compile import aether_include, eagle_include

#: The layout array's positional index of ``sizeof(aether::idx_t)``.
IDX_FIELD = 3

WIDE = ("AETHER_INDEX_T=std::uint64_t",)


@pytest.fixture(scope="module")
def wide_bundle(tmp_path_factory, cache_dir):
    """The SAME kernel, built under a 64-bit aether index."""
    return build_bundle([D.fraction], tmp_path_factory.mktemp("hawk_wideidx"),
                        targets=("cuda", "host"), cache_dir=cache_dir, defines=WIDE)


# --------------------------------------------------------------------------- #
# -- the include order.
# --------------------------------------------------------------------------- #
def test_the_mandated_include_order_is_aether_first(built):
    for suffix in (".cu", ".cpp"):
        text = (built["fraction"].directory / f"fraction{suffix}").read_text()
        lines = text.splitlines()
        aether = min(i for i, ln in enumerate(lines) if ln.startswith("#include <aether/"))
        eagle = next(i for i, ln in enumerate(lines) if 'plugin/gref_layout.h' in ln)
        assert aether < eagle, (
            "aether must be included FIRST -- EAGLE_ABI_INDEX_T binds to "
            "AETHER_INDEX_T only if that macro is already visible, and the reverse "
            f"order silently narrows the index width:\n{text[:800]}"
        )


def test_the_resolved_header_roots_are_recorded(built):
    sc = sidecar_of(built["fraction"], "fraction")
    assert sc["eagle_include"] == eagle_include()
    assert sc["aether_include"] == aether_include()
    assert (ctypes.c_char * 1) is not None  # keep the import honest


@pytest.mark.gpu
def test_the_exported_width_follows_the_define(built, wide_bundle):
    """The observation: the artifact reports its OWN index width, and it is
    the compile's, not a hand-written number."""
    import cupy as cp

    narrow = _layout(built["fraction"].directory / "fraction.so")
    assert narrow[IDX_FIELD] == 4, narrow

    wide = _layout(wide_bundle.directory / "fraction.so")
    assert wide[IDX_FIELD] == 8, (
        "under -DAETHER_INDEX_T=std::uint64_t the exported "
        f"eagle_layout_sizes[{IDX_FIELD}] must report 8; got {wide}"
    )
    module = cp.RawModule(path=str(device_artifact(wide_bundle.directory, "fraction")))
    module.get_function("fraction")
    sizes = [int(v) for v in cp.ndarray(
        (5,), dtype=cp.uint64,
        memptr=module.get_global("eagle_layout_sizes")).get()]
    assert sizes[IDX_FIELD] == 8, sizes


def _layout(so):
    lib = ctypes.CDLL(str(so))
    return list((ctypes.c_uint64 * 5).in_dll(lib, "eagle_layout_sizes"))


# --------------------------------------------------------------------------- #
# -- the nsamples ROLE's width.
# --------------------------------------------------------------------------- #
def test_the_nsamples_role_is_spelled_at_the_abi_index_width(built):
    cu = (built["fraction"].directory / "fraction.cu").read_text()
    entry = cu.split('extern "C" __global__ void fraction(')[1].split(")")[0]
    assert "EAGLE_ABI_INDEX_T p_nsm_n_samples" in entry, entry
    assert "long long p_nsm" not in entry and "std::int64_t p_nsm" not in entry, (
        "int64 is the TRIPLE only; emitting the nsamples ROLE at 64 "
        f"bits shifts every parameter after it by 4 bytes:\n{entry}"
    )
    # ... and the triple that FOLLOWS it is int64, in both entries.
    assert "long long base" in entry and "long long nSamples" in entry
    cpp = (built["fraction"].directory / "fraction.cpp").read_text()
    assert "std::int64_t base" in cpp and "std::int64_t nSamples" in cpp


def test_the_role_width_tracks_the_macro_not_a_literal(built, wide_bundle):
    """The SAME emitted spelling under both builds; only the macro moved. That
    is what makes the width a property of the build rather than of the emitter
    (and the exported field above is the measurement)."""
    narrow = (built["fraction"].directory / "fraction.cu").read_text()
    wide = (wide_bundle.directory / "fraction.cu").read_text()
    assert "EAGLE_ABI_INDEX_T p_nsm_n_samples" in narrow
    assert "EAGLE_ABI_INDEX_T p_nsm_n_samples" in wide
    assert narrow == wide, "the DEFINE is a compile flag, never an emitter branch"
    assert _layout(built["fraction"].directory / "fraction.so")[IDX_FIELD] !=\
        _layout(wide_bundle.directory / "fraction.so")[IDX_FIELD]


def test_the_parameter_block_is_not_shifted_at_the_default_width(built):
    """The executable half: a body whose ``nsamples`` role sat at the wrong
    width would read the parameter after it (or the triple) as its sample
    count. Running it is what says the block lines up."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    x = np.arange(n, dtype=float) + 1.0
    plugin = L.host_plugin(built["fraction"].directory, "fraction",
                           sidecar_of(built["fraction"], "fraction"))
    got = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x)
    np.testing.assert_allclose(got, D.ref_fraction(x, n))
