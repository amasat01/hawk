# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Local memory per thread: recorded in the sidecar, refused by name past the device limit, and the
keyed (block-sum) payload shape recorded beside its width.

A launch whose per-thread local memory passes CUDA's 512 KB limit fails, and the stream sync alone
reports nothing; the build refuses it by name instead (from ptxas's stack frame of a CUBIN, else from
the local depots the PTX declares). The device driver of ``_team_stages`` also asks the runtime for
the launch status after every launch."""

from __future__ import annotations

import types
import warnings

import pytest
from _block_kernels import gram_kernel
from _linalg_kernels import chol_kernel, eigh_kernel, lu_kernel
from conftest import sidecar_of

from hawk import HawkError
from hawk.artifact import bundle as B
from hawk.artifact import build_bundle
from hawk.artifact.sidecar import LOCAL_LIMIT_B
from hawk.trace.kernel import Kernel


def _unit(tmp_path, ptx, ptxas=None):
    path = tmp_path / "k.ptx"
    path.write_text(ptx)
    result = types.SimpleNamespace(artifact=path, ptxas=ptxas or {})
    return {"cuda": (result, "k.ptx", "k.cu", "ptx")}


def test_a_ptx_whose_local_depot_passes_the_limit_is_refused_by_name(tmp_path):
    ptx = (".visible .entry k(.param .u64 p)\n{\n.local .align 8 .b8 \t__local_depot0[%d];\n}\n" % (LOCAL_LIMIT_B + 8))
    with pytest.raises(HawkError, match=rf"kernel 'k': {LOCAL_LIMIT_B + 8:,} bytes of local memory per thread \(ptx\), "
                                        rf"past the device limit of {LOCAL_LIMIT_B:,}.*linalg_form='team'"):
        B._local_memory("k", _unit(tmp_path, ptx), "cuda")


def test_ptxas_frame_is_the_number_when_the_compile_ran_ptxas(tmp_path):
    row = {"k": {"regs": 40, "spill_store_B": 0, "spill_load_B": 0, "stack_B": LOCAL_LIMIT_B + 1}}
    with pytest.raises(HawkError, match=r"\(ptxas\)"):
        B._local_memory("k", _unit(tmp_path, "", row), "cuda")
    row["k"]["stack_B"] = 4096
    assert B._local_memory("k", _unit(tmp_path, "", row), "cuda") == (4096, "ptxas")


def test_an_entry_runs_alone_and_its_device_functions_add_to_it(tmp_path):
    ptx = (".visible .entry a()\n{\n.local .align 8 .b8 __local_depot0[1000];\n}\n"
           ".visible .entry b()\n{\n.local .align 8 .b8 __local_depot1[3000];\n}\n"
           ".func f()\n{\n.local .align 8 .b8 __local_depot2[500];\n}\n")
    assert B._local_memory("k", _unit(tmp_path, ptx), "cuda") == (3500, "ptx")


def test_a_128_matrix_in_one_thread_is_refused_by_name_and_as_a_team_it_holds_no_frame(tmp_path, cache_dir):
    mixed = Kernel("la_mix_128", (*lu_kernel(128).sinks, *chol_kernel(128).sinks, *eigh_kernel(128).sinks))
    with pytest.raises(HawkError, match=r"linalg_form='thread' keeps the matrix in one lane"):
        build_bundle([mixed], tmp_path / "thread", linalg_form="thread", targets=("host",), cache_dir=cache_dir)
    bundle = build_bundle([lu_kernel(71)], tmp_path / "team", targets=("cuda", "host"), cache_dir=cache_dir)
    launch = sidecar_of(bundle, "la_lu_71")["launch"]
    assert launch["local_limit"] == LOCAL_LIMIT_B and launch["local_source"] in ("ptx", "ptxas")
    assert 0 <= launch["local_bytes"] < LOCAL_LIMIT_B // 8       # the matrix lives in team scratch


def test_the_keyed_record_carries_the_payload_shape(tmp_path, cache_dir):
    kernel = gram_kernel(24, 5)
    bundle = build_bundle([kernel], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    stages = sidecar_of(bundle, kernel.name)["stages"]
    keyed = [st["team"]["keyed"] for st in stages if st.get("team", {}).get("keyed")]
    assert keyed and keyed[0]["payload"] == [24, 24] and keyed[0]["width"] == 24 * 24


def test_a_wide_value_placed_in_a_plane_holds_no_local_frame_and_the_thread_form_keeps_it(tmp_path, cache_dir):
    from _wide_kernels import fill_mask

    placed = build_bundle([fill_mask], tmp_path / "auto", targets=("cuda",), cache_dir=cache_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # a cache of its own: a unit another test built is not compiled again, and a build that does not compile does not warn
        pinned = build_bundle([fill_mask], tmp_path / "thread", targets=("cuda",), cache_dir=tmp_path / "cache", wide_form="thread")
    figures = (sidecar_of(pinned, "fill_mask").get("ptxas") or {}).values()
    if any(row.get("spill_store_B") is not None for row in figures):    # ptxas ran: the thread form spills on purpose, and the build says so
        assert any(issubclass(w.category, UserWarning) and "local frame" in str(w.message) for w in caught)
    assert sidecar_of(placed, "fill_mask")["launch"]["local_bytes"] == 0
    assert sidecar_of(pinned, "fill_mask")["launch"]["local_bytes"] > 0       # the 512-wide value in one thread


@pytest.mark.gpu
def test_a_failed_device_call_left_pending_is_raised_by_name():
    import ctypes

    import cupy as cp
    from _team_stages import _cudart, check_launch

    cp.zeros(1)                                           # a context exists
    check_launch("fine")                                  # nothing pending
    _cudart().cudaMalloc(ctypes.byref(ctypes.c_void_p()), ctypes.c_size_t(1 << 62))     # fails, sets the error
    with pytest.raises(RuntimeError, match=r"kernel 'k': the launch failed \(.*cuda error 2\)"):
        check_launch("k")
    check_launch("k")                                     # read once: cleared
