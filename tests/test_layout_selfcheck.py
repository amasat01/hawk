# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A wrong ``eagle_layout_sizes`` is REFUSED at load,
naming the field.

The artifact is built through the TEST-ONLY ``layout_sizes_override=`` door —
the ONLY caller of it — so a deliberately-wrong artifact exists
whose refusal this row observes; eagle needed a hand-written fixture
(``tests/fixtures/host_plugin_badlayout.cpp``) for exactly this purpose.

BOTH targets are checked, because they are read two different ways: ``dlsym``
on the host ``.so`` and ``cuModuleGetGlobal`` on the PTX module (there is
no dlsym on a PTX module). The tag is CORRECT in both, which is
the whole point: a tag alone can never catch a layout mismatch, and the
consequence of not catching it is deterministic garbage rather than a crash.

This test previously failed. Two plants: the ``layout_sizes_override=`` branch
disabled (``if False and ...``) --
``AssertionError: the test-only override must replace the DERIVED sizeof
expressions, or the row is asserting a refusal that cannot happen`` -- and,
the detector-deletion arm, the loader's ``eagle.exec.check_layout_sizes`` call
removed from ``_deploy.py`` --
``Failed: DID NOT RAISE ValueError``, i.e. the deliberately-wrong artifact
loaded cleanly. Both plants were then removed.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import pytest
from _device_file import device_artifact

from hawk.artifact import build_bundle

#: GRefMirror deliberately 41 instead of 40; every other field correct.
BAD = (41, 32, 32, 4, 24)


@pytest.fixture(scope="module")
def bad_bundle(tmp_path_factory, cache_dir):
    return build_bundle([D.axpb], tmp_path_factory.mktemp("hawk_badlayout"),
                        targets=("cuda", "host"), cache_dir=cache_dir,
                        layout_sizes_override=BAD)


def test_the_override_reaches_the_emitted_text(bad_bundle):
    for suffix in (".cpp", ".cu"):
        text = (bad_bundle.directory / f"axpb{suffix}").read_text()
        assert "41ull" in text and "sizeof(eagle::plugin::GRefMirror)" not in text, (
            "the test-only override must replace the DERIVED sizeof expressions, "
            "or the row is asserting a refusal that cannot happen:\n" + text
        )


def test_a_wrong_layout_host_artifact_is_refused_naming_the_field(bad_bundle):
    with pytest.raises(ValueError, match="GRefMirror") as excinfo:
        L.host_selfcheck(bad_bundle.directory / "axpb.so")
    assert "41" in str(excinfo.value) and "layout" in str(excinfo.value)


@pytest.mark.gpu
def test_a_wrong_layout_device_artifact_is_refused_naming_the_field(bad_bundle):
    import cupy as cp

    module = cp.RawModule(path=str(device_artifact(bad_bundle.directory, "axpb")))
    module.get_function("axpb")            # force the driver load
    with pytest.raises(ValueError, match="GRefMirror"):
        L.device_selfcheck(module)


def test_the_tag_is_correct_so_only_the_layout_can_have_caught_it(bad_bundle):
    """ point: a wrong-layout artifact can carry a perfectly correct ABI
    tag, so the tag comparison alone certifies nothing."""
    import ctypes

    import eagle.exec as eexec

    lib = ctypes.CDLL(str(bad_bundle.directory / "axpb.so"))
    raw = (ctypes.c_char * (len(eexec.ABI_TAG_V2) + 1)).in_dll(lib, "eagle_abi_tag")
    assert bytes(raw).split(b"\0", 1)[0].decode() == eexec.ABI_TAG_V2


def test_a_correct_artifact_passes_the_same_check(built):
    """Non-vacuity: the SAME reader accepts a correctly-built artifact."""
    import eagle.exec as eexec

    _tag, sizes, _lib = L.host_selfcheck(built["axpb"].directory / "axpb.so")
    assert sizes == eexec.layout_sizes()
