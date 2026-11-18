# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A build with no host target links no library: its units are the device artifacts themselves."""

from __future__ import annotations

from hawk.artifact.bundle import _link_set


def test_a_device_only_build_links_no_library(tmp_path):
    """Needs no device compiler: the link step is asked for a build whose targets hold no host target, and answers before any tool."""
    assert (
        _link_set(
            [], [], [], targets=("cuda",), stem="units", exports=None, root=tmp_path
        )
        is None
    )
