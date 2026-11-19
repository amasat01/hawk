# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A chain the host's register file cannot hold in one unit is cut by the library, and the sidecar names the stages it recorded."""

from __future__ import annotations

from test_chain import _stages, fan_in

import hawk


def test_a_host_cut_chain_names_its_stages_from_the_librarys_record(tmp_path):
    """The first stage carries the chain's name and each later one the name and its place in the cut, in launch order."""
    stages = _stages(hawk.chain(*fan_in(10), name="fan"), "float64", "host", tmp_path)
    assert len(stages) >= 2
    assert stages == ["fan", *(f"fan_{place}" for place in range(2, len(stages) + 1))]
