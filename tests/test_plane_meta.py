# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Opaque notes on a plane, ``Annotated[T, Meta(...)]``: HAWK writes them under ``plane_meta`` in the
sidecar by plane name and never reads them; eagle's reader hands them back from the loaded kernel."""

from __future__ import annotations

from typing import Annotated

import pathlib

import pytest
from conftest import sidecar_of

import hawk
from hawk import HawkError, Meta, Mutable, Scalar, Vector
from hawk.artifact import build_bundle


@hawk.kernel
def noted(x: Annotated[Vector[3], Meta(unit="km", frame="J2000")], y: Mutable[Scalar]):
    y = x[0] + x[1] + x[2]


@hawk.kernel
def plain(x: Vector[3], y: Mutable[Scalar]):
    y = x[0] + x[1] + x[2]


def test_the_notes_travel_in_the_sidecar_and_back_through_eagle(tmp_path):
    bundle = build_bundle([noted, plain], tmp_path, targets=("host",), host_profile="native")
    meta = sidecar_of(bundle, noted.name)
    assert meta["plane_meta"] == {"x": {"frame": "J2000", "unit": "km"}}
    assert "plane_meta" not in sidecar_of(bundle, plain.name)
    from eagle.host_launch import HostPluginLibrary

    def loaded(kernel):
        return HostPluginLibrary(pathlib.Path(bundle.directory) / f"{kernel.name}.so",
                                 sidecar_of(bundle, kernel.name))

    assert loaded(noted).plane_meta == {"x": {"frame": "J2000", "unit": "km"}}
    assert loaded(plain).plane_meta == {}


def test_a_note_that_is_not_json_is_refused():
    with pytest.raises(HawkError, match="Meta.*JSON"):
        Meta(unit=object())


def test_eagles_reader_refuses_notes_for_a_plane_the_kernel_lacks():
    from eagle.sidecar import read_plane_meta

    meta = {"arg_spec": [["per_sample", "x"]], "plane_meta": {"z": {"a": 1}}}
    with pytest.raises(ValueError, match="plane_meta must map each plane of the arg_spec"):
        read_plane_meta(meta, name="k")
