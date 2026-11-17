"""The sidecar of a banded device kernel, read from a real build under the stand-in device compiler.

A banded kernel builds for a device only, so no host build reaches this path. The stand-in compiler of ``conftest`` writes the
image from the real device text; the trace, the request, the lowering answer, the sidecar of each unit and the set library
are the real code."""

import pytest

import hawk
from hawk.artifact import build_bundle


@hawk.kernel
def banded_colsum(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x, over="col")


@pytest.fixture
def sidecars(tmp_path, stand_in):
    """``sidecars(mode)``: unit name to the sidecar of the device build of ``banded_colsum`` under ``mode``."""
    def build(mode):
        built = build_bundle([banded_colsum], tmp_path / mode, mode=mode, targets=("cuda",),
                             cache_dir=str(tmp_path / "cache"), options={"cuda.arch": "sm_80"})
        return {artifact.name: artifact.sidecar for artifact in built.artifacts}

    return build


def _partials(sidecars: dict) -> set:
    """The element types of the scratch planes that are not counters, over every unit of the kernel."""
    return {plane["dtype"] for side in sidecars.values() for plane in (side.get("team_planes") or {}).values()
            if plane["dtype"] != "int32"}


def test_a_banded_column_sum_keeps_its_partials_as_16_byte_band_partials(sidecars):
    banded = sidecars("banded")
    assert _partials(banded) == {"band_partial"}
    partial_planes = {name for side in banded.values() for name, plane in (side.get("team_planes") or {}).items()
                      if plane["dtype"] == "band_partial"}
    bound = [side["arg_dtypes"][name] for side in banded.values() for name in partial_planes if name in side["arg_dtypes"]]
    assert bound, "a stage binds the partials' plane as a slot of its own"
    assert set(bound) == {"band_partial"}


def test_the_same_kernel_in_float64_has_no_band_partial(sidecars):
    plain = sidecars("float64")
    assert "band_partial" not in _partials(plain)
    assert all("band_partial" not in side["arg_dtypes"].values() for side in plain.values())
