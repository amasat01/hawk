"""The sidecar of a banded device kernel, rebuilt from its unit records, with the compile stood in.

A banded kernel builds for a device only, so on a machine without a device compiler nothing else in the suite reaches
this path: here the compile plan and the compile itself are replaced, and everything before them (the trace, the
request, the lowering answer, the sidecar made from each unit record) is the real code."""

from types import SimpleNamespace

import pytest

import hawk
from hawk.artifact import build_bundle, bundle, request


@hawk.kernel
def banded_colsum(lane: hawk.Grid["key", "col":64], x: hawk.Scalar, y: hawk.Mutable["key"]):
    y = hawk.sum(x, over="col")


@pytest.fixture
def sidecars(monkeypatch, tmp_path):
    """``sidecars(mode)``: unit name to the sidecar a device build of ``banded_colsum`` drafts, no compiler run."""
    def no_toolchain(target, spec, options):
        raise request.ToolchainAbsent("stood in: no device toolchain")

    def not_compiled(lowered, sources, options, targets):
        result = SimpleNamespace(key="stood-in", hit=False, seconds=0.0, closure=())
        return {t: SimpleNamespace(files=[], entry=(result, "a." + t, "s." + t, "cubin"), facts={}, report_file=None)
                for t in targets}

    drafted = {}
    draft = bundle._draft

    def keep(*args, **kwargs):
        made = draft(*args, **kwargs)
        drafted[made.name] = made.sidecar
        return made

    monkeypatch.setattr(request, "_compile_plan", no_toolchain)
    monkeypatch.setattr(bundle, "_build_targets", not_compiled)
    monkeypatch.setattr(bundle, "_draft", keep)

    def build(mode):
        drafted.clear()
        build_bundle([banded_colsum], tmp_path / mode, mode=mode, targets=("cuda",), host_profile="native")
        return dict(drafted)

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
