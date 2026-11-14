# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A wing that cannot be loaded is refused by name, whole: a parent that is not installed, a manifest of another family than the group
it is declared under, a cycle, and two distributions that declare one name in a family."""

from __future__ import annotations

import importlib

import pytest

from hawk import wings
from hawk._errors import HawkError


def _wing(root, monkeypatch, name, *, family="raptor", parent=None, stated=None, group=None, dist=None, kinds=None):
    """A distribution ``dist`` (default ``<family>-<name>``) that declares the wing ``name`` under ``group`` (default ``<family>.wings``)."""
    dist = dist or f"{family}-{name}"
    package = f"zz_{dist.replace('-', '_')}"
    info = root / f"{package}-0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text(f"[{group or family + '.wings'}]\n{name} = {package}\n")
    (root / package).mkdir()
    text = f"MANIFEST = {{'family': {stated or family!r}, 'name': {name!r}, 'parent': {parent!r}}}\n"
    if kinds:
        text += f"from types import SimpleNamespace\nKINDS = SimpleNamespace(kinds={dict.fromkeys(kinds, 1)!r}, classes={{}})\n"
    (root / package / "__init__.py").write_text(text)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()


def test_a_parent_that_is_not_installed_is_refused_by_name(tmp_path, monkeypatch):
    _wing(tmp_path, monkeypatch, "zchild", parent="zmissing")
    with pytest.raises(HawkError, match="raptor wing zchild builds on the raptor wing zmissing: the raptor wing zmissing is not installed: "
                                        "install the distribution raptor-zmissing"):
        wings.wing("raptor", "zchild")


def test_a_wing_whose_manifest_states_another_family_is_refused_by_name(tmp_path, monkeypatch):
    _wing(tmp_path, monkeypatch, "zodd", family="aether", stated="raptor")
    with pytest.raises(HawkError, match="aether wing zodd is declared by aether-zodd under the group aether.wings, but its manifest states the family 'raptor'"):
        wings.wing("aether", "zodd")


def test_a_cycle_of_parents_is_refused_by_name(tmp_path, monkeypatch):
    _wing(tmp_path, monkeypatch, "zone", parent="ztwo")
    _wing(tmp_path, monkeypatch, "ztwo", parent="zone")
    with pytest.raises(HawkError, match="raptor wings form a cycle: zone -> ztwo -> zone"):
        wings.wing("raptor", "zone")


def test_two_distributions_that_declare_one_wing_name_are_refused_by_name(tmp_path, monkeypatch):
    _wing(tmp_path, monkeypatch, "zsame", dist="raptor-zsame")
    _wing(tmp_path, monkeypatch, "zsame", dist="other-zsame")
    with pytest.raises(HawkError, match=r"raptor wing zsame is declared by more than one distribution \(other-zsame, raptor-zsame\)"):
        wings.wing("raptor", "zsame")


def test_a_kind_defined_twice_in_a_chain_is_refused_by_name(tmp_path, monkeypatch):
    from hawk.dialects import KINDS

    _wing(tmp_path, monkeypatch, "zroot", kinds=["zz_twice_kind"])
    _wing(tmp_path, monkeypatch, "zleaf", parent="zroot", kinds=["zz_twice_kind"])
    with pytest.raises(HawkError, match="the kind 'zz_twice_kind' is defined twice in the chain of the raptor wing zleaf: by the raptor wing zroot "
                                        "and by the raptor wing zleaf"):
        wings.wing("raptor", "zleaf")
    assert "zz_twice_kind" not in KINDS.kinds


def test_a_host_build_on_arm64_does_not_require_the_x86_wing(monkeypatch):
    from types import MappingProxyType

    from hawk.targets import Machine, build_spec

    real = wings.entry_points
    monkeypatch.setattr(wings, "entry_points", lambda group: [e for e in real(group) if not (group == "raptor.wings" and e.name == "x86")])
    wings.wing.cache_clear()
    try:
        machine = Machine(isa="arm64", arch="", reserved_names=MappingProxyType({}), facts=MappingProxyType({}))
        build_spec(("host",), machine=machine)
        wings.wing.cache_clear()
        with pytest.raises(HawkError, match="raptor wing x86"):
            build_spec(("host",), machine=Machine(isa="x86_64", arch="", reserved_names=MappingProxyType({}), facts=MappingProxyType({})))
    finally:
        wings.wing.cache_clear()
