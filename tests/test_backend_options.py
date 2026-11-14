# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Backend options: ``options={"<backend id>.<key>": value}`` is checked by the driver against the manifests before the unit key,
enters the key by rule, and the old keywords are a deprecation window onto the same options."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import bundle as B
from raptor.errors import RaptorError as HawkError

from _ospreys import backend, host_reads, make

STUB = '''
from dataclasses import replace

from ospreys.cpu.backend import Host

EMITS = []


class Optb(Host):
    id = "optb"
    about = "for the CPU, with one option"

    def finish(self, source, lowered, options):
        EMITS.append(self.own_options(options))
        return replace(source, text=source.text + "// tag=%r\\n" % (self.own_options(options).get("tag"),))


BACKEND = Optb()
'''


@hawk.kernel
def op_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def op_lit(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x + 0.7071 * x + 1.2345


class Stop(Exception):
    pass


@pytest.fixture
def stub(monkeypatch, tmp_path):
    from hawk import registry

    root = tmp_path / "ext"
    root.mkdir()
    make(root, "opts-backend", {"backends": [backend("optb", "opts_plugin:BACKEND", device=False, builds=True, priority=5, wings={"raptor": "cpu"}, reads=host_reads(),
                                                      options=["tag"])]}, modules={"opts_plugin": STUB})
    monkeypatch.syspath_prepend(str(root))
    registry.reset()
    hawk.artifact.reset_unit_memo()
    yield sys.modules.get("opts_plugin") or __import__("opts_plugin")
    monkeypatch.undo()
    sys.modules.pop("opts_plugin", None)
    registry.reset()
    hawk.artifact.reset_unit_memo()


@pytest.fixture
def keys(monkeypatch):
    """Every unit key computed by a build; the build stops right after it (no emit, no compile)."""
    got = []
    real = B._unit_key

    def spy(*a, **k):
        got.append(real(*a, **k))
        raise Stop

    monkeypatch.setattr(B, "_unit_key", spy)
    return got


def _key(keys, **kw):
    n = len(keys)
    with pytest.raises(Stop):
        B.build_bundle([op_affine], "unused", **kw)
    assert len(keys) == n + 1
    return keys[-1]


def test_an_outside_option_changes_the_key_and_the_same_option_hits(stub, tmp_path, cache_dir):
    def build(sub, tag):
        return B.build_bundle([op_affine], tmp_path / sub, targets=("optb",), cache_dir=cache_dir, options={"optb.tag": tag})

    build("a", 1)
    assert len(stub.EMITS) == 1 and len(B._UNIT_MEMO) == 1
    build("b", 2)
    assert [e["tag"] for e in stub.EMITS] == [1, 2] and len(B._UNIT_MEMO) == 2
    assert "tag=2" in (tmp_path / "b" / "op_affine.cpp").read_text()
    build("c", 2)
    assert len(stub.EMITS) == 2 and len(B._UNIT_MEMO) == 2                         # the same option: the memo answers, nothing is emitted


def test_the_key_takes_every_option_of_a_built_backend(stub, keys):
    base = _key(keys, targets=("optb",), cache_dir="c")
    assert _key(keys, targets=("optb",), cache_dir="c", options={"optb.tag": 1}) != base
    assert _key(keys, targets=("optb",), cache_dir="c", options={"optb.tag": 1}) != _key(
        keys, targets=("optb",), cache_dir="c", options={"optb.tag": 2})
    assert _key(keys, targets=("optb",), cache_dir="c", options={"optb.tag": {"b": 1, "a": 2}}) == _key(
        keys, targets=("optb",), cache_dir="c", options={"optb.tag": {"a": 2, "b": 1}})
    # an installed backend this build does not build: accepted, ignored, not hashed
    assert _key(keys, targets=("host",), cache_dir="c", options={"optb.tag": 1}) == _key(keys, targets=("host",), cache_dir="c")


def test_the_key_takes_the_version_of_each_built_backend(stub, keys, monkeypatch):
    from hawk import registry

    base = _key(keys, targets=("optb",), cache_dir="c")
    monkeypatch.setattr(registry, "version", lambda backend_id: "9.9")
    assert _key(keys, targets=("optb",), cache_dir="c") != base


def test_a_bad_option_is_refused_before_the_key_and_any_emit(stub, keys):
    cases = [({"nope.x": 1}, "nope"), ({"optb.other": 1}, "optb.other"), ({"host.zzz": 1}, "host.zzz"), ({"bare": 1}, "bare"),
             ({"optb.tag": object()}, "cannot enter the unit key"), ({"optb.tag": {1, 2}}, "cannot enter the unit key")]
    for options, why in cases:
        for entry in (lambda: B.build_bundle([op_affine], "unused", targets=("optb",), options=options),
                      lambda: B.emit_source(op_affine, targets=("optb",), options=options),
                      lambda: B.build(op_affine, "unused", targets=("optb",), options=options)):
            with pytest.raises(HawkError, match=why):
                entry()
    assert keys == [] and stub.EMITS == [] and len(B._UNIT_MEMO) == 0


def test_a_memo_hit_still_checks_the_option(stub, tmp_path, cache_dir):
    B.build_bundle([op_affine], tmp_path / "a", targets=("optb",), cache_dir=cache_dir, options={"optb.tag": 1})
    with pytest.raises(HawkError, match="optb.other"):
        B.build_bundle([op_affine], tmp_path / "b", targets=("optb",), cache_dir=cache_dir, options={"optb.tag": 1, "optb.other": 1})


OLD = [("literal_table", True, "cuda.literal_table", True, ("cuda",)),
       ("embedded_profile", None, "c.profile", None, ("c",)),
       ("device_arch", "sm_90", "cuda.arch", "sm_90", ("cuda",)),
       ("host_profile", "portable", "host.profile", "portable", ("host",))]


@pytest.mark.parametrize("old, value, new, nvalue, targets", OLD)
def test_an_old_keyword_warns_and_is_its_option(keys, old, value, new, nvalue, targets):
    from hawk.targets import TARGETS

    if targets[0] not in TARGETS:
        pytest.skip(f"{targets[0]} is not installed")
    if old == "embedded_profile":
        from ospreys.c.profile import EmbeddedProfile

        value = nvalue = EmbeddedProfile(max_trips=7)
    with pytest.warns(DeprecationWarning, match=new.replace(".", r"\.")):
        k_old = _key(keys, targets=targets, cache_dir="c", **{old: value})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k_new = _key(keys, targets=targets, cache_dir="c", options={new: nvalue})
    assert k_old == k_new
    assert _key(keys, targets=targets, cache_dir="c") != k_new
    with pytest.raises(HawkError, match="both given"):
        B.build_bundle([op_affine], "unused", targets=targets, **{old: value}, options={new: nvalue})


def test_the_old_keyword_emits_the_same_text_as_its_option():
    from hawk.targets import TARGETS

    if "cuda" not in TARGETS:
        pytest.skip("cuda is not installed")
    with pytest.warns(DeprecationWarning):
        old = B.emit_source(op_lit, targets=("cuda",), device_arch="sm_90", literal_table=True)
    new = B.emit_source(op_lit, targets=("cuda",), options={"cuda.arch": "sm_90", "cuda.literal_table": True})
    plain = B.emit_source(op_lit, targets=("cuda",), options={"cuda.arch": "sm_90"})
    assert old.sources["cuda"].text == new.sources["cuda"].text != plain.sources["cuda"].text


def test_the_driver_names_no_backend_option():
    root = Path(hawk.__file__).parent
    names = ('"cuda.literal_table"', '"c.profile"', "'c.profile'", "'cuda.literal_table'")
    for path in [*(root / "artifact").glob("*.py"), root / "backends" / "__init__.py"]:
        text = path.read_text()
        assert not [n for n in names if n in text], path
    for path in (root / "artifact" / "unit_cache.py", root / "backends" / "__init__.py"):
        text = path.read_text()
        assert not [n for n in ("literal_table", "embedded_profile") if n in text], path
