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




OLD = [("literal_table", True, "cuda.literal_table", True, ("cuda",)),
       ("embedded_profile", None, "c.profile", None, ("c",)),
       ("device_arch", "sm_90", "cuda.arch", "sm_90", ("cuda",)),
       ("host_profile", "portable", "host.profile", "portable", ("host",))]




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
