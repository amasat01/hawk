# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target through ``hawk.build`` / ``build_bundle``.

``targets=("c", ...)`` routes through :mod:`hawk.embedded` (render, audit,
toolchain checks, report) and never through the eagle manifest: a ``c``
entry publishes ``<name>.c``, ``<name>.h`` and ``<name>.embedded.json``, the
manifest and the JSON sidecar describe the native targets only, and a
c-only unit writes neither.

Rows:

* **c-only** — exactly the three C files plus the stamp, byte-identical to
  :func:`hawk.embedded.export`'s; ``Bundle.manifest`` is ``None``;
* **stamp** — a second c-only build is a memo hit in-process and a stamp
  hit across a cold memo;
* **host + c** — the manifest, its plugin entry and the sidecar name the
  host target only; ``Entry("c").report`` is set;
* **tables** — ``_EXEC_TARGET`` and ``manifest.EXEC_TARGETS`` never learn
  ``c``;
* **load** — ``hawk.load`` on a c-only directory refuses with a
  :class:`HawkError` naming the ``c`` target and ``hawk.embedded``;
* **key** — the embedded profile is a term of the unit key;
* **message** — the unknown-target message names ``c``.
"""

from __future__ import annotations

import json

import _kernels as K
import pytest

import hawk
from hawk.artifact import (
    STAMP_NAME,
    build,
    build_bundle,
    plugins,
    reset_unit_memo,
    unit_stats,
)
from hawk.artifact import bundle as _bundle
from hawk.artifact import manifest as _manifest
from hawk.artifact.unit_cache import _unit_key
from hawk.embedded import EmbeddedProfile, export
from hawk.emit import scalar_mode
from hawk.ext import DEFAULT_KIND
from hawk.ir import HawkError

C_FILES = ["drag.c", "drag.embedded.json", "drag.h"]


@pytest.fixture
def fresh():
    """A publisher with no memory of any unit, before and after the row."""
    reset_unit_memo()
    yield
    reset_unit_memo()


def _c_only(directory, cache_dir, **kw):
    return build_bundle([K.drag], directory, targets=("c",), cache_dir=cache_dir, **kw)


def test_c_only_bundle_writes_sources_report_and_no_manifest(tmp_path, cache_dir,
                                                             fresh):
    d = tmp_path / "u"
    bundle = _c_only(d, cache_dir)
    assert sorted(p.name for p in d.iterdir()) == sorted([*C_FILES, STAMP_NAME])
    assert not (d / "manifest.json").exists() and not (d / "drag.json").exists()
    assert bundle.manifest is None and bundle.manifest_path is None
    assert sorted(p.name for p in bundle.files()) == C_FILES
    (art,) = bundle.artifacts
    assert list(art.entries) == ["c"]
    entry = art.entries["c"]
    assert (entry.backend, entry.artifact, entry.source, entry.report) == (
        "c", d / "drag.h", d / "drag.c", d / "drag.embedded.json")
    assert art.sidecar_path == d / "drag.embedded.json"
    assert art.sidecar == json.loads(entry.report.read_text())
    assert art.sidecar["checks"].get("cc"), "the bundle door runs the toolchain checks"
    # one code path: the bundle's C unit is byte-identical to export()'s
    exported = export(K.drag, tmp_path / "x")
    for f in C_FILES:
        assert (d / f).read_bytes() == (exported.directory / f).read_bytes(), f


def test_second_build_is_a_stamp_hit(tmp_path, cache_dir, fresh):
    d = tmp_path / "u"
    first = _c_only(d, cache_dir)
    again = _c_only(d, cache_dir)
    assert unit_stats()["memo_hits"] == 1
    assert again.artifacts[0].entries["c"].hit
    reset_unit_memo()
    cold = _c_only(d, cache_dir)
    assert unit_stats()["stamp_hits"] == 1 and unit_stats()["published"] == 0
    assert cold.digest == first.digest and cold.manifest is None


def test_host_and_c_together_keep_the_manifest_host_only(tmp_path, cache_dir, fresh):
    d = tmp_path / "u"
    bundle = build_bundle([K.drag], d, targets=("host", "c"), host_profile="native",
                          cache_dir=cache_dir)
    assert sorted(p.name for p in d.iterdir()) == sorted(
        ["drag.so", "drag.cpp", "drag.json", "manifest.json", *C_FILES, STAMP_NAME])
    assert bundle.manifest_path == d / "manifest.json"
    assert bundle.manifest["exec_targets"] == ["host"]
    assert json.loads((d / "manifest.json").read_text()) == bundle.manifest
    (plugin,) = bundle.manifest["plugins"]
    assert (plugin["artifact"], plugin["sidecar"], plugin["format"]) == (
        "drag.so", "drag.json", "host")
    (art,) = bundle.artifacts
    assert art.sidecar["exec_targets"] == ["host"]
    assert art.sidecar_path == d / "drag.json"
    assert art.entries["c"].report == d / "drag.embedded.json"
    assert art.entries["host"].report is None
    assert sorted(p.name for p in bundle.files()) == sorted(
        ["manifest.json", "drag.json", "drag.so", "drag.cpp", *C_FILES])
    # the plugin carries the host entry point and nothing for the C unit
    plugin_ns = plugins(bundle)["drag"]
    assert plugin_ns.host_entry and not hasattr(plugin_ns, "device_function")


def test_exec_targets_table_never_learns_c():
    assert "c" not in _bundle._EXEC_TARGET
    assert _manifest.EXEC_TARGETS == ("device", "host")
    assert _bundle._TARGET["c"] == (".h", ".c", "c")


def test_c_only_directory_refuses_cleanly_to_load(tmp_path, cache_dir, fresh):
    d = tmp_path / "u"
    _c_only(d, cache_dir)
    with pytest.raises(HawkError) as err:
        hawk.load(d, "drag")
    text = str(err.value)
    assert "'c'" in text and "hawk.embedded" in text


def test_unit_key_differs_by_profile(tmp_path, cache_dir, fresh):
    def key(profile: str) -> str:
        return _unit_key(["drag"], [K.drag.walk.digest], targets=("c",),
                         smode=scalar_mode("float64"), kind=DEFAULT_KIND, defines=(),
                         cache_dir=cache_dir, device_arch="", derivative_blocks=[None],
                         layout_sizes_override=None, embedded_profile=profile)

    assert key(repr(EmbeddedProfile())) != key(repr(EmbeddedProfile(max_trips=10**6)))
    # and through the door: a second profile is a real publish, not a memo hit
    _c_only(tmp_path / "u", cache_dir)
    other = _c_only(tmp_path / "u", cache_dir,
                    embedded_profile=EmbeddedProfile(max_trips=10**6))
    assert unit_stats()["memo_hits"] == 0 and unit_stats()["published"] == 2
    assert "max_trips=1000000" in other.artifacts[0].sidecar["profile"]


def test_unknown_target_message_names_c(tmp_path):
    with pytest.raises(HawkError) as err:
        build(K.drag, tmp_path / "u", targets=("device",))
    assert str(err.value) == (
        "unknown target 'device': the valid targets are 'cuda', 'host' and 'c' "
        "('cuda' builds for the GPU, 'host' for the CPU, 'c' a standalone MISRA C "
        "unit for an embedded build); pass targets=('cuda',), targets=('host',) "
        "or targets=('c',)")


def test_no_target_is_refused(tmp_path, cache_dir, fresh):
    """``targets=()`` must not publish an empty unit now that a draft may
    legitimately carry no native target."""
    with pytest.raises(HawkError, match="no target to build"):
        build_bundle([K.drag], tmp_path / "u", targets=(), cache_dir=cache_dir)
    assert not (tmp_path / "u").exists()


def test_plugins_refuses_a_c_only_bundle(tmp_path, cache_dir, fresh):
    """A c-only unit holds no eagle entry point: asking for plugins is a
    mistake, refused as ``hawk.load`` refuses the directory."""
    bundle = _c_only(tmp_path / "u", cache_dir)
    with pytest.raises(HawkError) as err:
        plugins(bundle)
    text = str(err.value)
    assert "'c'" in text and "hawk.embedded" in text


def _kernel_from(path, text: str):
    """Trace ``k`` out of ``text`` written to ``path``, so its source digest
    reads the real file."""
    path.write_text(text)
    ns = {"hawk": hawk}
    exec(compile(text, str(path), "exec"), ns)
    return ns["k"]


def test_same_walk_different_source_reports_its_own_digest(tmp_path, cache_dir, fresh):
    """Two kernels with one walk and different source text (a comment) are two
    C units: each report carries its own kernel's source digest."""
    body = "    y = a * 2.0\n"
    head = ("import hawk\n@hawk.kernel\n"
            "def k(a: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):\n")
    ka = _kernel_from(tmp_path / "a.py", head + body)
    kb = _kernel_from(tmp_path / "b.py", head + "    # a comment\n" + body)
    assert ka.walk.digest == kb.walk.digest
    assert ka.source_digest != kb.source_digest
    first = build_bundle([ka], tmp_path / "u", targets=("c",), cache_dir=cache_dir)
    second = build_bundle([kb], tmp_path / "u", targets=("c",), cache_dir=cache_dir)
    assert first.artifacts[0].sidecar["source_digest"] == ka.source_digest
    assert second.artifacts[0].sidecar["source_digest"] == kb.source_digest
    on_disk = json.loads((tmp_path / "u" / "k.embedded.json").read_text())
    assert on_disk["source_digest"] == kb.source_digest


def _key_for(targets, cache_dir, **kw) -> str:
    return _unit_key(["drag"], [K.drag.walk.digest], targets=targets,
                     smode=scalar_mode("float64"), kind=DEFAULT_KIND, defines=(),
                     cache_dir=cache_dir, device_arch="", derivative_blocks=[None],
                     layout_sizes_override=None, host_profile="native", **kw)


def test_unit_key_names_the_c_core_version_for_c_only(cache_dir, monkeypatch):
    from aether_dsc import c_core

    c_before = _key_for(("c",), cache_dir, embedded_profile=repr(EmbeddedProfile()))
    host_before = _key_for(("host",), cache_dir)
    monkeypatch.setattr(c_core, "CORE_VERSION", c_core.CORE_VERSION + "-other")
    assert _key_for(("c",), cache_dir,
                    embedded_profile=repr(EmbeddedProfile())) != c_before
    assert _key_for(("host",), cache_dir) == host_before
