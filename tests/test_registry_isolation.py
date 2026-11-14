# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An installed package never changes or breaks a build that did not ask for it: each row runs a real ``hawk.build``
in a subprocess, with throwaway distributions on the path."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _ospreys import backend, copy_of, host_reads, make

import hawk
from hawk import registry
from hawk.backends import need
from raptor.errors import RaptorError as HawkError

#: The first-party packages sit in the source tree's extensions folder; a wheel carries them beside hawk.
pytestmark = pytest.mark.repo_local

REPO = str(Path(hawk.__file__).resolve().parent.parent)
EXTENSIONS = [str(Path(REPO) / "extensions" / name) for name in ("hawk_cuda", "hawk_c")] + [str(Path(REPO) / "staging")]

HEAD = """
import json, os, sys
import hawk
from hawk import Mutable, Scalar, registry
from hawk.backends import need
from hawk.targets import DEFAULT_TARGETS

@hawk.kernel
def only(a: Scalar, y: Mutable[Scalar]):
    y = a + 1.0

here = os.path.dirname(os.path.abspath(__file__))
def state():
    return {"ids": registry.backend_ids(default=True), "default": tuple(DEFAULT_TARGETS)}
def host_build(tag):
    return sorted(hawk.build([only], os.path.join(here, "out_" + tag), targets=("host",),
                             cache_dir=os.path.join(here, "cache")).artifacts[0].entries)
def emitted():
    return {t: s.text for t, s in hawk.artifact.emit_source(only, device_arch="sm_80").sources.items()}
"""


#: Counts the reads of the installed distributions made for the EXTENSIONS group (variants and components are found through their own groups).
SCAN = """
from importlib import metadata
calls = []
real = metadata.entry_points
metadata.entry_points = lambda **k: (calls.append(k) if k.get("group") == "ospreys" else None) or real(**k)
"""

#: A state in which hawk's own distribution metadata is absent: no first-party OSPREY is declared.
NO_OWN = """
import importlib.metadata as _md
_orig = _md.distribution
def _dist(name):
    if name == "raptor-hawk":
        raise _md.PackageNotFoundError(name)
    return _orig(name)
_md.distribution = _dist
"""


def _run(tmp_path, body: str, paths, pre: str = "", env=None) -> dict:
    """Run ``body`` after :data:`HEAD` as a script in ``tmp_path`` (the script's folder is first on the path);
    its last line of output is a json object."""
    import raptor

    # hawk reads its IR definition from raptor, which an install carries; it goes last on the path
    paths = [*paths, os.path.dirname(os.path.dirname(raptor.__file__))]
    (tmp_path / "probe.py").write_text(pre + HEAD + body)
    done = subprocess.run([sys.executable, str(tmp_path / "probe.py")], capture_output=True, text=True, cwd=tmp_path,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths)), **(env or {})}, check=False,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


ATTEMPT = """
def attempt(fn):
    try:
        return {"ok": fn()}
    except BaseException as e:
        return {"raised": type(e).__name__, "said": str(e)}
"""


def _dist(site, name, entry=None, *, metadata=True, meta=None):
    """A dist-info that may be damaged: no METADATA, odd METADATA, an entry_points.txt of bytes or text."""
    d = site / f"{name.replace('-', '_')}-1.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    if metadata:
        (d / "METADATA").write_text(meta or f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    if entry is not None:
        (d / "entry_points.txt").write_bytes(entry if isinstance(entry, bytes) else entry.encode())


def _explode(root, package: str) -> None:
    (root / package / "__init__.py").write_text("raise SystemExit('imported')\n")


def test_a_distribution_that_reuses_a_first_party_manifest_name_changes_nothing(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "innocent-tool", {"name": "cuda", "version": "9"})
    _explode(site, "innocent_tool")
    out = _run(tmp_path, """
before = state()
cuda = need("cuda").id
host_build("a")
first = emitted()
host_build("b")
second = emitted()
print(json.dumps({"before": before, "after": state(), "cuda": cuda, "same": first == second, "again": need("cuda").id,
                  "targets": sorted(first), "refused": registry.refused()}))
""", [REPO, *EXTENSIONS, site])
    assert out["before"] == out["after"] == {"ids": ["cuda", "host", "c"], "default": ["cuda", "host"]}
    assert out["cuda"] == out["again"] == "cuda" and out["same"] and out["targets"] == ["cuda", "host"]
    assert out["refused"] == {}


def test_an_entry_point_at_a_first_party_manifest_is_refused_and_c_is_unchanged(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "innocent-tool", value="hawk_c:osprey.json")
    out = _run(tmp_path, """
before = state()
host_build("a")
print(json.dumps({"before": before, "after": state(), "c": need("c").id, "refused": registry.refused()}))
""", [REPO, *EXTENSIONS, site])
    assert out["before"] == out["after"] and "c" in out["after"]["ids"] and out["c"] == "c"
    assert list(out["refused"]) == ["innocent-tool"] and "hawk_c" in out["refused"]["innocent-tool"]


@pytest.mark.parametrize("manifest", [{"requires": [["x"]]},
                                      {"backends": [backend(["x"], "wrong_mod:BACKEND")]},
                                      {"backends": [backend("p", "wrong_mod:BACKEND", priority="high")]},
                                      {"backends": [backend("p", "wrong_mod:BACKEND", device="yes")]},
                                      {"backends": [{"name": "p"}]},
                                      {"spellings": [{"writer": 1, "dialect": "d", "ref": "m"}]},
                                      {"backends": "not a list"},
                                      {"version": 3}, {"contract": 1}])
def test_a_manifest_with_a_wrong_type_is_refused_and_a_host_build_succeeds(tmp_path, manifest):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "sloppy-ext", manifest)
    out = _run(tmp_path, """
print(json.dumps({"entries": host_build("a"), "state": state(), "refused": registry.refused()}))
""", [REPO, *EXTENSIONS, site])
    assert out["entries"] == ["host"] and "malformed" in out["refused"]["sloppy-ext"]
    assert out["state"]["ids"] == ["cuda", "host", "c"]


def test_a_stray_folder_named_like_a_first_party_package_never_runs(tmp_path):
    stray = tmp_path / "hawk_cuda"
    stray.mkdir()
    (stray / "osprey.json").write_text(json.dumps(
        {"name": "cuda", "version": "1", "contract": "1",
         "backends": [backend("cuda", "hawk_cuda.backend:BACKEND", device=True)]}))
    (stray / "backend.py").write_text(f"open({str(tmp_path / 'RAN')!r}, 'w').write('x')\nraise SystemExit('ran')\n")
    out = _run(tmp_path, """
print(json.dumps({"state": state(), "refused": registry.refused(),
                  "mods": [m for m in sys.modules if m.startswith("hawk_cuda")]}))
""", [REPO], NO_OWN)
    assert out["state"] == {"ids": [], "default": []} and out["mods"] == []
    assert not (tmp_path / "RAN").exists() and out["refused"] == {}      # no distribution declares it: the folder is never read


def test_on_a_host_only_path_a_third_party_cannot_take_the_cuda_id(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "fakecuda", {"backends": [backend("cuda", "fakecuda_mod:BACKEND", device=True)]},
         modules={"fakecuda_mod": f"open({str(tmp_path / 'RAN')!r}, 'w').write('x')\nraise SystemExit('ran')\n"})
    out = _run(tmp_path, """
try:
    hawk.build([only], os.path.join(here, "out"), targets=("cuda",), cache_dir=os.path.join(here, "cache"))
    said = None
except hawk.HawkError as e:
    said = str(e)
print(json.dumps({"said": said, "mods": [m for m in sys.modules if m.startswith("fakecuda")], "ids": registry.backend_ids()}))
""", [REPO, site], NO_OWN)
    assert out["said"] and "'cuda'" in out["said"] and "fakecuda" in out["said"] and "refused" in out["said"]
    assert out["mods"] == [] and out["ids"] == [] and not (tmp_path / "RAN").exists()



def test_a_build_scans_the_installed_distributions_only_for_a_name_that_is_not_built_in(tmp_path):
    out = _run(tmp_path, """
host_build("a")
cold = len(calls)
host_build("b")
warm = len(calls)
try:
    hawk.build([only], os.path.join(here, "out_c"), targets=("nonesuch",), cache_dir=os.path.join(here, "cache"))
except hawk.HawkError as e:
    said = str(e)
print(json.dumps({"cold": cold, "warm": warm, "unknown": len(calls), "said": said}))
""", [REPO, *EXTENSIONS], SCAN)
    assert (out["cold"], out["warm"], out["unknown"]) == (0, 0, 1) and "unknown target 'nonesuch'" in out["said"]


def test_walking_every_target_row_neither_scans_nor_imports_an_unnamed_osprey(tmp_path):
    """A runtime walks ``TARGETS.items()`` on every deploy to learn which rows it can enter (eagle does): that
    walk lists the default set only, so it reads no installed distribution and imports no backend nobody named;
    the unnamed OSPREY is still reached by its id."""
    site = tmp_path / "site"
    site.mkdir()
    make(site, "walk-ext", {"backends": [backend("walked", "walked_mod:BACKEND")]},
         modules={"walked_mod": "raise SystemExit('an unnamed backend was imported')"})
    out = _run(tmp_path, """
from hawk.targets import TARGETS
rows = [(t, row.entry) for t, row in TARGETS.items()]
walked = len(calls)
try:
    import eagle
    plan = eagle.deploy(only, targets=("host",), cache_dir=os.path.join(here, "cache"))
    deployed = len(calls)
except ImportError:
    deployed = walked
print(json.dumps({"ids": [t for t, _ in rows], "walked": walked, "deployed": deployed, "by_id": "walked" in TARGETS,
                  "after": len(calls)}))
""", [REPO, *EXTENSIONS, str(site)], SCAN)
    assert "walked" not in out["ids"] and "host" in out["ids"]
    assert (out["walked"], out["deployed"]) == (0, 0)
    assert out["by_id"] is True and out["after"] == 1


def _home(tmp_path, monkeypatch):
    """A folder that plays an install: copies of the first-party packages and the dist-info of ``raptor-hawk`` that declares them."""
    root = tmp_path.resolve() / "wheel"
    root.mkdir()
    dist = root / "raptor_hawk-0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: raptor-hawk\nVersion: 0.0\n")
    (dist / "entry_points.txt").write_text("[ospreys]\ncuda = hawk_cuda:osprey.json\nc = hawk_c:osprey.json\n"
                                            "cpu = osprey_cpu:osprey.json\n")
    for name in ("hawk_c", "hawk_cuda"):
        shutil.copytree(Path(REPO) / "extensions" / name / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(REPO) / "staging" / "osprey_cpu", root / "osprey_cpu", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(root))       # first: this is what makes monkeypatch restore the ORIGINAL path
    sys.path[:] = [p for p in sys.path if "extensions" not in p]
    for name in list(sys.modules):
        if name.split(".")[0] in ("hawk_c", "hawk_cuda", "osprey_cpu"):
            monkeypatch.delitem(sys.modules, name)
    registry.reset()
    return root


def test_a_first_party_package_that_cannot_be_read_is_an_error_at_the_default_build(tmp_path, monkeypatch):
    root = _home(tmp_path, monkeypatch)
    (root / "hawk_c" / "osprey.json").write_text('{"name": "c", "contract": "1", "backends": [{"name": ["c"]}]}')
    try:
        assert "malformed" in registry.refused()["c"]
        for _ in range(2):
            with pytest.raises(HawkError, match="first-party OSPREY 'c' is installed but refused"):
                list(hawk.targets.DEFAULT_TARGETS)
    finally:
        registry.reset()


def test_the_wider_scan_keeps_what_the_narrow_one_answered(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    site = tmp_path / "site"
    site.mkdir()
    make(site, "other-ext", {"name": "cuda", "backends": [backend("cuda", "other_mod:BACKEND")]},
         modules={"other_mod": "raise SystemExit('ran')\n"})
    monkeypatch.syspath_prepend(str(site))
    registry.reset()
    try:
        narrow = (registry.backend_ids(default=True), registry.traits("cuda"))
        assert narrow[0] == ("cuda", "host", "c")
        wide = (registry.backend_ids(), registry.backend_ids(default=True), registry.traits("cuda"))
        assert wide == (narrow[0], narrow[0], narrow[1]) and "other_mod" not in sys.modules
        assert "belongs to the built-in set" in registry.refused()["other-ext/backends/'cuda'"]
    finally:
        registry.reset()


def test_a_stray_module_that_shadows_a_first_party_package_is_a_named_error(tmp_path, monkeypatch):
    root = _home(tmp_path, monkeypatch)
    stray = tmp_path / "elsewhere"
    stray.mkdir()
    (stray / "hawk_cuda.py").write_text("raise SystemExit('ran')\n")
    monkeypatch.syspath_prepend(str(stray))
    registry.reset()
    try:
        for call in (lambda: list(hawk.targets.DEFAULT_TARGETS), lambda: need("cuda")):
            with pytest.raises(HawkError, match=r"resolves that name to .*elsewhere.*hawk_cuda\.py"):
                call()
        assert "hawk_cuda" not in sys.modules and root.exists()
    finally:
        registry.reset()


def test_a_genuine_copy_outside_the_trusted_places_is_a_named_error(tmp_path, monkeypatch):
    root = _home(tmp_path, monkeypatch)
    other = tmp_path.resolve() / "copy" / "hawk_c"
    other.parent.mkdir()
    other.mkdir()
    (other / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(other.parent))                           # a genuine copy, first on the path
    registry.reset()
    try:
        for call in (lambda: list(hawk.targets.DEFAULT_TARGETS), lambda: need("c")):
            with pytest.raises(HawkError, match=r"resolves that name to .*copy.*hawk_c"):
                call()
    finally:
        registry.reset()


BUILD_DEFAULT = ATTEMPT + """
print(json.dumps({"build": attempt(lambda: hawk.build([only], os.path.join(here, "out"), cache_dir=os.path.join(here, "cache"))
                                    and "built"),
                  "need": attempt(lambda: need("cuda").id), "host": attempt(lambda: host_build("h"))}))
"""


@pytest.mark.parametrize("stray", ["hawk_cuda.py", "hawk_cuda/__init__.py"])
def test_a_stray_file_named_like_a_first_party_package_is_a_named_error_not_a_silent_loss(tmp_path, stray):
    (tmp_path / stray).parent.mkdir(exist_ok=True)
    (tmp_path / stray).write_text(f"open({str(tmp_path / 'RAN')!r}, 'w').write('x')\n")
    out = _run(tmp_path, BUILD_DEFAULT, [REPO, *EXTENSIONS])
    for call in (out["build"], out["need"]):
        assert call["raised"] == "RaptorError" and "import path resolves that name to" in call["said"]
        assert "hawk_cuda" in call["said"] and "not installed" not in call["said"]
    assert out["host"] == {"ok": ["host"]} and not (tmp_path / "RAN").exists()


def test_a_genuine_first_party_copy_beside_nothing_is_a_named_error(tmp_path):
    import shutil

    shutil.copytree(Path(EXTENSIONS[0]) / "hawk_cuda", tmp_path / "elsewhere" / "hawk_cuda")
    out = _run(tmp_path, BUILD_DEFAULT, [REPO, tmp_path / "elsewhere"])
    for call in (out["build"], out["need"]):
        assert call["raised"] == "RaptorError" and "import path resolves that name to" in call["said"] and "elsewhere" in call["said"]
    assert out["host"] == {"ok": ["host"]}


DAMAGED = {"no metadata": dict(entry="[ospreys]\nbrk = brkpkg:osprey.json\n", metadata=False),
           "bad bytes": dict(entry=b"\xff\xfe[ospreys]\n\xff = x:y\n"),
           "a line without equals": dict(entry="[ospreys]\njust a line\n"),
           "no name": dict(entry="[ospreys]\nnn = nnpkg:osprey.json\n", meta="Metadata-Version: 2.1\nVersion: 1\n")}


@pytest.mark.parametrize("kind", DAMAGED)
def test_a_damaged_distribution_refuses_itself_and_blocks_nothing_else(tmp_path, kind):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "good-ext", {"backends": [backend("p", "good_mod:BACKEND", wings={"raptor": "cpu"}, reads=host_reads())]}, modules={"good_mod": copy_of("osprey_cpu.backend", "p")})
    _dist(site, "damaged", **DAMAGED[kind])
    out = _run(tmp_path, ATTEMPT + """
def build(t):
    return sorted(hawk.build([only], os.path.join(here, "o" + t), targets=(t,), cache_dir=os.path.join(here, "cache")).artifacts[0].entries)
print(json.dumps({"p": attempt(lambda: build("p")), "typo": attempt(lambda: build("hots")), "refused": registry.refused(),
                  "again": attempt(lambda: registry.refused() and "ok")}))
""", [REPO, site], env={"HAWK_OSPREYS": "good-ext"})
    assert out["p"] == {"ok": ["p"]} and out["again"] == {"ok": "ok"}
    assert out["typo"]["raised"] == "RaptorError" and "unknown target 'hots'" in out["typo"]["said"]
    assert any("damaged" in k and "cannot be read" in v for k, v in out["refused"].items()), out["refused"]


def test_hawk_load_of_a_missing_sidecar_imports_no_third_party_module(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "noisy-ext", {"backends": [backend("noisy", "noisy_mod:BACKEND")]},
         modules={"noisy_mod": f"open({str(tmp_path / 'RAN')!r}, 'w').write('x')\nraise RuntimeError('blew up')\n"})
    out = _run(tmp_path, ATTEMPT + """
host_build("a")
print(json.dumps({"load": attempt(lambda: hawk.load(os.path.join(here, "out_a"), "onyl")),
                  "mods": [m for m in sys.modules if m.startswith("noisy")]}))
""", [REPO, site])
    assert out["load"]["raised"] == "FileNotFoundError" and out["mods"] == [] and not (tmp_path / "RAN").exists()


def test_a_third_party_cannot_supply_a_first_party_writers_spellings(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "helpful-ext", {"spellings": [{"writer": "c", "dialect": "simt", "ref": "helpful_sp"}]},
         modules={"helpful_sp": f"open({str(tmp_path / 'RAN')!r}, 'w').write('x')\n"})
    out = _run(tmp_path, """
print(json.dumps({"miss": ("c", "simt") not in registry.spelling_refs(["host"]), "scans": len(calls),
                  "refused": registry.refused()}))
""", [REPO, *EXTENSIONS, site], SCAN)
    assert out["miss"] and out["scans"] == 0 and not (tmp_path / "RAN").exists()
    assert "reserved" in out["refused"]["helpful-ext/spellings/c.simt"]


def test_a_distribution_named_like_a_first_party_identity_is_refused_under_its_own_label(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "Osprey_CUDA", raw="{ not json")
    make(site, "hawk", {"backends": [backend("zz", "zz_mod:BACKEND")]}, modules={"zz_mod": copy_of("osprey_cpu.backend", "zz")})
    make(site, "needs-cuda", {"requires": ["cuda"], "backends": [backend("nc", "nc_mod:BACKEND")]},
         modules={"nc_mod": copy_of("osprey_cpu.backend", "nc")})
    out = _run(tmp_path, ATTEMPT + """
print(json.dumps({"refused": registry.refused(), "ext": sorted(registry.ospreys()), "nc": attempt(lambda: need("nc").id),
                  "zz": attempt(lambda: need("zz").id), "ids": registry.backend_ids(default=True)}))
""", [REPO, *EXTENSIONS, site])
    assert "cuda" not in out["refused"] and "hawk" not in out["refused"]
    assert sum("hawk's own" in v for v in out["refused"].values()) == 2 and "cuda" in out["ext"]
    assert out["nc"] == {"ok": "nc"} and out["zz"]["raised"] == "RaptorError" and out["ids"] == ["cuda", "host", "c"]


def test_a_manifest_that_is_a_pipe_or_huge_is_refused_without_hanging(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    make(site, "pipe-ext")
    (site / "pipe_ext" / "osprey.json").unlink()
    os.mkfifo(site / "pipe_ext" / "osprey.json")
    make(site, "huge-ext", raw='{"name": "x", "contract": "1", "pad": "' + "a" * (1 << 21) + '"}')
    out = _run(tmp_path, 'print(json.dumps({"refused": registry.refused()}))', [REPO, site])
    assert {"pipe-ext", "huge-ext"} <= set(out["refused"]) and all("malformed" in v for v in out["refused"].values())


@pytest.mark.parametrize("value, raises", [("typo-ext", True), ("OFF", True), (" ", False), (",,", False)])
def test_a_name_in_the_environment_that_is_no_osprey_is_an_error_at_the_build(tmp_path, value, raises):
    out = _run(tmp_path, ATTEMPT + """
print(json.dumps({"host": attempt(lambda: host_build("h")), "refused": registry.refused()}))
""", [REPO, *EXTENSIONS], env={"HAWK_OSPREYS": value})
    if raises:
        assert out["host"]["raised"] == "RaptorError" and value.lower() in out["host"]["said"].lower()
        assert value.lower() in out["refused"]
    else:
        assert out["host"] == {"ok": ["host"]} and out["refused"] == {}
