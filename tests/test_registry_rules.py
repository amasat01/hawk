# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The registry rules, one row each, against throwaway extensions (a dist-info, a package with its manifest and modules under ``tmp_path``, put on
``sys.path``): built-ins cannot be shadowed, a faulty extension is isolated, an installed third party changes no build
it was not asked for, items load lazily and once."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import _kernels as K
import pytest
from _extensions import backend, copy_of, host_reads, make

import hawk
from hawk import registry
from hawk.backends import need
from raptor.errors import RaptorError as HawkError
from hawk.targets import DEFAULT_TARGETS, TARGETS

REPO = str(Path(hawk.__file__).resolve().parent.parent)
EXTENSIONS = [str(Path(REPO) / "extensions" / name) for name in ("hawk_cuda", "hawk_c")] + [str(Path(REPO) / "staging")]


@pytest.fixture
def ext(tmp_path, monkeypatch):
    """``tmp_path`` on ``sys.path`` with a fresh registry; the modules it holds are forgotten afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.reset()
    yield tmp_path
    registry.reset()
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", "") or "").startswith(str(tmp_path)):
            del sys.modules[name]


def _fast(root, name="fast-ext", module="fast_backend", backend_id="fast", variant=None, **manifest):
    make(root, name, {"backends": [backend(backend_id, f"{module}:BACKEND", **({"variant": variant, "reads": host_reads()} if variant else {}))], **manifest},
         modules={module: copy_of("osprey_cpu.backend", backend_id)})


#: A state in which hawk's own distribution metadata is absent: no first-party extension is declared (a package-only install).
NO_OWN_DIST = ("import importlib.metadata as _md\n_orig = _md.distribution\n"
               "def _dist(name):\n    if name == 'raptor-hawk':\n        raise _md.PackageNotFoundError(name)\n    return _orig(name)\n"
               "_md.distribution = _dist\n")


def _run(code: str, paths, env=None, own: bool = True) -> str:
    import os

    import raptor

    code = code if own else NO_OWN_DIST + code

    # hawk reads its IR definition from raptor, which a package-only install carries
    script = f"import sys; sys.path[:0] = {[*paths, os.path.dirname(os.path.dirname(raptor.__file__))]!r}\n" + code
    done = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_a_dist_info_cannot_shadow_a_built_in(ext):
    host = need("host")
    make(ext, "evil", {"backends": [backend("host", "evil_host:BACKEND")]},
         modules={"evil_host": copy_of("osprey_cpu.backend", "host", "BACKEND.about = 'SHADOW'\n")})
    registry.reset()
    assert need("host").about == "for the CPU" and type(need("host")) is type(host)
    assert any("belongs to the built-in set" in why for key, why in registry.refused().items() if key.startswith("evil/"))
    assert "evil_host" not in sys.modules


@pytest.mark.repo_local  # builds the state without the extension packages from the source tree's layout; a wheel carries them beside hawk
def test_a_package_only_install_builds_a_host_kernel(tmp_path):
    script = tmp_path / "only_pkg.py"
    script.write_text("import sys, hawk\n"
                      "from hawk import Mutable, Scalar, registry\n"
                      "@hawk.kernel\n"
                      "def only(a: Scalar, y: Mutable[Scalar]):\n"
                      "    y = a + 1.0\n"
                      "assert registry.backend_ids() == (), registry.backend_ids()\n"
                      "try:\n"
                      f"    hawk.build([only], {str(tmp_path / 'out')!r}, cache_dir={str(tmp_path / 'cache')!r})\n"
                      "except hawk.HawkError as e:\n"
                      "    print('refused:', e)\n")
    out = _run(f"import runpy; runpy.run_path({str(script)!r}, run_name='__main__')", [REPO], own=False)
    assert out.startswith("refused:") and "osprey-cpu" in out, out      # the host backend is the cpu OSPREY: absent, refused by name


def test_a_malformed_manifest_is_isolated(ext):
    make(ext, "broken", raw="{not json")
    before = tuple(DEFAULT_TARGETS)
    assert "malformed" in registry.refused()["broken"]
    assert need("host").id == "host" and tuple(DEFAULT_TARGETS) == before


def test_a_wrong_contract_is_refused_and_an_error_when_named(ext):
    make(ext, "future", {"contract": "9", "backends": [backend("later", "later_mod:BACKEND")]})
    assert "contract '9'" in registry.refused()["future"]
    assert need("host").id == "host"
    with pytest.raises(HawkError, match=r"'later'.*future.*contract '9'"):
        need("later")


def test_a_missing_requirement_is_refused(ext):
    _fast(ext, requires=["not-installed"])
    assert "requires 'not-installed'" in registry.refused()["fast-ext"]
    with pytest.raises(HawkError, match="fast.*not-installed"):
        need("fast")


def test_a_ref_that_fails_to_import_is_refused_when_named(ext):
    make(ext, "dangling", {"backends": [backend("dangling", "no_such_module_xyz:BACKEND")]})
    assert "dangling" in registry.backend_ids() and not registry.refused()
    with pytest.raises(HawkError, match=r"extension 'dangling'.*cannot be loaded"):
        need("dangling")
    assert "dangling/backends/'dangling'" in registry.refused() and need("host").id == "host"


def test_a_duplicate_third_party_id_refuses_both_and_errors_only_when_named(ext):
    _fast(ext, "one-ext", "one_backend", "twin")
    _fast(ext, "two-ext", "two_backend", "twin")
    registry.reset()
    assert need("host").id == "host"
    assert {"one-ext/backends/'twin'", "two-ext/backends/'twin'"} <= set(registry.refused())
    assert "twin" not in registry.backend_ids()
    with pytest.raises(HawkError, match=r"'twin'.*declared more than once.*'one-ext', 'two-ext'"):
        need("twin")


def test_an_unnamed_build_ignores_an_installed_third_party(ext):
    before = tuple(DEFAULT_TARGETS)
    _fast(ext)
    registry.reset()
    assert "fast" in TARGETS and "fast" not in registry.backend_ids(default=True)
    assert tuple(DEFAULT_TARGETS) == before


def test_an_extension_listed_in_the_environment_joins_the_default_set(ext, monkeypatch):
    _fast(ext)
    monkeypatch.setenv("HAWK_EXTENSIONS", "fast-ext")
    registry.reset()
    assert "fast" in DEFAULT_TARGETS
    monkeypatch.setenv("HAWK_EXTENSIONS", "off")
    registry.reset()
    assert tuple(DEFAULT_TARGETS) == ()


def test_a_third_party_without_a_variant_is_refused_by_name_where_a_threshold_is_read(ext, tmp_path, cache_dir):
    _fast(ext)
    registry.reset()
    with pytest.raises(HawkError, match="form.rolled_fold_min"):
        hawk.build([K.drag], tmp_path / "u", targets=("fast",), cache_dir=cache_dir)


def test_a_named_third_party_is_used(ext, tmp_path, cache_dir):
    _fast(ext, variant={"x86_64": "x86", "arm64": "arm"})
    registry.reset()
    bundle = hawk.build([K.drag], tmp_path / "u", targets=("fast",), cache_dir=cache_dir)
    assert set(bundle.artifacts[0].entries) == {"fast"} and need("fast").id == "fast"


def test_an_unknown_or_refused_target_is_a_hawk_error_naming_it(ext, tmp_path, cache_dir):
    make(ext, "future", {"contract": "9", "backends": [backend("later", "later_mod:BACKEND")]})
    registry.reset()
    with pytest.raises(HawkError, match="'later' is refused"):
        hawk.build([K.drag], tmp_path / "u", targets=("later",), cache_dir=cache_dir)
    with pytest.raises(HawkError, match="unknown target 'nope'"):
        hawk.build([K.drag], tmp_path / "v", targets=("nope",), cache_dir=cache_dir)
    with pytest.raises(HawkError, match="no backend 'nope'"):
        need("nope")


def test_listing_and_need_host_import_no_extension_module():
    out = _run("from hawk import registry\n"
               "from hawk.backends import need\n"
               "registry.backend_ids(); registry.traits('cuda'); registry.refused()\n"
               "need('host')\n"
               "print(sorted(m for m in sys.modules if m.split('.')[0] in ('hawk_c', 'hawk_cuda')))\n"
               "need('c')\n"
               "print(sorted(m for m in sys.modules if m.split('.')[0] == 'hawk_c')[:1])\n", [REPO, *EXTENSIONS])
    assert out.splitlines() == ["[]", "['hawk_c']"]


@pytest.mark.repo_local  # builds the state without the extension packages from the source tree's layout; a wheel carries them beside hawk
def test_first_party_extensions_are_found_through_hawks_own_distribution():
    out = _run("from hawk import registry\n"
               "print(registry.backend_ids(default=True), registry.refused(), sorted(registry.extensions()))\n"
               "print([m for m in sys.modules if m.split('.')[0] in ('hawk_c', 'hawk_cuda')])\n", [REPO, *EXTENSIONS])
    assert out.splitlines() == ["('cuda', 'host', 'c') {} ['hawk-c', 'hawk-cuda', 'osprey-cpu']", "[]"]
    # no raptor-hawk metadata, no first-party extension: there is no search beside hawk's folder
    assert _run("from hawk import registry\nprint(registry.backend_ids())", [REPO, *EXTENSIONS], own=False).strip() == "()"
    assert _run("from hawk import registry\nprint(registry.backend_ids())", [REPO, *EXTENSIONS],
                {"HAWK_EXTENSIONS": "off"}).strip() == "()"


def test_a_third_party_cannot_claim_a_first_party_id(ext, monkeypatch):
    for path in EXTENSIONS:
        monkeypatch.syspath_prepend(path)
    _fast(ext, "fake-cuda", "fake_cuda", "cuda")
    registry.reset()
    assert any(key.startswith("fake-cuda/") and "belongs to the built-in set" in why for key, why in registry.refused().items())
    assert registry.traits("cuda")["about"] == "builds for the GPU" and "cuda" in registry.backend_ids(default=True)


@pytest.mark.parametrize("value, word", [("extension.json", "not '<top package>:<file>'"),
                                         ("no_such_pkg_xyz:extension.json", "no package"),
                                         ("odd_ext:nothing.json", "nothing.json")])
def test_a_bad_entry_point_value_is_refused_with_a_sentence(ext, value, word):
    make(ext, "odd-ext", {"backends": [backend("odd", "odd_mod:BACKEND")]}, value=value)
    registry.reset()
    assert "malformed" in registry.refused()["odd-ext"] and word in registry.refused()["odd-ext"]
    assert "odd" not in registry.backend_ids() and need("host").id == "host"


@pytest.mark.repo_local  # builds the state without the extension packages from the source tree's layout; a wheel carries them beside hawk
def test_a_third_party_manifest_is_read_without_importing_its_package(tmp_path):
    _fast(tmp_path, "lazy-ext", "lazy_backend", "lazy")
    (tmp_path / "lazy_ext" / "__init__.py").write_text("raise RuntimeError('imported')\n")
    out = _run("from hawk import registry\n"
               "print(registry.backend_ids(), registry.refused(), 'lazy_ext' in sys.modules)\n", [REPO, str(tmp_path)], own=False)
    assert out.strip() == "('lazy',) {} False"


def test_the_order_does_not_depend_on_sys_path(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _fast(a, "a-ext", "a_backend", "aa")
    _fast(b, "b-ext", "b_backend", "aa")
    _fast(b, "c-ext", "c_backend", "cc")
    orders = []
    for paths in ((a, b), (b, a)):
        monkeypatch.setattr(sys, "path", [*map(str, paths), *sys.path])
        registry.reset()
        orders.append((registry.backend_ids(), tuple(DEFAULT_TARGETS), sorted(registry.refused())))
        monkeypatch.undo()
    registry.reset()
    assert orders[0] == orders[1] and "cc" in orders[0][0] and "aa" not in orders[0][0]


def test_a_builds_spellings_are_those_its_backends_name_not_what_else_is_installed(ext):
    from hawk.targets import build_spec

    before = build_spec(("host",)).spellings
    assert before and registry.spelling_refs(["host"]).items() <= before.items()
    arm = {"spellings": [{"writer": "w_out", "dialect": "d_out", "ref": "w_out_spells"}]}
    make(ext, "writer-ext", arm, modules={"w_out_spells": "SPELLINGS = {('d_out', 'x'): lambda w, r: 1}\n"})
    registry.reset()
    assert build_spec(("host",)).spellings == before and "w_out_spells" not in sys.modules      # an unrelated extension adds nothing
    _fast(ext, spellings=[{"writer": "w_out", "dialect": "d_own", "ref": "w_own_spells"}])
    registry.reset()
    assert build_spec(("host", "fast")).spellings == {**before, ("w_out", "d_own"): "w_own_spells"}     # its own backend's arm does
    assert "w_own_spells" not in sys.modules


def test_eight_threads_load_an_item_once(ext):
    make(ext, "once-ext", {"backends": [backend("once", "once_backend:BACKEND")]},
         modules={"once_backend": "import builtins, time\nbuiltins._once_loads = getattr(builtins, '_once_loads', 0) + 1\n"
                  "time.sleep(0.1)\n" + copy_of("osprey_cpu.backend", "once")})
    registry.reset()
    import builtins

    builtins._once_loads = 0
    gate, got = threading.Barrier(8), []

    def ask():
        gate.wait()
        got.append(need("once"))

    threads = [threading.Thread(target=ask) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert builtins._once_loads == 1 and len({id(b) for b in got}) == 1 and len(got) == 8
    del builtins._once_loads


def test_an_extension_is_its_distribution_not_the_name_its_manifest_declares(ext):
    for dist, module, backend_id in (("one-ext", "one_backend", "aa"), ("two-ext", "two_backend", "bb")):
        make(ext, dist, {"name": "label", "backends": [backend(backend_id, f"{module}:BACKEND")]},
             modules={module: copy_of("osprey_cpu.backend", backend_id)})
    registry.reset()
    assert {"aa", "bb"} <= set(registry.backend_ids()) and not registry.refused()
    assert {"one-ext", "two-ext"} <= set(registry.extensions())
    dist = ext / "one_ext-0.0.dist-info" / "entry_points.txt"
    dist.write_text(dist.read_text() + "again = one_ext:extension.json\n")
    registry.reset()
    assert "aa" in registry.backend_ids() and not registry.refused()          # two entry points, one package: one extension
    dist.write_text(dist.read_text() + "other = two_ext:extension.json\n")
    registry.reset()
    assert "aa" not in registry.backend_ids() and "more than one entry point" in registry.refused()["one-ext"]


def _own(root, package, editable):
    """A distribution of the first-party kind under ``root/site``; ``editable``: its package lives in ``root/project`` and
    ``direct_url.json`` records that folder, else beside the metadata."""
    import json
    from importlib import metadata

    site = root / "site"
    info = site / "own_dist-0.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: own-dist\nVersion: 0.0\n")
    base = root / "project" if editable else site
    (base / package).mkdir(parents=True)
    (base / package / "__init__.py").write_text("")
    if editable:
        (info / "direct_url.json").write_text(json.dumps({"url": (root / "project").as_uri(), "dir_info": {"editable": True}}))
    return metadata.PathDistribution(info)


def test_an_editable_distribution_may_keep_its_package_in_its_project_folder(tmp_path, monkeypatch):
    dist = _own(tmp_path, "pc_editable_a", editable=True)
    monkeypatch.syspath_prepend(str(tmp_path / "project"))
    home, problem = registry._locate("pc_editable_a", dist)
    assert problem is None and home == (tmp_path / "project" / "pc_editable_a").resolve()


def test_a_same_named_package_earlier_on_the_path_is_refused_by_name_for_an_editable_distribution(tmp_path, monkeypatch):
    dist = _own(tmp_path, "pc_editable_b", editable=True)
    (tmp_path / "stray" / "pc_editable_b").mkdir(parents=True)
    (tmp_path / "stray" / "pc_editable_b" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "project"))
    monkeypatch.syspath_prepend(str(tmp_path / "stray"))
    home, problem = registry._locate("pc_editable_b", dist)
    assert home is None and "'pc_editable_b'" in problem
    assert str(tmp_path / "project" / "pc_editable_b") in problem and str(tmp_path / "stray" / "pc_editable_b") in problem


def test_a_normally_installed_distribution_is_accepted_beside_its_metadata(tmp_path, monkeypatch):
    dist = _own(tmp_path, "pc_normal", editable=False)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    home, problem = registry._locate("pc_normal", dist)
    assert problem is None and home == (tmp_path / "site" / "pc_normal").resolve()
