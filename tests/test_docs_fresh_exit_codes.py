# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""`tools/sync_perf_claims.py --check` says which failure it is, by exit code.

0 fresh, 1 stale, 3 a card the docs cite is missing, 4 the eagle checkout or its
claim engine is missing, 2 usage only. Each case runs a temporary copy of the
generator, README and snapshot against a temporary copy of eagle's cards.
A grep test keeps the workflows from reporting every failure as "out of date".

The card cases need an eagle checkout (``RAPTOR_EAGLE_ROOT``) and skip without one.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parent.parent
if not (_REPO / "tools" / "sync_perf_claims.py").is_file():  # the suite also runs from a copy of tests/ outside the repository
    pytest.skip("needs the repository's tools/sync_perf_claims.py", allow_module_level=True)
_EAGLE = os.environ.get("RAPTOR_EAGLE_ROOT")


def _run(hawk, *args, eagle=None):
    cmd = [sys.executable, str(hawk / "tools" / "sync_perf_claims.py"), "--check", *args]
    if eagle is not None:
        cmd += ["--eagle-root", str(eagle)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=hawk)


@pytest.fixture()
def hawk(tmp_path):
    """A copy of the generator, README and snapshot."""
    root = tmp_path / "hawk"
    (root / "tools").mkdir(parents=True)
    (root / "docs").mkdir()
    shutil.copy2(_REPO / "tools" / "sync_perf_claims.py", root / "tools")
    shutil.copy2(_REPO / "README.md", root)
    shutil.copy2(_REPO / "docs" / "perf_claims.json", root / "docs")
    return root


@pytest.fixture()
def eagle(tmp_path):
    if not _EAGLE or not (pathlib.Path(_EAGLE) / "tools" / "prose_claims.py").is_file():
        pytest.skip("needs RAPTOR_EAGLE_ROOT pointing at an eagle checkout")
    root = tmp_path / "eagle"
    (root / "tools").mkdir(parents=True)
    shutil.copy2(pathlib.Path(_EAGLE) / "tools" / "prose_claims.py", root / "tools")
    shutil.copytree(pathlib.Path(_EAGLE) / "benchmarks", root / "benchmarks",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_fresh_tree_exits_0(hawk, eagle):
    assert _run(hawk, eagle=eagle).returncode == 0


def test_stale_readme_exits_1_and_names_the_file(hawk, eagle):
    readme = hawk / "README.md"
    text = readme.read_text()
    i = text.index("<!--c ")
    j = text.index("-->", i) + 3
    k = text.index("<!--/c-->", j)
    readme.write_text(text[:j] + "hand edit" + text[k:])
    res = _run(hawk, eagle=eagle)
    assert res.returncode == 1
    assert "README.md" in res.stdout


def test_missing_card_exits_3_and_names_the_card(hawk, eagle):
    (eagle / "benchmarks" / "perf_card").joinpath("card_nvidia-h100-80gb-hbm3.json").unlink()
    res = _run(hawk, eagle=eagle)
    assert res.returncode == 3
    assert "benchmarks/perf_card/card_" in res.stderr


def test_eagle_without_claim_engine_exits_4_and_names_the_path(hawk, tmp_path):
    bare = tmp_path / "bare_eagle"
    bare.mkdir()
    res = _run(hawk, eagle=bare)
    assert res.returncode == 4
    assert "prose_claims.py" in res.stderr


def test_no_eagle_root_exits_4(hawk, monkeypatch):
    monkeypatch.delenv("RAPTOR_EAGLE_ROOT", raising=False)
    res = subprocess.run([sys.executable, str(hawk / "tools" / "sync_perf_claims.py"), "--check"],
                         capture_output=True, text=True, cwd=hawk,
                         env={k: v for k, v in os.environ.items() if k != "RAPTOR_EAGLE_ROOT"})
    assert res.returncode == 4
    assert "--eagle-root" in res.stderr


def test_usage_error_exits_2(hawk):
    assert _run(hawk, "--no-such-flag").returncode == 2


def test_workflows_say_out_of_date_only_in_the_stale_branch():
    for wf in sorted((_REPO / ".github" / "workflows").glob("*.yml")):
        assert "out of date" not in wf.read_text(), wf.name
    for line in (_REPO / "tools" / "docs_fresh.sh").read_text().splitlines():
        if "out of date" in line:
            assert line.lstrip().startswith("1)"), line
