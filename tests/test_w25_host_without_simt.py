# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A host build and a C export work with the device component and the device extension unimportable, and a broken extension never
breaks a CPU call. Each check runs in a fresh interpreter, so ``sys.modules`` starts clean."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from _extensions import backend, make

BLOCK = textwrap.dedent('''
    import importlib.abc, sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("simt", "hawk_cuda"):
                raise ImportError("blocked for this test: " + name)

    sys.meta_path.insert(0, _Block())
    import importlib.metadata as _md
    _orig = _md.distribution

    def _dist(name):
        if name == "raptor-simt":
            raise _md.PackageNotFoundError(name)
        return _orig(name)

    _md.distribution = _dist
''')

KERNEL = textwrap.dedent('''
    import numpy as np, hawk
    from hawk import Mutable, Param, Scalar

    @hawk.kernel
    def scale(x: Scalar, a: Param, y: Mutable[Scalar]):
        y = a * x
''')


def run(code: str, tmp_path, extra_path=(), **env):
    full = {**os.environ, "PYTHONPATH": os.pathsep.join([*extra_path, *sys.path]), "HAWK_CACHE_DIR": str(tmp_path / "cache"), **env}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full, cwd=str(tmp_path))
    return done


def test_a_host_kernel_builds_and_runs_with_the_device_pieces_unimportable(tmp_path):
    code = BLOCK + KERNEL + textwrap.dedent(f'''
        x = np.arange(1.0, 6.0)
        y = np.zeros(5)
        scale(x, 2.0, y)
        assert np.array_equal(y, 2.0 * x), y
        hawk.build([scale], r"{tmp_path / 'b'}", targets=("host",))
        from hawk import *
        print("OK", sorted(m for m in sys.modules if m.split(".")[0] in ("simt", "hawk_cuda")))
    ''')
    done = run(code, tmp_path)
    assert done.returncode == 0 and done.stdout.startswith("OK []"), done.stderr[-1500:] + done.stdout


def test_a_c_export_works_with_the_device_pieces_unimportable(tmp_path):
    code = BLOCK + KERNEL + textwrap.dedent(f'''
        import os
        out = hawk.embedded.export(scale, r"{tmp_path / 'c'}")
        print("OK", sorted(os.listdir(r"{tmp_path / 'c'}")) != [])
    ''')
    done = run(code, tmp_path)
    assert done.returncode == 0 and done.stdout.startswith("OK True"), done.stderr[-1500:] + done.stdout


def test_a_host_only_build_loads_no_device_piece(tmp_path):
    code = KERNEL + textwrap.dedent(f'''
        hawk.build([scale], r"{tmp_path / 'b'}", targets=("host",))
        print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in ("simt", "hawk_c", "hawk_cuda"))))
    ''').replace("json.dumps", "__import__('json').dumps").replace("sys.modules", "__import__('sys').modules")
    done = run(code, tmp_path)
    assert done.returncode == 0 and json.loads(done.stdout.strip().splitlines()[-1]) == [], done.stderr[-1500:] + done.stdout


@pytest.fixture
def broken(tmp_path):
    root = tmp_path / "ext"
    root.mkdir()
    make(root, "broken-ext", {"backends": [backend("broken", "broken_mod:BACKEND", entry="host")]},
         modules={"broken_mod": "raise RuntimeError('this extension is broken')\n"})
    return str(root)


def test_a_broken_extension_does_not_break_a_host_call(tmp_path, broken):
    code = KERNEL + textwrap.dedent('''
        x = np.arange(1.0, 4.0)
        y = np.zeros(3)
        scale(x, 3.0, y)
        assert np.array_equal(y, 3.0 * x), y
        print("OK")
    ''')
    done = run(code, tmp_path, extra_path=(broken,))     # installed, not named: a CPU call never reads it
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-1500:]


def test_a_broken_extension_is_reported_by_name_when_it_is_the_target(tmp_path, broken):
    code = KERNEL + textwrap.dedent(f'''
        try:
            hawk.build([scale], r"{tmp_path / 'b'}", targets=("broken",))
        except hawk.HawkError as e:
            print("REFUSED", e)
    ''')
    done = run(code, tmp_path, extra_path=(broken,))
    assert "REFUSED" in done.stdout and "broken" in done.stdout and "this extension is broken" in done.stdout, done.stderr[-1500:] + done.stdout


def test_a_broken_default_set_extension_is_skipped_by_an_implicit_call_with_one_warning(tmp_path, broken):
    code = KERNEL + textwrap.dedent('''
        import warnings
        x = np.arange(1.0, 4.0)
        y = np.zeros(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scale(x, 3.0, y)
            scale(x, 3.0, y)
        assert np.array_equal(y, 3.0 * x), y
        said = [str(w.message) for w in caught if "broken" in str(w.message)]
        assert len(said) == 1 and "this extension is broken" in said[0], said
        print("OK")
    ''')
    done = run(code, tmp_path, extra_path=(broken,), HAWK_EXTENSIONS="broken-ext")     # named: it is in the default set
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), done.stderr[-1800:] + done.stdout


def test_a_broken_default_set_extension_is_refused_by_name_when_requested(tmp_path, broken):
    code = KERNEL + textwrap.dedent(f'''
        try:
            hawk.build([scale], r"{tmp_path / 'b'}", targets=("broken",))
        except hawk.HawkError as e:
            print("REFUSED", e)
    ''')
    done = run(code, tmp_path, extra_path=(broken,), HAWK_EXTENSIONS="broken-ext")
    assert "REFUSED" in done.stdout and "broken" in done.stdout and "this extension is broken" in done.stdout, done.stderr[-1500:] + done.stdout


def test_an_implicit_call_on_a_machine_without_a_device_imports_no_device_backend(tmp_path):
    """Listing a default-set row costs no import, and eagle asks for device rows only where a device is present: ``scale(...)`` on a
    machine with no device never imports the device extension or the device component."""
    code = KERNEL + textwrap.dedent('''
        import sys
        from hawk.targets import entry_ids
        ids = entry_ids("device")
        assert not [m for m in sys.modules if m.split(".")[0] in ("simt", "hawk_cuda")], "listing imported a device piece"
        from eagle import _core
        try:
            devices = _core.device_count()
        except Exception:        # eagle's own probe: no device library is no device
            devices = 0
        if devices >= 1:
            print("SKIP a device is present")
            raise SystemExit(0)
        x = np.arange(1.0, 6.0)
        y = np.zeros(5)
        scale(x, 2.0, y)
        assert np.array_equal(y, 2.0 * x), y
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("simt", "hawk_cuda"))
        assert loaded == [], loaded
        print("OK", ids)
    ''')
    done = run(code, tmp_path)
    assert done.returncode == 0 and (done.stdout.strip().startswith("OK") or "SKIP" in done.stdout), done.stderr[-1800:] + done.stdout
