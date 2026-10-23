# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target's export door and its traceability.

Rows:

* **files** — :func:`hawk.embedded.export` writes exactly ``<name>.c``,
  ``<name>.h`` and ``<name>.embedded.json``;
* **report** — the report's keys come in the spec's order, its traceability
  stamps (hawk version, walk digest, kernel source digest) agree with the
  header's TRACEABILITY block, and it records the toolchain deviations;
* **digest** — ``Kernel.source_digest`` is the sha256 of the dedented
  source of the traced function, carried through ``hawk.steps`` and
  ``resolve_dispatch``; a hand-built ``Kernel`` reports ``"unavailable"``;
* **determinism** — a re-export is byte-identical, with and without the
  toolchain checks (no timestamps, no temp paths, no durations);
* **checks** — ``check=True`` runs the C compiler and cppcheck and records a
  zero-finding cppcheck block.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import textwrap

import _kernels as K
import pytest

import hawk
from hawk import Index, Int32, Mutable, Param, Scalar, Vector
from hawk.embedded import EmbeddedArtifact, export, render, toolchain
from hawk.math import norm, where
from hawk.trace.kernel import Kernel

#: The report's keys in the spec's order (``int_division_sites`` and
#: ``int_ub_sites`` sit where the verify pass puts them, after
#: ``fp_equality_sites``).
REPORT_KEYS = [
    "kernel", "generator", "hawk_version", "walk_digest", "source_digest",
    "scalar_mode", "profile", "interface", "budgets", "indices", "fp_domain",
    "fp_equality_sites", "int_division_sites", "int_ub_sites", "literal_specials",
    "libm",
    "helpers", "core", "core_version", "deviations", "unsupported_checked",
    "audit", "checks",
]


def _probe(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)

PROBE = hawk.kernel(_probe)


def _sha(fn) -> str:
    return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).encode()).hexdigest()


def _drag_source() -> str:
    """``K.drag``'s decorated definition, as ``inspect.getsource`` returns it
    (a :class:`Kernel` does not hold the traced function)."""
    text = inspect.getsource(K)
    m = re.search(r"^@kernel\ndef drag\(.*?(?=\n\n)", text, re.S | re.M)
    return m.group(0) + "\n"


# --------------------------------------------------------------------------- #
# files and report
# --------------------------------------------------------------------------- #
def test_export_writes_exactly_three_files(tmp_path):
    art = export(K.drag, tmp_path, check=False)
    assert isinstance(art, EmbeddedArtifact)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "drag.c", "drag.embedded.json", "drag.h"]
    assert (art.name, art.directory) == ("drag", tmp_path)
    assert (art.c_path, art.h_path, art.report_path) == (
        tmp_path / "drag.c", tmp_path / "drag.h", tmp_path / "drag.embedded.json")
    assert json.loads(art.report_path.read_text()) == art.report


def test_report_keys_follow_the_spec_order(tmp_path):
    art = export(K.drag, tmp_path, check=False)
    assert list(json.loads(art.report_path.read_text())) == REPORT_KEYS
    r = art.report
    assert r["kernel"] == "drag" and r["generator"] == "hawk.embedded/1"
    assert r["hawk_version"] == hawk.__version__
    assert r["walk_digest"] == K.drag.walk.digest
    assert r["scalar_mode"] == "float64"
    assert r["profile"].startswith("EmbeddedProfile(")
    assert r["audit"] == {"hits": []}
    assert r["checks"] == {"skipped": "not requested"}
    assert [d["rule"] for d in r["deviations"]] == [d["rule"] for d in toolchain.DEVIATIONS]
    assert not re.search(r"\b(19|20)\d\d-\d\d-\d\d", art.report_path.read_text())


@hawk.kernel
def float_equal(x: Scalar, z: Scalar, y: Mutable[Scalar]):
    y = where(x == z, x, z)


def test_float_equality_sites_become_a_deviation_note(tmp_path):
    art = export(float_equal, tmp_path, check=False)
    sites = art.report["fp_equality_sites"]
    assert sites
    (note,) = [d for d in art.report["deviations"] if d["rule"] == "float-equality"]
    assert note == {"rule": "float-equality", "category": "note", "sites": sites,
                    "reason": note["reason"]}


@hawk.kernel
def index_add(a: Index, b: Index, y: Mutable[Index]):
    y = a + b


@hawk.kernel
def int32_add(a: Int32, b: Int32, y: Mutable[Int32]):
    y = a + b


def test_int_ub_sites_become_a_defined_where_cpp_is_ub_note(tmp_path):
    art = export(index_add, tmp_path / "index_add", check=False)
    sites = art.report["int_ub_sites"]
    assert sites
    (note,) = [d for d in art.report["deviations"]
               if d["rule"] == "defined_where_cpp_is_ub"]
    assert note == {"rule": "defined_where_cpp_is_ub", "category": "note",
                    "sites": sites, "reason": note["reason"]}
    for needle in ("hawk_i64_*", "undefined", "wherever the C++ is defined"):
        assert needle in note["reason"]
    # Int32 arithmetic wraps through a helper on the host too: no note
    art = export(int32_add, tmp_path / "int32_add", check=False)
    assert art.report["int_ub_sites"] == []
    assert not [d for d in art.report["deviations"]
                if d["rule"] == "defined_where_cpp_is_ub"]
    # nor for a real-only kernel
    art = export(K.drag, tmp_path / "drag", check=False)
    assert art.report["int_ub_sites"] == []
    assert not [d for d in art.report["deviations"]
                if d["rule"] == "defined_where_cpp_is_ub"]


# --------------------------------------------------------------------------- #
# the kernel source digest
# --------------------------------------------------------------------------- #
def test_source_digest_is_the_sha256_of_the_dedented_source():
    assert PROBE.source_digest == _sha(_probe)
    assert K.drag.source_digest == hashlib.sha256(_drag_source().encode()).hexdigest()


def test_the_digest_rides_through_derived_kernels():
    import test_finish_own_sample as F

    from hawk.trace.steps import steps

    four = steps(F.osc_finish, 4)
    assert F.osc_finish.source_digest is not None
    assert four.source_digest == F.osc_finish.source_digest


def test_a_hand_built_kernel_reports_unavailable(tmp_path):
    hand = Kernel("x", PROBE.sinks)
    assert hand.source_digest is None
    art = export(hand, tmp_path, check=False)
    assert art.report["source_digest"] == "unavailable"
    assert re.search(r"\*   kernel sha256 +unavailable\n", art.h_path.read_text())


def test_the_header_traceability_block_shows_the_stamps(tmp_path):
    art = export(PROBE, tmp_path, check=False)
    header = art.h_path.read_text()
    assert re.search(rf"\*   hawk version +{re.escape(hawk.__version__)}\n", header)
    assert re.search(rf"\*   walk digest +{PROBE.walk.digest}\n", header)
    assert re.search(rf"\*   kernel sha256 +{_sha(_probe)}\n", header)
    assert art.report["source_digest"] == _sha(_probe)
    # render() alone already shows them
    assert render(PROBE).header == header


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def _bytes(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_a_re_export_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    export(K.drag, a, check=False)
    export(K.drag, b, check=False)
    export(K.drag, b, check=False)
    assert _bytes(a) == _bytes(b)
    text = (a / "drag.embedded.json").read_text()
    assert text.endswith("}\n") and json.loads(text)


def test_a_failed_check_leaves_no_stale_report(tmp_path, monkeypatch):
    """A re-export whose checks fail leaves its C for inspection and NO
    report: never an older clean ``.embedded.json`` beside failing C."""
    import test_int_planes as I

    from hawk.ir import HawkError

    assert toolchain.cppcheck() is not None, "install cppcheck or set $HAWK_CPPCHECK"
    export(K.drag, tmp_path, name="u", check=True)
    assert (tmp_path / "u.embedded.json").is_file()

    def refuse(*_args, **_kw):
        raise HawkError("planted toolchain failure")

    monkeypatch.setattr(toolchain, "run_checks", refuse)
    with pytest.raises(HawkError, match="planted toolchain failure"):
        export(I.casts, tmp_path, name="u", check=True)
    assert not (tmp_path / "u.embedded.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u.c", "u.h"]
    assert (tmp_path / "u.c").read_text() == render(I.casts, name="u").text


def test_check_true_runs_cppcheck_clean_and_stays_byte_identical(tmp_path):
    assert toolchain.cppcheck() is not None, "install cppcheck or set $HAWK_CPPCHECK"
    a, b = tmp_path / "a", tmp_path / "b"
    art = export(K.drag, a, check=True)
    export(K.drag, b, check=True)
    cpp = art.report["checks"]["cppcheck"]
    assert cpp["findings"] == [] and cpp["rc"] == 0
    assert [run["std"] for run in art.report["checks"]["cc"]] == ["c99", "c11"]
    assert "seconds" not in art.report["checks"]
    assert "hawk-embedded-" not in (a / "drag.embedded.json").read_text()
    assert _bytes(a) == _bytes(b)


def test_a_bare_import_hawk_resolves_the_embedded_submodule():
    """``import hawk; hawk.embedded.export`` works with no explicit submodule
    import (a subprocess: in-process the attribute is set once any test
    imported ``hawk.embedded``)."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", "import hawk; hawk.embedded.export"],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "AttributeError" not in done.stderr
