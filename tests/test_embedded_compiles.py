# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every C unit the corpus renders compiles strictly and passes the MISRA
checker.

For every corpus kernel (``energy`` is a sum over samples, refused by the C target)
in both scalar modes, :func:`hawk.embedded.check` runs the C compiler under
the strict flag set for ``-std=c99`` and ``-std=c11`` and then cppcheck with
the MISRA C:2012 addon; any diagnostic or finding raises. The rows run under
``EmbeddedProfile(require_cppcheck=True)``, so a machine without cppcheck
FAILS them (a skipped check certifies nothing): install cppcheck or point
``$HAWK_CPPCHECK`` at one. The deviations the checker runs under are
:data:`hawk.embedded.toolchain.DEVIATIONS`, each with its stated reason.
"""

from __future__ import annotations

import _kernels as K
import pytest

from hawk.embedded import EmbeddedProfile, check, render, toolchain

MODES = ("float64", "float32")
CORPUS = [n for n in K.CORPUS if n != "energy"]
STRICT = EmbeddedProfile(require_cppcheck=True)


def test_cppcheck_is_installed():
    assert toolchain.cppcheck() is not None, "install cppcheck or set $HAWK_CPPCHECK"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CORPUS)
def test_the_unit_compiles_and_passes_misra(name, mode):
    checks = check(render(K.CORPUS[name], mode=mode), profile=STRICT)
    assert [run["std"] for run in checks["cc"]] == ["c99", "c11"]
    assert all(run["rc"] == 0 for run in checks["cc"])
    assert "findings" in checks["cppcheck"], "install cppcheck or set $HAWK_CPPCHECK"
    assert checks["cppcheck"]["findings"] == []
    assert checks["cppcheck"]["suppressed"] == [d["rule"] for d in toolchain.DEVIATIONS]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("casts", "scaled"))
def test_a_saturating_real_to_int_cast_passes_misra(name, mode):
    """``hawk_*_from_real`` compares against real-literal bounds (no cast of a
    composite expression, Rule 10.8)."""
    import test_int_planes as I

    source = render(getattr(I, name), mode=mode)
    assert "_from_real(" in source.text
    checks = check(source, profile=STRICT)
    assert "findings" in checks["cppcheck"], "install cppcheck or set $HAWK_CPPCHECK"
    assert checks["cppcheck"]["findings"] == []


def test_every_deviation_states_its_rule_category_and_reason():
    for d in toolchain.DEVIATIONS:
        assert d["rule"].startswith("misra-c2012-")
        assert d["category"] in ("advisory", "required")
        assert len(d["reason"]) > 40
