# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The two compile guards beside the address-space cap: a translation unit over
the byte budget is refused before any compile, and a compiler subprocess that
outlives its timeout is stopped; each refusal names the limit and its variable."""

from __future__ import annotations

import pytest

from hawk.compile import drivers
from hawk.ir import HawkError


def test_a_source_over_the_budget_is_refused_naming_the_budget(monkeypatch):
    monkeypatch.setenv("HAWK_COMPILE_SOURCE_BUDGET", "100")
    with pytest.raises(HawkError, match=r"HAWK_COMPILE_SOURCE_BUDGET"):
        drivers.check_source_budget("kan", "x" * 101)
    drivers.check_source_budget("kan", "x" * 100)


def test_a_zero_budget_disables_the_check(monkeypatch):
    monkeypatch.setenv("HAWK_COMPILE_SOURCE_BUDGET", "0")
    drivers.check_source_budget("kan", "x" * (drivers.SOURCE_BYTES_BUDGET + 1))


def test_a_compiler_that_outlives_the_timeout_is_stopped(monkeypatch):
    monkeypatch.setenv("HAWK_COMPILE_TIMEOUT", "1")
    with pytest.raises(HawkError, match=r"HAWK_COMPILE_TIMEOUT"):
        drivers.run_compiler(["sleep", "30"], "device compile of 'kan'")


def test_a_bad_budget_is_refused(monkeypatch):
    monkeypatch.setenv("HAWK_COMPILE_SOURCE_BUDGET", "lots")
    with pytest.raises(HawkError, match=r"HAWK_COMPILE_SOURCE_BUDGET"):
        drivers.check_source_budget("kan", "x")
