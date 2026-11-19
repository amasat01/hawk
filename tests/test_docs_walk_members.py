# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A page of the documentation that reads a member of a kernel's ``walk`` names one the walk has.

The walk is hawk's handle on a lowered body; the members a reader may use are the ones it states, and the kernel's own
``arg_spec`` is where the slot order lives. A page that reads ``walk.<name>`` for a name the walk lacks fails when a reader
runs it, so this check reads every page and notebook for the names, without running one.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from hawk import _ir

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted(p for p in DOCS.rglob("*") if p.suffix in (".md", ".ipynb"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_a_page_reads_only_members_the_walk_has(page):
    named = set(re.findall(r"\.walk\.([A-Za-z_]\w*)", page.read_text(encoding="utf-8")))
    assert not {name for name in named if not hasattr(_ir.Walk, name)}
