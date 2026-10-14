# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""README.md's performance numbers are ``tools/sync_perf_claims.py``'s output.

Every measured number in the README's Performance section is a claim span whose
text comes from eagle's committed cards. hawk's CI has no eagle checkout, so the
rendered values live in ``docs/perf_claims.json``: the README must agree with
that file always, and that file (its card md5s, spans and headline) must agree with the cards
whenever ``RAPTOR_EAGLE_ROOT`` points at an eagle checkout; the release gate sets
it, and a set-but-unusable root fails.

RED: edit a number inside a ``<!--c ...-->...<!--/c-->`` span and
``test_readme_spans_match_the_snapshot`` fails; re-run a card without
``python tools/sync_perf_claims.py`` and ``test_snapshot_matches_the_cards``
fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_README = _REPO / "README.md"
_SNAPSHOT = _REPO / "docs" / "perf_claims.json"
_spec = importlib.util.spec_from_file_location(
    "hawk_sync_perf_claims", _REPO / "tools" / "sync_perf_claims.py"
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def _snapshot():
    return json.loads(_SNAPSHOT.read_text())


def test_readme_spans_match_the_snapshot():
    snap = _snapshot()
    text = _README.read_text()
    found = {m.group("spec"): m.group("body") for m in sync.SPAN.finditer(text)}
    assert found, "README.md has no claim spans"
    assert set(found) == set(snap["spans"]), (
        "README claim spans and docs/perf_claims.json name different claims: "
        "run tools/sync_perf_claims.py"
    )
    for spec, body in found.items():
        assert body == snap["spans"][spec], (
            f"README says {body!r} for '{spec}' but the cards give "
            f"{snap['spans'][spec]!r}: run tools/sync_perf_claims.py instead of "
            "editing a number by hand"
        )


def test_readme_headline_matches_the_snapshot():
    snap = _snapshot()
    if snap["hero"] is None:
        pytest.skip(
            f"headline card for device {snap['headline']['dev']!r} is not "
            "committed in eagle yet: run tools/sync_perf_claims.py once it is"
        )
    m = sync.HERO.search(_README.read_text())
    assert m, "README.md has no hero:begin/hero:end block"
    assert m.group(2) == snap["hero"]


def test_no_measured_number_is_typed_by_hand():
    text = _README.read_text()
    section = text[text.index("## Performance") : text.index("## Going deeper")]
    section = sync.SPAN.sub(" ", section)
    section = sync.HERO.sub(" ", section)
    left = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)? ?(?:µs|ms|×|MiB)(?![\w])", section)
    assert not left, f"measured numbers outside a claim span: {left}"


def _eagle_root():
    """The eagle checkout named by RAPTOR_EAGLE_ROOT, or None when it is unset.

    A root that is set but unusable FAILS: the gate was asked to compare against
    the cards and cannot, which must not read as a pass."""
    root = os.environ.get("RAPTOR_EAGLE_ROOT")
    if not root:
        return None
    root = pathlib.Path(root)
    assert (root / "tools" / "prose_claims.py").is_file(), (
        f"RAPTOR_EAGLE_ROOT={root} has no tools/prose_claims.py: not an eagle checkout "
        "new enough for claim spans"
    )
    return root


def test_snapshot_card_md5s_match_the_cards():
    root = _eagle_root()
    if root is None:
        pytest.skip(
            "RAPTOR_EAGLE_ROOT is not set (no eagle checkout to read cards from)"
        )
    stale = []
    for rel, md5 in _snapshot()["cards"].items():
        card = root / rel
        if not card.is_file():
            stale.append(f"{rel}: not in {root} (card deleted or renamed)")
        elif hashlib.md5(card.read_bytes()).hexdigest() != md5:
            stale.append(f"{rel}: md5 differs from docs/perf_claims.json")
    assert not stale, (
        "docs/perf_claims.json is stale against eagle's cards: "
        + "; ".join(stale)
        + ". Run tools/sync_perf_claims.py --eagle-root <eagle>"
    )


def test_snapshot_matches_the_cards():
    root = _eagle_root()
    if root is None:
        pytest.skip(
            "RAPTOR_EAGLE_ROOT is not set (no eagle checkout to read cards from)"
        )
    snap = _snapshot()
    assert snap["hero"] is not None, (
        "docs/perf_claims.json has no headline sentence although the eagle checkout "
        "is given: run tools/sync_perf_claims.py"
    )
    fresh = sync.build(root, with_headline=True)
    assert fresh["cards"] == snap["cards"], (
        "eagle's cards changed: re-run sync_perf_claims.py"
    )
    assert fresh["spans"] == snap["spans"]
    assert fresh["hero"] == snap["hero"]
