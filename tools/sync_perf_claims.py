#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the performance numbers in README.md from eagle's committed cards.

Every measured number in the README's Performance section is a claim span::

    <!--c wall card=rk78 dev=p2000 arm=eagle_simulate n=1000000-->13.0 s<!--/c-->

whose text is written by eagle's ``tools/prose_claims.py`` (the one rounding rule
and the span syntax are stated there) from the cards under eagle's
``benchmarks/``. The headline sentence between ``<!-- hero:begin -->`` and
``<!-- hero:end -->`` is eagle's headline: its configured device's card at the
larger N with the smaller N beside it.

hawk's CI has no eagle checkout, so the rendered values are kept in
``docs/perf_claims.json`` (span spec -> text, the headline sentence, the md5 of
each card read); ``tests/test_perf_claims.py`` checks the README against that
file always, and the file against the cards whenever ``RAPTOR_EAGLE_ROOT`` points
at an eagle checkout.

When the headline device has no card this script stops with an error naming the
missing file and writes nothing; ``--without-headline`` regenerates the rest.

Usage::

    python tools/sync_perf_claims.py --eagle-root ../eagle
    python tools/sync_perf_claims.py --eagle-root ../eagle --check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
README = REPO / "README.md"
SNAPSHOT = REPO / "docs" / "perf_claims.json"
HERO = re.compile(r"(<!-- hero:begin -->\n)(.*?)(\n<!-- hero:end -->)", re.DOTALL)
SPAN = re.compile(r"<!--c\s+(?P<spec>[^>]*?)\s*-->(?P<body>.*?)<!--/c-->", re.DOTALL)
SUBJECT = "hawk + eagle (`eagle.simulate`)"


_LOADED = {}


def load_eagle(root: pathlib.Path):
    """Import eagle's claim engine from an eagle checkout (once per root)."""
    path = root / "tools" / "prose_claims.py"
    if path in _LOADED:
        return _LOADED[path]
    if not path.is_file():
        raise SystemExit(f"{path} not found: pass --eagle-root of an eagle checkout")
    spec = importlib.util.spec_from_file_location("eagle_prose_claims", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOADED[path] = module
    return module


def build(root: pathlib.Path, with_headline: bool) -> dict:
    """The snapshot the README must agree with."""
    pcl = load_eagle(root)
    cards = pcl.Cards(root)
    text = README.read_text()
    spans = {}
    for m in SPAN.finditer(text):
        spans[m.group("spec")] = pcl._claim(cards, m.group("spec"), dict(pcl.BASE_CTX))
    used = set()
    for spec in spans:
        sel = dict(pcl.BASE_CTX)
        sel.update(pcl._kv(spec))
        used.add(str(cards.path(sel["card"], sel["dev"]).relative_to(root)))
    hero = None
    if with_headline:
        hero = pcl.render_hero_sentence(pcl.headline_facts(cards), SUBJECT)
        hl = cards.headline
        used.add(str(cards.path("rk4", hl["dev"]).relative_to(root)))
    return {
        "headline": {"dev": pcl.HEADLINE["dev"], "ns": list(pcl.HEADLINE["ns"])},
        "cards": {
            rel: hashlib.md5((root / rel).read_bytes()).hexdigest()
            for rel in sorted(used)
        },
        "spans": spans,
        "hero": hero,
    }


def render_readme(snapshot: dict) -> str:
    """README text with each span's body and the hero sentence from `snapshot`."""
    text = SPAN.sub(
        lambda m: (
            f"<!--c {m.group('spec')}-->{snapshot['spans'][m.group('spec')]}<!--/c-->"
        ),
        README.read_text(),
    )
    if snapshot["hero"] is not None:
        text = HERO.sub(lambda m: m.group(1) + snapshot["hero"] + m.group(3), text)
    return text


def main(argv=None) -> int:
    """Entry point; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--eagle-root", type=pathlib.Path, default=os.environ.get("RAPTOR_EAGLE_ROOT")
    )
    ap.add_argument(
        "--check", action="store_true", help="exit 1 on a diff, write nothing"
    )
    ap.add_argument(
        "--without-headline",
        action="store_true",
        help="regenerate everything except the headline sentence",
    )
    args = ap.parse_args(argv)
    if not args.eagle_root:
        print(
            "pass --eagle-root (or set RAPTOR_EAGLE_ROOT) to an eagle checkout",
            file=sys.stderr,
        )
        return 2
    root = pathlib.Path(args.eagle_root).resolve()
    pcl = load_eagle(root)
    try:
        snapshot = build(root, with_headline=not args.without_headline)
    except pcl.ClaimError as exc:
        print(f"headline card missing, nothing written: {exc}", file=sys.stderr)
        return 2
    if args.without_headline:
        old = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.is_file() else {}
        snapshot["hero"] = old.get("hero")
    snap_text = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n"
    readme = render_readme(snapshot)
    stale = []
    if readme != README.read_text():
        stale.append("README.md")
    if not SNAPSHOT.is_file() or SNAPSHOT.read_text() != snap_text:
        stale.append("docs/perf_claims.json")
    if args.check:
        if stale:
            print("stale (run tools/sync_perf_claims.py): " + ", ".join(stale))
            return 1
        print("README.md's performance numbers are up to date")
        return 0
    README.write_text(readme)
    SNAPSHOT.write_text(snap_text)
    print("wrote " + (", ".join(stale) if stale else "nothing (already up to date)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
