"""Run: HOST_LIB=<library> PYTHONPATH=<osprey build>:<shadow> <python> test_host_axpb.py
<texts.jsonl>
Opens the host library through the binding, prints its name and identity, and renders the recorded
unit `axpb` (ladder rung 1).
"""

import hashlib
import json
import os
import sys

import _osprey
import host_plugin as hp

NEEDED = ("02d7b16cedb128ab", 137)


def recorded_axpb(path):
    """The first error-free host record of the unit axpb whose recorded text is rung 1's."""
    with open(path) as handle:
        for line in handle:
            if '"target":"host"' not in line or '"unit":"axpb"' not in line:
                continue
            record = json.loads(line)
            if (
                record.get("seam") != "hawk.backends.Backend.render"
                or record.get("error")
                or record.get("unmirrorable")
            ):
                continue
            text = record["output"]["value"]["v"]["text"]["s"]
            if hashlib.sha256(text.encode()).hexdigest().startswith(NEEDED[0]):
                return record, text
    raise SystemExit("no recorded axpb unit with the rung 1 text")


def main(path):
    """Render the recorded axpb unit through the binding and check it against the record."""
    record, want = recorded_axpb(path)
    inputs = record["inputs"]
    ctx = _osprey.Ctx()
    lib = ctx.backend_open(os.environ["HOST_LIB"])
    print("library", ctx.backend_name(lib), ctx.backend_identity(lib))
    digest = hp.Unit(inputs, None).digest
    used = (record.get("walk_opts_used") or {}).get(digest)
    known = (record.get("walk_opts") or {}).get(digest) or [{}]
    unit = hp.Unit(inputs, used if used is not None else known[0].get("opts"))
    text, body, entry, meta = hp.render(ctx, lib, unit)
    digest_text = hashlib.sha256(text.encode()).hexdigest()
    print(
        "text sha", digest_text[:16], "lines", len(text.splitlines()), "entry", entry, "meta", meta
    )
    assert (
        digest_text.startswith(NEEDED[0]) and len(text.splitlines()) == NEEDED[1] and text == want
    )
    ctx.backend_close(lib)
    print("axpb rendered through the binding: identical to the recorded text")


main(sys.argv[1])
