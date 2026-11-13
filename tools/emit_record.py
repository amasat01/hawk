# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: record every emitted translation unit and bundle blob.

Use: EMIT_RECORD_DIR=<dir> PYTHONPATH=<tools> pytest -p emit_record ...
Writes <dir>/rec_<pid>.json: {test file: [[kind, sha256, tag], ...]} and the
texts once each, gzipped, under <dir>/blobs/<sha>.gz. Compare two runs with
emit_compare.py. kind "src" = every Source the shared writer builds;
kind "blob" = every text hawk.artifact.bundle turns into bundle bytes.
"""
import gzip
import hashlib
import json
import os

_OUT = os.environ.get("EMIT_RECORD_DIR")
_rows: dict = {}
_now = ["<collect>"]


def _note(kind: str, text: str, tag: str) -> None:
    sha = hashlib.sha256(text.encode()).hexdigest()
    _rows.setdefault(_now[0], []).append([kind, sha, tag])
    path = os.path.join(_OUT, "blobs", sha + ".gz")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}"
        with gzip.open(tmp, "wt") as fh:
            fh.write(text)
        os.replace(tmp, path)


def pytest_configure(config):
    if not _OUT:
        return
    os.makedirs(os.path.join(_OUT, "blobs"), exist_ok=True)
    from hawk.artifact import bundle
    try:
        from aether.wings.base import wrapper
    except ImportError:          # a tree from before the writer left hawk
        from hawk.emit import wrapper

    init = wrapper.Source.__init__

    def source_init(self, *a, **k):
        init(self, *a, **k)
        _note("src", self.text, f"{self.target}/{self.mode}/{self.entry}")

    wrapper.Source.__init__ = source_init
    blob = bundle._blob

    def blob_rec(text):
        _note("blob", text, "")
        return blob(text)

    bundle._blob = blob_rec

    # Every compiler command hawk runs, with machine- and run-specific paths replaced, so a change of flags or of
    # what gets compiled is visible even when the emitted text is the same.
    import re
    import tempfile
    from hawk.compile import drivers
    run = drivers.run_compiler
    subs = [(re.escape(os.environ.get("HAWK_CACHE_DIR", "\x00")), "<CACHE>"), (re.escape(os.getcwd()), "<TREE>"),
            (re.escape(os.path.realpath(tempfile.gettempdir())) + r"/[^ ]*", "<TMP>"),
            (r"/(?:private/)?(?:var/folders|tmp)/[^ ]*", "<TMP>"), (r"[0-9a-f]{16,64}", "<HEX>")]

    def run_rec(argv, what, *more, **named):
        line = " ".join(str(a) for a in argv)
        for pat, to in subs:
            line = re.sub(pat, to, line)
        _note("argv", line, what if isinstance(what, str) else "")
        return run(argv, what, *more, **named)

    drivers.run_compiler = run_rec


def pytest_runtest_setup(item):
    _now[0] = item.nodeid.split("::")[0]


_outcomes: dict = {}


_reasons: dict = {}
_details: dict = {}


def _plain(text: str) -> str:
    import re

    text = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", text)
    text = re.sub(r"(?:/[\w.+@-]+){2,}", "<PATH>", text)
    text = re.sub(r"[0-9a-f]{12,64}", "<HEX>", text)
    return re.sub(r"\d+\.\d+\.\d+\.dev\d+\+g[0-9a-f]+(?:\.d\d{8})?", "<VERSION>", text)


def pytest_runtest_logreport(report):
    if report.outcome != "passed" or report.when == "call":
        if _outcomes.get(report.nodeid, "passed") == "passed":
            _outcomes[report.nodeid] = report.outcome
    if report.outcome == "failed" and report.nodeid not in _reasons:
        crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
        text = (crash.message if crash is not None else str(report.longrepr)).strip().splitlines()[0] if report.longrepr else ""
        _reasons[report.nodeid] = text.split(":")[0][:120]
        # The class alone hides a changed sentence or a different failing line: keep the first line of the message
        # (paths, addresses and long hex masked) and the file:line where it was raised.
        where = f"{os.path.basename(str(crash.path))}:{crash.lineno}" if crash is not None else ""
        _details[report.nodeid] = [_plain(text)[:300], where]


def pytest_collectreport(report):
    # A test file that fails to import runs no test and so has no outcome row: record it, or it is invisible.
    if report.failed:
        _outcomes[report.nodeid + "::<collection>"] = "collect-error"
        _reasons[report.nodeid + "::<collection>"] = str(report.longrepr).strip().splitlines()[-1].split(":")[0][:120]


def pytest_sessionfinish(session):
    if _OUT and hasattr(session.config, "workerinput") or (_OUT and not session.config.getoption("numprocesses", 0)):
        with open(os.path.join(_OUT, f"rec_{os.getpid()}.json"), "w") as fh:
            json.dump(_rows, fh)
        with open(os.path.join(_OUT, f"out_{os.getpid()}.json"), "w") as fh:
            json.dump(_outcomes, fh)
        with open(os.path.join(_OUT, f"reasons_{os.getpid()}.json"), "w") as fh:
            json.dump(_reasons, fh)
        with open(os.path.join(_OUT, f"details_{os.getpid()}.json"), "w") as fh:
            json.dump(_details, fh)
