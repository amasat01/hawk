# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target through ``hawk.build`` / ``build_bundle``.

``targets=("c", ...)`` routes through :mod:`hawk.embedded` (render, audit,
toolchain checks, report) and never through the eagle manifest: a ``c``
entry publishes ``<name>.c``, ``<name>.h`` and ``<name>.embedded.json``, the
manifest and the JSON sidecar describe the native targets only, and a
c-only unit writes neither.

Rows:

* **c-only** — exactly the three C files plus the stamp, byte-identical to
  :func:`hawk.embedded.export`'s; ``Bundle.manifest`` is ``None``;
* **stamp** — a second c-only build is a memo hit in-process and a stamp
  hit across a cold memo;
* **host + c** — the manifest, its plugin entry and the sidecar name the
  host target only; ``Entry("c").report`` is set;
* **tables** — ``_EXEC_TARGET`` and ``manifest.EXEC_TARGETS`` never learn
  ``c``;
* **load** — ``hawk.load`` on a c-only directory refuses with a
  :class:`HawkError` naming the ``c`` target and ``hawk.embedded``;
* **key** — the embedded profile is a term of the unit key;
* **message** — the unknown-target message names ``c``.
"""

from __future__ import annotations
from _specs import HOST

import json

import _kernels as K
import pytest

import hawk
from hawk.artifact import (
    STAMP_NAME,
    build,
    build_bundle,
    plugins,
    reset_unit_memo,
    unit_stats,
)
from hawk.artifact import manifest as _manifest
from hawk.artifact.unit_cache import _unit_key
from aether.wings.base.wrapper import scalar_mode
from hawk.ext import DEFAULT_KIND
from raptor.errors import RaptorError as HawkError
from hawk.targets import TARGETS

C_FILES = ["drag.c", "drag.embedded.json", "drag.h"]


@pytest.fixture
def fresh():
    """A publisher with no memory of any unit, before and after the row."""
    reset_unit_memo()
    yield
    reset_unit_memo()


def _c_only(directory, cache_dir, **kw):
    return build_bundle([K.drag], directory, targets=("c",), cache_dir=cache_dir, **kw)


def test_no_target_is_refused(tmp_path, cache_dir, fresh):
    """``targets=()`` must not publish an empty unit now that a draft may
    legitimately carry no native target."""
    with pytest.raises(HawkError, match="no target to build"):
        build_bundle([K.drag], tmp_path / "u", targets=(), cache_dir=cache_dir)
    assert not (tmp_path / "u").exists()


def _kernel_from(path, text: str):
    """Trace ``k`` out of ``text`` written to ``path``, so its source digest
    reads the real file."""
    path.write_text(text)
    ns = {"hawk": hawk}
    exec(compile(text, str(path), "exec"), ns)
    return ns["k"]

