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



import hawk

C_FILES = ["drag.c", "drag.embedded.json", "drag.h"]








def _kernel_from(path, text: str):
    """Trace ``k`` out of ``text`` written to ``path``, so its source digest
    reads the real file."""
    path.write_text(text)
    ns = {"hawk": hawk}
    exec(compile(text, str(path), "exec"), ns)
    return ns["k"]

