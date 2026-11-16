# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The base aether wing (``aether.wings.base``): the shared writer. It renders a kernel's walk into aether C++ source through the spelling
seam and assembles the translation unit around it; it holds text and no graph rewrite.

It imports the standard library, raptor and the wings its manifest names (:mod:`raptor.wings.base`). The front end reads it through the
manifest (``wing.json``, in this package)."""

from __future__ import annotations

import json
from importlib import resources

#: The wing's manifest (data): its spellings arm and its emission modules.
MANIFEST = json.loads(resources.files(__name__).joinpath("wing.json").read_text(encoding="utf-8"))
