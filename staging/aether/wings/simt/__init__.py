# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether wing for the SIMT form: the source text and the spellings of the warp-form nodes (``aether.wings.simt``).

It holds text and no graph rewrite: the node kinds are the raptor SIMT wing's (:mod:`simt`), which it imports with raptor. The
front end reads it through the manifest (``component.json``, in this package) and the device wrapper argument."""

from __future__ import annotations

import json
from importlib import resources

#: The wing's manifest (data): its spellings arm, its emission modules and the entry list the front end calls.
MANIFEST = json.loads(resources.files(__name__).joinpath("component.json").read_text(encoding="utf-8"))
