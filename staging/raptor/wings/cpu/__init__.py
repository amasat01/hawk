# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The cpu raptor wing (``raptor.wings.cpu``): the graph analysis of the host's loop over samples (which loops run interchanged, whether
the lanes of a tile may run side by side). It holds no text. It imports the standard library, raptor and the lowering wing its manifest names."""

from __future__ import annotations

import json
from importlib import resources

#: The wing's manifest (data): its emission modules and the wings it builds on.
MANIFEST = json.loads(resources.files(__name__).joinpath("wing.json").read_text(encoding="utf-8"))
