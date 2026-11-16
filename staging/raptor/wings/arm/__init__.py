# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The arm raptor wing (``raptor.wings.arm``): a facts-only wing, child of the cpu wing. It holds no text and no code."""

from __future__ import annotations

import json
from importlib import resources

#: The wing's manifest (data): its family, its name and its parent.
MANIFEST = json.loads(resources.files(__name__).joinpath("wing.json").read_text(encoding="utf-8"))
