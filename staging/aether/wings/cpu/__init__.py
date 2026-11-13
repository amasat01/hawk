# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The cpu aether wing (``aether.wings.cpu``): the host wrapper and its vector form, the text of a host translation unit around the body
the base wing renders. It imports the standard library, raptor and the wings its manifest names (:mod:`aether.wings.base`,
:mod:`raptor.wings.cpu`, :mod:`lowering`)."""

from __future__ import annotations

import json
from importlib import resources

#: The wing's manifest (data): its emission modules and the wings it builds on.
MANIFEST = json.loads(resources.files(__name__).joinpath("component.json").read_text(encoding="utf-8"))
