# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Old home of :mod:`hawk.passes.policy`, kept so a released eagle that imports ``hawk.emit.policy`` keeps working.

This name IS that module (one object, one set of scopes), not a copy. Removal: with the first eagle release that
imports ``hawk.passes.policy``."""

import sys

from ..passes import policy as _policy

sys.modules[__name__] = _policy
