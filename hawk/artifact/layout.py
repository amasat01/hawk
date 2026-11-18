# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The positional field order of the layout table a unit exports (``eagle_layout_sizes``); the library writes the table itself.

``layout_sizes_override=`` is a test-only door that states five literal sizes in place of the derived ones, so a
deliberately-wrong artifact can exist whose refusal a row observes.
"""

from __future__ import annotations

#: The array's positional field order, fixed by the mirror PODs'
#: ``static_assert`` layout pins; authoritative over any prose list.
LAYOUT_FIELDS = ("sizeof(GRefMirror)", "sizeof(ScalarHandle)", "sizeof(IntHandle)",
                 "sizeof(aether::idx_t)", "sizeof(PartitionTriple)")
