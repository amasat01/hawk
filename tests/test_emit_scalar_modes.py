# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Two built scalar modes, one named but unbuilt slot.

The scalar-mode seam is the same principle as the backend seam: two modes
built, a third slot named rather than silently absent. So the row has three
halves -- ``float64`` and ``float32`` both emit, they DIFFER (a built mode that
were a no-op would be decorative), and ``banded`` raises a HawkError that
names who builds it and who asks for it.

This test previously failed:
  * ``ScalarMode.real_spelling`` for FLOAT32 set to ``"double"`` (a built mode
    made a no-op) --
      AssertionError: : float64 and float32 emitted IDENTICAL text for
      'scale' on cuda -- a built scalar mode that changes nothing is decorative. (The comparison drops the banner comment, which names the mode and
      would make any two modes 'differ' for free.)
    That parenthesis is itself a measured correction: the FIRST form of this row
    compared whole texts and stayed GREEN under the same plant, because the
    banner line carries the mode id.
  * ``_UNBUILT_MODES`` emptied, so ``scalar_mode('banded')`` fell through to the
    generic unknown-mode message --
      AssertionError: : 'banded' is the seam's NAMED third slot -- it is
      never silently absent from the vocabulary; assert 'banded' in {}
      AssertionError: an unknown mode's refusal must still show the seam's full
      vocabulary, third slot included; assert 'banded' in "unknown scalar mode
      'posit8'; the seam declares ('float64', 'float32')"
  both plants were then removed.
"""

from __future__ import annotations

import pytest
from _emitted import cases

from hawk.emit import FLOAT32, FLOAT64, SCALAR_MODES, render_source, scalar_mode
from hawk.emit.backend import _UNBUILT_MODES
from hawk.ir import HawkError


def test_the_seam_declares_exactly_the_two_built_modes():
    assert sorted(SCALAR_MODES) == ["float32", "float64"]
    assert scalar_mode("float64") is FLOAT64
    assert scalar_mode("float32") is FLOAT32


def test_both_built_modes_emit_and_differ():
    for name, sinks, walk in cases():
        for backend_id, backend in _backends():
            f64 = render_source(name, sinks, walk, backend, mode=FLOAT64).text
            f32 = render_source(name, sinks, walk, backend, mode=FLOAT32).text
            assert _substance(f64) != _substance(f32), (
                f"float64 and float32 emitted IDENTICAL text for {name!r} on "
                f"{backend_id} -- a built scalar mode that changes nothing is "
                "decorative. (The comparison drops the banner comment, which "
                "names the mode and would make any two modes 'differ' for free.)"
            )
            assert "using Real = double;" in f64 and "using Real = float;" in f32


def test_banded_is_a_named_slot_that_raises_with_its_wave_and_customer():
    assert "banded" in _UNBUILT_MODES, (
        "'banded' is the seam's NAMED third slot -- it is never silently "
        "absent from the vocabulary"
    )
    with pytest.raises(HawkError) as excinfo:
        scalar_mode("banded")
    message = str(excinfo.value)
    for needle in ("declared but not built yet", "emulated double precision",
                   "aether/banded/"):
        assert needle in message, (
            "selecting 'banded' must say what it is and where it lands; "
            f"it said: {message}"
        )


def test_an_unknown_mode_is_refused_naming_the_whole_vocabulary():
    with pytest.raises(HawkError) as excinfo:
        scalar_mode("posit8")
    assert "banded" in str(excinfo.value), (
        "an unknown mode's refusal must still show the seam's full vocabulary, "
        "third slot included"
    )


def _substance(text: str) -> str:
    """The unit without its banner comment -- the banner NAMES the mode, so a
    whole-text comparison would report any two modes as different for free."""
    return text.split("\n", 2)[2]


def _backends():
    from hawk.emit import BACKENDS

    return sorted(BACKENDS.items())
