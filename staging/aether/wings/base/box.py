# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The entry text of a thread box (:class:`lowering.box.Box`): the prologue line that binds a thread's first owned lane, and the
frame that runs the box body or, for a launch of fewer rows than the box holds, the plain body lane by lane."""

from __future__ import annotations

#: The prologue's name for the first lane a thread owns.
BOX_OWN = "hawk_box_own"


def owned_lane(ctype: str, thread: str, box: Box) -> str:
    """The prologue line binding thread ``thread``'s first owned lane."""
    return (f"        const {ctype} {BOX_OWN} = {thread} + {thread} / {box.stride} * "
            f"{(box.T - 1) * box.stride};")


def frame(box: Box, plain: str, boxed: str) -> str:
    """What follows :func:`owned_lane` in the entry: the ``boxed`` body over the
    thread's box, or, when the launch has fewer than ``T`` rows, the box-free
    ``plain`` body over each owned lane in turn. Both bodies bind ``hawk_i`` and
    ``i``, so both read exactly as the box-free entry's."""
    T, S = box.T, box.stride
    bind = ("const aether::SampleIndex i =\n"
            "{pad}    aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n"
            "{pad}(void)hawk_i;\n{pad}(void)nSamples;")
    pad8, pad12, pad16 = " " * 8, " " * 12, " " * 16
    return "\n".join([
        f"{pad8}if (nSamples < {T * S}) {{",
        f"{pad12}for (long long hawk_k = 0; hawk_k < {T}; ++hawk_k) {{",
        f"{pad16}const long long hawk_i = {BOX_OWN} + hawk_k * {S};",
        f"{pad16}if (hawk_i >= nSamples) break;",
        pad16 + bind.format(pad=pad16),
        plain,
        f"{pad12}}}",
        f"{pad8}}} else {{",
        f"{pad12}const long long hawk_i = {BOX_OWN} + {(T - 1) * S} < nSamples",
        f"{pad12}    ? {BOX_OWN} : nSamples - {T * S} + {BOX_OWN} % {S};",
        pad12 + bind.format(pad=pad12),
        boxed,
        f"{pad8}}}",
    ])
