# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""One animation shared by the domain pages: curves that grow with time and stop at a level.

Used by pages that run a batch of lanes until each one reaches a level (an ignition temperature, a landing height).
The curves come from ``times[level, lane]``, the time each lane first passed each level; a lane's line ends at a dot
on the final level, labelled with its name. One GIF is written for a light page and one for a dark page.
"""
import numpy as np

from anim import frame_of, write_anim
from themed_figure import DARK, LIGHT


def stopping_traces(times, levels, labels, stem, *, xlabel="time (ms)", ylabel="", stop_text="", count_text="still going",
                    key_text="", time_scale=1e3, n_frames=54, hold=6):
    """Write ``<stem>_light.gif`` and ``<stem>_dark.gif``; return ``{file name: size in bytes}``.

    ``times`` is ``(len(levels), len(labels))`` in seconds, increasing down each column; ``levels`` are the y values
    (the last one is the stop level, drawn dashed); ``labels`` name the lanes (one colour each, coolest first).
    ``time_scale`` converts seconds to the x unit (1e3: milliseconds)."""
    import matplotlib.pyplot as plt

    times, levels = np.asarray(times, dtype=float), np.asarray(levels, dtype=float)
    assert times.shape == (len(levels), len(labels)) and (np.diff(times, axis=0) > 0).all(), "each level comes later"
    x_lo = 3e-6 * time_scale
    t_frames = np.geomspace(x_lo / time_scale, times.max() * 1.15, n_frames)
    t_frames = np.concatenate([t_frames, np.full(hold, t_frames[-1])])           # hold the last frame
    top = levels[-1]
    spread = np.linspace(0.15, 0.85, len(labels))

    def render(theme, t_now):
        pal = LIGHT if theme == "light" else DARK
        fig = plt.figure(figsize=(5.0, 2.8), dpi=150, facecolor=pal["bg"])
        ax = fig.add_axes([0.14, 0.18, 0.83, 0.74], facecolor=pal["bg"])
        ax.set_xscale("log"); ax.set_xlim(x_lo, times.max() * time_scale * 1.5); ax.set_ylim(-0.04 * top, 1.22 * top)
        ax.set_xlabel(xlabel, color=pal["sub"], fontsize=8)
        ax.set_ylabel(ylabel, color=pal["sub"], fontsize=8)
        ax.tick_params(colors=pal["sub"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(pal["rule"])
        ax.axhline(top, ls="--", lw=1.0, color=pal["off"])
        ax.text(1.2 * x_lo, 1.03 * top, stop_text, fontsize=7, color=pal["sub"])
        ax.text(0.98, 0.17, key_text, transform=ax.transAxes, ha="right", fontsize=6.5, color=pal["sub"])
        for lane, label in enumerate(labels):
            colour = plt.cm.viridis(spread[lane])
            seen = times[:, lane] <= t_now
            ax.plot(np.r_[x_lo, times[seen, lane] * time_scale], np.r_[0.0, levels[seen]], lw=1.1, color=colour)
            if seen[-1]:
                x = times[-1, lane] * time_scale
                ax.plot(x, top, "o", ms=4, color=colour, zorder=3)
                ax.text(x, 1.08 * top, label, ha="center", fontsize=6, color=pal["ink"])
        ax.text(0.98, 0.07, f"{int((times[-1] > t_now).sum())} of {len(labels)} {count_text}", transform=ax.transAxes,
                ha="right", fontsize=8, color=pal["ink"])
        return frame_of(fig, pal["bg"])

    sizes = {}
    for theme in ("light", "dark"):
        sizes.update(write_anim([render(theme, t_now) for t_now in t_frames], f"{stem}_{theme}"))
    return sizes
