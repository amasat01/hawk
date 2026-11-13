# Variant data: ONE literal, no code. Base variant of the SIMT family (data only; extended by cuda, one level).
{
    "name": "simt",
    # The regime of a steps_reverse row is the batch per resident thread: N / (SM count x resident threads per SM).
    "steps_reverse": {"regime": "per_resident"},
    # The SIMT form's own limits (a device twin of the cpu values, equal today): up to max_dim one thread holds a matrix, a team
    # above it; a site is static up to rolled_fold_min elements, rolled above it.
    "linalg": {"max_dim": 32,
               # the largest dimension a team of threads works on (the team scratch is the bound); aether's linalg::team holds the same number
               "team_max_dim": 4096},
    # lazy_leaf_width: the element count above which a read-only leaf read only by static component reads is bound as a proxy, not copied
    # whole (``.get()`` is one register per element); lean_width: the widest vector whose single indexed accumulate keeps the select-chain
    # spelling when Real is four bytes (at width 4 ptxas allocates 32 registers for it against 38 for per-component selects on sm_61).
    "form": {"rolled_fold_min": 16, "lazy_leaf_width": 32, "lean_width": 4},
    # The lanes a warp spans: the ONE home of the warp width (threads per block are a multiple of it; a sliced value holds W / warp_size
    # elements per lane; a uniform access is shared over an aligned run of it). cuda inherits it.
    "launch": {"warp_size": 32},
    # Samples one row of a reduction over samples stands for: one warp reduces one row, so it must equal launch.warp_size.
    "cells": {"row": 32},
    # The most lane chunks an ordered scatter adds in (a chunk is one thread's run of lanes).
    # `threads`: the team of each of its stages (one does the adds, the rest of a warp idles).
    "scatter": {"max_chunks": 256, "threads": 32},
    # Lane-run and staging numbers of the SIMT lowerings (read by simt/lanes.py, refused by name when absent).
    "lanes": {
        # words a row of shared memory serves without a bank conflict
        "bank_words": 32,
        # the widest run of lanes a keyed commit reduces over (the coalescing width)
        "run_max": 32,
        # the lanes a block key-run spans: every block size divides it
        "block_run": 1024,
        # the threads of the block of a stage that only copies operands into team scratch (a block per lane)
        "copy_threads": 256,
    },
}
