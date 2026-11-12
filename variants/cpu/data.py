# Variant data: ONE literal, no code. Read by hawk/variants.py (ast.literal_eval).
{
    "name": "cpu",
    # steps_reverse="auto" rows for a host target: (state width, K, narrowest regime, widest regime); regime = N.
    # 8 Xeon cores, compiled hawk.run: the 4-scalar oscillator probe, fused 7x / 5x at N=1 / 64, 1.2x at 1024 and slower from
    # 16384 (K 8); K 4 unchanged. Width 2, K 8 is the wide matvec of the feature card (24 by 512 table): fused 30-250x at every N.
    "steps_reverse": {
        "regime": "samples",
        "rows": ((4, 8, 1, 1024), (4, 4, 1, 64), (2, 8, 1, 65536)),
    },
    # Per-lane linear algebra: up to max_dim one thread holds a matrix, above it a team of threads works on it.
    "linalg": {"max_dim": 32,
               # the largest dimension a team of threads works on in a host text (the host's own limit, at the value it has always had)
               "team_max_dim": 4096},
    # Lane-run numbers the host text's keyed commit reads (the host's own, at the values it has always used).
    "lanes": {"run_max": 32, "block_run": 1024},
    # Samples one row of a reduction over samples stands for: the host's own choice, a power of two (the host text masks with row - 1);
    # the runtime partitions and sizes its scratch by it.
    "cells": {"row": 32},
    # The most lane chunks an ordered scatter adds in (a chunk is one thread's run of lanes): the host's own threading choice.
    "scatter": {"max_chunks": 256},
    # A site is spelled static while its fold length and its result size are both at most rolled_fold_min elements,
    # rolled above it (also the element count above which a TeamCopy sink copies by a loop and a block is a team stage).
    "form": {"rolled_fold_min": 16},
    # The aligned run of lanes a uniform access is proven equal over, in the host text (the host's own decision, kept at the value it has
    # always used).
    "uniform_run": 32,
    # The thresholds the HOST's wide-value placement reads (a host build whose loops run interchanged places wide carries too): the host's
    # own decision, kept at the values the host has always placed by.
    "host_wide": {
        "name": "host",
        "runtime_min": {
            "read": {4: 16, 8: 2},
            "slot_read": {4: 16, 8: 4},
            "rotate": {4: 16, 8: 0},
            "write": {4: 96, 8: 8},
            "accum": {4: 4096, 8: 4096},
        },
        "wide_min": {4: 128, 8: 64},
    },
    # The host's own thread-box table (the thread box of a host text: T lanes a thread visits along a non-fastest axis) and the weight of a
    # select-written runtime-index store; the same numbers the device card measured, kept as the host's own facts.
    "thread_box": {
        "defaults": {'float64': 16, 'float32': 8},
        "contraction": {'float64': 'gemm', 'float32': 'gemm'},
        "select_weight": {"defaults": {'float64': 1.0, 'float32': 1.0}},
    },
}
