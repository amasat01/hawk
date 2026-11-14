# Facts of the cuda wing: ONE literal, no code. It states what it adds or changes over its parent, the simt wing.
{
    # steps_reverse="auto" rows measured on NVIDIA cards (the regime, per_resident, is the simt base's).
    # P2000 (5 SMs x 2048 threads, regime = N / 10240), the wide matvec: fused 0.43x at N=64, 2.2x at 1024, 2.8x at 16384
    # and 65536 (L4: 1.8x at 65536).
    "steps_reverse": {
        "rows": ((2, 8, 0.1, 6.4),),
    },
    # The linalg ops a device works one warp per lane, as rows (kind, n, scalar, outputs read beyond the base, aether::linalg::warp function).
    # A row is bit-identical to the thread form and faster at every lane count measured (P2000, f32: 256 to 131072 matrices, 1.5x to 5.7x), so
    # a device takes it with no fill test, except that a float64 row needs a device whose FP64 rate is at least 1/fast_fp64 of FP32: on a
    # 1:32 or 1:64 part the f64 fold is pipe-bound and the thread form stays.
    "linalg": {"warp_entries": (("eigh", 32, "float32", (), "eighValues"), ("eigh", 32, "float64", (), "eighValues")), "fast_fp64": 8},
    # Launch limits: threads per block are a multiple of warp_size; threads x blocks_per_sm at most max_threads_per_sm.
    "launch": {"max_threads_per_sm": 2048},
    # ptxas registers of a one-column sample-sum warp (20 for 1 row, 26 for 4, 43 for 8; width 1) and the cost of each further
    # column, against the register line under which every arch keeps full residency; k_max rows per warp at most.
    "fold": {"regs_base": 20, "regs_column": 2, "residency_regs": 32, "k_max": 8},
    # Hardware facts of the archs a build may target: compute capability -> (fp64_ratio, regs_per_sm, threads_per_sm, local_cached).
    "arch_features": {
        61: (32, 65536, 2048, True),
        70: (2, 65536, 2048, True),
        75: (32, 65536, 1024, True),
        80: (2, 65536, 2048, True),
        86: (64, 65536, 1536, True),
        89: (64, 65536, 1536, True),
        90: (2, 65536, 2048, True),
    },
    # The class a build with no device arch (a direct call of the pass) uses.
    "default_arch": "sm_61",
    # Measured classes: name -> (capability whose features it has, {pattern: {scalar bytes: widest width kept in registers}}, wide_min).
    # sm61: Quadro P2000, widths 2 to 128 x read / fill-then-read / rotate / write, three problem sizes
    # (hawk/targets/cards/_probe_wide.py); the faster form wins and a spill-free one only on a tie.
    "wide_classes": {
        "sm61": {
            "capability": 61,
            "runtime_min": {
                "read": {4: 16, 8: 2},
                "slot_read": {4: 16, 8: 4},
                "rotate": {4: 16, 8: 0},
                "write": {4: 96, 8: 8},
                "accum": {4: 4096, 8: 4096},   # an in-place accumulation (reverse sum): slower resident at 512, no width found
            },
            "wide_min": {4: 128, 8: 64},
        },
    },
}
