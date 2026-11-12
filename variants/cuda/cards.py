# Variant data: ONE literal, no code. The per-device cards of the cuda variant, merged under the key "cards" by hawk/variants.py.
# Rewritten by the sweep (python -m hawk_cuda.sweep --write); the sections below say what each number is.
# The per-device cards (they were files under hawk/targets/cards/). Defaults are the values measured on a Quadro P2000 (sm_61); a row
# {cc, device, dtype, ...} of a card replaces them on the nearest arch by device features, rows are written by the sweep.
# wide: SLICE_MIN the narrowest vector a warp-form kernel holds sliced across a warp (a multiple of 32), TEAM_MIN the shortest
# sliced value from which wide_form "auto" builds a warp per lane, TC the bytes of shared memory a block's output tiles may take.
# fusion: R0 registers of an empty kernel, R_file registers a lane may hold before spilling, L_us the cost of one more launch (us),
# c_plane_us_per_B the cost of a plane byte per sample (us), c_local_over_plane a local byte over a plane byte (None: not measured).
# thread_box: T lanes a thread computes (2..16) and the contraction route per scalar type, select_weight the cost of one add of a
# runtime-index store written as selects against an indexed access (rows keyed by compute capability alone), rows may carry
# tile = {kind, SB, T, R, EB, smem} (a block tile; absent by default); "examples" are documentation, never read.
{'wide': {'defaults': {'SLICE_MIN': 128, 'TEAM_MIN': 128, 'TC': 40960}, 'rows': []},
 'fusion': {'defaults': {'R0': 20,
                         'R_file': 255,
                         'L_us': 2.7,
                         'c_plane_us_per_B': 7.1e-06,
                         'c_local_over_plane': 1.0},
            'rows': []},
 'thread_box': {'defaults': {'float64': 16, 'float32': 8},
                'contraction': {'float64': 'gemm', 'float32': 'gemm'},
                'rows': [],
                'select_weight': {'defaults': {'float64': 1.0, 'float32': 1.0},
                                  'rows': [{'cc': 'sm_61', 'dtype': 'float64', 'select_weight': 4.0}]},
                'examples': [{'cc': 'sm_61',
                              'device': 'Quadro P2000',
                              'dtype': 'float32',
                              'tile': {'kind': 'BT', 'SB': 128, 'T': 16, 'R': 1, 'EB': 4, 'smem': 49152}},
                             {'cc': 'sm_61',
                              'device': 'Quadro P2000',
                              'dtype': 'float64',
                              'tile': {'kind': 'BF', 'SB': 64, 'T': 8, 'R': 2, 'EB': 1, 'smem': 49152}}]}}
