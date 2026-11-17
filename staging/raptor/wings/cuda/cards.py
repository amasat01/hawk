# Facts of the cuda wing: the per-device cards, read as the key "cards" by hawk/facts.py.
# Defaults are the values measured on a Quadro P2000 (sm_61); a row {cc, device, dtype, ...} of a card replaces them on the nearest arch
# by device features.
# fusion: R0 registers of an empty kernel, R_file registers a lane may hold before spilling, L_us the cost of one more launch (us),
# c_plane_us_per_B the cost of a plane byte per sample (us), c_local_over_plane a local byte over a plane byte (None: not measured).
{'fusion': {'defaults': {'R0': 20,
                         'R_file': 255,
                         'L_us': 2.7,
                         'c_plane_us_per_B': 7.1e-06,
                         'c_local_over_plane': 1.0},
            'rows': []}}
