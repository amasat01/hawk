# Variant data: ONE literal, no code. Extends the cpu base variant (one level, data only).
{
    "name": "x86",
    "extends": "cpu",
    "isa": "x86_64",
    # Chain-fusion constants of an x86 host's vector ISA, keyed by the ISA name the machine reports: the register line only (R_file: 16 ymm,
    # 32 zmm); their costs are measured per host by the CPU profile sweep.
    "fusion_host": {
        "sse2": {"R0": 0, "R_file": 16, "L_us": None, "c_plane_us_per_B": None, "c_local_over_plane": None},
        "avx2": {"R0": 0, "R_file": 16, "L_us": None, "c_plane_us_per_B": None, "c_local_over_plane": None},
        "avx512": {"R0": 0, "R_file": 32, "L_us": None, "c_plane_us_per_B": None, "c_local_over_plane": None},
    },
}
