# Variant data: ONE literal, no code. Extends the cpu base variant (one level, data only).
{
    "name": "arm",
    "extends": "cpu",
    "isa": "arm64",
    # Chain-fusion constants of an Arm host's vector ISA (32 NEON registers); costs are measured per host by the CPU profile sweep.
    "fusion_host": {
        "neon": {"R0": 0, "R_file": 32, "L_us": None, "c_plane_us_per_B": None, "c_local_over_plane": None},
    },
}
