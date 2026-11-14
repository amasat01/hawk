# Facts of the arm wing: ONE literal, no code. It states what it adds or changes over its parent, the cpu wing.
{
    "isa": "arm64",
    # Chain-fusion constants of an Arm host's vector ISA (32 NEON registers); costs are measured per host by the CPU profile sweep.
    "fusion_host": {
        "neon": {"R0": 0, "R_file": 32, "L_us": None, "c_plane_us_per_B": None, "c_local_over_plane": None},
    },
}
