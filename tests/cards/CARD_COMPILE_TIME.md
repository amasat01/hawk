# CARD_COMPILE_TIME — one card per consumer TU

Written BY THE RUN (`tests/test_compile_time_card.py`). Prose cites this file; no number here may be restated without citing it. It DECIDES NOTHING — it is the evidence gating aether's header partition  and whether the compile/cache driver moves to C++.

Host phases come from `g++ -ftime-report` (seconds, wall column); device phases from `nvcc --time` (milliseconds, per compilation phase).

md5: `00b09c7db2d6be29a899b95de94ccdc1`

```json
{
  "aether_include": "include",
  "card": "HAWK compile time (evidence)",
  "device_arch": "sm_61",
  "device_compiler": "nvcc||nvcc: NVIDIA (R) Cuda compiler driver",
  "eagle_include": "eagle-abi",
  "host_compiler": "/usr/bin/g++||g++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5.0.1)",
  "machine": "x86_64",
  "note": "every compile below is COLD (a fresh tmpdir, no cache slot); the machine was shared at measurement time, so these are ORDERS OF MAGNITUDE for a deferred decision, not a gate",
  "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
  "produced": "2026-09-23",
  "rows": {
    "eagle/host_plugin_execv2/host": {
      "phases_s": {
        "TOTAL": 0.45,
        "phase lang. deferred": 0.04,
        "phase opt and generate": 0.03,
        "phase parsing": 0.38,
        "phase setup": 0.0
      },
      "wall_s": 0.811
    },
    "hawk/axpb/device": {
      "phases_ms": {
        "cicc": 438.52,
        "g++ (preprocessing 1)": 128.862,
        "nvcc (driver)": 0.543
      },
      "wall_s": 0.62
    },
    "hawk/axpb/host": {
      "phases_s": {
        "TOTAL": 0.78,
        "phase lang. deferred": 0.06,
        "phase opt and generate": 0.02,
        "phase parsing": 0.69,
        "phase setup": 0.01
      },
      "wall_s": 1.333
    },
    "hawk/spin/device": {
      "phases_ms": {
        "cicc": 483.507,
        "g++ (preprocessing 1)": 126.757,
        "nvcc (driver)": 0.638
      },
      "wall_s": 0.658
    },
    "hawk/spin/host": {
      "phases_s": {
        "TOTAL": 0.89,
        "phase lang. deferred": 0.09,
        "phase opt and generate": 0.08,
        "phase parsing": 0.71,
        "phase setup": 0.01
      },
      "wall_s": 1.514
    },
    "hawk/vec3_scale/device": {
      "phases_ms": {
        "cicc": 456.138,
        "g++ (preprocessing 1)": 126.032,
        "nvcc (driver)": 0.653
      },
      "wall_s": 0.632
    },
    "hawk/vec3_scale/host": {
      "phases_s": {
        "TOTAL": 0.78,
        "phase lang. deferred": 0.07,
        "phase opt and generate": 0.04,
        "phase parsing": 0.66,
        "phase setup": 0.01
      },
      "wall_s": 1.332
    },
    "hawk/vocab/device": {
      "phases_ms": {
        "cicc": 520.409,
        "g++ (preprocessing 1)": 138.533,
        "nvcc (driver)": 0.653
      },
      "wall_s": 0.703
    },
    "hawk/vocab/host": {
      "phases_s": {
        "TOTAL": 0.84,
        "phase lang. deferred": 0.07,
        "phase opt and generate": 0.07,
        "phase parsing": 0.7,
        "phase setup": 0.0
      },
      "wall_s": 1.397
    }
  }
}
```
