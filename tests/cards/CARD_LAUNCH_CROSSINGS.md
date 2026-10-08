# CARD_LAUNCH_CROSSINGS — the per-launch boundary cost

Written BY THE RUN (`tests/test_launch_crossings.py`). Prose cites this file; no number here may be restated without citing it.

`crossings_per_launch` is the GATE: 1 for a stable-binding loop, 2 for a rebind-every-slot loop, whatever the slot count. The seconds beside them are a measurement on a shared machine — arms interleaved, spread reported — and gate nothing.

md5: `1cec034cdfa2279ff099f8f146035ce4`

```json
{
  "aether_include": "include",
  "ambient_conda_prefix": "<unset>",
  "card": "HAWK launch-boundary crossings and per-launch cost",
  "core_build_digest": "d318b186df2a35f27232947abd05cde705d0a8d4ed44130b339c14c60b967bc7",
  "core_compiler": "GNU 13.4.0",
  "core_md5": "1a87ee54ea3a0e0d6337461a4bd6b745",
  "core_path": "_core.cpython-312-x86_64-linux-gnu.so",
  "cuda_visible_devices": "1",
  "eagle_include": "eagle-abi",
  "interpreter": "python 3.12.13",
  "machine": "x86_64",
  "note": "the CROSSING COUNTS are the gate and are exact integers; the seconds are a MEASUREMENT on a shared box, interleaved arm by arm, reported with their spread and gating nothing",
  "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
  "produced": "2026-09-23",
  "rows": {
    "axpb": {
      "arg_slots": 4,
      "crossings_per_launch": {
        "rebind_every_slot": 2,
        "stable_binding": 1
      },
      "launches_per_repeat": 20000,
      "pointer_slots": 2,
      "rebind_every_slot": {
        "repeats": 5,
        "seconds_per_launch_max": 6.29e-07,
        "seconds_per_launch_median": 6.11e-07,
        "seconds_per_launch_min": 6.02e-07
      },
      "samples": 256,
      "stable_binding": {
        "repeats": 5,
        "seconds_per_launch_max": 3.54e-07,
        "seconds_per_launch_median": 3.45e-07,
        "seconds_per_launch_min": 3.43e-07
      }
    },
    "spin": {
      "arg_slots": 4,
      "crossings_per_launch": {
        "rebind_every_slot": 2,
        "stable_binding": 1
      },
      "launches_per_repeat": 20000,
      "pointer_slots": 4,
      "rebind_every_slot": {
        "repeats": 5,
        "seconds_per_launch_max": 1.732e-06,
        "seconds_per_launch_median": 1.706e-06,
        "seconds_per_launch_min": 1.696e-06
      },
      "samples": 256,
      "stable_binding": {
        "repeats": 5,
        "seconds_per_launch_max": 1.533e-06,
        "seconds_per_launch_median": 1.424e-06,
        "seconds_per_launch_min": 1.42e-06
      }
    },
    "vec3_scale": {
      "arg_slots": 3,
      "crossings_per_launch": {
        "rebind_every_slot": 2,
        "stable_binding": 1
      },
      "launches_per_repeat": 20000,
      "pointer_slots": 2,
      "rebind_every_slot": {
        "repeats": 5,
        "seconds_per_launch_max": 7.71e-07,
        "seconds_per_launch_median": 7.32e-07,
        "seconds_per_launch_min": 7.26e-07
      },
      "samples": 256,
      "stable_binding": {
        "repeats": 5,
        "seconds_per_launch_max": 5.69e-07,
        "seconds_per_launch_median": 4.59e-07,
        "seconds_per_launch_min": 4.54e-07
      }
    }
  }
}
```
