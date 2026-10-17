# autodiff_vs_torch card

Schema: `raptor-perf-card/1`  ·  generated 2026-10-17T07:50:17Z

## Facts

```json
{
  "device": {
    "name": "Quadro P2000"
  },
  "example_script": "examples/autodiff_vs_torch.py",
  "example_script_md5": "77bba2091f3ddf957de08a287036da65",
  "generated_utc": "2026-10-17T07:50:17Z",
  "peak": null,
  "script_md5": "1a3c8954f037b4a3d8643a137600e735",
  "software": {
    "compile": {
      "device_flags_note": "hawk.compile.toolchain.device_flags now carries the same opt_level explicitly: -O3 for nvcc's own host-side code generation and -Xptxas -O3 for ptxas; NVRTC device compiles always optimise and ignore opt_level",
      "host_flags": [
        "-O3",
        "-ffp-contract=fast",
        "-march=native",
        "-mprefer-vector-width=512",
        "-fno-trapping-math",
        "-DAETHER_HOST_VECTOR_MATH"
      ],
      "host_profile": "native-vector-math",
      "opt_level": "O3"
    },
    "cpu": "Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz",
    "cupy": "14.2.0",
    "eagle_commit": "3f9a26785f55bcd5caac37c950d9263451469a27-dirty",
    "hawk": "0.5.1",
    "hawk_commit": "127857f8a327e96c9becf32d1cb7e85df9f5e198",
    "numpy": "2.5.3",
    "omp_num_threads": null,
    "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
    "python": "3.12.14",
    "torch": "2.12.1"
  }
}
```

## Results

```json
[
  {
    "batch": 1,
    "hawk_jvp": 1.7786817625164986e-05,
    "hawk_primal": 1.8076086416840553e-05,
    "hawk_vjp": 1.725819893181324e-05,
    "ratio_jvp": 359.8003607622608,
    "ratio_vjp": 284.69422071842365,
    "torch_jvp": 0.006399703398346901,
    "torch_primal": 0.0006674312986433506,
    "torch_vjp": 0.004913309495896101
  },
  {
    "batch": 32,
    "hawk_jvp": 1.1104624718427658e-05,
    "hawk_primal": 1.1631799861788749e-05,
    "hawk_vjp": 1.1154869571328164e-05,
    "ratio_jvp": 598.0251520107352,
    "ratio_vjp": 455.1939811896522,
    "torch_jvp": 0.006640844885259867,
    "torch_primal": 0.0004669808316975832,
    "torch_vjp": 0.005077629489824176
  },
  {
    "batch": 4096,
    "hawk_jvp": 1.2325402349233627e-05,
    "hawk_primal": 1.2553995475172997e-05,
    "hawk_vjp": 1.2300163507461549e-05,
    "ratio_jvp": 515.6126429051783,
    "ratio_vjp": 341.7436019746804,
    "torch_jvp": 0.006355133280158043,
    "torch_primal": 0.0004174694884568453,
    "torch_vjp": 0.004203502181917429
  },
  {
    "batch": 65536,
    "hawk_jvp": 3.888672217726708e-05,
    "hawk_primal": 2.9808329418301583e-05,
    "hawk_vjp": 4.079770296812057e-05,
    "ratio_jvp": 163.73240600369303,
    "ratio_vjp": 111.18177678045575,
    "torch_jvp": 0.006367016583681107,
    "torch_primal": 0.0005690267775207758,
    "torch_vjp": 0.0045359611045569185
  },
  {
    "batch": 1048576,
    "hawk_jvp": 0.0004155570175498724,
    "hawk_primal": 0.000283416174352169,
    "hawk_vjp": 0.0004300342872738838,
    "ratio_jvp": 63.397394828496,
    "ratio_vjp": 91.57925258876205,
    "torch_jvp": 0.0263452323153615,
    "torch_primal": 0.008944086218252777,
    "torch_vjp": 0.039382218616083264
  }
]
```
