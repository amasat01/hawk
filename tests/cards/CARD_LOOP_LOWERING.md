# CARD_LOOP_LOWERING — a bounded `for`, lowered

Written BY THE RUN (`tests/test_loop_lowering_card.py`). Prose cites this file; no number here may be restated without citing it. It DECIDES NOTHING — rules that whether to unroll a lowered loop is the C++/CUDA compiler's decision, and this card is the evidence that HAWK hands it a translation unit it can make that decision about.

Every compile in `grid` ran under `ulimit -v 12000000` (the same 12 GiB `hawk.compile.drivers` installs with `RLIMIT_AS`) and was measured with `/usr/bin/time -v`. `kan_scale_cell` — the point, a downstream package's own KAN edge shape at (n_in=64, G=20, k=3) — went trace to run inside ONE child under `ulimit -v 6000000`, a cap that binds the TRACER too.

md5: `a90c857ace714bdc41f3dd4db2fabe0b`

```json
{
  "a2_additive_loops": [
    {
      "bound": 4,
      "primal": 6.95712584333502,
      "primal_minus_reference_abs": 0.0,
      "primal_verdict": "bit-identical",
      "vjp_g": 5.351635264103861,
      "vjp_g_minus_reference_abs": 0.0,
      "vjp_pos_max_abs_error": 0.0,
      "vjp_verdict": "bit-identical",
      "why": "the reverse of an accumulator loop is a forward loop over the same range with NOTHING stored, so both arms sum the same terms in the same order; any difference is the association a compiler's fma contraction chooses"
    },
    {
      "bound": 16,
      "primal": 27.82850337334008,
      "primal_minus_reference_abs": 0.0,
      "primal_verdict": "bit-identical",
      "vjp_g": 21.406541056415445,
      "vjp_g_minus_reference_abs": 0.0,
      "vjp_pos_max_abs_error": 0.0,
      "vjp_verdict": "bit-identical",
      "why": "the reverse of an accumulator loop is a forward loop over the same range with NOTHING stored, so both arms sum the same terms in the same order; any difference is the association a compiler's fma contraction chooses"
    },
    {
      "bound": 64,
      "primal": 111.31401349336018,
      "primal_minus_reference_abs": 0.0,
      "primal_verdict": "bit-identical",
      "vjp_g": 85.62616422566178,
      "vjp_g_minus_reference_abs": 0.0,
      "vjp_pos_max_abs_error": 0.0,
      "vjp_verdict": "bit-identical",
      "why": "the reverse of an accumulator loop is a forward loop over the same range with NOTHING stored, so both arms sum the same terms in the same order; any difference is the association a compiler's fma contraction chooses"
    }
  ],
  "address_space_cap_bytes": 12884901888,
  "aether_include": "include",
  "card": "HAWK loop lowering: what a lowered `for` costs to compile",
  "device_arch": "sm_61",
  "device_compiler": "nvcc||nvcc: NVIDIA (R) Cuda compiler driver",
  "eagle_include": "eagle-abi",
  "grid": [
    {
      "basis": 5,
      "device_peak_rss_kib": 175324,
      "device_ptx_bytes": 9495,
      "device_tu_bytes": 5974,
      "device_wall_s": 0.61,
      "edges": 8,
      "host_peak_rss_kib": 97952,
      "host_tu_bytes": 5880,
      "host_wall_s": 0.429,
      "top_level_loops": 1,
      "walk_nodes": 11
    },
    {
      "basis": 11,
      "device_peak_rss_kib": 175108,
      "device_ptx_bytes": 9497,
      "device_tu_bytes": 5977,
      "device_wall_s": 0.621,
      "edges": 32,
      "host_peak_rss_kib": 97736,
      "host_tu_bytes": 5883,
      "host_wall_s": 0.442,
      "top_level_loops": 1,
      "walk_nodes": 11
    },
    {
      "basis": 23,
      "device_peak_rss_kib": 175364,
      "device_ptx_bytes": 9498,
      "device_tu_bytes": 5977,
      "device_wall_s": 0.614,
      "edges": 64,
      "host_peak_rss_kib": 97804,
      "host_tu_bytes": 5883,
      "host_wall_s": 0.448,
      "top_level_loops": 1,
      "walk_nodes": 11
    }
  ],
  "host_compiler": "/usr/bin/g++||g++ (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5.0.1)",
  "kan_scale_cell": {
    "G": 20,
    "compiled_minus_interpreted_abs": 4.440892098500626e-16,
    "cuda_tu_bytes": 28751,
    "device_arch": "sm_61",
    "device_peak_rss_kib": 177440,
    "device_ptx_bytes": 26796,
    "device_wall_s": 0.646,
    "host_peak_rss_kib": 104268,
    "host_run_wall_s": 0.0029,
    "host_tu_bytes": 28657,
    "host_wall_s": 0.533,
    "interpreted_minus_numpy_abs": 4.440892098500626e-16,
    "interpreter_wall_s": 0.0578,
    "k": 3,
    "loop_body_nodes": 727,
    "n_basis": 23,
    "n_in": 64,
    "peak_rss_kib": 36280,
    "samples": 256,
    "top_level_loops": 1,
    "trace_canonicalise_peak_rss_kib": 33748,
    "trace_canonicalise_wall_s": 0.0126,
    "ulimit_v_kib": 6000000,
    "verdict": "banded",
    "walk_nodes": 92,
    "why": "banded: the emitted chain lets the compiler contract a multiply-add into an fma across the body's Cox-de Boor levels, which the interpreter's separate numpy operations cannot (the same low-bit consequence records for a fused ET)"
  },
  "machine": "x86_64",
  "note": "every compile in `grid` ran under `ulimit -v 12000000` and was measured with `/usr/bin/time -v`; the 64x23 point is the one HW7d could not assemble at all when the same body was unrolled by trace-time execution (25-27 GB of ptxas RSS). `kan_scale_cell` is the point: a downstream package's own `kan_edge_kernel_loop` shape (a DEEP Cox-de Boor body, not the shallow `grid` family) at (n_in=64, G=20, k=3), taken trace -> canonicalise -> emit -> compile both targets -> run, all inside ONE child under `ulimit -v 6000000`, which caps the TRACER as well as the compilers. Before the fixed-width body key that child exhausted 6 GB during canonicalisation and never reached a compiler. The card DECIDES NOTHING: rules that whether to unroll is the compiler's decision, and no threshold is derived here",
  "numerics": [
    {
      "basis": 5,
      "compiled_minus_interpreted_abs": 0.0,
      "edges": 8,
      "host_run_wall_s": 0.0005,
      "interpreted_minus_numpy_abs": 0.0,
      "interpreter_wall_s": 0.0006,
      "samples": 512,
      "verdict": "bit-identical",
      "why": "bit-identical: the compiled body evaluates the same operations in the same association order as the interpreter walks them"
    },
    {
      "basis": 11,
      "compiled_minus_interpreted_abs": 8.881784197001252e-16,
      "edges": 32,
      "host_run_wall_s": 0.0058,
      "interpreted_minus_numpy_abs": 0.0,
      "interpreter_wall_s": 0.0071,
      "samples": 512,
      "verdict": "banded",
      "why": "banded: the emitted chain lets the compiler contract a multiply-add into an fma across iterations, which the interpreter's separate numpy operations cannot (the same low-bit consequence a fused ET already records elsewhere)"
    },
    {
      "basis": 23,
      "compiled_minus_interpreted_abs": 8.881784197001252e-16,
      "edges": 64,
      "host_run_wall_s": 0.0087,
      "interpreted_minus_numpy_abs": 0.0,
      "interpreter_wall_s": 0.0149,
      "samples": 512,
      "verdict": "banded",
      "why": "banded: the emitted chain lets the compiler contract a multiply-add into an fma across iterations, which the interpreter's separate numpy operations cannot (the same low-bit consequence a fused ET already records elsewhere)"
    }
  ],
  "platform": "Linux-6.12.0-100.28.2.el9uek.x86_64-x86_64-with-glibc2.34",
  "produced": "2026-09-23",
  "straight_line_twin": {
    "abs_difference": 0.0,
    "basis": 2,
    "edges": 2,
    "looped": -1.661909229969345,
    "straight_line_twin": -1.661909229969345,
    "verdict": "bit-identical",
    "why": "the twin is written with the loop's own seeds and association order, so the two evaluate the identical expression tree; a difference here would mean the lowering changed the arithmetic rather than where it is written"
  },
  "ulimit_v_kib": 12000000
}
```
