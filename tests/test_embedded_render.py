# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target's renderer: ``hawk.embedded.render`` turns a kernel
into a ``.c``/``.h`` pair through the one body traversal and the C writer.

Rows:

* **header** — the args struct (one typed field per slot, pointer constness
  from the interface direction), the status-returning prototype and the
  ``HAWK_EMBEDDED_TYPES`` width guard;
* **body** — every rank>=1 value is a named C struct value, every fold a CORE
  function inlined above the entry, the SoA pitch is ``n_samples`` (never the
  partition's ``count``), a float32 unit never names ``double``;
* **corpus** — every corpus kernel (``energy`` is a sum over samples, refused) renders
  in both modes, deterministically, with no aether C++ left in it;
* **sites** — a scattered accumulate is a clamped ``+=``, a runtime-count loop
  clamps its count to the static trip, the guard and the finish epilogue spell
  the ``uint8_t`` masks.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import _deployable as D
import _kernels as K
import pytest

import hawk
from hawk import Mutable, Scalar
from hawk.embedded import render
from hawk.emit import FLOAT32, FLOAT64
from hawk.ir import HawkError, canonical

DRAG_H_STRUCT = ("typedef struct {\n    hawk_real_t*       mut_out;\n    const hawk_real_t* vin_state;\n"
                 "    hawk_real_t        uni_rho;\n} drag_args_t;\n")  # exact comment columns per abi.py's formatter

#: The field comments the spec's ``drag.h`` carries after each member; the
#: struct row compares the declarations themselves.
_FIELD_COMMENT = re.compile(r"[ ]*/\*[^\n]*?\*/(?=\n)")


def _derived(name, sinks):
    """A kernel-shaped object over a derived sink set (``vjp`` returns sinks,
    not a :class:`~hawk.trace.kernel.Kernel`): what ``render`` reads of one."""
    return SimpleNamespace(name=name, sinks=tuple(sinks), walk=canonical(sinks),
                           kind=None)


def test_drag_header_struct_and_prototype():
    s = render(K.drag, mode="float64")
    header = _FIELD_COMMENT.sub("", s.header)
    assert DRAG_H_STRUCT.replace("    ", " ") in header.replace("    ", " ")  # whitespace-normalised
    assert "hawk_status_t drag_c(const drag_args_t* args, int64_t base, int64_t count, int64_t n_samples);" in s.header
    assert "#define HAWK_EMBEDDED_TYPES 64" in s.header


def test_drag_body_names_every_tensor_and_uses_the_core():
    s = render(K.drag)
    assert "const aether_c_f64_vec3_t vin_state_i = hawk_vec3_load(vin_state, hawk_i, n_samples);" in s.text
    assert "aether_c_f64_vec3_norm(" in s.text and "hawk_vec3_store(mut_out, hawk_i, n_samples, " in s.text
    assert s.report["core"] == sorted(s.report["core"]) and "aether_c_f64_vec3_norm" in s.report["core"]
    for fn in s.report["core"]:
        assert s.text.count("/* aether-c: ") >= 1 and fn + "(" in s.text


def test_pitch_is_n_samples_not_count():
    loop = render(K.drag).text.split("for (int64_t hawk_i")[1]
    assert "hawk_vec3_load(vin_state, hawk_i, n_samples)" in loop and "hawk_i, count)" not in loop


def test_float32_text_never_promotes():
    s = render(K.drag, mode=FLOAT32.id)
    assert "sqrtf(" in s.text and "double" not in s.text and "aether_c_f32_vec3_t" in s.text


def test_every_corpus_kernel_renders_in_both_modes():
    for name, k in K.CORPUS.items():
        if name == "energy":
            continue
        for mode in (FLOAT64, FLOAT32):
            s = render(k, mode=mode.id)
            assert ".eval(" not in s.text and "aether::" not in s.text and "std::" not in s.text
            assert render(k, mode=mode.id).text == s.text


def test_int_arithmetic_routes_through_wrapping_helpers():
    s = render(K.scatter)   # `x * x` is Real; the Index lane is used as an index: expect hawk_clamp_index
    assert "hawk_clamp_index(" in s.text and "+=" in s.text
    import test_int_planes as I

    ints = render(I.ops_i32)
    for helper in ("hawk_i32_add(", "hawk_i32_mul(", "hawk_i32_floordiv(", "hawk_i32_mod("):
        assert helper in ints.text
    assert "hawk_u32_add(" in render(I.ops_u32).text


def test_an_integer_param_is_a_64_bit_field():
    import test_int_planes as I

    s = render(I.casts)
    assert "    hawk_int_t         uni_s;" in _FIELD_COMMENT.sub("", s.header)
    assert re.search(r"\*   uni_s +uniform +in +i64 ", s.header)
    assert "<stdbool.h>" not in s.header


def test_two_masks_guard_and_finish():
    import test_finish_own_sample as F

    two = render(D.diagnostic_two, kind=D.TWO_MASKS)
    assert ("if (!((trm_terminated[hawk_i] != (uint8_t)0) || "
            "(trm_rejected[hawk_i] != (uint8_t)0))) {") in two.text
    assert "const uint8_t* const trm_rejected = args->trm_rejected;" in two.text
    one = render(D.diagnostic)
    assert "if (!(trm_terminated[hawk_i] != (uint8_t)0)) {" in one.text
    fin = render(F.stop_only)
    assert "if ((trm_terminated[hawk_i] == (uint8_t)0) && " in fin.text
    assert "trm_terminated_w[hawk_i] = (uint8_t)1;" in fin.text
    assert ("lut_finished_count[0] = lut_finished_count[0] + 1U;" in fin.text)
    # the finish mask is written, so its field is non-const and the twin binds it
    assert "    uint8_t*           trm_terminated;" in fin.header
    assert "uint8_t* const trm_terminated_w = args->trm_terminated;" in fin.text
    assert "uint32_t*" in fin.header and "lut_finished_count" in fin.header


def test_two_masks_and_a_finish_in_one_kernel():
    """``two_fin.step`` under ``TWO_MASKS``: the guard reads BOTH mask bytes,
    the finish writes only its own mask (through the writable twin), and the
    other mask stays a ``const`` input."""
    one = render(D.two_fin.step, kind=D.TWO_MASKS)
    guard = ("if (!((trm_terminated[hawk_i] != (uint8_t)0) || "
             "(trm_rejected[hawk_i] != (uint8_t)0))) {")
    assert one.text.count(guard) == 1
    body = one.text.split(guard, 1)[1]
    assert "if ((trm_terminated[hawk_i] == (uint8_t)0) && " in body
    assert "trm_terminated_w[hawk_i] = (uint8_t)1;" in body
    assert "trm_rejected_w" not in one.text
    assert not re.search(r"trm_rejected\[hawk_i\] = ", one.text)
    assert "const uint8_t* const trm_rejected = args->trm_rejected;" in one.text
    assert "uint8_t* const trm_terminated_w = args->trm_terminated;" in one.text
    header = _FIELD_COMMENT.sub("", one.header)
    assert "    const uint8_t*     trm_rejected;" in header
    assert "    uint8_t*           trm_terminated;" in header
    rows = {r["field"]: r for r in one.report["interface"]}
    assert rows["trm_rejected"]["direction"] == "in"
    assert rows["trm_terminated"]["direction"] == "inout"
    assert rows["mut_y"]["reason"] == (
        "masked samples keep the prior value (guard: trm_terminated, trm_rejected)")


def test_runtime_count_loop_clamps_to_trip():
    """A vjp through a head ``break`` reads the forward loop's iteration count
    back in its reverse loop: a RUNTIME count the C clamps to the static trip."""
    import test_loop_early_exit as E

    from hawk.diff import vjp

    k = _derived("head_vjp", vjp(E.head, wrt=("x",)))
    for mode in (FLOAT64, FLOAT32):
        s = render(k, mode=mode.id)
        assert "hawk_clamp_trips(" in s.text
        assert re.search(r"const int32_t hawk_cnt\d+ = hawk_clamp_trips\(\w+, "
                         r"INT32_C\(12\)\);", s.text)
        assert "if (hawk_v" in s.body and "if ((" not in s.body and "break;" in s.body


def test_a_runtime_count_header_never_multiplies_by_one():
    """MISRA C:2012 Rule 2.2: an operation with no effect is dead code. A
    runtime-count header scales by the step's magnitude only when it is not 1
    (cppcheck 2.13 reports ``(12 - cnt) * INT32_C(1)``)."""
    import test_loop_early_exit as E

    from hawk.diff import vjp

    for name, fwd in (("head_vjp", E.head), ("tail_vjp", E.tail)):
        k = _derived(name, vjp(fwd, wrt=("x",)))
        for mode in (FLOAT64, FLOAT32):
            text = render(k, mode=mode.id).text
            assert "hawk_clamp_trips(" in text
            assert not re.search(r"\* (INT32_C|INT64_C)\(1\)\)", text), name


def test_a_vjp_kernel_renders():
    """``drag_vjp()`` is a sink set, not a Kernel; :func:`_derived` gives it the
    three attributes ``render`` reads (name, sinks, walk; kind defaults)."""
    s = render(_derived("drag_vjp", K.drag_vjp()))
    assert "hawk_status_t drag_vjp_c(const drag_vjp_args_t* args" in s.header
    assert "aether_c_f64_vec3_dot(" in s.text or "aether_c_f64_vec3_sum(" in s.text
    assert "aether_c_f64_vec3_scale(" in s.text
    assert "mut_bar_rho[hawk_i] = " in s.text
    assert "hawk_vec3_store(mut_bar_state, hawk_i, n_samples, " in s.text


def test_scattered_accumulate_is_a_clamped_plus_equals():
    s = render(D.scatter)
    assert "aout_acc[hawk_clamp_index(((hawk_int_t)psc_lane[hawk_i]), aout_acc_len)] += " in s.text
    assert "    int64_t            aout_acc_len;" in _FIELD_COMMENT.sub("", s.header)
    (row,) = [r for r in s.report["interface"] if r["field"] == "aout_acc"]
    assert row["direction"] == "inout" and row["runtime_length"] is True


@hawk.kernel
def eq_break(x: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(8):
        if s == lim:
            break
        s = s * 0.5 + 1.0
    y = s


def test_a_head_break_on_an_equality_is_parenthesised_once():
    """clang's ``-Wparentheses-equality`` rejects ``if ((a == b))``: the
    break's ``if`` wraps a condition only when it is not already one
    parenthesised expression."""
    for mode in (FLOAT64, FLOAT32):
        s = render(eq_break, mode=mode.id)
        assert "if ((" not in s.body
        assert re.search(r"if \(hawk_v\w+ == psc_lim\w*\) \{", s.body)


def test_the_branch_mix_plane_is_k_cells():
    import test_dispatch_mix_and_guard as G

    s = render(G.mix_auto)
    (row,) = [r for r in s.report["interface"] if r["field"] == "lut_dispatch0_mix"]
    assert row["shape"] == [3]
    assert re.search(r"uint32_t\* +lut_dispatch0_mix; +/\* lookup +inout +Vector\[3\] +"
                     r"3 cells, the caller zeroes them \*/", s.header)
    assert re.search(r"\*   lut_dispatch0_mix +lookup +inout +u32 +\(3,\) ", s.header)
    assert "counter, the caller zeroes it" not in s.header


def test_a_clamped_tape_read_is_refused_by_name(monkeypatch):
    """No fixture produces one today: the renderer spells a tape subscript
    itself, so a clamped tape read could only render raw."""
    from hawk.embedded import verify

    real = verify.ir_checks

    def planted(*args, **kwargs):
        report = real(*args, **kwargs)
        report["indices"].append({"node": "loop7.3", "kind": "tape_read",
                                  "status": "clamped", "reason": "planted",
                                  "length": 12})
        return report

    monkeypatch.setattr(verify, "ir_checks", planted)
    with pytest.raises(HawkError) as err:
        render(K.drag)
    message = str(err.value)
    for needle in ("'drag'", "node loop7.3", "(tape_read)", "index.unproven",
                   "the C target cannot clamp a tape subscript yet"):
        assert needle in message


def test_per_component_commits_stay_in_the_aether_writer(tmp_path):
    """hawk 0.6.4 stores a vector built from scalars one component at a time
    (``target(k, i.global()) = part``) so a wide value is never live whole. That
    is an aether-view optimisation: the C writer keeps the single store, and the
    unit passes every check (render, audit, strict cc, cppcheck)."""
    import test_math_parity as MP

    from hawk.embedded import export

    for mode in (FLOAT64, FLOAT32):
        body = render(MP.unary_all, mode=mode.id).body
        assert "i.global()" not in body and "hawk_vec" in body
        export(MP.unary_all, tmp_path / mode.id, mode=mode.id)


def test_a_runtime_index_accumulate_in_a_loop_is_refused_by_name():
    """A vjp through a rolled, runtime-indexed loop accumulates its gradient in
    place, ``bar[i].at(k) += g`` per iteration: the C target does not lower it
    and says so, rather than emitting aether view syntax."""
    import _spline_forms as SF
    import test_rolled_reverse as RR

    from hawk.diff import vjp

    k = _derived("kan_vjp", vjp(SF.kan_rolled("plain", 8), wrt=RR.WRT))
    with pytest.raises(HawkError) as err:
        render(k)
    text = str(err.value)
    assert "'kan_vjp'" in text and "the C target" in text and "in place" in text
