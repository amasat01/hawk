# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C target's verify pass: budgets, index sites, fp-domain sites and the
interface contract, read off real traced kernels before anything renders.

Every kernel here is planted through the authoring API (or is a corpus /
deployment fixture), so what the rows observe is what an author's kernel
actually produces. ``apply_caps`` is the one door a profile's caps refuse
through; ``ir_checks`` itself never refuses.
"""

from __future__ import annotations

import re

import _deployable as D
import _kernels as K
import pytest

import hawk
import hawk.math as hm
from hawk import Accum, Index, Mutable, Param, Scalar, Terminated, Vector, WideOut
from hawk.diff import vjp
from hawk.embedded.profile import EmbeddedProfile
from hawk.embedded.verify import (
    UNSUPPORTED,
    apply_caps,
    index_status,
    ir_checks,
    positions,
    site_status,
)
from hawk.ext import DEFAULT_KIND
from hawk.ir import HawkError, canonical
from hawk.ir.loop_nodes import Loop
from hawk.ir.nodes import At


def _report(kernel, profile=None):
    return ir_checks(kernel.name, kernel.sinks, kernel.walk, kernel.kind,
                     profile or EmbeddedProfile())


# --------------------------------------------------------------------------- #
# The profile.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field, bad, allowed", [
    ("fp_domain", "warn", "record"),
    ("index_policy", "wrap", "clamp"),
    ("c_std", "c89", "c99"),
])
def test_the_profile_refuses_an_unknown_policy_naming_the_field(field, bad, allowed):
    with pytest.raises(HawkError, match=rf"{field}.*{bad!r}.*{allowed!r}"):
        EmbeddedProfile(**{field: bad})


@pytest.mark.parametrize("field", ["max_trips", "max_ops", "max_stack_bytes"])
@pytest.mark.parametrize("bad", [-1, 2.5, "10", True])
def test_the_profile_refuses_a_cap_that_is_not_a_non_negative_int(field, bad):
    with pytest.raises(HawkError, match=rf"{field}.*non-negative int"):
        EmbeddedProfile(**{field: bad})


def test_a_zero_cap_is_a_cap():
    assert EmbeddedProfile(max_trips=0, max_ops=0, max_stack_bytes=0).max_trips == 0


def test_the_profile_repr_is_deterministic():
    assert repr(EmbeddedProfile(max_trips=8)) == repr(EmbeddedProfile(max_trips=8))
    assert repr(EmbeddedProfile()) != repr(EmbeddedProfile(c_std="c11"))


# --------------------------------------------------------------------------- #
# a. Bounds and budgets.
# --------------------------------------------------------------------------- #
@hawk.kernel
def long_loop(x: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(100000):
        s = s * 0.5 + 1.0
    y = s


@hawk.kernel
def nested_loops(x: Scalar, y: Mutable[Scalar]):
    s = x
    for _i in range(4):
        t = s
        for _j in range(3):
            t = t * 0.5 + 1.0
        s = t + s
    y = s


def test_a_hundred_thousand_trip_loop_exceeds_a_cap():
    report = _report(long_loop)
    assert report["budgets"]["max_trips"] == 100000
    apply_caps("long_loop", report, EmbeddedProfile(max_trips=100000))
    with pytest.raises(HawkError) as err:
        apply_caps("long_loop", report, EmbeddedProfile(max_trips=1000))
    message = str(err.value)
    for needle in ("'long_loop'", "node loop", "(loop)", "budget.trips", "the C target"):
        assert needle in message


def test_nested_loop_trips_multiply():
    assert _report(nested_loops)["budgets"]["max_trips"] == 4 * (1 + 3)


@hawk.kernel
def twenty_ops(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    s = x * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    s = s * a + b
    y = s * a + b             # ten mul + ten add, straight line


def test_ops_cap_uses_kernel_cost():
    report = _report(twenty_ops)
    ops = report["budgets"]["ops"]
    assert ops["arith64"] >= 20
    assert set(ops) >= {"arith32", "arith64", "special32", "special64", "loads",
                        "gathers", "stores"}
    with pytest.raises(HawkError, match=r"'twenty_ops'.*budget\.ops"):
        apply_caps("twenty_ops", report, EmbeddedProfile(max_ops=5))
    apply_caps("twenty_ops", report, EmbeddedProfile(max_ops=50))
    assert report["budgets"]["fp64_ops"] >= 20


@hawk.kernel
def recurrence(x: Scalar, g: Param, y: Mutable[Scalar]):
    s = x
    for _k in range(64):
        s = hm.tanh(g * s)
    y = s


def test_tape_bytes_feed_the_memory_estimate():
    sinks = vjp(recurrence)
    walk = canonical(sinks)
    report = ir_checks("recurrence_vjp", sinks, walk, DEFAULT_KIND, EmbeddedProfile())
    static = report["budgets"]["static_bytes"]
    (site,) = report["indices"]
    assert (site["kind"], site["status"], site["length"]) == ("tape_read", "proven", 64)
    taped_slots = 1                      # the forward loop's one f64 carry
    assert static["tapes"] == 64 * 8 * taped_slots
    # forward loop: one f64 carry; reverse loop: two f64 carries
    assert static["carries"] == 3 * 8
    assert static["estimate"] == static["tapes"] + static["carries"]
    assert static["temporaries"] is None and static["total"] is None
    with pytest.raises(HawkError, match=r"budget\.stack"):
        apply_caps("recurrence_vjp", report, EmbeddedProfile(max_stack_bytes=256))
    apply_caps("recurrence_vjp", report, EmbeddedProfile(max_stack_bytes=1024))


def test_a_loop_free_kernel_has_no_trips_and_no_static_bytes():
    report = _report(K.drag)
    assert report["budgets"]["max_trips"] == 0
    assert report["budgets"]["loops"] == []
    apply_caps("drag", report, EmbeddedProfile(max_trips=0, max_stack_bytes=0))
    assert report["budgets"]["static_bytes"]["estimate"] == 0
    assert report["unsupported_checked"] == sorted(UNSUPPORTED)


# --------------------------------------------------------------------------- #
# b. Index sites.
# --------------------------------------------------------------------------- #
def test_table_read_is_clamped_and_refusable():
    report = _report(K.gather)
    site = report["indices"][0]
    assert site["kind"] == "table_read"
    assert site["status"] == "clamped"
    assert site["length"] == "lut_table_len"
    assert index_status(report, site["node"]) == "clamped"
    apply_caps("gather", report, EmbeddedProfile())        # clamp is the default
    with pytest.raises(HawkError) as err:
        apply_caps("gather", report, EmbeddedProfile(index_policy="refuse"))
    message = str(err.value)
    assert f"node {site['node']} " in message
    assert "index.unproven" in message and "'gather'" in message


@hawk.kernel
def component_sum(v: Vector[3], y: Mutable[Scalar]):
    s = 0.0
    for k in range(3):
        s = s + v[k]
    y = s


def test_component_at_with_an_in_range_loop_index_is_proven():
    report = _report(component_sum)
    sites = [s for s in report["indices"] if s["kind"] == "component_at"]
    assert len(sites) == 1
    assert sites[0]["status"] == "proven"
    assert sites[0]["length"] == 3
    assert isinstance(sites[0]["node"], str) and sites[0]["node"].startswith("loop")
    assert index_status(report, sites[0]["node"]) == "proven"
    apply_caps("component_sum", report, EmbeddedProfile(index_policy="refuse"))


def test_a_site_is_found_from_its_node_object():
    report = _report(K.gather)
    walk, sinks = K.gather.walk, K.gather.sinks
    (at,) = [n for n in walk.order if isinstance(n, At)]
    (site,) = report["indices"]
    assert positions(sinks)[id(at)] == site["node"]
    assert site_status(report, sinks, at) == "clamped"
    with pytest.raises(HawkError, match="not an index site"):
        site_status(report, sinks, walk.order[-1])

    report = _report(component_sum)
    walk, sinks = component_sum.walk, component_sum.sinks
    (i, loop), = [(i, n) for i, n in enumerate(walk.order) if isinstance(n, Loop)]
    k, node = next((k, n) for k, n in enumerate(loop.body_nodes())
                   if n.kind == "component_at")
    assert positions(sinks)[id(node)] == f"loop{i}.{k}"
    assert site_status(report, sinks, node) == "proven"
    every = positions(sinks)
    assert len(every) == len(walk.order) + len(loop.body_nodes())


@hawk.kernel
def switched(k: Index, b0: Scalar, b1: Scalar, y: Mutable[Scalar]):
    y = hm.dispatch(k, [b0, b1], policy="switch")


def test_a_scattered_sink_and_a_dispatch_selector_are_clamped():
    report = _report(D.scatter)
    (site,) = report["indices"]
    assert (site["kind"], site["status"], site["length"]) == (
        "scatter", "clamped", "aout_acc_len")
    report = _report(switched)
    (site,) = report["indices"]
    assert (site["kind"], site["status"], site["length"]) == (
        "dispatch", "clamped", 2)


# --------------------------------------------------------------------------- #
# c. Floating-point domain.
# --------------------------------------------------------------------------- #
@hawk.kernel
def speed_of(v: Vector[3], y: Mutable[Scalar]):
    y = hm.sqrt(hm.dot(v, v))


@hawk.kernel
def root_of(x: Scalar, y: Mutable[Scalar]):
    y = hm.sqrt(x)


@hawk.kernel
def softened(x: Scalar, y: Mutable[Scalar]):
    y = hm.log(1.0 + x * x) / (hm.norm(hm.vec(x, x, x)) + 0.5)


def test_fp_domain_sqrt_of_dot_is_safe_and_sqrt_of_a_scalar_is_unknown():
    safe = _report(speed_of)
    (site,) = safe["fp_domain"]
    assert (site["op"], site["status"]) == ("sqrt", "safe")
    apply_caps("speed_of", safe, EmbeddedProfile(fp_domain="refuse"))

    unknown = _report(root_of)
    (site,) = unknown["fp_domain"]
    assert (site["op"], site["status"]) == ("sqrt", "unknown")
    apply_caps("root_of", unknown, EmbeddedProfile())      # record is the default
    with pytest.raises(HawkError) as err:
        apply_caps("root_of", unknown, EmbeddedProfile(fp_domain="refuse"))
    message = str(err.value)
    assert "fp.domain" in message and "'root_of'" in message
    assert f"node {site['node']} " in message


def test_positive_forms_make_log_and_div_safe():
    report = _report(softened)
    ops = {s["op"]: s["status"] for s in report["fp_domain"]}
    assert ops == {"log": "safe", "div": "safe"}


@hawk.kernel
def recip_exp(x: Scalar, y: Mutable[Scalar]):
    y = 1.0 / hm.exp(x)


@hawk.kernel
def log_exp(x: Scalar, y: Mutable[Scalar]):
    y = hm.log(hm.exp(x))


@hawk.kernel
def rsqrt_exp(x: Scalar, y: Mutable[Scalar]):
    y = hm.rsqrt(hm.exp(x))


@hawk.kernel
def log_sqrt(x: Scalar, y: Mutable[Scalar]):
    y = hm.log(hm.sqrt(x))


@hawk.kernel
def root_root(x: Scalar, y: Mutable[Scalar]):
    y = hm.sqrt(hm.sqrt(x))


@hawk.kernel
def lorentzian(x: Scalar, y: Mutable[Scalar]):
    y = 1.0 / (1.0 + x * x)


@pytest.mark.parametrize("kernel, op", [
    (recip_exp, "div"), (log_exp, "log"), (rsqrt_exp, "rsqrt")],
    ids=["1/exp", "log(exp)", "rsqrt(exp)"])
def test_exp_is_nonneg_not_positive(kernel, op):
    """``exp`` underflows to +0 for a large negative operand: a divisor, a
    ``log`` or an ``rsqrt`` over it is not provably in its domain."""
    report = _report(kernel)
    (site,) = report["fp_domain"]
    assert (site["op"], site["status"]) == (op, "unknown")
    assert site["reason"].endswith("its sign is nonneg")
    with pytest.raises(HawkError, match=r"fp\.domain"):
        apply_caps(kernel.name, report, EmbeddedProfile(fp_domain="refuse"))


def test_sqrt_of_an_unknown_sign_is_unknown():
    """A negative operand gives NaN, which is no sign at all."""
    report = _report(log_sqrt)
    sites = {s["op"]: s["status"] for s in report["fp_domain"]}
    assert sites == {"sqrt": "unknown", "log": "unknown"}
    outer = [s for s in _report(root_root)["fp_domain"] if s["status"] == "safe"]
    assert outer == []


def test_the_sound_positive_forms_stay_safe():
    (site,) = _report(lorentzian)["fp_domain"]
    assert (site["op"], site["status"]) == ("div", "safe")
    apply_caps("lorentzian", _report(lorentzian), EmbeddedProfile(fp_domain="refuse"))
    (site,) = _report(speed_of)["fp_domain"]
    assert (site["op"], site["status"]) == ("sqrt", "safe")


@hawk.kernel
def float_equal(x: Scalar, z: Scalar, y: Mutable[Scalar]):
    y = hm.where(x == z, x, z)


def test_float_equality_sites_are_listed():
    report = _report(float_equal)
    eq = [i for i, n in enumerate(float_equal.walk.order) if n.kind == "eq"]
    assert report["fp_equality_sites"] == eq
    assert _report(K.drag)["fp_equality_sites"] == []


@hawk.kernel
def int_divide(a: Index, b: Index, y: Mutable[Index]):
    y = a // b + a % b


def test_int_division_sites_are_listed():
    report = _report(int_divide)
    want = [i for i, n in enumerate(int_divide.walk.order)
            if n.kind in ("floordiv", "mod")]
    assert len(want) == 2
    assert report["int_division_sites"] == want
    assert report["fp_domain"] == []


@hawk.kernel
def index_add(a: Index, b: Index, y: Mutable[Index]):
    y = a + b


@hawk.kernel
def index_mix(a: Index, b: Index, s: Mutable[Index], d: Mutable[Index],
              p: Mutable[Index], n: Mutable[Index], sh: Mutable[Index]):
    s = a + b
    d = a - b
    p = a * b
    n = -a
    sh = (a << b) >> 1


def test_int_ub_sites_are_the_index_add_sub_mul_neg_and_true_div():
    """Only an Index ``+ - *``, negation and true division ``/`` part the two
    builds: the C++ spells them infix (signed overflow, a zero divisor and
    ``MIN / -1`` undefined), the C defines them through ``hawk_i64_*``.
    Int32/UInt32 arithmetic and every Index ``//``, ``%`` and shift go
    through the host's own helpers too, so they are no site."""
    report = _report(index_add)
    (add,) = [i for i, n in enumerate(index_add.walk.order) if n.kind == "add"]
    assert report["int_ub_sites"] == [add]
    report = _report(index_mix)
    kinds = sorted(index_mix.walk.order[p].kind for p in report["int_ub_sites"])
    assert kinds == ["add", "mul", "neg", "sub"]
    # the divisions are int_division_sites, never UB sites
    report = _report(int_divide)
    assert report["int_ub_sites"] == [
        i for i, n in enumerate(int_divide.walk.order) if n.kind == "add"]
    import test_int_planes as I

    assert _report(I.ops_i32)["int_ub_sites"] == []
    assert _report(I.ops_u32)["int_ub_sites"] == []
    # a real `x * x` is no site
    assert _report(K.drag)["int_ub_sites"] == []


@hawk.kernel
def index_true_div(a: Index, b: Index, y: Mutable[Index]):
    y = a / b


def test_an_index_true_division_is_a_ub_site_as_well_as_a_division_site():
    """``a / b`` on two ``Index`` values is infix in the C++ (undefined at
    ``b == 0`` and at ``MIN / -1``) and ``hawk_i64_div`` in the C (0 and MIN
    there), so it is a ``defined_where_cpp_is_ub`` site; ``//`` and ``%`` go
    through the host's own helpers and stay division sites only."""
    report = _report(index_true_div)
    (div,) = [i for i, n in enumerate(index_true_div.walk.order) if n.kind == "div"]
    assert report["int_division_sites"] == [div]
    assert report["int_ub_sites"] == [div]
    assert _report(int_divide)["int_ub_sites"] == [
        i for i, n in enumerate(int_divide.walk.order) if n.kind == "add"]


# --------------------------------------------------------------------------- #
# d. The interface contract.
# --------------------------------------------------------------------------- #
@hawk.kernel
def accumulate_in_place(x: Scalar, y: Mutable[Scalar]):
    y = y + x


@hawk.kernel
def own_partials(x: Scalar, acc: Accum[Scalar]):
    acc.add(x * x)


def _rows(kernel):
    return {row["field"]: row for row in _report(kernel)["interface"]}


def test_interface_directions():
    rows = _rows(D.diagnostic)
    assert list(rows) == ["mut_y", "psc_x", "trm_terminated"]
    assert rows["trm_terminated"]["direction"] == "in"
    assert rows["trm_terminated"]["init_by_caller"] is True
    # the store sits inside the default guard: a masked sample keeps the prior
    assert rows["mut_y"]["direction"] == "inout"
    assert rows["mut_y"]["init_by_caller"] is True
    assert rows["mut_y"]["reason"] == (
        "masked samples keep the prior value (guard: trm_terminated)")
    assert rows["psc_x"]["layout"] == "scalar"

    rows = _rows(accumulate_in_place)
    assert rows["mut_y"]["direction"] == "inout"
    assert rows["mut_y"]["init_by_caller"] is True

    rows = _rows(D.scatter)
    acc = rows["aout_acc"]
    assert acc["direction"] == "inout"
    assert acc["init_by_caller"] is True
    assert acc["runtime_length"] is True

    rows = _rows(own_partials)
    acc = rows["aout_acc"]
    assert acc["layout"] == "partials"
    assert acc["direction"] == "out"
    assert "partial" in acc["reason"]


@hawk.kernel
def guarded_partials(x: Scalar, terminated: Terminated, acc: Accum[Scalar]):
    acc.add(x * x)


@hawk.kernel
def guarded_vector(v: Vector[3], terminated: Terminated, y: Mutable[Vector[3]]):
    y = v * 2.0


@hawk.kernel
def guarded_wide(x: Scalar, terminated: Terminated, w: WideOut[Scalar]):
    w = x * 2.0


def _guarded_rows(kernel, kind=None):
    report = ir_checks(kernel.name, kernel.sinks, kernel.walk,
                       kind or kernel.kind, EmbeddedProfile())
    return {row["field"]: row for row in report["interface"]}


_KEEP = "masked samples keep the prior value (guard: {})"


@pytest.mark.parametrize("kernel, field", [
    (D.diagnostic, "mut_y"), (guarded_vector, "mut_y"), (guarded_wide, "wout_w")])
def test_a_guarded_own_column_store_is_inout_and_init_by_caller(kernel, field):
    """A masked sample never reaches the store, so the host keeps the prior
    value: the caller must hand over initialised memory."""
    row = _guarded_rows(kernel)[field]
    assert (row["direction"], row["init_by_caller"]) == ("inout", True)
    assert row["reason"] == _KEEP.format("trm_terminated")


def test_guarded_partials_are_zeroed_by_the_caller():
    row = _guarded_rows(guarded_partials)["aout_acc"]
    assert (row["direction"], row["init_by_caller"], row["layout"]) == (
        "inout", True, "partials")
    assert row["reason"] == (
        "one partial per sample, length n_samples; masked samples keep theirs "
        "(guard: trm_terminated): the caller zeroes it, then reduces")


def test_the_reason_names_every_bound_mask_and_a_mask_free_kind_is_unchanged():
    rows = _guarded_rows(D.diagnostic_two, D.TWO_MASKS)
    assert rows["mut_y"]["reason"] == _KEEP.format("trm_terminated, trm_rejected")
    # the default guard reads `terminated` alone, even with `rejected` bound
    rows = _guarded_rows(D.diagnostic_two)
    assert rows["mut_y"]["reason"] == _KEEP.format("trm_terminated")
    rows = _guarded_rows(D.diagnostic, D.MASK_FREE)
    assert (rows["mut_y"]["direction"], rows["mut_y"]["init_by_caller"]) == (
        "out", False)
    assert rows["mut_y"]["reason"] == "written every sample"
    # a guard naming a mask the kernel does not bind never fires
    rows = _guarded_rows(own_partials)
    assert (rows["aout_acc"]["direction"], rows["aout_acc"]["init_by_caller"]) == (
        "out", False)


def test_a_finishing_kernel_falls_under_the_same_rule():
    """The finish mask is bound in the ``terminated`` role, so the own-column
    store is guarded by it like any other."""
    import test_finish_own_sample as F

    rows = _guarded_rows(F.branch_finish)
    assert rows["trm_terminated"]["direction"] == "inout"
    assert (rows["mut_y"]["direction"], rows["mut_y"]["init_by_caller"]) == (
        "inout", True)
    assert rows["mut_y"]["reason"] == _KEEP.format("trm_terminated")


def test_every_row_carries_the_bound_guard_masks():
    """``guard`` is structural: the bound mask fields (as the renderer's
    ``_guard_slot`` computes them) on EVERY row, empty when unguarded."""
    for kernel, kind, want in [
            (D.diagnostic, None, ["trm_terminated"]),
            (D.diagnostic_two, D.TWO_MASKS, ["trm_terminated", "trm_rejected"]),
            (D.diagnostic_two, None, ["trm_terminated"]),
            (D.diagnostic, D.MASK_FREE, []),
            (own_partials, None, []),
            (K.drag, None, [])]:
        rows = _guarded_rows(kernel, kind)
        assert all(row["guard"] == want for row in rows.values()), (kernel.name, kind)


def test_the_header_selects_the_guard_lines_from_the_guard_field():
    """The header's per-field guard line and the guarded partials line are
    chosen by ``row["guard"]``, never by the reason text."""
    from hawk.embedded import abi, render
    from hawk.emit import FLOAT64

    for kernel, field in [(D.diagnostic, "mut_y"), (guarded_partials, "aout_acc")]:
        report = render(kernel).report
        reworded = {**report, "interface": [
            {**r, "reason": "reworded"} for r in report["interface"]]}
        text = "\n".join(abi._contract(kernel.name, reworded, FLOAT64))
        assert "(guard: trm_terminated)" in text and f"*   {field}" in text
        unguarded = {**report, "interface": [
            {**r, "guard": []} for r in report["interface"]]}
        text = "\n".join(abi._contract(kernel.name, unguarded, FLOAT64))
        assert "guard:" not in text


def test_the_header_contract_says_a_guarded_plane_is_initialised():
    from hawk.embedded import render

    header = render(guarded_partials).header
    assert re.search(r"\*   aout_acc +accum_out +inout +f64 +\(\) +yes\n", header)
    assert ("aout_acc holds ONE PARTIAL PER SAMPLE (length n_samples); masked "
            "samples keep theirs (guard: trm_terminated): the caller zeroes it, "
            "then reduces.") in header
    assert "the caller reduces it." not in header
    header = render(D.diagnostic).header
    assert re.search(r"\*   mut_y +mutable +inout +f64 +\(\) +yes\n", header)
    assert ("mut_y: masked samples keep the prior value (guard: trm_terminated); "
            "the caller initialises it.") in header
    # keyed on the structural guard: a prior-read plane under the guard is
    # skipped at a masked sample too, and says so
    import test_finish_own_sample as F

    header = render(F.osc_finish).header
    for field in ("mut_t", "mut_v", "mut_x"):
        assert (f"{field}: masked samples keep the prior value (guard: "
                "trm_terminated); the caller initialises it.") in header
    # an unguarded partial plane keeps the plain reduce line
    assert "aout_acc holds ONE PARTIAL PER SAMPLE (length n_samples); the caller " \
        "reduces it." in render(own_partials).header


def test_interface_rows_carry_dtype_shape_and_layout():
    rows = _rows(K.drag)
    assert rows["mut_out"] == {
        "field": "mut_out", "role": "mutable", "name": "out", "direction": "out",
        "dtype": "f64", "shape": [3], "layout": "soa", "runtime_length": False,
        "init_by_caller": False, "guard": [], "reason": rows["mut_out"]["reason"]}
    assert rows["uni_rho"]["layout"] == "value"
    assert rows["uni_rho"]["direction"] == "in"
    rows = _rows(K.gather)
    assert rows["lut_table"]["runtime_length"] is True


@hawk.kernel(steps=1, time="t", time_step="dt")
def taped_step(dt: Param, t_final: Scalar, terminated: Terminated,
               x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x - dt * x
    t = t + dt
    terminated = t >= t_final


def test_the_tape_twin_row_is_clamped_with_its_row_count():
    """The twin's row is ``a - (a / W) * W`` over the caller-writable
    ``applied_steps`` plane: negative for a negative ``a``, so not provable."""
    from hawk.diff import tape

    twin = tape(taped_step, 8)
    (site,) = _report(twin)["indices"]
    assert (site["kind"], site["status"], site["length"]) == ("tape_write", "clamped", 8)
