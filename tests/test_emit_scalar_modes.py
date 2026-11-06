# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Three built scalar modes, and what each one changes in the emitted text.

``float64`` and ``float32`` must DIFFER (a built mode that were a no-op would
be decorative; the comparison drops the banner comment, which names the mode
and would make any two modes 'differ' for free). ``banded`` stores 8-byte
cells and computes on the three-limb ``Band``, so its unit carries the
banded headers and three writer rules: a Real literal is three exact limbs, a
store into a plane packs (``Real(value)``), and a named rank-0 Real local is
a ``Band``. Each rule is held against the float64 text of the same kernel,
so a rule that stopped firing shows up as the float64 spelling."""

from __future__ import annotations

from fractions import Fraction

import pytest
from _emitted import cases

import hawk
import hawk.math as m
from hawk.emit import BANDED, FLOAT32, FLOAT64, SCALAR_MODES, literal_table, render_source, scalar_mode
from hawk.emit.writer_aether import band_literal
from hawk.ir import HawkError
from hawk.targets import TARGETS


def test_the_seam_declares_exactly_the_three_built_modes():
    assert sorted(SCALAR_MODES) == ["banded", "float32", "float64"]
    assert scalar_mode("float64") is FLOAT64
    assert scalar_mode("float32") is FLOAT32
    assert scalar_mode("banded") is BANDED


def test_both_float_modes_emit_and_differ():
    for name, sinks, walk in cases():
        for target_id, target in _targets():
            f64 = render_source(name, sinks, walk, target, mode=FLOAT64).text
            f32 = render_source(name, sinks, walk, target, mode=FLOAT32).text
            assert _substance(f64) != _substance(f32), (
                f"float64 and float32 emitted IDENTICAL text for {name!r} on "
                f"{target_id} -- a built scalar mode that changes nothing is "
                "decorative. (The comparison drops the banner comment, which "
                "names the mode and would make any two modes 'differ' for free.)"
            )
            assert "using Real = double;" in f64 and "using Real = float;" in f32


def test_an_unknown_mode_is_refused_naming_the_whole_vocabulary():
    with pytest.raises(HawkError) as excinfo:
        scalar_mode("posit8")
    assert "banded" in str(excinfo.value), (
        "an unknown mode's refusal must show the seam's full vocabulary")


@hawk.kernel
def kq(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    t = x * x * 0.5 + y
    out = m.exp(t) / (y + 1.5) - m.sin(x) * 0.25


@hawk.kernel
def shared(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    t = x * x + y
    out = m.abs(t) / (t + 1.0)


@pytest.mark.parametrize("target_id", ["cuda"])
def test_a_banded_unit_stores_cells_and_includes_the_banded_headers(target_id):
    text = _unit(kq, target_id, BANDED)
    assert "using Real = aether::banded::BandedReal;" in text
    assert "#include <aether/banded/banded.h>\n#include <aether/banded/BandedRealOps.h>\n" in text
    assert "aether/banded/" not in _unit(kq, target_id, FLOAT64)


@hawk.kernel
def literals(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = (x * 0.1 + 3.0 * x) + (x / 3.0) + (x / 4.0) + (x * 1.0) + (x + 0.0) + (x * -1.0) + (2.0 + x) - 0.75


@pytest.mark.parametrize("target_id", ["cuda"])
def test_a_banded_literal_is_classified_by_its_exact_value_and_a_store_packs(target_id):
    banded, f64 = _unit(kq, target_id, BANDED), _unit(kq, target_id, FLOAT64)
    for value in (0.5, 0.25):                    # powers of two scale exactly
        assert f"static_cast<Real>({value})" in f64
        assert "scalePow2f(" in banded and f"{value}f)" in banded
    assert "addRawS(psc_y_i, 1.5f)" in banded          # one limb: a float-exact literal
    assert "static_cast<Real>(" not in _body(banded)
    assert "mut_out[i] = Real((" in banded and "mut_out[i] = ((" in f64
    text = _body(_unit(literals, target_id, BANDED))
    assert "psc_x_i * aether::banded::Band(0.10000000149011612f, " in text                   # three limbs
    assert "mulRawS(psc_x_i, 3.0f)" in text and "scalePow2f(psc_x_i, 0.25f)" in text
    assert "psc_x_i * aether::banded::Band(0.3333333432674408f, " in text                    # x / 3.0
    assert "addRawS(psc_x_i, 2.0f)" in text and "(-psc_x_i)" in text and ", -0.75f)" in text
    assert "Band(1.0f" not in text and "Band(0.0f" not in text and ") + psc_x_i) + psc_x_i)" in text


@hawk.kernel
def order_reader(a: hawk.Scalar, b: hawk.Scalar, c: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    t = a * b
    out = m.maximum(t, c) + t


@hawk.kernel
def stepping(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    acc = x
    for _k in range(4):
        acc = acc * 0.5 + m.sin(acc) * x
    out = acc


def test_a_banded_local_is_raw_unless_its_reader_needs_the_limbs_in_order():
    text = _body(_unit(order_reader, "cuda", BANDED))
    assert "const aether::banded::Band t2 = (psc_a[i].eval() * psc_b[i].eval());" in text       # fmax reads in order
    raw = _body(_unit(kq, "cuda", BANDED))
    assert "const auto" not in raw and "const aether::banded::Band psc_x_i = psc_x[i].eval();" in raw   # one decode


def test_a_banded_loop_carry_is_a_band_and_is_encoded_once_after_the_loop():
    text = _body(_unit(stepping, "cuda", BANDED))
    assert "aether::banded::Band hawk_v2_0_acc = psc_x_i;" in text
    assert "const auto tb0 = aether::banded::detail::scalePow2f(hawk_v2_0_acc, 0.5f);" in text
    assert "mut_out[i] = Real(hawk_v2_0_acc);" in text


@hawk.kernel
def sin_cos(x: hawk.Scalar, a: hawk.Scalar, b: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.sin(x) * a + m.cos(x) * b


@hawk.kernel
def products(x: hawk.Scalar, y: hawk.Scalar, z: hawk.Scalar, w: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = x * y + z * w


def test_the_sin_and_cos_of_one_operand_share_one_banded_sincos():
    text = _body(_unit(sin_cos, "cuda", BANDED))
    assert text.count("aether::math::sincos(") == 1 and "aether::math::sin(" not in text
    assert "aether::math::cos(" not in text
    assert "(hawk_sin0 * psc_a[i].eval()) + (hawk_cos0 * psc_b[i].eval())" in text
    assert "sincos" not in _body(_unit(sin_cos, "cuda", FLOAT64))


def test_a_banded_statement_of_products_is_one_expression_with_no_band_local():
    text = _body(_unit(products, "cuda", BANDED))
    assert "aether::banded::Band t" not in text and "const auto" not in text
    assert "Real(((psc_x[i].eval() * psc_y[i].eval()) + (psc_z[i].eval() * psc_w[i].eval())))" in text


@hawk.kernel
def fused(x: hawk.Scalar, y: hawk.Scalar, z: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = x * y * z + y + (z + x * y * x)


@hawk.kernel
def raw_addend(x: hawk.Scalar, y: hawk.Scalar, z: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = x * y * z + (x + y)


def test_a_product_and_a_normalised_addend_fuse_in_either_order_and_a_raw_addend_does_not():
    text = _body(_unit(fused, "cuda", BANDED))
    assert text.count("aether::banded::detail::fma(") == 2
    assert "detail::fma(t2, psc_z_i, aether::banded::Band(psc_y_i))" in text and "detail::fma(t2, psc_x_i, aether::banded::Band(psc_z_i))" in text
    raw = _body(_unit(raw_addend, "cuda", BANDED))
    assert "fma(" not in raw


@hawk.kernel
def invariant(x: hawk.Scalar, k: hawk.Scalar, v: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    acc = x
    for _j in range(4):
        acc = acc * k + v
    out = acc


def test_a_loop_invariant_plane_element_is_decoded_once_before_the_loop():
    text = _body(_unit(invariant, "cuda", BANDED)).rpartition("#else")[2]
    loop = text.index("for (Int ")
    assert text.count("psc_k[i].eval()") == 1 and text.index("psc_k[i].eval()") < loop
    assert text.index("const aether::banded::Band psc_k_i = ") < loop
    assert "eval()" not in text[loop:]


@hawk.kernel
def twins(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.sin(x) * y - m.exp(y) * 0.5 + (x / y) * x + m.abs(m.cos(y))


def test_a_transcendental_read_by_a_chain_operator_takes_its_raw_exit_and_any_other_reader_the_facade():
    text = _body(_unit(twins, "cuda", BANDED))
    assert "detail::sinRaw(aether::banded::Band(psc_x_i)) * psc_y_i" in text                   # read by a product
    assert "scalePow2f(aether::banded::detail::expRaw(aether::banded::Band(psc_y_i)), 0.5f)" in text    # by a scale
    assert "aether::math::abs(aether::math::cos(psc_y_i))" in text and "cosRaw" not in text   # abs reads the limbs
    kq_text = _body(_unit(kq, "cuda", BANDED))
    assert "expRaw(" in kq_text and "sinRaw(" in kq_text                                      # a div numerator; a scale
    assert "Raw(" not in _body(_unit(twins, "cuda", FLOAT64))


def test_a_named_banded_local_is_a_band_and_a_float64_one_stays_auto():
    banded, f64 = _unit(shared, "cuda", BANDED), _unit(shared, "cuda", FLOAT64)
    named = [line.split("=")[0].split()[-1] for line in _body(f64).splitlines() if "const auto" in line]
    assert named, "the control kernel names no local: the row would hold nothing"
    for name in named:
        assert f"const aether::banded::Band {name} = " in banded
    assert "const auto" not in _body(banded)


def test_a_banded_literal_splits_exactly_and_refuses_what_three_limbs_cannot_hold():
    for value in (0.1, -1.0 / 3.0, 3.141592653589793, 2.0 ** -90 * 1.2345, 1e30):
        limbs = band_literal(value).removeprefix("aether::banded::Band(").removesuffix(")").split(", ")
        assert len(limbs) == 3 and all(limb.endswith("f") for limb in limbs)
        assert sum(Fraction(float(limb[:-1])) for limb in limbs) == Fraction(value), value
    assert band_literal(-0.0) == "aether::banded::Band(-0.0f, 0.0f, 0.0f)"
    for value in (1e300, 2.0 ** 130, 1e-300):
        with pytest.raises(HawkError, match=r"outside the banded range.*\[2\^-94, 2\^126\)"):
            band_literal(value)


def test_the_literal_table_leaves_a_banded_unit_as_the_writer_spelled_it():
    """The device literal table is a float64 rewrite; a banded unit's literals are already the writer's
    three-limb spelling and go through unchanged."""
    banded = render_source("kq", kq.sinks, kq.walk, TARGETS["cuda"], mode=BANDED)
    assert literal_table(banded) is banded
    f64 = render_source("kq", kq.sinks, kq.walk, TARGETS["cuda"], mode=FLOAT64)
    assert literal_table(f64).text != f64.text


def _unit(kernel, target_id: str, mode) -> str:
    return render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target_id], mode=mode).text


def _body(text: str) -> str:
    """The entry onward: the prelude's helpers carry their own spellings."""
    return text[text.index("// The self-check exports"):] if "// The self-check exports" in text else text


def _substance(text: str) -> str:
    """The unit without its banner comment -- the banner NAMES the mode, so a
    whole-text comparison would report any two modes as different for free."""
    return text.split("\n", 2)[2]


def _targets():
    return sorted((t, row) for t, row in TARGETS.items() if row.wrapper)
