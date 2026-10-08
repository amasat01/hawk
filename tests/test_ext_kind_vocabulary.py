# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Kind`` vocabulary, the defining-frame annotation scope, and
``Output.named``/``Output.returned``.

A kind's vocabulary supplies a declaration for a parameter the signature does
not itself annotate; precedence is not "own wins" (an own annotation must
agree with the vocabulary's, never override it), a ``terminated``-form entry
binds whether or not the signature names it, and every other entry binds only
when named. The annotation scope is the frame that DEFINED the traced
function, found by walking outward from the trace call rather than reading a
fixed stack position, so a wrapper decorator or a kernel-factory closure both
resolve a string annotation correctly."""

from __future__ import annotations

import linecache

import pytest

from hawk import Mutable, Scalar, Terminated, Vector, kernel
from hawk.emit import render_body
from hawk.ext import DEFAULT_KIND, Kind, Output
from hawk.ir import HawkError

VOCAB = {"terminated": Terminated, "scale": Scalar}


@Kind("vocab_kind", vocabulary=VOCAB)
def _vocab_kernel(x: Scalar, terminated, scale, y: Mutable[Scalar]):
    y = x * scale


@kernel
def _annotated_twin(x: Scalar, terminated: Terminated, scale: Scalar,
                    y: Mutable[Scalar]):
    y = x * scale


def test_vocabulary_kernel_matches_its_fully_annotated_twin():
    assert _vocab_kernel.arg_spec == _annotated_twin.arg_spec
    assert _vocab_kernel.walk.digest == _annotated_twin.walk.digest
    a = render_body(_vocab_kernel.sinks, _vocab_kernel.walk,
                    kind=_vocab_kernel.kind).text
    b = render_body(_annotated_twin.sinks, _annotated_twin.walk).text
    assert a == b


MASK_VOCAB = {"terminated": Terminated}


@Kind("mask_vocab", vocabulary=MASK_VOCAB)
def _omits_terminated(x: Scalar, y: Mutable[Scalar]):
    """The signature never names ``terminated`` — the vocabulary's
    `terminated`-form entry binds it anyway (unconditional, item 4)."""
    y = x + 1.0


def test_unconditional_terminated_binds_even_when_the_signature_omits_it():
    assert ("terminated", "terminated") in _omits_terminated.arg_spec
    text = render_body(_omits_terminated.sinks, _omits_terminated.walk,
                       kind=_omits_terminated.kind).text
    assert "if (!trm_terminated[i].eval())" in text


def test_a_conflicting_annotation_refuses():
    from hawk.ext import DATA_ONLY

    K = Kind("conflicting_annotation", vocabulary={"scale": Scalar}, guard=DATA_ONLY)
    with pytest.raises(HawkError, match="disagrees with"):
        @K
        def bad(x: Scalar, scale: Vector[3], y: Mutable[Scalar]):  # noqa: F841
            y = x


def _scale_positional(x: Scalar, scale, y: Mutable[Scalar]):
    y = x * scale


def _scale_keyword_only(x: Scalar, *, scale, y: Mutable[Scalar]):
    y = x * scale


def _scale_annotated_positional(x: Scalar, scale: Scalar, y: Mutable[Scalar]):
    y = x * scale


def _scale_annotated_keyword_only(x: Scalar, *, scale: Scalar, y: Mutable[Scalar]):
    y = x * scale


def _scale_disagreeing_keyword_only(x: Scalar, *, scale: Vector[3], y: Mutable[Scalar]):
    y = x


def _scale_kind():
    from hawk.ext import DATA_ONLY

    return Kind("kwonly_vocabulary", vocabulary={"scale": Scalar}, guard=DATA_ONLY)


@pytest.mark.parametrize("positional, keyword_only", [
    (_scale_positional, _scale_keyword_only),
    (_scale_annotated_positional, _scale_annotated_keyword_only),
])
def test_a_keyword_only_vocabulary_parameter_binds_like_the_positional_form(positional, keyword_only):
    K = _scale_kind()
    a, b = K(positional), K(keyword_only)
    assert a.arg_spec == b.arg_spec
    assert (render_body(a.sinks, a.walk, kind=a.kind).text
            == render_body(b.sinks, b.walk, kind=b.kind).text)


def test_a_keyword_only_vocabulary_parameter_with_a_disagreeing_annotation_refuses():
    with pytest.raises(HawkError, match="disagrees with"):
        _scale_kind()(_scale_disagreeing_keyword_only)


def test_a_guard_mask_outside_a_non_empty_vocabulary_refuses_at_construction():
    from hawk.ext import Guard

    with pytest.raises(HawkError, match="guard mask"):
        Kind("guard_mask_outside_vocabulary", vocabulary={"scale": Scalar},
             guard=Guard("rejected"))


def test_build_with_no_kind_uses_the_kernels_own_kind(tmp_path, cache_dir):
    from hawk.artifact import build
    from hawk.ext import compensated

    OWN_KIND = Kind("own_kind_comp", sink=compensated(into="c"))

    @OWN_KIND
    def own_kind_solo(x: Scalar, y: Mutable[Scalar]):
        y = x

    art = build(own_kind_solo, tmp_path / "own_kind_a", targets=("host",),
               cache_dir=cache_dir)
    text = (art.directory / "own_kind_solo.cpp").read_text()
    assert "hawk_abi::store_compensated(" in text

    other = Kind("other_kind")
    with pytest.raises(HawkError, match="differing from the kernel's own"):
        build(own_kind_solo, tmp_path / "own_kind_b", targets=("host",), kind=other,
             cache_dir=cache_dir)


def test_every_bare_kernel_gets_the_default_kind():
    import _deployable as D

    for name in ("axpb", "vec3_scale", "scatter", "diagnostic"):
        assert getattr(D, name).kind is DEFAULT_KIND


def test_a_wrapper_decorator_does_not_see_a_shadowed_vocabulary_name():
    """Old behaviour (`sys._getframe(1).f_locals` read directly in ``kernel``)
    grabbed the WRAPPER's own locals, where ``mine`` shadows ``Vector`` with a
    string; the fix walks OUTWARD to the frame that actually wrote
    ``def k(...):`` (the exec'd module below), which never shadowed it."""
    filename = "<shadow_wrapper>"
    src = (
        "from __future__ import annotations\n"
        "from hawk.trace import kernel, Mutable, Vector\n"
        "def mine(fn):\n"
        "    Vector = 'shadow'\n"
        "    return kernel(fn)\n"
        "@mine\n"
        "def k(v: Vector[3], y: Mutable[Vector[3]]):\n"
        "    y = v\n"
    )
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    ns: dict = {}
    exec(compile(src, filename, "exec"), ns)  # noqa: - the row's own fixture
    k = ns["k"]

    @kernel
    def bare(v: Vector[3], y: Mutable[Vector[3]]):
        y = v

    assert k.arg_spec == bare.arg_spec
    assert k.walk.digest == bare.walk.digest


def _make_factory(w):
    @kernel
    def k(v: Vector[w], out: Mutable[Vector[w]]):
        """``w`` is never read in the body — the defining frame (``_make_factory``'s
        own locals) is where the string annotation ``Vector[w]`` resolves it."""
        out = v
    return k


def test_a_kernel_factory_resolves_an_unused_extent_from_the_defining_frame():
    k3, k5 = _make_factory(3), _make_factory(5)
    assert k3.walk.slot_types[("vec_in", "v")].shape == (3,)
    assert k5.walk.slot_types[("vec_in", "v")].shape == (5,)
    assert k3.arg_spec == k5.arg_spec


def test_returned_output_matches_the_named_twin_byte_for_byte():
    RETURNED = Kind("returned_output", output=Output.returned(Vector[3], slot="acc"))

    @RETURNED
    def body_returns(e: Vector[3]):
        return e

    NAMED = Kind("named_output")

    @NAMED
    def body_named(e: Vector[3], acc: Mutable[Vector[3]]):
        acc = e

    assert body_returns.arg_spec == body_named.arg_spec
    assert body_returns.walk.digest == body_named.walk.digest
    a = render_body(body_returns.sinks, body_returns.walk, kind=RETURNED).text
    b = render_body(body_named.sinks, body_named.walk, kind=NAMED).text
    assert a == b


def test_a_parameter_named_like_the_synthesised_slot_refuses():
    K = Kind("returned_slot_collision", output=Output.returned(Scalar, slot="acc"))
    with pytest.raises(HawkError, match="may not also be a parameter name"):
        @K
        def bad(acc: Scalar):
            return acc


def test_a_returned_value_of_the_wrong_rank_refuses():
    K = Kind("returned_wrong_rank", output=Output.returned(Vector[3], slot="acc"))
    with pytest.raises(HawkError):
        @K
        def bad(e: Scalar):
            return e


# --------------------------------------------------------------------------- #
# The class-form spelling of Kind: sugar, one object, two spellings.
# --------------------------------------------------------------------------- #
def test_the_class_form_builds_the_same_kind_the_instance_form_does():
    from hawk.ext import KernelKind

    instance_kind = Kind("class_form_twin", vocabulary={"terminated": Terminated,
                                                        "scale": Scalar})

    class ClassFormTwin(KernelKind, slug="class_form_twin"):
        terminated: Terminated
        scale: Scalar

    assert ClassFormTwin.kind == instance_kind

    @Kind("class_form_twin_instance", vocabulary={"terminated": Terminated,
                                                  "scale": Scalar})
    def instance_spelled(x: Scalar, terminated, scale, y: Mutable[Scalar]):
        y = x * scale

    @ClassFormTwin
    def class_spelled(x: Scalar, terminated, scale, y: Mutable[Scalar]):
        y = x * scale

    assert instance_spelled.arg_spec == class_spelled.arg_spec
    assert instance_spelled.walk.digest == class_spelled.walk.digest
    a = render_body(instance_spelled.sinks, instance_spelled.walk,
                    kind=instance_spelled.kind).text
    b = render_body(class_spelled.sinks, class_spelled.walk, kind=class_spelled.kind).text
    assert a == b


def test_the_class_form_sets_kind_fields_by_plain_assignment():
    from hawk.ext import KernelKind, compensated

    class Compensated(KernelKind, slug="class_form_compensated"):
        sink = compensated(into="c")

    assert Compensated.kind == Kind("class_form_compensated", sink=compensated(into="c"))

    @Compensated
    def solo(x: Scalar, y: Mutable[Scalar]):
        y = x

    text = render_body(solo.sinks, solo.walk, kind=solo.kind).text
    assert "hawk_abi::store_compensated(" in text


def test_the_class_form_refuses_a_class_attribute_that_is_neither_form():
    from hawk.ext import KernelKind

    with pytest.raises(HawkError, match="neither an ANNOTATION"):
        class Bad(KernelKind, slug="class_form_bad_attribute"):
            def helper(self):  # noqa: ANN001 - the refusal is the point
                return 1


def test_the_class_form_needs_a_slug():
    from hawk.ext import KernelKind

    with pytest.raises(HawkError, match="slug"):
        class Bad(KernelKind):
            pass
