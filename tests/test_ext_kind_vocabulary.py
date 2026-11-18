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
from hawk.ext import DEFAULT_KIND, KernelKind, Kind, Output
from hawk.errors import HawkError
from hawk import _ir

VOCAB = {"terminated": Terminated, "scale": Scalar}


MASK_VOCAB = {"terminated": Terminated}


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
    exec(compile(src, filename, "exec"), ns)  # the row's own fixture
    k = ns["k"]

    @kernel
    def bare(v: Vector[3], y: Mutable[Vector[3]]):
        y = v

    assert k.arg_spec == bare.arg_spec
    assert _ir.digest(k.walk) == _ir.digest(bare.walk)


def _make_factory(w):
    @kernel
    def k(v: Vector[w], out: Mutable[Vector[w]]):
        """``w`` is never read in the body — the defining frame (``_make_factory``'s
        own locals) is where the string annotation ``Vector[w]`` resolves it."""
        out = v
    return k


def test_a_kernel_factory_resolves_an_unused_extent_from_the_defining_frame():
    k3, k5 = _make_factory(3), _make_factory(5)
    assert _ir.slot_type(k3.walk, "vec_in", "v").shape == (3,)
    assert _ir.slot_type(k5.walk, "vec_in", "v").shape == (5,)
    assert k3.arg_spec == k5.arg_spec


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


# --- the decorator takes the launch keywords hawk.kernel takes ------------------ #
LAUNCH = Kind("launch_kind", vocabulary={"terminated": Terminated})


def _launch_body():
    def body(x: Scalar, dt: Scalar, t: Mutable[Scalar], terminated: Terminated,
             y: Mutable[Scalar]):
        y = x * dt + t
        t = t + dt
        terminated = y > 3.0

    return body


@pytest.mark.parametrize("launch", [
    {"group": 4}, {"steps": 16}, {"time_step": "dt", "time": "t"}, {"jump": 0.05},
])
def test_a_kind_decorator_takes_the_launch_keywords_of_hawk_kernel(launch):
    spelled = kernel(_launch_body(), kind=LAUNCH, **launch)
    decorated = LAUNCH(**launch)(_launch_body())
    assert decorated.name == spelled.name
    assert _ir.digest(decorated.walk) == _ir.digest(spelled.walk)
    for field in ("group", "time", "time_step", "jump"):
        assert getattr(decorated, field) == getattr(spelled, field)


def test_a_kernel_kind_class_takes_them_too():
    class Launched(KernelKind, slug="launched"):
        terminated: Terminated

    @Launched(group=4)
    def body(x: Scalar, y: Mutable[Scalar]):
        y = x + 1.0

    assert body.group == 4 and body.kind is Launched.kind
