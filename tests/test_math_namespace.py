# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every name in ``hawk.math.__all__`` traces and emits.

Mirrors ``tests/test_trace_op_exports.py``, and for the same reason it
exists: a name in the authoring namespace that does not trace — or traces
and then has no aether spelling — is a hole an author only finds when their
kernel will not build. So the gate is the enumeration: :data:`SPELLINGS`
maps every exported name to a callable that uses it, each is run, and the
covered set must equal ``__all__``. A name added to ``hawk/math.py`` later
lands here until the table can say how to use it.

Emission is checked too, not just tracing. Three of these names are new IR
(``sample_index``, and the ``split_index`` pair built on it), and an IR node with
no renderer traces perfectly and fails at build time — which is the whole class
of defect ``test_trace_op_exports`` was written against.

THE TWO SEMANTIC ROWS at the bottom are what make the rest more than a smoke
test. ``split_index``'s contract is a claim about WHICH coordinate varies between
neighbouring lanes, and the identity ``lane == major * minor + minor_part``
either holds for every lane or the whole two-dimensional launch-domain story is
wrong; it is checked against the compiled HOST artifact, sample by sample, not
against the interpreter alone. And ``sample_index`` must be the GLOBAL index —
``base + flat`` — so the row runs the same kernel over a PARTITION and requires
the same answers in the same places (an oracle whose lane numbering moved
with the partitioning would be comparing a structure against itself).

This test previously failed via two plants, both observed and both removed; recorded
per row.
"""

from __future__ import annotations

import _oracle as O
import numpy as np
import pytest

import hawk
import hawk.math as M
import hawk.math as hm
import hawk.trace.value as T
from hawk.artifact import build
from hawk.emit import render_body
from hawk.ir import Assign, HawkError, canonical
from hawk.trace.value import TableRef, node_of
from hawk.types import TensorType

S = TensorType((), "f64")
B = TensorType((), "bool")
V3 = TensorType((3,), "f64")
Q = TensorType((4,), "f64", "quaternion")
M33 = TensorType((3, 3), "f64")

N = 12
MINOR = 4


def _v(t: TensorType, name: str = "a"):
    """A traced value over a fresh leaf of type ``t``."""
    from hawk.ir import Leaf

    if t.dtype == "bool":
        return hawk.Value(Leaf("terminated", "terminated", name, t))
    role = {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(t.shape)]
    return hawk.Value(Leaf("vocab_read", role, name, t))


def _table():
    return TableRef("tab", S)


#: ``exported name -> a callable that USES it and returns a traced value``.
#: ``split_index`` returns a pair, so its entry sums them: a name is covered when
#: what it returns reaches an emitted body, not when it merely runs.
SPELLINGS = {
    "acos": lambda: M.acos(_v(S)),
    "as_pure": lambda: M.as_pure(_v(V3)),
    "as_vec3": lambda: M.as_vec3(_v(Q)),
    "asin": lambda: M.asin(_v(S)),
    "atan": lambda: M.atan(_v(S)),
    "atan2": lambda: M.atan2(_v(S), _v(S, "b")),
    "cos": lambda: M.cos(_v(S)),
    "cross": lambda: M.cross(_v(V3), _v(V3, "b")),
    "dispatch": lambda: M.dispatch(_v(TensorType((), "i32"), "k"),
                                   [_v(S, "b"), _v(S, "c")]),
    "dot": lambda: M.dot(_v(V3), _v(V3, "b")),
    "exp": lambda: M.exp(_v(S)),
    "argmax": lambda: M.argmax(_v(V3)),
    "floor": lambda: M.floor(_v(S)),
    "land": lambda: M.select(M.land(_v(B), _v(B, "c")), _v(S), _v(S, "b")),
    "lnot": lambda: M.select(M.lnot(_v(B)), _v(S), _v(S, "b")),
    "log": lambda: M.log(_v(S)),
    "lor": lambda: M.select(M.lor(_v(B), _v(B, "c")), _v(S), _v(S, "b")),
    "max": lambda: M.max(_v(S), _v(S, "b")),
    "maximum": lambda: M.maximum(_v(S), _v(S, "b")),
    "min": lambda: M.min(_v(S), _v(S, "b")),
    "minimum": lambda: M.minimum(_v(S), _v(S, "b")),
    "n_samples": lambda: _v(S) * M.n_samples(),
    "norm": lambda: M.norm(_v(V3)),
    "outer": lambda: M.vsum(M.outer(_v(V3), _v(V3, "b")) @ _v(V3, "c")),
    "quat_conj": lambda: M.as_vec3(M.quat_conj(_v(Q))),
    "quat_mul": lambda: M.as_vec3(M.quat_mul(_v(Q), _v(Q, "b"))),
    "quat_recip": lambda: M.as_vec3(M.quat_recip(_v(Q))),
    "quat_rotate": lambda: M.quat_rotate(_v(Q), _v(V3, "b")),
    "random_normal": lambda: M.random_normal(_v(TensorType((), "i32"), "seed"),
                                             _v(TensorType((), "i32"), "counter")),
    "random_bernoulli": lambda: M.random_bernoulli(_v(TensorType((), "i32"), "seed"),
                                                 _v(TensorType((), "i32"), "counter"), 0.3),
    "random_exponential": lambda: M.random_exponential(_v(TensorType((), "i32"), "seed"),
                                                     _v(TensorType((), "i32"), "counter"), 2.0),
    "random_lognormal": lambda: M.random_lognormal(_v(TensorType((), "i32"), "seed"),
                                                 _v(TensorType((), "i32"), "counter"), 0.5, 0.25),
    "random_multivariate_normal": lambda: M.vsum(M.random_multivariate_normal(
        _v(TensorType((), "i32"), "seed"), _v(TensorType((), "i32"), "counter"),
        _v(V3, "mu"), _v(M33, "L"))),
    "random_uniform_int": lambda: M.random_uniform_int(_v(TensorType((), "i32"), "seed"),
                                                     _v(TensorType((), "i32"), "counter"), -2, 5),
    "random_poisson1": lambda: M.random_poisson1(_v(TensorType((), "i32"), "seed"),
                                               _v(TensorType((), "i32"), "counter")),
    "random_uniform": lambda: M.random_uniform(_v(TensorType((), "i32"), "seed"),
                                               _v(TensorType((), "i32"), "counter")),
    "rsqrt": lambda: M.rsqrt(_v(S)),
    "sample_index": lambda: _table().at(M.sample_index()),
    "select": lambda: M.select(_v(B), _v(S), _v(S, "b")),
    "sin": lambda: M.sin(_v(S)),
    "split_index": lambda: _table().at(sum(M.split_index(MINOR))),
    "sqrt": lambda: M.sqrt(_v(S)),
    "take": lambda: M.take(_table(), _v(S, "where")),
    "tan": lambda: M.tan(_v(S)),
    "tanh": lambda: M.tanh(_v(S)),
    "transpose": lambda: M.vsum(M.transpose(_v(M33)) @ _v(V3, "b")),
    "vec": lambda: M.vsum(M.vec([_v(S), _v(S, "b"), _v(S, "c")])),
    "vsum": lambda: M.vsum(_v(V3)),
    "where": lambda: M.where(_v(B), _v(S), _v(S, "b")),
    "exp2": lambda: M.exp2(_v(S)),
    "expm1": lambda: M.expm1(_v(S)),
    "log2": lambda: M.log2(_v(S)),
    "log10": lambda: M.log10(_v(S)),
    "log1p": lambda: M.log1p(_v(S)),
    "cbrt": lambda: M.cbrt(_v(S)),
    "sinh": lambda: M.sinh(_v(S)),
    "cosh": lambda: M.cosh(_v(S)),
    "asinh": lambda: M.asinh(_v(S)),
    "acosh": lambda: M.acosh(_v(S)),
    "atanh": lambda: M.atanh(_v(S)),
    "ceil": lambda: M.ceil(_v(S)),
    "trunc": lambda: M.trunc(_v(S)),
    "round": lambda: M.round(_v(S)),
    "rint": lambda: M.rint(_v(S)),
    "sign": lambda: M.sign(_v(S)),
    "erf": lambda: M.erf(_v(S)),
    "erfc": lambda: M.erfc(_v(S)),
    "hypot": lambda: M.hypot(_v(S), _v(S, "b")),
    "copysign": lambda: M.copysign(_v(S), _v(S, "b")),
    "fmod": lambda: M.fmod(_v(S), _v(S, "b")),
    "remainder": lambda: M.remainder(_v(S), _v(S, "b")),
    "fdim": lambda: M.fdim(_v(S), _v(S, "b")),
    "fma": lambda: M.fma(_v(S), _v(S, "b"), _v(S, "c")),
    "clip": lambda: M.clip(_v(S), _v(S, "b"), _v(S, "c")),
    "isnan": lambda: M.select(M.isnan(_v(S)), _v(S, "b"), _v(S, "c")),
    "isinf": lambda: M.select(M.isinf(_v(S)), _v(S, "b"), _v(S, "c")),
    "isfinite": lambda: M.select(M.isfinite(_v(S)), _v(S, "b"), _v(S, "c")),
    "abs": lambda: M.abs(_v(V3)),
    "absolute": lambda: M.absolute(_v(S)),
    "pow": lambda: M.pow(_v(S), _v(S, "b")),
    "power": lambda: M.power(_v(V3), 2.0),
}


def _used(spell):
    """Run one spelling; a failure is RETURNED so it counts as UNCOVERED."""
    try:
        return node_of(spell())
    except Exception as exc:                            # noqa: BLE001 - reported
        return exc


USED = {name: _used(spell) for name, spell in SPELLINGS.items()}


def test_the_table_covers_exactly_the_exported_namespace():
    """Needs every exported name covered; if ``split_index`` is missing
    from ``hawk/math.__all__``::

        AssertionError: this row spells ['split_index'], which hawk.math no
        longer exports
    """
    missing = sorted(set(M.__all__) - set(SPELLINGS))
    assert not missing, (
        f"these names are exported by hawk.math but no row uses them: {missing}. "
        "A name in the authoring namespace that no row traces is a name an "
        "author discovers is broken when their kernel will not build")
    stale = sorted(set(SPELLINGS) - set(M.__all__))
    assert not stale, f"this row spells {stale}, which hawk.math no longer exports"


@pytest.mark.parametrize("name", sorted(SPELLINGS), ids=sorted(SPELLINGS))
def test_every_name_traces_and_emits(name):
    """Traced AND rendered: an IR node with no aether spelling traces perfectly
    and then fails at build time, far from the name that minted it."""
    node = USED[name]
    assert not isinstance(node, Exception), (
        f"the recorded spelling for hawk.math.{name} does not run: {node!r}")
    sinks = (Assign("out", node, node.ttype),)
    text = render_body(sinks, canonical(sinks)).text
    assert text.strip(), f"hawk.math.{name} emitted an empty body"


#: The four names that are ADAPTERS rather than re-exports, and what each adapts.
#: The list is short and named on purpose: every other entry in ``__all__`` must
#: be the identical object ``hawk.trace`` exports, so "namespace, not layer" is
#: an assertion rather than an intention.
ADAPTERS = {
    "vec": "accepts vec([...]) as well as vec(a, b, c)",
    "where": "the np.where spelling of select",
    "split_index": "sample_index() composed with / and the remainder identity",
    "random_uniform": "accepts lo/hi: the primitive at the defaults, lo + (hi-lo)*u otherwise",
    "random_normal": "accepts mean/sd: the primitive at the defaults, mean + sd*z otherwise",
    "take": "the free-function spelling of a plane's own at()",
    "argmax": "a select chain over the components: the first index of the maximum, as i32",
}


def test_every_exported_name_is_resolvable_and_is_the_trace_surface_itself():
    """A namespace, not a layer: each re-exported name must BE the object
    :mod:`hawk.trace.value` defines, so there is one implementation and not
    two. ``hawk.trace.value`` (the deep submodule, not the ``hawk.trace``
    PACKAGE) is the comparison target: the package stopped re-exporting
    these names once their canonical public path became ``hawk.math`` alone,
    but the primitives themselves still
    live in this submodule, and that is exactly the "one implementation"
    this row is pinning."""
    shared = [n for n in M.__all__ if hasattr(T, n) and n not in ADAPTERS]
    assert len(shared) >= 30, shared
    for name in shared:
        assert getattr(M, name) is getattr(T, name), (
            f"hawk.math.{name} is not hawk.trace.{name} — a namespace module "
            "re-exports; it does not re-implement")
    assert M.max is M.maximum and M.min is M.minimum
    assert M.abs is M.absolute and M.pow is M.power
    assert set(ADAPTERS) <= set(M.__all__)
    for name, why in ADAPTERS.items():
        assert why, f"the adapter {name!r} is excused with no reason"
    # an adapter is still a composition of the SAME vocabulary: nothing here
    # mints an op kind the flat surface cannot.
    assert M.vec(1.0, 2.0).node.kind == hm.vec(1.0, 2.0).node.kind
    assert M.where(_v(B), _v(S), _v(S, "b")).node.kind == "select"


def test_both_backends_bind_the_name_a_sample_index_renders_as():
    """The body knows exactly ONE thing about its wrapper: the identifier the
    index prologue binds the lane's integer index to ( lets it know no more).
    Both backends must bind that name, and the name has one spelling
    (:data:`hawk.emit.aether.SAMPLE_INDEX_IDENT`) — otherwise a body that reads
    the lane index compiles on one target and not the other."""
    from hawk.emit import BACKENDS
    from hawk.emit.aether import SAMPLE_INDEX_IDENT

    for backend in BACKENDS.values():
        prologue = backend.index_prologue()
        assert f" {SAMPLE_INDEX_IDENT} =" in prologue, (
            f"the {backend.id} prologue binds no {SAMPLE_INDEX_IDENT!r}:\n{prologue}")
    lane = M.sample_index()
    sinks = (Assign("out", node_of(lane * 1.0), S),)
    assert SAMPLE_INDEX_IDENT in render_body(sinks, canonical(sinks)).text


def test_random_namespace_is_the_same_object_as_the_flat_names():
    """The spec's spelling, ``hawk.math.random.uniform`` /
    ``.normal``, is the SAME object the flat ``__all__`` names are — a tiny
    attribute namespace, never a second implementation (this module mints
    nothing, its own docstring rule)."""
    assert M.random.uniform is M.random_uniform
    assert M.random.normal is M.random_normal
    assert M.random.poisson1 is M.random_poisson1
    for name in ("lognormal", "exponential", "bernoulli", "uniform_int",
                 "multivariate_normal"):
        assert getattr(M.random, name) is getattr(M, "random_" + name), name


def test_uniform_and_normal_are_the_primitive_at_their_defaults():
    """``random.uniform(seed, counter)`` traces to ONE ``random_uniform`` node
    (the primitive, nothing wrapped around it); a range or a scale makes
    it the documented composition."""
    seed, counter = _v(TensorType((), "i32"), "seed"), _v(TensorType((), "i32"), "counter")
    assert M.random.uniform(seed, counter).node.kind == "random_uniform"
    assert M.random.normal(seed, counter).node.kind == "random_normal"
    assert M.random.uniform(seed, counter, -1.0, 3.0).node.kind == "add"
    assert M.random.normal(seed, counter, sd=0.5).node.kind == "add"
    assert M.random.uniform(seed, counter, 0, 1).node.kind == "random_uniform"


def test_split_index_refuses_a_non_constant_or_non_positive_divisor():
    with pytest.raises(HawkError, match="compile-time positive int"):
        M.split_index(_v(S))
    with pytest.raises(HawkError, match="must be positive"):
        M.split_index(0)


# --------------------------------------------------------------------------- #
# The three semantic rows, on the compiled host path.
# --------------------------------------------------------------------------- #
@hawk.kernel
def split_identity(y: hawk.Mutable[hawk.Vector[3]]):
    """``(major, minor, major * MINOR + minor)`` — the split and its inverse."""
    major, minor = M.split_index(MINOR)
    y = M.vec([major * 1.0, minor * 1.0, (major * MINOR + minor) * 1.0])


@hawk.kernel
def lane_value(y: hawk.Mutable[hawk.Scalar]):
    """The lane index itself, as a number an oracle can compare."""
    y = M.sample_index() * 1.0


@hawk.kernel
def floor_value(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """``floor(x)`` ( lever 2) — a name that mints and renders but never
    RUNS is still unverified (``hawk.math.floor``'s own SPELLINGS row above
    only traces and emits); this is the ONE device value check."""
    y = M.floor(x)


def test_floor_matches_numpy_on_the_compiled_host_path(tmp_path, cache_dir):
    xs = np.linspace(-3.4, 3.4, N)
    got = _run(floor_value, tmp_path / "floor", cache_dir, x=xs)
    np.testing.assert_array_equal(got, np.floor(xs))


def test_the_split_index_identity_holds_on_the_compiled_host_path(tmp_path,
                                                                  cache_dir):
    """RED (plant): the remainder in ``hawk/math.split_index`` spelled
    ``lane - major`` instead of ``lane - major * minor``::

        Mismatched elements: 8 / 12 (66.7%)
        Max absolute difference among violations: 6.

    — the minor component stopped being the remainder, and with it the third
    component stopped reproducing the lane, which is the identity the whole
    two-dimensional launch domain rests on.
    """
    got = _run(split_identity, tmp_path / "split", cache_dir)
    lane = np.arange(N)
    np.testing.assert_array_equal(got[0], lane // MINOR)
    np.testing.assert_array_equal(got[1], lane % MINOR)
    np.testing.assert_array_equal(got[2], lane)


def test_sample_index_is_the_global_index_under_a_partition(tmp_path, cache_dir):
    """The index a body reads is ``base + flat``, not the
    partition-local one — the same property ``n_samples`` has, and for the same
    reason: an expression that moved with the partitioning would make the serial
    oracle a comparison of a structure against itself."""
    build(lane_value, tmp_path / "lane", targets=("host",), cache_dir=cache_dir)
    loaded = O.load(tmp_path / "lane", "lane_value")
    whole = O.run_kernel(loaded, N)
    np.testing.assert_array_equal(whole, np.arange(N, dtype=float))

    half = N // 2
    tiled = np.zeros(N)
    for base in (0, half):
        planes = O.planes(loaded, N, {})
        from hawk import runtime
        runtime.run(loaded, base=base, count=half, n_samples=N, **planes)
        tiled[base:base + half] = np.asarray(planes["y"])[base:base + half]
    np.testing.assert_array_equal(tiled, whole)


def _run(kernel, directory, cache, **planes):
    build(kernel, directory, targets=("host",), cache_dir=cache)
    loaded = O.load(directory, kernel.name)
    wanted = {name: planes[name] for role, name in loaded.arg_spec
              if role in O.INPUT_ROLES or role == "uniform"}
    return O.run_kernel(loaded, N, **wanted)
