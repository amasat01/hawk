# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target computes the SAME BITS as hawk's own host build.

The C unit (:func:`hawk.embedded.export`) is compiled into a shared library
(:func:`_embedded.cc_shared`) and called through its args struct
(:func:`_embedded.run_c`); the host build runs the same kernel through
``hawk._core``. Both are serial, both are built without floating-point
contraction (``-ffp-contract=off``; the host under the ``native`` profile), and
both evaluate the formula table of the design spec, so the comparison is
``tobytes()`` equality, never a band.

Three groups:

* **the deployable fixtures** (session ``built`` fixture, which compiles the
  cuda target too, so these rows run on Linux CI): every in-scope
  ``_deployable.cases(N)`` case against ``_oracle.run_kernel`` on the built
  host artifact; ``energy`` (a mapreduce) is not in scope, and its refusal,
  with every other feature the C target refuses, is asserted by name;
* **the loop / dispatch / finish / derivative corpus** (this file's own
  host-only bundles, one per access class per mode, built once per session):
  head, middle and tail breaks, a per-sample stop running past the cap, the
  vjp through a head and a tail break (the reverse loop's RUNTIME trip count,
  clamped by ``hawk_clamp_trips``), switch / predicated / auto dispatch with
  the branch-mix plane, finishing kernels and ``hawk.steps(k, 4)``, ``vjp``
  of ``drag``, two ``tape()`` twins, a vocabulary sweep, ``Vector / Vector``,
  two masks and a finish in one kernel (under ``TWO_MASKS``, its own bundle)
  and own-column partials under a mask guard, in float64 and float32, every
  writable plane compared (masks and counters too), and every plane under a
  mask guard checked to keep the caller's value at a masked sample; plus a
  partition run (``base=2, count=3, n=8``) whose untouched samples must stay
  zero on both sides;
* **the integer kernels on their edge grids** (the same host-only bundles):
  ``test_int_planes``' ``ops_i32``, ``ops_u32``, ``casts`` (Param ``s`` in
  ``{-5, 7, 2^31 - 1}``), ``scaled`` and two Index kernels
  (``index_arith``: ``+ - *`` and negation; ``index_divshift``: ``// % << >>``),
  in float64 and float32, each fed the boundary grid its operators are pinned
  at (``INT32_EDGES``/``UINT32_EDGES`` pairs; NaN, +-inf, +-0, 2^31 +- 0.5,
  2^32 +- 0.5, +-2^63 and 1e300 for the casts; a 16x16 Index grid reaching
  +-2^31 and +-2^40), never a random draw. Int32/UInt32 arithmetic and the
  Index ``// % << >>`` are a wrapping helper on both sides, so they compare
  over the whole grid. Index ``+ - *`` and negation identity is "where the C++
  is defined": the C++ spells them infix (signed overflow is undefined) and
  the C wraps them through ``hawk_i64_*`` (the report's ``int_ub_sites`` and
  ``defined_where_cpp_is_ub`` note). So the split: the host comparison runs
  ``index_arith`` only on the pairs whose results stay inside int64, and a
  C-only row runs it at the overflow points (the grid plus the int64
  extremes) against NumPy's int64 wrapping arithmetic. ``abs``/``minimum``/
  ``maximum`` on an Index are not in the grid: they do not compile on the
  host, so the C's NumPy semantics for them have no host twin;
* **the interpreter row** (no host artifact and no ``hawk.runtime`` on its
  path; this module itself imports ``hawk``, which loads ``hawk._core``, so
  the row still needs the extension installed): the C library against
  ``tests/_eval.evaluate`` per sample, within the ``S x 2 x eps`` band of
  ``test_serial_oracle`` (restated, not imported: one test module does not
  import another).

Every failing row names the kernel, the mode, the plane, the first differing
(component, sample) and both bit patterns.
"""

from __future__ import annotations

import functools
import pathlib

import _deployable as D
import _embedded as E
import _kernels as K
import numpy as np
import pytest

import hawk
from hawk import Accum, Index, Mutable, Param, Scalar, Terminated, Vector
from hawk.embedded import export
from hawk.ir import HawkError

N = 96
SEED = 20261016
MODES = ("float64", "float32")
_NP = {"float64": np.float64, "float32": np.float32}
_EPS = {"float64": np.finfo(np.float64).eps, "float32": np.finfo(np.float32).eps}


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def _bits(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(value).reshape(1).view(np.uint8).tobytes()
    return "0x" + raw[::-1].hex()


def first_difference(got, want):
    """``None``, or ``(component, sample, got value, want value)`` of the first
    element whose BYTES differ (so ``-0.0`` vs ``0.0`` and differing NaN
    payloads count)."""
    g, w = np.ascontiguousarray(got), np.ascontiguousarray(want)
    if g.tobytes() == w.tobytes():
        return None
    size = g.dtype.itemsize
    bad = np.flatnonzero((g.reshape(-1).view(np.uint8) != w.reshape(-1).view(np.uint8))
                         .reshape(-1, size).any(axis=1))[0]
    shape = g.shape if g.ndim == 2 else (1, g.shape[0])
    comp, sample = np.unravel_index(bad, shape)
    return int(comp), int(sample), g.reshape(-1)[bad:bad + 1], w.reshape(-1)[bad:bad + 1]


def assert_bit_identical(got, want, *, kernel: str, mode: str, plane: str) -> None:
    g, w = np.ascontiguousarray(got), np.ascontiguousarray(want)
    assert g.dtype == w.dtype and g.shape == w.shape, (
        f"{kernel} [{mode}] plane {plane!r}: shape/dtype disagree, C "
        f"{g.shape}/{g.dtype} vs host {w.shape}/{w.dtype}")
    diff = first_difference(g, w)
    if diff is None:
        return
    comp, sample, gv, wv = diff
    count = int(np.sum((g.reshape(-1).view(np.uint8) != w.reshape(-1).view(np.uint8))
                       .reshape(-1, g.dtype.itemsize).any(axis=1)))
    raise AssertionError(
        f"{kernel} [{mode}] plane {plane!r}: NOT bit-identical to the host build; "
        f"first difference at component {comp}, sample {sample}: C {gv[0]!r} "
        f"({_bits(gv)}) vs host {wv[0]!r} ({_bits(wv)}); {count} element(s) "
        "differ. Both sides are serial and uncontracted, so this is a C spelling "
        "that evaluates a different formula than aether's host path")


def _tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _written(report) -> list:
    """The interface rows a call may write, in interface order."""
    return [r for r in report["interface"]
            if r["role"] != "nsamples" and r["direction"] != "in"]


# --------------------------------------------------------------------------- #
# the C side, compiled once per (kernel, mode) per session
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def c_units(tmp_path_factory):
    """``unit(kernel, mode, kind=None) -> (so_path, report)``, cached. The
    export runs its own checks (``check=True``: strict cc and cppcheck), so
    the unit compared is the unit a caller would ship."""
    root = pathlib.Path(tmp_path_factory.mktemp("embedded_c"))
    cache: dict = {}

    def unit(kernel, mode, kind=None):
        key = (kernel.name, mode, getattr(kind, "id", kind))
        if key not in cache:
            where = root / f"{kernel.name}_{mode}"
            art = export(kernel, where, mode=mode, kind=kind)
            cache[key] = (E.cc_shared(art.c_path, where), art.report)
        return cache[key]

    return unit


# --------------------------------------------------------------------------- #
# group 1: the deployable fixtures against the session `built` bundles (CI)
# --------------------------------------------------------------------------- #
#: The fixture cases the C target builds. ``energy`` is a mapreduce (refused,
#: asserted below); ``scatter_c`` is not a ``cases()`` entry, and its
#: compensated build is refused below too.
_OUT_OF_SCOPE = {"energy"}
_CASES = [c for c in D.cases(N) if c[1] not in _OUT_OF_SCOPE]


@pytest.mark.parametrize("bundle_name,kernel,kw,_want", _CASES,
                         ids=[c[0] for c in _CASES])
def test_built_fixture_c_matches_the_serial_oracle_bit_for_bit(
        built, c_units, bundle_name, kernel, kw, _want):
    import _oracle as O
    from conftest import sidecar_of

    bundle = built[bundle_name]
    sidecar = sidecar_of(bundle, kernel)
    mode = sidecar["scalar_type"]
    want = O.run_kernel(O.load(bundle.directory, kernel, sidecar), N, **kw)
    so, report = c_units(getattr(D, kernel), mode)
    got = E.run_c(so, kernel, report, N, mode=mode, **kw)
    rows = _written(report)
    assert len(_tuple(got)) == len(_tuple(want)) == len(rows)
    for row, g, w in zip(rows, _tuple(got), _tuple(want)):
        assert_bit_identical(g, w, kernel=kernel, mode=mode, plane=row["name"])


# --------------------------------------------------------------------------- #
# refusals: what the C target does not build is refused BY NAME, never run
# --------------------------------------------------------------------------- #
@hawk.kernel
def _auto_step(dt: Param, t_final: Scalar, terminated: Terminated,
               t: Mutable[Scalar]):
    """A finishing step: ``@hawk.kernel`` without ``steps=`` derives the
    automatic kernel, which branches on a runtime step word."""
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def _refused():
    import test_embedded_refusals as R

    return {
        "energy": (D.energy, None, "mapreduce"),
        "scatter_c_compensated": (D.scatter_c, D.COMPENSATED, "compensated"),
        "random": (R.draw, None, "random"),
        "active_set": (R.active, None, "active_set"),
        "segmented": (R.segmented, None, "segmented"),
        "steps_auto": (_auto_step, None, "two_path"),
    }


@pytest.mark.parametrize("case", ["energy", "scatter_c_compensated", "random",
                                  "active_set", "segmented", "steps_auto"])
@pytest.mark.parametrize("mode", MODES)
def test_a_refused_feature_is_named_and_nothing_is_written(tmp_path, case, mode):
    kernel, kind, feature = _refused()[case]
    with pytest.raises(HawkError) as err:
        export(kernel, tmp_path / "out", mode=mode, kind=kind, check=False)
    text = str(err.value)
    assert f"unsupported.{feature}" in text and "the C target" in text, text
    assert repr(kernel.name) in text, text
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- #
# group 2: the loop / dispatch / finish / derivative corpus against this
# file's own host-only bundles
# --------------------------------------------------------------------------- #
CAP = 12        # test_loop_early_exit.CAP, restated: no test-module import at collection
#: (kernel name, selector extent) of every dispatch fixture: an index plane
#: named ``k`` stays inside ``[0, K)``.
_SELECTOR = {"disp_r0_k5_pred": 5, "disp_r0_k5_switch": 5, "disp_r1_k3_pred": 3,
             "disp_r1_k3_switch": 3, "disp_int_pred": 2, "disp_int_switch": 2,
             "mix_auto": 3, "mix_switch": 3}


@hawk.kernel
def gp(x: Scalar, terminated: Terminated, acc: Accum[Scalar]):
    """Own-column partials under the default mask guard: a masked sample
    keeps the partial the caller handed over (the interface row is ``inout``,
    init by caller)."""
    acc.add(x * x)


@hawk.kernel
def index_arith(a: Index, b: Index, s: Mutable[Index], p: Mutable[Index],
                ng: Mutable[Index]):
    """Index ``+ - *`` and negation: infix in the C++ (signed overflow is
    undefined), ``hawk_i64_*`` in the C. Compared with the host only where the
    C++ is defined; pinned against NumPy at the overflow points."""
    s = a + b - 3
    p = a * b
    ng = -a


@hawk.kernel
def index_div(a: Index, b: Index, q: Mutable[Index]):
    """Index true division: infix in the C++ (a zero divisor and ``MIN / -1``
    are undefined), ``hawk_i64_div`` in the C. Compared with the host only
    where the C++ is defined; the two undefined points are pinned C-only."""
    q = a / b


@hawk.kernel
def index_divshift(a: Index, b: Index, q: Mutable[Index], r: Mutable[Index],
                   sh: Mutable[Index]):
    """Index ``// %`` and the shifts: a wrapping helper on BOTH sides, so the
    host comparison takes the whole grid."""
    q = a // b
    r = a % b
    sh = (a << b) >> 1


@hawk.kernel
def vdiv(a: Vector[3], b: Vector[3], y: Mutable[Vector[3]]):
    """Rank>=1 over rank>=1: the core's ``vec_cwise_div`` against the host's
    ``aether::cwiseDiv``, ``a_k / b_k``."""
    y = a / b


@functools.cache
def _corpus() -> dict:
    """``{name: kernel}``. The fixture modules are test modules with their own
    collection-time work, so they are imported here, never at collection."""
    import test_dispatch as T
    import test_dispatch_mix_and_guard as G
    import test_finish_own_sample as F
    import test_int_planes as I
    import test_loop_early_exit as LE
    import test_steps as S
    import test_tape_twin as TT

    from hawk import Kernel
    from hawk.diff import vjp
    from hawk.diff.tape import tape

    kernels = [
        LE.head, LE.middle, LE.tail, LE.gated_sum, LE.trajectory, LE.vec_head,
        Kernel("head_vjp", vjp(LE.head, wrt=("x",))),
        Kernel("tail_vjp", vjp(LE.tail, wrt=("x",))),
        T.disp_r0_k5_pred, T.disp_r0_k5_switch, T.disp_r1_k3_pred,
        T.disp_r1_k3_switch, T.disp_int_pred, T.disp_int_switch,
        G.mix_auto, G.mix_switch,
        F.osc_finish, F.stop_only, F.branch_finish,
        hawk.steps(S.osc, 4), hawk.steps(S.spin3, 4),
        K.drag, Kernel("drag_vjp", vjp(K.drag)), K.scale, K.chain6,
        tape(TT.osc, 8), tape(TT.spin, 8),
        D.vec3_scale, D.mat_apply, D.spin, D.vocab, D.diagnostic_two,
        D.scatter, gp, D.two_fin.step, vdiv,
        I.ops_i32, I.ops_u32, I.casts, I.scaled, index_arith, index_div,
        index_divshift,
    ]
    out = {k.name: k for k in kernels}
    assert len(out) == len(kernels), "two corpus kernels share a name"
    assert LE.CAP == CAP
    return out


#: Every group-2 kernel, by name (pinned against :func:`_corpus` in the
#: fixture, so the two lists cannot drift).
CORPUS_NAMES = (
    "head", "middle", "tail", "gated_sum", "trajectory", "vec_head", "head_vjp",
    "tail_vjp", "disp_r0_k5_pred", "disp_r0_k5_switch", "disp_r1_k3_pred",
    "disp_r1_k3_switch", "disp_int_pred", "disp_int_switch", "mix_auto",
    "mix_switch", "osc_finish", "stop_only", "branch_finish", "osc_x4",
    "spin3_x4", "drag", "drag_vjp", "scale", "chain6", "osc_tape", "spin_tape",
    "vec3_scale", "mat_apply", "spin", "vocab", "diagnostic_two", "scatter",
    "gp", "two_fin", "vdiv", "ops_i32", "ops_u32", "casts", "scaled", "index_arith",
    "index_div", "index_divshift",
)
#: The integer kernels: fed their edge grids (:data:`EDGE_CASES`), never the
#: random draw of :func:`_inputs`.
_EDGE_GRID = ("ops_i32", "ops_u32", "casts", "scaled", "index_arith", "index_div",
              "index_divshift")
#: Every group-2 kernel the random draw feeds.
RANDOM_NAMES = tuple(n for n in CORPUS_NAMES if n not in _EDGE_GRID)

#: ``name -> (bundle group, kind)`` of every kernel outside the default
#: ``local`` group: a bundle declares one execution axis (the scattering
#: kernel is its own unit) and renders under ONE kind.
_GROUPS = {"scatter": ("scattering", None), "two_fin": ("two_masks", D.TWO_MASKS)}


def _group(name: str) -> tuple:
    return _GROUPS.get(name, ("local", None))


@pytest.fixture(scope="session")
def host_units(tmp_path_factory, cache_dir):
    """``unit(name, mode) -> HostKernel``: one host-only bundle per mode and
    :data:`_GROUPS` group (the sample-local corpus, the scattering kernel, a
    kernel under another kind), native profile, built on first use and once
    per session."""
    from hawk import runtime
    from hawk.artifact import build_bundle

    corpus = _corpus()
    assert tuple(corpus) == CORPUS_NAMES
    root = pathlib.Path(tmp_path_factory.mktemp("embedded_host"))
    bundles: dict = {}
    loaded: dict = {}

    def unit(name, mode):
        if mode not in bundles:
            groups: dict = {}
            for n, k in corpus.items():
                groups.setdefault(_group(n), []).append(k)
            bundles[mode] = {
                group: build_bundle(members, root / f"{group}_{mode}", mode=mode,
                                    kind=kind, targets=("host",),
                                    host_profile="native", cache_dir=cache_dir)
                for (group, kind), members in groups.items()}
        if (name, mode) not in loaded:
            directory = bundles[mode][_group(name)[0]].directory
            loaded[(name, mode)] = runtime.load(directory, name)
        return loaded[(name, mode)]

    return unit


def _inputs(name: str, report: dict, n: int, mode: str, rng) -> dict:
    """One value per interface row (``nsamples`` aside), drawn once and copied
    to each side: reals in ``[-2, 2]`` (by-value reals rounded to the mode's
    width first), index planes in range, masks a mix of 0/1 bytes, counters
    and tapes zeroed, a per-sample stop up to ``CAP + 5``."""
    real = _NP[mode]
    out = {}
    for row in report["interface"]:
        role, rname, dtype = row["role"], row["name"], row["dtype"]
        width = int(np.prod(row["shape"])) if row["shape"] else 1
        cells = (width, n) if row["shape"] else (n,)
        if role == "nsamples":
            continue
        if role == "uniform" and dtype in ("f64", "f32"):
            out[rname] = float(real(rng.uniform(-2.0, 2.0)))
        elif role == "uniform":                     # an integer Param
            out[rname] = 1 if rname == "tape_on" else int(rng.integers(-3, 4))
        elif role == "terminated":
            out[rname] = rng.random(n) < 0.3
        elif role == "lookup" and row["direction"] == "inout":      # counters
            out[rname] = np.zeros(width, dtype=np.uint32)
        elif role == "out" or rname == "applied_steps":             # tape twin
            out[rname] = np.zeros(cells, dtype=real if role == "out" else np.int64)
        elif role == "accum_out" and row["init_by_caller"] and row["layout"] == "partials":
            # guarded partials: a masked sample keeps what the caller passed
            out[rname] = rng.uniform(-2.0, 2.0, n).astype(real)
        elif role == "accum_out":
            out[rname] = np.zeros(n, dtype=real)
        elif dtype in ("int32", "uint32"):
            raise AssertionError(f"{name}: an integer plane takes its edge grid "
                                 "(EDGE_CASES), not a random draw")
        elif dtype in ("i32", "i64"):
            if rname == "k" and name in _SELECTOR:
                out[rname] = rng.integers(0, _SELECTOR[name], cells).astype(np.int64)
            elif rname == "lane":
                out[rname] = rng.integers(0, n, cells).astype(np.int64)
            else:
                out[rname] = rng.integers(-1000, 1000, cells).astype(np.int64)
        elif row["direction"] == "out":
            out[rname] = np.zeros(cells, dtype=real)
        else:
            out[rname] = rng.uniform(-2.0, 2.0, cells).astype(real)
    if name == "trajectory":
        # the per-sample stop: some samples stop early, some run past the cap
        out["nstop"] = rng.integers(0, CAP + 6, n).astype(real)
        out["k"] = np.zeros(n, dtype=real)
    return out


def _copies(values: dict) -> dict:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in values.items()}


def _run_host(kernel, n: int, values: dict, *, base: int = 0, count=None) -> dict:
    """The host side: every plane supplied by the caller (so the masks and
    counters the oracle does not hand back are read in place)."""
    import _oracle as O

    from hawk import runtime

    bound = O.planes(kernel, n, values)
    runtime.run(kernel, base=base, count=n if count is None else count,
                n_samples=n, **bound)
    return bound


def _compare_written(name, mode, report, c_side: dict, host_side: dict) -> None:
    for row in _written(report):
        assert_bit_identical(c_side[row["name"]], host_side[row["name"]],
                             kernel=name, mode=mode, plane=row["name"])


def _assert_masked_kept(name, mode, report, values: dict, sides: dict) -> int:
    """Every per-sample plane a guarded kernel writes (a ``mutable``, own
    partials, the finish mask) still holds, at each sample one of the row's
    ``guard`` masks flagged on entry, exactly the value the caller handed
    over, on each side. Returns how many such planes were checked."""
    fields = {r["field"]: r["name"] for r in report["interface"]}
    checked = 0
    for row in _written(report):
        per_sample = (row["role"] in ("mutable", "terminated")
                      or row["layout"] == "partials")
        if not row["guard"] or not per_sample:
            continue
        assert (row["direction"], row["init_by_caller"]) == ("inout", True), row
        masked = np.zeros(np.asarray(values[row["name"]]).shape[-1], dtype=bool)
        for field in row["guard"]:
            masked |= np.asarray(values[fields[field]], dtype=bool)
        assert masked.any(), f"{name}: no sample is masked; the row proves nothing"
        want = np.ascontiguousarray(values[row["name"]][..., masked])
        for side, planes in sides.items():
            got = np.ascontiguousarray(planes[row["name"]][..., masked])
            assert got.tobytes() == want.tobytes(), (
                f"{name} [{mode}] plane {row['name']!r}: the {side} side changed a "
                f"masked sample (guard: {', '.join(row['guard'])})")
        checked += 1
    return checked


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_c_matches_the_host_bundle_bit_for_bit(host_units, c_units, name, mode):
    kernel = _corpus()[name]
    so, report = c_units(kernel, mode, _group(name)[1])
    host = host_units(name, mode)
    values = _inputs(name, report, N, mode, np.random.default_rng(SEED))
    c_side, host_side = _copies(values), _copies(values)
    E.run_c(so, name, report, N, mode=mode, **c_side)
    _run_host(host, N, host_side)
    _compare_written(name, mode, report, c_side, host_side)
    _assert_masked_kept(name, mode, report, values, {"C": c_side, "host": host_side})


# --------------------------------------------------------------------------- #
# the integer kernels on their edge grids
# --------------------------------------------------------------------------- #
#: The 16x16 Index grid: small values, shift counts at and past the width,
#: and magnitudes past 2^31 and 2^40 (so products overflow 64 bits).
_INDEX_EDGES = [0, 1, -1, 2, -2, 7, -7, 31, 63, 64, 1000, -1000, 2**31 - 1, -2**31,
                2**40, -2**40]
#: Reals at every boundary a saturating conversion pins.
_CAST_XS = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 0.5, -0.5, -0.999,
            0.999, 1.0, -1.0, 2147483647.0, 2147483647.5, 2147483648.0, 2147483648.5,
            -2147483647.5, -2147483648.0, -2147483648.5, -2147483649.0, 4294967295.0,
            4294967295.5, 4294967296.0, 4294967296.5, 1e300, -1e300, 2.0**63,
            -(2.0**63), 123456.789, -123456.789]


def _pairs(edges, dtype) -> dict:
    a, b = np.meshgrid(np.array(edges, dtype=dtype), np.array(edges, dtype=dtype))
    return {"a": np.ascontiguousarray(a.reshape(-1)),
            "b": np.ascontiguousarray(b.reshape(-1))}


_I64 = (-(2**63), 2**63 - 1)


def _wrap64(x: int) -> int:
    """``x`` reduced mod 2^64 into the signed int64 range: NumPy's (and the
    C's ``hawk_i64_*``) two's-complement result."""
    return (x + 2**63) % 2**64 - 2**63


def _index_arith(a: int, b: int) -> dict:
    """``index_arith``'s outputs for one pair, exact (Python ints)."""
    return {"s": a + b - 3, "p": a * b, "ng": -a}


def _defined(planes: dict) -> dict:
    """The pairs of ``planes`` at which every ``index_arith`` result (and
    the intermediate ``a + b``) stays inside int64: the inputs whose C++
    behaviour is defined."""
    keep = [k for k, (a, b) in enumerate(zip(planes["a"].tolist(), planes["b"].tolist()))
            if all(_I64[0] <= v <= _I64[1]
                   for v in (a + b, *_index_arith(a, b).values()))]
    return {k: np.ascontiguousarray(v[keep]) for k, v in planes.items()}


def _defined_div(planes: dict) -> dict:
    """The pairs of ``planes`` whose ``a / b`` the C++ defines: no zero
    divisor and not ``MIN / -1``."""
    keep = [k for k, (a, b) in enumerate(zip(planes["a"].tolist(), planes["b"].tolist()))
            if b != 0 and not (a == _I64[0] and b == -1)]
    return {k: np.ascontiguousarray(v[keep]) for k, v in planes.items()}


def _cast_planes(mode: str) -> dict:
    import test_int_planes as I

    with np.errstate(over="ignore"):         # +-1e300 is +-inf in float32
        xs = np.array(_CAST_XS, dtype=np.float64).astype(_NP[mode])
    k = xs.shape[0]
    return {"x": xs, "n": np.resize(np.array(I.INT32_EDGES, np.int32), k),
            "u": np.resize(np.array(I.UINT32_EDGES, np.uint32), k)}


def _edge_values(case: str, mode: str) -> tuple[str, dict]:
    """``(kernel name, planes)`` of one edge-grid case: every input on its
    grid (a by-value real rounded to the mode's width first)."""
    import test_int_planes as I

    if case == "ops_i32":
        return case, _pairs(I.INT32_EDGES, np.int32)
    if case == "ops_u32":
        return case, _pairs(I.UINT32_EDGES, np.uint32)
    if case == "index_arith":
        # only where the C++ is defined: + - * and negation inside int64
        return case, _defined(_pairs(_INDEX_EDGES, np.int64))
    if case == "index_div":
        # only where the C++ is defined: no zero divisor, not MIN / -1
        return case, _defined_div(_pairs(_INDEX_EDGES, np.int64))
    if case == "index_divshift":
        return case, _pairs(_INDEX_EDGES, np.int64)
    if case.startswith("casts_s"):
        return "casts", {**_cast_planes(mode), "s": int(case.removeprefix("casts_s"))}
    if case == "scaled":
        return case, {**_cast_planes(mode), "a": float(_NP[mode](1.25))}
    raise KeyError(case)


EDGE_CASES = ("ops_i32", "ops_u32", "index_arith", "index_div", "index_divshift",
              "casts_s-5",
              "casts_s7",
              "casts_s2147483647", "scaled")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", EDGE_CASES)
def test_integer_kernels_match_the_host_bundle_on_their_edge_grids(
        host_units, c_units, case, mode):
    name, values = _edge_values(case, mode)
    so, report = c_units(_corpus()[name], mode)
    n = next(v for v in values.values() if isinstance(v, np.ndarray)).shape[-1]
    for row in _written(report):
        # every written plane supplied at its own dtype on both sides
        cells = (int(np.prod(row["shape"])), n) if row["shape"] else (n,)
        values.setdefault(row["name"], np.zeros(cells, dtype=E.element(row, mode)[1]))
    c_side, host_side = _copies(values), _copies(values)
    E.run_c(so, name, report, n, mode=mode, **c_side)
    _run_host(host_units(name, mode), n, host_side)
    _compare_written(f"{name} ({case})", mode, report, c_side, host_side)


#: The overflow points the host row leaves out, plus the int64 extremes so
#: a sum and a negation overflow too.
_OVERFLOW_EDGES = [*_INDEX_EDGES, 2**63 - 1, -(2**63)]


@pytest.mark.parametrize("mode", MODES)
def test_index_arithmetic_wraps_like_numpy_at_the_overflow_points(c_units, mode):
    """C only: at the inputs where the C++ build's signed overflow is
    undefined there is no host result to compare, so the C unit's
    ``hawk_i64_*`` results are pinned against NumPy's int64 wrapping
    arithmetic (mod 2^64 into the signed range)."""
    so, report = c_units(_corpus()["index_arith"], mode)
    planes = _pairs(_OVERFLOW_EDGES, np.int64)
    n = planes["a"].shape[0]
    for row in _written(report):
        planes[row["name"]] = np.zeros(n, dtype=np.int64)
    E.run_c(so, "index_arith", report, n, mode=mode, **planes)
    exact = [_index_arith(a, b) for a, b in zip(planes["a"].tolist(),
                                                planes["b"].tolist())]
    for out in ("s", "p", "ng"):
        want = np.array([_wrap64(e[out]) for e in exact], dtype=np.int64)
        overflowed = sum(not _I64[0] <= e[out] <= _I64[1] for e in exact)
        assert overflowed, f"index_arith.{out}: the grid reaches no overflow"
        assert_bit_identical(planes[out], want, kernel="index_arith", mode=mode,
                             plane=out)


@pytest.mark.parametrize("mode", MODES)
def test_index_division_is_defined_by_the_c_where_the_cpp_is_not(c_units, mode):
    """C only: a zero divisor and ``MIN / -1`` are undefined in the C++ build,
    so there is no host result; the C unit's ``hawk_i64_div`` returns 0 and
    ``MIN`` there (the defined results the ``defined_where_cpp_is_ub`` note
    and the docs state), and plain truncating division everywhere else."""
    so, report = c_units(_corpus()["index_div"], mode)
    lo = _I64[0]
    a = np.array([5, -5, lo, lo, 7, -7, lo], dtype=np.int64)
    b = np.array([0, 0, -1, 1, 2, 2, 2], dtype=np.int64)
    want = np.array([0, 0, lo, lo, 3, -3, lo // 2], dtype=np.int64)
    planes = {"a": a, "b": b, "q": np.zeros(a.shape[0], dtype=np.int64)}
    E.run_c(so, "index_div", report, a.shape[0], mode=mode, **planes)
    assert_bit_identical(planes["q"], want, kernel="index_div", mode=mode, plane="q")


#: Kernels whose own-column planes sit under a mask guard: each one's group-2
#: row must check at least one masked plane (pins the interface rule).
_GUARDED = ("diagnostic_two", "gp", "branch_finish", "two_fin")


@pytest.mark.parametrize("name", _GUARDED)
def test_a_guarded_plane_keeps_the_callers_value_at_a_masked_sample(
        host_units, c_units, name):
    mode = "float64"
    so, report = c_units(_corpus()[name], mode, _group(name)[1])
    values = _inputs(name, report, N, mode, np.random.default_rng(SEED))
    c_side, host_side = _copies(values), _copies(values)
    E.run_c(so, name, report, N, mode=mode, **c_side)
    _run_host(host_units(name, mode), N, host_side)
    assert _assert_masked_kept(name, mode, report, values,
                               {"C": c_side, "host": host_side}) >= 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["drag", "scatter"])
def test_a_partition_writes_its_window_alone_on_both_sides(host_units, c_units,
                                                           name, mode):
    n, base, count = 8, 2, 3
    kernel = _corpus()[name]
    so, report = c_units(kernel, mode)
    host = host_units(name, mode)
    values = _inputs(name, report, n, mode, np.random.default_rng(SEED))
    c_side, host_side = _copies(values), _copies(values)
    E.run_c(so, name, report, n, base=base, count=count, mode=mode, **c_side)
    _run_host(host, n, host_side, base=base, count=count)
    _compare_written(name, mode, report, c_side, host_side)
    if name == "drag":
        for side in (c_side, host_side):
            out = side["out"]
            assert not out[:, :base].any() and not out[:, base + count:].any(), (
                f"drag [{mode}]: a sample outside [{base}, {base + count}) was "
                "written")
            assert out[:, base:base + count].all()
    else:
        hit = set(values["lane"][base:base + count].tolist())
        idle = [j for j in range(n) if j not in hit]
        for side in (c_side, host_side):
            assert not side["acc"][idle].any(), (
                f"scatter [{mode}]: a lane no sample of [{base}, {base + count}) "
                "targets was written")
            assert side["acc"][sorted(hit)].any()


# --------------------------------------------------------------------------- #
# group 3: the C library against the numpy interpreter (no host artifact)
# --------------------------------------------------------------------------- #
def band(s: int, a, b, mode: str) -> float:
    """``S x 2 x eps(dtype)``, relative to the larger operand (and 1): the one
    anchor ``test_serial_oracle.band`` states, restated here so this file
    does not import another test module."""
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return s * 2.0 * _EPS[mode] * scale


def _ops(report) -> int:
    """``S``: the per-sample arithmetic, special and chain operations the
    verify pass counted (derived from the kernel, never fitted)."""
    ops = report["budgets"]["ops"]
    return max(1, int(sum(v for k, v in ops.items()
                          if k.startswith(("arith", "special", "chain")))))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kernel", [K.drag, K.chain6, D.vec3_scale],
                         ids=["drag", "chain6", "vec3_scale"])
def test_c_matches_the_numpy_interpreter_within_band(c_units, kernel, mode):
    from _eval import evaluate

    so, report = c_units(kernel, mode)
    rng = np.random.default_rng(SEED)
    values = {}
    for row in report["interface"]:
        if row["role"] == "uniform":
            # chain6 takes sqrt(k): its Param stays positive
            values[row["name"]] = float(_NP[mode](rng.uniform(0.25, 2.0)))
        elif row["direction"] == "in":
            cells = (int(np.prod(row["shape"])), N) if row["shape"] else (N,)
            values[row["name"]] = rng.uniform(-2.0, 2.0, cells).astype(_NP[mode])
    got = _tuple(E.run_c(so, kernel.name, report, N, mode=mode, **values))
    rows = _written(report)
    s = _ops(report)
    for i in range(N):
        env = {k: (v[..., i].astype(np.float64) if isinstance(v, np.ndarray) else v)
               for k, v in values.items()}
        want = evaluate(kernel.sinks, env, lanes=N, lane=i)
        for row, plane in zip(rows, got):
            g = np.asarray(plane[..., i], dtype=np.float64)
            w = np.asarray(want[row["name"]], dtype=np.float64)
            limit = band(s, g, w, mode)
            worst = float(np.max(np.abs(g - w)))
            assert worst <= limit, (
                f"{kernel.name} [{mode}] plane {row['name']!r}, sample {i}: the C "
                f"result {g} differs from the interpreter's {w} by {worst}, band "
                f"S*2*eps*scale = {limit} (S = {s})")
