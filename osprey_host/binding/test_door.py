"""Checks of the binding's door calls, run from test_binding.py on a host context.

The door is the runtime of the opened cpu library: every Door is made from a backend handle the
caller opened. The units come from the door's own fixture, built beside the extension as
libfixture_unit.<dylib or so>. numpy is the oracle.
"""

import gc
import math
import os
import platform
import struct
import sys
import weakref

import numpy as np

import _osprey as o

EXT = "dylib" if sys.platform == "darwin" else "so"
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "build", f"libfixture_unit.{EXT}")
REVERSE = os.path.join(HERE, "build", f"libfixture_reverse.{EXT}")
GRAPH = os.path.join(HERE, "build", f"libfixture_graph.{EXT}")
GEMM = os.path.join(HERE, "build", f"libfixture_gemm.{EXT}")
# the unit libraries with device images are built by the door's own device build; they are
# read only on a device context, which a library without a device group never makes
DEVICE_FIXTURE = os.path.join(HERE, "build", f"libfixture_device.{EXT}")
DEVICE_GEMM = os.path.join(HERE, "build", f"libfixture_gemm.{EXT}")
N = 1000


def digest(byte):
    """The 32-byte digest of a fixture unit: its one byte repeated."""
    return bytes([byte]) * o.EG_DIGEST_BYTES


def uniform(value):
    """A float64 uniform as the bytes of its slot's dtype."""
    return struct.pack("<d", value)


def refusal_text(door, error):
    """The reason an exception carries, which must be what the door stored in the context."""
    assert str(error) and str(error) == door.error(), (str(error), door.error())
    return str(error)


class Export:
    """A DLPack producer over a capsule the door made, as numpy's from_dlpack takes it."""

    def __init__(self, capsule):
        """Hold the capsule."""
        self.capsule = capsule

    def __dlpack__(self, **options):
        """Give the capsule."""
        return self.capsule

    def __dlpack_device__(self):
        """Say host memory (kDLCPU)."""
        return (1, 0)


def product(a, b, flags, c_before):
    """The oracle of C = op(A) op(B) over stored arrays, accumulated or not, with its bound.

    Returns the float64 product and the largest difference a float64 or float32 implementation may
    show from it per element. A sum of k terms is rounded k - 1 times, each by at most eps / 2 of
    the sum of the terms' magnitudes, on both sides of the comparison (the implementation and this
    float64 oracle): (k - 1) * eps * sum |a * b|. Accumulating adds one term, the one in C.
    """
    left = a.T if flags & o.EG_GEMM_TRANS_A else a
    right = b.T if flags & o.EG_GEMM_TRANS_B else b
    left, right = left.astype(np.float64), right.astype(np.float64)
    result, magnitude = left @ right, np.abs(left) @ np.abs(right)
    if flags & o.EG_GEMM_TRANS_C:
        result, magnitude = result.T, magnitude.T
    terms = left.shape[1]
    if flags & o.EG_GEMM_ACCUMULATE:
        result = result + c_before
        magnitude = magnitude + np.abs(c_before)
        terms += 1
    return result, (terms - 1) * np.finfo(a.dtype).eps * magnitude


def within(got, want, bound):
    """True when every element of `got` is within its bound of the oracle `want`."""
    return bool((np.abs(got.astype(np.float64) - want) <= bound).all())


def to_device(door, array):
    """A device plane holding a copy of `array`; the door's copy returns once the data is there."""
    plane = door.plane_new(list(array.shape), 2, array.dtype.itemsize * 8, kind=o.EG_MEM_DEVICE)
    host = door.plane_wrap(array)
    door.copy(host, plane)
    door.plane_free(host)
    return plane


def from_device(door, plane, like):
    """The values of a device plane as a new array shaped and typed like `like`."""
    out = np.zeros_like(like)
    host = door.plane_wrap(out)
    door.copy(plane, host)
    door.plane_free(host)
    return out


def check_gemm(door, done, raises):
    """Rows of the matrix-product calls, and of the units that state products, against numpy."""
    rng = np.random.default_rng(11)
    m, k, n = 3, 5, 4
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        for flags in range(16):
            a_shape = (k, m) if flags & o.EG_GEMM_TRANS_A else (m, k)
            b_shape = (n, k) if flags & o.EG_GEMM_TRANS_B else (k, n)
            c_shape = (n, m) if flags & o.EG_GEMM_TRANS_C else (m, n)
            a, b = (rng.standard_normal(s).astype(dtype) for s in (a_shape, b_shape))
            c = rng.standard_normal(c_shape).astype(dtype)
            want, bound = product(a, b, flags, c.copy())
            gemm = door.gemm_new(a_shape, b_shape, c_shape, bits, flags)
            planes = [door.plane_wrap(x) for x in (a, b, c)]
            door.gemm_run(gemm, *planes)
            assert within(c, want, bound), (bits, flags)
            door.gemm_run(gemm, *planes)  # a second run of the plan; accumulating adds again
            for plane in planes:
                door.plane_free(plane)
            door.gemm_free(gemm)
    done.append("gemm_new, gemm_run, gemm_free: 32 products (f32, f64, every flag) equal numpy")

    wide_a, b, wide_c = (
        rng.standard_normal((3, 7)),
        rng.standard_normal((5, 4)),
        np.full((3, 6), 9.0),
    )
    gemm = door.gemm_new((3, 5), (5, 4), (3, 4), 64)
    planes = [door.plane_wrap(x) for x in (wide_a, b, wide_c)]
    door.gemm_run(gemm, *planes)
    want, bound = product(wide_a[:, :5], b, 0, None)
    assert within(wide_c[:, :4], want, bound) and (wide_c[:, 4:] == 9.0).all()
    done.append("gemm_run: wider planes serve a column slice and leave the rest alone")

    a, b, c = rng.standard_normal((3, 5)), rng.standard_normal((5, 4)), np.zeros((3, 4))
    pa, pb, pc = (door.plane_wrap(x) for x in (a, b, c))
    door.record_begin()
    door.gemm_run(gemm, pa, pb, pc)
    assert not c.any()
    graph = door.record_end()
    door.gemm_free(gemm)
    door.graph_launch(graph)
    assert within(c, *product(a, b, 0, None))
    door.graph_free(graph)
    done.append("gemm_run recorded: runs on replay, and gemm_free before the replay is safe")

    gemm = door.gemm_new((3, 5), (5, 4), (3, 4), 64)

    def run_with(a_array=a, c_array=c, plane_c=None):
        """Run the product over arrays that break one rule."""
        pa2 = door.plane_wrap(a_array)
        door.gemm_run(gemm, pa2, pb, plane_c or door.plane_wrap(c_array))

    read_only = np.zeros((3, 4))
    read_only.flags.writeable = False
    for what, call, fragment in (
        (
            "dtype_bits 16",
            lambda: door.gemm_new((3, 5), (5, 4), (3, 4), 16),
            "is not a float width",
        ),
        ("an extent of 0", lambda: door.gemm_new((0, 5), (5, 4), (3, 4), 64), "is outside 1 .. "),
        (
            "an extent past 2^31 - 1",
            lambda: door.gemm_new((3, 1 << 31), (5, 4), (3, 4), 64),
            "outside",
        ),
        (
            "contracted extents",
            lambda: door.gemm_new((3, 5), (4, 4), (3, 4), 64),
            "contracted extents",
        ),
        (
            "a C stored otherwise",
            lambda: door.gemm_new((3, 5), (5, 4), (4, 3), 64),
            "is stored (4, 3)",
        ),
        (
            "an unknown flag",
            lambda: door.gemm_new((3, 5), (5, 4), (3, 4), 64, 1 << 7),
            "unknown bits",
        ),
    ):
        e = raises(o.BadRequest, call, f"gemm_new: {what}")
        assert fragment in refusal_text(door, e), str(e)
    for what, call, fragment in (
        ("a plane of the wrong rank", lambda: run_with(a_array=np.zeros(15)), "must be 2-D"),
        ("a plane of the wrong rows", lambda: run_with(a_array=np.zeros((4, 5))), "with 3 rows"),
        (
            "a row stride below the columns",
            lambda: run_with(a_array=np.zeros((3, 4))),
            "row stride",
        ),
        (
            "a plane of the wrong width",
            lambda: run_with(a_array=np.zeros((3, 5), np.float32)),
            "float64",
        ),
        ("a read-only output", lambda: run_with(c_array=read_only), "read-only"),
        ("an output over an input", lambda: door.gemm_run(gemm, pa, pb, pa), "overlaps"),
    ):
        e = raises(o.BadRequest, call, f"gemm_run: {what}")
        assert fragment in refusal_text(door, e), str(e)
    door.record_begin()
    e = raises(
        o.BadRequest, lambda: door.gemm_new((3, 5), (5, 4), (3, 4), 64), "gemm_new: recording"
    )
    assert "while recording" in refusal_text(door, e)
    door.graph_free(door.record_end())
    raises(TypeError, lambda: door.gemm_run(pa, pa, pb, pc), "gemm_run: a plane for a product")
    door.gemm_free(gemm)
    raises(ValueError, lambda: door.gemm_free(gemm), "gemm_free: a second free of one handle")
    done.append(
        "gemm refusals: each stated sentence of eg_gemm_new and eg_gemm_run by its stable part"
    )

    # units that state matrix products: the host entry is the oracle of the record, numpy of both
    shapes = {
        "sq": (0, (4, 4), (4, 4), (4, 4)),
        "rect": (0, (3, 5), (5, 4), (3, 4)),
        "ta": (o.EG_GEMM_TRANS_A, (5, 3), (5, 4), (3, 4)),
        "tb": (o.EG_GEMM_TRANS_B, (3, 5), (4, 5), (3, 4)),
        "tc": (o.EG_GEMM_TRANS_C, (3, 5), (5, 4), (4, 3)),
        "all": (7, (5, 3), (4, 5), (4, 3)),
        "lane": (0, (3, 5), (5, 4), (3, 4)),  # columns are n_samples / lane_fanin: 12 / 3
        "wide": (0, (3, 5), (5, 4), (3, 4)),
    }
    for bits, base, dtype in ((64, 0x30, np.float64), (32, 0x40, np.float32)):
        for index, (name, (flags, a_s, b_s, c_s)) in enumerate(shapes.items()):
            slack = 2 if name == "wide" else 0
            full = [(r, c + slack) for r, c in (a_s, b_s, c_s)]
            a, b, c = (rng.standard_normal(s).astype(dtype) for s in full)
            c[:] = 5.0
            unit = door.unit_load(GEMM, f"f{bits}_{name}", digest(base + index))
            plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_TEAM, o.EG_REDUCE_COMPENSATED, 12)
            planes = [door.plane_wrap(x) for x in (a, b, c)]
            door.bind(plan, [0, 1, 2], planes)
            assert door.plan_entry_figures(plan)["flags"] & o.EG_ENTRY_GEMM == 0
            door.launch(plan)
            cut = lambda x, s: x[: s[0], : s[1]]  # noqa: E731
            assert within(cut(c, c_s), *product(cut(a, a_s), cut(b, b_s), flags, None)), name
            for plane in planes:
                door.plane_free(plane)
            door.plan_free(plan)
            door.unit_free(unit)
    done.append(
        "gemm units: all eight shapes, f32 and f64, equal numpy; no gemm entry on a host plan"
    )

    for bits, byte, dtype in ((64, 0x50, np.float64), (32, 0x51, np.float32)):
        w, x, dy = (rng.standard_normal(s).astype(dtype) for s in ((3, 5), (5, 4), (3, 4)))
        dw, dx = np.zeros((3, 5), dtype), rng.standard_normal((5, 4)).astype(dtype)
        before = dx.copy()
        unit = door.unit_load(GEMM, f"f{bits}_vjp", digest(byte))
        plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 1)
        planes = [door.plane_wrap(v) for v in (w, x, dy, dw, dx)]
        door.bind(plan, [0, 1, 2, 3, 4], planes)
        door.launch(plan)
        assert within(dw, *product(dy, x, o.EG_GEMM_TRANS_B, None))
        assert within(dx, *product(w, dy, o.EG_GEMM_TRANS_A | o.EG_GEMM_ACCUMULATE, before))
    done.append(
        "gemm unit with two calls: dW = dY X^T, dX += W^T dY (ACCUMULATE over a prefilled plane)"
    )

    e = raises(
        o.NotWritten,
        lambda: door.unit_load(GEMM, "bad_banded", digest(0x67)),
        "unit_load: a banded unit with a gemm record",
    )
    assert "banded unit with a gemm record" in refusal_text(door, e)
    e = raises(
        o.BadRequest,
        lambda: door.unit_load(GEMM, "bad_extent", digest(0x64)),
        "unit_load: a gemm call extent of 0",
    )
    assert "extent 0" in refusal_text(door, e)
    done.append("gemm record refusals at load: a banded unit, an extent of 0")

    # banded cells ride as unsigned 64-bit planes, never as doubles
    cells = np.array(door.banded_pack([0.5, -2.0, 3.25]), dtype=np.uint64)
    plane = door.plane_wrap(cells)
    door.plane_free(plane)
    made = door.plane_new([3], 1, 64, flags=o.EG_PLANE_ZERO)
    view = np.from_dlpack(Export(door.plane_dlpack(made)))
    assert view.dtype == np.uint64
    view[:] = cells
    assert door.banded_unpack(view.tolist()) == (0.5, -2.0, 3.25)
    door.plane_free(made)
    done.append(
        "banded planes: packed cells wrap and export as uint64 planes and unpack to the values"
    )


def check_segmented(door, done, raises):
    """Rows of a runner over a SEGMENTED group, on the door's static and dynamic fixtures."""
    samples = 200

    def load_group(prefix, first, branches, extras, fused):
        """The units of a group (branch 0 last, with the others in deps) and its fused unit."""
        others = [door.unit_load(FIXTURE, name, digest(byte)) for name, byte in branches + extras]
        head = door.unit_load(FIXTURE, f"{prefix}__k0", digest(first), deps=others)
        return head, door.unit_load(FIXTURE, f"{prefix}f", digest(fused))

    class Set:
        """The planes of one run over a unit: k, limit, mask and, for the dynamic group, pos."""

        def __init__(self, unit, dynamic, with_pos):
            """Bind fresh inputs to a plan of the unit."""
            index = np.arange(samples)
            self.k = 0.02 * ((index * 37) % samples) if dynamic else np.zeros(samples)
            self.limit = np.full(samples, 7.0) if dynamic else (2 + index % 9).astype(float)
            self.mask = np.zeros(samples, dtype=np.bool_)
            self.pos = np.full(samples, -1.0)
            arrays, slots = [self.k, self.limit, self.mask], [0, 1, 2]
            if with_pos:
                arrays, slots = arrays + [self.pos], slots + [7]
            self.plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_TEAM, 0, samples)
            door.bind(self.plan, slots, [door.plane_wrap(a) for a in arrays])

        def launch_until_done(self, max_steps):
            """Launch the plan by hand with the runner's finish rule; answer the launches made."""
            launches = 0
            while launches < max_steps and not self.mask.all():
                door.launch(self.plan)
                launches += 1
            return launches

        def same(self, other, with_pos):
            """Bit-for-bit equal planes."""
            same = np.array_equal(self.k, other.k) and np.array_equal(self.mask, other.mask)
            return same and (not with_pos or np.array_equal(self.pos, other.pos))

    static = load_group("sg", 0xE2, [("sg__k1", 0xE3)], [("sg__cls", 0xE1)], 0xE4)
    dynamic = load_group("dg", 0xE5, [("dg__k1", 0xE6)], [("dg__sweep", 0xE7)], 0xE8)
    for name, (head, fused), is_dynamic in (("static", static, False), ("dynamic", dynamic, True)):
        hand, run, fus = (
            Set(u, is_dynamic, w) for u, w in ((head, is_dynamic),) * 2 + ((fused, False),)
        )
        want = hand.launch_until_done(1000)
        assert want > 1 and hand.mask.all(), (name, want)
        # the runner and the hand launches: the same compiled units launched two ways, bit for bit
        runner = door.runner_new([run.plan], 1000)
        report = door.run(runner)
        assert run.same(hand, is_dynamic) and report["launches"] == want, (name, report)
        assert report["steps"] == want and report["finished"] == samples and report["mode"] == 0
        assert report["flags"] == o.EG_REPORT_DONE | o.EG_REPORT_EXACT_STEPS
        assert report["mapped"] == want * samples
        fused_report = door.run(door.runner_new([fus.plan], 1000))
        # the fused unit is a different computation of k: the one rule is |k_fused - k_group| <=
        # n * eps * M. In the fixture (door test folder, fixture_unit.c) every launch adds one step
        # to each live sample (seg_step runs once per sample and launch: its branch, or the sweep
        # for a stale slot), so n is the launches made; a step is at most 2.0 in the static group
        # (branch j adds j + 1) and 0.125 * (class + 1) <= 0.25 in the dynamic one; k stays below
        # its limit before its last addition, so M is the largest limit plus the largest step.
        step_max = 0.25 if is_dynamic else 2.0
        bound = want * np.finfo(np.float64).eps * (hand.limit.max() + step_max)
        assert np.abs(fus.k - hand.k).max() <= bound, (name, np.abs(fus.k - hand.k).max(), bound)
        assert fused_report["launches"] == want, name
        run.k[:] = Set(head, is_dynamic, False).k
        run.mask[:] = False
        run.pos[:] = -1.0
        door.runner_reset(runner)
        again = door.run(runner)
        assert run.same(hand, is_dynamic) and again == report, (name, again, report)
        fresh = Set(head, is_dynamic, is_dynamic)
        assert door.run(door.runner_new([fresh.plan], 1000)) == report and fresh.same(
            hand, is_dynamic
        )
        cut_hand, cut = Set(head, is_dynamic, is_dynamic), Set(head, is_dynamic, is_dynamic)
        part = cut_hand.launch_until_done(3)
        cut_report = door.run(door.runner_new([cut.plan], 3))
        assert cut.same(cut_hand, is_dynamic) and cut_report["launches"] == part
        assert cut_report["finished"] < samples and cut_report["flags"] == o.EG_REPORT_EXACT_STEPS
    done.append("a SEGMENTED group under a runner (static, dynamic): hand launches and fused unit")
    done.append(
        "runner_reset over a SEGMENTED group: the second run equals the first and a fresh runner"
    )

    head, _ = static
    group = Set(head, False, False)
    axpb = door.unit_load(FIXTURE, "axpb", digest(0xA1))
    other = door.plan_new(axpb, o.EG_STRUCTURE_HOST_TEAM, 0, samples)
    for what, call, kind, fragment in (
        (
            "beside another plan",
            lambda: door.runner_new([group.plan, other], 10),
            o.NotWritten,
            "the whole step of its runner",
        ),
        (
            "REORDER",
            lambda: door.runner_new([group.plan], 10, flags=o.EG_RUN_REORDER),
            o.BadRequest,
            "nothing to reorder",
        ),
        (
            "every",
            lambda: door.runner_new([group.plan], 10, every=4),
            o.BadRequest,
            "compaction does not apply",
        ),
        (
            "events REFINE",
            lambda: door.runner_new([group.plan], 10, events=o.EG_EVENTS_REFINE),
            o.NotWritten,
            "is not recorded or refined",
        ),
        (
            "RECORD",
            lambda: door.runner_new(
                [group.plan],
                10,
                flags=o.EG_RUN_RECORD,
                derived_roles=[o.EG_DERIVED_TAPE],
                derived_units=[axpb],
            ),
            o.NotWritten,
            "is not recorded or refined",
        ),
    ):
        e = raises(kind, call, f"runner_new: a SEGMENTED group, {what}")
        assert fragment in refusal_text(door, e), str(e)
    done.append("SEGMENTED runner refusals: several plans, REORDER, every, REFINE, RECORD")


def check_auto_set(door, done, raises):
    """The segmented set of an AUTO group as the lowering writes it loads; a torn set is refused."""
    import re
    import subprocess
    import tempfile

    source = open(os.path.join(HERE, "fixture", "auto_set_meta.c")).read()
    entries = re.findall(r'M\(host_entry, "([^"]+)"\)', source)
    root = os.environ["OSPREY_EAGLE_SRC"]
    # the meta carries no code: the entries are empty functions, and the size block is the one every
    # unit library exports (GRefMirror, ScalarHandle, IntHandle, the index type, the triple)
    stubs = "".join(f"void {name}(void) {{}}\n" for name in entries) + (
        '#include <stdint.h>\nconst char eagle_abi_tag[] = "aether-abi/2";\n'
        "const uint64_t eagle_layout_sizes[5] = {40, 32, 32, 4, 24};\n"
    )
    flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
    names = 'M(name, "dispatch0"), M(k, 3), M(reserved0, 0), M(branches, u0_d0_branches)'
    classify = "M(classify, &u0_d0_classify), " + names

    def swap(old, new):
        assert old in source, old
        return source.replace(old, new, 1)

    def other_group(text=source):
        head = text.index("static const struct eg_unit_dispatch u2_d0")
        end = text.index("\n", head)
        return text[:head] + text[head:end].replace('"dispatch0"', '"dispatch1"') + text[end:]

    torn = {
        "branches without a classify unit": swap(
            classify, classify.replace("&u0_d0_classify", "NULL")
        ),
        "a classify unit without branches": swap(
            classify, classify.replace("u0_d0_branches", "NULL")
        ),
        "branches without costs": swap("M(branch_costs, u0_d0_costs)", "M(branch_costs, NULL)"),
        "a branch record of another group": other_group(),
    }

    def load_set(directory, tag, text):
        """Loads the set in the order the door plans a group.

        The units other than branch 0 come first, branch 0 follows with them in deps, and the
        fused unit last with the whole group in deps.
        """
        base = os.path.join(directory, tag)
        open(base + ".c", "w").write(text)
        open(base + "_stubs.c", "w").write(stubs)
        library = base + (".dylib" if sys.platform == "darwin" else ".so")
        cc = ["cc", *flags, "-std=c11", "-w", "-I", root, base + ".c", base + "_stubs.c"]
        subprocess.check_call([*cc, "-o", library])
        found = re.findall(r'M\(name, "([^"]+)"\), M\(digest, \{([^}]*)\}\), M\(pattern', text)
        digests = {name: bytes(int(x, 16) for x in raw.split(",")) for name, raw in found}

        def load(name, deps=()):
            return door.unit_load(library, name, digests[name], deps=list(deps))

        others = [load(name) for name in ("mix_auto__k1", "mix_auto__k2", "mix_auto__cls")]
        head = load("mix_auto__k0", others)
        return load("mix_auto", [head] + others)

    with tempfile.TemporaryDirectory() as directory:
        assert load_set(directory, "whole", source) is not None
        done.append("unit_load: an AUTO group naming its segmented set (branches, classify, costs)")
        for what, text in torn.items():
            tag = what.replace(" ", "_")
            e = raises(
                o.BadRequest,
                lambda: load_set(directory, tag, text),
                f"unit_load: a torn set, {what}",
            )
            assert "contradicts itself" in str(e) and "dispatch" in str(e), str(e)


def check_fft(door, done, raises):
    """Rows of the transform calls; numpy.fft is the oracle, for every kind, norm and width."""
    rng = np.random.default_rng(7)
    norms = {o.EG_FFT_NORM_BACKWARD: "backward", o.EG_FFT_NORM_FORWARD: "forward"}
    norms[o.EG_FFT_NORM_ORTHO] = "ortho"

    def planes(source, out):
        """Plane handles over two numpy arrays, as float planes."""
        return door.plane_wrap(source), door.plane_wrap(out)

    def close(got, want, dtype):
        """Equal to the oracle within the float width's rounding."""
        scale = max(1.0, float(np.abs(want).max()))
        assert np.abs(got - want).max() <= (1e-4 if dtype == np.float32 else 1e-11) * scale

    count = 0
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        for shape, axes in (((6, 10), (1,)), ((7, 12), (0, 1)), ((9,), (0,)), ((5, 7), (0,))):
            z = rng.standard_normal(shape + (2,)).astype(dtype)
            zc = z[..., 0] + 1j * z[..., 1]
            for norm, name in norms.items():
                for inverse in (0, o.EG_FFT_INVERSE):
                    out = np.zeros_like(z)
                    plan = door.fft_new(o.EG_FFT_C2C, norm, bits, shape, axes, inverse)
                    pin, pout = planes(z, out)
                    door.fft_run(plan, pin, pout)
                    fn = np.fft.ifftn if inverse else np.fft.fftn
                    want = fn(zc, axes=axes, norm=name)
                    close(out[..., 0] + 1j * out[..., 1], want, dtype)
                    count += 1
                    z2 = rng.standard_normal(z.shape).astype(dtype)  # a second run of the plan
                    z[:] = z2
                    door.fft_run(plan, pin, pout)
                    want = fn(z2[..., 0] + 1j * z2[..., 1], axes=axes, norm=name)
                    close(out[..., 0] + 1j * out[..., 1], want, dtype)
                    z = z.copy()
                    for handle in (pin, pout):
                        door.plane_free(handle)
                    door.fft_free(plan)
                    zc = z[..., 0] + 1j * z[..., 1]
    done.append(
        f"fft_new, fft_run, fft_free: {count} C2C plans equal numpy.fft, a second run, free"
    )
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        for shape, axes in (((4, 9), (1,)), ((6, 10), (0, 1)), ((8, 7), (1, 0)), ((11,), (0,))):
            x = rng.standard_normal(shape).astype(dtype)
            halved = list(shape)
            halved[axes[-1]] = shape[axes[-1]] // 2 + 1
            spec = np.zeros(tuple(halved) + (2,), dtype=dtype)
            for norm, name in norms.items():
                plan = door.fft_new(o.EG_FFT_R2C, norm, bits, shape, axes)
                pin, pout = planes(x, spec)
                door.fft_run(plan, pin, pout)
                want = np.fft.rfftn(x, axes=axes, norm=name)
                close(spec[..., 0] + 1j * spec[..., 1], want, dtype)
                back = np.zeros(shape, dtype=dtype)
                keep = spec.copy()
                inverse = door.fft_new(o.EG_FFT_C2R, norm, bits, shape, axes)
                sp, pback = planes(spec, back)
                door.fft_run(inverse, sp, pback)
                assert np.array_equal(spec, keep)
                sizes = [shape[a] for a in axes]
                close(back, np.fft.irfftn(want, s=sizes, axes=axes, norm=name), dtype)
                close(back, x, dtype)
                for handle in (pin, pout, sp, pback):
                    door.plane_free(handle)
                door.fft_free(plan)
                door.fft_free(inverse)
    done.append("fft R2C and C2R on one and two axes, even and odd lengths, f32 and f64: numpy.fft")

    z = rng.standard_normal((5, 6, 2))
    keep = z.copy()
    plan = door.fft_new(o.EG_FFT_C2C, o.EG_FFT_NORM_BACKWARD, 64, [5, 6], [0, 1])
    pz = door.plane_wrap(z)
    door.fft_run(plan, pz, pz)
    want = np.fft.fft2(keep[..., 0] + 1j * keep[..., 1])
    close(z[..., 0] + 1j * z[..., 1], want, np.float64)
    done.append("fft_run: a C2C plan runs in place")

    def new(kind=o.EG_FFT_C2C, bits=64, shape=(4,), axes=(0,), flags=0):
        """A plan description with one thing changed."""
        return lambda: door.fft_new(
            kind, o.EG_FFT_NORM_BACKWARD, bits, list(shape), list(axes), flags
        )

    for what, call, kind in (
        ("an axis of length 0", new(shape=(0,)), o.BadRequest),
        ("no axes", new(axes=()), o.BadRequest),
        ("a repeated axis", new(shape=(4, 4), axes=(0, 0)), o.BadRequest),
        ("an axis out of range", new(axes=(1,)), o.BadRequest),
        ("a negative extent", new(shape=(-3,)), o.BadRequest),
        ("INVERSE on R2C", new(kind=o.EG_FFT_R2C, flags=o.EG_FFT_INVERSE), o.BadRequest),
        ("a float width of 24", new(bits=24), o.BadRequest),
        ("an unknown flag", new(flags=1 << 9), o.BadRequest),
    ):
        refusal_text(door, raises(kind, call, f"fft_new: {what}"))
    for bits in (16, 128):
        e = raises(o.NotWritten, new(bits=bits), f"fft_new: a float width of {bits}")
        refusal_text(door, e)
    done.append(
        "fft_new refusals: length 0, no axes, a repeated or foreign axis, INVERSE on R2C, widths"
    )

    good = door.fft_new(o.EG_FFT_R2C, o.EG_FFT_NORM_BACKWARD, 64, [4, 6], [1])
    real = door.plane_wrap(np.zeros((4, 6)))
    spec_ok = door.plane_wrap(np.zeros((4, 4, 2)))
    spec_bad = door.plane_wrap(np.zeros((4, 3, 2)))
    narrow = door.plane_wrap(np.zeros((4, 4, 2), dtype=np.float32))
    e = raises(o.BadRequest, lambda: door.fft_run(good, real, spec_bad), "fft_run: a wrong shape")
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.fft_run(good, real, narrow), "fft_run: a wrong width")
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.fft_run(good, real, real), "fft_run: R2C in place")
    refusal_text(door, e)
    door.fft_run(good, real, spec_ok)
    done.append("fft_run refusals: a wrong shape, a wrong float width, a real transform in place")

    # a recording keeps the plan: free it after the recording, replay still runs
    x = rng.standard_normal((4, 6))
    xin, spec = door.plane_wrap(x), np.zeros((4, 4, 2))
    pspec = door.plane_wrap(spec)
    door.record_begin()
    e = raises(o.BadRequest, new(), "fft_new: while recording")
    assert "while recording" in refusal_text(door, e)
    door.fft_run(good, xin, pspec)
    assert not spec.any()
    graph = door.record_end()
    door.fft_free(good)
    door.graph_launch(graph)
    want = np.fft.rfftn(x, axes=(1,))
    close(spec[..., 0] + 1j * spec[..., 1], want, np.float64)
    door.graph_free(graph)
    raises(ValueError, lambda: door.fft_free(good), "fft_free: a second free of one handle")
    done.append("fft_new refused while recording; a recorded run replays after fft_free")


def check_graph_reverse(door, done, raises):
    """Rows of record_end(diff) and graph_reverse; the closed forms in numpy are the oracle."""
    n = 7
    scale, scale_vjp = (
        door.unit_load(GRAPH, name, digest(b)) for name, b in (("scale", 0x10), ("scale_vjp", 0x11))
    )
    square, square_vjp = (
        door.unit_load(GRAPH, name, digest(b))
        for name, b in (("square", 0x12), ("square_vjp", 0x13))
    )
    vjps = [(0, scale_vjp), (1, square_vjp)]
    x0 = 0.5 + 0.3 * np.arange(n)

    def rig(second_reads):
        """Plans y = w x and a second launch over `second_reads` (planes by slot), recorded."""
        arrays = {k: np.zeros(n) for k in ("y", "z", "gx", "ones")}
        arrays.update(x=x0.copy(), w=np.array([1.7]), ones=np.ones(n), gw=np.zeros(1))
        planes = {k: door.plane_wrap(a) for k, a in arrays.items()}
        first = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(first, [0, 1, 2], [planes[k] for k in ("x", "w", "y")], [], [])
        second = door.plan_new(square, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(second, [0, 1], [planes[k] for k in second_reads], [], [])
        door.record_begin()
        door.launch(first)
        door.launch(second)
        return arrays, planes, first, second

    def release(planes, *plans, graph=None):
        for handle in plans:
            door.plan_free(handle)
        for plane in planes.values():
            door.plane_free(plane)
        if graph is not None:
            door.graph_free(graph)

    # y = w x, z = y^2, J = sum z: dJ/dx = 2 w^2 x, dJ/dw = 2 w sum x^2
    arrays, planes, first, second = rig(("y", "z"))
    graph = door.record_end(vjps)
    door.graph_launch(graph)
    door.graph_reverse(
        graph,
        [(planes["z"], planes["ones"])],
        [(planes["x"], planes["gx"]), (planes["w"], planes["gw"])],
    )
    w = 1.7
    assert np.allclose(arrays["gx"], 2 * w * w * x0, rtol=1e-12)
    assert np.allclose(arrays["gw"], 2 * w * np.sum(x0 * x0), rtol=1e-12)
    done.append(
        "record_end(diff), graph_reverse: y = w x, z = y^2 gives dJ/dx and dJ/dw as the closed form"
    )

    # a rebind of a plan of the recorded graph: the reverse is refused by the door's text
    door.bind(first, [0], [planes["x"]], [], [])
    e = raises(
        o.BadRequest,
        lambda: door.graph_reverse(
            graph, [(planes["z"], planes["ones"])], [(planes["x"], planes["gx"])]
        ),
        "graph_reverse: after a rebind",
    )
    assert "rebound" in refusal_text(door, e), str(e)
    release(planes, first, second, graph=graph)
    done.append("graph_reverse: a graph whose plan was rebound is refused by name")

    # x <- y^2 in place after y = w x read x: the snapshot gives the gradient at the original x
    arrays, planes, first, second = rig(("y", "x"))
    graph = door.record_end(vjps)
    door.graph_launch(graph)
    door.graph_reverse(
        graph,
        [(planes["x"], planes["ones"])],
        [(planes["x"], planes["gx"]), (planes["w"], planes["gw"])],
    )
    assert np.allclose(arrays["gx"], 2 * w * w * x0, rtol=1e-12)
    assert np.allclose(arrays["gw"], 2 * w * np.sum(x0 * x0), rtol=1e-12)
    release(planes, first, second, graph=graph)
    done.append(
        "graph_reverse: a launch that overwrites a plane an earlier launch read is reversed "
        "at the original values"
    )

    # a graph ended without a description is plain: it launches and graph_reverse refuses it
    arrays, planes, first, second = rig(("y", "z"))
    graph = door.record_end()
    door.graph_launch(graph)
    assert np.allclose(arrays["z"], (w * x0) ** 2)
    e = raises(
        o.BadRequest,
        lambda: door.graph_reverse(
            graph, [(planes["z"], planes["ones"])], [(planes["x"], planes["gx"])]
        ),
        "graph_reverse: a plain graph",
    )
    assert "not ended with EG_RECORD_DIFF" in refusal_text(door, e), str(e)
    release(planes, first, second, graph=graph)
    for what, call in (
        ("a pair of three", lambda: door.record_end([(0, scale_vjp, 1)])),
        ("a plane in place of a unit", lambda: door.record_end([(0, door.plane_wrap(x0))])),
    ):
        raises(TypeError, call, f"record_end: {what}")
    done.append(
        "record_end with no description: the graph launches as before; graph_reverse refuses it"
    )
    for unit in (scale, scale_vjp, square, square_vjp):
        door.unit_free(unit)


def check_fork_and_cond(door, done, raises):
    """Rows of forks, waits, the conditional's fields and graph_describe; numpy gives the values."""

    def cells(*values):
        """Int32 arrays of the values, and a plane over each."""
        arrays = [np.array([v], dtype=np.int32) for v in values]
        return arrays, [door.plane_wrap(a) for a in arrays]

    def end_replay():
        """End the recording and replay it once."""
        graph = door.record_end()
        door.graph_launch(graph)
        return graph

    # two branches over disjoint planes, then the join
    arrays, (s1, s2, d1, d2) = cells(11, 22, 0, 0)
    door.record_begin()
    door.record_fork_begin()
    door.copy(s1, d1)
    door.record_fork_branch()
    door.copy(s2, d2)
    door.record_fork_end()
    graph = door.record_end()
    assert arrays[2][0] == 0 and arrays[3][0] == 0
    door.graph_launch(graph)
    assert (arrays[2][0], arrays[3][0]) == (11, 22)
    door.graph_free(graph)
    done.append(
        "record_fork_*: two disjoint branches then a join replay to the recording-order values"
    )

    # a race refused at fork_end with the door's text; a wait resolves it
    door.record_begin()
    door.record_fork_begin()
    door.copy(s1, d1)
    door.record_fork_branch()
    door.copy(d1, d2)
    e = raises(o.BadRequest, door.record_fork_end, "record_fork_end: a race between branches")
    assert "RAW race between node 2" in refusal_text(door, e), str(e)
    # the refused race leaves the fork open: the recording is still open, and abort ends it
    e = raises(o.BadRequest, door.record_begin, "record_begin: the refused race left it open")
    assert "already recording" in refusal_text(door, e), str(e)
    door.record_abort()
    door.record_begin()
    door.record_fork_begin()
    door.copy(s1, d1)
    door.record_fork_branch()
    door.record_wait([d1])
    door.copy(d1, d2)
    door.record_fork_end()
    graph = end_replay()
    assert (arrays[2][0], arrays[3][0]) == (11, 11)
    done.append("record_wait: a race refused by the door's text at record_fork_end is resolved")
    door.graph_free(graph)

    # graph_describe of fork + wait, then an IF_ELSE; the launches of a fork of units
    cell_array, (cell,) = cells(1)
    door.record_begin()
    door.record_fork_begin()
    door.copy(s1, d1)
    door.record_fork_branch()
    door.record_wait([d1])
    door.copy(d1, d2)
    door.record_fork_end()
    door.record_cond_begin(o.EG_COND_IF_ELSE, cell)
    door.copy(s2, d1)
    door.record_cond_arm()
    door.copy(s1, d1)
    door.record_cond_end()
    graph = door.record_end()
    desc = door.graph_describe(graph, [s1, s2, d1, d2, cell])
    nodes = desc["nodes"]
    assert desc["flags"] == 0 and desc["n_levels"] == 9 and len(nodes) == 12
    kinds = [
        o.EG_GNODE_FORK, o.EG_GNODE_BRANCH, o.EG_GNODE_COPY, o.EG_GNODE_BRANCH, o.EG_GNODE_WAIT,
        o.EG_GNODE_COPY, o.EG_GNODE_JOIN, o.EG_GNODE_COND, o.EG_GNODE_ARM, o.EG_GNODE_COPY,
        o.EG_GNODE_ARM, o.EG_GNODE_COPY,
    ]  # fmt: skip
    assert [n["kind"] for n in nodes] == kinds
    assert [n["parent"] for n in nodes] == [-1, 0, 1, 0, 3, 3, -1, -1, 7, 8, 7, 10]
    assert [n["level"] for n in nodes] == [0, 1, 2, 1, 3, 4, 5, 6, 7, 8, 7, 8]
    assert nodes[7]["arm"] == o.EG_COND_IF_ELSE and [nodes[8]["arm"], nodes[10]["arm"]] == [0, 1]
    assert nodes[2]["planes"] == (0, 2) and nodes[2]["access"] == (1, 2)
    assert nodes[2]["unit"] is None and nodes[2]["launch"] == -1 and not nodes[7]["has_baseline"]
    edges = {(e["kind"], e["from"], e["to"]): e for e in desc["edges"]}
    assert edges[(o.EG_GEDGE_WAIT, 2, 4)]["plane"] == 2
    for key in ((o.EG_GEDGE_JOIN, 2, 6), (o.EG_GEDGE_JOIN, 5, 6), (o.EG_GEDGE_ORDER, 6, 7)):
        assert key in edges and edges[key]["plane"] is None, key
    assert sum(e["kind"] == o.EG_GEDGE_WAIT for e in desc["edges"]) == 1
    door.graph_free(graph)
    raises(ValueError, lambda: door.graph_describe(graph, []), "graph_describe: a freed graph")
    done.append(
        "graph_describe: node kinds, parents, levels, planes, WAIT and JOIN edges of a fork"
    )

    # each launch node carries the ordinal of its eg_launch call
    unit_a, unit_b = (
        door.unit_load(GRAPH, name, digest(b)) for name, b in (("scale", 0x10), ("square", 0x12))
    )
    xs = np.arange(1.0, 8.0)
    arrays = [xs.copy(), np.array([1.7]), np.zeros(7), np.zeros(7), xs.copy(), np.zeros(7)]
    ps = [door.plane_wrap(a) for a in arrays]
    scale = door.plan_new(unit_a, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 7)
    door.bind(scale, [0, 1, 2], ps[:3], [], [])
    square = door.plan_new(unit_b, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 7)
    door.bind(square, [0, 1], [ps[2], ps[3]], [], [])
    other = door.plan_new(unit_b, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 7)
    door.bind(other, [0, 1], [ps[4], ps[5]], [], [])
    door.record_begin()
    door.record_fork_begin()
    door.launch(scale)
    door.record_fork_branch()
    door.record_wait([ps[2]])
    door.launch(square)
    door.record_fork_end()
    door.launch(scale)
    graph = door.record_end()
    desc = door.graph_describe(graph, ps)
    nodes = desc["nodes"]
    first = next(n for n in nodes if n["kind"] == o.EG_GNODE_LAUNCH)
    assert first["planes"] == (0, 1, 2)
    unlisted = door.graph_describe(graph, ps[:1])["nodes"]
    assert next(n for n in unlisted if n["kind"] == o.EG_GNODE_LAUNCH)["planes"] == (0, -1, -1)
    launches = [(n["unit"], n["launch"]) for n in nodes if n["kind"] == o.EG_GNODE_LAUNCH]
    assert launches == [("scale", 0), ("square", 1), ("scale", 2)], launches
    door.graph_launch(graph)
    assert np.allclose(arrays[3], (1.7 * xs) ** 2)
    door.graph_free(graph)
    done.append("graph_describe: a launch node's ordinal is its place among the launches")

    # a race between launches, refused; then the same fork over disjoint planes
    door.record_begin()
    door.record_fork_begin()
    door.launch(scale)
    door.record_fork_branch()
    door.launch(square)
    e = raises(o.BadRequest, door.record_fork_end, "record_fork_end: a race between launches")
    assert "race between node" in refusal_text(door, e), str(e)
    door.record_abort()  # the refused race left the fork open
    door.record_begin()
    door.record_fork_begin()
    door.launch(scale)
    door.record_fork_branch()
    door.launch(other)
    door.record_fork_end()
    graph = end_replay()
    assert np.allclose(arrays[2], 1.7 * xs) and np.allclose(arrays[5], xs**2)
    door.graph_free(graph)
    door.plan_free(scale)
    door.plan_free(square)
    door.plan_free(other)
    done.append("record_fork_end: launches over overlapping planes are a race, disjoint ones run")

    # negate: IF takes its arm when the cell is zero
    flag, flag_plane = cells(1)
    for negate, flag_value in ((True, 1), (True, 0), (False, 1), (False, 0)):
        flag[0][0] = flag_value
        out, (src, dst) = cells(11, 0)
        door.record_begin()
        door.record_cond_begin(o.EG_COND_IF, flag_plane[0], negate=negate)
        door.copy(src, dst)
        door.record_cond_end()
        graph = end_replay()
        assert (out[1][0] == 11) == ((flag_value != 0) != negate), (negate, flag_value)
        door.graph_free(graph)
    done.append("record_cond_begin: negate runs an IF when the cell is zero, and not otherwise")

    # max_iters: a shift register {1,1,1,1,0} leaves one place per pass; the cell is its first place
    start = np.array([1, 1, 1, 1, 0], dtype=np.int32)
    leading_ones = int(np.argmin(start))
    for cap in (0, 2):
        reg, planes = cells(*start)
        door.record_begin()
        door.record_cond_begin(o.EG_COND_WHILE, planes[0], max_iters=cap)
        for i in range(4):
            door.copy(planes[i + 1], planes[i])
        door.record_cond_end()
        graph = end_replay()
        passes = min(cap or leading_ones, leading_ones)
        want = np.concatenate((start[passes:], np.zeros(passes, dtype=np.int32)))
        assert np.array_equal(np.concatenate(reg), want), (cap, reg)
        door.graph_free(graph)
    done.append("record_cond_begin: a WHILE with max_iters stops after that many passes")

    # cell_index and baseline: the test reads lanes[2], then count[0] against base[1]
    lane_array = np.array([9, 9, 0, 9], dtype=np.int32)
    lane_plane = door.plane_wrap(lane_array)
    out, (src, dst) = cells(11, 0)
    door.record_begin()
    door.record_cond_begin(o.EG_COND_IF, lane_plane, cell_index=2)
    door.copy(src, dst)
    door.record_cond_end()
    graph = door.record_end()
    door.graph_launch(graph)
    assert out[1][0] == 0
    lane_array[2] = 1
    door.graph_launch(graph)
    assert out[1][0] == 11
    door.graph_free(graph)
    count = np.array([3], dtype=np.int32)
    base = np.array([0, 3], dtype=np.int32)
    count_plane, base_plane = door.plane_wrap(count), door.plane_wrap(base)
    for negate in (False, True):
        door.record_begin()
        door.record_cond_begin(
            o.EG_COND_IF, count_plane, negate=negate, baseline=base_plane, baseline_index=1
        )
        door.copy(src, dst)
        door.record_cond_end()
        graph = door.record_end()
        for value in (3, 4):
            count[0], out[1][0] = value, 0
            door.graph_launch(graph)
            assert (out[1][0] == 11) == ((value != base[1]) != negate), (negate, value)
        node = door.graph_describe(graph, [count_plane, base_plane])["nodes"][0]
        assert node["has_baseline"] and node["cond_flags"] == (o.EG_COND_NEGATE if negate else 0)
        door.graph_free(graph)
    done.append("record_cond_begin: cell_index and a baseline element decide the test")


def check_reverse_structure(door, done):
    """Rows of graph_reverse through a fork, an IF and a WHILE; closed forms in numpy."""
    n = 7
    names = (("scale", 0x10), ("scale_vjp", 0x11), ("mul2", 0x14), ("mul2_vjp", 0x15))
    units = {name: door.unit_load(GRAPH, name, digest(b)) for name, b in names}
    units["tick"] = door.unit_load(GRAPH, "tick", digest(0x1E))
    x = 0.5 + 0.3 * np.arange(n)

    def planes(*arrays):
        """A plane over each array."""
        return [door.plane_wrap(a) for a in arrays]

    def plan(unit, bound):
        """A serial plan of the unit with the planes bound at slots 0, 1, ..."""
        handle = door.plan_new(units[unit], o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(handle, list(range(len(bound))), bound, [], [])
        return handle

    # a fork: y1 = w1 x and y2 = w2 x in two branches, z = y1 y2: dJ/dx = 2 w1 w2 x
    w1, w2 = np.array([1.7]), np.array([-0.6])
    arrays = [x.copy(), np.zeros(n), np.zeros(n), np.zeros(n), w1, w2, np.ones(n), np.zeros(n)]
    px, py1, py2, pz, pw1, pw2, one, gx = planes(*arrays)
    f1, f2, m = (
        plan("scale", [px, pw1, py1]),
        plan("scale", [px, pw2, py2]),
        plan("mul2", [py1, py2, pz]),
    )
    door.record_begin()
    door.record_fork_begin()
    door.launch(f1)
    door.record_fork_branch()
    door.launch(f2)
    door.record_fork_end()
    door.launch(m)
    graph = door.record_end(
        [(0, units["scale_vjp"]), (1, units["scale_vjp"]), (2, units["mul2_vjp"])]
    )
    door.graph_launch(graph)
    door.graph_reverse(graph, [(pz, one)], [(px, gx)])
    assert np.allclose(arrays[7], 2 * w1[0] * w2[0] * x, rtol=1e-12)
    done.append(
        "graph_reverse: a fork of two branches then a join sums both paths, as the closed form"
    )

    # an IF on lane 2 of a flag array: y = w1 x, and when the flag is set y = w2 x2 (taken or not)
    x2 = -1.5 + 0.3 * np.arange(n)
    flags = np.array([7, 7, 0, 7], dtype=np.int32)
    arrays = [x.copy(), x2, np.zeros(n), w1, w2, np.ones(n), np.zeros(n), np.zeros(n)]
    px, px2, py, pw1, pw2, one, gx, gx2 = planes(*arrays)
    flag = door.plane_wrap(flags)
    f1, f2 = plan("scale", [px, pw1, py]), plan("scale", [px2, pw2, py])
    door.record_begin()
    door.launch(f1)
    door.record_cond_begin(o.EG_COND_IF, flag, cell_index=2)
    door.launch(f2)
    door.record_cond_end()
    graph = door.record_end([(0, units["scale_vjp"]), (1, units["scale_vjp"])])
    for taken in (1, 0):
        flags[2] = taken
        door.graph_launch(graph)
        door.graph_reverse(graph, [(py, one)], [(px, gx), (px2, gx2)])
        # a taken arm overwrites y, so the first launch reaches nothing; a skipped one adds nothing
        assert np.allclose(arrays[6], 0.0 if taken else w1[0]), taken
        assert np.allclose(arrays[7], w2[0] if taken else 0.0), taken
    done.append("graph_reverse: an IF taken follows its arm, an IF not taken contributes nothing")

    # a WHILE whose counter ticks down per pass (t = a x, x = 1 t), capped at 3 passes
    a, unit_gain = np.array([0.9]), np.ones(1)
    for start, cap in ((2, 4), (6, 3), (0, 4)):
        counter = np.full(n, start, dtype=np.int32)
        arrays = [x.copy(), np.zeros(n), a, unit_gain, np.ones(n), np.zeros(n), np.zeros(1)]
        px, pt, pa, pone, seed, gx, ga = planes(*arrays)
        pc = door.plane_wrap(counter)
        f1, f2 = plan("scale", [px, pa, pt]), plan("scale", [pt, pone, px])
        tick = plan("tick", [pc])
        door.record_begin()
        door.record_cond_begin(o.EG_COND_WHILE, pc, max_iters=cap)
        door.launch(f1)
        door.launch(f2)
        door.launch(tick)
        door.record_cond_end()
        graph = door.record_end([(0, units["scale_vjp"]), (1, units["scale_vjp"])])
        door.graph_launch(graph)
        passes = min(start, cap)
        assert np.allclose(arrays[0], a[0] ** passes * x), (start, cap)
        door.graph_reverse(graph, [(px, seed)], [(px, gx), (pa, ga)])
        # J = sum of x after `passes` multiplications: dJ/dx0 = a^p, dJ/da = p a^(p-1) sum x0
        assert np.allclose(arrays[5], a[0] ** passes), (start, cap)
        assert np.allclose(arrays[6], passes * a[0] ** (passes - 1) * x.sum() if passes else 0.0)
    done.append(
        "graph_reverse: a WHILE stopped by its counter or by max_iters gives the unrolled gradient"
    )


def check_step_graphs_and_loops(door, done, raises):
    """Rows of runner_new(steps=) and of the limit, counter and brk members of record_cond_begin."""
    n = 4
    count1 = door.unit_load(FIXTURE, "count1", digest(0xD1))

    def counting_plan(limit):
        """A finishing plan that steps k by one per launch up to `limit`: its k array and plane."""
        arrays = [np.zeros(n), np.full(n, float(limit)), np.zeros(n, dtype=np.bool_)]
        planes = [door.plane_wrap(a) for a in arrays]
        handle = door.plan_new(count1, o.EG_STRUCTURE_HOST_SERIAL, 0, n)
        door.bind(handle, [0, 1, 2], planes)
        return arrays[0], planes[0], handle

    # two step graphs of an unrecorded runner of two plans (the finishing plan counts, the second
    # one y = 2 x): the first graph does x += 1 ahead of plan 0, the second x *= 10 ahead of plan 1
    k_a, plane_a, plan_a = counting_plan(5)
    scale, scale_vjp = (
        door.unit_load(GRAPH, name, digest(b)) for name, b in (("scale", 0x10), ("scale_vjp", 0x11))
    )
    x_steps, y_steps, two = np.arange(1.0, n + 1), np.zeros(n), np.array([2.0])
    plane_x, plane_y, plane_two = (door.plane_wrap(v) for v in (x_steps, y_steps, two))
    plan_b = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
    door.bind(plan_b, [0, 1, 2], [plane_x, plane_two, plane_y], [], [])
    x_oracle = x_steps.copy()
    seen = []

    def step_graph(tag, change):
        def work(views):
            seen.append((tag, float(np.from_dlpack(views[0])[0])))
            x = np.from_dlpack(views[1])
            x[:] = change(x)

        node = door.host_node(tag, work)
        door.record_begin()
        door.host_call(
            node, [plane_a, plane_x], [o.EG_ACCESS_READ, o.EG_ACCESS_READ | o.EG_ACCESS_WRITE]
        )
        return door.record_end()

    ahead_of_a = step_graph("ahead_of_a", lambda x: x + 1.0)
    ahead_of_b = step_graph("ahead_of_b", lambda x: x * 10.0)
    report = door.run(
        door.runner_new([plan_a, plan_b], 100, steps=[(ahead_of_a, 0), (ahead_of_b, 1)])
    )
    # launch L (1 based): plan a has stepped L - 1 times before the first graph and L before the
    # second
    expected = []
    for launch in range(1, 6):
        expected += [("ahead_of_a", launch - 1.0), ("ahead_of_b", float(launch))]
        x_oracle = (x_oracle + 1.0) * 10.0
    assert report["launches"] == 5 and seen == expected, (report, seen)
    assert np.array_equal(k_a, np.full(n, 5.0)), k_a
    assert np.array_equal(x_steps, x_oracle) and np.array_equal(y_steps, 2.0 * x_oracle), x_steps
    done.append("runner_new(steps=): two graphs run ahead of plan 0 and plan 1 of each launch")

    # a recorded runner takes a graph only ahead of its first plan: before_plan 1 is refused by name
    size = 37
    lin = door.unit_load(REVERSE, "lin", digest(0xE0))
    tape = door.unit_load(REVERSE, "lin", digest(0xE1))
    vjp = door.unit_load(REVERSE, "lin_vjp", digest(0xE4))
    state = [
        1.0 + np.arange(size) / 8.0,
        np.zeros(size),
        np.zeros(size, dtype=np.bool_),
        1.0 + (np.arange(size) * 7) % 10,
    ]
    lin_plan = door.plan_new(lin, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, size)
    door.bind(
        lin_plan,
        [0, 1, 2, 4],
        [door.plane_wrap(a) for a in state],
        [5, 6],
        [uniform(0.5), uniform(0.25)],
    )
    gx, gw, gy = np.arange(1.0, n + 1), np.array([0.5]), np.zeros(n)
    scale_plan = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
    door.bind(scale_plan, [0, 1, 2], [door.plane_wrap(a) for a in (gx, gw, gy)], [], [])
    door.record_begin()
    door.launch(scale_plan)
    differentiated = door.record_end([(0, scale_vjp)])
    recorded = dict(
        flags=o.EG_RUN_RECORD,
        derived_roles=[o.EG_DERIVED_TAPE, o.EG_DERIVED_STEP_VJP],
        derived_units=[tape, vjp],
        window=1,
    )
    e = raises(
        o.BadRequest,
        lambda: door.runner_new([lin_plan], 10, steps=[(differentiated, 1)], **recorded),
        "runner_new: a recorded runner with a step graph ahead of no plan",
    )
    assert "before_plan" in refusal_text(door, e), str(e)
    done.append("runner_new(steps=): a recorded runner refuses before_plan 1 by the door's text")

    # WHILE loops over x <- a x (two scale plans), against numpy
    x0, a, unit_gain = 0.5 + 0.3 * np.arange(n), np.array([0.9]), np.ones(1)
    x, t = x0.copy(), np.zeros(n)
    px, pt, pa, pone = (door.plane_wrap(v) for v in (x, t, a, unit_gain))

    def body_plans():
        first = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(first, [0, 1, 2], [px, pa, pt], [], [])
        second = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(second, [0, 1, 2], [pt, pone, px], [], [])
        return first, second

    first, second = body_plans()
    counter, limit = np.full(1, 99, dtype=np.uint32), np.array([9, 3], dtype=np.uint32)
    door.record_begin()
    door.record_cond_begin(
        o.EG_COND_WHILE,
        None,
        counter=door.plane_wrap(counter),
        limit=door.plane_wrap(limit),
        limit_index=1,
    )
    door.launch(first)
    door.launch(second)
    door.record_cond_end()
    door.graph_launch(door.record_end())
    assert np.allclose(x, a[0] ** 3 * x0) and counter[0] == 3, (x, counter)
    done.append(
        "record_cond_begin: a WHILE stops at limit[limit_index] and the counter reads the count"
    )

    # a brk written in the body stops the loop after that iteration (the cap of 50 would show
    # otherwise)
    tick = door.unit_load(GRAPH, "tick", digest(0x1E))
    # (the loop reads the flag as uint32; the tick unit counts the same word down as an int32, so
    # one iteration leaves it non-zero)
    brk = np.zeros(1, dtype=np.uint32)
    stop = door.plan_new(tick, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 1)
    pbrk = door.plane_wrap(brk)
    door.bind(stop, [0], [door.plane_wrap(brk.view(np.int32))], [], [])
    x[:] = x0
    door.record_begin()
    door.record_cond_begin(o.EG_COND_WHILE, None, max_iters=50, brk=pbrk)
    door.launch(first)
    door.launch(second)
    door.launch(stop)
    door.record_cond_end()
    door.graph_launch(door.record_end())
    assert np.allclose(x, a[0] * x0), x
    done.append("record_cond_begin: a brk written in the body stops the loop")

    # the same loop under a DIFF recording is refused by name, at the end of the recording
    door.record_begin()
    door.record_cond_begin(o.EG_COND_WHILE, None, max_iters=50, brk=pbrk)
    door.launch(first)
    door.launch(stop)
    door.record_cond_end()
    e = raises(
        o.NotWritten,
        lambda: door.record_end([(0, scale_vjp)]),
        "record_end(diff): a WHILE with a brk",
    )
    assert "holds a break" in refusal_text(door, e), str(e)
    door.record_abort()  # the refused end leaves the recording open
    done.append("record_end(diff): a WHILE holding a brk is refused by name")


def check_gather_mask(door, done, raises):
    """A gathered matrix with a keep mask binds with its column count, not with another."""
    samples = 40
    unit = door.unit_load(FIXTURE, "gathb", digest(0x61))
    plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_TEAM, 0, samples)
    y = door.plane_wrap(np.zeros(samples))
    # the mask keeps the one static axis of four columns: the matrix is (2, 2, 4)
    good = door.plane_wrap(np.zeros((2, 2, 4)))
    door.bind(plan, [0, 1], [y, good], [], [])
    short = door.plane_wrap(np.zeros((2, 2, 3)))
    e = raises(o.BadRequest, lambda: door.bind(plan, [1], [short], [], []), "bind: a short matrix")
    assert "a plane of shape (2, 2, 4) is expected" in refusal_text(door, e), str(e)
    done.append(
        "bind: a gathered matrix of the mask's column count binds, a shorter one is refused"
    )


def check_gather_table(door, done, raises):
    """A broadcast table (a gathered LOOKUP) binds with one value per kept lane, sized by the door.

    The unit's launch has two static lane axes (3 and 6) and its table varies along both: 18 values.
    """
    samples = 18
    unit = door.unit_load(FIXTURE, "gathl", digest(0xEB))
    plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_TEAM, 0, samples)
    y = door.plane_wrap(np.zeros(samples))
    door.bind(plan, [0, 1], [y, door.plane_wrap(np.zeros(samples))], [], [])
    short = door.plane_wrap(np.zeros(6))
    for what, call in (
        ("bind: a table short of the lanes it varies along", lambda: door.bind(plan, [1], [short])),
        ("bind: a short table into a bound plan", lambda: door.bind(plan, [0, 1], [y, short])),
    ):
        e = raises(o.BadRequest, call, what)
        assert "a table of 18 elements is expected (18 for the lane axes it varies along)" in (
            refusal_text(door, e)
        ), str(e)
    done.append(
        "bind: a gathered table of the lane count binds, a shorter one is refused by the door"
    )
    e = raises(
        o.BadRequest,
        lambda: door.unit_load(FIXTURE, "gathls", digest(0xEC)),
        "unit_load: a gathered table that is also shared",
    )
    assert "is GATHER (it varies along lane axes) and SHARED" in refusal_text(door, e), str(e)
    done.append("unit_load: a table both gathered and shared contradicts itself and is refused")


def check_reverse_options(door, done, raises):
    """Rows of the report graph_reverse returns and of the options it takes."""
    n = 7
    names = (("scale", 0x10), ("scale_vjp", 0x11), ("scale_scatter_vjp", 0x1F))
    units = {name: door.unit_load(GRAPH, name, digest(b)) for name, b in names}
    x = 0.5 + 0.3 * np.arange(n)
    w1, w2 = 1.7, -0.6

    def build(second_vjp):
        """Two launches computing y = w2 (w1 x), ended with the given vjp for the second."""
        arrays = [x.copy(), np.zeros(n), np.zeros(n), np.array([w1]), np.array([w2]), np.ones(n)]
        arrays += [np.full(n, -999.0), np.zeros(1), np.zeros(1)]
        px, pu, py, pw1, pw2, one, gx, gw1, gw2 = (door.plane_wrap(a) for a in arrays)
        plans = []
        for unit, bound in (("scale", [px, pw1, pu]), ("scale", [pu, pw2, py])):
            plan = door.plan_new(units[unit], o.EG_STRUCTURE_HOST_SERIAL, 0, n)
            door.bind(plan, [0, 1, 2], bound, [], [])
            plans.append(plan)
        door.record_begin()
        for plan in plans:
            door.launch(plan)
        graph = door.record_end([(0, units["scale_vjp"]), (1, units[second_vjp])])
        door.graph_launch(graph)
        grads = [(px, gx), (pw1, gw1), (pw2, gw2)]
        return arrays, graph, [(py, one)], grads

    # a plain chain: every gradient is bitwise, one pass, the closed forms
    arrays, graph, seeds, grads = build("scale_vjp")
    report = door.graph_reverse(graph, seeds, grads)
    assert report == ((True, True, True), 1), report
    assert np.allclose(arrays[6], w1 * w2) and np.allclose(arrays[7], w2 * x.sum())
    assert np.allclose(arrays[8], w1 * x.sum())
    done.append("graph_reverse: a plain chain reports every gradient bitwise and one pass")

    # a scatter-add vjp taints the gradients its adjoint reaches, and strict refuses it
    arrays, graph, seeds, grads = build("scale_scatter_vjp")
    report = door.graph_reverse(graph, seeds, grads)
    assert report == ((False, False, True), 1), report
    assert np.allclose(arrays[6], w1 * w2) and np.allclose(arrays[8], w1 * x.sum())
    arrays[6][:] = -999.0
    e = raises(
        o.BadRequest,
        lambda: door.graph_reverse(graph, seeds, grads, strict=True),
        "graph_reverse: strict over a tainted gradient",
    )
    assert "inputs[0]" in refusal_text(door, e) and "launch 1" in str(e), str(e)
    assert (arrays[6] == -999.0).all()
    done.append("graph_reverse: a scatter-add vjp is reported not bitwise and strict refuses it")
    assert door.graph_reverse(graph, seeds, grads[2:])[0] == (True,)

    # an integer input has no gradient
    flags = np.array([7, 7, 7, 7], dtype=np.int32)
    arrays, _, seeds, grads = build("scale_vjp")
    flag, gflag = door.plane_wrap(flags), door.plane_wrap(np.zeros(1))
    plan = door.plan_new(units["scale"], o.EG_STRUCTURE_HOST_SERIAL, 0, n)
    door.bind(plan, [0, 1, 2], [grads[0][0], grads[1][0], seeds[0][0]], [], [])
    door.record_begin()
    door.record_cond_begin(o.EG_COND_IF, flag)
    door.launch(plan)
    door.record_cond_end()
    graph = door.record_end([(0, units["scale_vjp"])])
    door.graph_launch(graph)
    e = raises(
        o.BadRequest,
        lambda: door.graph_reverse(graph, seeds, [(flag, gflag)]),
        "graph_reverse: an integer input",
    )
    assert "is not a float plane" in refusal_text(door, e), str(e)
    done.append("graph_reverse: an integer plane asked for a gradient is refused by name")


def check_host_nodes(door, backend, done, raises):
    """Rows of host nodes: a Python callable run by the door, recorded, replayed and reversed.

    numpy is the oracle: the callables compute with it, and the closed forms are written by hand.
    """
    n = 8
    x0 = 0.5 + 0.3 * np.arange(n)

    # a node that doubles a plane in place and checks what its views are
    seen = {"device": None, "stale": [], "inside": []}

    def double(views):
        a = np.from_dlpack(views[0])
        assert a.flags.writeable and a.shape == (n,) and a.dtype == np.float64
        a *= 2.0
        seen["device"] = views[0].__dlpack_device__()
        seen["stale"].append(views[0])
        try:
            door.sync()
        except o.BadRequest as inner:
            seen["inside"].append(str(inner))

    node = door.host_node("double", double)
    assert node.kind == "host_node" and node.live
    now = x0.copy()
    plane = door.plane_wrap(now)
    door.host_call(node, [plane], [o.EG_ACCESS_READ | o.EG_ACCESS_WRITE])
    assert np.array_equal(now, 2.0 * x0)
    assert seen["device"] == (1, 0), seen["device"]  # kDLCPU, device 0
    assert seen["inside"] and "double" in seen["inside"][0], seen["inside"]
    done.append(
        "host_node, host_call: a callable doubles a plane in place through numpy.from_dlpack,"
        " and a door call from inside is refused naming the node"
    )

    # a view kept past the call is refused by name, for both of its protocol calls
    for use in (
        lambda v: v.__dlpack__(max_version=(1, 0)),
        lambda v: v.__dlpack_device__(),
        np.from_dlpack,
    ):
        e = raises(ValueError, lambda use=use: use(seen["stale"][0]), "a view kept past the call")
        assert "host node double" in str(e) and "only during the call" in str(e), str(e)
    done.append("host_call: a view kept past the call is refused by name")

    # recorded, not run; replayed twice
    recorded = x0.copy()
    plane_r = door.plane_wrap(recorded)
    door.record_begin()
    door.host_call(node, [plane_r], [o.EG_ACCESS_READ | o.EG_ACCESS_WRITE])
    assert np.array_equal(recorded, x0), "a recording runs nothing"
    graph = door.record_end()
    for launches in (1, 2):
        door.graph_launch(graph)
        assert np.array_equal(recorded, x0 * 2.0**launches), launches
    nodes = door.graph_describe(graph, [plane_r])["nodes"]
    assert [(m["kind"], m["unit"], m["planes"]) for m in nodes] == [
        (o.EG_GNODE_HOST, "double", (0,))
    ]
    done.append("host_call recorded, replayed twice, and described as a host node with its name")

    # a read-only view: a plane declared read refuses a write in the callable
    refused = []

    def copy_to_second(views):
        source, target = (np.from_dlpack(v) for v in views)
        assert not source.flags.writeable and target.flags.writeable
        try:
            source[0] = -1.0
        except ValueError:
            refused.append(True)
        target[:] = source

    src_a, dst_a = x0.copy(), np.zeros(n)
    copy_node = door.host_node("copy_to_second", copy_to_second)
    door.host_call(
        copy_node,
        [door.plane_wrap(src_a), door.plane_wrap(dst_a)],
        [o.EG_ACCESS_READ, o.EG_ACCESS_WRITE],
    )
    assert refused == [True] and np.array_equal(dst_a, x0) and np.array_equal(src_a, x0)
    done.append("host_call: a plane declared read is a read-only view; one declared write is not")

    # an exception of the callable arrives as itself, from the run and from the replay
    class Boom(Exception):
        pass

    def fails(views):
        raise Boom("the callable failed: 42")

    def replayed(node, plane):
        """Record a call of `node` over `plane` and launch the graph, as a replay does."""
        door.record_begin()
        door.host_call(node, [plane], [o.EG_ACCESS_READ])
        door.graph_launch(door.record_end())

    boom = door.host_node("fails", fails)
    cell = door.plane_wrap(np.zeros(1))
    for call in (
        lambda: door.host_call(boom, [cell], [o.EG_ACCESS_READ]),
        lambda: replayed(boom, cell),
    ):
        try:
            call()
        except Boom as e:
            assert str(e) == "the callable failed: 42"
        else:
            raise AssertionError("the callable's exception was lost")
    assert door.sync() is None, "the context is usable after a callable failed"
    done.append("host_call, graph_launch: an exception of the callable is raised as itself")

    # reversed: y = w x (a unit), z = 3 y^2 (a host node with its vjp), J = sum z
    scale, scale_vjp = (
        door.unit_load(GRAPH, name, digest(b)) for name, b in (("scale", 0x10), ("scale_vjp", 0x11))
    )

    def triple_square(views):
        y, z = (np.from_dlpack(v) for v in views)
        z[:] = 3.0 * y * y

    def triple_square_vjp(primals, adjoints_in, adjoints_out):
        assert primals[1] is None and adjoints_in[0] is None and adjoints_out[1] is None
        y = np.from_dlpack(primals[0])
        adjoint_z = np.from_dlpack(adjoints_in[1])
        adjoint_y = np.from_dlpack(adjoints_out[0])
        assert not np.from_dlpack(primals[0]).flags.writeable and not adjoint_z.flags.writeable
        assert np.all(adjoint_y == 0.0)
        adjoint_y[:] = 6.0 * y * adjoint_z

    def rig(host_node):
        arrays = {k: np.zeros(n) for k in ("y", "z", "gx")}
        arrays.update(x=x0.copy(), w=np.array([1.7]), ones=np.ones(n), gw=np.zeros(1))
        planes = {k: door.plane_wrap(a) for k, a in arrays.items()}
        plan = door.plan_new(scale, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
        door.bind(plan, [0, 1, 2], [planes[k] for k in ("x", "w", "y")], [], [])
        door.record_begin()
        door.launch(plan)
        door.host_call(
            host_node, [planes[k] for k in ("y", "z")], [o.EG_ACCESS_READ, o.EG_ACCESS_WRITE]
        )
        return arrays, planes, door.record_end([(0, scale_vjp)])

    reversing = door.host_node("triple_square", triple_square, triple_square_vjp)
    arrays, planes, graph_d = rig(reversing)
    door.graph_launch(graph_d)
    w = 1.7
    assert np.allclose(arrays["z"], 3.0 * (w * x0) ** 2, rtol=1e-12)
    door.graph_reverse(
        graph_d,
        [(planes["z"], planes["ones"])],
        [(planes["x"], planes["gx"]), (planes["w"], planes["gw"])],
    )
    assert np.allclose(arrays["gx"], 6.0 * w * w * x0, rtol=1e-12)
    assert np.allclose(arrays["gw"], 6.0 * w * np.sum(x0 * x0), rtol=1e-12)
    done.append("graph_reverse: a host node's vjp is a leaf: dJ/dx and dJ/dw equal the closed form")

    # no vjp and a gradient flowing through it: refused by name at the reverse
    plain = door.host_node("triple_square_no_vjp", triple_square)
    arrays, planes, graph_n = rig(plain)
    door.graph_launch(graph_n)
    e = raises(
        o.BadRequest,
        lambda: door.graph_reverse(
            graph_n, [(planes["z"], planes["ones"])], [(planes["x"], planes["gx"])]
        ),
        "graph_reverse: a host node with no vjp",
    )
    assert "triple_square_no_vjp" in refusal_text(door, e), str(e)
    done.append("graph_reverse: a host node with no vjp under a gradient is refused naming it")

    # the callables are released once: not while a node handle or a graph holds the node
    collected = []

    def make_node(name):
        def work(views):
            pass

        ref = weakref.ref(work, lambda _: collected.append(name))
        return door.host_node(name, work), ref

    kept, ref_kept = make_node("kept")
    cell_array = np.zeros(1)
    cell_plane = door.plane_wrap(cell_array)
    door.record_begin()
    door.host_call(kept, [cell_plane], [o.EG_ACCESS_READ])
    graph_k = door.record_end()
    door.host_node_free(kept)
    assert collected == [] and ref_kept() is not None, "the graph holds the node"
    door.graph_launch(graph_k)
    door.graph_free(graph_k)
    assert collected == ["kept"] and ref_kept() is None
    free_only, ref_free = make_node("free_only")
    door.host_node_free(free_only)
    assert collected == ["kept", "free_only"] and ref_free() is None
    second = o.Door(backend)

    def left_work(views):
        pass

    ref_left = weakref.ref(left_work, lambda _: collected.append("left"))
    left = second.host_node("left", left_work)
    del left_work
    assert left.live and collected == ["kept", "free_only"]
    second.ctx_free()
    assert collected == ["kept", "free_only", "left"] and ref_left() is None, collected
    done.append("host_node_free, graph_free, ctx_free: the callables are released once, at the end")

    # device_props answers or refuses as the header states, on this context
    check_device_props(door, done, raises)

    # a step graph of a host node runs once per step of the runner, ahead of the step's plans
    counted = []
    dummy = door.plane_wrap(np.zeros(1))
    step = door.host_node("count_step", lambda views: counted.append(1))
    door.record_begin()
    door.host_call(step, [dummy], [o.EG_ACCESS_READ])
    step_graph = door.record_end()
    index = np.arange(200)
    arrays = [np.zeros(200), (2 + index % 9).astype(float), np.zeros(200, dtype=np.bool_)]
    count1 = door.unit_load(FIXTURE, "count1", digest(0xD1))
    plan = door.plan_new(count1, o.EG_STRUCTURE_HOST_SERIAL, 0, 200)
    door.bind(plan, [0, 1, 2], [door.plane_wrap(a) for a in arrays])
    report = door.run(door.runner_new([plan], 1000, steps=[(step_graph, 0)]))
    assert report["launches"] > 1 and len(counted) == report["launches"], (report, len(counted))
    done.append("runner_new(steps=): a host node in it runs once per launch of the runner")


def check_device_props(door, done, raises):
    """Rows of device_props here: the door's answer, or its refusal as the header states."""
    members = (
        "cc_major cc_minor sm_count warp_size regs_per_sm regs_per_block shared_mem_per_block "
        "shared_mem_per_sm name clock_rate_khz memory_clock_rate_khz memory_bus_width_bits "
        "peak_bytes_per_s peak_flops_sp peak_flops_dp fp64_ratio ridge_flops_per_byte_sp "
        "ridge_flops_per_byte_dp"
    ).split()
    count = door.device_count()
    try:
        props = door.device_props(0)
    except o.NotWritten as e:
        # a door built without the device group refuses every device call by name
        assert count == 0 and "the device group is not built" in refusal_text(door, e)
        done.append("device_props: a door without the device group refuses by name")
        return
    except o.BadRequest as e:
        # a door with the group and no driver, or no device at that ordinal
        assert count == 0 and refusal_text(door, e)
        done.append("device_props: an ordinal the driver does not see is refused by the door")
        return
    assert count > 0 and list(props) == members, (count, list(props))
    assert isinstance(props["name"], str) and props["name"] and props["cc_major"] > 0
    assert all(isinstance(props[k], float) for k in members[-6:])
    raises(
        o.BadRequest, lambda: door.device_props(count), "device_props: an ordinal past the count"
    )
    done.append("device_props: every member under the struct's names; a missing ordinal refused")


def check_record_abort(door, done, raises):
    """Rows of record_abort: nothing open, an open fork left by refusals, and a recording after."""
    e = raises(o.BadRequest, door.record_abort, "record_abort: nothing open")
    assert "eg_record_abort: no open recording" in refusal_text(door, e), str(e)
    values = [np.array([v], dtype=np.int32) for v in (7, 0, 0)]
    src, mid, dst = (door.plane_wrap(a) for a in values)
    door.record_begin()
    door.record_fork_begin()
    raises(o.BadRequest, lambda: door.record_wait([mid]), "record_wait: no earlier sibling writes")
    door.copy(src, mid)
    door.record_fork_branch()  # the second branch is empty
    door.record_abort()
    raises(o.BadRequest, door.record_abort, "record_abort: the recording is gone")
    # a bind (refused while recording) and a new recording work after the abort
    unit = door.unit_load(GRAPH, "scale", digest(0x10))
    plan = door.plan_new(unit, o.EG_STRUCTURE_HOST_SERIAL, 0, 1)
    door.bind(plan, [], [], [], [])
    assert values[1][0] == 0, "an aborted recording runs nothing"
    door.record_begin()
    door.copy(src, dst)
    door.graph_launch(door.record_end())
    assert values[2][0] == 7
    done.append(
        "record_abort: refused with nothing open; from an open fork it frees bind and recording"
    )


def device_door(backend):
    """A context on device 0, or None when the door was built without a device group."""
    try:
        return o.Door(backend, device=0)
    except o.NotWritten:
        return None


def check_device(door, done, raises):
    """Rows on a device context: a banded slot's refusal, routed gemm units, direct gemm calls.

    Data moves through the door's copy between host planes and device planes; numpy is the oracle,
    with the bound of `product`.
    """
    rng = np.random.default_rng(11)

    unit = door.unit_load(DEVICE_FIXTURE, "banded", digest(0xC4))
    plan = door.plan_new(unit, o.EG_STRUCTURE_DEVICE, o.EG_REDUCE_COMPENSATED, N)
    packed = door.plane_new([N], 1, 64, kind=o.EG_MEM_DEVICE)
    door.bind(plan, [0], [packed])  # the packed cell is the banded real slot's own element type
    doubles = door.plane_new([N], 2, 64, kind=o.EG_MEM_DEVICE)
    e = raises(o.BadRequest, lambda: door.bind(plan, [0], [doubles]), "bind: a double on a band")
    assert "a banded cell is not a double; pack it with eg_banded_pack" in refusal_text(door, e)
    for plane in (packed, doubles):
        door.plane_free(plane)
    door.plan_free(plan)
    door.unit_free(unit)
    done.append("banded slot on a device plan: a packed cell binds, a float64 plane is refused")

    m, k, n = 3, 5, 4
    for bits, dtype in ((64, np.float64), (32, np.float32)):
        for flags in range(16):
            a_shape = (k, m) if flags & o.EG_GEMM_TRANS_A else (m, k)
            b_shape = (n, k) if flags & o.EG_GEMM_TRANS_B else (k, n)
            c_shape = (n, m) if flags & o.EG_GEMM_TRANS_C else (m, n)
            a, b = (rng.standard_normal(s).astype(dtype) for s in (a_shape, b_shape))
            c = rng.standard_normal(c_shape).astype(dtype)
            want, bound = product(a, b, flags, c)
            gemm = door.gemm_new(a_shape, b_shape, c_shape, bits, flags)
            planes = [to_device(door, x) for x in (a, b, c)]
            door.gemm_run(gemm, *planes)
            door.sync()
            assert within(from_device(door, planes[2], c), want, bound), (bits, flags)
            for plane in planes:
                door.plane_free(plane)
            door.gemm_free(gemm)
    done.append("device gemm_new, gemm_run: 32 products (f32, f64, every flag) equal numpy")

    shapes = {
        "sq": (0, (4, 4), (4, 4), (4, 4)),
        "rect": (0, (3, 5), (5, 4), (3, 4)),
        "ta": (o.EG_GEMM_TRANS_A, (5, 3), (5, 4), (3, 4)),
        "tb": (o.EG_GEMM_TRANS_B, (3, 5), (4, 5), (3, 4)),
        "tc": (o.EG_GEMM_TRANS_C, (3, 5), (5, 4), (4, 3)),
        "all": (7, (5, 3), (4, 5), (4, 3)),
        "lane": (0, (3, 5), (5, 4), (3, 4)),
        "wide": (0, (3, 5), (5, 4), (3, 4)),
    }
    for bits, base, dtype in ((64, 0x30, np.float64), (32, 0x40, np.float32)):
        for index, (name, (flags, a_s, b_s, c_s)) in enumerate(shapes.items()):
            slack = 2 if name == "wide" else 0
            full = [(r, c + slack) for r, c in (a_s, b_s, c_s)]
            a, b, c = (rng.standard_normal(s).astype(dtype) for s in full)
            c[:] = 5.0
            unit = door.unit_load(DEVICE_GEMM, f"f{bits}_{name}", digest(base + index))
            plan = door.plan_new(unit, o.EG_STRUCTURE_DEVICE, o.EG_REDUCE_COMPENSATED, 12)
            planes = [to_device(door, x) for x in (a, b, c)]
            door.bind(plan, [0, 1, 2], planes)
            assert door.plan_entry_figures(plan)["flags"] & o.EG_ENTRY_GEMM != 0, (bits, name)
            door.launch(plan)
            door.sync()
            cut = lambda x, s: x[: s[0], : s[1]]  # noqa: E731
            got = cut(from_device(door, planes[2], c), c_s)
            assert within(got, *product(cut(a, a_s), cut(b, b_s), flags, None)), (bits, name)
            for plane in planes:
                door.plane_free(plane)
            door.plan_free(plan)
            door.unit_free(unit)
    done.append(
        "gemm units on a device plan: EG_ENTRY_GEMM set, eight shapes, f32 and f64, equal numpy"
    )


def run(done, raises, not_run, backends):
    """Run every row; `done` collects the passed rows, `not_run` the rows the door cannot run.

    `raises` is the harness's check. `backends` holds the opened backend handles the doors are
    made from: "cpu" and "second" (a second open of the cpu library), and "cuda" when given.
    """
    cpu = backends["cpu"]
    # ---- the context: opening, refusals of the library, information ----
    e = raises(ValueError, lambda: o.Door(0), "Door: a handle that is no opened backend")
    assert "not an opened backend" in str(e), str(e)
    if "cuda" in backends:
        # the cuda library carries the door only when built with a CUDA toolkit (a Mac has none)
        try:
            built_with = o.Door(backends["cuda"])
        except o.NotWritten as refused:
            assert "states no runtime" in str(refused), str(refused)
            done.append("Door: the cuda library, which states no runtime, refuses by name")
            not_run.append("the device rows: the cuda library is built without the device runtime")
        else:
            assert sys.platform != "darwin", "a Mac has no CUDA toolkit to build the door with"
            assert built_with.ctx_info()["device"] == -1
            done.append("Door: the cuda library built with the device runtime opens a host context")
            on_device = device_door(backends["cuda"])
            if on_device is None:
                not_run.append("the device rows on the cuda library: no device context opens")
            elif not os.path.exists(DEVICE_FIXTURE):
                not_run.append(
                    "the device rows on the cuda library: no device fixture "
                    "(binding/build.sh builds it when the cuda library found a CUDA toolkit)"
                )
            else:
                check_device(on_device, done, raises)
    else:
        not_run.append("Door: the cuda library refuses by name: no CUDA_LIB")
    door = o.Door(cpu)
    device = device_door(cpu)
    info = door.ctx_info()
    assert info["abi_major"] == 1 and info["device"] == -1 and info["stream"] == -1, info
    assert info["capabilities"] & o.EG_CAP_HOST, info
    done.append("Door, ctx_info: a host context")
    # the class key of the dispatch weights: row -1 is the built-in row (no card states a class
    # row for a host); the key is what the machine reads, a CPUID vendor and family on x86 and
    # vendor 0 ("other") where there is no CPUID
    assert info["weight_row"] == -1, info
    if platform.machine() in ("x86_64", "AMD64"):
        assert info["weight_vendor"] != 0 and info["weight_family"] > 0, info
    else:
        assert info["weight_vendor"] == 0, info
    assert isinstance(info["weight_family"], int) and isinstance(info["weight_model"], int)
    done.append("ctx_info: the weight class key of this machine, and the built-in row")
    assert door.error() == "", door.error()
    done.append("error: no reason stored after OK calls")
    door.host_threads(2)
    assert door.ctx_info()["host_threads"] == 2
    e = raises(o.BadRequest, lambda: door.host_threads(-1), "host_threads: a negative count")
    refusal_text(door, e)
    door.host_threads(0)
    done.append("host_threads: set, read back, a negative count refused with the door's reason")

    # ---- units ----
    e = raises(
        ValueError, lambda: door.unit_load(FIXTURE, "axpb", b"short"), "unit_load: a short digest"
    )
    assert "32" in str(e)
    e = raises(
        o.BadRequest,
        lambda: door.unit_load(FIXTURE, "axpb", digest(0x00)),
        "unit_load: a wrong digest",
    )
    refusal_text(door, e)
    e = raises(
        o.Fault,
        lambda: door.unit_load("/nonexistent/unit.so", "axpb", digest(0xA1)),
        "unit_load: no library",
    )
    refusal_text(door, e)
    e = raises(
        o.BadRequest,
        lambda: door.unit_load(FIXTURE, "axpb_split", digest(0xC3)),
        "unit_load: a stage missing",
    )
    assert "is not in deps" in refusal_text(door, e)
    axpb = door.unit_load(FIXTURE, "axpb", digest(0xA1))
    assert axpb.kind == "unit" and axpb.live
    stage = door.unit_load(FIXTURE, "axpb", digest(0xA1))
    split = door.unit_load(FIXTURE, "axpb_split", digest(0xC3), deps=[stage])
    for handle in (split, stage):
        door.unit_free(handle)
    done.append(
        "unit_load: with and without dependencies; a digest, a library and a missing stage refused"
    )

    # ---- planes, plans, bind, launch: one host launch end to end ----
    x = np.arange(N, dtype=np.float64) * 0.5
    y = np.zeros(N)
    px, py = door.plane_wrap(x), door.plane_wrap(y)
    for structure in (o.EG_STRUCTURE_HOST_TEAM, o.EG_STRUCTURE_HOST_SERIAL):
        plan = door.plan_new(axpb, structure, o.EG_REDUCE_COMPENSATED, N)
        door.bind(plan, [0, 1], [py, px], [2, 3], [uniform(2.0), uniform(-1.5)])
        y[:] = 0.0
        tiles = door.launch(plan)
        assert isinstance(tiles, int) and tiles >= 1, tiles
        assert np.array_equal(y, 2.0 * x - 1.5)
        door.plan_free(plan)
    done.append("plane_wrap, plan_new, bind, launch: y = a*x + b equals numpy (team and serial)")
    plan = door.plan_new(
        axpb, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, N, [0, N // 2], [N // 2, N // 2]
    )
    door.bind(plan, [0, 1], [py, px], [2, 3], [uniform(3.0), uniform(0.0)])
    door.launch(plan)
    assert np.array_equal(y, 3.0 * x)
    door.plan_free(plan)
    e = raises(
        o.BadRequest,
        lambda: door.plan_new(axpb, o.EG_STRUCTURE_HOST_SERIAL, 0, N, [0, 1], [N, N]),
        "plan_new: partitions that overlap",
    )
    refusal_text(door, e)
    raises(
        ValueError,
        lambda: door.plan_new(axpb, 0, 0, N, [0], []),
        "plan_new: parts of different length",
    )
    done.append("plan_new with partitions; overlapping partitions refused with the door's reason")

    plan = door.plan_new(axpb, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, N)
    e = raises(o.BadRequest, lambda: door.bind(plan, [99], [px]), "bind: an unknown slot")
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.launch(plan), "launch: a slot unbound")
    refusal_text(door, e)
    raises(
        ValueError,
        lambda: door.bind(plan, [0, 1], [py]),
        "bind: slots and planes of different length",
    )
    raises(
        ValueError,
        lambda: door.bind(plan, [0], [py], [2], [b"123456789"]),
        "bind: a uniform of 9 bytes",
    )
    raises(TypeError, lambda: door.bind(plan, [0], [plan]), "bind: a handle of another kind")
    raises(TypeError, lambda: o.Door(cpu).launch(plan), "launch: a handle of another Door")
    done.append("bind and launch refusals: the door's reasons, and the binding's argument checks")

    # ---- clear and set_count ----
    door.bind(plan, [0, 1], [py, px], [2, 3], [uniform(1.0), uniform(5.0)])
    door.launch(plan)
    assert np.array_equal(y, x + 5.0)
    door.clear(plan, [0])
    assert not y.any()
    door.clear(plan, [])
    e = raises(o.BadRequest, lambda: door.clear(plan, [1]), "clear: an input slot")
    assert "not an output plane" in refusal_text(door, e)
    e = raises(
        o.BadRequest, lambda: door.set_count(plan, 1), "set_count: a unit with no index count slot"
    )
    assert "has no index count slot" in refusal_text(door, e)
    raises(
        OverflowError,
        lambda: door.set_count(plan, 1 << 32),
        "set_count: a count of more than 32 bits",
    )
    door.plan_free(plan)
    prefix = door.unit_load(FIXTURE, "prefix", digest(0x92))
    count = np.zeros(1, dtype=np.uint32)
    pc = door.plane_wrap(count)
    plan = door.plan_new(prefix, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 64)
    px64, py64 = door.plane_wrap(x[:64]), door.plane_wrap(y[:64])
    door.bind(plan, [0, 1, 2], [py64, px64, pc])
    y[:] = -1.0
    door.set_count(plan, 21)
    assert count[0] == 21
    door.launch(plan)
    assert np.array_equal(y[:21], 2.0 * x[:21] + 1.0) and (y[21:] == -1.0).all()
    e = raises(o.BadRequest, lambda: door.set_count(plan, 65), "set_count: above the sample count")
    refusal_text(door, e)
    for handle in (px64, py64, pc):
        door.plane_free(handle)
    door.plan_free(plan)
    door.unit_free(prefix)
    done.append(
        "clear and set_count: outputs zeroed, a prefix count written, each refusal by reason"
    )

    # ---- the runner: a finishing unit run to done ----
    count1 = door.unit_load(FIXTURE, "count1", digest(0xD1))
    limit = np.array([1 + (i * 37) % 50 for i in range(N)], dtype=np.float64)
    k = np.zeros(N)
    mask = np.zeros(N, dtype=np.bool_)
    planes = [door.plane_wrap(a) for a in (k, limit, mask)]
    count_plan = door.plan_new(count1, o.EG_STRUCTURE_HOST_TEAM, o.EG_REDUCE_COMPENSATED, N)
    door.bind(count_plan, [0, 1, 2], planes)
    runner = door.runner_new([count_plan], 1000)
    report = door.run(runner)
    assert report["flags"] & o.EG_REPORT_DONE and report["mode"] == o.EG_MODE_HOST, report
    assert report["finished"] == N and report["launches"] == int(limit.max()), report
    assert np.array_equal(k, limit) and mask.all()
    done.append("runner_new, run: a finishing unit runs until every sample is done")
    e = raises(o.BadRequest, lambda: door.run_events(runner), "run_events: no refine run")
    assert "no refine run" in refusal_text(door, e)
    e = raises(
        o.BadRequest,
        lambda: door.reverse(runner, ([0], [planes[0]]), ([], [])),
        "reverse: not recorded",
    )
    refusal_text(door, e)
    e = raises(
        o.BadRequest,
        lambda: door.runner_new([count_plan], 10, flags=o.EG_RUN_RECORD),
        "runner_new: record without derived units",
    )
    assert "needs its derived units" in refusal_text(door, e)
    raises(
        ValueError,
        lambda: door.runner_new([count_plan], 10, derived_roles=[o.EG_DERIVED_TAPE]),
        "runner_new: roles without units",
    )
    done.append("run_events and reverse: refused by name when nothing was refined or recorded")

    # ---- reduce, copy, recording, graphs ----
    cols = np.array([[0.1 * i for i in range(N)]])
    total = np.zeros(1)
    copy_to = np.zeros(1)
    pin, pout, pcopy = (door.plane_wrap(a) for a in (cols, total, copy_to))
    door.reduce(pin, pout, o.EG_REDUCE_SUM, o.EG_REDUCE_COMPENSATED)
    assert total[0] == math.fsum(cols[0]), (total[0], math.fsum(cols[0]))
    door.reduce(pin, pout, o.EG_REDUCE_MAX, o.EG_REDUCE_FIXED)
    assert total[0] == cols.max()
    e = raises(
        o.BadRequest,
        lambda: door.reduce(pin, pin, o.EG_REDUCE_SUM, o.EG_REDUCE_FIXED),
        "reduce: a shape that disagrees",
    )
    refusal_text(door, e)
    door.copy(pout, pcopy)
    assert copy_to[0] == total[0]
    done.append("reduce and copy: a compensated sum equals fsum, a max, a copy; a shape refused")
    ties = np.array([[4.0, 2.0, 7.0, 2.0, 7.0]])
    low, lane = np.zeros(1), np.full(1, -1, dtype=np.int32)
    pties, plow, plane = (door.plane_wrap(a) for a in (ties, low, lane))
    door.reduce(pties, plow, o.EG_REDUCE_MIN, o.EG_REDUCE_COMPENSATED, index=plane)
    assert (low[0], lane[0]) == (2.0, 1), (low, lane)
    door.reduce(pties, plow, o.EG_REDUCE_MAX, o.EG_REDUCE_COMPENSATED, index=plane)
    assert (low[0], lane[0]) == (7.0, 2), (low, lane)
    raises(
        o.BadRequest,
        lambda: door.reduce(pties, plow, o.EG_REDUCE_SUM, o.EG_REDUCE_COMPENSATED, index=plane),
        "reduce: an index with a sum",
    )
    flags_in, flag_out = np.array([[0, 0, 1]], dtype=np.bool_), np.zeros(1, dtype=np.bool_)
    pflags, pflag = door.plane_wrap(flags_in), door.plane_wrap(flag_out)
    door.reduce(pflags, pflag, o.EG_REDUCE_ANY, o.EG_REDUCE_COMPENSATED)
    assert flag_out[0]
    done.append("reduce with index and any: the lowest lane of a tie, a refusal by name, the or of a mask")
    e = raises(o.BadRequest, lambda: door.record_end(), "record_end: not recording")
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.record_cond_end(), "record_cond_end: no conditional open")
    refusal_text(door, e)
    y2 = np.zeros(N)
    py2 = door.plane_wrap(y2)
    plan2 = door.plan_new(axpb, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, N)
    door.bind(plan2, [0, 1], [py2, px], [2, 3], [uniform(2.0), uniform(1.0)])
    door.record_begin()
    e = raises(o.BadRequest, lambda: door.record_begin(), "record_begin: already recording")
    refusal_text(door, e)
    door.launch(plan2)
    assert not y2.any()
    e = raises(o.BadRequest, lambda: door.bind(plan2, [0], [py2]), "bind: while recording")
    assert "while recording" in refusal_text(door, e)
    graph = door.record_end()
    assert graph.kind == "graph"
    door.graph_launch(graph)
    assert np.array_equal(y2, 2.0 * x + 1.0)
    y2[:] = 0.0
    door.graph_launch(graph)
    assert np.array_equal(y2, 2.0 * x + 1.0)
    door.graph_free(graph)
    done.append(
        "record_begin, record_end, graph_launch, graph_free: a recorded launch runs only on replay"
    )
    pcell = door.plane_wrap(np.zeros(1, dtype=np.int32))
    door.record_begin()
    door.record_cond_begin(o.EG_COND_IF, pcell)
    door.record_cond_end()
    door.graph_free(door.record_end())
    done.append(
        "record_cond_begin, record_cond_end: a conditional opened and closed inside a recording"
    )

    # ---- the device group on a door built without it: no device is counted, the calls that
    # need one are NOT_WRITTEN, sync answers ----
    if device is None:
        assert door.device_count() == 0
        for call in (
            lambda: door.device_props(0),
            lambda: door.stream_use(0),
        ):
            e = raises(o.NotWritten, call, "the device group: not built")
            assert "the device group is not built" in refusal_text(door, e)
        assert door.sync() is None
        e = raises(
            o.NotWritten,
            lambda: o.Door(cpu, device=0),
            "Door: a device context without a device",
        )
        assert "EG_CTX_DEVICE" in str(e), str(e)
        done.append("the device group without a device: none counted, NOT_WRITTEN by reason")
    else:
        not_run.append("the device group without a device: this door has a device group")

    # ---- a host context is one queue: the three stream calls refuse it by the same words ----
    for call in (
        lambda: door.stream_new("cuda:0"),
        lambda: door.stream_free(1),
        lambda: door.stream_sync(1),
    ):
        e = raises(o.BadRequest, call, "stream calls on a host context")
        assert "host-only context: it has one queue" in refusal_text(door, e)
    done.append("stream_new, stream_free, stream_sync: a host context refuses them by name")

    # ---- events: a host context stamps a steady clock; an event never recorded is refused by name ----
    first, second = door.event_new(), door.event_new()
    e = raises(o.BadRequest, lambda: door.event_elapsed(first, second), "event_elapsed: never recorded")
    assert "start event was never recorded" in refusal_text(door, e)
    door.event_record(first)
    door.event_record(second)
    door.event_wait(first)
    assert door.event_elapsed(first, second) >= 0.0
    door.event_free(first)
    door.event_free(second)
    raises(ValueError, lambda: door.event_record(first), "an event handle after its free")
    done.append("event_new, event_record, event_wait, event_elapsed, event_free on a host context")

    # ---- scan: the running fold of a plane, numpy as the oracle ----
    values = np.arange(1, 2051, dtype=np.int32) % 17 - 8
    result = np.zeros_like(values)
    source, scanned = door.plane_wrap(values), door.plane_wrap(result)
    door.scan(source, scanned, o.EG_SCAN_SUM)
    assert (result == np.cumsum(values, dtype=np.int32)).all()
    door.scan(source, scanned, o.EG_SCAN_MAX, o.EG_SCAN_EXCLUSIVE)
    e = raises(o.BadRequest, lambda: door.scan(source, source, o.EG_SCAN_SUM), "scan: out is in")
    assert "overlaps" in refusal_text(door, e)
    door.plane_free(source)
    door.plane_free(scanned)
    done.append("scan: an inclusive sum equals numpy cumsum, an exclusive maximum runs, out = in is refused by name")

    # ---- compact: the ascending lanes a bool plane keeps, numpy as the oracle ----
    wanted_mask = np.arange(3000) % 7 == 0
    compacted = np.full(3000, -1, dtype=np.int32)
    compact_total = np.zeros(1, dtype=np.uint32)
    keep_p, kept_p, compact_total_p = door.plane_wrap(wanted_mask), door.plane_wrap(compacted), door.plane_wrap(compact_total)
    door.compact(keep_p, kept_p, compact_total_p)
    wanted_ids = np.flatnonzero(wanted_mask)
    assert compact_total[0] == len(wanted_ids) and (compacted[: compact_total[0]] == wanted_ids).all() and (compacted[compact_total[0] :] == -1).all()
    e = raises(o.BadRequest, lambda: door.compact(keep_p, keep_p, compact_total_p), "compact: kept is keep")
    refusal_text(door, e)
    for handle in (keep_p, kept_p, compact_total_p):
        door.plane_free(handle)
    done.append("compact: kept[:count] equals flatnonzero, the tail is untouched, a wrong kept is refused by name")

    # ---- order_by_key: the stable permutation by key and the first place of each key, numpy as the oracle ----
    sort_keys = (np.arange(3000, dtype=np.int32) * 7919) % 13
    sorted_order = np.zeros(3000, dtype=np.int32)
    sorted_starts = np.zeros(14, dtype=np.int32)
    keys_p, order_p, starts_p = door.plane_wrap(sort_keys), door.plane_wrap(sorted_order), door.plane_wrap(sorted_starts)
    door.order_by_key(keys_p, order_p, starts_p, 13)
    assert (sorted_order == np.argsort(sort_keys, kind="stable")).all()
    assert (sorted_starts == np.searchsorted(sort_keys[sorted_order], np.arange(14))).all()
    short_starts_p = door.plane_wrap(np.zeros(13, dtype=np.int32))
    e = raises(o.BadRequest, lambda: door.order_by_key(keys_p, order_p, short_starts_p, 12), "order_by_key: a key outside the bound")
    assert "keys are outside [0, 12)" in refusal_text(door, e)
    for handle in (keys_p, order_p, starts_p, short_starts_p):
        door.plane_free(handle)
    done.append("order_by_key: order and starts equal numpy argsort and searchsorted, a key outside the bound is refused with the count")

    # ---- reorder: planes permuted in place by an order, numpy fancy indexing as the oracle ----
    reorder_order = np.argsort(sort_keys, kind="stable").astype(np.int32)
    reorder_ints = (np.arange(3000, dtype=np.int64) * 31) % 1009
    reorder_reals = np.linspace(0.0, 1.0, 3000)
    reorder_rows = np.arange(9000, dtype=np.float32).reshape(3, 3000)
    reorder_bufs = (reorder_ints.copy(), reorder_reals.copy(), reorder_rows.copy())
    reorder_ps = [door.plane_wrap(b) for b in reorder_bufs]
    reorder_order_p = door.plane_wrap(reorder_order)
    door.reorder(reorder_ps, reorder_order_p)
    assert (reorder_bufs[0] == reorder_ints[reorder_order]).all() and (reorder_bufs[1] == reorder_reals[reorder_order]).all()
    assert (reorder_bufs[2] == reorder_rows[:, reorder_order]).all()
    reorder_order[7] = 3000
    before = [b.copy() for b in reorder_bufs]
    e = raises(o.BadRequest, lambda: door.reorder(reorder_ps, reorder_order_p), "reorder: an entry outside the extent")
    assert "1 entries of order are outside [0, 3000)" in refusal_text(door, e)
    assert all((b == c).all() for b, c in zip(reorder_bufs, before))
    e = raises(o.BadRequest, lambda: door.reorder([], reorder_order_p), "reorder: no planes")
    assert "n_planes is 0" in refusal_text(door, e)
    for handle in (*reorder_ps, reorder_order_p):
        door.plane_free(handle)
    done.append("reorder: planes of three dtypes and a vector value equal numpy p[order], an entry outside the extent is refused with the count and moves nothing")

    # ---- the arms of a conditional, by kind ----
    cell = door.plane_wrap(np.zeros(1, dtype=np.int32))
    for kind, n_arms, arms_ok in (
        (o.EG_COND_IF, 0, 0),
        (o.EG_COND_IF_ELSE, 0, 1),
        (o.EG_COND_SWITCH, 3, 2),
    ):
        door.record_begin()
        door.record_cond_begin(kind, cell, n_arms)
        for _ in range(arms_ok):
            door.record_cond_arm()
        e = raises(
            o.BadRequest, door.record_cond_arm, f"record_cond_arm: past the last arm ({kind})"
        )
        refusal_text(door, e)
        door.record_cond_end()
        door.graph_free(door.record_end())
    done.append("record_cond_arm: IF refused, IF_ELSE takes one more arm, SWITCH one per case")
    door.plane_free(cell)

    # ---- plane ownership: the door holds the producer's array until the plane is freed ----
    owned = np.zeros(8)
    alive = weakref.ref(owned)
    plane = door.plane_wrap(owned)
    del owned
    gc.collect()
    assert alive() is not None
    door.plane_free(plane)
    gc.collect()
    assert alive() is None

    def refused_wrap():
        """Wrap an array of a dtype the door does not run; answer a weak reference to it."""
        half = np.zeros(8, dtype=np.float16)
        e = raises(
            o.NotWritten, lambda: door.plane_wrap(half), "plane_wrap: a dtype it does not run"
        )
        refusal_text(door, e)
        return weakref.ref(half)

    alive = refused_wrap()
    gc.collect()
    assert alive() is None
    raises(AttributeError, lambda: door.plane_wrap(3), "plane_wrap: an object with no DLPack")

    class NoCapsule:
        """A producer that answers with something that is no capsule."""

        def __dlpack__(self, **options):
            """Answer an integer."""
            return 42

    raises(
        TypeError,
        lambda: door.plane_wrap(NoCapsule()),
        "plane_wrap: a producer that gives no capsule",
    )
    done.append("plane_wrap: the door owns the tensor after a wrap, the caller after a refusal")

    # ---- banded cells ----
    values = [0.0, 1.0, -2.5, 0.125, 3.75]
    cells = door.banded_pack(values)
    assert all(isinstance(c, int) and 0 <= c < 1 << 64 for c in cells) and len(cells) == 5
    assert door.banded_unpack(cells) == tuple(values)
    assert door.banded_pack_scalar(-2.5) == cells[2]
    assert door.banded_pack([]) == () and door.banded_unpack(()) == ()
    inf = float("inf")
    special = door.banded_unpack(door.banded_pack([float("nan"), inf, -inf]))
    assert math.isnan(special[0]) and special[1:] == (inf, -inf), special
    flushed = door.banded_unpack(door.banded_pack([5e-324, -5e-324]))
    assert flushed == (0.0, 0.0) and all(math.copysign(1.0, v) == 1.0 for v in flushed), flushed
    for bad in (1e300, 1e-300):
        e = raises(
            o.BadRequest, lambda: door.banded_pack([1.0, bad]), "banded_pack: outside the band"
        )
        assert "outside tier 1" in refusal_text(door, e)
        raises(o.BadRequest, lambda: door.banded_pack_scalar(bad), "banded_pack_scalar: outside")
    done.append(
        "banded cells: round trip, NaN and inf kept, subnormals flushed, out-of-band refused"
    )

    # ---- planes the door makes, exported through DLPack ----
    made = door.plane_new([3, 4], 2, 64, flags=o.EG_PLANE_ZERO)
    export = Export(door.plane_dlpack(made))
    door.plane_free(made)
    view = np.from_dlpack(export)
    assert view.shape == (3, 4) and view.dtype == np.float64 and not view.any()
    mine = door.plane_new([N], 2, 64, flags=o.EG_PLANE_ZERO)
    mine_view = np.from_dlpack(Export(door.plane_dlpack(mine)))
    plan = door.plan_new(axpb, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, N)
    door.bind(plan, [0, 1], [mine, px], [2, 3], [uniform(4.0), uniform(1.0)])
    door.launch(plan)
    assert np.array_equal(mine_view, 4.0 * x + 1.0)
    done.append("plane_new, plane_dlpack: numpy sees shape, dtype, zeros and a launch's results")
    if device is None:
        for kind in (o.EG_MEM_PINNED, o.EG_MEM_DEVICE):
            e = raises(
                o.NotWritten, lambda: door.plane_new([2], 2, 64, kind=kind), "plane_new: no device"
            )
            refusal_text(door, e)
        done.append("plane_new: pinned and device memory refused on a door without a device")
    else:
        not_run.append("plane_new on a door without a device: this door has a device group")
    e = raises(
        o.NotWritten, lambda: door.plane_new([2], 2, 16), "plane_new: a dtype it does not run"
    )
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.plane_new([-1], 2, 64), "plane_new: a negative extent")
    refusal_text(door, e)
    done.append("plane_new: a dtype and an extent refused by reason")
    del mine_view, view
    door.plan_free(plan)
    door.plane_free(mine)

    # ---- figures, reset ----
    figures = door.runner_figures(runner)
    assert figures == {
        "mode": o.EG_MODE_HOST,
        "block": -1,
        "chunk": -1,
        "steps_reverse": o.EG_STEPS_REVERSE_PER_STEP,
        "grid": -1,
        "reverse_passes": 0,
        "reverse_bitwise": 0,
    }, figures
    entry = door.plan_entry_figures(count_plan)
    assert entry["flags"] == 0 and not any(v for k, v in entry.items()), entry
    done.append(
        "runner_figures, plan_entry_figures: a host runner has no launch, a host plan no entry"
    )
    k0, mask0 = k.copy(), mask.copy()
    k[:], mask[:] = 0.0, False
    door.runner_reset(runner)
    again = door.run(runner)
    assert again == report and np.array_equal(k, k0) and np.array_equal(mask, mask0)
    done.append("runner_reset: a second run over rewritten state equals the first")

    # ---- reset drops a recorded run: reverse refused by name (fixture x <- a*x + b, 10 steps) ----
    n, steps = 37, 10
    lin = door.unit_load(REVERSE, "lin", digest(0xE0))
    tape = door.unit_load(REVERSE, "lin", digest(0xE1))
    vjp = door.unit_load(REVERSE, "lin_vjp", digest(0xE4))
    state = {
        "x": 1.0 + np.arange(n) / 8.0,
        "t": np.zeros(n),
        "T": 1.0 + (np.arange(n) * 7) % steps,
        "done": np.zeros(n, dtype=np.bool_),
    }
    start = {name: a.copy() for name, a in state.items()}
    lin_planes = [door.plane_wrap(state[name]) for name in ("x", "t", "done", "T")]
    lin_plan = door.plan_new(lin, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, n)
    door.bind(lin_plan, [0, 1, 2, 4], lin_planes, [5, 6], [uniform(0.5), uniform(0.25)])
    recorded = door.runner_new(
        [lin_plan],
        steps,
        flags=o.EG_RUN_RECORD,
        derived_roles=[o.EG_DERIVED_TAPE, o.EG_DERIVED_STEP_VJP],
        derived_units=[tape, vjp],
        window=1,
    )
    first = door.run(recorded)
    assert first["finished"] == n and state["done"].all()
    first_x = state["x"].copy()
    door.runner_reset(recorded)
    seed = door.plane_wrap(np.ones(n))
    e = raises(
        o.BadRequest,
        lambda: door.reverse(recorded, ([0], [seed]), ([], [])),
        "reverse: after reset",
    )
    assert "no recorded run to reverse" in refusal_text(door, e), str(e)
    for name, a in state.items():
        a[:] = start[name]
    assert door.run(recorded) == first and np.array_equal(state["x"], first_x)
    done.append(
        "runner_reset: the recorded run is dropped, eg_reverse refuses it by name; a rerun equals"
    )

    check_fft(door, done, raises)
    check_gemm(door, done, raises)
    check_graph_reverse(door, done, raises)
    check_host_nodes(door, cpu, done, raises)
    check_fork_and_cond(door, done, raises)
    check_reverse_structure(door, done)
    check_step_graphs_and_loops(door, done, raises)
    check_gather_mask(door, done, raises)
    check_gather_table(door, done, raises)
    check_reverse_options(door, done, raises)
    check_record_abort(door, done, raises)
    check_segmented(door, done, raises)
    check_auto_set(door, done, raises)
    if device is None:
        not_run.extend(
            f"{row}: no device"
            for row in (
                "banded slot on a device plan",
                "device gemm_new, gemm_run",
                "gemm units on a device plan",
            )
        )
    else:
        check_device(device, done, raises)

    # ---- handles and the context's end ----
    plane = door.plane_wrap(np.zeros(2))
    door.plane_free(plane)
    assert not plane.live
    raises(ValueError, lambda: door.plane_free(plane), "plane_free: a second free of one handle")
    raises(TypeError, lambda: o.Handle(), "Handle(): handles are made by the door")
    other = o.Door(cpu)
    raises(
        TypeError,
        lambda: other.plane_free(door.plane_wrap(np.zeros(2))),
        "free: a handle of another Door",
    )
    done.append("handles: freed once, never made by the caller, never crossing Doors")

    # two opened handles of one cpu library are two contexts: a handle of one is refused by the
    # other, and freeing the first context leaves the second one working
    first, second = o.Door(cpu), o.Door(backends["second"])
    mine = first.plane_new([2], 2, 64)
    raises(TypeError, lambda: second.plane_free(mine), "free: a handle of the other opened backend")
    first.ctx_free()
    raises(ValueError, first.ctx_info, "a call on the first context, freed")
    assert second.ctx_info()["abi_major"] == 1
    second.ctx_free()
    done.append("two opened handles of the cpu library: two contexts, no shared door state")

    if hasattr(os, "fork"):
        child = os.fork()
        if child == 0:
            code = 3
            try:
                door.ctx_info()
            except o.BadRequest:
                code = 0
            os._exit(code)
        assert os.waitpid(child, 0)[1] == 0, "a forked child was not refused"
        done.append("fork: a call on the context in a forked child is refused")
    # a Door dropped, and a context freed, while an export is outstanding: the array stays readable
    for _ in range(50):
        owner = o.Door(cpu)
        plane = owner.plane_new([4], 2, 64, flags=o.EG_PLANE_ZERO)
        outstanding = np.from_dlpack(Export(owner.plane_dlpack(plane)))
        owner.plane_free(plane)
        owner.ctx_free()
        raises(ValueError, owner.ctx_info, "a call on a closed context")
        outstanding[:] = 7.0
        assert (outstanding == 7.0).all()
        del owner
        outstanding = None
    kept = o.Door(cpu).plane_new([2], 2, 64)  # the Handle keeps its Door alive
    dropped = o.Door(cpu)
    held = np.from_dlpack(Export(dropped.plane_dlpack(dropped.plane_new([3], 2, 64, flags=1))))
    del dropped
    gc.collect()
    assert held.shape == (3,) and not held.any()
    del held, kept
    gc.collect()
    done.append(
        "ctx_free and a dropped Door with an export outstanding: no error, the array stays readable"
    )
    door.ctx_free()
    raises(ValueError, door.ctx_info, "a call after ctx_free")
    raises(ValueError, lambda: door.plan_free(plan), "a handle after ctx_free")
    door.ctx_free()
    done.append(
        "ctx_free: the context is freed once; later calls are refused, a second free is a no-op"
    )
    names = sorted(n for n in dir(o) if n.startswith("EG_"))
    assert "EG_STRUCTURE_HOST_TEAM" in names and "Door" in dir(o) and "Handle" in dir(o)
    done.append(f"door names: Door, Handle and {len(names)} constants")
