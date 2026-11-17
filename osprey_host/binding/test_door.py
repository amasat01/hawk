"""Checks of the binding's door calls, run from test_binding.py on a host context.

The door library is EAGLE_DOOR_DIR/libeagle_door.<dylib or so>; the units come from the door's own
fixture, built beside the extension as libfixture_unit.<dylib or so>. numpy is the oracle.
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
DOOR_LIB = os.path.join(os.environ["EAGLE_DOOR_DIR"], f"libeagle_door.{EXT}")
FIXTURE = os.path.join(HERE, "build", f"libfixture_unit.{EXT}")
REVERSE = os.path.join(HERE, "build", f"libfixture_reverse.{EXT}")
GEMM = os.path.join(HERE, "build", f"libfixture_gemm.{EXT}")
# the unit libraries with device images are built by the door's own device build, beside its library
DEVICE_FIXTURE = os.path.join(os.environ["EAGLE_DOOR_DIR"], f"libfixture_device.{EXT}")
DEVICE_GEMM = os.path.join(os.environ["EAGLE_DOOR_DIR"], f"libfixture_gemm.{EXT}")
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
    root = os.environ["EAGLE_DOOR_DIR"]
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


def device_door():
    """A context on device 0, or None when the door was built without a device group."""
    try:
        return o.Door(DOOR_LIB, device=0)
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


def run(done, raises, not_run):
    """Run every row; `done` collects the passed rows, `not_run` the rows the door cannot run.

    `raises` is the harness's check.
    """
    # ---- the context: opening, refusals of the library, information ----
    e = raises(
        o.Fault, lambda: o.Door("/nonexistent/libnothing"), "Door: a path that is no library"
    )
    assert "nonexistent" in str(e), str(e)
    e = raises(o.Fault, lambda: o.Door(FIXTURE), "Door: a library without the door's functions")
    assert "no function eg_" in str(e), str(e)
    door = o.Door(DOOR_LIB)
    device = device_door()
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
    raises(TypeError, lambda: o.Door(DOOR_LIB).launch(plan), "launch: a handle of another Door")
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

    # ---- the device group on a door built without it: NOT_WRITTEN everywhere, sync excepted ----
    if device is None:
        for call in (
            door.device_count,
            lambda: door.device_props(0),
            lambda: door.stream_use(0),
            lambda: door.record_child(0),
        ):
            e = raises(o.NotWritten, call, "the device group: not built")
            assert "the device group is not built" in refusal_text(door, e)
        assert door.sync() is None
        e = raises(
            o.NotWritten,
            lambda: o.Door(DOOR_LIB, device=0),
            "Door: a device context without a device",
        )
        assert "EG_CTX_DEVICE" in str(e), str(e)
        done.append("the device group without a device: NOT_WRITTEN by reason, sync answers")
    else:
        not_run.append("the device group without a device: this door has a device group")

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
    other = o.Door(DOOR_LIB)
    raises(
        TypeError,
        lambda: other.plane_free(door.plane_wrap(np.zeros(2))),
        "free: a handle of another Door",
    )
    done.append("handles: freed once, never made by the caller, never crossing Doors")

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
        owner = o.Door(DOOR_LIB)
        plane = owner.plane_new([4], 2, 64, flags=o.EG_PLANE_ZERO)
        outstanding = np.from_dlpack(Export(owner.plane_dlpack(plane)))
        owner.plane_free(plane)
        owner.ctx_free()
        raises(ValueError, owner.ctx_info, "a call on a closed context")
        outstanding[:] = 7.0
        assert (outstanding == 7.0).all()
        del owner
        outstanding = None
    kept = o.Door(DOOR_LIB).plane_new([2], 2, 64)  # the Handle keeps its Door alive
    dropped = o.Door(DOOR_LIB)
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
