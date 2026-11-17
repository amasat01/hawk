"""Checks of the binding's door calls, run from test_binding.py on a host context.

The door library is EAGLE_DOOR_DIR/libeagle_door.<dylib or so>; the units come from the door's own
fixture, built beside the extension as libfixture_unit.<dylib or so>. numpy is the oracle.
"""

import gc
import math
import os
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


def run(done, raises):
    """Run every row; `done` collects the passed rows, `raises` is the harness's check."""
    # ---- the context: opening, refusals of the library, information ----
    e = raises(
        o.Fault, lambda: o.Door("/nonexistent/libnothing"), "Door: a path that is no library"
    )
    assert "nonexistent" in str(e), str(e)
    e = raises(o.Fault, lambda: o.Door(FIXTURE), "Door: a library without the door's functions")
    assert "no function eg_" in str(e), str(e)
    door = o.Door(DOOR_LIB)
    info = door.ctx_info()
    assert info["abi_major"] == 1 and info["device"] == -1 and info["stream"] == -1, info
    assert info["capabilities"] & o.EG_CAP_HOST, info
    done.append("Door, ctx_info: a host context")
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
        o.NotWritten, lambda: o.Door(DOOR_LIB, device=0), "Door: a device context without a device"
    )
    assert "EG_CTX_DEVICE" in str(e), str(e)
    done.append("the device group without a device: NOT_WRITTEN by reason, sync answers")

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
    for kind in (o.EG_MEM_PINNED, o.EG_MEM_DEVICE):
        e = raises(
            o.NotWritten, lambda: door.plane_new([2], 2, 64, kind=kind), "plane_new: no device"
        )
        refusal_text(door, e)
    e = raises(
        o.NotWritten, lambda: door.plane_new([2], 2, 16), "plane_new: a dtype it does not run"
    )
    refusal_text(door, e)
    e = raises(o.BadRequest, lambda: door.plane_new([-1], 2, 64), "plane_new: a negative extent")
    refusal_text(door, e)
    done.append(
        "plane_new: pinned and device memory on a host door, a dtype, an extent refused by reason"
    )
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
