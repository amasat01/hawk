"""Planes with axes through the cpu library, from an IR document to a run on the host runtime.

Each kernel is written as IR nodes, lowered, built on the host compiler, linked and launched
through the library's own runtime; the values are compared with NumPy indexing, assignment and
`np.add.at`, which are the independent oracle. The runtime fills the extent of each runtime axis
from the bound array's shape, and refuses an array of the wrong shape by name.
"""

import ctypes
import json
import os
import sys
import tempfile

import numpy as np
import test_end_to_end as e2

import _osprey as o

HERE = os.path.dirname(os.path.abspath(__file__))
LEAF = (
    "(('kind', 'vocab_read'), ('role', '%s'), ('name', '%s'), ('quantity', None), "
    "('wire_index', 0), ('sorted', False), ('axes', %s))"
)
EDGES = ("clamp", "wrap", "zero", "unchecked")
INT64 = (0, 64)
FLOAT64 = (2, 64)


class Host:
    """The cpu library opened once, with the store its builds go to."""

    def __init__(self):
        """Open the library of the dist folder (or of OSPREY_DIST) and ask it for this machine."""
        dist = os.environ.get("OSPREY_DIST") or os.path.join(HERE, "..", "dist")
        _, path = e2.library_of(os.path.join(dist, "osprey_cpu"))
        self.ctx = o.Ctx()
        self.lib = self.ctx.backend_open(path)
        self.path = path
        self.root = tempfile.mkdtemp(prefix="osprey_axes_")
        ask = {"target": "host", "options": {}, "opt_level": "", "root": self.root}
        self.machine = json.loads(self.ctx.backend_machine(self.lib, json.dumps(ask)))

    def unit(self, graph):
        """Lower, build and link the kernel `graph`; return unit, host text and library."""
        ctx, nodes, sinks = graph
        request = {
            **e2.REQUEST,
            "machine": {"Host": self.machine},
            "exports_block": False,
            "body": {
                "Plain": {"name": e2.REQUEST["name"], "sinks": sinks, "declared": [], "exempt": []}
            },
        }
        answer = json.loads(ctx.backend_lower(self.lib, ctx.dump(nodes), "", json.dumps(request)))
        unit = answer["units"][0]
        text = next(t for t in unit["texts"] if t["target"] == "Host")["text"]
        asked = json.dumps({"name": unit["meta"]["name"], "text": text, "machine": self.machine})
        built = json.loads(self.ctx.backend_build(self.lib, asked, self.root))
        set_text = self.ctx.backend_set_text(
            self.lib, json.dumps({"v": 1, "units": [unit["meta"]]})
        )
        ext = "dylib" if sys.platform == "darwin" else "so"
        linked = json.loads(
            self.ctx.backend_link(
                self.lib, json.dumps([built]), set_text, f"libaxes.{ext}", self.root
            )
        )
        return unit, text, linked["library"]

    def launch(self, graph, arrays, written, samples):
        """Run the kernel on `arrays` (name -> NumPy array); return the `written` ones."""
        unit, _, library = self.unit(graph)
        door = o.Door(self.lib)
        loaded = door.unit_load(
            library, unit["meta"]["name"], bytes.fromhex(unit["meta"]["digest"])
        )
        slots = [s["name"] for s in unit["meta"]["body"]["slots"]]
        bound = [name for name in slots if name in arrays]
        planes, views = [], {}
        for name in bound:
            planes.append(make_plane(door, arrays[name], views, name))
        plan = door.plan_new(loaded, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, samples)
        door.bind(plan, [slots.index(name) for name in bound], planes)
        door.launch(plan)
        return {name: views[name].copy() for name in written}

    def close(self):
        """Release the library."""
        self.ctx.backend_close(self.lib)


def make_plane(door, array, views, name):
    """A plane holding the array's values; `views[name]` is the NumPy view of the plane's memory."""
    code, bits = INT64 if array.dtype.kind == "i" else FLOAT64
    plane = door.plane_new(list(array.shape), code, bits, flags=o.EG_PLANE_ZERO)
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    capsule = door.plane_dlpack(plane)
    tensor = e2.ManagedTensor.from_address(get(capsule, b"dltensor_versioned"))
    raw = (ctypes.c_char * array.nbytes).from_address(tensor.data)
    view = np.frombuffer(raw, dtype=array.dtype).reshape(array.shape)
    view[...] = array
    views[name] = view
    views[name + "#capsule"] = capsule
    return plane


def leaf(ctx, role, name, dtype, axes="()"):
    """A leaf node of a plane with the axes given as IR text."""
    return ctx.make("Leaf", dtype, [], None, [], LEAF % (role, name, axes))


def at_kernel(edge, table_axes, index_names):
    """s[target] += table[index...] with `edge` on the read, one index plane per axis."""
    ctx = o.Ctx()
    table = leaf(ctx, "lookup", "p", "f64", table_axes)
    indices = [leaf(ctx, "plane_in", n, "i64") for n in index_names]
    target = leaf(ctx, "plane_in", "tg", "i64")
    read = ctx.make(
        "At",
        "f64",
        [],
        None,
        [table, *indices],
        f"(('kind', 'at'), ('literal', None), ('edge', '{edge}'))",
    )
    sink = ctx.make(
        "AccumWrite",
        "f64",
        [],
        None,
        [read, target],
        "(('kind', 'accum_write'), ('name', 's'), ('role', 'accum_out'), ('contract', 'auto'), "
        "('reduced', None), ('edge', 'unchecked'), ('axes', (('t', None),)))",
    )
    nodes = [table, *indices, target, read, sink]
    return ctx, nodes, [len(nodes) - 1]


def store_kernel(name, role, node_class, sink_fields, sink_axes):
    """A store of leaf v at [ii, kk] into a plane of two axes."""
    ctx = o.Ctx()
    value = leaf(ctx, "plane_in", "v", "f64")
    ii, kk = leaf(ctx, "plane_in", "ii", "i64"), leaf(ctx, "plane_in", "kk", "i64")
    sink = ctx.make(
        node_class, "f64", [], None, [value, ii, kk], sink_fields % (name, role, sink_axes)
    )
    return ctx, [value, ii, kk, sink], [3]


def read_oracle(table, edge, rows, cols):
    """NumPy indexing of `table` at the index arrays under the edge rule, stated in words."""
    shape = np.array(table.shape)
    rows = [np.asarray(r) for r in rows]
    if edge == "clamp":
        return table[tuple(np.clip(r, 0, n - 1) for r, n in zip(rows, shape))]
    if edge == "wrap":
        return table[tuple(r % n for r, n in zip(rows, shape))]
    inside = np.logical_and.reduce([(r >= 0) & (r < n) for r, n in zip(rows, shape)])
    safe = table[tuple(np.where(inside, r, 0) for r in rows)]
    return np.where(inside, safe, 0.0)


def check_reads(host, table, index_names, table_axes, label):
    """A table read at random indices equals NumPy indexing for each edge rule."""
    rng = np.random.default_rng(7)
    samples = 40
    out = []
    for edge in EDGES:
        low, high = (0, 0) if edge == "unchecked" else (-3, 3)
        indices = [rng.integers(low, n + high, samples) for n in table.shape]
        arrays = {"p": table, "tg": np.arange(samples, dtype=np.int64), "s": np.zeros(samples)}
        arrays.update({n: i.astype(np.int64) for n, i in zip(index_names, indices)})
        got = host.launch(at_kernel(edge, table_axes, index_names), arrays, ["s"], samples)["s"]
        want = read_oracle(table, edge, indices, None)
        assert np.array_equal(got, want), (label, edge, got, want)
        out.append(edge)
    return out


def test_a_table_with_a_runtime_and_a_static_axis_reads_as_numpy_indexes_it(host):
    """Acceptance (a): `p[i, k]` for each edge rule; the oracle is NumPy indexing."""
    table = np.random.default_rng(1).random((5, 3))
    assert check_reads(host, table, ("ii", "kk"), "(('i', None), ('k', 3))", "2 axes") == list(
        EDGES
    )


def test_a_table_of_three_axes_with_two_runtime_axes_reads_as_numpy_indexes_it(host):
    """Acceptance (b): `p[i, j, k]` with runtime i and k; the oracle is NumPy indexing."""
    table = np.random.default_rng(2).random((4, 3, 6))
    axes = "(('i', None), ('j', 3), ('k', None))"
    assert check_reads(host, table, ("ii", "jj", "kk"), axes, "3 axes") == list(EDGES)


def test_the_two_index_read_is_the_views_call_form_without_a_stride(host):
    """Acceptance (e): the text is the view call with one index per axis, with no stride."""
    _, text, _ = host.unit(at_kernel("clamp", "(('i', None), ('k', 3))", ("ii", "kk")))
    line = next(row for row in text.splitlines() if "lut_p(" in row)
    assert "lut_p(aether::edge::clamp(" in line and line.count("lut_p.extent(") == 2, line
    assert "*" not in line.split("lut_p(", 1)[1], line


def test_a_store_and_an_accumulate_at_two_indices_equal_assignment_and_add_at(host):
    """Acceptance (c): the oracles are NumPy assignment and `np.add.at`."""
    rng = np.random.default_rng(3)
    rows, cols, samples = 8, 3, 20
    ii = rng.integers(0, rows, samples).astype(np.int64)
    kk = rng.integers(0, cols, samples).astype(np.int64)
    v = rng.integers(1, 9, samples).astype(np.float64)
    axes = "(('i', None), ('k', 3))"
    # a store: the pairs are unique, so assignment has one result
    pairs = rng.permutation(rows * cols)[:samples]
    ii_u, kk_u = (pairs // cols).astype(np.int64), (pairs % cols).astype(np.int64)
    stored = host.launch(
        store_kernel(
            "w",
            "wide_out",
            "WideWrite",
            "(('kind', 'wide_write'), ('name', '%s'), ('role', '%s'), ('edge', 'unchecked'), "
            "('axes', %s))",
            axes,
        ),
        {"v": v, "ii": ii_u, "kk": kk_u, "w": np.zeros((rows, cols))},
        ["w"],
        samples,
    )["w"]
    want = np.zeros((rows, cols))
    want[ii_u, kk_u] = v
    assert np.array_equal(stored, want), (stored, want)
    added = host.launch(
        store_kernel(
            "a",
            "accum_out",
            "AccumWrite",
            "(('kind', 'accum_write'), ('name', '%s'), ('role', '%s'), ('contract', 'auto'), "
            "('reduced', None), ('edge', 'unchecked'), ('axes', %s))",
            axes,
        ),
        {"v": v, "ii": ii, "kk": kk, "a": np.zeros((rows, cols))},
        ["a"],
        samples,
    )["a"]
    want = np.zeros((rows, cols))
    np.add.at(want, (ii, kk), v)
    assert np.array_equal(added, want), (added, want)


def test_an_array_of_the_wrong_shape_is_refused_by_plane_axis_and_extent(host):
    """Acceptance (d): the refusal names the plane, the axis and the expected and given extents."""
    graph = at_kernel("clamp", "(('i', None), ('k', 3))", ("ii", "kk"))
    unit, _, library = host.unit(graph)
    door = o.Door(host.lib)
    loaded = door.unit_load(library, unit["meta"]["name"], bytes.fromhex(unit["meta"]["digest"]))
    slots = [s["name"] for s in unit["meta"]["body"]["slots"]]
    plan = door.plan_new(loaded, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 4)
    views = {}
    wrong = make_plane(door, np.zeros((5, 4)), views, "p")
    try:
        door.bind(plan, [slots.index("p")], [wrong])
    except o.BadRequest as refused:
        message = str(refused)
    else:
        raise AssertionError("a table of shape (5, 4) was bound to axes (i, k=3)")
    for part in ("'p'", "axis 1", "'k'", "has extent 4", "named axis has 3"):
        assert part in message, (part, message)


def bound_p(host, table_axes, array):
    """Bind `array` to the table `p` of a read over `table_axes`; the refusal text, or None."""
    unit, _, library = host.unit(at_kernel("clamp", table_axes, ("ii", "kk")))
    door = o.Door(host.lib)
    loaded = door.unit_load(library, unit["meta"]["name"], bytes.fromhex(unit["meta"]["digest"]))
    slots = [s["name"] for s in unit["meta"]["body"]["slots"]]
    plan = door.plan_new(loaded, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, 4)
    plane = make_plane(door, array, {}, "p")
    try:
        door.bind(plan, [slots.index("p")], [plane])
    except o.BadRequest as refused:
        return str(refused)
    return None


def test_a_compact_array_binds_when_one_axis_is_runtime(host):
    """Rule (ii): the element count is the stated axes times the one runtime axis."""
    assert bound_p(host, "(('i', None), ('k', 3))", np.zeros(12)) is None
    message = bound_p(host, "(('i', None), ('k', 3))", np.zeros(7))
    assert message is not None and "7 elements" in message, message


def test_a_compact_array_is_refused_when_two_axes_are_runtime(host):
    """With two runtime axes only the shape (*payload, *axes) is accepted."""
    assert bound_p(host, "(('i', None), ('k', None))", np.zeros((3, 4))) is None
    message = bound_p(host, "(('i', None), ('k', None))", np.zeros(12))
    assert message is not None and "2 runtime axes only the shape" in message, message


def run_all():
    """Run each check on one host library; returns the number run."""
    host = Host()
    tests = [t for name, t in sorted(globals().items()) if name.startswith("test_")]
    for check in tests:
        check(host)
    host.close()
    return len(tests)


if __name__ == "__main__":
    print(f"test_plane_axes: {run_all()} checks passed")
