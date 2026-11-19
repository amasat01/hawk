"""One kernel through the cpu library's folder, from Python only, on the real host compiler.

The caller knows nothing of the target: the manifest, `machine`, `lower`, one `build` for each
unit, `merge`, `set_text`, `link`, `state`. It hands each answer on as given and reads no card,
no compiler identity and no flag. A cuda folder on a machine with no device tool answers `machine`
with `tools_absent` and refuses `build` by name; on a machine with one it states the tool and builds
a device unit.
"""

import ctypes
import json
import os
import subprocess
import sys
import tempfile

LEAF = (
    "(('kind', 'vocab_read'), ('role', 'per_sample'), ('name', '%s'), ('quantity', None), "
    "('wire_index', 0), ('sorted', False))"
)
ADD = "(('kind', 'add'), ('literal', None))"
ASSIGN = "(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))"
REQUEST = {
    "v": 1,
    "name": "axpb",
    "scalar": "Float64",
    "targets": ["Host"],
    "kind": {"guard": "Default", "active_set": False, "sink": "Plain"},
    "body": {"Plain": {"name": "axpb", "sinks": [3], "declared": [], "exempt": []}},
}


def library_of(folder):
    """Return the manifest of an OSPREY folder and the path of the library it holds."""
    with open(os.path.join(folder, "osprey.json")) as manifest_file:
        manifest = json.load(manifest_file)
    ext = "dylib" if sys.platform == "darwin" else "so"
    return manifest, os.path.join(folder, f"libosprey_{manifest['name']}.{ext}")


def exported(path):
    """Return the names of the global symbols the library at `path` defines."""
    version = subprocess.run(["nm", "--version"], capture_output=True, text=True).stdout
    # GNU nm has no -U and the macOS nm has no --defined-only
    defined = ["-g", "--defined-only"] if "GNU" in version else ["-gU"]
    shown = subprocess.run(["nm", *defined, path], capture_output=True, text=True, check=True)
    return {line.split()[-1].lstrip("_") for line in shown.stdout.splitlines() if line.strip()}


class ManagedTensor(ctypes.Structure):
    """The leading members of a DLPack 1.0 managed tensor, up to its data pointer."""

    _fields_ = [
        ("version", ctypes.c_uint32 * 2),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("data", ctypes.c_void_p),
    ]


def lanes(capsule, count):
    """Return the float64 memory of a plane's DLPack capsule as a ctypes array of `count` lanes.

    The array reads and writes the plane's own memory and is valid while the capsule is alive.
    """
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    tensor = ManagedTensor.from_address(get(capsule, b"dltensor_versioned"))
    return (ctypes.c_double * count).from_address(tensor.data)


def run_through_runtime(o, lib, library, unit):
    """Load the linked unit through the cpu library's runtime, run it on planes, compare with a + b.

    The oracle is plain Python: the sum of two floats, which is the one rounding the unit does.
    """
    count = 8
    a_values = [0.5 * i for i in range(count)]
    b_values = [3.0 * i - 1.25 for i in range(count)]
    door = o.Door(lib)
    loaded = door.unit_load(library, unit["meta"]["name"], bytes.fromhex(unit["meta"]["digest"]))
    # the unit's slots in its meta order: y (written), a, b
    planes = [door.plane_new([count], 2, 64, flags=o.EG_PLANE_ZERO) for _ in range(3)]
    capsules = [door.plane_dlpack(plane) for plane in planes]
    y_lanes, a_lanes, b_lanes = (lanes(capsule, count) for capsule in capsules)
    for i in range(count):
        a_lanes[i], b_lanes[i] = a_values[i], b_values[i]
    plan = door.plan_new(loaded, o.EG_STRUCTURE_HOST_SERIAL, o.EG_REDUCE_COMPENSATED, count)
    door.bind(plan, [0, 1, 2], planes)
    assert door.launch(plan) >= 1
    assert list(y_lanes) == [a + b for a, b in zip(a_values, b_values)], list(y_lanes)
    door.ctx_free()


def same_build_asked_otherwise(c, lib, ask, unset, profile):
    """Check that the stated or environment default is the unset ask, and another level is not."""

    def machine_of(level, options):
        stated_ask = {**ask, "opt_level": level, "options": options}
        return json.loads(c.backend_machine(lib, json.dumps(stated_ask)))

    stated = machine_of("O3", {"profile": profile})
    assert (stated["key"], stated["stated"]["asked"]) == (unset["key"], unset["stated"]["asked"])
    os.environ["HAWK_OPT_LEVEL"] = "O3"
    try:
        from_env = machine_of("", {})
        os.environ["HAWK_OPT_LEVEL"] = "O1"
        lower = machine_of("", {})
    finally:
        del os.environ["HAWK_OPT_LEVEL"]
    assert from_env["key"] == unset["key"]
    assert from_env["stated"]["asked"] == unset["stated"]["asked"]
    assert lower["key"] != unset["key"] and lower["stated"]["asked"]["opt_level"] == "O1"


def run(done, o, raises, folders, header_abi, make_nodes):
    """Check the cpu folder end to end and the cuda folder's refusals; folders maps name to path."""
    work = tempfile.mkdtemp(prefix="osprey_e2e_")
    root = os.path.join(work, "store")
    c = o.Ctx()
    manifest, path = library_of(folders["cpu"])
    assert manifest["door"] == header_abi
    assert manifest["targets"][0]["request"] == "host"
    assert manifest["targets"][0]["target"] == "Host"
    done.append("end to end: the cpu manifest names its door and its target")
    lib = c.backend_open(path)
    ask = {"target": "host", "options": {}, "opt_level": "", "root": root}
    machine = json.loads(c.backend_machine(lib, json.dumps(ask)))
    assert machine["tools_absent"] is None, machine["tools_absent"]
    asked = machine["stated"]["asked"]
    assert asked["opt_level"] == "O3", asked
    profile = asked["options"]["profile"]
    assert set(asked["options"]) == {"profile"}, asked
    done.append("end to end: an unset ask is stated with the default level and profile")
    for named, expected in (("native", True), ("native-vector-math", False)):
        ask_named = {**ask, "options": {"profile": named}}
        stated_named = json.loads(c.backend_machine(lib, json.dumps(ask_named)))["stated"]
        assert stated_named["exact"] is expected, (named, stated_named)
    done.append("end to end: the machine states whether the resolved profile is exact")
    same_build_asked_otherwise(c, lib, ask, machine, profile)
    done.append("end to end: the default stated or set in the environment is one key and record")
    compiler = machine["stated"]["tools"]["compiler"]
    done.append("end to end: machine states the host compiler " + compiler["path"])
    # the kernel: y = a + b
    c3, dump_nodes, sinks = make_nodes()
    # a set carries the self-check exports block once, in its own text: the unit is asked without it
    request = {**REQUEST, "machine": {"Host": machine}, "exports_block": False}
    request["body"]["Plain"]["sinks"] = sinks
    answer = json.loads(c3.backend_lower(lib, c3.dump(dump_nodes), "", json.dumps(request)))
    units = answer["units"]
    assert units, "the kernel lowered to no unit"
    done.append(f"end to end: lower answers {len(units)} unit(s) for the host from the machine")
    parts, metas, keys = [], [], set()
    for unit in units:
        host = next(t for t in unit["texts"] if t["target"] == "Host")
        name = unit["meta"]["name"]
        asked = json.dumps({"name": name, "text": host["text"], "machine": machine})
        built = json.loads(c.backend_build(lib, asked, root))
        assert os.path.isfile(built["files"][built["artifact"]]), built
        assert os.path.isfile(built["files"][built["source"]]), built
        # two units of one text and one name share a key: the later one is served from the store
        assert built["hit"] is (built["key"] in keys), (name, len(units), len(keys))
        keys.add(built["key"])
        again = json.loads(c.backend_build(lib, asked, root))
        assert again["hit"] is True and again["key"] == built["key"], "a second build ran a tool"
        parts.append(built)
        metas.append(unit["meta"])
    done.append("end to end: build compiled each unit once and answered a hit the second time")
    merged = json.loads(c.backend_merge(lib, json.dumps([answer])))
    assert [u["meta"]["name"] for u in merged["units"]] == [m["name"] for m in metas], merged
    done.append("end to end: merge answers the one answer of the lowering")
    set_text = c.backend_set_text(lib, json.dumps({"v": 1, "units": metas}))
    ext = "dylib" if sys.platform == "darwin" else "so"
    linked = json.loads(c.backend_link(lib, json.dumps(parts), set_text, f"libe2e.{ext}", root))
    library = linked["library"]
    ctypes.CDLL(library)
    names = exported(library)
    entries = {u["meta"]["host_entry"] for u in units}
    assert entries <= names, f"entries {sorted(entries - names)} are not exported"
    done.append(
        f"end to end: link made {os.path.basename(library)}; it loads and exports "
        f"{len(entries)} entry(ies) (nm: calling one needs the runtime)"
    )
    run_through_runtime(o, lib, library, units[0])
    done.append("end to end: the linked unit runs through the cpu library's runtime: y = a + b")
    stats = json.loads(c.backend_state(lib, "stats", root))
    assert stats["tool_runs"] >= len(units) + 2 and stats["hits"] >= len(units), stats
    done.append("end to end: state counts the tool runs and the hits")
    e = raises(o.BadRequest, lambda: c.backend_merge(lib, "[]"), "merge: no answers")
    assert "a merge of no answers" in str(e), str(e)
    done.append("end to end: merge refuses an empty list of answers by name")
    # a refusal of the tool side reaches the caller as NotWritten, whatever its kind
    unknown = json.dumps({**ask, "opt_level": "O9"})
    e = raises(o.NotWritten, lambda: c.backend_machine(lib, unknown), "machine: unknown level")
    assert "unknown opt level" in str(e) and "the build's opt_level" in str(e), str(e)
    assert "O3" in str(e), str(e)
    os.environ["HAWK_OPT_LEVEL"] = "O9"
    try:
        e = raises(
            o.NotWritten,
            lambda: c.backend_machine(lib, json.dumps(ask)),
            "machine: unknown level from the variable",
        )
    finally:
        del os.environ["HAWK_OPT_LEVEL"]
    assert "unknown opt level" in str(e) and "$HAWK_OPT_LEVEL" in str(e), str(e)
    assert "O3" in str(e), str(e)
    done.append("end to end: machine refuses an unknown level by name, from either source")
    neither = {k: v for k, v in request.items() if k != "machine"}
    e = raises(
        o.BadRequest,
        lambda: c3.backend_lower(lib, c3.dump(dump_nodes), "", json.dumps(neither)),
        "lower: neither machine nor facts",
    )
    assert "neither `machine`" in str(e) and "nor `facts`" in str(e), str(e)
    done.append("end to end: lower refuses a request with neither machine nor facts by name")
    c.backend_close(lib)
    # the cuda folder: with no device tool, machine says why and build refuses by name;
    # with one, machine states the tool that was found
    _, cuda_path = library_of(folders["cuda"])
    cuda = c.backend_open(cuda_path)
    answered = json.loads(
        c.backend_machine(cuda, json.dumps({"target": "cuda", "options": {}, "opt_level": ""}))
    )
    tooled = answered["tools_absent"] is None
    if tooled:
        stated = answered["stated"]
        assert "compiler" in stated["tools"] or "runtime" in stated, stated
        done.append("end to end: cuda machine states the device tool it found")
        # with a device tool, a device unit is built: its image and the record the host link takes
        unit_request = {"name": "k", "text": "__global__ void k() {}", "machine": answered}
        built = json.loads(c.backend_build(cuda, json.dumps(unit_request), root))
        assert built["artifact"] in ("k.ptx", "k.cubin"), built["artifact"]
        assert built["embed"]["symbol"] == "k__image", built["embed"]
        done.append("end to end: cuda build writes " + built["artifact"] + " with its embed record")
    else:
        assert "HAWK_NVCC" in answered["tools_absent"], answered
        refused_unit = {"name": "k", "text": "int x;", "machine": answered}
        e = raises(
            o.NotWritten,
            lambda: c.backend_build(cuda, json.dumps(refused_unit), root),
            "cuda build: no compiler on this machine",
        )
        assert getattr(e, "code", None) == "tool_absent", (e, getattr(e, "code", None))
        # a device unit's text needs no tool: without a device compiler it takes the card's release
    device = {
        **REQUEST,
        "targets": ["Device"],
        "machine": {"Device": answered},
    }
    if not tooled:
        # a tooled machine takes the format its answer states: a newer compiler refuses "Ptx"
        device["image_format"] = "Ptx"
    device["body"] = {"Plain": {**REQUEST["body"]["Plain"], "sinks": sinks}}
    written = json.loads(c3.backend_lower(cuda, c3.dump(dump_nodes), "", json.dumps(device)))
    unit = written["units"][0]
    assert [t["target"] for t in unit["texts"]] == ["Device"], unit["texts"]
    if not tooled:
        assert unit["meta"]["device_compiler"] == "nvcc 12.6", unit["meta"]["device_compiler"]
    done.append("end to end: cuda lower writes the device text")
    c.backend_close(cuda)
