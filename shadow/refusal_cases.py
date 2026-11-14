# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One case per declared constructor check (42, in core.json order), plus the quantity and type cases.
Each case: raptor's constructor call that trips it (exception class and message), and the door-side description
(a pool of valid nodes and the bad node as class, dtype, shape, tag, operands, state). Standard library and raptor only.
Usage: PYTHONPATH=<overlay> python refusal_cases.py <out.json>"""
import json, re, sys
from raptor.ir import nodes as N
from raptor.ir.types import TensorType, Wire
from raptor.ir.compound import Quantity

class R:  # a node reference by pool name
    def __init__(s, n): s.n = n

def kt(v):
    if isinstance(v, R): return "#" + v.n
    if isinstance(v, tuple): return "(" + ", ".join(kt(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list): return "[" + ", ".join(kt(x) for x in v) + "]"
    if isinstance(v, dict): return "Wire(" + ", ".join(f"{k}={kt(x)}" for k, x in v.items()) + ")"
    return repr(v)

def T(dtype="f64", shape=(), tag=None): return TensorType(shape, dtype, tag)

def spec(cls, ops=(), dtype="f64", shape=(), tag=None, **f):
    return {"class": cls, "dtype": dtype, "shape": list(shape), "tag": tag, "ops": list(ops), "fields": kt(tuple(f.items()))}

LEAF = lambda role, name: spec("Leaf", kind="vocab_read", role=role, name=name, quantity=None, wire_index=0, sorted=False)
POOL = [  # name, python constructor, door spec
    ("x", lambda P: N.Leaf("vocab_read", "vec_in", "x", T()), LEAF("vec_in", "x")),
    ("m", lambda P: N.Leaf("vocab_read", "mutable", "m", T()), LEAF("mutable", "m")),
    ("c1", lambda P: N.Const(1.0, T()), spec("Const", literal=1.0)),
    ("ci", lambda P: N.Const(0, T("i32")), spec("Const", dtype="i32", literal=0)),
    ("cb", lambda P: N.Const(True, T("bool")), spec("Const", dtype="bool", literal=True)),
    ("v3", lambda P: N.Const(0.0, T("f64", (3,))), spec("Const", shape=(3,), literal=0.0)),
    ("m3", lambda P: N.Const(0.0, T("f64", (1, 1, 1))), spec("Const", shape=(1, 1, 1), literal=0.0)),
    ("idx", lambda P: N.LoopIndex(0, "i"), spec("LoopIndex", dtype="i32", depth=0, name="i", bounds=None)),
    ("car", lambda P: N.LoopCarry(0, 0, "c", T()), spec("LoopCarry", depth=0, slot=0, name="c")),
    ("car2", lambda P: N.LoopCarry(0, 1, "c2", T()), spec("LoopCarry", depth=0, slot=1, name="c2")),
    ("car3", lambda P: N.LoopCarry(0, 0, "c3", T("f64", (1, 1, 1))), spec("LoopCarry", shape=(1, 1, 1), depth=0, slot=0, name="c3")),
]

def loop(carries, inits, nexts, start=0, stop=2, step=1, body_sinks=(), exit_cond=None, exit_values=(), count=None, dynamic=None, ttype=None):
    """(python thunk, door spec) of a Loop: the state the constructor would set, by the `out` list of core.json."""
    def trip(a, b, c): return ((b - a + c - 1) // c if b > a else 0) if c > 0 else (((a - b - c - 1) // -c if a > b else 0) if c < 0 else 0)
    r = lambda n: R(n) if n else None
    py = lambda P: N.Loop(P["idx"], tuple(P[c] for c in carries), tuple(P[i] for i in inits), tuple(P[n] for n in nexts), start=start, stop=stop, step=step,
                          body_sinks=tuple(P[b] for b in body_sinks), exit_cond=P[exit_cond] if exit_cond else None, exit_values=tuple(P[e] for e in exit_values),
                          count=P[count] if count else None, dynamic=dynamic)
    ops = list(inits) + ([count] if count else [])
    sp = spec("Loop", ops, **ttype or {}, index=R("idx"), carries=tuple(R(c) for c in carries), nexts=tuple(R(n) for n in nexts), start=start, stop=stop, step=step,
              trip=trip(start, stop, step), carry_count=len(carries), body_sinks=tuple(R(b) for b in body_sinks), _body_key=None, exit_cond=r(exit_cond),
              exit_values=tuple(R(e) for e in exit_values), count=r(count), count_tail=False, dynamic=dynamic, _free_end=len(inits))
    return py, sp

L0 = loop(["car"], ["c1"], ["x"])
POOL.append(("L0", lambda P: L0[0](P), L0[1]))

def sink(cls, kind, role, ops, name="o", **f): return spec(cls, ops, kind=kind, role=role, name=name, **f)
AS = lambda ops, cls="Assign", kind="assign", role="mutable", **f: sink(cls, kind, role, ops, **f)

# check number -> (python thunk, door spec, expect, standing refusal for 'c')
C = {}
def case(n, py, door, expect="a", standing=None): C[n] = (py, door, expect, standing)
n_ = lambda P: None
case(1, lambda P: N.Node(1, ()), None, "c", {"status": "RX_BADARG", "msg": "dtype is NULL", "door": spec("Node", dtype=None)})
case(2, lambda P: N.Node(TensorType((), "f77"), ()), None, "c", {"status": "RX_BADARG", "msg": "unknown dtype", "door": spec("Node", dtype="f77")})
case(3, lambda P: N.Node(T(), (1,)), None, "c", {"status": "RX_REFUSED", "msg": "operand handle 99 is not in this context", "door": spec("Node", ops=[99], fields="()")})
case(4, lambda P: N.Select(P["cb"], P["x"], P["x"], T("f64", (), "lu_factor")), spec("Select", ["cb", "x", "x"], tag="lu_factor"))
case(5, lambda P: N.Sink("assign", "bogus", "y", P["x"]), sink("Sink", "assign", "bogus", ["x"], name="y"))
case(6, lambda P: N.Leaf("bogus", "vec_in", "l", T()), spec("Leaf", kind="bogus", role="vec_in", name="l", quantity=None, wire_index=0, sorted=False))
case(7, lambda P: N.Leaf("vocab_read", "bogus", "l", T()), spec("Leaf", kind="vocab_read", role="bogus", name="l", quantity=None, wire_index=0, sorted=False))
case(8, lambda P: N.Const([1], T()), spec("Const", literal=[1]))
case(9, lambda P: N.Op("add", (P["x"], P["x"]), T(), literal=[1]), spec("Op", ["x", "x"], kind="add", literal=[1]))
case(10, lambda P: N.Op("quat_mul", (P["x"], P["x"]), T("f64", (4,))), spec("Op", ["x", "x"], shape=(4,), kind="quat_mul", literal=None))
case(11, lambda P: N.At(P["m"], P["ci"], T()), spec("At", ["m", "ci"], kind="at", literal=None), "b")
case(12, lambda P: N.Dispatch(P["ci"], (P["x"],), T()), spec("Dispatch", ["ci", "x"]))
case(13, lambda P: N.Dispatch(P["x"], (P["x"], P["x"]), T()), spec("Dispatch", ["x", "x", "x"]))
case(14, lambda P: N.Dispatch(P["ci"], (P["x"], P["ci"]), T()), spec("Dispatch", ["ci", "x", "ci"]))
case(15, lambda P: N.AccumWrite("o", P["x"], contract="bogus"), AS(["x"], "AccumWrite", "accum_write", "accum_out", contract="bogus", reduced=None))
RS = lambda **f: spec("ReduceSamples", ["x"], **{"op": "sum", "key": None, "count": None, "contract": "auto", "kept": (), **f})
case(16, lambda P: N.ReduceSamples("bogus", P["x"]), RS(op="bogus"))
case(17, lambda P: N.ReduceSamples("sum", P["ci"]), spec("ReduceSamples", ["ci"], dtype="i32", op="sum", key=None, count=None, contract="auto", kept=()))
case(18, lambda P: N.ReduceSamples("max", P["x"], key=P["ci"]), spec("ReduceSamples", ["x"], op="max", key=R("ci"), count=None, contract="auto", kept=()))
case(19, lambda P: N.ReduceSamples("sum", P["x"], contract="bogus"), RS(contract="bogus"))
SS = lambda ops, shape=(), **f: spec("SampleSum", ops, shape=shape, **{"kind": "sample_sum", "role": "accum_out", "name": "s", "cell": "s", "columns": 1, "op": "sum", **f})
case(20, lambda P: N.SampleSum("s", P["x"], op="bogus"), SS(["x"], op="bogus"))
case(21, lambda P: N.SampleSum("s", P["v3"]), SS(["v3"], (3,)))
case(22, lambda P: N.TapeWrite("t", P["ci"], P["cb"], (P["x"],), T()), spec("TapeWrite", ["ci", "cb", "x"], kind="tape_write", role="out", name="t", width=1))
case(23, lambda P: N.Finish("f", P["x"]), spec("Finish", ["x"], dtype="bool", kind="finish", role="terminated", name="f"))
def lc(n, **kw):
    cars = kw.pop("carries", ["car"]); ini = kw.pop("inits", ["c1"]); nx = kw.pop("nexts", ["x"])
    py, sp = loop(cars, ini, nx, **kw)
    case(n, py, sp, "b" if n == 31 else "a")
lc(24, exit_values=["c1"]); lc(25, exit_cond="x", exit_values=["c1"]); lc(26, exit_cond="cb"); lc(27, exit_cond="cb", exit_values=["ci"])
lc(28, dynamic="bogus", count="ci"); lc(29, dynamic="max"); lc(30, count="x"); lc(31, body_sinks=["x"])
lc(32, carries=[], inits=[], nexts=[]); lc(33, carries=["car", "car2"], inits=["c1"], nexts=["x", "x"]); lc(34, step=0)
lc(35, inits=["ci"]); lc(36, nexts=["ci"]); lc(37, carries=["car3"], inits=["m3"], nexts=["m3"], ttype=dict(shape=(1, 1, 1)))
case(38, lambda P: N.LoopValue(P["x"], 0), spec("LoopValue", ["x"], kind="loop_value", literal=0), "b")
case(39, lambda P: N.LoopValue(P["L0"], 5), spec("LoopValue", ["L0"], kind="loop_value", literal=5))
case(40, lambda P: N.LoopCount(P["L0"]), spec("LoopCount", ["L0"], dtype="i32", kind="loop_count", literal=None), "b")
case(41, lambda P: N.TapeRead(P["x"], 0, P["ci"]), spec("TapeRead", ["x", "ci"], kind="tape_read", literal=0, slot=0), "b")
case(42, lambda P: N.TapeRead(P["L0"], 5, P["ci"]), spec("TapeRead", ["L0", "ci"], kind="tape_read", literal=5, slot=5))

def build():
    P = {}
    for n, f, _ in POOL: P[n] = f(P)
    return P

def run(py, P):
    try: py(P)
    except Exception as e: return type(e).__name__, str(e)
    return None, None

def main(out):
    P = build()
    checks = []
    cj = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    if cj:
        def walk(x):
            if isinstance(x, list):
                if x and x[0] == "check": yield x
                else:
                    for y in x: yield from walk(y)
            elif isinstance(x, dict):
                for y in x.values(): yield from walk(y)
        order = [c for k, v in cj["checks"].items() for c in walk(v["steps"])] + [(k["class"], c) for k in cj["kinds"] if "ctor" in k for c in walk(k["ctor"]["steps"])]
        checks = [c if isinstance(c, list) else c[1] for c in order]
        kinds = [("check:" + k) for k, v in cj["checks"].items() for c in walk(v["steps"])] + [k["class"] for k in cj["kinds"] if "ctor" in k for c in walk(k["ctor"]["steps"])]
    cases = []
    for n in range(1, 43):
        py, door, expect, standing = C[n]
        exc, msg = run(py, P)
        cases.append({"group": "check", "id": n, "kind": kinds[n - 1] if checks else "", "check": checks[n - 1][1] if checks else "", "today": {"exc": exc, "msg": msg},
                      "raw": msg, "expect": expect, "bad": door, "standing": standing})
    W = lambda **k: dict({"name": "a", "role": "vec_in", "shape": (), "dtype": "f64", "tag": None}, **k)
    def q(slug="q", wires=None, pack="auto"): return (slug, tuple(wires if wires is not None else [W()]), pack)
    qcases = [("pack", q(pack="bogus")), ("slug", q(slug="")), ("no wires", q(wires=[])), ("not a wire", q(wires=[1])),
              ("role", q(wires=[W(role="bogus")])), ("duplicate", q(wires=[W(), W()]))]
    for i, (what, (slug, wires, pack)) in enumerate(qcases):
        def py(P, slug=slug, wires=wires, pack=pack): Quantity(slug, None, wires=tuple(Wire(**w) if isinstance(w, dict) else w for w in wires), pack=pack)
        exc, msg = run(py, P)
        quant = "Quantity(slug=" + kt(slug) + ", name=None, wires=" + kt(wires) + ", pack=" + kt(pack) + ")"
        cases.append({"group": "quantity", "id": i + 1, "kind": what, "check": "", "today": {"exc": exc, "msg": msg}, "raw": msg, "expect": "a",
                      "bad": spec("Leaf", kind="vocab_read", role="vec_in", name="l", quantity=None, wire_index=0, sorted=False) | {"fields": "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', 'l'), ('quantity', " + quant + "), ('wire_index', 0), ('sorted', False))"},
                      "standing": None})
    tcases = [("dtype", TensorType((), "f77"), {"status": "RX_BADARG", "msg": "unknown dtype", "door": spec("Node", dtype="f77", fields="()")}),
              ("negative extent", TensorType((-1,), "f64"), {"status": "RX_BADARG", "msg": "negative extent", "door": spec("Node", shape=(-1,), fields="()")}),
              ("shape not a tuple", TensorType(5, "f64"), {"none": "the shape is an int64 array at the door"}),
              ("shape holds a non-int", TensorType(("a",), "f64"), {"none": "the shape is an int64 array at the door"})]
    for i, (what, tt, standing) in enumerate(tcases):
        exc, msg = run(lambda P, tt=tt: N.Node(tt, ()), P)
        cases.append({"group": "ttype", "id": i + 1, "kind": what, "check": "", "today": {"exc": exc, "msg": msg}, "raw": msg, "expect": "c", "bad": None, "standing": standing})
    pool = [dict(s, name=n) for n, _, s in POOL]
    json.dump({"pool": pool, "cases": cases}, open(out, "w"), indent=1)
    bad = [c["id"] for c in cases if c["group"] != "ttype" and c["today"]["exc"] is None]
    print("cases", len(cases), "not raised by today's constructor:", bad)

if __name__ == "__main__": main(sys.argv[1])
