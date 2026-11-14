# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Live objects (through the recorder's tagged values) into the C binding: value text, dialect manifests, nodes. Standard library only."""
import json

_TAGS = ("n", "i", "f", "s", "t", "l", "fs", "d", "r", "u")


def _tag(j):
    return next(k for k in _TAGS if k in j)


def val(j, h):
    """A tagged value as value text; {"n": i} becomes the handle made for node i."""
    if j is None:
        return "None"
    if j is True or j is False:
        return "True" if j else "False"
    k = _tag(j)
    x = j[k]
    if k == "n":
        return "#%d" % h[x]
    if k in ("i", "f"):
        return x
    if k == "s":
        return repr(x)
    if k == "t":
        items = [val(i, h) for i in x]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if k in ("l", "fs"):
        return "[" + ", ".join(val(i, h) for i in x) + "]"
    if k == "d":
        return "[" + ", ".join("(%s, %s)" % (val(a, h), val(b, h)) for a, b in x) + "]"
    if k == "r":
        return "%s(%s)" % (x, ", ".join("%s=%s" % (f, val(v, h)) for f, v in sorted(j["v"].items())))
    return "Unknown(?)"


def plain(j):
    if j is None or j is True or j is False:
        return j
    k = _tag(j)
    x = j[k]
    return int(x) if k == "i" else float(x) if k == "f" else x if k == "s" else [plain(i) for i in x] if k in ("t", "l") else None


def manifests(inp):
    """One manifest (JSON text) and its first-party key text per extension, parents first, from the declarations the walk carries."""
    by_ext, seen = {}, set()
    for chain in inp.get("kinds", []):
        for item in reversed(chain):
            d = item["decl"]["v"]
            name = d["name"]["s"]
            if name in seen:
                continue
            seen.add(name)
            g = lambda k: plain(d[k]) if k in d else None
            ops = d["operands"]["t"]
            fields = []
            for f in (d["fields"]["t"] if d.get("fields") else []):
                t = f["t"]
                row = {"name": plain(t[0]), "type": plain(t[1])}
                if t[3] is not True:
                    row["default"] = plain(t[2])
                fields.append(row)
            sp = d.get("spelling")
            sp = {"dialect": plain(sp["t"][0]), "name": plain(sp["t"][1]), "args": plain(sp["t"][2])} if sp and "t" in sp else None
            row = {"name": g("name"), "family": g("family"), "refines": g("refines"),
                   "operands": {"min": plain(ops[0]), "max": "many" if ops[1] is None else plain(ops[1]), "families": plain(ops[2])},
                   "fields": fields, "result": g("result"), "identity": g("identity"), "feathers": g("feathers"), "spelling": sp,
                   "wire": g("wire"), "no_writer": g("no_writer"), "diff": g("diff")}
            ext = d["ext"]["s"]
            by_ext.setdefault(ext, (d["dialect"]["s"], []))[1].append(row)
    return [(json.dumps({"name": e, "dialect": dl, "kinds": rows}, separators=(",", ":")), '("%s",)' % e) for e, (dl, rows) in by_ext.items()]


def declare(ctx, ms):
    for m, fp in ms:
        ctx.declare_dialect(m, fp)


def mirror(ctx, inp):
    """Make every recorded node (one `make` each) in ctx; returns the node objects in entry order."""
    nodes = []
    h = []
    for nd in inp["nodes"]:
        v = nd["v"]
        tt = v["ttype"]["v"]
        shape = [int(d["i"]) for d in tt["shape"]["t"]]
        tag = tt["tag"]["s"] if tt.get("tag") else None
        ops = [nodes[o["n"]] for o in v["operands"]["t"]]
        fields = ["(%s, %s)" % (repr(k), val(x, h)) for k, x in sorted(v.items()) if k not in ("ttype", "operands")]
        ft = "(" + ", ".join(fields) + ("," if len(fields) == 1 else "") + ")"
        n = ctx.make(nd["c"], tt["dtype"]["s"], shape, tag, ops, ft)
        nodes.append(n)
        h.append(len(h))        # a handle is the position of the make: the proxy does not expose it
    return nodes
