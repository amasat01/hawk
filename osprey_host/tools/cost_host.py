"""cost_host.py pick <texts.jsonl> <count> <picklist>   and   cost_host.py table <counter folder>
<picklist> <out>
pick: the `count` largest host units of the recorded corpus (by recorded text lines) that the
library writes identically, as
  lines `unit<TAB>lines<TAB>test id`. Needs HOST_LIB and the binding on PYTHONPATH.
table: merges the per-worker counters of a HOST_MODE=cost run and writes the table (unit, lines, old
ms, new ms, crossing ms).
Standard library only.
"""

import glob
import json
import os
import sys


def written(record, ctx, library):
    """The recorded text when the library writes the recorded unit identically, else None."""
    import host_plugin as hp

    inputs = record["inputs"]
    probe = hp.Unit(inputs, None)
    used = (record.get("walk_opts_used") or {}).get(probe.digest)
    known = (record.get("walk_opts") or {}).get(probe.digest) or [{}]
    unit = hp.Unit(inputs, used if used is not None else known[0].get("opts"))
    want = record["output"]["value"]["v"]["text"]["s"]
    try:
        return want if hp.render(ctx, library, unit)[0] == want else None
    except ValueError:
        return None


def pick(corpus, count, out):
    """Write the picklist of the largest host units the library writes identically."""
    import _osprey

    ctx = _osprey.Ctx()
    library = ctx.backend_open(os.environ["HOST_LIB"])
    rows = []
    with open(corpus) as handle:
        for line in handle:
            if '"target":"host"' not in line or '"seam":"hawk.backends.Backend.render"' not in line:
                continue
            record = json.loads(line)
            if (
                record.get("error")
                or record.get("unmirrorable")
                or record.get("exclude")
                or not record.get("tests")
            ):
                continue
            rows.append((len(record["output"]["value"]["v"]["text"]["s"].splitlines()), record))
    rows.sort(key=lambda row: -row[0])
    chosen, seen = [], set()
    for lines, record in rows:
        if len(chosen) == count:
            break
        if record["unit"] in seen:
            continue
        try:
            ok = written(record, ctx, library)
        except Exception:
            ok = None
        if ok is not None:
            seen.add(record["unit"])
            chosen.append("%s\t%d\t%s" % (record["unit"], lines, record["tests"][0]))
    with open(out, "w") as handle:
        handle.write("\n".join(chosen) + "\n")
    print("picked", len(chosen), "units of", len(rows), "host records")


def table(folder, picklist, out):
    """Write the cost table from the merged counters of a cost run, and print it."""
    costs = {}
    for path in sorted(glob.glob(os.path.join(folder, "h*.json"))):
        with open(path) as handle:
            costs.update(json.load(handle).get("costs", {}))
    picks = [line.rstrip("\n").split("\t") for line in open(picklist) if line.strip()]
    lines = [
        "COST_HOST: render time per host unit, the old Python render against the library path",
        "NOTE: measured on a loaded machine (other jobs compile at the same time): read the "
        "ratios, not the absolute times.",
        "old = the old render call, median of 5 calls in one process after one warm-up; new = "
        "mirror + dump + load + canonical +",
        "render through the binding, measured the same way; crossing = the part of new that is "
        "not the render (mirror, dump, load,",
        "canonical: the same path with a render that refuses at once). The unit's text is already "
        "the library's, equal to the old.",
        "CAUTION: the old render is timed on the same call repeated in one process, so memos of "
        "the old build (planning, spelling tables)",
        "may make its repeats cheaper than a first call; the new path rebuilds its context each "
        "call by design.",
        "",
        "%-28s %6s %9s %9s %9s %7s" % ("unit", "lines", "old ms", "new ms", "crossing", "new/old"),
    ]
    missing = []
    for unit, count, _test in picks:
        row = costs.get(unit)
        if row is None:
            missing.append(unit)
            continue
        lines.append(
            "%-28s %6d %9.2f %9.2f %9.2f %7.2f"
            % (
                unit[:28],
                row["lines"],
                row["old_ms"],
                row["new_ms"],
                row["crossing_ms"],
                row["new_ms"] / row["old_ms"],
            )
        )
    done = [costs[u] for u, _, _ in picks if u in costs]
    if done:
        ratios = sorted(r["new_ms"] / r["old_ms"] for r in done)
        lines.append("")
        lines.append(
            "units measured %d of %d; median new/old %.2f (min %.2f, max %.2f)"
            % (len(done), len(picks), ratios[len(ratios) // 2], ratios[0], ratios[-1])
        )
    if missing:
        lines.append(
            "not measured (the picked test did not render the unit as a host call): "
            + ", ".join(missing)
        )
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if sys.argv[1] == "pick":
    pick(sys.argv[2], int(sys.argv[3]), sys.argv[4])
else:
    table(sys.argv[2], sys.argv[3], sys.argv[4])
