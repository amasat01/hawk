"""Compare two results files of the sweep: python compare.py results_A.json results_B.json.

Prints every case and target whose outcome differs, grouped as LOST (good on A, not on B),
GAINED (the reverse), CHANGED MESSAGE (the stages' outcomes or the message differ) and VALUE
DIFFERENCE (both ran and the output sums or maxima differ by more than the bound), then counts.
"""

import json
import sys

VALUE_BOUND = 1e-12  # relative to max(1, |value on A|)


def load(path):
    """Return {(id, target): row} of one results file."""
    with open(path) as handle:
        data = json.load(handle)
    return data, {(r["id"], r["target"]): r for r in data["rows"]}


def good(row):
    """True when no stage of the row failed (a skipped stage is not a failure)."""
    return (row["describe"] in ("ok", "skipped") and row["build"] in ("ok", "skipped")
            and row["run"] in ("ok", "skipped"))


def stages(row):
    """The three stage outcomes as one text."""
    return f"{row['describe']}/{row['build']}/{row['run']}"


def value_differences(a, b):
    """Names of the outputs whose sum or absolute maximum differ beyond VALUE_BOUND."""
    names = []
    for key in sorted(set(a.get("outputs", {})) & set(b.get("outputs", {}))):
        for stat in ("sum", "absmax"):
            x, y = a["outputs"][key][stat], b["outputs"][key][stat]
            if abs(x - y) > VALUE_BOUND * max(1.0, abs(x)):
                names.append(f"{key}.{stat}: {x:.17g} vs {y:.17g}")
    return names


def main(argv):
    """Print the grouped differences; return 0."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    (meta_a, rows_a), (meta_b, rows_b) = load(argv[1]), load(argv[2])
    groups = {"LOST": [], "GAINED": [], "CHANGED MESSAGE": [], "VALUE DIFFERENCE": []}
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    same = 0
    for key in sorted(set(rows_a) & set(rows_b)):
        a, b = rows_a[key], rows_b[key]
        line = f"{key[0]} [{key[1]}]  A {stages(a)}  B {stages(b)}"
        if good(a) and not good(b):
            groups["LOST"].append(f"{line}\n      B: {b['message']}")
        elif good(b) and not good(a):
            groups["GAINED"].append(f"{line}\n      A: {a['message']}")
        elif stages(a) != stages(b) or a["message"] != b["message"]:
            groups["CHANGED MESSAGE"].append(f"{line}\n      A: {a['message']}\n      B: {b['message']}")
        elif value_differences(a, b):
            groups["VALUE DIFFERENCE"].append(line + "\n      " + "\n      ".join(value_differences(a, b)))
        else:
            same += 1
    print(f"A = {meta_a['label']} (hawk {meta_a['hawk_version']})   "
          f"B = {meta_b['label']} (hawk {meta_b['hawk_version']})")
    print(f"value bound: {VALUE_BOUND:g} relative to max(1, |A|)\n")
    for name, items in groups.items():
        print(f"== {name} ({len(items)})")
        for item in items:
            print("  " + item)
        print()
    for label, keys in (("only in A", only_a), ("only in B", only_b)):
        if keys:
            print(f"== {label} ({len(keys)}): " + ", ".join(f"{i}[{t}]" for i, t in keys) + "\n")
    print("counts: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items())
          + f", unchanged {same}, only in A {len(only_a)}, only in B {len(only_b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
