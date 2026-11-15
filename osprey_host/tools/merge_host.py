"""merge_host.py <counter folder>: merges the per-worker counters of host_plugin and prints the line
   calls, identical, different, refused (by class), crossing errors, routed, not routed, not host.
Prints what it examined and exits 1 when no host call was examined, or any call was different or a
crossing error. Standard library only.
"""

import glob
import json
import os
import sys

NUMBERS = (
    "calls",
    "identical",
    "different",
    "crossing",
    "routed",
    "not_routed",
    "not_host",
    "raised",
)


def merge(folder):
    """The sum of every worker's counters in the folder, with the lists of differences and crossing
    errors.
    """
    total = {name: 0 for name in NUMBERS}
    total.update(
        {
            "refused": {},
            "differences": [],
            "crossings": [],
            "workers": 0,
            "identities": set(),
            "key_example": None,
            "modes": set(),
        }
    )
    for path in sorted(glob.glob(os.path.join(folder, "h*.json"))):
        with open(path) as handle:
            part = json.load(handle)
        total["workers"] += 1
        total["identities"].add(part.get("identity"))
        total["modes"].add(part.get("mode"))
        total["key_example"] = total["key_example"] or part.get("key_example")
        for name in NUMBERS:
            total[name] += part.get(name, 0)
        for cls, count in part["refused"].items():
            total["refused"][cls] = total["refused"].get(cls, 0) + count
        total["differences"] += part["differences"]
        total["crossings"] += part["crossings"]
    return total


def main(folder):
    """Print the merged counters of the folder; return the exit status (1 on failure)."""
    total = merge(folder)
    refused = sum(total["refused"].values())
    host = total["identical"] + total["different"] + refused + total["crossing"]
    print(
        "examined: %d workers, %d seam calls, %d host units"
        % (total["workers"], total["calls"], host)
    )
    print(
        "calls %d, identical %d, different %d, refused %d, crossing errors %d, routed %d, "
        "not routed %d, not host %d"
        % (
            total["calls"],
            total["identical"],
            total["different"],
            refused,
            total["crossing"],
            total["routed"],
            total["not_routed"],
            total["not_host"],
        )
    )
    by_class = sorted(total["refused"].items(), key=lambda item: (-item[1], item[0]))
    for cls, count in by_class[:12]:
        print("  refused %5d  %s" % (count, cls))
    for item in total["differences"][:10]:
        print("DIFFERENT", item)
    for item in total["crossings"][:10]:
        print("CROSSING", item)
    print(
        "modes %s, library identity %s"
        % (sorted(m for m in total["modes"] if m), sorted(i for i in total["identities"] if i))
    )
    if total["key_example"]:
        print("unit key example", total["key_example"])
    if host == 0:
        print("FAIL: no host unit was examined")
        return 1
    if total["different"] or total["crossing"]:
        print("FAIL: different or crossing errors above")
        return 1
    return 0


sys.exit(main(sys.argv[1]))
