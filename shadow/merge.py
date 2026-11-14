# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""merge.py <corpus dir>: merge w*.jsonl into walks.jsonl (unique inputs) and SUMMARY.json. Standard library only."""
import glob
import hashlib
import json
import os
import sys


def canon(j):
    return json.dumps(j, sort_keys=True, separators=(",", ":"))


def main(d):
    uniq = {}
    n = raised = unm = 0
    findings = []
    files = sorted(glob.glob(os.path.join(d, "w*.jsonl")))
    for f in files:
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                n += 1
                raised += "error" in rec
                unm += bool(rec.get("unmirrorable"))
                key = hashlib.sha256(canon(rec["inputs"]).encode()).hexdigest()
                outcome = canon({"result": rec.get("result"), "error": rec.get("error"), "preimage": rec.get("preimage")})
                u = uniq.get(key)
                if u is None:
                    uniq[key] = {"key": key, "calls": 1, "tests": [rec["test"]], "inputs": rec["inputs"],
                                 "result": rec.get("result"), "error": rec.get("error"), "preimage": rec.get("preimage"),
                                 "unknown": rec.get("unknown", []), "unmirrorable": rec.get("unmirrorable", False),
                                 "_o": outcome}
                else:
                    u["calls"] += 1
                    if rec["test"] not in u["tests"] and len(u["tests"]) < 3:
                        u["tests"].append(rec["test"])
                    if u["_o"] != outcome:
                        findings.append({"key": key, "test": rec["test"], "file": os.path.basename(f), "seq": rec.get("seq")})
    out = os.path.join(d, "walks.jsonl")
    with open(out, "w") as fh:
        for u in uniq.values():
            u.pop("_o")
            fh.write(json.dumps(u, sort_keys=True, separators=(",", ":")) + "\n")
    summ = {"files": len(files), "calls_N": n, "unique_U": len(uniq), "raised_calls": raised,
            "raised_unique": sum(1 for u in uniq.values() if u["error"]), "unmirrorable_calls": unm,
            "unmirrorable_unique": sum(1 for u in uniq.values() if u["unmirrorable"]),
            "equal_inputs_equal_results": not findings, "findings": findings[:20], "findings_count": len(findings)}
    with open(os.path.join(d, "SUMMARY.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "findings"}))


main(sys.argv[1])
