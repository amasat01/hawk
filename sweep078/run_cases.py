"""Run every case of the sweep against the installed hawk and write results_<label>.json.

For each case and target the runner tries, in order: describe (emit the source, no compiler),
build, and, when the case has inputs and the target is runnable here, run and compare with the
case's oracle. One failing case never stops the run.

Usage: python run_cases.py --label <name> [--targets host,cuda] [--only <id>,<id>] [--cases <dir>]
"""

import argparse
import copy
import importlib
import importlib.util
import json
import pathlib
import platform
import re
import sys
import tempfile
import traceback

import numpy as np

# The one place to adjust when a public call is named differently in another version. Each entry
# is a dotted path from the `hawk` package, as the documentation of that version spells it.
API = {
    "error_type": "HawkError",  # hawk's own error type: a refusal
    "build": "build",  # build(kernels, directory, targets=..., **options)
    "load": "load",  # load(directory, name) -> runnable kernel
    "run": "run",  # run(loaded, **planes) fills the Mutable planes in place
}
# What "describe" (the text and sidecar of a list of kernels, no compiler) is per version, tried in this
# order; the first that exists is used. A version with none records describe as "skipped": the
# released line's emit_source refuses kernels that run as several launches, so it is not a
# stand-in for describe and is left out on purpose.
DESCRIBE_CALLS = ["artifact.describe"]
KERNEL_NAME_ATTR = "name"  # attribute holding the name a bundle loads a kernel by
RUNNABLE_TARGETS = {"host"}  # targets whose artifacts hawk.run executes without another package
DEFAULT_BOUND = {"float64": 1e-12, "float32": 1e-5}  # relative, scaled by max(1, |expected|)

CASES_DIR = pathlib.Path(__file__).resolve().parent / "cases"


def resolve(hawk, dotted):
    """Return the attribute of `hawk` named by a dotted path, importing submodules on the way."""
    obj = hawk
    for part in dotted.split("."):
        if not hasattr(obj, part) and isinstance(obj, type(hawk)):
            importlib.import_module(f"{obj.__name__}.{part}")
        obj = getattr(obj, part)
    return obj


def find_describe(hawk):
    """The version's describe call from DESCRIBE_CALLS, or None when it has none."""
    for dotted in DESCRIBE_CALLS:
        try:
            return resolve(hawk, dotted)
        except (AttributeError, ImportError):
            continue
    return None


def first_line(exc):
    """First non-empty line of an exception text, prefixed by the exception type."""
    text = re.sub(r"^(\S+\.py:\d+|line \d+): ", "", str(exc).strip())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return f"{type(exc).__name__}: {lines[0].strip() if lines else ''}"[:300]


def load_case(path):
    """Import one case module; its own refusal or error surfaces to the caller."""
    spec = importlib.util.spec_from_file_location(f"case_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_groups(case):
    """Kernel-name groups built together, as the source's build calls group them."""
    return case.get("build_groups") or [[name] for name in case["kernels"]]


def stats(value, prefix, table):
    """Record sum and absolute maximum of every array inside a plane, a result or a container."""
    if isinstance(value, dict):
        for key, item in value.items():
            stats(item, f"{prefix}.{key}", table)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            stats(item, f"{prefix}.{index}", table)
    else:
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return
        if arr.size:
            table[prefix] = {"sum": float(arr.sum()), "absmax": float(np.abs(arr).max())}


def compare_value(got, want, tolerance):
    """Return (max scaled error, bound text, ok) of got against want."""
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        return float("inf"), "shape", False
    dtype = "float32" if got.dtype == np.float32 else "float64"
    if tolerance:
        rtol, atol = tolerance.get("rtol", 1e-5), tolerance.get("atol", 1e-8)
        bound = f"source rtol={rtol} atol={atol}"
        diff = np.abs(got - want)
        ok = bool(np.all(diff <= atol + rtol * np.abs(want)))
    else:
        rtol = DEFAULT_BOUND[dtype]
        bound = f"default bound {rtol:g} ({dtype})"
        diff = np.abs(got - want)
        ok = bool(np.all(diff <= rtol * np.maximum(1.0, np.abs(want))))
    scaled = diff / np.maximum(1.0, np.abs(want))
    return float(scaled.max()) if scaled.size else 0.0, bound, ok


def check_oracle(case, module, planes_before, planes_after, result):
    """Compare the run with the case's oracle; return (status, max_error, bound)."""
    expected = getattr(module, case["oracle"])(planes_before)
    worst, bound, ok = 0.0, "", True
    for key, want in expected.items():
        got = result if key == "result" else planes_after[key]
        pairs = zip(got, want) if isinstance(got, (list, tuple)) else [(got, want)]
        for g, w in pairs:
            err, bound, good = compare_value(g, w, case.get("tolerance"))
            worst, ok = max(worst, err), ok and good
    return ("ok" if ok else "mismatch"), worst, bound


def run_one(case, module, hawk, target, dirs):
    """Run the case once on a target; return the run part of a row."""
    out = {"run": "skipped", "max_error": None, "bound": None, "outputs": {}, "message": ""}
    if not case.get("inputs") or not case.get("run_kernel"):
        return out
    call_form = case.get("call_form", False)
    if target not in RUNNABLE_TARGETS or (not call_form and target not in dirs):
        out["message"] = "run: target not runnable here"
        return out
    planes = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
              for k, v in getattr(module, case["inputs"])().items()}
    before = copy.deepcopy(planes)
    result = None
    if call_form:
        result = getattr(module, case["run_kernel"])(**planes)
    else:
        directory, _ = dirs[target]
        named = getattr(getattr(module, case["run_kernel"], None), KERNEL_NAME_ATTR, case["run_kernel"])
        loaded = resolve(hawk, API["load"])(directory, named)
        resolve(hawk, API["run"])(loaded, **planes)
    stats(planes, "plane", out["outputs"])
    stats(result, "result", out["outputs"])
    out["run"] = "ok"
    if case.get("oracle"):
        out["run"], out["max_error"], out["bound"] = check_oracle(case, module, before, planes, result)
    return out


def sweep_case(path, hawk, targets, error_type):
    """Rows (one per target) for one case file."""
    row = {"id": path.stem, "describe": "error", "build": "skipped", "run": "skipped",
           "message": "", "max_error": None}
    try:
        module = load_case(path)
    except error_type as exc:
        return [dict(row, target=t, describe="refused", message="import: " + first_line(exc))
                for t in targets or ["host"]]
    except Exception as exc:  # noqa: BLE001 - any failure of a case is an outcome to record
        return [dict(row, target=t, message="import: " + first_line(exc)) for t in targets or ["host"]]
    case = module.CASE
    row["id"] = case["id"]
    wanted = targets or list(case["targets"])
    kernels = {name: getattr(module, name) for name in case["kernels"] if hasattr(module, name)}
    definers = case.get("definers", [])
    describe_msg = ""
    status = "ok"
    try:
        if definers:
            for name in definers:
                getattr(module, name)()
        else:
            describe = find_describe(hawk)
            if describe is None:
                status = "skipped"
            if describe:
                describe(list(kernels.values()))
    except error_type as exc:
        status, describe_msg = "refused", "describe: " + first_line(exc)
    except Exception as exc:  # noqa: BLE001 - any failure of a case is an outcome to record
        status, describe_msg = "error", "describe: " + first_line(exc)
    rows = []
    for target in wanted:
        entry = dict(row, target=target, describe=status, message=describe_msg, outputs={}, bound=None)
        if definers or not kernels:
            rows.append(entry)
            continue
        dirs = {}
        try:
            for group in build_groups(case):
                directory = pathlib.Path(tempfile.mkdtemp())
                members = [kernels[n] for n in group if n in kernels]
                resolve(hawk, API["build"])(members, directory, targets=(target,),
                                            **case.get("build_options", {}))
                dirs.setdefault(target, (directory, group))
                if case.get("run_kernel") in group or case.get("run_kernel") in [
                        getattr(k, KERNEL_NAME_ATTR, None) for k in members]:
                    dirs[target] = (directory, group)
            entry["build"] = "ok"
        except error_type as exc:
            entry["build"], entry["message"] = "refused", entry["message"] or "build: " + first_line(exc)
        except Exception as exc:  # noqa: BLE001 - any failure of a case is an outcome to record
            entry["build"], entry["message"] = "error", entry["message"] or "build: " + first_line(exc)
        if entry["build"] == "ok" or case.get("call_form"):
            try:
                part = run_one(case, module, hawk, target, dirs)
                note = part.pop("message")
                entry["message"] = entry["message"] or note
                entry.update(part)
                if part["run"] == "mismatch" and not entry["message"]:
                    entry["message"] = (f"run: differs from the oracle, max error "
                                        f"{part['max_error']:.3g} over {part['bound']}")
            except error_type as exc:
                entry.update(run="refused", message=entry["message"] or "run: " + first_line(exc))
            except Exception as exc:  # noqa: BLE001 - any failure of a case is an outcome to record
                entry.update(run="error", message=entry["message"] or "run: " + first_line(exc))
                entry["trace"] = traceback.format_exc().splitlines()[-3:]
        rows.append(entry)
    return rows


def main():
    """Parse the command line, sweep the selected cases, write the results file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="name of the version under test")
    parser.add_argument("--targets", default="", help="comma list overriding each case's targets")
    parser.add_argument("--cases", default=str(CASES_DIR), help="directory of case files")
    parser.add_argument("--only", default="", help="comma list of case ids (or id prefixes)")
    args = parser.parse_args()
    import hawk

    error_type = resolve(hawk, API["error_type"])
    targets = [t for t in args.targets.split(",") if t]
    only = [s for s in args.only.split(",") if s]
    rows = []
    for path in sorted(pathlib.Path(args.cases).glob("*.py")):
        if only and not any(path.stem.startswith(o) or o in path.stem for o in only):
            continue
        rows.extend(sweep_case(path, hawk, targets, error_type))
    result = {"label": args.label, "hawk_version": getattr(hawk, "__version__", "unknown"),
              "python": platform.python_version(), "machine": platform.platform(), "rows": rows}
    out = pathlib.Path(f"results_{args.label}.json")
    out.write_text(json.dumps(result, indent=1))
    counts = {}
    for r in rows:
        counts[(r["describe"], r["build"], r["run"])] = counts.get((r["describe"], r["build"], r["run"]), 0) + 1
    print(f"{len(rows)} rows written to {out}")
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  describe={key[0]} build={key[1]} run={key[2]}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
