"""Records the argv the old host driver builds for every combination its tables span.

This is the oracle of the host build instructions. Run it against the old tree (read only):

    PYTHONPATH=<old tree>:<old overlay> python tools/host_argv_old.py tests/data/host_argv_old.jsonl

Nothing of the old tree is imported by the product; the file written is recorded old behaviour,
like the other recordings.

The old command is `Host.compile_plan` followed by `Host.compile_hook`
(staging/ospreys/cpu/backend.py:57-90), called with the driver's own option and input objects.
What the old code reads from the machine is fed through these seams, none of which replaces an
old function:
  * the compiler (toolchain.py:host_compiler reads $HAWK_CXX): a stand-in script per (family,
    witness verdict). It answers `-dM -E` with the predefined macros of the family (the probe at
    toolchain.py:compiler_macros), answers `--version`
    (hawk/compile/toolchain.py:compiler_identity) and, when asked to build the masked-call
    witness (toolchain.py:_witness_run), writes a program that exits with the verdict, so the
    old probes run as written;
  * the include roots ($HAWK_AETHER_INCLUDE, $HAWK_EAGLE_INCLUDE, hawk/_roots.py): two empty
    folders;
  * the folder of the facts header: every host build passes the driver one include flag of the
    raptor wing's folder as the option `extra` (hawk/backends/__init__.py:115, made by
    hawk/facts.py:include_flags); the same flag class is passed here over a fixed folder name;
  * the cache folder ($HAWK_CACHE_DIR): one per stand-in compiler, so no verdict is shared;
  * the profile and the opt level: the compile options (drivers.py:CompileOptions), the
    environment variables unset.
ONE thing has no seam: the architecture is `platform.machine()` read directly
(toolchain.py:_is_x86_64, _is_arm64 and host_codegen_flags). The rows of the machine this runs on
are recorded unpatched; the rows of the other architecture are recorded with `platform.machine`
answering for it, and every such row says so in its `machine_patched` field.
"""

import itertools
import json
import os
import platform
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

#: The predefined macros of each compiler family, as `-dM -E` prints them (only those the old
#: code reads). A release stated only in part: a major alone counts as minor 0, and a minor that
#: is no number is no release (toolchain.py:gcc_maskcall_affected).
FAMILIES = {
    "gcc-12.3": ("gcc", "12.3", "g++ (GCC) 12.3.0", {"__GNUC__": 12, "__GNUC_MINOR__": 3}),
    "gcc-13.2": ("gcc", "13.2", "g++ (GCC) 13.2.0", {"__GNUC__": 13, "__GNUC_MINOR__": 2}),
    "gcc-13.4": ("gcc", "13.4", "g++ (GCC) 13.4.0", {"__GNUC__": 13, "__GNUC_MINOR__": 4}),
    "gcc-14.2": ("gcc", "14.2", "g++ (GCC) 14.2.0", {"__GNUC__": 14, "__GNUC_MINOR__": 2}),
    "gcc-14.3": ("gcc", "14.3", "g++ (GCC) 14.3.0", {"__GNUC__": 14, "__GNUC_MINOR__": 3}),
    "gcc-13-major-only": ("gcc", None, "g++ (GCC) 13", {"__GNUC__": 13}),
    "gcc-13-minor-unreadable": (
        "gcc",
        None,
        "g++ (GCC) x.y",
        {"__GNUC__": 13, "__GNUC_MINOR__": "x"},
    ),
    "clang-17": (
        "clang",
        None,
        "clang version 17.0.1",
        {"__GNUC__": 4, "__GNUC_MINOR__": 2, "__clang__": 1},
    ),
    "apple-clang-21": (
        "clang",
        None,
        "Apple clang version 21.0.0 (clang-2100.0.1)",
        {
            "__GNUC__": 4,
            "__GNUC_MINOR__": 2,
            "__clang__": 1,
            "__apple_build_version__": 21000001,
        },
    ),
}
PROFILES = (
    "",
    "native",
    "portable",
    "native-vector-math",
    "native-vector-math-nofma",
    "no-such-profile",
)
OPT_LEVELS = ("", "O0", "O1", "O2", "O3", "O4")
SCALARS = ("float64", "float32")
VERDICTS = ("fixed", "miscompile", "unrun")
WITNESS_EXIT = {"fixed": 0, "miscompile": 1, "unrun": 2}


def stand_in(folder: Path, version_line: str, macros: dict, verdict: str) -> Path:
    """Write a compiler stand-in into `folder` and return its path.

    It prints the macros and the version, and builds a witness that exits with `verdict`.
    """
    folder.mkdir(parents=True)
    defines = "".join(f"#define {k} {v}\n" for k, v in macros.items())
    (folder / "macros.txt").write_text(defines)
    (folder / "version.txt").write_text(version_line + "\nsecond line\n")
    script = folder / "c++"
    script.write_text(f"""#!/bin/sh
d=$(dirname "$0")
case "$*" in
  *-dM*) cat "$d/macros.txt"; exit 0;;
  *--version*) cat "$d/version.txt"; exit 0;;
esac
out=
while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done
[ -n "$out" ] || exit 1
printf '#!/bin/sh\\nexit {WITNESS_EXIT[verdict]}\\n' > "$out"
chmod +x "$out"
""")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def main(out_path: str) -> None:
    """Write one JSON line for every combination of the tables to `out_path`."""
    root = Path(tempfile.mkdtemp(prefix="host_argv_old_"))
    aether, eagle = root / "aether", root / "eagle"
    aether.mkdir()
    eagle.mkdir()
    os.environ.update(HAWK_AETHER_INCLUDE=str(aether), HAWK_EAGLE_INCLUDE=str(eagle))
    for name in ("HAWK_HOST_PROFILE", "HAWK_OPT_LEVEL", "CXX"):
        os.environ.pop(name, None)
    from hawk.compile.drivers import CompileOptions, _Inputs
    from hawk.facts import IncludeFlag
    from ospreys.cpu.backend import BACKEND

    real_machine = platform.machine
    native = "arm64" if real_machine().lower() in ("arm64", "aarch64") else "x86_64"
    placeholders = {
        str(aether): "{include:aether}",
        str(eagle): "{include:eagle}",
        "-I/FACTS": "-I{include:facts}",
        "/SRC": "{source}",
        "/OUT": "{output}",
        "/DEP": "{depfile}",
    }

    def places(compiler: Path) -> list:
        """The paths this script made for one stand-in compiler, and what stands for each."""
        return [
            (str(compiler), "{compiler}"),
            (str(compiler.parent / "cache"), "{cache}"),
            (str(root), "{scratch}"),
        ]

    rows = []
    combos = itertools.product(FAMILIES.items(), VERDICTS)
    for (family, (kind, release, line, macros)), verdict in combos:
        compiler = stand_in(root / f"{family}_{verdict}", line, macros, verdict)
        os.environ["HAWK_CXX"] = str(compiler)
        os.environ["HAWK_CACHE_DIR"] = str(compiler.parent / "cache")
        for machine in ("arm64", "x86_64"):
            if machine == native:
                platform.machine = real_machine
            else:
                platform.machine = lambda m=machine: m
            for profile, opt, scalar in itertools.product(PROFILES, OPT_LEVELS, SCALARS):
                row = {
                    "machine": machine,
                    "machine_patched": machine != native,
                    "compiler": kind,
                    "family": family,
                    "gcc_release": release,
                    "identity_line": line,
                    "witness": verdict,
                    "profile": profile,
                    "opt_level": opt,
                    "scalar": scalar,
                }
                options = CompileOptions(
                    target="host",
                    mode=scalar,
                    host_profile=profile,
                    opt_level=opt,
                    extra=(IncludeFlag("-I/FACTS"),),
                )
                try:
                    plan = BACKEND.compile_plan(options, _Inputs())
                    argv = []
                    job = SimpleNamespace(
                        src=Path("/SRC"), out=Path("/OUT"), dep=Path("/DEP"), served=None
                    )
                    BACKEND.compile_hook(plan, job, lambda a: argv.extend(a))
                    row["argv"] = [placeholders.get(a, a) for a in argv]
                    row["argv"][0] = "{compiler}"
                except Exception as why:  # a refusal of the old code is recorded, with its words
                    text = f"{type(why).__name__}: {why}"
                    known = [*places(compiler), *placeholders.items()]
                    for path, name in sorted(known, key=lambda pair: -len(pair[0])):
                        text = text.replace(path, name)
                    row["refused"] = text
                rows.append(row)
    platform.machine = real_machine
    # the scalar mode must not change the argv: check it here
    seen = {}
    for row in rows:
        key = tuple(row[k] for k in ("machine", "family", "profile", "opt_level"))
        got = json.dumps(row.get("argv", row.get("refused")))
        if row["scalar"] == "float32" and seen.get(key + (row["witness"],)) not in (None, got):
            raise SystemExit(f"the scalar mode changes the argv at {key}")
        seen.setdefault(key + (row["witness"],), got)
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    argvs = {json.dumps(r["argv"]) for r in rows if "argv" in r}
    refused = sum("refused" in r for r in rows)
    print(
        f"combinations {len(rows)}, with an argv {len(rows) - refused} "
        f"(distinct {len(argvs)}), refused {refused}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1])
