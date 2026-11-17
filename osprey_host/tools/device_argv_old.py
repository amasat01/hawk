"""Records the image form and argv the old device driver builds for every combination of its tables.

This is the oracle of the device build instructions. Run it against the old tree (read only):

    PYTHONPATH=<old tree>:<old tree>/staging:<old overlay> \
        python tools/device_argv_old.py tests/data/device_argv_old.jsonl

Nothing of the old tree is imported by the product; the file written is recorded old behaviour,
like the other recordings.

The old command is `Cuda.compile_plan` followed by `Cuda.compile_hook`
(staging/ospreys/cuda/backend.py:184-229), the nvcc branch, called with the driver's own option
objects. What the old code reads from the machine is fed through these seams, none of which
replaces a rule of the old code:
  * the device compiler ($HAWK_DEVICE_COMPILER, $HAWK_NVCC, toolchain.py:device_compiler): a file
    that is never run, so no compiler is needed;
  * the compiler's release (toolchain.py:nvcc_version, a probe of `nvcc --version`) and the
    driver's release (nvrtc.py:driver_cuda_version, a probe of the GPU driver; unreachable raises
    the error `_probed` turns into "unknown", nvrtc.py:358): both are replaced by the stated value
    the row records;
  * the architecture (toolchain.py:resolve_arch, a policy that reads the toolkit and the GPU):
    the architecture is the input of the row, handed on as it is;
  * the header roots (hawk/compile/drivers.py:_Inputs.include_flags): two folders, whose paths are
    replaced by the places `{include:aether}` and `{include:eagle}`;
  * the facts header folder: one include flag of a fixed folder name, as the host oracle passes it
    (hawk/backends/__init__.py:115);
  * the cache folder (the probe folder the plan sets) and the compiler identity: a folder and a
    constant, which only reach the cache key.
The old driver never passes the host compiler in the argv: nvcc reads it from its environment
(toolchain.py:process_env), so no row has one.
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

#: The releases of the device compiler the rows state, as `(major, minor)`.
COMPILERS = ((12, 6), (12, 9))
#: The driver releases the rows state; none is a driver that cannot be reached.
DRIVERS = (None, (11, 8), (12, 6), (12, 9), (13, 0))
#: The resolved architectures: two physical and two virtual (the virtual ones have no binary image).
ARCHS = ("sm_61", "sm_80", "compute_61", "compute_80")
#: The caller's defines.
DEFINES = ((), ("FOO",), ("FOO", "BAR=2"))
#: The optimisation levels asked for: none (the default), the four levels and one that is none.
OPT_LEVELS = ("", "O0", "O1", "O2", "O3", "O4")


def main(out_path: str) -> None:
    """Write one JSON line for every combination of the tables to `out_path`."""
    root = Path(tempfile.mkdtemp(prefix="device_argv_old_"))
    aether, eagle = root / "aether", root / "eagle"
    aether.mkdir()
    eagle.mkdir()
    nvcc = root / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    os.environ.update(HAWK_DEVICE_COMPILER="nvcc", HAWK_NVCC=str(nvcc))
    os.environ.pop("HAWK_OPT_LEVEL", None)
    from hawk.compile import toolchain as hawk_toolchain
    from hawk.compile.drivers import CompileOptions
    from hawk.facts import IncludeFlag
    from ospreys.cuda import nvrtc
    from ospreys.cuda import toolchain as device_toolchain
    from ospreys.cuda.backend import BACKEND
    from raptor.errors import RaptorError

    stated = {"nvcc": None, "driver": None}

    def driver_release():
        if stated["driver"] is None:
            raise RaptorError("no driver is reachable")
        return stated["driver"]

    device_toolchain.nvcc_version = lambda compiler: stated["nvcc"]
    device_toolchain.resolve_arch = lambda arch="": arch
    nvrtc.driver_cuda_version = driver_release
    inputs = SimpleNamespace(
        cache_dir=str(root),
        identity=lambda compiler: "identity",
        opt_level=hawk_toolchain.opt_level,
        include_flags=lambda: ["-I", str(aether), "-I", str(eagle)],
    )
    placeholders = {
        str(nvcc.resolve()): "{compiler}",
        str(aether): "{include:aether}",
        str(eagle): "{include:eagle}",
        "-I/FACTS": "-I{include:facts}",
        "/SRC": "{source}",
        "/OUT": "{output}",
        "/DEP": "{depfile}",
    }
    rows = []
    combos = itertools.product(COMPILERS, DRIVERS, ARCHS, DEFINES, OPT_LEVELS)
    for compiler, driver, arch, defines, opt in combos:
        stated.update(nvcc=compiler, driver=driver)
        row = {
            "compiler_release": f"{compiler[0]}.{compiler[1]}",
            "driver_release": None if driver is None else f"{driver[0]}.{driver[1]}",
            "arch": arch,
            "defines": list(defines),
            "opt_level": opt,
        }
        options = CompileOptions(
            target="cuda",
            mode="float64",
            arch=arch,
            defines=defines,
            opt_level=opt,
            extra=(IncludeFlag("-I/FACTS"),),
        )
        try:
            plan = BACKEND.compile_plan(options, inputs)
            argv = []
            job = SimpleNamespace(src=Path("/SRC"), out=Path("/OUT"), dep=Path("/DEP"))
            BACKEND.compile_hook(plan, job, lambda a: argv.extend(a) or SimpleNamespace(stderr=""))
            row["image_format"] = plan["meta"]["device_target"]
            row["argv"] = [placeholders.get(a, a) for a in argv]
        except Exception as why:  # a refusal of the old code is recorded, with its words
            row["refused"] = f"{type(why).__name__}: {why}"
        rows.append(row)
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    refused = sum("refused" in r for r in rows)
    forms = {r["image_format"] for r in rows if "image_format" in r}
    print(
        f"combinations {len(rows)}, with an argv {len(rows) - refused}, refused {refused}, "
        f"forms {sorted(forms)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1])
