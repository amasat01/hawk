# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The CUDA backend: device source, compiled for the GPU."""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from pathlib import Path

from hawk.artifact.sidecar import LOCAL_LIMIT_B
from hawk.backends import Backend
from hawk.compile import drivers
from hawk.emit import literal_table
from hawk.emit.writer_aether import AETHER
from raptor.errors import RaptorError
from hawk.passes.cost import working_set_bytes

import hawk_cuda

from . import driver
from . import toolchain as dtc
from .emit import CUDA


_PTX_DEPOT = re.compile(r"^\s*\.local\s+[^\n;]*__local_depot\d+\[(\d+)\]", re.M)


_PTX_BODY = re.compile(r"^\.(visible\s+)?\.(entry|func)\b", re.M)


#: The published extension for each real device format
#: :func:`_sniff_device_format` can return (NVRTC/nvcc here never emit a
#: ``fatbin``, the third ``eagle.roles.MANIFEST_FORMATS`` member).
_DEVICE_EXT = {"ptx": ".ptx", "cubin": ".cubin"}


def _sniff_device_format(path: Path) -> str:
    r"""``"cubin"`` if the compiled device object at ``path`` is an ELF
    CUBIN, ``"ptx"`` otherwise — the artifact's own bytes decide, never the
    target recorded at compile time, since the compiler picks PTX or CUBIN
    from the box's own toolchain. A CUBIN opens with the ELF magic
    (``\\x7fELF``); anything else is read as PTX without inspecting further."""
    with open(path, "rb") as f:
        magic = f.read(4)
    return "cubin" if magic == b"\x7fELF" else "ptx"


def _resource_report(kernel, ptxas: dict | None, spec, scalar_type: str = "float64", arch: str = "") -> dict | None:
    """The sidecar's ptxas figures with the kernel's estimated registers beside them (:func:`hawk.passes.fusion.estimate`,
    plus a quarter of the declared working set, :func:`hawk.passes.cost.working_set_bytes`, as words) and a ``spills`` flag:
    registers spilled, or registers and frame words past the fence (:func:`hawk.passes.fusion.fence`). A flagged kernel
    warns; a flagged chain (:func:`hawk.chain`) is refused, naming its members and the boundary where to cut it."""
    if not ptxas:
        return ptxas
    from hawk.passes import fusion
    from simt.wide_class import select, tables

    working = working_set_bytes(kernel.sinks, spec.primary["form"]["rolled_fold_min"], spec.settings.form)
    real = 4 if scalar_type in ("float32", "banded") else 8
    card = fusion.row(scalar_type, spec)
    est = fusion.estimate(kernel.walk, card, scalar_type,
                          tables(select(arch, spec.primary), spec.primary).wide_min.get(real)) + working // 4
    report = {}
    for entry, row in ptxas.items():
        inside = fusion.fence(row["regs"], row["stack_B"], est)
        spills = (row["spill_store_B"] or 0) > 0 or (row["stack_B"] > 0 and not inside)
        report[entry] = {**row, "working_set_B": working, "est_regs": est, "spills": spills}
        members = getattr(kernel, "members", None)
        if not spills and (members is None or inside):
            continue
        said = (f"kernel {entry!r} uses {row['regs']} registers and a {row['stack_B']}-byte local frame "
                f"({'spills unknown' if row['spill_store_B'] is None else str(row['spill_store_B']) + ' bytes spilled'}) "
                f"against an estimate of {est} registers")
        if members is not None:
            crossing = fusion.bind(members).crossing
            widest = max(range(len(crossing)), key=crossing.__getitem__) + 1 if crossing else 0
            raise RaptorError(f"hawk: chain {kernel.name!r} of {[k.name for k in members]}: {said}; cut it where the most "
                            f"values cross, before {members[widest].name!r}: chain the members before it and the members "
                            "from it as two chains")
        warnings.warn(f"hawk: {said}", stacklevel=2)
    return report


#: An FP64 instruction in SASS (``cuobjdump -sass``), and an FP64 opcode in PTX (``add.f64``, ``cvt.f64.f32``).
_SASS_FP64 = re.compile(r"\b(DADD|DMUL|DFMA|DSETP|F2F\.F64|F2F\.F32\.F64|I2F\.F64|D2F|F2D|DMNMX)\b")


_PTX_FP64 = re.compile(r"^\s*(?:@!?%p\d+\s+)?[a-z][a-z0-9]*(?:\.[a-z0-9]+)*\.f64\b", re.M)


def _refuse_fp64(name: str, target: str, artifact, fmt: str) -> None:
    """A banded build is FP32 only: refuse a unit that computes in FP64. The device object is read for FP64
    instructions (the SASS of a CUBIN through ``cuobjdump`` when it is installed, else the opcodes of PTX)."""
    hit = None
    if fmt == "cubin":
        tool = shutil.which("cuobjdump")
        if tool is not None:
            sass = subprocess.run([tool, "-sass", str(artifact)], capture_output=True, text=True, check=False).stdout
            found = _SASS_FP64.search(sass)
            hit = found and f"the FP64 instruction {found.group(1)} in its SASS"
    else:
        found = _PTX_FP64.search(Path(artifact).read_text())
        hit = found and f"the FP64 PTX instruction {found.group(0).strip()!r}"
    if hit:
        raise RaptorError(
            f"kernel {name!r} ({target}) was built in the banded scalar mode but contains {hit}; a banded kernel "
            "computes in FP32 only. Find the double-typed value (a literal, a cast or a host helper) and make it "
            "float, or build this kernel in float64")


def _local_memory(name: str, entries: dict, lead: str) -> tuple | None:
    """``(bytes per thread, source)`` of the device unit, refused by name past the device limit.
    ``source`` is ``"ptxas"`` (the compiler's stack frame of a CUBIN, the real number) or ``"ptx"``
    (the local depots the PTX declares: ptxas can only shrink them, so it is a ceiling). ``None`` for a
    unit with no device object."""
    result, _, _, fmt = entries[lead]
    if result.ptxas:
        held, source = max(row["stack_B"] for row in result.ptxas.values()), "ptxas"
    elif fmt == "ptx":
        text = Path(result.artifact).read_text()
        cuts = [(m.start(), m.group(2)) for m in _PTX_BODY.finditer(text)] + [(len(text), "")]
        frames = [(kind, sum(int(d.group(1)) for d in _PTX_DEPOT.finditer(text[at:end])))
                  for (at, kind), (end, _) in zip(cuts, cuts[1:])]
        # an entry runs alone; the device functions it may call add to it
        held = (max((b for k, b in frames if k == "entry"), default=0)
                + sum(b for k, b in frames if k == "func"))
        source = "ptx"
    else:
        return None
    if held > LOCAL_LIMIT_B:
        raise RaptorError(
            f"kernel {name!r}: {held:,} bytes of local memory per thread ({source}), past the device limit "
            f"of {LOCAL_LIMIT_B:,}; the launch would fail. Hold fewer live factorisations or matrices per "
            "lane, build with linalg_form='team' so a team of threads works the matrix in scratch, or use "
            "a smaller n")
    return held, source


class Cuda(Backend):
    id = "cuda"
    about = "builds for the GPU"
    writer = AETHER
    wrapper = CUDA
    entry = "device"
    format = "ptx"
    artifact = ".ptx"
    source = ".cu"
    device = True
    api = hawk_cuda

    def compile(self, source, name, opts):
        if dtc.device_compiler_kind() == "nvrtc":
            return driver._compile_nvrtc_source(source, name, opts, self)
        return drivers.compile_cached(
            source, name, opts, self, compiler=dtc.device_compiler(),
            argv_flags=lambda o, target: dtc.device_flags(arch=o.arch, target=target,
                                                     extra=(*drivers.define_args(o), *o.extra), opt=o.opt_level or None),
            pick_target=lambda compiler: driver._device_target(compiler, dtc.resolve_arch(opts.arch)),
            terms=lambda flags, target: (*flags, f"device_target:{target}", dtc.device_host_compiler_identity()),
            parse=driver.parse_ptxas_info)

    def object_format(self, path):
        # the real format, read off the compiled bytes: the same build lands PTX or CUBIN by the box's toolchain
        fmt = _sniff_device_format(path)
        return fmt, _DEVICE_EXT[fmt]

    def check(self, name, smode, artifact, fmt):
        if smode.cells:
            _refuse_fp64(name, self.id, artifact, fmt)

    def facts(self, kernel, *, name, smode, block, spec, built=None, arch=""):
        out = {"block": block, "ptxas": None, "local": None}
        if built is not None:
            out["ptxas"] = _resource_report(kernel, built[0].ptxas, spec, smode.id, arch)
            out["local"] = _local_memory(name, {self.id: built}, self.id)
        return out

    def finish(self, source, lowered, options):
        return literal_table(source) if self.own_options(options).get("literal_table") else source

    def resolve_arch(self, device_arch=""):
        return dtc.resolve_arch(device_arch)

    def probe_device(self):
        from . import nvrtc

        return nvrtc.current_device_arch(), nvrtc.current_device_name()

    def process_env(self, env):
        return dtc.process_env(env)


BACKEND = Cuda()
