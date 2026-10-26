# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target: hawk kernels rendered as freestanding MISRA-C
instead of aether expression templates. The package holds the C writer
(``writer``), the spelling tables (``spelling``), the runtime helpers
(``helpers``), the header and ABI layout (``abi``), the verification harness
(``verify``), the rule audit (``audit``), the toolchain driver (``toolchain``)
and the target profile (``profile``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .profile import EmbeddedProfile

#: The ``generator`` stamp of every report this package writes.
GENERATOR = "hawk.embedded/1"

#: The ``<name>.embedded.json`` keys, in order: the spec's list, with
#: ``int_division_sites`` and ``int_ub_sites`` where the verify pass puts
#: them (after ``fp_equality_sites``), and ``core``/``core_version``/
#: ``deviations`` after ``helpers``.
REPORT_KEYS = (
    "kernel", "generator", "hawk_version", "walk_digest", "source_digest",
    "scalar_mode", "profile", "interface", "budgets", "indices", "fp_domain",
    "fp_equality_sites", "int_division_sites", "int_ub_sites", "linalg.status_dropped",
    "literal_specials", "libm", "helpers", "core", "core_version", "deviations",
    "unsupported_checked", "audit", "checks",
)


@dataclass(frozen=True)
class EmbeddedSource:
    """One rendered C unit: the ``.c`` text, the ``.h`` text, the bare body
    (the renderer's output, indented for the sample loop), the verify report
    the render completed, the kernel name and the scalar mode id."""

    text: str
    header: str
    body: str
    report: dict
    name: str
    mode: str


@dataclass(frozen=True)
class EmbeddedArtifact:
    """One exported C unit: the kernel name, the directory it was written to,
    the three file paths and the report as written."""

    name: str
    directory: Path
    c_path: Path
    h_path: Path
    report_path: Path
    report: dict


def render(kernel, *, name=None, mode="float64", kind=None,
           profile=None) -> EmbeddedSource:
    """Render ``kernel`` as a standalone C ``.c``/``.h`` pair.

    ``kernel`` is anything with ``sinks`` and ``walk`` (and ``name``/``kind``
    unless given): the kind resolves as ``build()`` resolves it, the verify
    pass refuses what the C target does not build and classifies every index
    site, the one body traversal renders through :class:`CWriter`, and the
    report gains what the render learned (helpers, core functions, libm,
    floating-point equality sites, special literals, static bytes) before
    the profile's caps apply. Deterministic: the same call gives the same
    bytes."""
    from ..artifact.bundle import _resolve_kind
    from ..emit import render_body
    from ..emit.wrapper import scalar_mode
    from . import abi, verify
    from .writer import CWriter
    from .profile import EmbeddedProfile

    kernel = _without_companions(kernel)
    name = name if name is not None else kernel.name
    kind = _resolve_kind(name, getattr(kernel, "kind", None), kind)
    smode = scalar_mode(getattr(mode, "id", mode))
    profile = profile if profile is not None else EmbeddedProfile()
    sinks, walk = tuple(kernel.sinks), kernel.walk
    verify.refuse_unsupported(name, kernel, kind, walk, sinks)
    report = {**_traceability(kernel, name, walk, smode, profile),
              **verify.ir_checks(name, sinks, walk, kind, profile)}
    # verify's rows are the one interface source; a branch-mix plane is K cells
    abi.size_mix_rows(report["interface"], walk)
    refuse_unclampable(name, report)
    where = verify.positions(sinks)
    by_pos = {site["node"]: site for site in report["indices"]}
    sites = {nid: by_pos[p]["status"] for nid, p in where.items() if p in by_pos}
    lengths = {nid: by_pos[p]["length"] for nid, p in where.items() if p in by_pos}
    writer = CWriter(smode, name, sites, lengths=lengths, positions=where)
    body = render_body(sinks, walk, indent="        ", kind=kind, writer=writer).text
    core_text, core_names = abi.core_closure(writer)
    helper_text, helper_names, helper_libm = abi.helper_closure(writer)
    core_libm = {m for n in core_names for m in _core_libm(writer, n)}
    report["helpers"] = sorted(helper_names)
    report["core"] = core_names
    report["core_version"] = _core_version()
    report["libm"] = sorted(set(writer.libm) | core_libm | set(helper_libm))
    # a switch renders a branch-exclusive node once per case: one site each
    report["fp_equality_sites"] = list(dict.fromkeys(
        p for p in writer.fp_eq_sites if p is not None))
    report["literal_specials"] = list(writer.literal_specials)
    static = report["budgets"]["static_bytes"]
    static["temporaries"] = sum(writer.sizeof(t) for _n, t in writer.declared)
    static["total"] = static["tapes"] + static["carries"] + static["temporaries"]
    verify.apply_caps(name, report, profile)
    header = abi.header_text(name, walk, smode, report, profile)
    text = abi.tu_text(name, walk, smode, body, writer, core_text, helper_text,
                       rows=report["interface"])
    return EmbeddedSource(text, header, body, report, name, smode.id)


def _without_companions(kernel):
    """``kernel`` as the C target lowers it: a keyed ``hawk.scatter`` under
    ``auto`` is the serial add into its target, so its companion plane and the
    commits to it are left out. An explicit ``compensated`` keeps its companion
    and is refused by the writer."""
    from ..ir.nodes import KEYED_AUTO, AccumWrite
    from ..trace.kernel import Kernel

    auto = {s.name for s in kernel.sinks
            if isinstance(s, AccumWrite) and s.policy == KEYED_AUTO}
    gone = {c for c, t in kernel.walk.companions.items() if t in auto}
    if not auto:
        return kernel
    sinks = tuple(AccumWrite(s.name, s.value, s.index, s.ttype) if s.name in auto else s
                  for s in kernel.sinks if s.name not in gone)
    planes = {n: p for n, p in kernel.planes.items() if n not in gone}
    return Kernel(kernel.name, sinks, planes, kind=kernel.kind,
                  source_digest=kernel.source_digest, returns=kernel.returns)


def _traceability(kernel, name: str, walk, smode, profile) -> dict:
    """The report's leading stamps: what was built, by which hawk, from which
    walk and which kernel source (``"unavailable"`` when the kernel holds no
    source digest: a hand-built or derived sink set), in which mode, under
    which profile. No timestamp: the same call gives the same bytes."""
    from .. import __version__

    digest = getattr(kernel, "source_digest", None)
    return {"kernel": name, "generator": GENERATOR, "hawk_version": __version__,
            "walk_digest": walk.digest,
            "source_digest": "unavailable" if digest is None else digest,
            "scalar_mode": smode.id, "profile": repr(profile)}


def check(source: EmbeddedSource, *, profile=None) -> dict:
    """Run the C compiler (``-std=c99`` and ``-std=c11``, strict flags) and
    cppcheck's MISRA addon over ``source`` in a temp dir: the report's
    ``checks`` block (:func:`hawk.embedded.toolchain.run_checks`). Any
    diagnostic or finding raises :class:`~hawk.ir.HawkError`."""
    return _checks(source, profile)


def _checks(source: EmbeddedSource, profile) -> dict:
    from . import toolchain

    return toolchain.run_checks(source.text, source.header, source.name, profile)


def export(kernel, directory, *, mode="float64", kind=None, name=None,
           profile=None, check=True) -> EmbeddedArtifact:
    """Export ``kernel`` as ``<name>.c``, ``<name>.h`` and
    ``<name>.embedded.json`` in ``directory``.

    :func:`render` (verify, render, caps), then the text audit
    (:func:`hawk.embedded.audit.audit`; any hit refuses with rule
    ``audit.<kind>`` and nothing is written), then the ``.c`` and ``.h`` are
    written, so a failing toolchain check leaves them for inspection, then
    :func:`check` when ``check`` is true, then the report, whose keys follow
    :data:`REPORT_KEYS`. Deterministic: a re-export writes the same bytes
    (the ``checks`` block names the checked files relative to ``directory``
    and drops the run time)."""
    profile = profile if profile is not None else EmbeddedProfile()
    source = render(kernel, name=name, mode=mode, kind=kind, profile=profile)
    name = source.name
    refuse_audit_hits(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c_path = directory / f"{name}.c"
    h_path = directory / f"{name}.h"
    report_path = directory / f"{name}.embedded.json"
    # a report is written only after its own C passed: never leave an older
    # clean report beside C that fails its checks
    report_path.unlink(missing_ok=True)
    c_path.write_bytes(source.text.encode())
    h_path.write_bytes(source.header.encode())
    checks = _checks(source, profile) if check else {"skipped": "not requested"}
    report = finish_report(source, checks)
    report_path.write_bytes(report_text(report).encode())
    return EmbeddedArtifact(name, directory, c_path, h_path, report_path, report)


def refuse_audit_hits(source: EmbeddedSource) -> None:
    """Run the text audit (:func:`hawk.embedded.audit.audit`) over ``source``
    and refuse any hit with rule ``audit.<kind>``: the step every door
    (:func:`export`, ``build_bundle(targets=("c",))``) runs before it writes
    or publishes anything."""
    from ..ir import HawkError
    from . import audit

    hits = audit.audit(source.text, source.header, source.name)
    if hits:
        raise HawkError("\n".join(
            f"{source.name!r}: the C target refuses line {line} under {rule}: {text}"
            for line, rule, text in hits))


def finish_report(source: EmbeddedSource, checks: dict) -> dict:
    """The ``<name>.embedded.json`` document for an audited ``source`` whose
    toolchain ``checks`` block (:func:`check`, or ``{"skipped": ...}``) is in
    hand: the render's report plus the deviations, the clean audit and the
    checks made stable (no run time, no temp directory), in
    :data:`REPORT_KEYS` order. Every door builds its report here."""
    from . import toolchain

    return _ordered({**source.report,
                     "deviations": _deviations(toolchain.DEVIATIONS, source.report),
                     "audit": {"hits": []}, "checks": _stable_checks(checks)})


def report_text(report: dict) -> str:
    """The exact text a ``<name>.embedded.json`` is written as: indented
    JSON in the report's own key order, no NaN, one trailing newline."""
    return json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n"


def _deviations(deviations, report: dict) -> list:
    """The toolchain's rule deviations, plus a ``float-equality`` note naming
    every walk position that compares reals for exact equality and a
    ``defined_where_cpp_is_ub`` note naming every Index ``+ - *``, negation
    and true division ``/``, which the C defines where the C++ build (infix
    there) is undefined."""
    out = [dict(d) for d in deviations]
    sites = report.get("fp_equality_sites") or []
    if sites:
        out.append({"rule": "float-equality", "category": "note", "sites": list(sites),
                    "reason": "the kernel compares reals for exact equality at these "
                              "walk positions as its author wrote it; the C keeps "
                              "the comparison (it mirrors the host build bit for "
                              "bit) and records each site for review"})
    ub = report.get("int_ub_sites") or []
    if ub:
        out.append({"rule": "defined_where_cpp_is_ub", "category": "note",
                    "sites": list(ub),
                    "reason": "the C defines these Index +, -, *, negations and / "
                              "through hawk_i64_* (two's-complement wrapping; 0 for "
                              "a zero divisor, MIN for MIN / -1); the C++ build "
                              "spells them infix, where signed overflow, a zero "
                              "divisor and MIN / -1 are undefined, so the two agree "
                              "wherever the C++ is defined"})
    return out


def _stable_checks(checks: dict) -> dict:
    """``checks`` with its run time dropped and the temp directory the check
    ran in replaced by ``.``: the checked files are byte-identical to the
    exported ones, so the argv reads relative to the export directory and a
    re-export writes the same report."""
    import os

    runs = checks.get("cc") or []
    tmp = os.path.dirname(runs[0]["argv"][-1]) if runs else None

    def rel(value):
        if isinstance(value, str) and tmp:
            return value.replace(tmp, ".")
        if isinstance(value, list):
            return [rel(v) for v in value]
        if isinstance(value, dict):
            return {k: rel(v) for k, v in value.items()}
        return value

    return {k: rel(v) for k, v in checks.items() if k != "seconds"}


def _ordered(report: dict) -> dict:
    """``report`` in :data:`REPORT_KEYS` order; a key the list does not know
    (a later pass's addition) follows, never dropped."""
    known = {k: report[k] for k in REPORT_KEYS if k in report}
    return {**known, **{k: v for k, v in report.items() if k not in known}}


def refuse_unclampable(name: str, report: dict) -> None:
    """Refuse a ``clamped`` index site the C target cannot clamp: the renderer
    spells a tape read's subscript itself, so only a PROVEN one may render."""
    from .verify import _refusal

    for site in report["indices"]:
        if site["kind"] == "tape_read" and site["status"] == "clamped":
            raise _refusal(
                name, site["node"], site["kind"], "index.unproven",
                f"the tape subscript is not provably in range ({site['reason']})",
                "the reader's range must lie inside the taped loop's range; the C "
                "target cannot clamp a tape subscript yet")


def _core_version() -> str:
    from aether_dsc import c_core

    return c_core.CORE_VERSION


def _core_libm(writer, name: str) -> tuple:
    """The libm names a core function calls; a helper's core dependency (the
    scalar ``sign`` under ``hawk_sign_nan``) calls none."""
    held = writer.core_functions.get(name)
    return () if held is None else held.libm


__all__ = ["EmbeddedArtifact", "EmbeddedProfile", "EmbeddedSource", "check",
           "export", "finish_report", "refuse_audit_hits", "render", "report_text"]
