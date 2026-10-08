# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The HAWK artifact eagle's rank bed is pointed at.

THE DOOR, TAKEN. eagle's ``python/tests/mpi/test_rank_bed.py`` certifies
``eagle.exec.RankPartition`` against a deployed ``aether-abi/2`` ARTIFACT rather
than a linked fixture, precisely so can swap in a real HAWK emission: its
``$EAGLE_RANK_BED_ARTIFACT`` is documented as "the door, where the artifact
is a real HAWK artifact". No row over there names a kernel — each asks for "the
unit DECLARING this access class" — so what this module owes is a root of units
whose DECLARATIONS cover the four classes the bed certifies.

WHAT THE BED'S ROWS REQUIRE OF A UNIT, and why these kernels are shaped as they
are (this is narrower than ``tests/_deployable``'s deployment fixture set, which
is why the kernels live here and not there):

* exactly ONE ``per_sample`` input and exactly ONE output plane — the bed binds
  by ROLE, not by name, and asserts those counts;
* every unit carries BOTH targets: the manifest ``format`` discriminant is
  device-only (``ptx``/``cubin``/``fatbin``), so a host-only unit would not load
  at all, and the bed's device row launches the ``sample_local`` unit on GPU-0;
* one manifest per unit, because the execution axis is a MANIFEST-level
  declaration (``exec_access`` is one value) — a single manifest over four
  classes could only do so by misdeclaring three of them;
* the sidecar carries ``host_artifact``, the one additive key that contract adds
  (``eagle.host_launch`` already reads ``host_entry``; the manifest's own
  ``artifact`` names the DEVICE bytes). HAWK's sidecar writer does not emit it —
  it is the BED's convention, not the artifact schema's — so it is added here,
  where the bed's contract is being satisfied.

THE CROSS-SAMPLE READ USES ``@raw_device`` ON PURPOSE. HAWK's traced ``at()``
form reads a ``lookup``-role plane, and a ``lookup`` slot would be a
SECOND input the bed does not bind. the escape hatch is the sanctioned way to
express an access form the DAG cannot: the block declares
``access="cross_sample_read"`` explicitly (an unannotated one is refused at trace
time), and that declared class folds into the walk's inferred class, so the
manifest declares what the body actually does. The text reads the SAME shifted
index eagle's own hand-written fixture does, off the TRUE ``nSamples`` — which
is what makes the replicated-input ruling checkable: the answer is bit-identical
under any partitioning only because every rank holds the whole plane.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from hawk.artifact import build_bundle
from hawk.trace import Accum, Index, Mutable, Param, Reduce, Scalar, kernel, raw_device

#: The four classes eagle's bed certifies (``v2_artifact.REQUIRED_ACCESS_CLASSES``).
REQUIRED_ACCESS_CLASSES = ("sample_local", "cross_sample_read", "mapreduce",
                           "cross_sample_write")

#: The shifted absolute read, spelled in the two identifiers BOTH backends bind:
#: the plane's own view (``psc_<name>``, ``hawk.emit.aether.binding_name``) and
#: the loop's global index ``hawk_i`` / the triple's ``nSamples``, which the host
#: and device prologues each define. Same index expression as
#: ``eagle/tests/fixtures/host_plugin_execv2.cpp``'s ``exec_gather``.
SHIFTED_READ = ("psc_table[aether::SampleIndex::make(static_cast<std::size_t>("
                "(hawk_i * 7 + 3) % nSamples))].eval()")


@raw_device(SHIFTED_READ, access="cross_sample_read", reads=("table",))
def _shifted_read():
    """The stub spelling: names the block, carries no body of its own."""


@kernel
def rank_local(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    """``sample_local``. Sample-dependent (the bed asserts ``y[0] != y[1]``) and
    uniform-carrying, so the bed's ``_uniforms`` binding is exercised."""
    y = a * x + b


@kernel
def rank_gather(table: Scalar, y: Mutable[Scalar]):
    """``cross_sample_read``: sample ``i`` reads a sample that is NOT its own, at
    an absolute index derived from the TRUE ``nSamples``. ``table`` is passed to
    the block so the leaf is REACHED and therefore bound — the text reads the
    plane's view, and an unbound plane would be a slot the entry never takes."""
    y = _shifted_read(table)


@kernel
def rank_partial(x: Scalar, total: Reduce("sum")):
    """``mapreduce``. The body stays ELEMENTWISE — one per-sample contribution,
    never a reduction: eagle owns the COMBINE, which is what makes the
    combine ORDER a property of the partitioning and not of the plugin."""
    total.contribute(x)


@kernel
def rank_scatter(x: Scalar, lane: Index, acc: Accum[Scalar]):
    """``cross_sample_write``: refused above one partition, and refused under the
    rank structure at EVERY world size including one (F-e). It is in the artifact
    so the bed's placement row has a real body to be refused, not a stub."""
    acc.add(x, at=lane)


#: ``(kernel, unit name)`` — the unit DIRECTORIES carry no information, exactly
#: as eagle's default artifact does: a bed row must find its unit by the
#: manifest's declaration, never by a path it could have guessed.
UNITS = ((rank_local, "unit0"), (rank_gather, "unit1"),
         (rank_partial, "unit2"), (rank_scatter, "unit3"))


def build(root, *, cache_dir: str | None = None) -> pathlib.Path:
    """Build every unit under ``root`` and return it.

    ONE bundle per unit (one manifest, one execution axis), both targets, and
    the sidecar's ``host_artifact`` key added afterwards."""
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for kern, unit in UNITS:
        bundle = build_bundle([kern], root / unit, targets=("cuda", "host"),
                              cache_dir=cache_dir)
        for artifact in bundle.artifacts:
            meta = dict(artifact.sidecar)
            meta["host_artifact"] = f"{artifact.name}.so"
            artifact.sidecar_path.write_text(json.dumps(meta, indent=2) + "\n")
    declared = {json.loads((root / unit / "manifest.json").read_text())["exec_access"]
                for _k, unit in UNITS}
    missing = [a for a in REQUIRED_ACCESS_CLASSES if a not in declared]
    if missing:
        raise AssertionError(
            f"the bed artifact declares {sorted(declared)} and is missing "
            f"{missing}; eagle's rank bed certifies every class in "
            f"{list(REQUIRED_ACCESS_CLASSES)} and cannot substitute one for another"
        )
    return root


def digest(root) -> str:
    """A content digest of the built artifact — what the gate NAMES in its
    verdict, so a green run says which bytes it certified.

    Over the DEPLOYED files only (manifests, sidecars, ``.so``, ``.ptx``/``.cubin``
    — #33: the device leg may be either, by the compiled bytes): the
    emitted ``.cpp``/``.cu`` are published beside them for the source audits and
    are not what any rank loads."""
    root = pathlib.Path(root)
    stream = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".so", ".ptx", ".cubin"):
            continue
        stream.append(f"{path.relative_to(root).as_posix()}\n"
                      f"{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    if not stream:
        raise AssertionError(f"no deployed files under {root}: nothing to digest")
    return hashlib.sha256("".join(stream).encode()).hexdigest()


def units_by_access(root) -> dict:
    """``{exec_access: (unit dir, sidecar dict)}`` for every unit under ``root``.

    A tiny manifest walk rather than a call into eagle's loader: the HAWK-side
    rank row drives ``hawk._core`` directly and must be able to find its unit
    without depending on the bed's own reader."""
    root = pathlib.Path(root)
    out = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        entry = next(e for e in manifest["plugins"] if e.get("enabled", True))
        sidecar = json.loads((manifest_path.parent / entry["sidecar"]).read_text())
        access = manifest["exec_access"]
        if access in out:
            raise AssertionError(
                f"{root}: two units declare exec_access={access!r}; a row must not "
                "have to choose between them"
            )
        out[access] = (manifest_path.parent, sidecar)
    return out


if __name__ == "__main__":                     # the gate's own entry point
    import sys

    built = build(sys.argv[1], cache_dir=(sys.argv[2] if len(sys.argv) > 2 else None))
    sys.stdout.write(f"{built}\n{digest(built)}\n")
