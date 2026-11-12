# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The one registry: what hawk can build with and spell with, found once and loaded on demand.

An extension is a distribution with one entry point in the group :data:`GROUP`, whose value is
``<top package>:extension.json``: a static file shipped as package data, located with ``find_spec`` so that reading
a manifest imports no extension code. The first-party extensions (:data:`FIRST_PARTY`) ship in hawk's own wheel and declare their
entry points in hawk's pyproject: they are read from the metadata of the distribution ``raptor-hawk`` (:data:`OWN_DIST`), the same
way (in a source tree, ``tools/dev_dists.py`` writes that metadata; there is no search beside hawk's folder). An item's ``ref`` is imported on the first
:func:`load` of that item. The built-in backend and spellings, and the first-party extensions, are added through
the same :func:`_State.add` path and cannot be shadowed. A faulty extension is refused with a sentence
(:func:`refused`) and is an error only when one of its items is asked for by name.

Arms: ``backends`` (``{name, ref, about, device, builds, priority, options}``, ``options`` the keys it reads under ``options={"<name>.<key>": ...}``) and ``spellings`` (``{writer, dialect, ref}``,
``ref`` the module that registers that writer's spellings of that dialect). Other arms in a manifest are ignored and
listed by :func:`extensions`.

One lock guards the scan, every load and every write to the registry state."""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

GROUP = "raptor.extensions"
CONTRACT = "1"
ARMS = ("backends", "spellings")
#: ``emission_modules``: the modules whose plain tables the package emits with (they enter a unit key); ``runtime_modules``: the ones holding
#: memos and probes (never a key term). Every module with plain tables is one or the other (tests/test_emission_modules_declared.py).
_META = ("name", "version", "contract", "requires", "emission_modules", "runtime_modules", *ARMS)
#: A manifest key that is read by another loader (the declared kinds); every other key outside :data:`_META` is refused by name.
_FOREIGN = ("kinds",)
#: Backend ids and writer / dialect names are lowercase identifiers (decision E58); an extension's manifest ``name`` is a
#: distribution name: the same with ``-`` allowed.
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_DIST = re.compile(r"[a-z][a-z0-9_-]*\Z")

#: The top packages of the first-party extensions (each with ``extension.json`` beside its ``__init__``): a third party may not
#: claim them. They are found through the entry points of the distribution :data:`OWN_DIST`, like any other.
FIRST_PARTY = ("hawk_cuda", "hawk_c")
OWN_DIST = "raptor-hawk"
#: The distribution that provides a first-party backend id, for the sentences that say what to install.
KNOWN = {"c": "hawk-c", "cuda": "hawk-cuda"}
#: Backend ids only the built-in and the first-party extensions may use, installed or not.
RESERVED = ("host", *KNOWN)
#: Writer ids of the spellings arm that belong to the built-in and first-party backends.
RESERVED_WRITERS = ("aether", *RESERVED)
_OWN = ("hawk", "hawk-cuda", "hawk-c")

_LOCK = threading.RLock()


def _error(message: str):
    from ._errors import HawkError

    return HawkError(message)


def missing(backend_id: str) -> str:
    """The sentence that refuses a backend ``backend_id`` that no installed extension provides."""
    dist = KNOWN.get(backend_id)
    return (f"the {backend_id!r} backend is not installed; install the {dist} extension"
            if dist else f"no backend {backend_id!r} is installed")


def missing_device() -> str:
    """The sentence that refuses a need for a device backend when none is installed."""
    return f"no device backend is installed; install the {KNOWN['cuda']} extension"


#: hawk's own emission modules (``hawk.emit``), declared here as its manifest's data.
_EMIT_MODULES = ("dispatch", "fused", "host", "policy", "render", "rtstore", "spelling", "wrapper", "writer", "writer_aether")


#: hawk's emission tables outside ``hawk.emit``: what an emitted artifact (its text, its sidecar, its manifest) is written with.
_EMIT_ALSO = ("hawk._contracts", "hawk._plane_layout", "hawk.artifact.frame", "hawk.artifact.layout", "hawk.artifact.manifest",
              "hawk.artifact.sidecar", "hawk.passes.chain", "hawk.passes.contraction", "hawk.passes.cost", "hawk.passes.fusion",
              "hawk.passes.lstsq", "hawk.passes.policy", "hawk.passes.segment", "hawk.passes.slotwise", "hawk.passes.vectorise")


def _builtin() -> dict:
    return {"identity": "hawk", "version": "0", "contract": CONTRACT, "requires": (),
            "emission_modules": (*(f"hawk.emit.{m}" for m in _EMIT_MODULES), *_EMIT_ALSO), "runtime_modules": (),
            "backends": [{"name": "host", "ref": "hawk.backends.host:BACKEND", "about": "for the CPU", "entry": "host", "priority": 1, "options": ["profile"],
                          "policy": ["emit_form", "linalg_form", "wide_form", "contraction"],
                          "reads": ["cells.row", "form.rolled_fold_min", "host_wide", "lanes.block_run", "lanes.run_max", "linalg.max_dim", "linalg.team_max_dim", "scatter.max_chunks"],
                          "variant": {"x86_64": "x86", "arm64": "arm"}, "components": ["lowering"]}],
            "spellings": [{"writer": "aether", "dialect": "lowering", "ref": "lowering.aether"}]}


def _key(arm: str, item: dict):
    return item["name"] if arm == "backends" else (item["writer"], item["dialect"])


def _label(arm: str, key) -> str:
    return repr(key) if arm == "backends" else f"{key[0]}.{key[1]}"


def _reserved(arm: str, key) -> bool:
    fold = (lambda x: str(x).strip().casefold())
    return fold(key) in RESERVED if arm == "backends" else fold(key[0]) in RESERVED_WRITERS


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class _State:
    """What the scans found. The narrow scan reads the built-in and first-party extensions; :meth:`widen` adds the
    installed third-party ones to the same state, so nothing already answered (an item, an object imported, a
    failure) can change."""

    def __init__(self) -> None:
        self.extensions: dict = {}     # identity -> manifest
        self.claims: dict = {}         # (arm, key) -> [(identity, item, built-in)]
        self.items: dict = {}          # (arm, key) -> (identity, item, in the default set)
        self.denied: dict = {}         # (arm, key) -> [sentence], from a refused extension
        self.shadowed: dict = {}       # (arm, key) -> [sentence], from the settled claims
        self.refused: dict = {}        # identity or "<identity>/<arm>/<id>" -> sentence
        self.broken: dict = {}         # identity -> sentence, for a first-party extension that is refused
        self.loaded: dict = {}         # (arm, key) -> imported object (spellings: True)
        self.failed: dict = {}         # (arm, key) -> sentence
        self.spelled: dict = {}        # (writer, dialect, name) -> module, for the spellings a third party's module registered
        self.homes: dict = {}          # identity -> directory, for each first-party package installed at its trusted place
        self.dists: dict = {}          # identity -> the distribution that declares it (raptor-hawk)
        self.unreadable: dict = {}     # identity -> sentence, for a manifest too damaged to say what it declared
        self.skipped: set = set()      # backend ids an implicit call skipped, each warned once
        self.checked = False           # the first-party packages were checked for shadowing at the first default build
        self.named_bad: dict = {}      # name -> sentence, for a HAWK_EXTENSIONS name that is unknown or refused
        self.third = False             # the installed third-party extensions were read

    def add(self, ext: dict, refusal: str | None = None, builtin: bool = False) -> None:
        name = ext["identity"]
        if refusal:
            self.refused[name] = refusal
            if "declared" in ext and ext["declared"] is None:
                self.unreadable[name] = refusal
            for declared in ext.get("declared") or ():
                self.denied.setdefault(("backends", declared), []).append(f"extension {name!r} is refused: {refusal}")
        else:
            self.extensions[name] = ext
        for arm in ARMS:
            for item in ext.get(arm, ()):
                key = (arm, _key(arm, item))
                if refusal:
                    self.denied.setdefault(key, []).append(f"extension {name!r} is refused: {refusal}")
                else:
                    self.claims.setdefault(key, []).append((name, item, builtin))

    def settle(self, allow: set) -> None:
        self.items, self.shadowed = {}, {}
        for key, claims in self.claims.items():
            arm, k = key
            built = [c for c in claims if c[2]]
            keep = built[0] if built else claims[0] if len(claims) == 1 and not _reserved(arm, k) else None
            why = (f"{arm[:-1]} {_label(arm, k)} belongs to the built-in set: not shadowed" if built else
                   f"{arm[:-1]} {_label(arm, k)} is reserved for the built-in and first-party backends" if keep is None
                   and _reserved(arm, k) else
                   f"{arm[:-1]} {_label(arm, k)} is declared more than once, by "
                   f"{', '.join(repr(c[0]) for c in claims)}: all refused")
            for name, _, _ in [c for c in claims if c is not keep]:
                self.refused[f"{name}/{arm}/{_label(arm, k)}"] = why
                self.shadowed.setdefault(key, []).append(f"extension {name!r}: {why}")
            if keep:
                name, item, is_builtin = keep
                self.items[key] = (name, item, is_builtin or name in allow)

    def judge(self, found: list) -> dict:
        """The refusal of each of the extensions ``found`` that cannot be accepted, by identity."""
        out = {e["identity"]: e["refusal"] for e in found if e.get("refusal")}
        for e in found:
            if e["identity"] not in out and e["contract"] != CONTRACT:
                out[e["identity"]] = (f"extension {e['identity']!r} targets contract {e['contract']!r}, "
                                      f"hawk is contract {CONTRACT!r}")
        have = set(self.extensions) | {e["identity"] for e in found}
        changed = True
        while changed:
            changed = False
            for e in found:
                for need in e.get("requires", ()):
                    if e["identity"] not in out and (need not in have or need in out or need in self.refused):
                        out[e["identity"]] = (f"extension {e['identity']!r} requires {need!r}, which is "
                                              + ("refused" if need in out or need in self.refused else "not installed"))
                        changed = True
        return out

    def take(self, found: list, builtin: bool) -> None:
        refusals = self.judge(found)
        for e in found:
            self.add(e, refusals.get(e["identity"]), builtin=builtin)
            if builtin and e["identity"] in refusals:
                self.broken[e["identity"]] = refusals[e["identity"]]

    def narrow(self) -> None:
        self.add(_builtin(), builtin=True)
        found = []
        for ep, dist in () if os.environ.get("HAWK_EXTENSIONS") == "off" else _own_entries():
            identity, package = _norm(ep.name), ep.value.partition(":")[0]
            home, problem = _locate(package, dist)
            if problem:
                self.broken[identity] = problem
            else:
                self.homes[identity] = home
                self.dists[identity] = dist
                found.append(_read(identity, home))
        self.take(found, builtin=True)
        self.settle(set())

    def widen(self) -> None:
        names, found = _names(), []
        try:
            values, bad = _entries() if os.environ.get("HAWK_EXTENSIONS") != "off" else ({}, {})
            self.refused.update(bad)
            for identity, each in sorted(values.items()):
                package = min(each).partition(":")[0]
                if identity == OWN_DIST:
                    continue                                    # read by the narrow scan
                if identity in _OWN:
                    found.append({"identity": f"{identity} (third-party distribution)",
                                  "refusal": f"a distribution named {identity!r} cannot be an extension: the name is hawk's own"})
                elif len(each) > 1:
                    found.append({"identity": identity, "refusal": f"extension {identity!r} declares more than one "
                                  f"entry point: {', '.join(sorted(each))}"})
                elif package in FIRST_PARTY:
                    found.append({"identity": identity, "refusal": f"its entry point names the first-party package {package!r}"})
                else:
                    found.append(_read(identity, min(each)))
        except Exception as e:
            self.refused["<installed distributions>"] = f"the scan stopped early ({type(e).__name__}: {e})"
        self.take(found, builtin=False)
        self.settle(set(names))
        for name in names:
            if name not in self.extensions:
                self.named_bad[name] = (f"HAWK_EXTENSIONS names {name!r}, which is refused: {self.refused[name]}"
                                        if name in self.refused else
                                        f"HAWK_EXTENSIONS names {name!r}, which is not an installed extension")
                self.refused.setdefault(name, self.named_bad[name])
        self.third = True


def _names() -> list:
    """The extensions ``HAWK_EXTENSIONS`` names (``off`` alone: none; empty entries are ignored)."""
    selected = os.environ.get("HAWK_EXTENSIONS", "")
    return [] if selected == "off" else [_norm(x.strip()) for x in selected.split(",") if x.strip()]


def _entries() -> tuple:
    """``({distribution: {entry point values}}, {where: sentence})`` for the group :data:`GROUP`: one call of
    ``importlib.metadata.entry_points``; when that fails on a damaged distribution, each distribution is read alone
    and the damaged ones are refused one by one."""
    from importlib import metadata

    found: dict = {}
    try:
        for ep in metadata.entry_points(group=GROUP):
            found.setdefault(_norm(ep.dist.name if getattr(ep, "dist", None) else ep.name), set()).add(ep.value)
        return found, {}
    except Exception:
        found = {}
    bad: dict = {}
    each = iter(metadata.distributions())
    while True:
        dist = None
        try:
            dist = next(each)
            values = [ep.value for ep in dist.entry_points if ep.group == GROUP]
            if values:
                found.setdefault(_norm(dist.metadata["Name"]), set()).update(values)
        except StopIteration:
            return found, bad
        except Exception as e:
            bad[str(getattr(dist, "_path", None) or "<distribution>")] = (
                f"an installed distribution cannot be read ({type(e).__name__}: {e})")


_STATE: _State | None = None


def reset() -> None:
    """Forget what was scanned and loaded, and the spellings that extensions registered; the next use scans again
    (the path or the environment changed)."""
    global _STATE
    with _LOCK:
        if _STATE is not None:
            for key, module in _STATE.spelled.items():
                _SPELLS.pop(key, None)
                sys.modules.pop(module, None)                # imported again, it registers again
        _STATE = None
    from . import variants

    variants.reset()


def _spec(package: str):
    """The import path's answer for a top-level name, never importing it; used only to see what shadows a first-party name."""
    from importlib import util

    try:
        return util.find_spec(package)
    except Exception:
        return None


def _locate(package: str, dist) -> tuple:
    """``(home, problem)`` for the first-party ``package`` that the distribution ``dist`` (``raptor-hawk``) declares: ``home`` is
    where that distribution's metadata puts the package, when the import path resolves the name to exactly that directory;
    ``problem`` is an error sentence when it resolves elsewhere (a copy shadows it) or the declared package is not there."""
    home = Path(dist.locate_file(package)).resolve()
    if not (home / "__init__.py").is_file():
        return None, f"the distribution {OWN_DIST!r} declares the package {package!r}, which is not installed at {home}"
    spec = _spec(package)
    try:
        regular = spec is not None and bool(spec.origin) and spec.origin.endswith("__init__.py")
        where = None if spec is None else (Path(spec.origin).resolve().parent if regular else
                                           Path(spec.origin or next(iter(spec.submodule_search_locations or ("?",)))).resolve())
    except (OSError, TypeError):
        where = None
    if where is not None and where != home:
        return None, (f"the first-party package {package!r} is installed at {home}, but the import path resolves "
                      f"that name to {where}: remove or rename it")
    return home, None


def _own_entries() -> list:
    """``[(entry point, distribution)]`` of the group :data:`GROUP` that hawk's own distribution declares (one distribution
    lookup, no scan); ``[]`` when it is not installed."""
    from importlib import metadata

    try:
        dist = metadata.distribution(OWN_DIST)
    except metadata.PackageNotFoundError:
        return []
    return [(ep, dist) for ep in dist.entry_points if ep.group == GROUP]


def _typed(table: dict, fields: dict, where: str, needed: tuple = ()) -> None:
    if not isinstance(table, dict):
        raise TypeError(f"{where} is not an object")
    for field in needed:
        if field not in table:
            raise KeyError(f"{where} has no {field!r}")
    for field, kinds in fields.items():
        if field in table and type(table[field]) not in kinds:
            raise TypeError(f"{where}: {field!r} is not a {' or '.join(k.__name__ for k in kinds)}")


def _module_exists(dotted: str) -> bool:
    """Whether the dotted module name is a file or package on the import path, found without importing any of it."""
    parts = dotted.split(".")
    if not all(p.isidentifier() for p in parts):
        return False
    spec = _spec(parts[0])
    if spec is None:
        return False
    if len(parts) == 1:
        return True
    here = [Path(x) for x in (spec.submodule_search_locations or ())]
    for i, part in enumerate(parts[1:], 1):
        last = i == len(parts) - 1
        hit = [d for d in here if (d / part / "__init__.py").is_file()]
        if last and (any((d / f"{part}.py").is_file() for d in here) or hit):
            return True
        if not hit:
            return False
        here = [d / part for d in hit]
    return False


def _variant_names(entry) -> list:
    """The variant names a manifest ``variant`` entry gives (a name, or a map from isa to name), each one identifier segment."""
    from . import variants

    names = [entry] if isinstance(entry, str) else list(entry.values())
    for name in names:
        variants._segment(name)
    return names


def _check_reads(item: dict, path) -> None:
    """The variant names of a backend item are one identifier segment each; when the item states ``reads`` (the dotted keys of the
    variant its policy reads), every variant it names must hold them: refused when the manifest is loaded, not where a build reads."""
    if item.get("variant") is None:
        return
    names = _variant_names(item["variant"])
    reads = item.get("reads")
    if not reads:
        return
    from . import variants

    for name in names:
        data = variants.load(name)
        for dotted in reads:
            node = data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f"{path}: backend {item.get('name')!r} reads {dotted!r}, which its variant {name!r} does not state")
                node = node[part]


def _read(identity: str, where) -> dict:
    """The manifest of extension ``identity``: ``where`` is the package directory, or the entry point's value
    ``<top package>:<file>``. A manifest that cannot be read has a ``refusal`` sentence instead of its content."""
    import json

    fields = {"backends": {"name": (str,), "ref": (str,), "about": (str,), "device": (bool,), "builds": (bool,),
                           "priority": (int, float), "options": (list,), "policy": (list,), "variant": (str, dict), "components": (list,),
                           "entry": (str, type(None)), "process_env": (bool,), "reads": (list,)},
              "spellings": {"writer": (str,), "dialect": (str,), "ref": (str,)}}
    needed = {"backends": ("name", "ref"), "spellings": ("writer", "dialect", "ref")}
    try:
        if isinstance(where, str):
            package, colon, file = where.partition(":")
            if not (package.isidentifier() and colon and file):
                raise ValueError("the value is not '<top package>:<file>'")
            spec = _spec(package)
            if spec is None or not spec.submodule_search_locations:
                raise ModuleNotFoundError(f"no package {package!r}")
            home = Path(next(iter(spec.submodule_search_locations)))
        else:
            home, file = where, "extension.json"
        path = home / file
        if not path.is_file() or path.stat().st_size > 1 << 20:
            raise ValueError(f"{path} is not a regular file of at most 1 MiB")
        raw = json.loads(path.read_text())
        _typed(raw, {"name": (str,), "version": (str,), "contract": (str,)}, "the manifest", ("contract",))
        for key in sorted(set(raw) - set(_META) - set(_FOREIGN)):
            raise ValueError(f"{path}: unknown manifest key {key!r} (the keys are {sorted(_META)}); a misspelt key is refused, never ignored")
        if "name" in raw and not _DIST.match(raw["name"]):
            raise ValueError(f"{path}: the extension name {raw['name']!r} is not a lowercase name [a-z][a-z0-9_-]*")
        requires = raw.get("requires", [])
        if not isinstance(requires, list) or any(type(r) is not str for r in requires):
            raise TypeError("requires is not a list of names")
        for key in ("emission_modules", "runtime_modules"):
            if not isinstance(raw.get(key, []), list) or any(type(m) is not str for m in raw.get(key, [])):
                raise TypeError(f"{key} is not a list of module names")
            for module in raw.get(key, []):
                if not _module_exists(module):
                    raise ValueError(f"{path}: {key} names the module {module!r}, which does not exist")
        ext = {"identity": identity, "version": raw.get("version", ""), "contract": raw["contract"],
               "emission_modules": tuple(raw.get("emission_modules", ())), "runtime_modules": tuple(raw.get("runtime_modules", ())),
               "requires": tuple(_norm(r) for r in requires), "ignored": sorted(set(raw) - set(_META))}
        for arm in ARMS:
            if not isinstance(raw.get(arm, []), list):
                raise TypeError(f"{arm} is not a list")
            ext[arm] = [dict(item) for item in raw.get(arm, [])]
            for item in ext[arm]:
                _typed(item, fields[arm], f"an item of {arm}", needed[arm])
                for key in sorted(set(item) - set(fields[arm])):
                    raise ValueError(f"{path}: an item of {arm} has the unknown key {key!r} (the keys are {sorted(fields[arm])})")
                for key in ("name", "writer", "dialect"):
                    if key in item and not _ID.match(item[key]):
                        raise ValueError(f"{path}: the {arm[:-1]} {key} {item[key]!r} is not a lowercase identifier [a-z][a-z0-9_]*")
                for key in ("options", "policy", "components", "reads"):
                    if any(type(o) is not str for o in item.get(key, ())):
                        raise TypeError(f"{key} is not a list of names")
                if arm == "backends":
                    import dataclasses
                    from .build_policy import BuildPolicy

                    known = {f.name for f in dataclasses.fields(BuildPolicy)}
                    for key in item.get("policy", ()):
                        if key not in known:
                            raise ValueError(f"{path}: backend {item.get('name')!r} declares the policy key {key!r}, which hawk's build policy "
                                             f"does not know (the keys are {sorted(known)})")
                    for key in item.get("options", ()):
                        if not _ID.match(key):
                            raise ValueError(f"{path}: backend {item.get('name')!r} declares the option {key!r}, which is not a lowercase identifier")
                _check_reads(item, path)
        if not isinstance(raw.get("kinds", []), list):
            raise TypeError("kinds is not a list")
        return ext
    except Exception as e:
        out = {"identity": identity, "refusal": f"its manifest {where} is malformed ({type(e).__name__}: {e})"}
        try:       # what it declared, when that can still be read: a later 'unknown target' then names this refusal
            if not path.is_file() or path.stat().st_size > 1 << 20:
                raise ValueError("not a regular file")
            out["declared"] = [i["name"] for i in json.loads(path.read_text()).get("backends", []) if isinstance(i.get("name"), str)]
        except Exception:                          # noqa: BLE001 - unreadable: every unknown id will mention it
            out["declared"] = None
        return out


def _scan(third: bool = False) -> _State:
    """The registry state. Third-party extensions are read only when asked for (``third``) or named in
    ``HAWK_EXTENSIONS``: a build that names none never pays the scan of the installed distributions."""
    global _STATE
    with _LOCK:
        if _STATE is None:
            state = _State()
            state.narrow()
            _STATE = state
        if (third or _names()) and not _STATE.third:
            _STATE.widen()
        return _STATE


def refused() -> dict:
    """Every refusal, one sentence each: by extension (its distribution), or ``<extension>/<arm>/<id>`` for one item."""
    with _LOCK:
        return dict(_scan(third=True).refused)


def extensions() -> dict:
    """The extensions that were read and accepted: distribution -> {version, contract, ignored arms}."""
    with _LOCK:
        return {n: {"version": e["version"], "contract": e["contract"], "ignored": e["ignored"]}
                for n, e in _scan(third=True).extensions.items() if n != "hawk"}


def backend_ids(default: bool = False) -> tuple:
    """Backend ids without importing any, sorted by (priority, id); ``default``: only the default set
    (built-in, first-party, and the extensions named in ``HAWK_EXTENSIONS``)."""
    with _LOCK:
        items = _scan(third=not default).items
        ids = [k[1] for k, (_, _, dflt) in items.items() if k[0] == "backends" and (dflt or not default)]
        return tuple(sorted(ids, key=lambda i: (items["backends", i][1].get("priority", 0), i)))


def loadable_backend_ids(ids) -> tuple:
    """``ids`` without the backends that fail to import: an IMPLICIT call (no target was named) builds only rows that load.
    Each skipped row is warned about once, naming it and the cause; a row asked for by name is not filtered, it is refused
    with the same cause when it is loaded."""
    import warnings

    keep = []
    for backend_id in ids:
        try:
            load("backends", backend_id)
        except Exception as e:                   # noqa: BLE001 - the cause is the warning's text
            with _LOCK:
                first = backend_id not in _scan().skipped
                _scan().skipped.add(backend_id)
            if first:
                warnings.warn(f"target {backend_id!r} is skipped: {e}", RuntimeWarning, stacklevel=3)
            continue
        keep.append(backend_id)
    return tuple(keep)


def check(default: bool = False) -> None:
    """A HawkError when ``HAWK_EXTENSIONS`` names an extension that is unknown or refused; with ``default``, also when a
    first-party package is installed but refused, found only outside hawk's folders, or shadowed on the import path."""
    with _LOCK:
        s = _scan()
        if default and not s.checked:
            s.checked = True
            for identity, home in s.homes.items():
                if problem := _locate(home.name, s.dists[identity])[1]:
                    s.broken[identity] = problem
        for name, why in (s.broken.items() if default else ()):
            raise _error(f"the first-party extension {name!r} is installed but refused: {why}")
        for why in s.named_bad.values():
            raise _error(why)


def has(arm: str, key) -> bool:
    """Whether the item is available; the installed third-party extensions are read only when it is not built in."""
    with _LOCK:
        return (arm, key) in _scan().items or (arm, key) in _scan(third=True).items


def traits(backend_id: str) -> dict:
    """The manifest data of a backend id (no import): about, device, builds, priority."""
    with _LOCK:
        has("backends", backend_id)
        return dict(_scan().items["backends", backend_id][1])


def shared_emission_modules() -> tuple:
    """hawk's own emission modules (``hawk.emit.*``, its builtin manifest): every target emits with them."""
    return _builtin()["emission_modules"]


def emission_lists(backend_id: str) -> tuple:
    """``(emission modules, runtime modules)`` that the manifest of the extension providing ``backend_id`` declares (no import)."""
    with _LOCK:
        has("backends", backend_id)
        s = _scan()
        e = s.extensions[s.items["backends", backend_id][0]]
        return tuple(e.get("emission_modules", ())), tuple(e.get("runtime_modules", ()))


def version(backend_id: str) -> str:
    """The version of the distribution that provides ``backend_id`` (no import): its manifest's, hawk's own for the built-in."""
    with _LOCK:
        has("backends", backend_id)
        s = _scan()
        owner = s.items["backends", backend_id][0]
        if owner == "hawk":
            from . import __version__

            return __version__
        return s.extensions[owner]["version"]


def denial(arm: str, key) -> str | None:
    """Why an item that some extension declared is not available, or ``None``."""
    with _LOCK:
        s = _scan(third=True)
        why = s.broken.get(_norm(KNOWN.get(key, ""))) if arm == "backends" and (arm, key) not in s.denied else None
        maybe = ([f"extension {n!r} could not be read and may have declared it: {w}" for n, w in s.unreadable.items()]
                 if (arm, key) not in s.items else [])
        found = (s.denied.get((arm, key), []) + s.shadowed.get((arm, key), []) + maybe + ([why] if why else [])
                 + ([s.failed[arm, key]] if (arm, key) in s.failed else []))
        return "; ".join(found) or None


def _import(ref: str):
    import importlib

    module, _, attr = ref.partition(":")
    found = importlib.import_module(module)
    return getattr(found, attr) if attr else found


def load(arm: str, key):
    """The imported item (a spellings item: its module), imported once; a HawkError that says what to install
    or why it is refused when there is none."""
    with _LOCK:
        has(arm, key)                          # not built in: it may be an installed extension, named here
        s = _scan()
        if (arm, key) not in s.items:
            raise _error(f"{arm[:-1]} {_label(arm, key)} is refused: {denial(arm, key)}" if denial(arm, key)
                         else missing(key) if arm == "backends" else f"no {arm[:-1]} {_label(arm, key)} is installed")
        if (arm, key) in s.failed:
            raise _error(s.failed[arm, key])
        if (arm, key) not in s.loaded:
            ext, item, _ = s.items[arm, key]
            try:
                if ext in s.homes and (problem := _locate(s.homes[ext].name, s.dists[ext])[1]):
                    raise ValueError(problem)
                before = set(_SPELLS)
                obj = _import(item["ref"])
                if arm == "spellings":         # a module that publishes its spellings as data: SPELLINGS[(dialect, name)] = fn
                    for (dialect, name), fn in getattr(obj, "SPELLINGS", {}).items():
                        if (key[0], dialect, name) not in _SPELLS:
                            register_spelling(key[0], dialect, name, fn)
                if arm == "spellings" and ext not in _OWN:
                    s.spelled.update({k: item["ref"] for k in set(_SPELLS) - before})
                if arm == "backends" and (getattr(obj, "id", None) != key or obj.device != item.get("device", False)
                                          or obj.builds != item.get("builds", True)):
                    raise ValueError("the object's id, device or builds disagree with the manifest")
                if arm == "backends" and "entry" in item and obj.entry != item["entry"]:
                    raise ValueError(f"the object enters as {obj.entry!r}, the manifest says {item['entry']!r}")
                if arm == "backends":
                    obj.components = tuple(item.get("components", ()))      # the components its builds need installed
                    obj.variant = item.get("variant")           # data: a variant name, or a map from host isa to name
            except (Exception, SystemExit) as e:
                s.failed[arm, key] = (f"extension {ext!r}: {arm[:-1]} {_label(arm, key)} cannot be loaded "
                                      f"({type(e).__name__}: {e})")
                s.refused[f"{ext}/{arm}/{_label(arm, key)}"] = s.failed[arm, key]
                raise _error(s.failed[arm, key]) from e
            s.loaded[arm, key] = obj
        return s.loaded[arm, key]


_SPELLS: dict = {}


def register_spelling(writer: str, dialect: str, name: str, fn) -> None:
    """Record ``fn`` as ``writer``'s spelling of ``dialect``'s ``name``; a second definition is an error."""
    with _LOCK:
        if (writer, dialect, name) in _SPELLS:
            raise _error(f"the {writer!r} writer already spells {dialect}.{name}: a second definition is refused")
        _SPELLS[writer, dialect, name] = fn


def spelling_of(writer: str, dialect: str, name: str):
    """The writer's spelling, loading the module that registers that (writer, dialect) on the first miss; ``None`` if none."""
    fn = _SPELLS.get((writer, dialect, name))
    if fn is None:
        with _LOCK:
            key = ("spellings", (writer, dialect))
            # a first-party writer's spellings are its own: a miss is final; any other writer's are read from the installed extensions
            if key in _scan().items or (writer not in RESERVED_WRITERS and has(*key)):
                load("spellings", (writer, dialect))
            fn = _SPELLS.get((writer, dialect, name))
    return fn
