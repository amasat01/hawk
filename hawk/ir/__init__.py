# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The IR's old home. The definition (nodes, ops, walk, loops, types, ...) is :mod:`raptor.ir`; every name and every
``hawk.ir.<module>`` path here is the SAME object, so old imports keep working. The lowering passes that used to sit in
this package (``chain``, ``slotwise``, ``contraction``, ``lanes``, ``segment``) are in :mod:`hawk.passes` and are
reached lazily through the same old names. The importers are not rewritten yet."""

import importlib
import pkgutil
import sys

import raptor.ir as _ir
from raptor.errors import RaptorError as HawkError  # noqa: F401  (the public name, kept as a module attribute; not in __all__)

#: ``segment`` is split: its names are raptor's, its rewrite is a pass; the old path is the pass module.
_SPLIT = ("segment",)
globals().update({k: v for k, v in vars(_ir).items() if not k.startswith("__") and k not in _SPLIT})
__all__ = [*_ir.__all__, "recognize"]
for _m in pkgutil.iter_modules(_ir.__path__):
    if _m.name in _SPLIT:
        continue
    sys.modules[f"{__name__}.{_m.name}"] = importlib.import_module(f"raptor.ir.{_m.name}")
importlib.import_module("hawk.types")        # the same module as raptor.ir.types

#: Pass-like names the package used to carry: module, resolved on first use.
_MOVED = {"recognize": "contraction", "ContractionOperand": "contraction", "RecognizedContraction": "contraction",
          "split_segmented": "segment"}


def __getattr__(name):
    if name in _MOVED:
        return getattr(importlib.import_module(f"hawk.passes.{_MOVED[name]}"), name)
    raise AttributeError(name)
