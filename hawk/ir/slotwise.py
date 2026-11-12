"""Moved: :mod:`hawk.passes.slotwise` (this name stays the same module)."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("hawk.passes.slotwise")
