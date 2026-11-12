# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusal type has a neutral home: ``hawk.HawkError`` is raptor's ``RaptorError``, and the toolchain refusal is raptor's."""

import ast
from pathlib import Path

import hawk
import raptor.errors
import raptor.protocols.refusal

_EXT = Path(__file__).resolve().parent.parent / "extensions"


def test_the_public_name_is_the_raptor_class():
    assert hawk.HawkError is raptor.errors.RaptorError
    from hawk.ir import HawkError
    assert HawkError is raptor.errors.RaptorError


def test_a_refusal_raised_inside_raptor_is_caught_by_the_public_name():
    from raptor.ir.nodes import Const
    from raptor.ir.types import TensorType
    try:
        Const(1.0, TensorType((), "f16"))
    except hawk.HawkError as e:
        assert type(e) is raptor.errors.RaptorError
    else:
        raise AssertionError("no refusal")


def test_the_toolchain_refusal_is_raptors():
    from hawk.compile import toolchain
    assert toolchain.ToolchainAbsent is raptor.protocols.refusal.ToolchainAbsent
    assert issubclass(raptor.protocols.refusal.ToolchainAbsent, raptor.errors.RaptorError)
    assert raptor.protocols.refusal.ToolchainAbsent("m", toolchain="x").toolchain == "x"


def test_the_extensions_do_not_import_the_toolchain_refusal_from_hawk():
    for rel in ("hawk_c/hawk_c/toolchain.py", "hawk_cuda/hawk_cuda/nvrtc.py"):
        for node in ast.walk(ast.parse((_EXT / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "hawk.compile.toolchain", rel
