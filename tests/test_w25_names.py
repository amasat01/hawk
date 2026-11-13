# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusal type has a neutral home: ``hawk.HawkError`` is raptor's ``RaptorError``, and the toolchain refusal is the base raptor wing's."""

import ast
from pathlib import Path

import hawk
import raptor.errors
import lowering.exchange

_EXT = Path(__file__).resolve().parent.parent / "extensions"


def test_the_public_name_is_the_raptor_class():
    assert hawk.HawkError is raptor.errors.RaptorError
    from raptor.errors import RaptorError as HawkError
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


def test_the_toolchain_refusal_is_the_base_wings():
    from hawk.compile import toolchain
    assert not hasattr(toolchain, "ToolchainAbsent")
    assert issubclass(lowering.exchange.ToolchainAbsent, raptor.errors.RaptorError)
    assert lowering.exchange.ToolchainAbsent("m", toolchain="x").toolchain == "x"


def test_the_extensions_do_not_import_the_toolchain_refusal_from_hawk():
    for rel in ("hawk_c/hawk_c/toolchain.py", "hawk_cuda/hawk_cuda/nvrtc.py"):
        for node in ast.walk(ast.parse((_EXT / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "hawk.compile.toolchain", rel
