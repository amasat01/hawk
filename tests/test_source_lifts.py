# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel whose source is not in a file: ``python -c`` is lifted, ``exec`` through
``raptor.flow.exec_source``; anything else is refused with the way out."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import hawk
from hawk.ir import HawkError

_KERNEL = textwrap.dedent("""
    import hawk
    from hawk import Scalar
    @hawk.kernel
    def k(x: Scalar) -> Scalar:
        return x * 2.0
    print(k.returns)
""")


def test_python_dash_c_source_is_lifted():
    done = subprocess.run([sys.executable, "-c", _KERNEL], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "out"


def test_exec_source_is_traced():
    from raptor import flow

    ns = flow.exec_source("import hawk\nfrom hawk import Scalar\n" + textwrap.dedent("""
        @hawk.kernel
        def k(x: Scalar) -> Scalar:
            return x * 2.0
    """))
    assert ns["k"].returns == "out"


def test_exec_without_the_lift_names_the_way_out():
    scope = {"hawk": hawk, "Scalar": hawk.Scalar}
    with pytest.raises(HawkError) as exc:
        exec("@hawk.kernel\ndef k(x: Scalar) -> Scalar:\n    return x * 2.0\n", scope)
    message = str(exc.value)
    assert "its source is not on disk" in message
    assert ".py file" in message and "IPython" in message
    assert "raptor.flow.exec_source" in message and "linecache.cache" in message


def test_dash_c_lift_refuses_a_kernel_exec_defined_inside_it():
    code = textwrap.dedent("""
        import hawk
        from hawk import Scalar
        try:
            exec("@hawk.kernel\\ndef k(x: Scalar):\\n    return x * 2.0\\n", {"hawk": hawk, "Scalar": Scalar})
        except hawk.HawkError as exc:
            print("refused" if "not on disk" in str(exc) else exc)
    """)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.stdout.strip() == "refused", done.stderr
