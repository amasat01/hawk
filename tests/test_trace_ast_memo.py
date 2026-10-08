# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``ast`` transform is memoised PER FUNCTION CODE OBJECT.

``hawk.trace.kernel._retrace`` used to run ``inspect.getsourcelines`` +
``ast.parse`` + ``astpass.transform`` + a final ``compile`` on EVERY
``kernel(fn)`` call, even a second trace of the identical definition — pure
waste, since the transform is a function of the source text and the resolved
sink/mutable sets alone (measured at ~0.65 ms of a 1.8 ms warm call). This
file pins the memo's three behaviours: a second trace of the SAME function
object skips the transform pipeline entirely; a DIFFERENT function that
happens to share a ``__name__`` does not share the memo; and a body the pass
REFUSES keeps refusing on every call (a refusal is never cached, so the memo
buys nothing there — the transform is always re-run and always raises)."""

from __future__ import annotations

import linecache
from unittest.mock import patch

import pytest

from hawk import Mutable, Param, Scalar, kernel
from hawk.ir import HawkError
from hawk.trace import astpass


def _defn(source: str, name: str, filename: str):
    """Compile ``source`` under a real, ``linecache``-registered filename (so
    ``kernel()``'s own ``inspect.getsourcelines`` reads it back exactly as it
    would an authored file) and hand back the UNDECORATED function ``name``.

    ``tests/_kernels.py``'s corpus is traced through ``@kernel`` at import
    time, which leaves no undecorated function object to call ``kernel()`` on
    twice — this helper is what lets a test do that."""
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {"Scalar": Scalar, "Param": Param, "Mutable": Mutable}
    exec(compile(source, filename, "exec"), scope)  # noqa: - test fixture
    return scope[name]


_SCALE_SRC = """
def scale(x: Scalar, a: Param, out: Mutable[Scalar]):
    out = a * x
"""

_AFFINE_SRC = """
def scale(x: Scalar, a: Param, b: Param, out: Mutable[Scalar]):
    out = a * x + b
"""

_REFUSED_SRC = """
def refused(x: Scalar, out: Mutable[Scalar]):
    while x > 0.0:
        out = x
"""


def test_a_second_trace_of_the_same_function_skips_the_ast_transform():
    fn = _defn(_SCALE_SRC, "scale", "<memo_same_fn>")
    with patch.object(astpass, "transform", wraps=astpass.transform) as spy:
        k1 = kernel(fn)
        assert spy.call_count == 1
        k2 = kernel(fn)
        assert spy.call_count == 1, (
            "a second kernel(fn) on the identical function object must not "
            "re-run astpass.transform")
    assert k1.walk.digest == k2.walk.digest, "the walk a memo hit produces "\
        "must digest identically to the walk the first, cold trace produced"


def test_a_different_function_with_the_same_name_does_not_share_the_memo():
    fn_a = _defn(_SCALE_SRC, "scale", "<memo_diff_fn_a>")
    fn_b = _defn(_AFFINE_SRC, "scale", "<memo_diff_fn_b>")
    with patch.object(astpass, "transform", wraps=astpass.transform) as spy:
        k_a = kernel(fn_a)
        assert spy.call_count == 1
        k_b = kernel(fn_b)
        assert spy.call_count == 2, (
            "a DIFFERENT function that happens to share fn_a's __name__ "
            "('scale') must still run its own transform, not reuse fn_a's "
            "cached one")
    assert k_a.walk.digest != k_b.walk.digest


def test_a_refused_body_refuses_on_every_call():
    fn = _defn(_REFUSED_SRC, "refused", "<memo_refused>")
    with pytest.raises(HawkError, match="While"):
        kernel(fn)
    # A refusal is never memoised (the cache is only ever written on a
    # SUCCESSFUL compile) -- a second call re-runs the same pipeline and
    # refuses again, not "hangs onto" a stale non-answer.
    with pytest.raises(HawkError, match="While"):
        kernel(fn)
