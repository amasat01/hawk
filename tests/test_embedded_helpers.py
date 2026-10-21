# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hawk-owned C helpers of the standalone C target."""

import importlib
import shutil
import subprocess

import pytest

from hawk.embedded import spelling as S
from hawk.emit import FLOAT32, FLOAT64
from hawk.ir import HawkError


class _Lazy:
    """A module imported on first attribute access, not at collection. Jobs that
    run only the ``ft`` rows install a released aether-dsc without ``c_core``
    (tests/sanitize/tsan_ft.sh); collecting this file must not import it there.
    A row that runs still imports it, and fails, never skips, when it is missing."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str):
        return getattr(importlib.import_module(self._name), attr)


c_core = _Lazy("aether_dsc.c_core")
H = _Lazy("hawk.embedded.helpers")

SHAPES = ((1,), (3,), (2, 2), (2, 3))
MODES = (FLOAT64, FLOAT32)


def test_load_uses_the_pitch():
    text = H.helper("load", mode=FLOAT64, shape=(3,)).text
    assert "plane[(2 * pitch) + i]" in text
    assert "out.v[0] = plane[i];" in text
    assert "out.v[1] = plane[pitch + i];" in text


def test_mat_load_flattens_row_major():
    text = H.helper("load", mode=FLOAT64, shape=(2, 3)).text
    assert "out.m[1][2] = plane[(5 * pitch) + i];" in text


def test_i64_div_defines_zero_and_min_over_minus_one():
    text = H.helper("i64_div", mode=FLOAT64).text
    assert "if (b == INT64_C(0))" in text
    assert "(a == (-INT64_C(9223372036854775807) - INT64_C(1))) && (b == INT64_C(-1))" in text
    assert text.count("return") == 1


def test_i64_from_real_saturates_and_maps_nan_to_zero():
    text = H.helper("i64_from_real", mode=FLOAT64).text
    assert "isnan(x) ? INT64_C(0)" in text
    assert "(x <= ((hawk_real_t)-9223372036854775808.0))" in text
    assert "(x >= ((hawk_real_t)9223372036854775807.0))" in text
    assert "? (-INT64_C(9223372036854775807) - INT64_C(1))" in text


def test_select_is_per_component_ternary():
    text = H.helper("select", mode=FLOAT64, shape=(2, 2)).text
    assert "out.m[1][0] = c ? a.m[1][0] : b.m[1][0];" in text
    assert "bool c" in text


def test_at_clamps_and_at_raw_does_not():
    assert "x.v[hawk_clamp_index(k, 3)]" in H.helper("at", mode=FLOAT64, shape=(3,)).text
    assert "x.v[k]" in H.helper("at_raw", mode=FLOAT64, shape=(3,)).text
    assert "hawk_clamp_index" not in H.helper("set_raw", mode=FLOAT64, shape=(3,)).text
    assert "[j / 3][j % 3]" in H.helper("at", mode=FLOAT64, shape=(2, 3)).text


def test_sign_nan_depends_on_the_core_sign():
    for mode in MODES:
        w = S.WIDTH[mode.id]
        h = H.helper("sign_nan", mode=mode)
        assert h.deps == (c_core.function("sign", width=w).name,)
        assert h.libm == ()
        assert f"aether_c_{w}_sign(x)" in h.text


def test_deps_and_libm():
    assert H.helper("load", mode=FLOAT32, shape=(3,)).deps == ("aether_c_f32_vec3_t",)
    assert H.helper("spacing", mode=FLOAT32).libm == ("nextafterf",)
    assert H.helper("spacing", mode=FLOAT64).libm == ("nextafter",)
    assert H.helper("clamp_index", mode=FLOAT64).deps == ()
    for name in ("at", "set"):
        h = H.helper(name, mode=FLOAT64, shape=(3,))
        assert h.deps == ("aether_c_f64_vec3_t", "hawk_clamp_index")
    assert H.helper("at_raw", mode=FLOAT64, shape=(3,)).deps == ("aether_c_f64_vec3_t",)


def test_header_line_and_no_line_comments():
    for name in H.HELPER_NAMES:
        shape = (3,) if name in H._SHAPED else None
        for mode in MODES:
            text = H.helper(name, mode=mode, shape=shape).text
            assert text.startswith(f"/* hawk: {name} {shape or ()} {mode.id} */\n")
            assert "//" not in text
            assert "static inline" in text


def test_unknown_name_and_missing_shape():
    with pytest.raises(HawkError):
        H.helper("nope", mode=FLOAT64)
    with pytest.raises(HawkError):
        H.helper("load", mode=FLOAT64)


def _unit(mode):
    w = S.WIDTH[mode.id]
    real = "double" if w == "f64" else "float"
    funcs = []
    for name in H.HELPER_NAMES:
        if name in H._SHAPED:
            funcs += [H.helper(name, mode=mode, shape=s) for s in SHAPES]
        else:
            funcs.append(H.helper(name, mode=mode))
    by_name = {f.name: f for f in funcs}
    core = [c_core.real_typedef(w).name]
    core += [c_core.struct(s, w).name for s in SHAPES]
    core += [c_core.function("sign", width=w).name]
    ordered, seen = [], set()

    def visit(f):
        if f.name in seen:
            return
        seen.add(f.name)
        for dep in f.deps:
            if dep.startswith("hawk_"):
                visit(by_name[dep])
            else:
                core.append(dep)
        ordered.append(f)

    for f in funcs:
        visit(f)
    parts = [
        "#include <math.h>\n#include <stdint.h>\n#include <stdbool.h>\n",
        f"typedef {real} hawk_real_t;\ntypedef int64_t hawk_int_t;\n",
        c_core.emit(core),
    ]
    parts += [f.text for f in ordered]
    return "\n".join(parts)


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.id)
def test_every_helper_compiles_as_strict_c99(tmp_path, mode):
    cc = shutil.which("cc")
    assert cc is not None, "a C compiler (cc) is required"
    src = tmp_path / "helpers.c"
    src.write_text(_unit(mode))
    proc = subprocess.run(
        [cc, "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Wconversion",
         "-Wsign-conversion", "-Wdouble-promotion", "-Wno-unused-function",
         "-Werror", "-fsyntax-only", str(src)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
