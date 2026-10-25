# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The emitted-text audit of the standalone C target (``hawk.embedded.audit``).

Rows:

* **clean** — every rendered unit the C target builds (the corpus in both
  modes, a loop with each break form, a switch dispatch, a vjp through a
  head break) audits with no hit;
* **planted** — each rule fires on a real unit mutated in one place: a
  banned token (``malloc(``, ``#pragma``), a second loop-level ``break``, a
  ``switch`` clause that falls through, an unreferenced ``static inline``
  helper, a ``//`` comment, a disallowed include, a ``= {`` brace
  initialiser, two header identifiers that agree in their first 31
  characters;
* **export** — :func:`hawk.embedded.export` refuses an audit hit by name
  (rule and line) and writes nothing;
* **vocabulary** — the banned list carries every launch-geometry token,
  imported rather than respelled.
"""

from __future__ import annotations

from types import SimpleNamespace

import _kernels as K
import pytest

from hawk.diff import vjp
from hawk.embedded import audit, render
from hawk.emit.spelling import GEOMETRY_TOKENS
from hawk.ir import HawkError, canonical

MODES = ("float64", "float32")
CORPUS = [n for n in K.CORPUS if n != "energy"]


def _derived(name, sinks):
    return SimpleNamespace(name=name, sinks=tuple(sinks), walk=canonical(sinks),
                           kind=None)


def _rules(hits):
    return {rule for _line, rule, _text in hits}


def _audit(source, c_text=None, h_text=None):
    return audit.audit(source.text if c_text is None else c_text,
                       source.header if h_text is None else h_text, source.name)


# --------------------------------------------------------------------------- #
# clean
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CORPUS)
def test_every_corpus_unit_audits_clean(name, mode):
    assert _audit(render(K.CORPUS[name], mode=mode)) == []


def _loop_and_dispatch_units():
    import test_dispatch as DI
    import test_finish_own_sample as F
    import test_loop_early_exit as E

    yield from (E.head, E.middle, E.tail, E.gated_sum, E.trajectory, E.vec_head)
    yield from (DI.disp_r0_k5_switch, DI.disp_r1_k3_switch, DI.disp_int_switch,
                DI.disp_r0_k5_pred)
    yield from (F.osc_finish, F.branch_finish)
    yield _derived("head_vjp", vjp(E.head, wrt=("x",)))
    yield _derived("drag_vjp", K.drag_vjp())


@pytest.mark.parametrize("mode", MODES)
def test_loop_dispatch_and_finish_units_audit_clean(mode):
    for k in _loop_and_dispatch_units():
        s = render(k, mode=mode)
        assert _audit(s) == [], s.name


# --------------------------------------------------------------------------- #
# planted
# --------------------------------------------------------------------------- #
def test_a_helper_calling_malloc_is_a_banned_token(monkeypatch):
    from hawk.embedded import abi

    real = abi.helper_closure

    def planted(writer):
        text, names, libm = real(writer)
        return text.replace("out.v[0] = plane[i];",
                            "out.v[0] = plane[i];\n    (void)malloc(8U);", 1), names, libm

    monkeypatch.setattr(abi, "helper_closure", planted)
    s = render(K.drag)
    assert "malloc(" in s.text
    hits = _audit(s)
    assert [(rule, "malloc" in text) for _l, rule, text in hits] == [("audit.token", True)]
    (line, _rule, _text) = hits[0]
    assert "malloc" in s.text.splitlines()[line - 1]


def test_a_pragma_is_a_banned_token():
    s = render(K.drag)
    hits = _audit(s, c_text=s.text.replace('#include <math.h>',
                                           '#pragma once\n#include <math.h>'))
    assert _rules(hits) == {"audit.token"}
    assert any("#pragma" in text for _l, _r, text in hits)


def test_a_word_that_merely_contains_a_banned_token_is_not_a_hit():
    s = render(K.drag)
    planted = s.text.replace("static void drag_run",
                             "static void drag_run_freely_unioned", 1)
    planted = planted.replace("drag_run(args", "drag_run_freely_unioned(args", 1)
    assert _audit(s, c_text=planted) == []


def test_a_banned_token_inside_a_comment_is_not_a_hit():
    """The header's own contract says ``allocation-free``: comments are prose."""
    s = render(K.drag)
    assert "allocation-free" in s.header
    assert _audit(s) == []


def test_a_cxx_namespace_is_a_banned_token():
    s = render(K.drag)
    planted = s.text.replace("sqrt(", "std::sqrt(", 1)
    assert _rules(_audit(s, c_text=planted)) == {"audit.token"}


def test_a_second_break_at_the_loops_level_is_refused():
    import test_loop_early_exit as E

    s = render(E.middle)
    assert s.text.count("break;") == 1
    planted = s.text.replace("hawk_v5_1_t = hawk_n5_1;\n",
                             "hawk_v5_1_t = hawk_n5_1;\n"
                             "            if (tb3 > psc_lim_i) {\n"
                             "                break;\n"
                             "            }\n", 1)
    assert planted.count("break;") == 2
    hits = _audit(s, c_text=planted)
    assert _rules(hits) == {"audit.break"}


def test_a_loop_without_a_break_and_one_with_a_switch_pass():
    """The sample loop has no ``break`` of its own; a switch's clause breaks
    belong to the switch."""
    import test_dispatch as DI

    s = render(DI.disp_r0_k5_switch)
    assert s.text.count("break;") == 5 and "for (" in s.text
    assert _audit(s) == []


def test_a_switch_clause_without_a_break_is_refused():
    import test_dispatch as DI

    s = render(DI.disp_r0_k5_switch)
    body = s.text.split("default: {", 1)[1]
    planted_default = body.replace("                break;\n", "", 1)
    planted = s.text.split("default: {", 1)[0] + "default: {" + planted_default
    assert planted.count("break;") == 4
    assert _rules(_audit(s, c_text=planted)) == {"audit.switch"}


def test_a_switch_without_a_default_is_refused():
    import test_dispatch as DI

    s = render(DI.disp_r0_k5_switch)
    planted = s.text.replace("default: {", "case 4: {", 1)
    assert _rules(_audit(s, c_text=planted)) == {"audit.switch"}


def test_an_unreferenced_static_inline_helper_is_refused():
    s = render(K.drag)
    planted = s.text.replace(
        "static void drag_run",
        "static inline int64_t hawk_orphan(int64_t a)\n{\n    return a;\n}\n\n"
        "static void drag_run", 1)
    hits = _audit(s, c_text=planted)
    assert _rules(hits) == {"audit.unused"}
    assert any("hawk_orphan" in text for _l, _r, text in hits)


def test_a_line_comment_is_refused_in_either_file():
    s = render(K.drag)
    in_c = s.text.replace("static void drag_run", "// the run\nstatic void drag_run", 1)
    in_h = s.header.replace("#endif\n", "#endif // guard\n", 1)
    assert _rules(_audit(s, c_text=in_c)) == {"audit.comment"}
    assert _rules(_audit(s, h_text=in_h)) == {"audit.comment"}


def test_a_brace_initialiser_is_refused():
    """``audit.init`` keeps the unit free of ``= {``, which is what lets the
    unit-wide Rule 9.3 deviation stand."""
    s = render(K.drag)
    planted = s.text.replace("    const int64_t hawk_end = base + count;\n",
                             "    const int64_t hawk_end = base + count;\n"
                             "    hawk_real_t hawk_pad[2] = {0};\n", 1)
    hits = _audit(s, c_text=planted)
    assert _rules(hits) == {"audit.init"}
    (line, _rule, text) = hits[0]
    assert "hawk_pad[2] = {0};" in planted.splitlines()[line - 1] and "drag.c" in text
    # a comparison or a brace in a comment is not an initialiser
    assert _audit(s, c_text=s.text.replace(
        "static void drag_run", "/* x = { } */\nstatic void drag_run", 1)) == []


def test_a_disallowed_include_is_refused():
    s = render(K.drag)
    planted = s.text.replace("#include <math.h>", "#include <math.h>\n#include <stdio.h>", 1)
    hits = _audit(s, c_text=planted)
    assert _rules(hits) == {"audit.include"}
    assert any("stdio.h" in text for _l, _r, text in hits)
    other = s.text.replace('#include "drag.h"', '#include "other.h"', 1)
    assert _rules(_audit(s, c_text=other)) == {"audit.include"}


def test_header_identifiers_must_differ_in_their_first_31_characters():
    """A 31-character kernel name makes ``<name>_c`` and ``<name>_args_t``
    agree in their first 31 characters (MISRA C:2012 Rule 5.1)."""
    long = "k" * 31
    s = render(K.drag, name=long)
    hits = _audit(s)
    assert _rules(hits) == {"audit.ident"}
    assert render(K.drag, name="k" * 20).name and _audit(render(K.drag, name="k" * 20)) == []


# --------------------------------------------------------------------------- #
# the export door refuses a hit
# --------------------------------------------------------------------------- #
def test_export_refuses_an_audit_hit_by_rule_and_line(monkeypatch, tmp_path):
    from hawk import embedded
    from hawk.embedded import abi

    real = abi.helper_closure

    def planted(writer):
        text, names, libm = real(writer)
        return text + "\nstatic inline int64_t hawk_orphan(int64_t a)\n{\n    return a;\n}", names, libm

    monkeypatch.setattr(abi, "helper_closure", planted)
    with pytest.raises(HawkError, match=r"'drag': the C target refuses line \d+ under "
                                        r"audit\.unused: .*hawk_orphan"):
        embedded.export(K.drag, tmp_path, check=False)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
def test_banned_tokens_carry_every_geometry_token():
    assert set(GEOMETRY_TOKENS) <= set(audit.BANNED_TOKENS)
    for token in ("malloc", "goto", "union", "volatile", "eagle", "aether::",
                  "std::", "#pragma", "tgmath", 'extern "C"'):
        assert token in audit.BANNED_TOKENS
    assert "aether" not in audit.BANNED_TOKENS


def test_the_include_whitelist():
    assert set(audit.INCLUDE_WHITELIST) == {"<math.h>", "<stdbool.h>", "<stdint.h>",
                                            "<stddef.h>", "<float.h>"}
