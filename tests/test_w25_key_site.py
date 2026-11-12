# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The compile key does not depend on where the variant's header directory is installed: it hashes the include flag's key term
(the variant name and the header content), while the compile line keeps the path."""

from __future__ import annotations

import hawk.variants as V
from hawk.compile import drivers


def test_the_key_reads_the_flag_key_term_and_the_line_keeps_the_path():
    flag = V.include_flags(V.load("cpu"))[0]
    assert str(flag).startswith("-I") and flag.key_term.startswith("variant:cpu:")
    assert drivers.key_flags(("-O2", flag)) == ("-O2", flag.key_term)
    assert str(flag) not in drivers.key_flags((flag,))


def test_a_plain_flag_is_its_own_key_term():
    assert drivers.key_flags(("-O2", "-Ifoo")) == ("-O2", "-Ifoo")
