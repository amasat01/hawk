# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Backend options: ``options={"<target id>.<key>": value}`` is checked before anything is built, and the old keywords are a
deprecation window onto the same options."""

from __future__ import annotations

import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar
from hawk.artifact import build, build_bundle, emit_source


@hawk.kernel
def op_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def op_literals(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x + 0.7071 * x + 1.2345


BAD_OPTIONS = [
    ({"nope.x": 1}, "nope"),
    ({"host.zzz": 1}, r"option \"zzz\" is declared by the target host"),
    ({"bare": 1}, "must be spelled"),
    ({"host.profile": object()}, "not plain data"),
    ({"host.profile": {1, 2}}, "not plain data"),
]


@pytest.mark.parametrize(("options", "why"), BAD_OPTIONS)
def test_a_bad_option_is_refused_by_name_on_every_entry_and_nothing_is_written(
    options, why, tmp_path
):
    """An unknown target, an unknown key, a name without its target and a value that is not plain data are each refused."""
    cache = tmp_path / "cache"
    entries = (
        lambda: build_bundle(
            [op_affine],
            tmp_path / "bundle",
            targets=("host",),
            options=options,
            cache_dir=cache,
        ),
        lambda: build(
            op_affine,
            tmp_path / "single",
            targets=("host",),
            options=options,
            cache_dir=cache,
        ),
        lambda: emit_source(op_affine, targets=("host",), options=options),
    )
    for entry in entries:
        with pytest.raises(HawkError, match=why):
            entry()
    assert (
        not cache.exists()
        and not (tmp_path / "bundle").exists()
        and not (tmp_path / "single").exists()
    )


def test_an_old_keyword_gives_the_text_of_its_option_and_warns_naming_the_option():
    """``device_arch=`` and ``literal_table=`` state ``cuda.arch`` and ``cuda.literal_table``; the emitted text is the same,
    and differs from a build without the table."""
    with pytest.warns(DeprecationWarning) as seen:
        old = emit_source(
            op_literals, targets=("cuda",), device_arch="sm_90", literal_table=True
        )
    said = " ".join(str(w.message) for w in seen)
    assert (
        "options={'cuda.arch': ...}" in said
        and "options={'cuda.literal_table': ...}" in said
    )
    new = emit_source(
        op_literals,
        targets=("cuda",),
        options={"cuda.arch": "sm_90", "cuda.literal_table": True},
    )
    plain = emit_source(op_literals, targets=("cuda",), options={"cuda.arch": "sm_90"})
    assert (
        old.sources["cuda"].text
        == new.sources["cuda"].text
        != plain.sources["cuda"].text
    )


def test_an_option_given_both_by_keyword_and_in_options_is_refused_naming_the_option():
    """Two statements of one option cannot both win: the refusal says to use ``options`` only."""
    with pytest.raises(HawkError, match=r"both given: use options\['cuda\.arch'\]"):
        emit_source(
            op_affine,
            targets=("cuda",),
            device_arch="sm_90",
            options={"cuda.arch": "sm_80"},
        )
