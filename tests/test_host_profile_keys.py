# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host profile and the optimisation level of a build: where they come from, what they key, and that every level builds.

The libraries state the machine for real. The key of a build covers the profile, the level and the host's own facts, so two
profiles never share a cache entry and a build asked for the default is the build asked for nothing.
"""

from __future__ import annotations

import platform

import pytest

from hawk import HawkError, Mutable, Scalar, kernel
from hawk.artifact import build_bundle, build_key, emit_source
from hawk.artifact import bundle as _bundle

HOST = ("host",)
PROFILES = ("native", "portable")


@kernel
def axpb(x: Scalar, y: Mutable[Scalar]):
    y = 2.0 * x + 1.0


def host_machine(tmp_path, options=None, opt_level=""):
    """The host machine answer of the library for the ask of ``options`` and ``opt_level``."""
    asked = _bundle._user_options(options, False, None, "", "")
    return _bundle._machines(HOST, asked, opt_level, tmp_path)["host"]


def test_the_two_exact_profiles_never_share_a_key():
    keys = {
        p: build_key([axpb], targets=HOST, options={"host.profile": p})
        for p in PROFILES
    }
    assert len(set(keys.values())) == len(keys)


def test_a_build_asked_for_the_default_profile_is_the_build_asked_for_nothing(tmp_path):
    unset = host_machine(tmp_path)
    profile = unset["stated"]["asked"]["options"]["profile"]
    assert unset["stated"]["asked"]["opt_level"] == "O3"
    nothing = build_key([axpb], targets=HOST)
    assert build_key([axpb], targets=HOST, options={"host.profile": profile}) == nothing
    assert build_key([axpb], targets=HOST, opt_level="O1") != nothing


def test_a_build_asked_for_the_default_level_is_the_build_asked_for_nothing():
    assert build_key([axpb], targets=HOST, opt_level="O3") == build_key(
        [axpb], targets=HOST
    )


def test_the_host_machine_states_the_cpu_its_key_covers(tmp_path):
    host = host_machine(tmp_path)["stated"]["host"]
    assert host["arch"] and host["vector_isa"]
    assert (
        host["l1d_cache_bytes"] > 0
        and host["l2_cache_bytes"] >= host["l1d_cache_bytes"]
    )
    assert host["cache_source"]


def test_the_profile_comes_from_the_options_then_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_HOST_PROFILE", "portable")
    asked = lambda options=None: host_machine(tmp_path, options)["stated"]["asked"][
        "options"
    ]["profile"]  # noqa: E731
    assert asked() == "portable"
    assert asked({"host.profile": "native"}) == "native"
    monkeypatch.setenv("HAWK_HOST_PROFILE", "fastest")
    with pytest.raises(HawkError, match="unknown host profile"):
        emit_source(axpb, targets=HOST)


@pytest.mark.skipif(
    platform.machine().lower() not in ("arm64", "aarch64"), reason="arm64 only"
)
def test_the_emitted_text_is_the_same_under_every_profile_on_arm64():
    texts = {
        p: emit_source(axpb, targets=HOST, options={"host.profile": p})
        .sources["host"]
        .text
        for p in (*PROFILES, "native-vector-math")
    }
    assert len(set(texts.values())) == 1


@pytest.mark.parametrize("level", ["O0", "O1", "O2", "O3"])
def test_a_host_build_compiles_at_every_level_of_the_ladder(tmp_path, level):
    built = build_bundle(
        [axpb],
        tmp_path / level,
        targets=HOST,
        cache_dir=str(tmp_path / "store"),
        opt_level=level,
    )
    assert (built.directory / "manifest.json").is_file()
