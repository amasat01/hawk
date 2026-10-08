# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One body, two wrappers.

The body text is one source for both targets, structurally rather than by
convention: the renderer produces a body STRING and a backend supplies the
entry wrapper around it, so the row compares the STRING and never has to
decide whether a host ``for`` header or a ``params[]`` unpack counts as
prologue -- neither is in the string.

The second half: each backend's entry signature takes its parameters
in ``Walk.arg_spec`` ORDER. That is the ONE contractual ordering property; eagle's role vocabulary is itself an unordered frozenset.
"""

from __future__ import annotations

import re

from _emitted import MODES, cases

from hawk.emit import BACKENDS, CUDA, FLOAT64, render_body, render_source
from hawk.emit.aether import binding_name, param_name
from hawk.ir import Assign, Leaf, canonical
from hawk.ir import make as mk
from hawk.types import TensorType


def test_both_backends_consume_the_same_body_string():
    for name, sinks, walk in cases():
        for mode in MODES:
            host = render_source(name, sinks, walk, BACKENDS["host"], mode=mode)
            cuda = render_source(name, sinks, walk, CUDA, mode=mode)
            assert host.body == cuda.body, (
                f"the two backends did not consume the SAME body string "
                f"for {name!r} ({mode.id}).\nhost:\n{host.body}\ncuda:\n{cuda.body}"
            )
            reference = render_body(sinks, walk, indent="        ").text
            assert host.body == reference, (
                f"{name!r}: a backend's body diverged from the renderer's own output"
            )
            assert host.body in host.text and cuda.body in cuda.text, (
                f"{name!r}: the body string is not embedded verbatim in the emitted "
                "translation unit -- the wrapper rewrote it"
            )


def _signature_params(signature: str, expected: list) -> list:
    """The entry's parameter identifiers, in declaration order."""
    return [ident for ident in re.findall(r"\bp_\w+", signature)]


def test_device_signature_parameter_order_equals_arg_spec_order():
    for name, sinks, walk in cases():
        want = [param_name(role, slot) for role, slot in walk.arg_spec]
        got = _signature_params(CUDA.entry_signature(walk, name), want)
        assert got == want, (
            f"cuda's entry signature parameter order {got} != arg_spec "
            f"order {want} for {name!r}"
        )


def test_host_params_block_is_read_in_arg_spec_order():
    """The host entry takes ONE packed ``params[]``; its ORDER is the contract."""
    for name, sinks, walk in cases():
        unpack = BACKENDS["host"].unpack(walk)
        got = [int(k) for k in re.findall(r"params\[(\d+)\]", unpack)]
        assert got == list(range(len(walk.arg_spec))), (
            f"{name!r}'s host unpack reads params{got}, not arg_spec's "
            f"0..{len(walk.arg_spec) - 1}"
        )
        idents = re.findall(r"\b(?:const )?aether::View<[^\n]*> (\w+) =|"
                            r"\b(?:const )?(?:Real|Int|bool|EAGLE_ABI_INDEX_T) (\w+) =",
                            unpack)
        bound = [a or b for a, b in idents]
        assert bound == [binding_name(r, n) for r, n in walk.arg_spec], (
            f"{name!r}: the host unpack binds {bound}, not arg_spec's own order"
        )


def test_both_built_backends_are_registered_and_no_dead_arm_exists():
    assert sorted(BACKENDS) == ["cuda", "host"], (
        "exactly the two BUILT backends are registered; metal is later and a "
        "dead stub would prove nothing about the seam"
    )


def test_a_unit_without_a_random_op_renders_no_random_include():
    """The corpus this file's own rows already sweep (the SAME
    ``cases()`` `_emitted.py` builds -- 26 kernels + one derivative IR)
    must still render byte-identically to before it:
    none of them use ``hawk.math.random.{uniform,normal}``, so none may carry
    the ``aether/random`` include :func:`hawk.emit.backend.prelude` adds only
    when a body actually used one (:attr:`hawk.emit.aether.Body.
    needs_random`)."""
    for name, sinks, walk in cases():
        for mode in MODES:
            for backend in (BACKENDS["host"], CUDA):
                source = render_source(name, sinks, walk, backend, mode=mode)
                assert "aether/random" not in source.text, (
                    f"{name!r} ({backend.id}, {mode.id}) uses no random op but "
                    "its emitted TU carries the aether/random include")


def test_a_unit_with_a_random_op_does_carry_the_include():
    """The other half of the same claim: a unit that DOES use a random op
    carries EXACTLY the one extra header ((b)'s "aether first" order
    preserved -- the include lands among aether's own, before eagle's
    ``plugin/gref_layout.h``), on both backends, and the body — 
    OWN claim — stays the ONE string both wrap."""
    seed = Leaf("vocab_read", "per_sample", "seed", TensorType((), "i32"))
    counter = Leaf("vocab_read", "per_sample", "counter", TensorType((), "i32"))
    draw = mk("random_uniform", (seed, counter))
    sinks = (Assign("y", draw, TensorType((), "f64")),)
    walk = canonical(sinks)
    host = render_source("rnd", sinks, walk, BACKENDS["host"], mode=FLOAT64)
    cuda = render_source("rnd", sinks, walk, CUDA, mode=FLOAT64)
    assert host.body == cuda.body
    for source in (host, cuda):
        assert source.text.count('#include "aether/random/detail/Backend.h"') == 1
        assert source.text.index('#include "aether/random/detail/Backend.h"') <\
            source.text.index('#include "plugin/gref_layout.h"'), (
            "aether's own headers must precede eagle's plugin header")
