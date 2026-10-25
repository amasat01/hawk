# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The forms that are gone, each refused with the line that replaces it.

``hawk.all_samples`` in every form, the masked store through it, the samples
axis as a payload axis, and its row / column / component views: one test per
form, each asserting the message names the replacement."""

from __future__ import annotations

import pytest

import hawk
from hawk import HawkError, Index, Matrix, Mutable, Scalar, Vector, kernel


def test_all_samples_is_gone_in_every_form():
    """The reduce, the gather and the by-key store each name their door."""
    with pytest.raises(HawkError, match="hawk.all_samples is gone") as got:
        hawk.all_samples
    for needle in ("hawk.sum(x)", "over=", "hawk.scatter(y)[key] += v", "u.at(i)"):
        assert needle in str(got.value)
    assert "all_samples" not in hawk.__all__ and not hasattr(hawk.trace, "all_samples")
    with pytest.raises(HawkError, match=r"all_samples\(x\)\.sum\(\) is hawk\.sum\(x\)"):
        @kernel
        def reduce(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x).sum()
    with pytest.raises(HawkError, match=r"a store by key is hawk\.scatter\(y\)\[key\] \+= v"):
        @kernel
        def by_key(x: Scalar, lane: Index, y: Mutable[Scalar]):
            hawk.all_samples(y)[lane] += x


def test_a_masked_store_is_gone_naming_where():
    with pytest.raises(HawkError, match=r"y = m\.where\(mask, v, y\)"):
        @kernel
        def masked(x: Scalar, y: Mutable[Scalar]):
            y = x
            hawk.all_samples(y)[x > 0.0] = 2.0 * x
    with pytest.raises(HawkError, match=r"a mask is not a key.*y = m\.where\(mask, v, y\)"):
        @kernel
        def masked2(x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y)[x > 0.0] += x


def test_the_samples_axis_is_gone_naming_hawk_sum():
    with pytest.raises(HawkError, match=r"axis='samples'\) is gone: the sum over every lane "
                                        r"is hawk\.sum\(x\)"):
        @kernel
        def k(v: Vector[3], t: Mutable[Scalar]):
            t = v.sum(axis="samples")


def test_the_row_views_of_every_sample_are_gone_naming_hawk_sum():
    with pytest.raises(HawkError, match=r"all_samples\(m\)\.rows\[0\]\.sum\(\) is "
                                        r"hawk\.sum\(m\.rows\[0\]\)"):
        @kernel
        def k(mm: Matrix[2, 3], t: Mutable[Vector[3]]):
            t = hawk.all_samples(mm).rows[0].sum()
