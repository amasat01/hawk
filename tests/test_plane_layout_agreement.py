# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""hawk's plane classifier on every corpus array, every one-sample shape and the ``samples_first`` / ``samples_last`` markers.

hawk's rule is ``hawk._plane_layout.plane_layout``; the other side is ``eagle._layout.classify``. The two must give the same word
for the same array, so a plane is read the same way whichever package binds it: the two agreement rows import the older package
and are skipped without it. The one-sample rows state the class hawk gives and need hawk alone."""

from __future__ import annotations

import numpy as np
import pytest

from hawk._plane_layout import plane_layout

W, N = 3, 8


def _cls(a, head=(W,)):
    _layout = pytest.importorskip("eagle._layout")
    return _layout.classify(a.shape, [s // a.itemsize for s in a.strides], head)


def test_hawk_and_eagle_classify_every_corpus_array_identically():
    base = np.zeros((W, N))
    corpus = [base, base.T, np.zeros((N, W)), np.zeros((W, W)),
              np.zeros((W, W)).T, np.zeros((1, W)), np.zeros((W, 1)),
              np.zeros((N, 2 * W))[:, ::2], np.zeros((2 * W, N))[::2].T,
              np.zeros((2, N)), np.zeros(N)]
    for a in corpus:
        eagle_says = _cls(a)
        hawk_says = plane_layout(memoryview(a), W)
        assert eagle_says == hawk_says, (a.shape, a.strides, eagle_says, hawk_says)


#: ONE SAMPLE: ``(head, flat width, shape, expected)``. A shape that is both a
#: batch and one sample is a batch (native is never guessed away).
_SINGLE_CORPUS = [
    ((), 1, (), "single"),               # a 0-d array (a number's shape)
    ((), 1, (1,), "native"),             # a batch of one
    ((), 1, (N,), "native"),
    ((3,), 3, (3,), "single"),
    ((3,), 3, (3, 1), "native"),         # a batch of one
    ((3,), 3, (3, 3), "ambiguous"),      # a batch of w: both readings, refused
    ((3,), 3, (1, 3), "view"),           # sample-major batch of one
    ((4,), 4, (4,), "single"),           # a quaternion
    ((4,), 4, (4, 1), "native"),
    ((4,), 4, (1, 4), "view"),
    ((2, 3), 6, (2, 3), "single"),       # a matrix as (r, c)
    ((2, 3), 6, (6,), "single"),         # ... or flat (r*c,)
    ((2, 3), 6, (6, 1), "native"),
    ((3, 1), 3, (3, 1), "native"),       # Matrix[r, 1] given (r, 1): a batch
]


@pytest.mark.parametrize("head,width,shape,expected", _SINGLE_CORPUS)
def test_hawk_classifies_every_single_sample_shape_as_stated(head, width, shape, expected):
    assert plane_layout(memoryview(np.zeros(shape)), width) == expected


def test_hawk_and_eagle_agree_on_the_ambiguous_cases_and_markers():
    _layout = pytest.importorskip("eagle._layout")
    for a in (np.zeros((W, W)), np.zeros((W, W)).T, np.asfortranarray(np.zeros((W, W)))):
        st = [s // 8 for s in a.strides]
        for axis in (None, "first", "last"):
            for layout in (None, "samples_first", "samples_last"):
                assert _layout.classify(a.shape, st, (W,), axis, layout) == \
                    plane_layout(memoryview(a), W, axis, layout), (a.strides, axis, layout)
    for a in (np.zeros((W, N)), np.zeros((W, N)).T, np.zeros((N, W))):
        st = [s // 8 for s in a.strides]
        for axis in (None, "first", "last"):
            assert _layout.classify(a.shape, st, (W,), axis) == plane_layout(memoryview(a), W, axis)
