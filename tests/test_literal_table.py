# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``literal_table=`` build option: the device source reads its ``double``
literals from one ``__constant__`` table, the same bits as the immediates.

* **off is the default** and changes nothing: the device source of a build
  without the option is the source the renderer emits;
* **on** moves every distinct finite literal into the table, once, and a text
  round trip (each table slot written back in place) gives the off source;
* **float32** and the host build are not touched;
* **the device result** is bit-identical on and off (``gpu`` row).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from _toolchain import real_arch

import hawk
import hawk.math as M
from hawk import Mutable, Scalar, kernel
from hawk.emit import literal_table, render_source, scalar_mode
from hawk.targets import TARGETS


@kernel(steps=1)
def poly(x: Scalar, y: Mutable[Scalar]):
    y = ((0.1 * x + 2.5e-3) * x - 1.75) * x + M.exp(x * 0.5) * 0.1 + 3.141592653589793


def _source(mode="float64"):
    return render_source(poly.name, poly.sinks, poly.walk, TARGETS["cuda"],
                         mode=scalar_mode(mode), kind=poly.kind)


def test_the_option_is_off_by_default_and_changes_no_byte(tmp_path, cache_dir):
    arch = real_arch("nvcc")
    hawk.build([poly], tmp_path / "default", targets=("cuda",), cache_dir=cache_dir, device_arch=arch)
    hawk.build([poly], tmp_path / "off", targets=("cuda",), cache_dir=cache_dir, device_arch=arch,
               literal_table=False)
    source = (tmp_path / "default" / "poly.cu").read_text()
    assert source == (tmp_path / "off" / "poly.cu").read_text()
    assert source.startswith(_source().text.split("\n", 1)[0])
    assert "hawk_f64_literals" not in source


def test_every_distinct_literal_lands_in_the_table_once():
    off, on = _source().text, literal_table(_source()).text
    (decl,) = re.findall(r"static __constant__ double hawk_f64_literals\[(\d+)\] = \{([^}]*)\};", on)
    size, values = int(decl[0]), decl[1].split(", ")
    assert size == len(values) == len(set(values)) >= 5
    assert not re.search(r"static_cast<Real>\(-?[0-9][0-9.eE+-]*\)", on)
    back = re.sub(r"static_cast<Real>\(hawk_f64_literals\[(\d+)\]\)",
                  lambda m: f"static_cast<Real>({values[int(m.group(1))]})", on)
    back = back.replace(f"static __constant__ double hawk_f64_literals[{size}] = "
                        f"{{{decl[1]}}};\n", "")
    assert back == off


def test_a_float32_source_is_returned_unchanged():
    f32 = _source("float32")
    assert literal_table(f32) is f32


def test_a_device_build_with_the_table_compiles(tmp_path, cache_dir):
    hawk.build([poly], tmp_path, targets=("cuda",), cache_dir=cache_dir,
               device_arch=real_arch("nvcc"), literal_table=True)
    assert "hawk_f64_literals" in (tmp_path / "poly.cu").read_text()


def test_the_host_build_is_the_same_with_the_option(tmp_path, cache_dir):
    off, on = tmp_path / "off", tmp_path / "on"
    hawk.build([poly], off, targets=("host",), cache_dir=cache_dir, host_profile="native")
    hawk.build([poly], on, targets=("host",), cache_dir=cache_dir, host_profile="native",
               literal_table=True)
    assert (off / "poly.cpp").read_text() == (on / "poly.cpp").read_text()
    x = np.linspace(-2.0, 2.0, 257)
    got = []
    for d in (off, on):
        y = np.zeros_like(x)
        hawk.run(hawk.load(d, "poly"), x=x.copy(), y=y)
        got.append(y)
    np.testing.assert_array_equal(*got)


@pytest.mark.gpu
def test_the_device_result_is_bit_identical_with_the_table(tmp_path, cache_dir):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    x = np.linspace(-2.0, 2.0, 257)
    got = []
    for tag, flag in (("off", False), ("on", True)):
        d = tmp_path / tag
        hawk.build([poly], d, targets=("cuda",), cache_dir=cache_dir,
                   device_arch=real_arch("nvcc"), literal_table=flag)
        sidecar = json.loads((d / "poly.json").read_text())
        xs, y = cp.asarray(x), cp.zeros(len(x))
        eplan.plan(L.device_plugin(d, "poly", sidecar),
                   structure=eexec.DeviceKernel).bind(x=xs, y=y).launch()
        got.append(cp.asnumpy(y))
    np.testing.assert_array_equal(*got)
    assert np.any(got[0] != 0.0)
