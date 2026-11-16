# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The banded carrier compiles through hawk's NVRTC driver from the sealed
payload: `aether/banded/*.h` (and `detail/`) are NVRTC-clean headers listed
in `aether_dsc/payload_headers.txt`, so a device source can
`#include <aether/banded/banded.h>` with no checkout and no `nvcc`.

Two arms, so the green one cannot be a blind compiler: a device chain over
`BandedReal` (add/mul/div/sqrt/exp/log/sin/atan2) must compile to a CUBIN
for the running (or toolkit-minimum) arch, and the same source calling the
host-only `BandedReal::fromDouble` must FAIL with a log that names it.
This is compile-only: no device is launched."""

from __future__ import annotations

import pytest

from _toolchain import real_arch
from hawk.compile import cubin_available, device
from raptor.errors import RaptorError as HawkError

def _skip_without_nvrtc_and_gpu():
    if not cubin_available():
        pytest.skip("no NVRTC binding or no CUDA device on this box")


_HEAD = """\
#include <aether/banded/banded.h>
#include <aether/banded/BandedRealOps.h>

using aether::banded::BandedReal;
"""

_CHAIN = _HEAD + """
extern "C" __global__ void band_chain(BandedReal* out, const BandedReal* a,
                                      const BandedReal* b, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const BandedReal x = a[i];
    const BandedReal y = b[i];
    BandedReal r = (x + y) * x / y;
    r = aether::math::sqrt(r) + aether::math::exp(x) - aether::math::log(y);
    r = r + aether::math::sin(x) + aether::math::atan2(x, y);
    out[i] = r;
}
"""

_HOST_ONLY = _HEAD + """
extern "C" __global__ void band_host_symbol(BandedReal* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = BandedReal::fromDouble(1.5);
}
"""


@pytest.mark.gpu
def test_banded_device_chain_compiles_to_a_cubin_from_the_sealed_payload():
    _skip_without_nvrtc_and_gpu()
    image = device(_CHAIN, target="cubin", arch=real_arch("nvrtc"), name="band_chain")
    assert image.target == "cubin"
    assert len(image.image) > 0


@pytest.mark.gpu
def test_banded_host_only_symbol_is_refused_and_named():
    _skip_without_nvrtc_and_gpu()
    with pytest.raises(HawkError) as err:
        device(_HOST_ONLY, target="cubin", arch=real_arch("nvrtc"), name="band_host_symbol")
    assert "fromDouble" in str(err.value)
