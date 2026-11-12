"""The warp-per-lane kernel's block of k warps is a card-kit switch: k = 1 emits the one-warp block's text, k > 1 reads the
lane off its warp and the lane count off nSamples."""
from hawk.passes import policy
from hawk_cuda.emit import CudaWrapper

PLAIN = """    const long long hawk_flat =
        static_cast<long long>(blockIdx.x);
    if (hawk_flat >= count) return;"""


def test_one_warp_block_text_unchanged():
    text = CudaWrapper().index_prologue(team=32)
    assert PLAIN in text and "blockDim" not in text and "threadIdx.x >> 5" not in text.split("hawk_wl")[0]


def test_k_warps_block_reads_lane_and_count():
    text = CudaWrapper().index_prologue(team=32, warps=8)
    assert "static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5);" in text
    assert "hawk_flat >= nSamples" in text
    assert "blockIdx.x) * 8" not in CudaWrapper().index_prologue(team=128, warps=8)
