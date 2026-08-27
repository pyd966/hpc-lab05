from __future__ import annotations

import pytest
import torch

from hpc101_infer.layers.linear import QuantizedLinear
from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import QuantizedWeight


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton W4A16 tests require CUDA"
)


def _quantized_weight(
    out_features: int,
    in_features: int,
    group_size: int,
    symmetric: bool,
) -> QuantizedWeight:
    padded = (in_features + group_size - 1) // group_size * group_size
    codes = torch.randint(
        0, 16, (out_features, padded), dtype=torch.uint8, device="cuda"
    )
    scales = (
        torch.rand(
            out_features,
            padded // group_size,
            dtype=torch.float16,
            device="cuda",
        )
        * 0.02
        + 0.005
    )
    zeros = None
    if not symmetric:
        zeros = torch.randint(
            2,
            14,
            scales.shape,
            dtype=torch.uint8,
            device="cuda",
        )
    return QuantizedWeight(
        qweight=pack_int4(codes),
        scales=scales,
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )


@pytest.mark.parametrize(
    (
        "leading_shape",
        "out_features",
        "in_features",
        "group_size",
        "symmetric",
        "bias",
    ),
    [
        ((1,), 96, 128, 64, True, False),
        ((3,), 75, 130, 32, False, True),
        ((2, 17), 160, 192, 64, True, False),
    ],
)
def test_w4a16_matches_reference(
    leading_shape: tuple[int, ...],
    out_features: int,
    in_features: int,
    group_size: int,
    symmetric: bool,
    bias: bool,
) -> None:
    torch.manual_seed(0)
    quantized = _quantized_weight(
        out_features, in_features, group_size, symmetric
    )
    bias_tensor = (
        torch.randn(out_features, device="cuda", dtype=torch.bfloat16) * 0.01
        if bias
        else None
    )
    reference = QuantizedLinear.from_quantized_weight(
        quantized, bias_tensor, backend="reference"
    )
    fused = QuantizedLinear.from_quantized_weight(
        quantized, bias_tensor, backend="triton"
    )
    inputs = torch.randn(
        *leading_shape, in_features, device="cuda", dtype=torch.bfloat16
    )

    expected = reference(inputs)
    actual = fused(inputs)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
