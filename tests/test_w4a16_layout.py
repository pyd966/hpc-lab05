from __future__ import annotations

import pytest
import torch

from hpc101_infer.quantization.packing import (
    W4A16_K_BLOCK,
    W4A16_N_BLOCK,
    pack_w4a16_group_tensor,
    pack_w4a16_qweight,
    unpack_w4a16_group_tensor,
    unpack_w4a16_qweight,
)


@pytest.mark.parametrize(
    ("out_features", "padded_in_features", "num_groups"),
    [
        (128, 64, 1),
        (75, 160, 5),
        (3840, 192, 3),
    ],
)
def test_w4a16_layout_round_trip(
    out_features: int,
    padded_in_features: int,
    num_groups: int,
) -> None:
    torch.manual_seed(0)
    qweight = torch.randint(
        0,
        256,
        (out_features, padded_in_features // 2),
        dtype=torch.uint8,
    )
    scales = torch.randn(out_features, num_groups, dtype=torch.float16)
    zeros = torch.randint(
        0,
        16,
        (out_features, num_groups),
        dtype=torch.uint8,
    )

    blocked_qweight = pack_w4a16_qweight(
        qweight,
        out_features=out_features,
        padded_in_features=padded_in_features,
    )
    blocked_scales = pack_w4a16_group_tensor(
        scales,
        out_features=out_features,
        num_groups=num_groups,
    )
    blocked_zeros = pack_w4a16_group_tensor(
        zeros,
        out_features=out_features,
        num_groups=num_groups,
    )

    assert blocked_qweight.shape == (
        (padded_in_features + W4A16_K_BLOCK - 1) // W4A16_K_BLOCK,
        (out_features + W4A16_N_BLOCK - 1) // W4A16_N_BLOCK,
        W4A16_K_BLOCK // 2,
        W4A16_N_BLOCK,
    )
    assert torch.equal(
        unpack_w4a16_qweight(
            blocked_qweight,
            out_features=out_features,
            padded_in_features=padded_in_features,
        ),
        qweight,
    )
    assert torch.equal(
        unpack_w4a16_group_tensor(
            blocked_scales,
            out_features=out_features,
            num_groups=num_groups,
        ),
        scales,
    )
    assert torch.equal(
        unpack_w4a16_group_tensor(
            blocked_zeros,
            out_features=out_features,
            num_groups=num_groups,
        ),
        zeros,
    )


def test_w4a16_layout_preserves_nibble_order() -> None:
    qweight = torch.zeros(129, 64, dtype=torch.uint8)
    qweight[5, 33] = 0xA3
    qweight[128, 0] = 0x7C

    blocked = pack_w4a16_qweight(
        qweight,
        out_features=129,
        padded_in_features=128,
    )

    assert blocked[1, 0, 1, 5].item() == 0xA3
    assert blocked[0, 1, 0, 0].item() == 0x7C
