from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from hpc101_infer.quantization.types import QuantizedWeight

W4A16_LAYOUT = "uint8_k64_n128_blocked_little_nibble_v1"
W4A16_K_BLOCK = 64
W4A16_N_BLOCK = 128


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("int4 values must use an integer dtype")
    if values.shape[-1] % 2:
        values = F.pad(values, (0, 1))
    values = values.to(torch.uint8) & 0x0F
    return values[..., 0::2] | (values[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor, length: int | None = None) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("packed int4 tensor must use torch.uint8")
    unpacked = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    if length is not None:
        if length < 0 or length > unpacked.shape[-1]:
            raise ValueError("invalid unpacked length")
        unpacked = unpacked[..., :length]
    return unpacked


def pack_w4a16_qweight(
    qweight: torch.Tensor,
    *,
    out_features: int,
    padded_in_features: int,
) -> torch.Tensor:
    """Reorder canonical [N, K/2] INT4 bytes for N-contiguous W4A16 loads."""
    if qweight.dtype != torch.uint8:
        raise TypeError("packed int4 tensor must use torch.uint8")
    canonical_k_bytes = (padded_in_features + 1) // 2
    if qweight.shape != (out_features, canonical_k_bytes):
        raise ValueError("canonical qweight shape does not match metadata")

    k_blocks = math.ceil(padded_in_features / W4A16_K_BLOCK)
    n_blocks = math.ceil(out_features / W4A16_N_BLOCK)
    padded = torch.zeros(
        n_blocks * W4A16_N_BLOCK,
        k_blocks * (W4A16_K_BLOCK // 2),
        dtype=qweight.dtype,
        device=qweight.device,
    )
    padded[:out_features, :canonical_k_bytes].copy_(qweight)
    return (
        padded.reshape(
            n_blocks,
            W4A16_N_BLOCK,
            k_blocks,
            W4A16_K_BLOCK // 2,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
    )


def unpack_w4a16_qweight(
    qweight: torch.Tensor,
    *,
    out_features: int,
    padded_in_features: int,
) -> torch.Tensor:
    """Restore canonical [N, K/2] bytes from the W4A16 blocked layout."""
    if qweight.dtype != torch.uint8:
        raise TypeError("packed int4 tensor must use torch.uint8")
    k_blocks = math.ceil(padded_in_features / W4A16_K_BLOCK)
    n_blocks = math.ceil(out_features / W4A16_N_BLOCK)
    expected = (
        k_blocks,
        n_blocks,
        W4A16_K_BLOCK // 2,
        W4A16_N_BLOCK,
    )
    if qweight.shape != expected:
        raise ValueError("blocked qweight shape does not match metadata")
    canonical_k_bytes = (padded_in_features + 1) // 2
    return (
        qweight.permute(1, 3, 0, 2)
        .reshape(
            n_blocks * W4A16_N_BLOCK,
            k_blocks * (W4A16_K_BLOCK // 2),
        )[:out_features, :canonical_k_bytes]
        .contiguous()
    )


def pack_w4a16_group_tensor(
    tensor: torch.Tensor,
    *,
    out_features: int,
    num_groups: int,
) -> torch.Tensor:
    """Reorder [N, groups] scales or zeros into [groups, N/128, 128]."""
    if tensor.shape != (out_features, num_groups):
        raise ValueError("group tensor shape does not match metadata")
    n_blocks = math.ceil(out_features / W4A16_N_BLOCK)
    padded = torch.zeros(
        num_groups,
        n_blocks * W4A16_N_BLOCK,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :out_features].copy_(tensor.transpose(0, 1))
    return padded.reshape(num_groups, n_blocks, W4A16_N_BLOCK).contiguous()


def unpack_w4a16_group_tensor(
    tensor: torch.Tensor,
    *,
    out_features: int,
    num_groups: int,
) -> torch.Tensor:
    """Restore canonical [N, groups] scales or zeros."""
    n_blocks = math.ceil(out_features / W4A16_N_BLOCK)
    expected = (num_groups, n_blocks, W4A16_N_BLOCK)
    if tensor.shape != expected:
        raise ValueError("blocked group tensor shape does not match metadata")
    return (
        tensor.reshape(num_groups, n_blocks * W4A16_N_BLOCK)[:, :out_features]
        .transpose(0, 1)
        .contiguous()
    )


def dequantize_weight(
    quantized: QuantizedWeight,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if quantized.bits != 4:
        raise ValueError("only int4 weights are supported")
    out_features, padded_in_features = quantized.padded_shape
    encoded = unpack_int4(quantized.qweight, padded_in_features)
    encoded = encoded.view(out_features, -1, quantized.group_size).to(dtype)
    scales = quantized.scales.to(dtype).unsqueeze(-1)
    if quantized.symmetric:
        values = encoded - 8.0
    else:
        if quantized.zeros is None:
            raise ValueError("asymmetric quantization requires zero points")
        values = encoded - quantized.zeros.to(dtype).unsqueeze(-1)
    weight = (values * scales).reshape(out_features, padded_in_features)
    return weight[:, : quantized.original_shape[1]].to(dtype)
