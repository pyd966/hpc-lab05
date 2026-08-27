"""Fused W4A16 dequantization and matrix multiplication in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from hpc101_infer.quantization.packing import W4A16_K_BLOCK, W4A16_N_BLOCK


@triton.jit
def _w4a16_gemm_kernel(
    inputs_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    bias_ptr,
    output_ptr,
    m_size,
    n_size,
    k_size,
    n_blocks,
    stride_am,
    stride_ak,
    stride_om,
    stride_on,
    GROUP_SIZE: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    LAYOUT_K_BLOCK: tl.constexpr,
    LAYOUT_N_BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(m_size, BLOCK_M)
    programs_n = tl.cdiv(n_size, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    program_in_group = program_id % programs_per_group
    program_m = first_program_m + (program_in_group % group_m)
    program_n = program_in_group // group_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_n = tl.max_contiguous(offsets_n, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k_size, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            inputs_ptr
            + offsets_m[:, None] * stride_am
            + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < m_size)
            & (offsets_k[None, :] < k_size),
            other=0.0,
        )

        packed_offsets = (
            (
                (offsets_k[:, None] // LAYOUT_K_BLOCK) * n_blocks
                + offsets_n[None, :] // LAYOUT_N_BLOCK
            )
            * (LAYOUT_K_BLOCK // 2)
            + (offsets_k[:, None] % LAYOUT_K_BLOCK) // 2
        ) * LAYOUT_N_BLOCK + offsets_n[None, :] % LAYOUT_N_BLOCK
        packed = tl.load(
            qweight_ptr + packed_offsets,
            mask=(offsets_k[:, None] < k_size)
            & (offsets_n[None, :] < n_size),
            other=0,
        ).to(tl.int32)
        shifts = ((offsets_k & 1) * 4)[:, None]
        codes = (packed >> shifts) & 0x0F

        if GROUP_SIZE == BLOCK_K:
            group_index = k_start // GROUP_SIZE
            group_offsets = group_index * n_blocks * LAYOUT_N_BLOCK + offsets_n
            scale_values = tl.load(
                scales_ptr + group_offsets,
                mask=offsets_n < n_size,
                other=0.0,
            ).to(tl.float32)
            if HAS_ZEROS:
                zero_values = tl.load(
                    zeros_ptr + group_offsets,
                    mask=offsets_n < n_size,
                    other=0,
                ).to(tl.float32)
            else:
                zero_values = 8.0
            weight_values = (
                codes.to(tl.float32) - zero_values
            ) * scale_values
        else:
            group_indices = offsets_k // GROUP_SIZE
            group_offsets = (
                group_indices[:, None] * n_blocks * LAYOUT_N_BLOCK
                + offsets_n[None, :]
            )
            scale_values = tl.load(
                scales_ptr + group_offsets,
                mask=(offsets_k[:, None] < k_size)
                & (offsets_n[None, :] < n_size),
                other=0.0,
            ).to(tl.float32)
            if HAS_ZEROS:
                zero_values = tl.load(
                    zeros_ptr + group_offsets,
                    mask=(offsets_k[:, None] < k_size)
                    & (offsets_n[None, :] < n_size),
                    other=0,
                ).to(tl.float32)
            else:
                zero_values = 8.0
            weight_values = (codes.to(tl.float32) - zero_values) * scale_values
        if INPUT_BF16:
            weight_values = weight_values.to(tl.bfloat16)
        else:
            weight_values = weight_values.to(tl.float16)
        accumulator += tl.dot(input_values, weight_values)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offsets_n,
            mask=offsets_n < n_size,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        accumulator,
        mask=(offsets_m[:, None] < m_size)
        & (offsets_n[None, :] < n_size),
    )


def _launch_config(
    m_size: int,
    group_size: int,
) -> tuple[int, int, int, int, int, int]:
    block_k = 64 if group_size >= 64 else 32
    if m_size <= 16:
        return 16, 128, block_k, 4, 3, 1
    if m_size <= 64:
        return 32, 128, block_k, 4, 3, 4
    return 128, 64, block_k, 4, 3, 1


def w4a16_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    in_features: int,
    out_features: int,
    padded_in_features: int,
    group_size: int,
) -> torch.Tensor:
    """Compute fused dequantization and linear without a full weight tensor."""
    if not inputs.is_cuda:
        raise ValueError("the Triton W4A16 kernel requires CUDA inputs")
    if inputs.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("the Triton W4A16 kernel requires FP16 or BF16 inputs")
    if inputs.ndim == 0 or inputs.shape[-1] != in_features:
        raise ValueError("input shape does not match in_features")
    if qweight.dtype != torch.uint8:
        raise TypeError("qweight must use torch.uint8")
    if padded_in_features < in_features or padded_in_features % group_size:
        raise ValueError("invalid padded input size or group size")
    n_blocks = triton.cdiv(out_features, W4A16_N_BLOCK)
    k_blocks = triton.cdiv(padded_in_features, W4A16_K_BLOCK)
    expected_qweight_shape = (
        k_blocks,
        n_blocks,
        W4A16_K_BLOCK // 2,
        W4A16_N_BLOCK,
    )
    if qweight.shape != expected_qweight_shape:
        raise ValueError("qweight shape does not match the blocked layout")
    expected_scale_shape = (
        padded_in_features // group_size,
        n_blocks,
        W4A16_N_BLOCK,
    )
    if scales.shape != expected_scale_shape:
        raise ValueError("scale shape does not match the blocked layout")
    if zeros is not None and zeros.shape != expected_scale_shape:
        raise ValueError("zero-point shape does not match the blocked layout")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("bias shape does not match out_features")

    tensors = [qweight, scales]
    if zeros is not None:
        tensors.append(zeros)
    if bias is not None:
        tensors.append(bias)
    if any(tensor.device != inputs.device for tensor in tensors):
        raise ValueError("all W4A16 tensors must be on the input CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all W4A16 weight tensors must be contiguous")

    flattened = inputs.reshape(-1, in_features)
    if flattened.stride(1) != 1:
        flattened = flattened.contiguous()
    m_size = flattened.shape[0]
    output = torch.empty(
        (m_size, out_features),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    if m_size == 0:
        return output.reshape(*inputs.shape[:-1], out_features)

    (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        group_m,
    ) = _launch_config(m_size, group_size)
    grid = (
        triton.cdiv(m_size, block_m) * triton.cdiv(out_features, block_n),
    )
    zeros_argument = qweight if zeros is None else zeros
    bias_argument = qweight if bias is None else bias
    with torch.cuda.device(inputs.device):
        _w4a16_gemm_kernel[grid](
            flattened,
            qweight,
            scales,
            zeros_argument,
            bias_argument,
            output,
            m_size,
            out_features,
            in_features,
            n_blocks,
            flattened.stride(0),
            flattened.stride(1),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=group_size,
            HAS_ZEROS=zeros is not None,
            HAS_BIAS=bias is not None,
            INPUT_BF16=inputs.dtype == torch.bfloat16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            LAYOUT_K_BLOCK=W4A16_K_BLOCK,
            LAYOUT_N_BLOCK=W4A16_N_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output.reshape(*inputs.shape[:-1], out_features)
