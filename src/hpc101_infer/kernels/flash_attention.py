"""Self-written Triton Flash/Paged Attention kernels.

The kernel keeps only one score tile in SRAM/registers and applies online
softmax. It therefore never materializes the full ``[query, key]`` matrix.
Paged mode resolves logical token positions through the KV block table inside
the kernel, while dense mode is useful without a cache and for long prefill
into a ring cache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    query_positions_ptr,
    key_positions_ptr,
    sequence_lengths_ptr,
    block_table_ptr,
    query_length,
    key_span,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_qpb,
    stride_qpm,
    stride_kpb,
    stride_kpn,
    stride_btb,
    stride_bts,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    PAGED: tl.constexpr,
    RING: tl.constexpr,
    INPUT_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    output_block = tl.program_id(2)
    batch_index = batch_head // num_query_heads
    query_head = batch_head % num_query_heads
    kv_head = query_head // (num_query_heads // num_kv_heads)

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_dv = output_block * BLOCK_DV + tl.arange(0, BLOCK_DV)
    query_mask = offsets_m < query_length
    query_positions = tl.load(
        query_positions_ptr
        + batch_index * stride_qpb
        + offsets_m * stride_qpm,
        mask=query_mask,
        other=0,
    ).to(tl.int64)
    sequence_length = tl.load(sequence_lengths_ptr + batch_index).to(tl.int64)
    row_valid = query_mask & (query_positions < sequence_length)

    if PAGED:
        if RING:
            key_start = tl.maximum(sequence_length - SLIDING_WINDOW, 0)
        else:
            key_start = 0
    else:
        key_start = tl.load(key_positions_ptr + batch_index * stride_kpb).to(
            tl.int64
        )

    # Causal attention never needs keys after the newest query in this block.
    # Sliding attention can also skip blocks entirely before the oldest query's
    # window. Align the lower bound so every iteration remains a full dot tile.
    first_query = tl.min(tl.where(row_valid, query_positions, sequence_length))
    last_query = tl.max(tl.where(row_valid, query_positions, -1))
    loop_start = 0
    if SLIDING_WINDOW > 0:
        loop_start = tl.maximum(
            first_query - SLIDING_WINDOW + 1 - key_start,
            0,
        )
        loop_start = (loop_start // BLOCK_N) * BLOCK_N
    loop_end = tl.minimum(
        tl.maximum(last_query - key_start + 1, 0),
        key_span,
    )

    # Slice the output dimension so 512-wide global heads do not exhaust
    # registers. Each slice repeats QK but never writes scores to global memory.
    accumulator = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for key_offset in range(loop_start, loop_end, BLOCK_N):
        offsets_n = key_offset + tl.arange(0, BLOCK_N)
        if PAGED:
            key_positions = key_start + offsets_n
            logical_blocks = key_positions // BLOCK_SIZE
            if RING:
                table_slots = logical_blocks % MAX_BLOCKS
            else:
                table_slots = logical_blocks
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * stride_btb
                + table_slots * stride_bts,
                mask=offsets_n < key_span,
                other=-1,
            ).to(tl.int64)
            token_offsets = key_positions % BLOCK_SIZE
            physical_tokens = physical_blocks * BLOCK_SIZE + token_offsets
            key_exists = physical_blocks >= 0
        else:
            key_positions = tl.load(
                key_positions_ptr
                + batch_index * stride_kpb
                + offsets_n * stride_kpn,
                mask=offsets_n < key_span,
                other=0,
            ).to(tl.int64)
            physical_tokens = offsets_n
            key_exists = offsets_n < key_span

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for dim_offset in range(0, HEAD_DIM, BLOCK_K):
            offsets_k = dim_offset + tl.arange(0, BLOCK_K)
            query_values = tl.load(
                query_ptr
                + batch_index * stride_qb
                + query_head * stride_qh
                + offsets_m[:, None] * stride_qm
                + offsets_k[None, :] * stride_qd,
                mask=query_mask[:, None] & (offsets_k[None, :] < HEAD_DIM),
                other=0.0,
            )
            if PAGED:
                key_values = tl.load(
                    key_ptr
                    + physical_tokens[None, :] * stride_kn
                    + kv_head * stride_kh
                    + offsets_k[:, None] * stride_kd,
                    mask=key_exists[None, :]
                    & (offsets_k[:, None] < HEAD_DIM),
                    other=0.0,
                )
            else:
                key_values = tl.load(
                    key_ptr
                    + batch_index * stride_kb
                    + kv_head * stride_kh
                    + offsets_n[None, :] * stride_kn
                    + offsets_k[:, None] * stride_kd,
                    mask=(offsets_n[None, :] < key_span)
                    & (offsets_k[:, None] < HEAD_DIM),
                    other=0.0,
                )
            scores += tl.dot(query_values, key_values)

        allowed = row_valid[:, None] & key_exists[None, :]
        allowed &= key_positions[None, :] < sequence_length
        allowed &= key_positions[None, :] <= query_positions[:, None]
        if SLIDING_WINDOW > 0:
            allowed &= key_positions[None, :] > (
                query_positions[:, None] - SLIDING_WINDOW
            )
        scores = tl.where(allowed, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        has_value = new_max != -float("inf")
        alpha = tl.where(
            has_value,
            tl.exp2((row_max - new_max) * log2e),
            0.0,
        )
        probabilities = tl.exp2((scores - new_max[:, None]) * log2e)
        probabilities = tl.where(allowed, probabilities, 0.0)
        new_sum = row_sum * alpha + tl.sum(probabilities, axis=1)

        if PAGED:
            value_values = tl.load(
                value_ptr
                + physical_tokens[:, None] * stride_vn
                + kv_head * stride_vh
                + offsets_dv[None, :] * stride_vd,
                mask=key_exists[:, None]
                & (offsets_dv[None, :] < HEAD_DIM),
                other=0.0,
            )
        else:
            value_values = tl.load(
                value_ptr
                + batch_index * stride_vb
                + kv_head * stride_vh
                + offsets_n[:, None] * stride_vn
                + offsets_dv[None, :] * stride_vd,
                mask=(offsets_n[:, None] < key_span)
                & (offsets_dv[None, :] < HEAD_DIM),
                other=0.0,
            )
        if INPUT_BF16:
            probabilities = probabilities.to(tl.bfloat16)
        else:
            probabilities = probabilities.to(tl.float16)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities, value_values
        )
        row_max = tl.where(has_value, new_max, row_max)
        row_sum = new_sum

    denominator = tl.where(row_sum > 0, row_sum, 1.0)
    output = accumulator / denominator[:, None]
    output = tl.where(row_valid[:, None], output, 0.0)
    tl.store(
        output_ptr
        + batch_index * stride_ob
        + query_head * stride_oh
        + offsets_m[:, None] * stride_om
        + offsets_dv[None, :] * stride_od,
        output,
        mask=query_mask[:, None] & (offsets_dv[None, :] < HEAD_DIM),
    )


def _validate_common(
    query: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    num_kv_heads: int,
) -> tuple[int, int, int, int]:
    if not query.is_cuda:
        raise ValueError("the Triton attention kernel requires CUDA inputs")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("query must use FP16 or BF16")
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence, head_dim]")
    batch, query_heads, query_length, head_dim = query.shape
    if query_positions.shape != (batch, query_length):
        raise ValueError("query_positions has an invalid shape")
    if sequence_lengths.shape != (batch,):
        raise ValueError("sequence_lengths has an invalid shape")
    if query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if head_dim <= 0 or head_dim % 16:
        raise ValueError("head_dim must be a positive multiple of 16")
    if any(
        tensor.device != query.device
        for tensor in (sequence_lengths, query_positions)
    ):
        raise ValueError("attention metadata must be on the query device")
    return batch, query_heads, query_length, head_dim


def _launch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_table: torch.Tensor,
    *,
    key_span: int,
    num_kv_heads: int,
    sliding_window: int,
    paged: bool,
    ring: bool,
    block_size: int,
    max_blocks: int,
) -> torch.Tensor:
    batch, query_heads, query_length, head_dim = _validate_common(
        query, sequence_lengths, query_positions, num_kv_heads
    )
    output = torch.empty_like(query)
    if query_length == 0:
        return output
    block_m = 16
    block_n = 32
    block_k = 64
    block_dv = 256
    grid = (
        triton.cdiv(query_length, block_m),
        batch * query_heads,
        triton.cdiv(head_dim, block_dv),
    )
    with torch.cuda.device(query.device):
        _flash_attention_kernel[grid](
            query,
            key,
            value,
            output,
            query_positions,
            key_positions,
            sequence_lengths,
            block_table,
            query_length,
            key_span,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0) if not paged else 0,
            key.stride(1) if not paged else key.stride(2),
            key.stride(2) if not paged else key.stride(0) // block_size,
            key.stride(3),
            value.stride(0) if not paged else 0,
            value.stride(1) if not paged else value.stride(2),
            value.stride(2) if not paged else value.stride(0) // block_size,
            value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            query_positions.stride(0), query_positions.stride(1),
            key_positions.stride(0), key_positions.stride(1),
            block_table.stride(0), block_table.stride(1),
            num_query_heads=query_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            MAX_BLOCKS=max_blocks,
            SLIDING_WINDOW=sliding_window,
            PAGED=paged,
            RING=ring,
            INPUT_BF16=query.dtype == torch.bfloat16,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            BLOCK_DV=block_dv,
            num_warps=8,
            num_stages=3,
        )
    return output


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    sliding_window: int = -1,
) -> torch.Tensor:
    """Run dense custom Flash Attention without materializing score tensors."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key/value must have matching [batch, heads, key, dim] shapes")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    batch, kv_heads, key_span, head_dim = key.shape
    if query.shape[0] != batch or query.shape[3] != head_dim:
        raise ValueError("query and key/value dimensions do not match")
    if key_positions.shape != (batch, key_span):
        raise ValueError("key_positions has an invalid shape")
    if key_span <= 0:
        raise ValueError("key sequence must not be empty")
    return _launch(
        query, key, value, query_positions, key_positions, sequence_lengths,
        key_positions,
        key_span=key_span,
        num_kv_heads=kv_heads,
        sliding_window=sliding_window,
        paged=False,
        ring=False,
        block_size=1,
        max_blocks=1,
    )


def paged_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_key_length: int,
    block_size: int,
    max_blocks: int,
    ring: bool,
    sliding_window: int = -1,
) -> torch.Tensor:
    """Run custom Flash Attention directly over paged KV storage."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("paged key/value must have matching four-dimensional shapes")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if key.shape[1] != block_size:
        raise ValueError("paged key/value block size does not match metadata")
    kv_heads, head_dim = key.shape[2], key.shape[3]
    if query.shape[3] != head_dim:
        raise ValueError("query and paged key/value head dimensions do not match")
    if block_table.shape != (query.shape[0], max_blocks):
        raise ValueError("block_table has an invalid shape")
    if max_key_length <= 0:
        raise ValueError("max_key_length must be positive")
    key_span = min(max_key_length, sliding_window) if ring else max_key_length
    return _launch(
        query, key, value, query_positions, query_positions, sequence_lengths,
        block_table,
        key_span=key_span,
        num_kv_heads=kv_heads,
        sliding_window=sliding_window,
        paged=True,
        ring=ring,
        block_size=block_size,
        max_blocks=max_blocks,
    )
