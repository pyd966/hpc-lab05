from __future__ import annotations

import pytest
import torch

from hpc101_infer.kernels.flash_attention import (
    flash_attention,
    paged_flash_attention,
)
from hpc101_infer.layers.attention import make_attention_mask, repeat_kv
from hpc101_infer.runtime.kv_cache import PagedLayerKVCache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    sliding_window: int = -1,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key = repeat_kv(key, repeats)
    value = repeat_kv(value, repeats)
    scores = query @ key.transpose(2, 3)
    mask, query_valid = make_attention_mask(
        query_positions,
        sequence_lengths,
        key.shape[2],
        "sliding_attention" if sliding_window > 0 else "full_attention",
        scores.dtype,
        sliding_window=sliding_window,
        key_positions=key_positions,
    )
    probabilities = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
    probabilities = probabilities.to(query.dtype)
    output = probabilities @ value
    return output * query_valid[:, None, :, None]


@pytest.mark.parametrize(
    ("head_dim", "sliding_window"),
    [(256, -1), (256, 16), (512, -1)],
)
def test_dense_flash_matches_eager(head_dim: int, sliding_window: int) -> None:
    torch.manual_seed(17 + head_dim)
    batch, query_heads, kv_heads, length = 2, 4, 2, 33
    if head_dim == 512:
        query_heads, kv_heads, length = 2, 1, 17
    query = torch.randn(
        batch, query_heads, length, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    key = torch.randn(
        batch, kv_heads, length, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    value = torch.randn_like(key)
    positions = torch.arange(length, device="cuda").expand(batch, -1)
    sequence_lengths = torch.tensor(
        [length, length - 7], device="cuda", dtype=torch.long
    )
    expected = _reference(
        query, key, value, positions, positions, sequence_lengths, sliding_window
    )
    actual = flash_attention(
        query,
        key,
        value,
        positions,
        positions,
        sequence_lengths,
        sliding_window=sliding_window,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def _paged_cache(
    *,
    batch: int,
    kv_heads: int,
    head_dim: int,
    max_length: int,
    block_size: int,
    max_blocks: int,
    ring: bool,
    window: int | None,
) -> PagedLayerKVCache:
    pool_blocks = batch * max_blocks
    return PagedLayerKVCache(
        key=torch.empty(
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        value=torch.empty(
            pool_blocks,
            block_size,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        lengths=torch.zeros(batch, device="cuda", dtype=torch.long),
        block_table=torch.full(
            (batch, max_blocks), -1, device="cuda", dtype=torch.long
        ),
        max_batch_size=batch,
        max_sequence_length=max_length,
        block_size=block_size,
        max_blocks_per_sequence=max_blocks,
        ring=ring,
        window_size=window,
    )


def test_paged_flash_matches_eager_with_gqa_and_padding() -> None:
    torch.manual_seed(29)
    batch, query_heads, kv_heads, length, head_dim = 2, 4, 2, 35, 256
    query = torch.randn(
        batch, query_heads, length, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    key = torch.randn(
        batch, kv_heads, length, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    value = torch.randn_like(key)
    positions = torch.arange(length, device="cuda").expand(batch, -1)
    sequence_lengths = torch.tensor([35, 27], device="cuda", dtype=torch.long)
    cache = _paged_cache(
        batch=batch,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_length=length,
        block_size=16,
        max_blocks=3,
        ring=False,
        window=None,
    )
    cache.reset(batch)
    cache.write(positions, key, value, sequence_lengths)
    expected = _reference(
        query, key, value, positions, positions, sequence_lengths
    )
    actual = paged_flash_attention(
        query,
        cache.key,
        cache.value,
        cache.block_table,
        positions,
        sequence_lengths,
        max_key_length=length,
        block_size=16,
        max_blocks=3,
        ring=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_paged_ring_decode_matches_visible_window() -> None:
    torch.manual_seed(41)
    query_heads, kv_heads, length, head_dim, window = 4, 2, 80, 256, 32
    key = torch.randn(
        1, kv_heads, length, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    value = torch.randn_like(key)
    positions = torch.arange(length, device="cuda").unsqueeze(0)
    sequence_lengths = torch.tensor([length], device="cuda", dtype=torch.long)
    cache = _paged_cache(
        batch=1,
        kv_heads=kv_heads,
        head_dim=head_dim,
        max_length=length,
        block_size=16,
        max_blocks=3,
        ring=True,
        window=window,
    )
    cache.reset(1)
    cache.write(positions, key, value, sequence_lengths)
    query = torch.randn(
        1, query_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    query_position = positions[:, -1:]
    visible_key = key[:, :, -window:, :]
    visible_value = value[:, :, -window:, :]
    visible_positions = positions[:, -window:]
    expected = _reference(
        query,
        visible_key,
        visible_value,
        query_position,
        visible_positions,
        sequence_lengths,
        window,
    )
    actual = paged_flash_attention(
        query,
        cache.key,
        cache.value,
        cache.block_table,
        query_position,
        sequence_lengths,
        max_key_length=length,
        block_size=16,
        max_blocks=3,
        ring=True,
        sliding_window=window,
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
