from __future__ import annotations

import torch

from hpc101_infer.layers.attention import make_attention_mask
from hpc101_infer.runtime.kv_cache import LayerKVCache, PagedLayerKVCache


def _cache(capacity: int = 4, batch_size: int = 1) -> LayerKVCache:
    return LayerKVCache(
        key=torch.full((batch_size, 1, capacity, 1), -1.0),
        value=torch.full((batch_size, 1, capacity, 1), -1.0),
        lengths=torch.zeros(batch_size, dtype=torch.long),
        max_batch_size=batch_size,
        max_sequence_length=16,
        ring=True,
        window_size=capacity,
    )


def test_ring_view_returns_recent_tokens_in_logical_order() -> None:
    cache = _cache()
    cache.reset(1)
    positions = torch.arange(6, dtype=torch.long)[None, :]
    values = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    cache.write(positions, values, values + 100, torch.tensor([6]))

    view = cache.view(6, torch.tensor([6]))

    assert view.key_positions.tolist() == [[2, 3, 4, 5]]
    assert view.key.flatten().tolist() == [2, 3, 4, 5]
    assert view.value.flatten().tolist() == [102, 103, 104, 105]


def test_ring_write_ignores_padding_for_variable_length_batch() -> None:
    cache = _cache(batch_size=2)
    cache.reset(2)
    positions = torch.arange(6, dtype=torch.long).expand(2, -1)
    values = torch.stack(
        [torch.arange(6, dtype=torch.float32), torch.arange(10, 16, dtype=torch.float32)]
    ).reshape(2, 1, 6, 1)
    cache.write(positions, values, values, torch.tensor([2, 6]))

    view = cache.view(6, torch.tensor([2, 6]))

    assert view.key_positions.tolist() == [[0, 1, 2, 3], [2, 3, 4, 5]]
    assert view.key[1].flatten().tolist() == [12, 13, 14, 15]
    assert cache.key[0, 0, :2, 0].tolist() == [0, 1]


def test_paged_ring_keeps_a_window_crossing_page_boundary() -> None:
    cache = PagedLayerKVCache(
        key=torch.full((2, 4, 1, 1), -1.0),
        value=torch.full((2, 4, 1, 1), -1.0),
        lengths=torch.zeros(1, dtype=torch.long),
        block_table=torch.full((1, 2), -1, dtype=torch.long),
        max_batch_size=1,
        max_sequence_length=16,
        block_size=4,
        max_blocks_per_sequence=2,
        ring=True,
        window_size=5,
    )
    cache.reset(1)
    positions = torch.arange(11, dtype=torch.long)[None, :]
    values = torch.arange(11, dtype=torch.float32).reshape(1, 1, 11, 1)
    cache.write(positions, values, values + 100, torch.tensor([11]))

    view = cache.view(11, torch.tensor([11]))

    assert view.key_positions.tolist() == [[6, 7, 8, 9, 10]]
    assert view.key.flatten().tolist() == [6, 7, 8, 9, 10]
    assert view.value.flatten().tolist() == [106, 107, 108, 109, 110]
    assert cache.active_block_count == 2

    cache.release([0])
    assert cache.active_block_count == 0
    assert cache.block_table.tolist() == [[-1, -1]]


def test_paged_full_cache_collects_noncontiguous_blocks() -> None:
    cache = PagedLayerKVCache(
        key=torch.full((3, 4, 2, 1), -1.0),
        value=torch.full((3, 4, 2, 1), -1.0),
        lengths=torch.zeros(1, dtype=torch.long),
        block_table=torch.full((1, 3), -1, dtype=torch.long),
        max_batch_size=1,
        max_sequence_length=10,
        block_size=4,
        max_blocks_per_sequence=3,
        ring=False,
    )
    cache.reset(1)
    positions = torch.arange(10, dtype=torch.long)[None, :]
    values = torch.stack(
        [torch.arange(10, dtype=torch.float32), torch.arange(20, 30, dtype=torch.float32)]
    ).reshape(1, 2, 10, 1)
    cache.write(positions, values, values + 100, torch.tensor([10]))

    view = cache.view(10, torch.tensor([10]))

    assert view.key_positions.tolist() == [list(range(10))]
    assert view.key[0, 0].flatten().tolist() == list(range(10))
    assert view.key[0, 1].flatten().tolist() == list(range(20, 30))
    assert view.value[0, 0].flatten().tolist() == list(range(100, 110))
    assert view.value[0, 1].flatten().tolist() == list(range(120, 130))
    assert cache.active_block_count == 3


def test_sliding_mask_uses_per_batch_ring_key_positions() -> None:
    positions = torch.tensor([[4, 5], [7, 8]])
    sequence_lengths = torch.tensor([6, 9])
    key_positions = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])

    mask, query_valid = make_attention_mask(
        positions,
        sequence_lengths,
        key_length=4,
        layer_type="sliding_attention",
        dtype=torch.float32,
        sliding_window=4,
        key_positions=key_positions,
    )

    assert query_valid.tolist() == [[True, True], [True, True]]
    assert mask.shape == (2, 1, 2, 4)
    assert mask[0, 0, 0].isfinite().tolist() == [True, True, True, False]
    assert mask[0, 0, 1].isfinite().tolist() == [True, True, True, True]
    assert mask[1, 0, 0].isfinite().tolist() == [True, True, True, False]
    assert mask[1, 0, 1].isfinite().tolist() == [True, True, True, True]
