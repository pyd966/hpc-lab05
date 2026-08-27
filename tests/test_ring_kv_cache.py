from __future__ import annotations

import torch

from hpc101_infer.layers.attention import make_attention_mask
from hpc101_infer.runtime.kv_cache import LayerKVCache


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
