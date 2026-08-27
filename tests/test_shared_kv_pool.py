from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hpc101_infer.runtime.kv_cache import KVCache, PagedLayerKVCache
from hpc101_infer.scheduler import ContinuousBatchScheduler, RequestState
from hpc101_infer.types import GenerationRequest


def _paged_cache(
    pool_blocks: int,
    *,
    max_batch_size: int,
    max_sequence_length: int,
    block_size: int,
    max_blocks_per_sequence: int,
    ring: bool = False,
    window_size: int | None = None,
) -> PagedLayerKVCache:
    shape = (pool_blocks, block_size, 1, 1)
    return PagedLayerKVCache(
        key=torch.empty(shape),
        value=torch.empty(shape),
        lengths=torch.zeros(max_batch_size, dtype=torch.long),
        block_table=torch.full(
            (max_batch_size, max_blocks_per_sequence),
            -1,
            dtype=torch.long,
        ),
        max_batch_size=max_batch_size,
        max_sequence_length=max_sequence_length,
        block_size=block_size,
        max_blocks_per_sequence=max_blocks_per_sequence,
        ring=ring,
        window_size=window_size,
    )


def _state(length: int, max_new_tokens: int) -> RequestState:
    return RequestState(
        request=GenerationRequest(
            input_ids=list(range(1, length + 1)),
            max_new_tokens=max_new_tokens,
        ),
        prompt_token_ids=list(range(1, length + 1)),
    )


def test_allocate_keeps_full_tables_but_shrinks_physical_pools() -> None:
    config = SimpleNamespace(
        layer_types=("full_attention", "sliding_attention"),
        num_global_key_value_heads=1,
        global_head_dim=2,
        num_key_value_heads=1,
        head_dim=2,
        sliding_window=8,
    )

    cache = KVCache.allocate(
        config,
        max_batch_size=3,
        max_sequence_length=16,
        dtype=torch.bfloat16,
        device="cpu",
        paged_kv_block_size=4,
        paged_kv_global_pool_blocks=7,
        paged_kv_sliding_pool_blocks=5,
    )

    full, sliding = cache.layers
    assert isinstance(full, PagedLayerKVCache)
    assert isinstance(sliding, PagedLayerKVCache)
    assert full.pool_blocks == 7
    assert full.block_table.shape == (3, 4)
    assert sliding.pool_blocks == 5
    assert sliding.block_table.shape == (3, 3)


def test_lifecycle_credit_accounts_for_unaligned_ring_window() -> None:
    full = _paged_cache(
        300,
        max_batch_size=2,
        max_sequence_length=2048,
        block_size=16,
        max_blocks_per_sequence=128,
    )
    sliding = _paged_cache(
        130,
        max_batch_size=2,
        max_sequence_length=2048,
        block_size=16,
        max_blocks_per_sequence=65,
        ring=True,
        window_size=1024,
    )
    cache = KVCache([full, sliding], 2, 2048)
    cache.reset(2)

    cache.admit(0, 1024)
    cache.admit(1, 1025)

    assert cache._slot_block_credits == [[64, 65], [64, 65]]
    assert KVCache._block_demand(sliding, 2048) == 65


def test_scheduler_refills_after_credit_and_pages_are_released() -> None:
    layer = _paged_cache(
        4,
        max_batch_size=3,
        max_sequence_length=16,
        block_size=4,
        max_blocks_per_sequence=4,
    )
    cache = KVCache([layer], 3, 16)
    cache.reset(3)
    scheduler = ContinuousBatchScheduler(
        max_batch_size=3,
        prefill_token_budget=32,
        default_stop_token_ids=(),
        can_admit=lambda state: cache.can_admit(
            len(state.prompt_token_ids) + state.request.max_new_tokens
        ),
        on_admit=lambda slot, state: cache.admit(
            slot,
            len(state.prompt_token_ids) + state.request.max_new_tokens,
        ),
    )
    first = _state(4, 1)
    second = _state(4, 3)
    third = _state(4, 2)
    for state in (first, second, third):
        scheduler.add_request(state)

    prefill = scheduler.schedule_prefill()
    assert prefill.cache_slots == (0, 1)
    cache.reserve(prefill.cache_slots, ((0, 4), (0, 4)))
    completed = scheduler.update(prefill, [7, 8])
    assert completed == (0,)
    cache.release(completed)

    refill = scheduler.schedule_prefill()
    assert refill.cache_slots == (0,)
    assert refill.requests[0].request is third


def test_reserve_preflight_does_not_partially_allocate_layers() -> None:
    first = _paged_cache(
        2,
        max_batch_size=2,
        max_sequence_length=4,
        block_size=4,
        max_blocks_per_sequence=1,
    )
    second = _paged_cache(
        1,
        max_batch_size=2,
        max_sequence_length=4,
        block_size=4,
        max_blocks_per_sequence=1,
    )
    cache = KVCache([first, second], 2, 4)
    cache.reset(2)

    with pytest.raises(RuntimeError, match="block pool is exhausted"):
        cache.reserve((0, 1), ((0, 1), (0, 1)))

    assert first.active_block_count == 0
    assert second.active_block_count == 0
    assert first.block_table.eq(-1).all()
    assert second.block_table.eq(-1).all()


def test_public_and_distribution_upper_bound_block_counts() -> None:
    public_lengths = (298, 282, 516, 548, 1032, 1032, 1548, 1516, 2032, 2016)
    global_blocks = sum((length + 15) // 16 for length in public_lengths)
    sliding_blocks = sum(min((length + 15) // 16, 65) for length in public_lengths)

    assert global_blocks == 680
    assert sliding_blocks == 495
    assert 776 > global_blocks
    assert 530 > sliding_blocks
