from __future__ import annotations

from types import SimpleNamespace

import torch

from hpc101_infer.runner import Runner
from hpc101_infer.runtime.kv_cache import KVCache, PagedLayerKVCache
from hpc101_infer.scheduler import ContinuousBatchScheduler, RequestState, RequestStatus
from hpc101_infer.types import GenerationRequest


def _state(length: int, max_new_tokens: int) -> RequestState:
    return RequestState(
        request=GenerationRequest(
            input_ids=list(range(1, length + 1)),
            max_new_tokens=max_new_tokens,
        ),
        prompt_token_ids=list(range(1, length + 1)),
    )


def test_continuous_scheduler_refills_completed_slot() -> None:
    scheduler = ContinuousBatchScheduler(
        max_batch_size=2,
        prefill_token_budget=8,
        default_stop_token_ids=(99,),
    )
    first = _state(4, 1)
    second = _state(4, 3)
    third = _state(8, 2)
    for state in (first, second, third):
        scheduler.add_request(state)

    prefill = scheduler.schedule_prefill()
    assert prefill.cache_slots == (0, 1)
    assert scheduler.update(prefill, [7, 8]) == (0,)
    assert first.status is RequestStatus.COMPLETED

    refill = scheduler.schedule_prefill()
    assert refill.cache_slots == (0,)
    assert refill.requests[0].request is third
    assert scheduler.update(refill, [9]) == ()

    decode = scheduler.schedule_decode()
    assert decode.cache_slots == (0, 1)
    assert [item.request for item in decode.requests] == [third, second]
    assert scheduler.update(decode, [10, 11]) == (0,)
    assert third.status is RequestStatus.COMPLETED
    assert scheduler.has_unfinished_requests()


def test_continuous_prefill_budget_uses_padded_work() -> None:
    scheduler = ContinuousBatchScheduler(
        max_batch_size=4,
        prefill_token_budget=12,
        default_stop_token_ids=(),
    )
    for length in (3, 4, 7):
        scheduler.add_request(_state(length, 2))
    schedule = scheduler.schedule_prefill()
    assert [len(item.request.prompt_token_ids) for item in schedule.requests] == [3, 4]
    assert schedule.batch_max_length == 4


def test_paged_cache_maps_compact_rows_to_stable_slots() -> None:
    cache = PagedLayerKVCache(
        key=torch.empty(6, 2, 1, 2),
        value=torch.empty(6, 2, 1, 2),
        lengths=torch.zeros(3, dtype=torch.long),
        block_table=torch.full((3, 2), -1, dtype=torch.long),
        max_batch_size=3,
        max_sequence_length=4,
        block_size=2,
        max_blocks_per_sequence=2,
    )
    kv_cache = KVCache([cache], max_batch_size=3, max_sequence_length=4)
    kv_cache.reset(3)
    positions = torch.arange(3).expand(2, -1)
    key = torch.tensor(
        [
            [[[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]],
            [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]],
        ]
    )
    value = key + 100
    lengths = torch.tensor([3, 3])
    slots = torch.tensor([2, 0])
    slot_ids = (2, 0)
    sequence_ranges = ((0, 3), (0, 3))
    kv_cache.reserve(slot_ids, sequence_ranges)
    kv_cache.write(
        0,
        positions,
        key,
        value,
        lengths,
        slots,
        slot_ids,
        sequence_ranges,
    )
    kv_cache.commit(lengths, batch_indices=slots)

    view = cache.view(3, lengths, slots)
    torch.testing.assert_close(view.key, key)
    torch.testing.assert_close(view.value, value)
    assert cache.lengths.tolist() == [3, 0, 3]
    assert cache.block_table[1].tolist() == [-1, -1]

    before_release = cache.active_block_count
    cache.release((2,))
    assert cache.lengths.tolist() == [3, 0, 0]
    assert cache.active_block_count < before_release


def test_runner_passes_the_entire_queue_to_continuous_engine() -> None:
    requests = [GenerationRequest(input_ids=[index + 1]) for index in range(5)]

    class FakeEngine:
        config = SimpleNamespace(scheduler_backend="continuous")

        def __init__(self) -> None:
            self.seen = []

        def generate(self, queued):
            self.seen = list(queued)
            return list(queued)

    engine = FakeEngine()
    outputs = Runner(engine).run(requests)
    assert engine.seen == requests
    assert outputs == requests
