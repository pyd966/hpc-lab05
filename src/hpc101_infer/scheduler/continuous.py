from __future__ import annotations

import heapq
from collections import deque

from hpc101_infer.scheduler.base import (
    RequestState,
    RequestStatus,
    ScheduledOutput,
    ScheduledRequest,
)


class ContinuousBatchScheduler:
    """Maintain a waiting queue and stable KV slots for continuous batching."""

    def __init__(
        self,
        max_batch_size: int,
        prefill_token_budget: int,
        default_stop_token_ids: tuple[int, ...],
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_token_budget <= 0:
            raise ValueError("prefill_token_budget must be positive")
        self.max_batch_size = max_batch_size
        self.prefill_token_budget = prefill_token_budget
        self.default_stop_token_ids = default_stop_token_ids
        self.pending: deque[RequestState] = deque()
        self.active: dict[int, RequestState] = {}
        self._free_slots = list(range(max_batch_size))
        heapq.heapify(self._free_slots)

    def add_request(self, request: RequestState) -> None:
        if request.status is not RequestStatus.PENDING:
            raise ValueError("new requests must be pending")
        if request.request.max_new_tokens == 0:
            request.status = RequestStatus.COMPLETED
            request.finish_reason = "length"
            return
        self.pending.append(request)

    def can_schedule_prefill(self) -> bool:
        return bool(self.pending and self._free_slots)

    def schedule_prefill(self) -> ScheduledOutput:
        if not self.can_schedule_prefill():
            raise RuntimeError("no pending request can be admitted")
        scheduled: list[ScheduledRequest] = []
        padded_length = 0
        while self.pending and self._free_slots:
            prompt_length = len(self.pending[0].prompt_token_ids)
            candidate_length = max(padded_length, prompt_length)
            candidate_work = (len(scheduled) + 1) * candidate_length
            if scheduled and candidate_work > self.prefill_token_budget:
                break
            state = self.pending.popleft()
            slot = heapq.heappop(self._free_slots)
            state.cache_slot = slot
            state.num_computed_tokens = prompt_length
            state.status = RequestStatus.PREFILLING
            self.active[slot] = state
            scheduled.append(
                ScheduledRequest(state, num_scheduled_tokens=prompt_length)
            )
            padded_length = candidate_length
        return ScheduledOutput(mode="prefill", requests=scheduled)

    def schedule_decode(self) -> ScheduledOutput:
        scheduled = []
        for slot in sorted(self.active):
            state = self.active[slot]
            if state.status is not RequestStatus.DECODING:
                continue
            state.num_computed_tokens += 1
            scheduled.append(ScheduledRequest(state, num_scheduled_tokens=1))
        if not scheduled:
            raise RuntimeError("no request is ready for decode")
        return ScheduledOutput(mode="decode", requests=scheduled)

    def update(
        self,
        schedule: ScheduledOutput,
        token_ids: list[int],
    ) -> tuple[int, ...]:
        if len(token_ids) != len(schedule.requests):
            raise ValueError("token_ids must match the scheduled batch")
        completed_slots: list[int] = []
        for scheduled, token_id in zip(schedule.requests, token_ids, strict=True):
            state = scheduled.request
            if state.status not in (
                RequestStatus.PREFILLING,
                RequestStatus.DECODING,
            ):
                raise RuntimeError("scheduled request is not runnable")
            state.output_token_ids.append(token_id)
            stop_token_ids = (
                self.default_stop_token_ids
                if state.request.stop_token_ids is None
                else state.request.stop_token_ids
            )
            if token_id in stop_token_ids:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "stop"
            elif len(state.output_token_ids) >= state.request.max_new_tokens:
                state.status = RequestStatus.COMPLETED
                state.finish_reason = "length"
            else:
                state.status = RequestStatus.DECODING

            if state.status is RequestStatus.COMPLETED:
                if state.cache_slot is None:
                    raise RuntimeError("completed request has no cache slot")
                slot = state.cache_slot
                del self.active[slot]
                heapq.heappush(self._free_slots, slot)
                state.cache_slot = None
                completed_slots.append(slot)
        return tuple(completed_slots)

    def has_active_requests(self) -> bool:
        return bool(self.active)

    def has_unfinished_requests(self) -> bool:
        return bool(self.pending or self.active)
