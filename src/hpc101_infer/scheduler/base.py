from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol

from hpc101_infer.types import GenerationRequest


class RequestStatus(Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RequestState:
    request: GenerationRequest
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.PENDING
    finish_reason: str = ""
    cache_slot: int | None = None


@dataclass
class ScheduledRequest:
    request: RequestState
    num_scheduled_tokens: int = 0


@dataclass
class ScheduledOutput:
    mode: Literal["prefill", "decode"]
    requests: list[ScheduledRequest]

    @property
    def batch_max_length(self) -> int:
        return max(scheduled.request.num_computed_tokens for scheduled in self.requests)

    @property
    def cache_slots(self) -> tuple[int, ...]:
        slots = tuple(scheduled.request.cache_slot for scheduled in self.requests)
        if any(slot is None for slot in slots):
            raise RuntimeError("scheduled request has no cache slot")
        return tuple(int(slot) for slot in slots)


class Scheduler(Protocol):
    def add_request(self, request: RequestState) -> None: ...

    def schedule(self) -> ScheduledOutput: ...

    def update(self, token_ids: list[int]) -> None: ...

    def has_unfinished_requests(self) -> bool: ...
