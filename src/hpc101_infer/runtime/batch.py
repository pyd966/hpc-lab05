from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    sequence_lengths: torch.Tensor
    mode: Literal["prefill", "decode"]
    curr_max_seq_len: int = 0
    cache_indices: torch.Tensor | None = None
    cache_slots: tuple[int, ...] | None = None
    cache_ranges: tuple[tuple[int, int], ...] | None = None
    past_max_seq_len: int = 0

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def query_length(self) -> int:
        return self.input_ids.shape[1]
