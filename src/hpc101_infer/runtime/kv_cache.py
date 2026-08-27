"""全局 KV 与滑动窗口 Ring KV cache。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from hpc101_infer.models.config import Gemma4TextConfig


@dataclass
class LayerKVCache:
    """单层 attention 的 K/V 存储。

    全局层按绝对位置保存完整前缀；滑动窗口层使用固定容量的环形缓冲区。
    tensor 布局为 ``[max_batch, kv_heads, capacity, head_dim]``。
    """

    key: torch.Tensor
    value: torch.Tensor
    lengths: torch.Tensor
    max_batch_size: int
    max_sequence_length: int
    ring: bool = False
    window_size: int | None = None
    batch_size: int = 0

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def reset(self, batch_size: int) -> None:
        """开始新的静态 batch；旧 tensor 不清零，只重置有效长度。"""
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(
                f"batch_size must be in [1, {self.max_batch_size}], "
                f"got {batch_size}"
            )
        self.batch_size = batch_size
        self.lengths.zero_()

    def write(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> None:
        """按绝对 token 位置写入 K/V，Ring 层使用模运算映射物理槽位。"""
        batch_size, query_length = positions.shape
        if batch_size != self.batch_size:
            raise ValueError(
                f"positions batch size {batch_size} does not match cache batch "
                f"size {self.batch_size}"
            )
        expected_prefix = (batch_size, self.key.shape[1])
        expected_suffix = (query_length, self.key.shape[3])
        if key.shape != expected_prefix + expected_suffix:
            raise ValueError(
                f"invalid key shape {tuple(key.shape)}, expected "
                f"{expected_prefix + expected_suffix}"
            )
        if value.shape != key.shape:
            raise ValueError(
                f"invalid value shape {tuple(value.shape)}, expected {tuple(key.shape)}"
            )
        if positions.numel() and bool((positions < 0).any()):
            raise ValueError("KV cache positions must be non-negative")
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths.to(device=positions.device)
            if sequence_lengths.shape != (batch_size,):
                raise ValueError(
                    f"sequence_lengths must have shape ({batch_size},), "
                    f"got {tuple(sequence_lengths.shape)}"
                )
        if self.ring:
            targets = positions.remainder(self.capacity)
        else:
            if positions.numel() and bool((positions >= self.capacity).any()):
                raise ValueError(
                    f"position exceeds non-ring cache capacity {self.capacity}"
                )
            targets = positions
        for batch_idx in range(batch_size):
            source_indices = torch.arange(query_length, device=positions.device)
            if sequence_lengths is not None:
                source_indices = source_indices[
                    positions[batch_idx] < sequence_lengths[batch_idx]
                ]
            # Only the newest window can be observed by a sliding layer. Besides
            # saving work, this avoids duplicate physical indices in long prefill.
            if self.ring and source_indices.numel() > self.capacity:
                source_indices = source_indices[-self.capacity :]
            target = targets[batch_idx].index_select(0, source_indices)
            if target.numel() == 0:
                continue
            if self.ring and torch.unique(target).numel() != target.numel():
                # This is uncommon (non-monotonic external positions); preserve
                # last-write-wins semantics explicitly instead of relying on CUDA
                # duplicate-index behavior.
                for source_index in source_indices.tolist():
                    slot = targets[batch_idx, source_index]
                    self.key[batch_idx, :, slot, :] = key[
                        batch_idx, :, source_index, :
                    ]
                    self.value[batch_idx, :, slot, :] = value[
                        batch_idx, :, source_index, :
                    ]
                continue
            self.key[batch_idx].index_copy_(
                1, target, key[batch_idx].index_select(1, source_indices)
            )
            self.value[batch_idx].index_copy_(
                1, target, value[batch_idx].index_select(1, source_indices)
            )

    def view(
        self,
        max_length: int,
        sequence_lengths: torch.Tensor | None = None,
    ) -> "LayerKVView":
        """按绝对位置顺序暴露当前 cache 的可见逻辑前缀。"""
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        batch_size = self.batch_size
        if sequence_lengths is None:
            sequence_lengths = self.lengths[:batch_size]
        else:
            sequence_lengths = sequence_lengths.to(device=self.lengths.device)
        if sequence_lengths.shape != (batch_size,):
            raise ValueError(
                f"sequence_lengths must have shape ({batch_size},), "
                f"got {tuple(sequence_lengths.shape)}"
            )
        if sequence_lengths.numel() and bool((sequence_lengths < 0).any()):
            raise ValueError("sequence lengths must be non-negative")
        if sequence_lengths.numel() and bool(
            (sequence_lengths > self.max_sequence_length).any()
        ):
            raise ValueError("sequence length exceeds cache max_sequence_length")
        if max_length < int(sequence_lengths.max().item() or 0):
            raise ValueError("max_length must cover every sequence length")

        key_length = min(max_length, self.capacity) if self.ring else max_length
        key_positions = torch.arange(
            key_length,
            device=self.key.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        if not self.ring:
            return LayerKVView(
                key=self.key[:batch_size, :, :key_length, :],
                value=self.value[:batch_size, :, :key_length, :],
                key_positions=key_positions,
            )

        logical_start = (sequence_lengths - key_length).clamp_min(0)
        key_positions = logical_start[:, None] + key_positions
        # Before the first wrap the physical layout is already logical order.
        if key_length == 0:
            raise RuntimeError("Ring KV view unexpectedly has zero capacity")
        if bool((logical_start == 0).all()):
            return LayerKVView(
                key=self.key[:batch_size, :, :key_length, :],
                value=self.value[:batch_size, :, :key_length, :],
                key_positions=key_positions,
            )

        physical = key_positions.remainder(self.capacity)
        gather_index = physical[:, None, :, None].expand(
            batch_size,
            self.key.shape[1],
            key_length,
            self.key.shape[3],
        )
        return LayerKVView(
            key=self.key[:batch_size].gather(2, gather_index),
            value=self.value[:batch_size].gather(2, gather_index),
            key_positions=key_positions,
        )

    def release(self, batch_indices: Iterable[int]) -> None:
        """连续布局无法释放物理空间，只清除对应 slot 的有效长度。"""
        for batch_idx in batch_indices:
            if not 0 <= batch_idx < self.batch_size:
                raise IndexError("batch index is outside the active cache")
            self.lengths[batch_idx] = 0

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        """在一次模型 forward 完成后提交新的有效序列长度。"""
        if sequence_lengths.shape != (self.batch_size,):
            raise ValueError(
                f"sequence_lengths must have shape ({self.batch_size},), "
                f"got {tuple(sequence_lengths.shape)}"
            )
        self.lengths[: self.batch_size].copy_(sequence_lengths)


@dataclass
class PagedLayerKVCache:
    """按 token block 分页的单层 KV cache。

    ``key``/``value`` 的布局为 ``[physical_blocks, block_size, kv_heads, head_dim]``，
    ``block_table`` 将每个 batch slot 的逻辑 block 映射到物理 block。Ring 模式
    只在 sliding layer 使用，并为未对齐的窗口额外保留一个 page。
    """

    key: torch.Tensor
    value: torch.Tensor
    lengths: torch.Tensor
    block_table: torch.Tensor
    max_batch_size: int
    max_sequence_length: int
    block_size: int
    max_blocks_per_sequence: int
    ring: bool = False
    window_size: int | None = None
    batch_size: int = 0

    def __post_init__(self) -> None:
        if self.block_table.shape != (
            self.max_batch_size,
            self.max_blocks_per_sequence,
        ):
            raise ValueError("block_table has an invalid shape")
        self._free_blocks: list[int] = []
        self._slot_logical: list[list[int]] = []
        self._slot_physical: list[list[int]] = []
        self._reset_allocator()

    @property
    def capacity(self) -> int:
        return self.max_blocks_per_sequence * self.block_size

    @property
    def pool_blocks(self) -> int:
        return self.key.shape[0]

    @property
    def active_block_count(self) -> int:
        return sum(
            physical >= 0
            for slots in self._slot_physical
            for physical in slots
        )

    def _reset_allocator(self) -> None:
        self._free_blocks = list(reversed(range(self.pool_blocks)))
        self._slot_logical = [
            [-1] * self.max_blocks_per_sequence
            for _ in range(self.max_batch_size)
        ]
        self._slot_physical = [
            [-1] * self.max_blocks_per_sequence
            for _ in range(self.max_batch_size)
        ]

    def reset(self, batch_size: int) -> None:
        """重置 block table，并把所有物理 block 放回空闲池。"""
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError(
                f"batch_size must be in [1, {self.max_batch_size}], "
                f"got {batch_size}"
            )
        self.batch_size = batch_size
        self.lengths.zero_()
        self.block_table.fill_(-1)
        self._reset_allocator()

    def _ensure_block(self, batch_idx: int, logical_block: int) -> None:
        if logical_block < 0:
            raise ValueError("logical KV block must be non-negative")
        if self.ring:
            slot = logical_block % self.max_blocks_per_sequence
        else:
            if logical_block >= self.max_blocks_per_sequence:
                raise ValueError("logical KV block exceeds cache capacity")
            slot = logical_block
        if self._slot_logical[batch_idx][slot] == logical_block:
            return
        old_physical = self._slot_physical[batch_idx][slot]
        if old_physical >= 0:
            self._free_blocks.append(old_physical)
        if not self._free_blocks:
            raise RuntimeError("paged KV cache block pool is exhausted")
        physical = self._free_blocks.pop()
        self._slot_logical[batch_idx][slot] = logical_block
        self._slot_physical[batch_idx][slot] = physical
        self.block_table[batch_idx, slot] = physical

    def write(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> None:
        """将 K/V 写入逻辑 block 对应的物理 page。"""
        batch_size, query_length = positions.shape
        if batch_size != self.batch_size:
            raise ValueError(
                f"positions batch size {batch_size} does not match cache batch "
                f"size {self.batch_size}"
            )
        expected = (batch_size, self.key.shape[2], query_length, self.key.shape[3])
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                f"invalid key/value shape, expected {expected}, got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if positions.numel() and bool((positions < 0).any()):
            raise ValueError("KV cache positions must be non-negative")
        if sequence_lengths is not None:
            sequence_lengths = sequence_lengths.to(device=positions.device)
            if sequence_lengths.shape != (batch_size,):
                raise ValueError(
                    f"sequence_lengths must have shape ({batch_size},), "
                    f"got {tuple(sequence_lengths.shape)}"
                )
        flat_shape = (
            self.pool_blocks * self.block_size,
            self.key.shape[2],
            self.key.shape[3],
        )
        flat_key = self.key.view(flat_shape)
        flat_value = self.value.view(flat_shape)
        for batch_idx in range(batch_size):
            source_indices = torch.arange(query_length, device=positions.device)
            if sequence_lengths is not None:
                source_indices = source_indices[
                    positions[batch_idx] < sequence_lengths[batch_idx]
                ]
            if self.ring and sequence_lengths is not None:
                keep_start = max(
                    int(sequence_lengths[batch_idx].item()) - self.capacity,
                    0,
                )
                source_indices = source_indices[
                    positions[batch_idx].index_select(0, source_indices) >= keep_start
                ]
            if source_indices.numel() == 0:
                continue
            logical_positions = positions[batch_idx].index_select(0, source_indices)
            logical_blocks = logical_positions // self.block_size
            for logical_block in torch.unique(logical_blocks).tolist():
                self._ensure_block(batch_idx, int(logical_block))
            page_slots = logical_blocks
            if self.ring:
                page_slots = page_slots.remainder(self.max_blocks_per_sequence)
            physical_blocks = self.block_table[batch_idx].index_select(0, page_slots)
            flat_positions = (
                physical_blocks * self.block_size
                + logical_positions.remainder(self.block_size)
            )
            source_key = key[batch_idx].index_select(1, source_indices).transpose(0, 1)
            source_value = value[batch_idx].index_select(1, source_indices).transpose(0, 1)
            flat_key.index_copy_(0, flat_positions, source_key)
            flat_value.index_copy_(0, flat_positions, source_value)

    def view(
        self,
        max_length: int,
        sequence_lengths: torch.Tensor | None = None,
    ) -> "LayerKVView":
        """按逻辑 token 顺序收集 block，供现有 eager attention 使用。"""
        if max_length <= 0 or max_length > self.max_sequence_length:
            raise ValueError("max_length exceeds paged cache capacity")
        batch_size = self.batch_size
        if sequence_lengths is None:
            sequence_lengths = self.lengths[:batch_size]
        else:
            sequence_lengths = sequence_lengths.to(device=self.lengths.device)
        if sequence_lengths.shape != (batch_size,):
            raise ValueError(
                f"sequence_lengths must have shape ({batch_size},), "
                f"got {tuple(sequence_lengths.shape)}"
            )
        if sequence_lengths.numel() and bool((sequence_lengths < 0).any()):
            raise ValueError("sequence lengths must be non-negative")
        if sequence_lengths.numel() and bool(
            (sequence_lengths > self.max_sequence_length).any()
        ):
            raise ValueError("sequence length exceeds cache max_sequence_length")
        if max_length < int(sequence_lengths.max().item() or 0):
            raise ValueError("max_length must cover every sequence length")

        key_length = (
            min(max_length, self.window_size)
            if self.ring and self.window_size is not None
            else max_length
        )
        offsets = torch.arange(
            key_length,
            device=self.key.device,
            dtype=torch.long,
        )
        if self.ring:
            logical_positions = (
                sequence_lengths - key_length
            ).clamp_min(0)[:, None] + offsets
            page_slots = (logical_positions // self.block_size).remainder(
                self.max_blocks_per_sequence
            )
        else:
            logical_positions = offsets.expand(batch_size, -1)
            page_slots = logical_positions // self.block_size
        physical_blocks = self.block_table.gather(1, page_slots).clamp_min(0)
        flat_positions = (
            physical_blocks * self.block_size
            + logical_positions.remainder(self.block_size)
        )
        flat_shape = (
            self.pool_blocks * self.block_size,
            self.key.shape[2],
            self.key.shape[3],
        )
        flat_key = self.key.view(flat_shape)
        flat_value = self.value.view(flat_shape)
        gather_positions = flat_positions.reshape(-1)
        key = flat_key.index_select(0, gather_positions).view(
            batch_size, key_length, self.key.shape[2], self.key.shape[3]
        ).permute(0, 2, 1, 3)
        value = flat_value.index_select(0, gather_positions).view(
            batch_size, key_length, self.value.shape[2], self.value.shape[3]
        ).permute(0, 2, 1, 3)
        return LayerKVView(key=key, value=value, key_positions=logical_positions)

    def release(self, batch_indices: Iterable[int]) -> None:
        """立即回收完成请求占用的物理 block。"""
        for batch_idx in batch_indices:
            if not 0 <= batch_idx < self.batch_size:
                raise IndexError("batch index is outside the active cache")
            for slot, physical in enumerate(self._slot_physical[batch_idx]):
                if physical < 0:
                    continue
                self._free_blocks.append(physical)
                self._slot_physical[batch_idx][slot] = -1
                self._slot_logical[batch_idx][slot] = -1
            self.block_table[batch_idx].fill_(-1)
            self.lengths[batch_idx] = 0

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        if sequence_lengths.shape != (self.batch_size,):
            raise ValueError(
                f"sequence_lengths must have shape ({self.batch_size},), "
                f"got {tuple(sequence_lengths.shape)}"
            )
        self.lengths[: self.batch_size].copy_(sequence_lengths)


@dataclass(frozen=True)
class LayerKVView:
    key: torch.Tensor
    value: torch.Tensor
    key_positions: torch.Tensor | None = None


class KVCache:
    """管理全局 cache 与滑动窗口 Ring cache，并统一提交序列长度。"""

    def __init__(
        self,
        layers: list[LayerKVCache | PagedLayerKVCache],
        max_batch_size: int,
        max_sequence_length: int,
    ) -> None:
        if not layers:
            raise ValueError("KV cache must contain at least one layer")
        self.layers = layers
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length

    @classmethod
    def allocate(
        cls,
        config: Gemma4TextConfig,
        max_batch_size: int,
        max_sequence_length: int,
        dtype: torch.dtype,
        device: str | torch.device,
        *,
        ring_kv_cache: bool = True,
        paged_kv_cache: bool = True,
        paged_kv_block_size: int = 16,
    ) -> "KVCache":
        if max_batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("cache capacities must be positive")
        if paged_kv_block_size <= 1 or (
            paged_kv_block_size & (paged_kv_block_size - 1)
        ):
            raise ValueError(
                "paged_kv_block_size must be a power of two greater than 1"
            )
        device = torch.device(device)
        layers: list[LayerKVCache | PagedLayerKVCache] = []
        for layer_type in config.layer_types:
            if layer_type == "full_attention":
                kv_heads = config.num_global_key_value_heads
                head_dim = config.global_head_dim
                capacity = max_sequence_length
                ring = False
                window_size = None
            elif layer_type == "sliding_attention":
                kv_heads = config.num_key_value_heads
                head_dim = config.head_dim
                window_size = config.sliding_window
                if window_size <= 0:
                    raise ValueError("sliding_window must be positive")
                ring = ring_kv_cache
                capacity = min(max_sequence_length, window_size) if ring else max_sequence_length
            else:
                raise ValueError(f"unsupported attention type: {layer_type!r}")

            if paged_kv_cache:
                if ring:
                    # The retained window may start in the middle of a block,
                    # so a wrapped window can span one extra physical page.
                    if max_sequence_length > window_size:
                        max_blocks = (
                            window_size + 2 * paged_kv_block_size - 2
                        ) // paged_kv_block_size
                    else:
                        max_blocks = (
                            max_sequence_length + paged_kv_block_size - 1
                        ) // paged_kv_block_size
                else:
                    max_blocks = (
                        max_sequence_length + paged_kv_block_size - 1
                    ) // paged_kv_block_size
                pool_blocks = max_batch_size * max_blocks
                shape = (
                    pool_blocks,
                    paged_kv_block_size,
                    kv_heads,
                    head_dim,
                )
                layers.append(
                    PagedLayerKVCache(
                        key=torch.empty(shape, dtype=dtype, device=device),
                        value=torch.empty(shape, dtype=dtype, device=device),
                        lengths=torch.zeros(
                            max_batch_size, dtype=torch.long, device=device
                        ),
                        block_table=torch.full(
                            (max_batch_size, max_blocks),
                            -1,
                            dtype=torch.long,
                            device=device,
                        ),
                        max_batch_size=max_batch_size,
                        max_sequence_length=max_sequence_length,
                        block_size=paged_kv_block_size,
                        max_blocks_per_sequence=max_blocks,
                        ring=ring,
                        window_size=window_size,
                    )
                )
            else:
                shape = (max_batch_size, kv_heads, capacity, head_dim)
                layers.append(
                    LayerKVCache(
                        key=torch.empty(shape, dtype=dtype, device=device),
                        value=torch.empty(shape, dtype=dtype, device=device),
                        lengths=torch.zeros(
                            max_batch_size, dtype=torch.long, device=device
                        ),
                        max_batch_size=max_batch_size,
                        max_sequence_length=max_sequence_length,
                        ring=ring,
                        window_size=window_size,
                    )
                )
        return cls(layers, max_batch_size, max_sequence_length)

    @property
    def lengths(self) -> torch.Tensor:
        return self.layers[0].lengths

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.lengths.device

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].key.dtype

    @property
    def allocated_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            total += (
                layer.key.numel() * layer.key.element_size()
                + layer.value.numel() * layer.value.element_size()
                + layer.lengths.numel() * layer.lengths.element_size()
            )
            block_table = getattr(layer, "block_table", None)
            if block_table is not None:
                total += block_table.numel() * block_table.element_size()
        return total

    @property
    def active_block_count(self) -> int:
        return sum(
            getattr(layer, "active_block_count", 0) for layer in self.layers
        )

    @property
    def ring_layer_count(self) -> int:
        return sum(layer.ring for layer in self.layers)

    def reset(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset(batch_size)

    def write(
        self,
        layer_id: int,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_lengths: torch.Tensor | None = None,
    ) -> None:
        self.layers[layer_id].write(positions, key, value, sequence_lengths)

    def view(
        self,
        layer_id: int,
        max_length: int,
        sequence_lengths: torch.Tensor | None = None,
    ) -> LayerKVView:
        return self.layers[layer_id].view(max_length, sequence_lengths)

    def release(self, batch_indices: Iterable[int]) -> None:
        indices = tuple(batch_indices)
        for layer in self.layers:
            layer.release(indices)

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        for layer in self.layers:
            layer.commit(sequence_lengths)
