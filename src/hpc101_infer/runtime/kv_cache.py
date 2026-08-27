"""全局 KV 与滑动窗口 Ring KV cache。"""

from __future__ import annotations

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

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        """在一次模型 forward 完成后提交新的有效序列长度。"""
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
        layers: list[LayerKVCache],
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
    ) -> "KVCache":
        if max_batch_size <= 0 or max_sequence_length <= 0:
            raise ValueError("cache capacities must be positive")
        device = torch.device(device)
        layers = []
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
        return sum(
            layer.key.numel() * layer.key.element_size()
            + layer.value.numel() * layer.value.element_size()
            + layer.lengths.numel() * layer.lengths.element_size()
            for layer in self.layers
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

    def commit(self, sequence_lengths: torch.Tensor) -> None:
        for layer in self.layers:
            layer.commit(sequence_lengths)
