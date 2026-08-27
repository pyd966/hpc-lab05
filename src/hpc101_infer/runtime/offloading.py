"""异步逐层权重 Offloading。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class _TensorSlot:
    owner: nn.Module
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    parameter: bool
    requires_grad: bool


@dataclass(frozen=True)
class _LayerPayload:
    host: torch.Tensor
    slots: tuple[_TensorSlot, ...]


class AsyncLayerOffloader:
    """用两个可复用 GPU buffer 异步搬运逐层权重。

    每层的参数和 buffer 先合并到一个连续的 CPU byte payload。传输流在
    计算当前层时预取下一层；计算流通过 ready event 等待 H2D 完成。推理
    权重是只读的，因此释放层时只切回 CPU 视图，不再做无意义的 D2H 回拷。
    """

    _BUFFER_SLOTS = 2

    def __init__(
        self,
        layers: list[nn.Module] | tuple[nn.Module, ...],
        device: str | torch.device,
        *,
        pin_memory: bool = True,
        tensor_filter: Callable[[nn.Module, str, torch.Tensor], bool] | None = None,
    ) -> None:
        self.layers = tuple(layers)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("async weight offloading requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not self.layers:
            raise ValueError("at least one layer is required")

        self._non_blocking = pin_memory
        self._payloads = [
            self._pack_layer(
                layer,
                pin_memory=pin_memory,
                tensor_filter=tensor_filter,
            )
            for layer in self.layers
        ]
        self._buffer_bytes = max(payload.host.numel() for payload in self._payloads)
        with torch.cuda.device(self.device):
            self._device_buffers = [
                torch.empty(
                    self._buffer_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                )
                for _ in range(self._BUFFER_SLOTS)
            ]
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self._ready_events = [torch.cuda.Event() for _ in self.layers]

        self._resident = [False] * len(self.layers)
        self._layer_buffer: list[int | None] = [None] * len(self.layers)
        self._buffer_layer: list[int | None] = [None] * self._BUFFER_SLOTS
        self._buffer_reuse_events: list[torch.cuda.Event | None] = [
            None
        ] * self._BUFFER_SLOTS
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0
        self._prefetch_calls = 0

    @classmethod
    def _pack_layer(
        cls,
        layer: nn.Module,
        *,
        pin_memory: bool,
        tensor_filter: Callable[[nn.Module, str, torch.Tensor], bool] | None,
    ) -> _LayerPayload:
        """将一层的所有 tensor 合并为一个按字节寻址的连续 payload。"""
        entries: list[tuple[nn.Module, str, torch.Tensor, bool, bool]] = []
        for owner in layer.modules():
            for name, parameter in tuple(owner._parameters.items()):
                if parameter is None:
                    continue
                if tensor_filter is not None and not tensor_filter(
                    owner, name, parameter
                ):
                    continue
                if parameter.device.type != "cpu":
                    raise ValueError(
                        "offloaded layer parameters must start on CPU, "
                        f"got {parameter.device} for {name}"
                    )
                entries.append(
                    (owner, name, parameter, True, parameter.requires_grad)
                )
            for name, buffer in tuple(owner._buffers.items()):
                if buffer is None:
                    continue
                if tensor_filter is not None and not tensor_filter(
                    owner, name, buffer
                ):
                    continue
                if buffer.device.type != "cpu":
                    raise ValueError(
                        "offloaded layer buffers must start on CPU, "
                        f"got {buffer.device} for {name}"
                    )
                entries.append((owner, name, buffer, False, False))

        if not entries:
            raise ValueError("each offloaded layer must contain tensors")

        offset = 0
        descriptors: list[_TensorSlot] = []
        for owner, name, tensor, parameter, requires_grad in entries:
            itemsize = tensor.element_size()
            offset = (offset + itemsize - 1) // itemsize * itemsize
            nbytes = tensor.numel() * itemsize
            descriptors.append(
                _TensorSlot(
                    owner=owner,
                    name=name,
                    offset=offset,
                    nbytes=nbytes,
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    parameter=parameter,
                    requires_grad=requires_grad,
                )
            )
            offset += nbytes

        host = torch.empty(offset, dtype=torch.uint8, pin_memory=pin_memory)
        for descriptor, (_, _, tensor, _, _) in zip(
            descriptors, entries, strict=True
        ):
            source = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            host[
                descriptor.offset : descriptor.offset + descriptor.nbytes
            ].copy_(source)

        for descriptor in descriptors:
            cls._assign_tensor(descriptor, host)
        return _LayerPayload(host=host, slots=tuple(descriptors))

    @staticmethod
    def _view(storage: torch.Tensor, descriptor: _TensorSlot) -> torch.Tensor:
        byte_view = storage.narrow(
            0,
            descriptor.offset,
            descriptor.nbytes,
        )
        return byte_view.view(descriptor.dtype).reshape(descriptor.shape)

    @classmethod
    def _assign_tensor(
        cls,
        descriptor: _TensorSlot,
        storage: torch.Tensor,
    ) -> None:
        tensor = cls._view(storage, descriptor)
        if descriptor.parameter:
            descriptor.owner._parameters[descriptor.name] = nn.Parameter(
                tensor,
                requires_grad=descriptor.requires_grad,
            )
        else:
            descriptor.owner._buffers[descriptor.name] = tensor

    def _choose_buffer(self) -> int:
        for buffer_index, layer_index in enumerate(self._buffer_layer):
            if layer_index is None:
                return buffer_index
        raise RuntimeError(
            "both GPU weight buffers are resident; release a layer before "
            "prefetching another one"
        )

    def prefetch(self, layer_index: int) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(layer_index)
        if self._resident[layer_index]:
            return

        buffer_index = self._choose_buffer()
        payload = self._payloads[layer_index]
        buffer = self._device_buffers[buffer_index]
        with torch.cuda.stream(self.transfer_stream):
            reuse_event = self._buffer_reuse_events[buffer_index]
            if reuse_event is not None:
                self.transfer_stream.wait_event(reuse_event)
            buffer[: payload.host.numel()].copy_(
                payload.host,
                non_blocking=self._non_blocking,
            )
            for descriptor in payload.slots:
                self._assign_tensor(descriptor, buffer)
            self._ready_events[layer_index].record(self.transfer_stream)

        self._resident[layer_index] = True
        self._layer_buffer[layer_index] = buffer_index
        self._buffer_layer[buffer_index] = layer_index
        self._prefetch_calls += 1
        self.host_to_device_bytes += payload.host.numel()

    def wait(self, layer_index: int) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(layer_index)
        if not self._resident[layer_index]:
            raise RuntimeError(f"layer {layer_index} was not prefetched")
        torch.cuda.current_stream(self.device).wait_event(
            self._ready_events[layer_index]
        )

    def release(self, layer_index: int) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(layer_index)
        if not self._resident[layer_index]:
            return

        buffer_index = self._layer_buffer[layer_index]
        if buffer_index is None:
            raise RuntimeError(f"layer {layer_index} has no GPU buffer")
        # 所有前向权重都是只读的；记录计算完成后即可安全复用 buffer，
        # 不需要把相同数据再拷回 CPU。
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream(self.device))
        for descriptor in self._payloads[layer_index].slots:
            self._assign_tensor(descriptor, self._payloads[layer_index].host)

        self._buffer_reuse_events[buffer_index] = compute_done
        self._buffer_layer[buffer_index] = None
        self._layer_buffer[layer_index] = None
        self._resident[layer_index] = False

    @property
    def resident_layers(self) -> tuple[int, ...]:
        return tuple(
            index for index, resident in enumerate(self._resident) if resident
        )

    def stats(self) -> dict[str, int]:
        return {
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "prefetch_calls": self._prefetch_calls,
            "prefetched_layers": len(self.layers),
            "merged_payload_bytes": sum(
                payload.host.numel() for payload in self._payloads
            ),
            "gpu_buffer_slots": self._BUFFER_SLOTS,
            "gpu_buffer_bytes": self._buffer_bytes * self._BUFFER_SLOTS,
        }
