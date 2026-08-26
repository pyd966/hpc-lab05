"""异步逐层权重 Offloading。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class _TensorSlot:
    owner: nn.Module
    name: str
    host: torch.Tensor
    parameter: bool
    requires_grad: bool


class AsyncLayerOffloader:
    """在独立 CUDA stream 上预取/回收 decoder layer 的权重。

    每次最多让当前层和下一层同时驻留 GPU。CPU 侧 tensor 使用 pinned
    memory，计算流通过 CUDA Event 等待 H2D 拷贝完成。
    """

    def __init__(
        self,
        layers: list[nn.Module] | tuple[nn.Module, ...],
        device: str | torch.device,
        *,
        pin_memory: bool = True,
    ) -> None:
        self.layers = tuple(layers)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("async weight offloading requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not self.layers:
            raise ValueError("at least one layer is required")

        self._slots: list[list[_TensorSlot]] = []
        self._resident = [False] * len(self.layers)
        self._non_blocking = pin_memory
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0

        for layer in self.layers:
            slots = self._collect_slots(layer, pin_memory=pin_memory)
            if not slots:
                raise ValueError("each offloaded layer must contain tensors")
            self._slots.append(slots)

        with torch.cuda.device(self.device):
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self._ready_events = [torch.cuda.Event() for _ in self.layers]

    @staticmethod
    def _collect_slots(
        layer: nn.Module,
        *,
        pin_memory: bool,
    ) -> list[_TensorSlot]:
        slots: list[_TensorSlot] = []
        for owner in layer.modules():
            for name, parameter in tuple(owner._parameters.items()):
                if parameter is None:
                    continue
                if parameter.device.type != "cpu":
                    raise ValueError(
                        "offloaded layer parameters must start on CPU, "
                        f"got {parameter.device} for {name}"
                    )
                if pin_memory and not parameter.is_pinned():
                    parameter.data = parameter.detach().pin_memory()
                slots.append(
                    _TensorSlot(
                        owner=owner,
                        name=name,
                        host=parameter,
                        parameter=True,
                        requires_grad=parameter.requires_grad,
                    )
                )
            for name, buffer in tuple(owner._buffers.items()):
                if buffer is None:
                    continue
                if buffer.device.type != "cpu":
                    raise ValueError(
                        "offloaded layer buffers must start on CPU, "
                        f"got {buffer.device} for {name}"
                    )
                if pin_memory and not buffer.is_pinned():
                    buffer = buffer.pin_memory()
                    owner._buffers[name] = buffer
                slots.append(
                    _TensorSlot(
                        owner=owner,
                        name=name,
                        host=buffer,
                        parameter=False,
                        requires_grad=False,
                    )
                )
        return slots

    def _is_target_device(self, device: torch.device) -> bool:
        return device.type == self.device.type and (
            self.device.index is None or device.index == self.device.index
        )

    @staticmethod
    def _current(slot: _TensorSlot) -> torch.Tensor:
        if slot.parameter:
            tensor = slot.owner._parameters[slot.name]
        else:
            tensor = slot.owner._buffers[slot.name]
        if tensor is None:
            raise RuntimeError(f"offloaded tensor disappeared: {slot.name}")
        return tensor

    def _move_to_device(self, slot: _TensorSlot) -> None:
        current = self._current(slot)
        if self._is_target_device(current.device):
            return
        if current.device.type != "cpu":
            raise RuntimeError(
                f"cannot prefetch tensor from {current.device}; expected CPU"
            )
        moved = current.detach().to(
            device=self.device,
            non_blocking=self._non_blocking,
        )
        if slot.parameter:
            slot.owner._parameters[slot.name] = nn.Parameter(
                moved,
                requires_grad=slot.requires_grad,
            )
        else:
            slot.owner._buffers[slot.name] = moved
        self.host_to_device_bytes += moved.numel() * moved.element_size()

    def _move_to_host(self, slot: _TensorSlot) -> None:
        current = self._current(slot)
        if current.device.type == "cpu":
            return
        if not self._is_target_device(current.device):
            raise RuntimeError(
                f"cannot release tensor on {current.device}; expected {self.device}"
            )
        if slot.host.is_pinned():
            if slot.parameter:
                slot.host.data.copy_(current.detach(), non_blocking=True)
                slot.owner._parameters[slot.name] = slot.host
            else:
                slot.host.copy_(current.detach(), non_blocking=True)
                slot.owner._buffers[slot.name] = slot.host
            current.record_stream(self.transfer_stream)
        else:
            # Pageable CPU memory cannot be the destination of an asynchronous
            # D2H copy. Keep this configuration correct with a synchronous copy.
            moved = current.detach().to(device="cpu")
            if slot.parameter:
                slot.owner._parameters[slot.name] = nn.Parameter(
                    moved,
                    requires_grad=slot.requires_grad,
                )
            else:
                slot.owner._buffers[slot.name] = moved
        self.device_to_host_bytes += current.numel() * current.element_size()

    def prefetch(self, layer_index: int) -> None:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(layer_index)
        if self._resident[layer_index]:
            return
        with torch.cuda.stream(self.transfer_stream):
            for slot in self._slots[layer_index]:
                self._move_to_device(slot)
            self._ready_events[layer_index].record(self.transfer_stream)
        self._resident[layer_index] = True

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
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_event(compute_done)
            for slot in self._slots[layer_index]:
                self._move_to_host(slot)
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
            "prefetched_layers": len(self.layers),
        }
