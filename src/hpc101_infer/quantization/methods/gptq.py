from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hpc101_infer.quantization.packing import pack_int4
from hpc101_infer.quantization.types import (
    LayerContext,
    LayerQuantizationResult,
    QuantizedWeight,
    SCALE_DTYPES,
)


@dataclass(frozen=True)
class GPTQOptions:
    block_size: int
    damp_percent: float


@dataclass(frozen=True)
class GPTQModuleState:
    activations: torch.Tensor
    activation_tokens: int


@dataclass(frozen=True)
class GPTQLayerState:
    modules: Mapping[str, GPTQModuleState]
    options: GPTQOptions


def _calibration_option(
    calibration: Mapping[str, Any],
    name: str,
    default: int | float,
) -> int | float:
    gptq = calibration.get("gptq", calibration)
    if not isinstance(gptq, Mapping):
        raise TypeError("config.calibration['gptq'] must be a mapping")
    return gptq.get(name, default)


def _parse_gptq_options(calibration: Mapping[str, Any]) -> GPTQOptions:
    block_size = _calibration_option(calibration, "block_size", 128)
    damp_percent = _calibration_option(calibration, "damp_percent", 0.01)

    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")

    gptq = calibration.get("gptq", calibration)
    if isinstance(gptq, Mapping) and gptq.get("desc_act", False):
        raise ValueError("GPTQ desc_act is not supported by this checkpoint format")
    return GPTQOptions(
        block_size=block_size,
        damp_percent=float(damp_percent),
    )


def quantize_weight_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    group_size: int,
    *,
    block_size: int = 128,
    damp_percent: float = 0.01,
    symmetric: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[QuantizedWeight, dict[str, float | int]]:
    """
    使用 GPTQ 算法将权重量化为 INT4。

    参数：
        weight: 待量化的权重张量，形状为 (out_features, in_features)。
        activations: 校准数据集的输入激活，形状为 (calibration_tokens, in_features)。
        group_size: 量化粒度，即每个 group 中的列数。
        block_size: 分块计算时每个 block 中的列数。
        damp_percent: 阻尼比例，用于改善 Hessian 的数值稳定性。
        symmetric: 是否使用对称量化。
        scale_dtype: 缩放因子的 dtype。

    返回：
        quantized_weight: 量化后的权重对象，具体详见 QuantizedWeight 类的定义。
        metadatas: 量化过程中的统计信息，不影响评测，用于分析和调试。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if activations.ndim != 2 or not activations.is_floating_point():
        raise ValueError("activations must be a floating-point matrix")
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation width does not match weight input size")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 < damp_percent < 1.0:
        raise ValueError("damp_percent must be in (0, 1)")
    if scale_dtype not in SCALE_DTYPES.values():
        raise ValueError("unsupported scale dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")
    if activations.shape[0] == 0:
        raise ValueError("activations must contain at least one token")
    if not torch.isfinite(activations).all():
        raise ValueError("activations must contain only finite values")

    out_features, in_features = weight.shape
    padded_in_features = math.ceil(in_features / group_size) * group_size
    num_groups = padded_in_features // group_size

    # GPTQ's Hessian is formed in FP32. Calibration activations are kept on
    # the host by the pipeline, so explicitly move them to the weight device.
    device = weight.device
    work = F.pad(
        weight.detach().to(device=device, dtype=torch.float32),
        (0, padded_in_features - in_features),
    )
    calibration = F.pad(
        activations.detach().to(device=device, dtype=torch.float32),
        (0, padded_in_features - in_features),
    )
    hessian = calibration.transpose(0, 1).matmul(calibration)
    hessian.div_(float(calibration.shape[0]))
    del calibration

    # GEMM accumulates H in FP32, so explicitly restore symmetry before
    # factorization. Zero-activation columns carry no information; give them
    # an identity entry so Cholesky remains well-defined.
    hessian = (hessian + hessian.transpose(0, 1)).mul_(0.5)
    diagonal = torch.diagonal(hessian)
    dead = diagonal <= 0
    dead_columns = int(dead.sum().item())
    if dead_columns:
        indices = torch.arange(padded_in_features, device=device)[dead]
        hessian[indices, indices] = 1.0
        work[:, dead] = 0.0

    diagonal = torch.diagonal(hessian)
    base_damping = float(damp_percent) * diagonal.mean()
    if not torch.isfinite(base_damping) or base_damping <= 0:
        base_damping = torch.tensor(
            torch.finfo(torch.float32).eps, device=device, dtype=torch.float32
        )
    diagonal.add_(base_damping)
    chol = None
    for retry in range(6):
        chol, info = torch.linalg.cholesky_ex(hessian, check_errors=False)
        if int(info.item()) == 0:
            break
        del chol
        diagonal.add_(base_damping * (10.0 ** (retry + 1)))
    else:
        raise RuntimeError("failed to Cholesky-factorize the damped Hessian")
    inverse = torch.cholesky_inverse(chol)
    del hessian, chol
    inverse = (inverse + inverse.transpose(0, 1)).mul_(0.5)
    inverse_diagonal = torch.diagonal(inverse)
    inverse_damping = torch.finfo(torch.float32).eps * inverse_diagonal.mean()
    if not torch.isfinite(inverse_damping) or inverse_damping <= 0:
        inverse_damping = torch.tensor(
            torch.finfo(torch.float32).eps, device=device, dtype=torch.float32
        )
    inverse_diagonal.add_(inverse_damping)
    hessian_factor = None
    for retry in range(6):
        # U is upper triangular and satisfies H^-1 = U^T U, matching the
        # column-wise GPTQ error propagation formula.
        hessian_factor, info = torch.linalg.cholesky_ex(
            inverse, upper=True, check_errors=False
        )
        if int(info.item()) == 0:
            break
        del hessian_factor
        inverse_diagonal.add_(inverse_damping * (10.0 ** (retry + 1)))
    else:
        raise RuntimeError("failed to factorize the inverse Hessian")
    del inverse

    scales_fp32 = torch.empty(
        out_features, num_groups, dtype=torch.float32, device=device
    )
    zeros = None
    if not symmetric:
        zeros = torch.empty(
            out_features, num_groups, dtype=torch.uint8, device=device
        )
    encoded = torch.empty(
        out_features, padded_in_features, dtype=torch.uint8, device=device
    )
    predicted_loss = torch.zeros((), dtype=torch.float64, device=device)

    def group_values(
        block: torch.Tensor,
        block_start: int,
        block_end: int,
        group_start: int,
        group_end: int,
    ) -> torch.Tensor:
        """Read a group's current (possibly cross-block) weight values."""
        parts = []
        if group_start < block_end:
            local_end = min(group_end, block_end)
            parts.append(block[:, group_start - block_start : local_end - block_start])
        if group_end > block_end:
            parts.append(work[:, block_end:group_end])
        return torch.cat(parts, dim=1)

    for block_start in range(0, padded_in_features, block_size):
        block_end = min(block_start + block_size, padded_in_features)
        block = work[:, block_start:block_end].clone()
        block_width = block_end - block_start
        block_errors = torch.empty_like(block)
        factor_block = hessian_factor[block_start:block_end, block_start:block_end]

        for offset in range(block_width):
            column = block_start + offset
            group_index = column // group_size
            if column % group_size == 0:
                group_end = min(column + group_size, padded_in_features)
                values = group_values(
                    block, block_start, block_end, column, group_end
                )
                if symmetric:
                    scale = values.abs().amax(dim=1) / 7.0
                else:
                    minimum = torch.minimum(
                        values.amin(dim=1), torch.zeros(out_features, device=device)
                    )
                    maximum = torch.maximum(
                        values.amax(dim=1), torch.zeros(out_features, device=device)
                    )
                    scale = (maximum - minimum) / 15.0
                scale = scale.clamp_min(torch.finfo(torch.float32).eps)
                scales_fp32[:, group_index] = scale
                if not symmetric:
                    zeros[:, group_index] = torch.round(
                        -values.amin(dim=1).clamp_max(0.0) / scale
                    ).clamp(0, 15).to(torch.uint8)

            scale = scales_fp32[:, group_index]
            column_values = block[:, offset]
            if symmetric:
                codes = torch.round(column_values / scale).clamp(-8, 7)
                reconstructed = codes * scale
                encoded[:, column] = (codes.to(torch.int16) + 8).to(torch.uint8)
            else:
                zero = zeros[:, group_index].to(torch.float32)
                codes = torch.round(column_values / scale + zero).clamp(0, 15)
                reconstructed = (codes - zero) * scale
                encoded[:, column] = codes.to(torch.uint8)

            diagonal_factor = factor_block[offset, offset]
            if not torch.isfinite(diagonal_factor) or diagonal_factor <= 0:
                raise RuntimeError("inverse Hessian has an invalid diagonal")
            error = column_values - reconstructed
            block_errors[:, offset] = error / diagonal_factor
            predicted_loss += (
                error.double().square() / diagonal_factor.double().square()
            ).sum()

            if offset + 1 < block_width:
                block[:, offset + 1 :] -= block_errors[:, offset : offset + 1].matmul(
                    factor_block[offset : offset + 1, offset + 1 :]
                )

        if block_end < padded_in_features:
            work[:, block_end:] -= block_errors.matmul(
                hessian_factor[block_start:block_end, block_end:]
            )

    del work, hessian_factor

    quantized = QuantizedWeight(
        qweight=pack_int4(encoded),
        scales=scales_fp32.to(scale_dtype),
        zeros=zeros,
        original_shape=(out_features, in_features),
        padded_shape=(out_features, padded_in_features),
        bits=4,
        group_size=group_size,
        symmetric=symmetric,
        packing="uint8_little_nibble",
    )
    metadata: dict[str, float | int] = {
        "activation_tokens": activations.shape[0],
        "block_size": block_size,
        "damp_percent": damp_percent,
        "dead_columns": dead_columns,
        "predicted_loss": float(predicted_loss.item()),
    }
    return quantized, metadata


class GPTQQuantizationMethod:
    name = "gptq"
    version = "1"

    def calibrate_layer(self, context: LayerContext) -> GPTQLayerState:
        if context.activations is None:
            raise ValueError("GPTQ requires calibration activations")

        options = _parse_gptq_options(context.config.calibration or {})
        modules = dict(context.layer.named_modules())
        states: dict[str, GPTQModuleState] = {}
        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            if name not in context.activations:
                raise ValueError(f"missing calibration activations for {name}")

            activations = context.activations[name]
            if activations.ndim != 2 or activations.shape[1] != module.in_features:
                raise ValueError(
                    f"invalid calibration activation shape for {name}: "
                    f"expected [tokens, {module.in_features}], "
                    f"got {tuple(activations.shape)}"
                )
            if activations.shape[0] == 0:
                raise ValueError(f"calibration activations for {name} are empty")
            if not activations.is_floating_point():
                raise TypeError(f"calibration activations for {name} must be floating")
            if not torch.isfinite(activations).all():
                raise ValueError(f"calibration activations for {name} are not finite")

            states[name] = GPTQModuleState(
                activations=activations.detach(),
                activation_tokens=activations.shape[0],
            )

        return GPTQLayerState(modules=states, options=options)

    def quantize_layer(
        self, context: LayerContext, state: GPTQLayerState
    ) -> LayerQuantizationResult:
        if not isinstance(state, GPTQLayerState):
            raise TypeError("state must be a GPTQLayerState")

        scale_dtype = SCALE_DTYPES[context.config.scale_dtype]
        modules = dict(context.layer.named_modules())
        weights: dict[str, QuantizedWeight] = {}
        module_metadata: dict[str, dict[str, float | int]] = {}

        for name in context.target_modules:
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"target is not a Linear module: {name}")
            module_state = state.modules.get(name)
            if module_state is None:
                raise ValueError(f"missing GPTQ calibration state for {name}")

            try:
                weights[name], module_metadata[name] = quantize_weight_gptq(
                    module.weight,
                    module_state.activations,
                    context.config.group_size,
                    block_size=state.options.block_size,
                    damp_percent=state.options.damp_percent,
                    symmetric=context.config.symmetric,
                    scale_dtype=scale_dtype,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"GPTQ failed for layer {context.layer_index} module {name} "
                    f"with {module_state.activation_tokens} calibration tokens: {error}"
                ) from error

        return LayerQuantizationResult(
            weights=weights,
            metadata={
                "gptq": {
                    "block_size": state.options.block_size,
                    "damp_percent": state.options.damp_percent,
                    "modules": module_metadata,
                }
            },
        )
