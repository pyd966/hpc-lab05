"""远程验证并 profile 自写 Triton W4A16 融合 kernel。"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

from hpc101_infer import EngineConfig, GenerationRequest, InferenceEngine, Runner
from hpc101_infer.kernels.w4a16 import w4a16_linear
from hpc101_infer.quantization.packing import dequantize_weight, pack_int4
from hpc101_infer.quantization.types import QuantizedWeight


def make_weight(n_size: int, k_size: int, group_size: int) -> QuantizedWeight:
    padded = (k_size + group_size - 1) // group_size * group_size
    codes = torch.randint(
        0, 16, (n_size, padded), device="cuda", dtype=torch.uint8
    )
    scales = (
        torch.rand(
            n_size,
            padded // group_size,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.02
        + 0.005
    )
    return QuantizedWeight(
        qweight=pack_int4(codes),
        scales=scales,
        zeros=None,
        original_shape=(n_size, k_size),
        padded_shape=(n_size, padded),
        bits=4,
        group_size=group_size,
        symmetric=True,
        packing="uint8_little_nibble",
    )


def fused_call(inputs: torch.Tensor, weight: QuantizedWeight) -> torch.Tensor:
    return w4a16_linear(
        inputs,
        weight.qweight,
        weight.scales,
        weight.zeros,
        None,
        in_features=weight.original_shape[1],
        out_features=weight.original_shape[0],
        padded_in_features=weight.padded_shape[1],
        group_size=weight.group_size,
    )


def reference_call(inputs: torch.Tensor, weight: QuantizedWeight) -> torch.Tensor:
    dequantized = dequantize_weight(weight, dtype=inputs.dtype)
    return torch.nn.functional.linear(inputs, dequantized)


def elapsed_ms(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def peak_temporary_bytes(operation: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return peak - baseline


def validate_and_benchmark() -> None:
    torch.manual_seed(0)
    correctness_cases = ((1, 96, 128), (3, 75, 130), (34, 160, 192))
    for m_size, n_size, k_size in correctness_cases:
        weight = make_weight(n_size, k_size, 64)
        inputs = torch.randn(
            m_size, k_size, device="cuda", dtype=torch.bfloat16
        )
        reference = reference_call(inputs, weight)
        fused = fused_call(inputs, weight)
        difference = (reference.float() - fused.float()).abs()
        print(
            f"CORRECTNESS M={m_size} N={n_size} K={k_size} "
            f"max_abs={difference.max().item():.8f} "
            f"mean_abs={difference.mean().item():.8f}"
        )

    benchmark_cases = (
        (1, 15360, 3840, 20, "decode_gate"),
        (1, 3840, 15360, 20, "decode_down"),
        (128, 15360, 3840, 5, "prefill_gate"),
    )
    for m_size, n_size, k_size, iterations, label in benchmark_cases:
        weight = make_weight(n_size, k_size, 64)
        inputs = torch.randn(
            m_size, k_size, device="cuda", dtype=torch.bfloat16
        )
        reference_ms = elapsed_ms(
            lambda: reference_call(inputs, weight), iterations
        )
        fused_ms = elapsed_ms(lambda: fused_call(inputs, weight), iterations)
        reference_peak = peak_temporary_bytes(
            lambda: reference_call(inputs, weight)
        )
        fused_peak = peak_temporary_bytes(lambda: fused_call(inputs, weight))
        print(
            f"MICROBENCH name={label} M={m_size} N={n_size} K={k_size} "
            f"reference_ms={reference_ms:.6f} fused_ms={fused_ms:.6f} "
            f"speedup={reference_ms / fused_ms:.4f} "
            f"reference_peak_bytes={reference_peak} "
            f"fused_peak_bytes={fused_peak}"
        )
        del inputs, weight


def load_request(input_path: str, max_new_tokens: int) -> GenerationRequest:
    record = json.loads(
        Path(input_path).read_text(encoding="utf-8").splitlines()[0]
    )
    return GenerationRequest(
        prompt=record.get("prompt"),
        input_ids=record.get("input_ids"),
        max_new_tokens=min(
            int(record.get("max_new_tokens", max_new_tokens)),
            max_new_tokens,
        ),
    )


def profile_request(
    model_path: str,
    input_path: str,
    max_sequence_length: int,
    max_new_tokens: int,
    *,
    linear_backend: str,
    collect_profile: bool,
) -> None:
    request = load_request(input_path, max_new_tokens)
    config = EngineConfig(
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=1,
        scheduler_batch_size=1,
        max_sequence_length=max_sequence_length,
        linear_backend=linear_backend,
        seed=0,
        synchronize_metrics=True,
        weight_offloading=True,
        weight_offloading_prefetch=True,
        weight_offloading_pin_memory=True,
        ring_kv_cache=True,
        paged_kv_cache=True,
        paged_kv_block_size=16,
    )
    engine = InferenceEngine.from_pretrained(model_path, config)
    runner = Runner(engine)

    warmup_request = GenerationRequest(
        prompt=request.prompt,
        input_ids=request.input_ids,
        max_new_tokens=min(2, request.max_new_tokens),
    )
    runner.run([warmup_request])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    profile = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        )
        if collect_profile
        else None
    )
    wall_start = perf_counter()
    with profile if profile is not None else nullcontext():
        outputs = runner.run([request])
    torch.cuda.synchronize()
    wall_time = perf_counter() - wall_start

    output = outputs[0]
    metrics = output.metrics
    print(
        f"E2E backend={linear_backend} prompt_tokens={output.prompt_tokens} "
        f"generated_tokens={output.generated_tokens} "
        f"ttft={metrics.ttft_s:.6f} "
        f"mean_tpot={metrics.mean_tpot_s:.6f} "
        f"total_latency={metrics.total_latency_s:.6f} "
        f"wall_time={wall_time:.6f} "
        f"peak_allocated={torch.cuda.max_memory_allocated()}"
    )
    if profile is not None:
        print(
            profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=25
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--input")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--linear-backend",
        choices=("int4_reference", "int4_triton"),
        default="int4_triton",
    )
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()

    validate_and_benchmark()
    if not args.skip_e2e:
        if args.model is None or args.input is None:
            parser.error(
                "--model and --input are required unless --skip-e2e is set"
            )
        profile_request(
            args.model,
            args.input,
            args.max_sequence_length,
            args.max_new_tokens,
            linear_backend=args.linear_backend,
            collect_profile=not args.no_profile,
        )


if __name__ == "__main__":
    main()
