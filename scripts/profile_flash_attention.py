"""Profile the custom Triton Flash/Paged Attention in a real request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from hpc101_infer import EngineConfig, GenerationRequest, InferenceEngine, Runner


def load_request(path: str, max_new_tokens: int) -> GenerationRequest:
    record = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
    return GenerationRequest(
        prompt=record.get("prompt"),
        input_ids=record.get("input_ids"),
        max_new_tokens=min(
            int(record.get("max_new_tokens", max_new_tokens)), max_new_tokens
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    request = load_request(args.input, args.max_new_tokens)
    config = EngineConfig(
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=1,
        scheduler_batch_size=1,
        max_sequence_length=args.max_sequence_length,
        attention_backend="triton_flash",
        linear_backend="int4_triton",
        seed=42,
        synchronize_metrics=True,
        weight_offloading=True,
        weight_offloading_prefetch=True,
        weight_offloading_pin_memory=True,
        ring_kv_cache=True,
        paged_kv_cache=True,
        paged_kv_block_size=16,
    )
    engine = InferenceEngine.from_pretrained(args.model, config)
    runner = Runner(engine)
    runner.run(
        [
            GenerationRequest(
                prompt=request.prompt,
                input_ids=request.input_ids,
                max_new_tokens=min(2, request.max_new_tokens),
            )
        ]
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profile:
        start = perf_counter()
        output = runner.run([request])[0]
        torch.cuda.synchronize()
        wall_time = perf_counter() - start

    metrics = output.metrics
    print(
        "E2E attention_backend=triton_flash "
        f"prompt_tokens={output.prompt_tokens} "
        f"generated_tokens={output.generated_tokens} "
        f"ttft={metrics.ttft_s:.6f} "
        f"mean_tpot={metrics.mean_tpot_s:.6f} "
        f"total_latency={metrics.total_latency_s:.6f} "
        f"wall_time={wall_time:.6f} "
        f"peak_allocated={torch.cuda.max_memory_allocated()} "
        f"peak_reserved={torch.cuda.max_memory_reserved()}"
    )
    print(
        profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=30,
        )
    )


if __name__ == "__main__":
    main()
