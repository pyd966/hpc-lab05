"""Profile the complete public request queue with continuous batching."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from hpc101_infer import EngineConfig, GenerationRequest, InferenceEngine, Runner
from run_generation_queue import parse_request


def load_requests(path: Path) -> list[GenerationRequest]:
    requests = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            requests.append(parse_request(json.loads(line), default_max_new_tokens=32))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    requests = load_requests(args.input)
    config = replace(
        EngineConfig.from_yaml(args.config),
        max_batch_size=len(requests),
        scheduler_batch_size=len(requests),
        scheduler_backend="continuous",
        synchronize_metrics=True,
    )
    engine = InferenceEngine.from_pretrained(args.model, config)
    runner = Runner(engine)

    # Compile all prefill shapes and one decode iteration outside the profile.
    warmup_requests = [replace(request, max_new_tokens=2) for request in requests]
    runner.run(warmup_requests)
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
        started = perf_counter()
        outputs = runner.run(requests)
        torch.cuda.synchronize()
        wall_time = perf_counter() - started

    generated_tokens = sum(output.generated_tokens for output in outputs)
    summary = {
        "requests": len(outputs),
        "generated_tokens": generated_tokens,
        "wall_time_s": wall_time,
        "generated_tokens_per_s": generated_tokens / wall_time,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print("PROFILE_SUMMARY " + json.dumps(summary))
    print(
        profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
    )


if __name__ == "__main__":
    main()
