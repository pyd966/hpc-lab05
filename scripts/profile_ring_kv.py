"""远程验证 Ring KV cache 的容量、注意力正确性和 CUDA profile。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hpc101_infer import EngineConfig, GenerationRequest, InferenceEngine, Runner
from hpc101_infer.layers.attention import SlidingAttentionLayer
from hpc101_infer.models.config import Gemma4TextConfig
from hpc101_infer.runtime.kv_cache import KVCache


def print_cache_sizes(model_path: str, max_sequence_length: int) -> None:
    model_config = Gemma4TextConfig.from_pretrained(model_path)
    sizes: dict[bool, KVCache] = {}
    for ring in (False, True):
        cache = KVCache.allocate(
            model_config,
            max_batch_size=1,
            max_sequence_length=max_sequence_length,
            dtype=torch.bfloat16,
            device="cuda",
            ring_kv_cache=ring,
        )
        sizes[ring] = cache
        capacities = {}
        for layer in cache.layers:
            capacities[layer.capacity] = capacities.get(layer.capacity, 0) + 1
        print(
            f"CACHE ring={ring} bytes={cache.allocated_bytes} "
            f"gib={cache.allocated_bytes / 2**30:.4f} "
            f"ring_layers={cache.ring_layer_count} capacities={capacities}"
        )
    reduction = 1.0 - sizes[True].allocated_bytes / sizes[False].allocated_bytes
    print(f"CACHE_REDUCTION={reduction:.6%}")


def tiny_attention_correctness() -> None:
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        layer_types=("sliding_attention",),
        sliding_window=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=None,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
        },
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    rotary = config.get_rope_config("sliding_attention")
    ring_layer = SlidingAttentionLayer(
        4, 2, 4, 16, rotary, config.rms_norm_eps, 16, 4
    ).cuda()
    plain_layer = SlidingAttentionLayer(
        4, 2, 4, 16, rotary, config.rms_norm_eps, 16, 4
    ).cuda()
    plain_layer.load_state_dict(ring_layer.state_dict())
    ring_layer.rotary.materialize("cuda")
    plain_layer.rotary.materialize("cuda")
    hidden = torch.randn(1, 6, 16, device="cuda")
    positions = torch.arange(6, device="cuda").expand(1, -1)
    lengths = torch.tensor([6], device="cuda")
    ring_cache = KVCache.allocate(config, 1, 8, torch.float32, "cuda", ring_kv_cache=True)
    plain_cache = KVCache.allocate(config, 1, 8, torch.float32, "cuda", ring_kv_cache=False)
    ring_cache.reset(1)
    plain_cache.reset(1)
    ring_out = ring_layer(hidden, positions, lengths, 6, 0, ring_cache)
    plain_out = plain_layer(hidden, positions, lengths, 6, 0, plain_cache)
    ring_cache.commit(lengths)
    plain_cache.commit(lengths)
    prefill_diff = (ring_out[:, -1] - plain_out[:, -1]).abs().max().item()

    next_position = torch.tensor([[6]], device="cuda")
    next_lengths = torch.tensor([7], device="cuda")
    next_hidden = torch.randn(1, 1, 16, device="cuda")
    ring_decode = ring_layer(next_hidden, next_position, next_lengths, 7, 0, ring_cache)
    plain_decode = plain_layer(next_hidden, next_position, next_lengths, 7, 0, plain_cache)
    decode_diff = (ring_decode - plain_decode).abs().max().item()
    print(f"TINY_PREFILL_LAST_MAX_ABS_DIFF={prefill_diff:.8e}")
    print(f"TINY_DECODE_MAX_ABS_DIFF={decode_diff:.8e}")


def profile_request(
    model_path: str,
    input_path: str,
    max_sequence_length: int,
    *,
    ring_kv_cache: bool,
    collect_profile: bool,
) -> None:
    record = json.loads(Path(input_path).read_text(encoding="utf-8").splitlines()[0])
    request = GenerationRequest(
        prompt=record["prompt"],
        max_new_tokens=min(int(record.get("max_new_tokens", 32)), 32),
    )
    config = EngineConfig(
        dtype=torch.bfloat16,
        device="cuda",
        max_batch_size=1,
        scheduler_batch_size=1,
        max_sequence_length=max_sequence_length,
        linear_backend="int4_reference",
        seed=0,
        synchronize_metrics=True,
        weight_offloading=True,
        weight_offloading_prefetch=True,
        weight_offloading_pin_memory=True,
        ring_kv_cache=ring_kv_cache,
    )
    engine = InferenceEngine.from_pretrained(model_path, config)
    runner = Runner(engine)
    torch.cuda.reset_peak_memory_stats()
    if collect_profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as profile:
            outputs = runner.run([request])
    else:
        outputs = runner.run([request])
    output = outputs[0]
    label = "PROFILE" if collect_profile else "RUN"
    print(
        f"{label} ring={ring_kv_cache} generated={output.generated_tokens} "
        f"total_latency={output.metrics.total_latency_s:.6f} "
        f"peak_allocated={torch.cuda.max_memory_allocated()}"
    )
    if collect_profile:
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument(
        "--ring-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    print_cache_sizes(args.model, args.max_sequence_length)
    tiny_attention_correctness()
    if not args.skip_profile:
        profile_request(
            args.model,
            args.input,
            args.max_sequence_length,
            ring_kv_cache=args.ring_kv_cache,
            collect_profile=not args.no_profile,
        )


if __name__ == "__main__":
    main()
