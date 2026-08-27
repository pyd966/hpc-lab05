from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from hpc101_infer.quantization.checkpoint import (
    load_quantization_cache,
    quantization_cache_key,
    write_quantization_cache,
)
from hpc101_infer.quantization.config import QuantizationConfig
from hpc101_infer.quantization.types import QuantizedModuleManifest
import hpc101_infer.quantization.pipeline as pipeline_module


def _write_quantized_output(output_dir: Path) -> dict[str, QuantizedModuleManifest]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qweight_key = "layers.0.q_proj.qweight"
    scales_key = "layers.0.q_proj.scales"
    save_file(
        {
            qweight_key: torch.zeros((1, 1), dtype=torch.uint8),
            scales_key: torch.ones((1, 1), dtype=torch.float16),
        },
        output_dir / "model-00001-of-00001.safetensors",
    )
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    qweight_key: "model-00001-of-00001.safetensors",
                    scales_key: "model-00001-of-00001.safetensors",
                },
            }
        )
    )
    (output_dir / "config.json").write_text("{}\n")
    (output_dir / "quantization_config.json").write_text(
        json.dumps(QuantizationConfig().to_dict()) + "\n"
    )
    manifest = {
        "layers.0.q_proj": QuantizedModuleManifest(
            original_shape=(1, 1),
            padded_shape=(1, 1),
            qweight_key=qweight_key,
            scales_key=scales_key,
            zeros_key=None,
            bits=4,
            group_size=128,
            symmetric=True,
            packing="uint8_little_nibble",
        )
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "method": {"name": "rtn", "version": "1"},
                "modules": {name: value.to_dict() for name, value in manifest.items()},
            }
        )
        + "\n"
    )
    return manifest


def _key(source_dir: Path, config: QuantizationConfig, ids: torch.Tensor) -> str:
    return quantization_cache_key(
        source_dir,
        config,
        calibration_input_ids=ids,
        calibration_micro_batch_size=1,
        max_calibration_tokens=4096,
        max_shard_size_bytes=1 << 30,
    )


def test_cache_round_trip_and_invalidation(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}\n")
    (source_dir / "model.safetensors").write_bytes(b"source")
    output_dir = tmp_path / "output"
    expected = _write_quantized_output(output_dir)
    ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
    config = QuantizationConfig()
    cache_key = _key(source_dir, config, ids)

    write_quantization_cache(output_dir, cache_key)
    loaded = load_quantization_cache(output_dir, cache_key)
    assert loaded == expected
    assert load_quantization_cache(output_dir, _key(source_dir, QuantizationConfig(group_size=64), ids)) is None

    shard = output_dir / "model-00001-of-00001.safetensors"
    corrupted = bytearray(shard.read_bytes())
    corrupted[0] ^= 1
    shard.write_bytes(corrupted)
    assert load_quantization_cache(output_dir, cache_key) is None


def test_quantize_checkpoint_runs_once_for_matching_cache(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}\n")
    (source_dir / "model.safetensors").write_bytes(b"source")
    output_dir = tmp_path / "output"
    calls = 0
    expected = {
        "layers.0.q_proj": QuantizedModuleManifest(
            original_shape=(1, 1),
            padded_shape=(1, 1),
            qweight_key="layers.0.q_proj.qweight",
            scales_key="layers.0.q_proj.scales",
            zeros_key=None,
            bits=4,
            group_size=128,
            symmetric=True,
            packing="uint8_little_nibble",
        )
    }

    class FakePipeline:
        def __init__(self, _source, output, _config, **_kwargs):
            self.output = Path(output)

        def run(self):
            nonlocal calls
            calls += 1
            return _write_quantized_output(self.output)

    monkeypatch.setattr(pipeline_module, "QuantizationPipeline", FakePipeline)
    config = QuantizationConfig()
    first = pipeline_module.quantize_checkpoint(source_dir, output_dir, config)
    second = pipeline_module.quantize_checkpoint(source_dir, output_dir, config)
    assert first == expected
    assert second == expected
    assert calls == 1
