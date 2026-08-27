# 长期量化缓存记录（2026-08-27）

## 固定路径

真实量化结果保存在共享 home 的长期目录：

```text
/home/h3250106394/quantized/gemma-4-12b-gptq-group64
```

该目录不是远程任务的 `/tmp`，后续任务和 DevPod 都可以通过这个绝对路径访问。目录当前约 `7.3 GiB`，包含 7 个最终 safetensors 分片、`manifest.json`、`model.safetensors.index.json`、`quantization_config.json` 和 `quantization_cache.json`。

## 生成方式

缓存由当前仓库的 `scripts/quantize.py`、`QuantizationPipeline` 和 GPTQ 实现生成，使用：

- 源模型：`/checkpoints/gemma-4-12b`
- 算法：GPTQ
- 权重量化：INT4，group size `64`
- scale dtype：`float16`
- 对称量化：开启
- 校准集：`datasets/calibration-256.jsonl`，limit `256`
- GPTQ block size：`128`
- damp percent：`0.01`
- 量化模块：`328` 个 Linear

远程构建任务为 `175438`，状态 `Succeeded`，退出码 `0`，耗时约 `18 分 8 秒`。构建过程中没有使用已有量化权重；生成的结果就是当前程序的 GPTQ 输出。

## 缓存验证

任务 `175588` 使用最终代码重建 sidecar，输出：

```text
CACHE_VALID True
CACHE_ARTIFACT_COUNT 11
```

任务 `175606` 使用完整 CLI、相同模型和相同输出路径再次执行：

```text
量化结果已准备好：328 个 Linear 模块，目录 /home/h3250106394/quantized/gemma-4-12b-gptq-group64
CACHE_REUSE_ELAPSED 11.737 s
```

这次执行没有重新进行 GPTQ，只完成缓存校验并复用已有结果。以后提交 offloading、Ring KV、Paged Attention 等推理优化时，统一使用上述路径即可，不需要重新量化。

只有修改 GPTQ 算法、量化配置或校准数据时，才需要使用 `--force` 重新生成。
