# 异步权重 Offloading 报告

日期：2026-08-26
源代码版本：本轮提交前的工作区（最终提交见 Git 历史）
远程验证作业：`171932`（首次诊断，定位设备比较 bug）、`172187`（修复后正确性与 profile）

## 本轮目标

按照 `docs/Lab5-Gemma4/index.md` 的权重 Offloading 要求，把 DecoderLayer 的权重保留在 CPU，仅让当前层和预取的下一层驻留 GPU，并使用独立 CUDA stream 和 CUDA Event 隐藏 Host-to-Device 搬运。INT4 的 `qweight`、`scales`、`zeros` 作为同一个层的完整 tensor 集合搬运。

本轮只实现 Offloading，暂不实现 Ring KV cache 或 Paged Attention。

## 实现内容

- 新增 `AsyncLayerOffloader`（`src/hpc101_infer/runtime/offloading.py`）。
- CPU 权重在初始化时转为 pinned memory；H2D 使用独立 transfer stream 的 non-blocking copy。
- `prefetch(i)` 在 transfer stream 上搬运第 `i` 层并记录 ready event；计算流在使用前等待该 event。
- 计算第 `i` 层时预取第 `i+1` 层；当前层计算完成后记录 compute event，由 transfer stream 异步 D2H 回收当前层。
- 释放路径支持 pageable CPU memory 的同步回退，且规范化 `cuda`/`cuda:0` 设备比较。
- `EngineConfig` 增加 `weight_offloading`、`weight_offloading_prefetch` 和 `weight_offloading_pin_memory` 配置；`config.yaml` 已打开 pinned asynchronous offload。
- `load_gemma4` 在启用 Offloading 时先把模型权重加载到 CPU，embedding、最终 norm 和 KV cache 仍位于 GPU。
- 修复 `measure_operation` 在 `finally` 中返回值导致 CUDA OOM 被吞掉的问题，确保后续显存错误能直接暴露。

## 正确性与显存

验证使用 g64 GPTQ INT4 checkpoint、CUDA/bfloat16、`int4_reference` 后端、batch 1，输入 token 为 `[1, 2, 3, 4]`，`max_sequence_length=64`。先运行全量 GPU 权重版本作为参考，再运行 Offloading 版本：

| 模式 | 峰值 GPU allocated | 说明 |
| --- | ---: | --- |
| 无 Offloading | `8,252,107,776` B（约 7.69 GiB） | 所有 DecoderLayer 权重常驻 GPU |
| 异步 Offloading | `2,673,564,672` B（约 2.49 GiB） | embedding/norm、KV 和当前/下一层临时权重 |

- `max_abs_diff = 0.0`
- `torch.allclose(..., atol=1e-4, rtol=1e-4) = True`
- Offloading 初始化后 allocation 为 `2,068,872,704` B（约 1.93 GiB）。
- 48 层 prefill 共搬运 H2D `5,799,469,152` B，D2H `5,799,469,152` B；prefill 结束时没有驻留层。

因此本轮在短 prefill 上同时满足了输出一致性和显存下降目标，峰值 allocation 下降约 67.6%。

## PyTorch Profiler

远程 profile 使用 `performance_small.jsonl` 第一条请求（单请求、最大输出 32 token），配置为 g64、batch 1、最大序列长度 2048；profile trace 位于远程任务容器的 `/tmp/offload-verify-profile.json`（约 956 MiB）。本轮 profile 的总 Self CUDA time 为 69.980 s：

| 条目 | CUDA self time | 占比 | 调用次数 |
| --- | ---: | ---: | ---: |
| `aten::copy_` | 47.973 s | 68.55% | 163,758 |
| Memcpy HtoD（Pinned -> Device） | 14.387 s | 20.56% | 33,280 |
| Memcpy DtoH（Device -> Pinned） | 14.136 s | 20.20% | 33,311 |
| unrolled elementwise kernel | 9.745 s | 13.93% | 20,992 |
| 128-thread elementwise kernel | 9.578 s | 13.69% | 20,992 |
| `aten::mul` | 9.099 s | 13.00% | 41,344 |
| `aten::sub` | 6.156 s | 8.80% | 11,777 |
| `aten::mm` | 3.694 s | 5.28% | 10,528 |
| `Command Buffer Full` | 6.924 s | 9.89% | 60,070 |

该 profile 中每个生成 token 都要重新搬运 48 层量化权重，累计 H2D/D2H 各 `192,864,324,608` B（约 179.7 GiB）。与基线 profile 的 `aten::copy_` 46.87% 相比，Offloading 将搬运提升为首要瓶颈；它主要换取显存空间，当前实现尚未带来端到端加速。

## 与入分布目标的关系

本轮没有把完整 `performance_public.jsonl` 作为通过条件。原因是当前 attention 仍然是 eager 实现，并且 KV cache 仍按 `max_sequence_length` 静态预分配；基线在入分布长 prompt 的 prefill 阶段已经 OOM。Offloading 只降低权重常驻显存，不能消除 `QK^T`、mask、softmax 等 `O(QK)` 临时张量，也不能回收静态 KV 的预留空间。因此 `<=36 s` 的入分布目标尚未达成，后续必须继续实现 Ring KV cache 和 Paged Attention，并结合 fused W4A16/SDPA 降低 attention 与反量化开销。

## 下一轮优化建议

1. **先实现 Ring KV cache**：滑动窗口层只保留最近 1024 个位置，直接降低长序列 KV 常驻量。
2. **再实现 Paged Attention**：按请求实际长度分配 KV block，避免 10 个不同长度请求共享静态最大容量；这一步对给定 4/2/4 输入长度分布最直接。
3. **减少 Offloading 搬运开销**：将每层多个 tensor 合并为预分配的双缓冲，减少 33,000 级别的小 copy；保留 Event/stream 顺序。
4. **优化 INT4 reference kernel**：profile 中 elementwise、`mul`、`sub` 和位移仍占大量时间，应实现 fused dequant + GEMM，避免完整 BF16 临时权重。
5. **attention 使用 SDPA/FlashAttention 或 chunked prefill**：不物化完整 score/prob，解决 1024-2000 token prompt 的 attention 峰值。

本轮完成后应先由用户判断是否接受该 Offloading 设计，再开始 Ring KV cache。
