# Ring KV Cache 实验报告

日期：2026-08-27
本轮目标：在保留全局注意力完整上下文的前提下，为 Gemma4 的滑动注意力层实现固定容量 Ring KV cache。

## 实现概要

Gemma4 同时包含 `full_attention` 和 `sliding_attention` 层。本轮没有把两类层混用：

- 全局注意力层仍按 `max_sequence_length` 分配，并使用绝对位置保存完整前缀。
- 滑动注意力层在 `ring_kv_cache=true` 时按 `min(max_sequence_length, sliding_window)` 分配固定容量；写入位置使用 `position % capacity`。
- 一次 prefill 超过窗口时，只保留最新的 `capacity` 个有效 token；右侧 padding 和 decode 阶段 inactive slot 不写入 cache。
- `view()` 在未发生环回时直接返回连续前缀；发生环回后按逻辑绝对位置生成 `[start, ..., end]`，通过物理位置取回 K/V，并把这些逻辑位置传给 attention mask。
- causal、padding 和 sliding-window 条件都基于每个 batch 的逻辑绝对 key position 计算，避免把环形物理下标误当成 token 位置。

配置入口为 `EngineConfig.ring_kv_cache`，默认值为 `true`，仓库的 `config.yaml` 也显式开启。验证脚本支持 `--no-ring-kv-cache` 作为对照开关。

## 远程正确性验证

硬件仍为 `lab5` 分区的 1 张 H800 MIG `1g.10gb`、4 vCPU、24 GiB 主机内存；量化 checkpoint 使用长期缓存：
`/home/h3250106394/quantized/gemma-4-12b-gptq-group64`。

- 作业 `175882`：仓库全部测试通过，`5 passed in 3.22s`。
- 作业 `175726`：Ring KV 专项 CPU 测试通过，`3 passed in 3.12s`。
- 作业 `175769`：真实 Gemma4 配置下的 cache 分配和小型 `SlidingAttentionLayer` 集成验证。
  - 长度 6、窗口 4 的 prefill 最后一个有效 token：`max_abs_diff = 0`。
  - 追加一个 decode token：`max_abs_diff = 1.19209290e-7`。
  - 两个结果均来自 Ring 与非 Ring 使用相同权重、相同输入的对照。

这组测试只比较实际需要的最后一个 prefill 位置；prefill 中窗口外的早期 query 不会影响后续层的有效窗口，因此不要求它们输出相同。

## KV 显存容量

作业 `175769` 直接从真实 Gemma4 配置构造 cache，`max_batch_size=1`、`max_sequence_length=2048`、dtype 为 bfloat16：

| 配置 | 全局层 | 滑动层 | 物理容量 | cache allocated |
| --- | ---: | ---: | --- | ---: |
| Ring 关闭 | 8 | 40 | 48 层均 2048 | `704,643,456 B`（0.6563 GiB） |
| Ring 开启 | 8 | 40 | 8 层 2048；40 层 1024 | `369,099,136 B`（0.3438 GiB） |

仅 KV cache 预留减少 `335,544,320 B`（320 MiB），相对减少 **47.619%**。全局层仍保留完整 2048 槽，因此 Ring 不会改变全局注意力的上下文能力。

## 短请求端到端对照

使用量化 g64、异步权重 offloading、`int4_reference`、batch 1、`performance_small.jsonl` 第一条请求、最多生成 32 token；以下两次均未启用 profiler：

| 配置 | 生成 token | 总耗时 | 峰值 GPU allocated |
| --- | ---: | ---: | ---: |
| Ring 关闭（作业 `175837`） | 32 | `52.065366 s` | `3,382,125,568 B` |
| Ring 开启（作业 `175864`） | 32 | `53.466660 s` | `3,046,581,248 B` |

Ring 版本在这个短上下文样本上慢约 **2.7%**，但峰值 GPU allocated 少 `335,544,320 B`。该样本的有效上下文没有覆盖 1024 token 窗口，Ring 主要体现为静态预留减少，不能期待明显加速；耗时差异也不应解释为 Ring 的长期性能趋势。

## CUDA Profile

作业 `175773` 对同一数据集第一条请求采集 profile，Ring 开启、最多生成 32 token。推理指标中的端到端时间为 `51.412584 s`，峰值 GPU allocated 为 `3,046,581,248 B`；profile 汇总的 Self CUDA 总时间为 `54.762 s`。

| 算子/事件 | Self CUDA | 占比 | 调用次数 |
| --- | ---: | ---: | ---: |
| `aten::copy_` | `33.382 s` | 60.96% | 104,822 |
| `Memcpy HtoD (Pinned -> Device)` | `14.627 s` | 26.71% | 1,536 |
| unrolled elementwise kernel | `9.380 s` | 17.13% | 20,992 |
| 128-thread elementwise kernel | `9.243 s` | 16.88% | 20,992 |
| `aten::mul` | `8.811 s` | 16.09% | 41,344 |
| `aten::sub` | `5.973 s` | 10.91% | 13,057 |
| `aten::mm` | `3.562 s` | 6.50% | 10,528 |

profile 中没有出现 Ring gather 成为主要热点，原因是该短请求尚未跨过窗口；主要瓶颈仍是每个 decode token 的权重 H2D 和 INT4 reference 反量化。Ring 的直接价值是为长 prompt 降低滑动层的 KV 常驻显存，为后续 chunked prefill/paged attention 留出空间。

## 本轮结论

本轮已完成并保留 Ring KV cache，默认开启，且通过了 cache、mask、attention 集成和全套测试。它将真实模型 2048 上限下的 KV 预留降低 47.62%，而短上下文端到端时间基本持平但有约 2.7% 的测量差异；这符合当前 workload 不足以触发窗口环回的预期。

本轮没有声称达到入分布总时间 `<=36 s`：profile 仍显示权重搬运和 INT4 reference kernel 占主导，Ring 解决的是 KV 容量而不是这些计算/带宽热点。下一轮再根据判断实现 Paged Attention。
