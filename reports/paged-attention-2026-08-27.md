# Paged Attention 实验报告

日期：2026-08-27

## 本轮目标

在已实现异步权重 Offloading 和 Ring KV cache 的基础上，实现文档要求的功能正确版 Paged Attention：每个请求通过 block table 把逻辑 KV block 映射到共享物理 block pool，按需扩容，并在请求结束时回收 block；attention 前按逻辑顺序收集 K/V，继续复用现有 eager attention。

## Block Size 选择

本实现采用 **16 tokens/block**，并允许通过 `engine.paged_kv_block_size` 配置，但只接受大于 1 的 2 次幂。

选择依据来自常见开源实现：

- vLLM 当前 `CacheConfig.DEFAULT_BLOCK_SIZE` 为 16：<https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py>。
- vLLM 的原生 Paged Attention 后端把 16 和 32 作为常见支持尺寸，并要求相关布局按 16 对齐：<https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py>。
- TensorRT-LLM 的官方 KV cache 文档要求 tokens-per-block 是大于 1 的 2 次幂，并按需把 block 分配给请求：<https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md>。

对本实验而言，16 还能整除 Gemma4 的 1024-token sliding window，普通请求最后一个 block 最多浪费 15 个 token，同时保留后续编写直接读取 block table 的 CUDA kernel 时的对齐条件。

## 实现细节

核心实现位于 `src/hpc101_infer/runtime/kv_cache.py`：

- `PagedLayerKVCache` 的 K/V 布局为 `[physical_blocks, block_size, kv_heads, head_dim]`。
- 每个 layer 的物理 block pool 在该层所有 batch slot 间共享；每个 batch slot 维护自己的 block table。
- `write()` 根据绝对 token 位置计算逻辑 block 和 block offset，在第一次访问逻辑 block 时从 free list 分配物理 block。
- `view()` 根据 block table 按逻辑 token 顺序执行 `index_select`，整理成现有 attention 所需的连续 `[batch, heads, sequence, head_dim]`。
- `release(batch_indices)` 可以立即回收完成请求的全部物理 block；当前静态 scheduler 在下一次 prefill 的 `reset()` 中统一回收，接口为后续 continuous batching 保留。
- `paged_kv_cache=false` 时仍保留上一轮的连续 Ring KV 实现作为 fallback。

Paged 与 Ring 同时开启时，滑动层的 block table 使用逻辑 block 的环形槽位。由于窗口起点可能落在 block 中间，长度为 W 的窗口最多跨越 `ceil((W + P - 1) / P)` 个 page；因此实现会额外保留一个边界 page，并仍由绝对 key position mask 排除窗口外 token。

配置默认开启：

```yaml
engine:
  ring_kv_cache: true
  paged_kv_cache: true
  paged_kv_block_size: 16
```

## 正确性验证

远程节点仍为 `lab5` 分区的 H800 MIG `1g.10gb`，4 vCPU、24 GiB 主机内存。

- 作业 `176601`：全部测试通过，`7 passed in 3.14s`。
- 测试覆盖非分页 Ring、分页 full attention、多 page 收集、未对齐 sliding window、padding batch、block release 和 attention mask。
- 作业 `176608`：使用 `W=5, P=4`、长度 11 的小型 `SlidingAttentionLayer`，直接比较 Paged Ring 与旧的非分页连续 cache：
  - prefill 最后一个 token：`max_abs_diff = 0`；
  - 追加一个 decode token：`max_abs_diff = 5.96046448e-8`。

该场景会跨 3 个逻辑 page，并发生物理 page 复用，不是只验证未环回的简单前缀。

## 真实 Gemma4 Cache 布局

作业 `176608` 使用真实 g64 Gemma4 配置、batch 1、最大长度 2048、bfloat16 KV：

| 配置 | 物理布局 | cache allocated |
| --- | --- | ---: |
| Paged、Ring 关闭 | 48 层各 128 block（2048 token） | `704,692,608 B`（0.6563 GiB） |
| Paged、Ring 开启 | 40 个 sliding layer 各 65 block；8 个 full layer 各 128 block | `374,371,008 B`（0.3487 GiB） |

Paged Ring 相比不启用 Ring 减少 **46.875%** KV pool 容量。滑动层的 65 个 block 对应 1040 个物理槽，其中额外 16 个槽用于保护未对齐窗口边界；attention mask 仍只允许最近 1024 token。

当前 pool tensor 按最坏并发容量预分配，这是生产推理框架的常见做法；Paged 的收益在于请求只占用所需 block、完成后可复用，而不是把 pool 中未分配 block 归还 PyTorch allocator。作业的短请求完成时只有 `192` 个 block 被请求占用，pool 其余 block 仍可服务其他请求。

## CUDA Profile

作业 `176617` 使用长期 g64 checkpoint、异步 Offloading、batch 1、`performance_small.jsonl` 第一条请求、最多 32 个输出 token：

- 端到端指标：`53.973381 s`；
- 峰值 GPU allocated：`3,051,873,280 B`；
- Self CUDA 总时间：`56.691 s`；
- 请求完成时活跃 block：`192`。

| 算子/事件 | Self CUDA | 占比 | 调用次数 |
| --- | ---: | ---: | ---: |
| `aten::copy_` | `34.873 s` | 61.51% | 105,022 |
| `Memcpy HtoD (Pinned -> Device)` | `15.630 s` | 27.57% | 1,536 |
| unrolled elementwise kernel | `9.746 s` | 17.19% | 20,992 |
| 128-thread elementwise kernel | `9.342 s` | 16.48% | 20,992 |
| `aten::mul` | `9.092 s` | 16.04% | 44,416 |
| `aten::sub` | `5.979 s` | 10.55% | 13,057 |
| `aten::mm` | `3.583 s` | 6.32% | 10,528 |

与上一轮非分页 Ring profile（作业 `175773`）相比，端到端从 `51.412584 s` 变为 `53.973381 s`，约慢 **4.98%**；Self CUDA 从 `54.762 s` 变为 `56.691 s`，增加约 **3.52%**。功能正确版的 block 收集没有进入前 20 个热点，主导项仍是每 token 权重 H2D 和 INT4 reference 反量化。

## 量化缓存兼容

远程全套测试曾发现快速原地重写 shard 时，只有 size/mtime/ctime/inode 的输出签名可能漏判。本轮为新 sidecar 增加首尾各 64 KiB 的采样 SHA-256，同时对缺少该字段的旧 sidecar 保持兼容，因此不会扫描 7.3 GiB 权重或触发重新量化。

作业 `176656` 使用完整量化 CLI 和长期路径再次运行，直接返回：

```text
量化结果已准备好：328 个 Linear 模块
```

没有重新运行 GPTQ，长期缓存仍可继续使用。

## 结论与限制

Paged Attention 已默认开启，block size 固定采用开源项目常见的 16；block pool、block table、按需逻辑分配、Ring page 复用和释放接口均已实现并验证。

当前版本按照实验文档的第一阶段建议，在 attention 前把离散 page 收集成连续 K/V，因此主要价值是内存管理和后续 continuous batching 的基础，不是直接加速 eager attention。它尚未让入分布总时间达到 `<=36 s`；下一阶段若继续优化，需要让 attention kernel 直接读取 block table，并优先解决 profile 中占主导的权重搬运和 INT4 reference kernel。
