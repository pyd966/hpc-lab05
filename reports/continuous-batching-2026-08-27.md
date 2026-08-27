# Continuous Batching 实现与测试报告

## 本轮目标与评测口径

本轮在已有 GPTQ g64、异步权重 Offloading、Ring KV cache、Paged KV cache、自写
Triton W4A16 GEMM 和自写 Triton FlashAttention 的基础上实现 continuous batching。
量化 checkpoint 继续直接复用长期缓存：

```text
/home/h3250106394/quantized/gemma-4-12b-gptq-group64
```

正式性能测试严格使用 `datasets/performance_public.jsonl` 的全部 10 个请求，不截断
输出，长度分布为：

| 项目 | 长度与请求数 |
| --- | --- |
| 输入 | 250 x2、500 x2、1000 x2、1500 x2、2000 x2 |
| 输出 | 16 x3、32 x4、48 x3，共 320 tokens |

评测允许请求组成 batch。所有请求在推理开始前均已到达，因此配置采用
`scheduler_backend=continuous`、`scheduler_batch_size=10`、
`max_batch_size=10`、`prefill_token_budget=2048`。其余参数保持 g64、BF16、
`int4_triton`、`triton_flash`、max sequence length 2048、seed 42 和同步指标计时。

## 实现

### 1. 可补位的连续调度器

`ContinuousBatchScheduler` 维护 FIFO 等待队列、按 slot 编号排序的活跃请求和最小堆
空闲 slot。请求完成后立即回收 Paged KV blocks 和 slot；下一轮 decode 前优先接纳
等待请求。decode batch 每步只包含仍活跃请求，不再为已完成请求保留空行。

Prefill 使用 padded token budget：加入一个请求后的工作量按
`batch_size * max_prompt_length` 计算，超过 2048 时结束当前 prefill batch，但始终至少
接纳一个请求。公开数据因此形成 4、2、1、1、1、1 六个 prefill batch，并把 10 个
请求全部放入稳定 KV slot 后进行紧凑 decode。

### 2. 稳定 KV slot 与 Paged cache 回收

模型 batch 行和 KV slot 解耦。attention 接收 `cache_indices`，将紧凑 batch 行映射到
稳定 slot；Paged block table、cache length、commit、view 和 release 都支持该映射。
完成请求的物理 pages 立即回到空闲池，新请求可以复用同一个 slot，而不会移动其他
请求的 KV。

调度器在 CPU 上已经知道 slot 和即将写入的 token 范围，因此在模型 forward 前统一
reserve pages。只有跨越 16-token page 边界、页表实际发生变化时，才把 CPU 页表批量
同步到 GPU。

### 3. 消除调度路径同步与无用 logits

最初实现仍在每层用 GPU 布尔索引去掉 prefill padding。首轮 profile 显示
`aten::nonzero` 共调用 2880 次，CPU total 为 30.186 秒；这些同步点还会破坏 pinned
H2D 与 W4 GEMM 的重叠。最终实现把 CPU 已知的 `(start, end)` token range 一直传到
KV 写入，直接构造有效 source indices，`aten::nonzero` 已从最终热点表消失。

此外，continuous prefill 在最终 RMSNorm 后先收集每行最后一个有效 hidden state，再做
LM head，只产生 `[batch, 1, vocab]` logits，避免长 prompt 的完整 vocab 临时张量。

## 正确性与质量

最终 GPU 回归作业 `178624`：

```text
14 passed, 5 warnings in 9.07s
```

测试覆盖调度器补位和 padded budget、compact row 到 stable slot 的 Paged KV 映射与回收、
Ring KV，以及 dense/paged/ring 三类自写 Triton FlashAttention。

完整质量作业 `178579` 评测 62 个序列、20,418 个 token：

| 指标 | 结果 |
| --- | ---: |
| mean NLL | 2.3948107879 |
| reference mean NLL | 2.3084555301 |
| delta NLL | **0.0863552578** |
| 质量门槛 | **通过，< 0.1** |

最终生成结果还与消除同步前的 continuous batch 10 逐请求比较，10 条请求的全部 token id
完全一致。

## 完整 10 请求成绩

最终作业 `178631` 直接读取仓库中的 `config.yaml`，未通过命令行覆盖 batch 或 scheduler：

```text
requests=10
generated_tokens=320
elapsed_s=111.39575808501104
generated_tokens_per_s=2.8726408033939124
peak_allocated_bytes=6297783808
peak_reserved_bytes=6639583232
```

| 版本 | 作业 | 10 请求总时间 | tokens/s | 相对上一项 |
| --- | ---: | ---: | ---: | ---: |
| 串行自写 FlashAttention 基线 | 177681 | 366.652 s | 0.8728 | - |
| 首个可运行 continuous 版本 | 178201 | 133.088 s | 2.4044 | -63.70% |
| 最终无同步 continuous 版本 | 178631 | **111.396 s** | **2.8726** | -16.30% |

最终版本相对串行基线总时间减少 **69.62%**，吞吐为 **3.291x**。但距离 36 秒目标仍差
75.396 秒，当前还需要约 **3.09x** 加速，尚未达到最终目标。

## Profile

最终代表性 profile 作业 `178661` 使用 `performance_small.jsonl` 的全部 4 个请求、60 个
输出 token，先以每请求 2 tokens 预热相同 prompt 形状和一次 decode，再采样完整请求：

```text
wall_time_s=25.809076492034364
generated_tokens_per_s=2.3247635388472045
peak_allocated_bytes=3822362112
peak_reserved_bytes=4242538496
Self CUDA time total=38.113s
```

| CUDA 热点 | Self CUDA | 占比 | 调用数 |
| --- | ---: | ---: | ---: |
| `_w4a16_gemm_kernel` | 22.833 s | **59.91%** | 10,496 |
| `Memcpy HtoD (Pinned -> Device)` | 14.280 s | **37.47%** | 1,536 |
| embedding/LM head `aten::mm` kernel | 278.233 ms | 0.73% | 32 |
| `_flash_attention_kernel` | 57.222 ms | 0.15% | 1,536 |

Self CUDA 总和大于 wall time，说明双 buffer 已让部分 pinned H2D 与 W4 GEMM 重叠。
消除 `nonzero` 前的同口径 profile 作业 `178480` 为 37.745 秒，最终为 25.809 秒，
profile wall time 下降 **31.62%**。完整 10 请求 profile 作业 `178424` 已完成 CUDA
采样并得到 127.494 秒 wall time、6.071 GB allocated，但在聚合事件表时触及 24 GiB
主机内存限制；正式成绩不使用 profiler，因此不受该问题影响。

## 下一轮优化建议

1. **继续优化自写 W4A16 kernel。** 它占最终 Self CUDA 的 59.91%，应分别为长 prefill
   和小 M、大 batch decode 调整 tile、warps/stages 与持久化策略，并减少剩余的拆包和
   边界开销。这是最大的纯计算收益来源。
2. **改成混合常驻与 Offloading。** 完整 10 请求峰值 allocated 约 6.30 GB，在 10 GiB
   MIG 上仍有可利用空间。应按实测安全余量让尽可能多的量化层常驻 GPU，只异步搬运
   其余层，直接降低 37.47% 的 pinned H2D 工作量。
3. **让 prefill 在 layer 维复用权重。** 当前 2048 token budget 需要六次完整模型
   forward，即同一层权重为六个 prefill microbatch 重复 H2D。可以在同一 layer 驻留
   期间依次处理多个 prefill microbatch，再进入下一层，以显著减少长 prompt 阶段搬运。
4. **再处理小 kernel 和 launch 开销。** W4/H2D 之后才考虑融合 RMSNorm、RoPE、残差和
   激活等小算子；FlashAttention 仅占 0.15%，当前不应继续优先优化 attention。

## 产物

- `results/continuous-public-generation.jsonl`：完整 10 请求生成结果；
- `results/continuous-public-summary.json`：正式端到端汇总；
- `results/continuous-public-quality.json`：完整质量结果；
- `scripts/profile_continuous_batching.py`：continuous profile 脚本。
