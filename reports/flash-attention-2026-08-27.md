# 自写 Triton Flash/Paged Attention 实验报告

日期：2026-08-27

## 本轮目标与规则

本轮将 eager attention 替换为项目内自行编写的 Triton Flash Attention kernel，并让
kernel 直接读取 Paged KV cache 的 block table。实现没有调用 FlashAttention、xFormers、
bitsandbytes、cuBLASLt 封装、PyTorch SDPA 或其他现成 attention 算子库。

量化结果继续复用长期 checkpoint：

    /home/h3250106394/quantized/gemma-4-12b-gptq-group64

本轮没有重新执行 GPTQ，权重量化参数仍是 INT4、group size 64。

## Kernel 设计

核心实现位于 `src/hpc101_infer/kernels/flash_attention.py`，backend 名称为
`triton_flash`。数据流如下：

1. grid 按 query tile、batch/query head、输出维 tile 三维展开；
2. 每次只加载一个 `16 x 32` 的 QK tile，在寄存器中计算分数；
3. 使用 FP32 online softmax 持续更新 row max、归一化和输出 accumulator；
4. kernel 结束时才写回 `[B,H,Q,D]` 输出，从不向全局显存写完整 `[Q,K]` scores 或
   probability；
5. GQA 在 kernel 内通过 `kv_head = query_head // group_count` 映射，不再调用
   `repeat_kv` 物化重复 K/V；
6. causal、padding、全局 attention 和 sliding window mask 均融合在 score tile 内；
7. 根据当前 query block 的最早/最晚绝对位置裁掉窗口前和 causal 上界后的 key tile。

Gemma 4 的 sliding head dimension 为 256，全局 head dimension 为 512。为避免 512 维
FP32 输出 accumulator 耗尽寄存器，输出维以 256 为 tile；512 维全局头由两个 program
分别累积，因此会重复一次 QK，但仍不产生全量 scores。

## Paged 与 Ring KV 接入

Paged 模式不再调用 `PagedLayerKVCache.view()` 收集连续 K/V。kernel 根据逻辑 token
位置计算 logical block 和 block offset，再通过 block table 找到 physical block，直接从
`[physical_blocks, block_size, kv_heads, head_dim]` 读取 K/V。block size 延续上一轮选定
的 16。

具体覆盖三种路径：

- 全局层 prefill/decode：直接读取非 Ring block table；
- sliding decode 和不超过窗口的 prefill：直接读取 Ring block table；
- 超过 1024-token 窗口的首次 prefill：直接对本轮新算出的 dense K/V 做 tiled Flash
  Attention，同时只把最新窗口写入 Ring，避免先写 Ring 后覆盖早期 query 所需的 K/V。

teacher-forcing 的多-token chunk 还需要旧历史。代码在写入新 chunk 前收集一次旧窗口，
再把旧窗口与本轮 K/V 交给 dense Triton kernel；KV cache 记录上次提交的最大长度，避免
仅为判断历史状态引入新的 GPU 到 CPU 同步。常规生成的首次 prefill 和单-token decode
不走该收集路径。

## 正确性与质量

远程作业 `177568`、`177610` 和最终回归 `177763` 在 H800 PCIe MIG 1g.10gb 上验证了：

- BF16 dense full/sliding attention；
- 256 和 512 head dimension；
- GQA、padding、非整除 query/key tile；
- Paged KV block table；
- Ring KV 跨页、跨逻辑 block 的 decode。

最终仓库回归为 `15 passed in 9.96s`。Triton 3.7.1 在 Python 3.13 下产生 5 条
`AnnAssign` deprecation warning，不影响本次编译和数值结果。

完整公开质量集作业 `177656` 评测 62 个序列、20,418 个 token：

| 指标 | eager W4A16 上一轮 | `triton_flash` 本轮 |
| --- | ---: | ---: |
| mean NLL | 2.39504889 | 2.39481079 |
| delta NLL | 0.08659336 | 0.08635526 |
| 门槛 | `< 0.1` | `< 0.1`，通过 |

attention 数值路径造成的 mean NLL 变化约 `-2.38e-4`，未改变质量结论。正式结果保存在
`results/flash-attention-public-quality.json`。

## 完整 10 请求正式成绩

作业 `177681` 按实验文档口径完整运行 `datasets/performance_public.jsonl`，没有截断任一
请求的 `max_new_tokens`。输入为 10,500 token，输出为 320 token，进程正常退出并生成
10 行结果。

| 指标 | 上一轮 W4A16 eager | 本轮 `triton_flash` | 变化 |
| --- | ---: | ---: | ---: |
| 正式 `elapsed_s` | 400.511 s | 366.652 s | -8.45% |
| 生成吞吐 | 0.79898 token/s | 0.87276 token/s | +9.23% |
| 累计 prefill latency | 70.683 s | 72.370 s | +2.39% |
| 累计 decode latency | 322.497 s | 287.135 s | -10.97% |
| 加权 Mean TPOT | 1.0403 s | 0.9262 s | -10.97% |
| 最大 GPU allocated | 5,856,150,528 B | 5,856,150,528 B | 不变 |
| 最大 GPU reserved | 10,005,512,192 B | 10,005,512,192 B | 不变 |

正式结果位于：

- `results/flash-attention-public-generation.jsonl`
- `results/flash-attention-public-summary.json`

本轮虽然实测快 8.45%，但不能把全部差值归因于 attention。后述 profile 表明 attention
只占很小比例，而不同作业的 H2D 时间有明显波动。当前 `366.652 s` 仍是 36 秒目标的
10.18 倍，尚未达到入分布目标。

## CUDA Profile

作业 `177742` 预热后 profile `performance_public.jsonl` 第一条请求，限制为 32 个输出
token。profile 会扰动时间，因此只用于归因，不替代上节无 profiler 的正式成绩。

- TTFT：1.845 s；
- Mean TPOT：0.955 s；
- total latency：32.038 s；
- peak allocated：2,697,664,000 B；
- peak reserved：3,189,768,192 B；
- Self CUDA 总计：32.628 s。

| 算子/事件 | Self CUDA | 占比 | 调用次数 |
| --- | ---: | ---: | ---: |
| `_w4a16_gemm_kernel` | 17.885 s | 54.82% | 10,496 |
| `Memcpy HtoD (Pinned -> Device)` | 13.984 s | 42.86% | 1,536 |
| `aten::mm` | 280.389 ms | 0.86% | 32 |
| `_flash_attention_kernel` | 79.552 ms | 0.24% | 1,536 |
| unrolled elementwise kernel | 93.801 ms | 0.29% | 30,816 |

每次模型 forward 有 48 次 attention kernel 和 48 次层权重 H2D；32 次 forward 因而均为
1,536 次。W4A16 共 328 个投影/MLP linear，每次 forward 对应 10,496 次调用。W4A16 与
pinned H2D 合计约占 Self CUDA 的 97.68%，attention 已不是端到端主瓶颈。

## 为什么仍有 1 GiB 分配失败提示

2000-token 请求仍报告一次可恢复的 allocator warning，请求大小正好为
`1,048,576,000 B`。这与模型词表和 BF16 logits 的尺寸完全吻合：

    2000 tokens x 262144 vocab x 2 bytes = 1,048,576,000 bytes

因此该请求来自 prefill 末尾 `F.linear(hidden_states, embed_weight)` 生成所有 prompt token
的完整 logits，不是 attention 的 `QK^T` scores。当前生成只使用每条请求最后一个有效
prompt token 的 logits，却先生成了全部 logits；这也解释了 Flash Attention 已消除全量
scores 后，正式结果的最大 allocated/reserved 仍与上一轮相同。

## 后续优化建议

1. **只计算 prefill 所需 logits**：在 LM head 前按每条请求的最后有效位置 gather hidden
   state，再做 vocab projection。它可直接消除长 prompt 的 1 GiB logits 张量和 allocator
   warning，并减少 TTFT。
2. **提高 decode 有效 batch**：当前 batch 1 让每个 token 都支付 48 层权重 H2D；应让
   scheduler 同批推进多个请求，以摊薄 1,536 次 H2D 和层加载等待。要达到 36 秒，这比
   继续微调 attention tile 更关键。
3. **优化 M=1 W4A16**：profile 第一热点是 10,496 次 W4A16。应设计 decode 专用 INT4
   GEMV/持久化归约和更合适的数据布局，并与更大 decode batch 联合调参。
4. **保留当前 Flash/Paged kernel**：它已经把 attention 压到 0.24% 并消除全量 scores，
   是后续增大 batch、释放显存和扩展上下文的基础；当前不应继续优先投入 attention 微调。

## 结论

本轮完成了实验规则要求的自写 Triton Flash Attention，并直接支持 Paged/Ring KV、GQA、
causal、padding、全局和 sliding attention。质量门槛继续通过，完整 10 请求从
`400.511 s` 降至 `366.652 s`，但尚未达到 `<= 36 s`。profile 已把下一步优先级收敛到
prefill logits 显存、decode batching 和 M=1 W4A16，而不是 attention 本身。
