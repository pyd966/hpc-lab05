# 共享 Paged KV Pool 与混合 MLP 常驻报告

## 结论

本轮在 Round 1 blocked W4A16 kernel 基础上实现共享物理 KV block pool、生命周期
admission credit 和 phase-aware MLP residency。正式 workload 的完整 10 请求连续运行三次：

```text
31.90793757000938 s
31.67921580100665 s
31.238719261949882 s
```

中位数为 **31.679 s**，最大值为 **31.908 s**，三次均满足 `elapsed_s <= 36 s`。
最慢一次仍有 4.092 秒余量，并且最终 44/48 层常驻配置没有 allocator OOM warning。

评测继续直接复用长期 GPTQ checkpoint：

```text
/home/h3250106394/quantized/gemma-4-12b-gptq-group64
```

本轮没有重新量化，也没有修改 GPTQ code、scale 或 zero。

## 共享 Paged KV Pool

### 原实现的问题

原 Paged KV allocator 已经支持按需分 page 和完成后回收，但初始化仍使用：

```text
pool_blocks = max_batch_size * max_blocks_per_sequence
```

因此 batch 10 时仍为每个 slot 预留 2048 token 的最坏容量。48 层 KV tensor 加 metadata
共 `3,743,710,080 B`；分页只改变映射方式，没有释放足够的物理显存。

### 新容量

block size 保持 16。block table 仍为固定的
`[max_batch_size, max_blocks_per_sequence]`，只缩小每层 K/V tensor 的物理 block 维，
所以现有自写 Triton Paged Attention 的地址和 metadata 接口不变。

正式配置为：

```yaml
paged_kv_block_size: 16
paged_kv_global_pool_blocks: 776
paged_kv_sliding_pool_blocks: 530
```

这两个数表示每一个同类 layer 中、跨全部 request slot 共享的物理 block 数。公开 10
请求最终长度为：

```text
298, 282, 516, 548, 1032, 1032, 1548, 1516, 2032, 2016
```

实际 global 需求为 680 blocks，sliding 需求为 495 blocks。配置的 776/530 按题目给定
输入与输出区间的上界计算，分别保留 96/35 blocks 水位，而不是只适配公开样本的精确
长度。新 KV pool 含 metadata 为 `2,982,443,904 B`，比原实现释放
`761,266,176 B`，约 726 MiB。

### 生命周期 credit 与原子 reserve

只缩小 tensor 仍可能在 decode 跨 page 时中途耗尽。现在 continuous scheduler 在请求
获得 stable slot 前，按 `prompt_length + max_new_tokens` 一次性检查并预留整个生命周期的
block credit：

- global demand 为 `ceil(max_length / block_size)`；
- sliding demand 截断到 ring 的 65 blocks；1024 窗口在未对齐时可能跨 65 页，不能按
  64 计算；
- 队首暂时放不下时停止 prefill admission，先继续 active decode；
- 请求完成后，`release()` 同时归还实际物理页和未来 credit；
- 单请求本身无法装入 pool 时立即报错，不进入无限等待。

`KVCache.reserve()` 也先为全部 paged layer 构造 plan 并检查 free blocks，再修改任何页表。
即使直接调用路径配置错误，也不会出现前几层已经分配、后层失败的半提交状态。

## 44/48 层 MLP 常驻

### Payload 边界

Round 1 每次模型遍历都搬运 48 个完整 decoder layer，约 5.799 GB；6 次 prefill 和 47 次
decode 合计约 307.37 GB H2D。现在按 layer phase 拆分：

- embedding、final norm 和 44 层 MLP/norm 常驻 GPU；
- 48 层 attention projection 仍由异步 offloader 管理；
- 另外 4 个 sliding layer 的 MLP 保留在 offload payload 中，用于增加显存余量；
- offloader 仍使用一个 transfer stream、当前 compute stream 和两个固定 GPU byte
  buffer，没有增加 stream 数量。

forward 时序保持：

1. transfer stream 预取 layer 0；
2. compute stream 等待当前 layer ready event；
3. 在当前 layer 计算前立即发射下一 layer H2D；
4. 当前 attention 和 resident MLP 计算覆盖下一 layer 搬运；
5. release 在 compute stream 记录完成 event，只切回 pinned CPU view，不执行 D2H；
6. transfer stream 复用 buffer 前等待 compute-done event。

最终 profile 区间的 offloader 统计为：

```text
host_to_device_bytes=87,695,737,256
device_to_host_bytes=0
prefetch_calls=2544
merged_payload_bytes=1,654,636,552
gpu_buffer_slots=2
gpu_buffer_bytes=238,204,932
```

H2D 相对 Round 1 的约 307.37 GB 下降 **71.47%**。

### RoPE cache 去重

原每层 attention payload 还包含独立 FP32 RoPE cache，48 份合计约 224 MiB，并在每次
遍历重复 H2D。现在相同 RoPE 配置共享一份 GPU cache：40 个 sliding layer 共享一份，
8 个 global layer 共享一份。`AsyncLayerOffloader` 的 tensor filter 排除这些 resident
buffer，只搬运真正的 layer payload。

### 为什么不是全部 48 层常驻

全部 MLP 常驻作业 `180693` 得到 28.149 秒，RoPE 去重后的作业 `180866` 得到 28.813
秒，但两者都在 64 MiB activation 申请时触发 allocator OOM warning，清理缓存重试后才
完成。对应峰值 allocated 接近 9.95 GB，工程余量不足。

最终显式配置 `weight_resident_mlp_layers: 44`。选择 4 个 sliding layer 放回 payload 后：

- 正式峰值 allocated：`9,742,262,784 B`；
- 正式峰值 reserved：`10,039,066,624 B`；
- 相对全常驻释放约 207 MB allocated；
- 三次正式运行和一次 profile 均无 allocator warning。

这用约 3 秒端到端时间换取了可重复性，仍保留 4 秒以上评分余量。

### cuBLAS 初始化顺序

第一版在完成最大 prefill 后首次进入 LM head，才延迟创建 cuBLAS handle，因持久显存已
分配而报 `CUBLAS_STATUS_ALLOC_FAILED`。engine 现在在 MLP/KV 大块分配前用 16x16 BF16
矩阵乘初始化 cuBLAS handle/workspace；初始化发生在正式 generation 计时之前，不改变
官方计时区间。

## 正确性

最终全部测试作业 `181333`：

```text
28 passed, 5 warnings in 9.99s
```

warning 仅为 Triton 对 Python 3.15 的 `AnnAssign` 弃用提示。新增测试覆盖：

- 物理 pool 小于 slot 最坏容量时，固定 block table 形状不变；
- 1024/1025 token 的 64/65 ring block 边界；
- lifecycle credit 容量竞争、完成释放和 slot refill；
- 跨 layer reserve 失败不产生部分页分配；
- 公开长度的 global/sliding 需求为 680/495 blocks。

三次正式生成结果忽略 metrics 时延后互相完全一致，也与 Round 1 的 10 请求 token ids、
文本、token 数和 finish reason 逐字段一致。量化数值与 kernel 算术未变化，因此质量沿用
完整 62 序列作业 `180064`：

```text
mean_nll=2.394810787939536
reference_mean_nll=2.3084555301012055
delta_nll=0.08635525783833042
passed=true
```

## 完整 10 请求正式成绩

| 版本 | 作业 | elapsed_s | tokens/s | 结果 |
| --- | ---: | ---: | ---: | --- |
| continuous 基线 | 178631 | 111.395758 | 2.872641 | 未达标 |
| Round 1 blocked W4 | 179945 | 49.234414 | 6.499519 | 未达标 |
| Round 2 稳定 run 1 | 181028 | 31.907938 | 10.028853 | **通过** |
| Round 2 稳定 run 2 | 181162 | 31.679216 | 10.101260 | **通过** |
| Round 2 稳定 run 3 | 181188 | 31.238719 | 10.243698 | **通过** |

最终中位数相对 Round 1 再下降 **35.66%**；相对 111.396 秒 continuous 基线总时间下降
**71.56%**，总体加速 **3.52x**。

## 完整 10 请求 Profile

最终 CUDA-only profile 作业 `181123` 在正式采样前预热真实 shape，并清理仅预热产生的
unused allocator cache。正式区间仍完整运行 10 请求和 320 tokens：

```text
wall_time_s=26.38207982800668
generated_tokens_per_s=12.129445520830185
peak_allocated_bytes=9516695552
peak_reserved_bytes=10039066624
Self CUDA time total=31.678s
```

| CUDA 热点 | Self CUDA | 占比 | 调用数 |
| --- | ---: | ---: | ---: |
| `_w4a16_gemm_kernel` | **13.590 s** | **42.90%** | 17,384 |
| pinned H2D | **10.168 s** | **32.10%** | 2,544 |
| `_flash_attention_kernel` | 2.925 s | 9.23% | 2,544 |
| 两类主要 elementwise kernel | 1.897 s | 5.99% | 84,217 |
| embedding/LM head GEMM | 0.465 s | 1.47% | 53 |

Round 1 profile 的 wall time 为 43.347 秒、pinned H2D 为 23.554 秒、W4 为 20.557 秒。
本轮 profile wall time 下降 39.14%，H2D 不再是不可跨越的 23 秒下界；W4 和 H2D 仍有
异步重叠，因此 Self CUDA 总和高于 wall time。

## 产物

- `src/hpc101_infer/runtime/kv_cache.py`：共享 pool、credit、原子 reserve；
- `src/hpc101_infer/scheduler/continuous.py`：容量感知 admission/refill；
- `src/hpc101_infer/runtime/offloading.py`：resident tensor filter；
- `src/hpc101_infer/models/gemma4.py`：部分 MLP residency 与 RoPE cache 共享；
- `src/hpc101_infer/config.py`、`config.yaml`：显式 pool 和 residency 配置；
- `tests/test_shared_kv_pool.py`：共享池与生命周期回归；
- `scripts/profile_continuous_batching.py`：full10 低内存 profile 和区间 offloader stats；
- `results/hybrid-resident-public-*.json*`：三次完整正式结果。

## 后续可选优化

当前目标已经稳定达成。若继续追求更低延迟，profile 表明优先级应为 W4 小 M 专用 kernel
和 gate/up epilogue 融合，其次才是 CUDA Graph；FlashAttention 只有 9.23%，不应重新
成为第一优先级。任何后续改动仍应保持三次 full10 最大值 `<36 s` 和
`delta_nll <0.1`。
