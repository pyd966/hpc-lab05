# Blocked W4A16 布局与 Kernel 优化报告

## 本轮目标与评测口径

本轮优化反量化-GEMM 融合算子的权重布局和 Triton kernel，目标是消除原 canonical
`[N, K/2]` qweight 在输出维 N 上的跨行访存，并分别优化 continuous batching 的
decode 小 M 和 prefill 大 M。实现仍是自行编写的 Triton kernel，没有调用
FlashAttention、xFormers、bitsandbytes、cuBLASLt 封装或 PyTorch SDPA。

正式成绩严格运行 `datasets/performance_public.jsonl` 的全部 10 个请求、320 个输出
token，不截断请求，也不改变 continuous batching 的输入/输出分布。量化结果直接复用
长期缓存：

```text
/home/h3250106394/quantized/gemma-4-12b-gptq-group64
```

本轮没有运行 GPTQ、没有更新量化参数，也没有修改长期缓存中的 canonical checkpoint。
运行时只对 qweight、scale 和 zero 做可逆的字节重排；该步骤发生在正式计时之前，也在
offloader 建立 pinned payload 之前。

## 根因与实现

### 1. N-contiguous INT4 blocked layout

原 qweight 为 `[N, K/2]`。Triton CTA 计算 `[BM, BN]` 输出 tile 时，需要同时读取 BN
个输出通道，但相邻 N lane 的地址相隔 `K/2` 字节，无法形成有效的合并访问。新布局为：

```text
qweight: [ceil(K/64), ceil(N/128), 32, 128]
Qb[k//64, n//128, (k%64)//2, n%128]

scales/zeros: [K/group_size, ceil(N/128), 128]
Sb[k//group_size, n//128, n%128]
```

每个 byte 仍保存相邻 K 位置的两个 INT4 code，低 nibble 对应偶数 K，高 nibble 对应
奇数 K，因此布局转换不改变任何量化数值。Gemma4 当前 328 个量化矩阵的 N 都能被 128
整除、K 都能被 64 整除，不会为正式模型增加 padding 字节。

`QuantizedLinear.prepare_triton_layout()` 兼容现有 canonical checkpoint；reference 路径
需要反量化时再无损恢复 canonical view。OJ 重新量化时仍由项目自身量化流程生成 code，
随后同一 loader bridge 在 kernel/offloader 使用前生成 blocked buffer，不依赖外部工具。

### 2. group64 scale 广播和 program ordering

kernel 的 `BLOCK_K=64` 与 GPTQ group64 对齐。每个 K tile 只加载 BN 个 scale，并沿 K
广播，不再为同一个 group 重复加载 `BK * BN` 份 scale。qweight 和 scale 都沿 N 连续。

CTA 使用 grouped program ordering。decode 保留 N=128 的宽 tile；大 M prefill 改为
N-fast 的 `128x64x64`，让一次解包后的权重服务 128 行输入。相较初版
`64x128x64`，每个 CTA 的输出元素数不变，但权重反量化次数减半。

最终离线固化的配置为：

| M 区间 | BM | BN | BK | warps | stages | GROUP_M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M <= 16` | 16 | 128 | 64 | 4 | 3 | 1 |
| `16 < M <= 64` | 32 | 128 | 64 | 4 | 3 | 4 |
| `M > 64` | 128 | 64 | 64 | 4 | 3 | 1 |

布局尺寸作为显式 `tl.constexpr` 参数传入 kernel，避免 Triton JIT 捕获 Python 全局变量。

## 正确性

远端 GPU 回归作业 `179754`：

```text
7 passed in 7.83s
```

提交前相关全套回归作业 `180128` 为 `23 passed, 5 warnings in 10.10s`；warning 仅来自
Triton 针对 Python 3.15 的 `AnnAssign` 弃用提示。

测试覆盖：

- canonical -> blocked -> canonical 的逐字节完全一致；
- 显式 sentinel 的 N/K block 映射和低/高 nibble 顺序；
- ragged N/K、group tensor 和零点往返；
- FP16/BF16 Triton kernel 对 canonical 反量化 reference 的数值误差。

最终微基准作业 `179936` 的三个独立正确性形状最大绝对误差分别为
`0.015625`、`0.0078125`、`0.03125`。完整生成结果还与上一轮 continuous batching
结果做了结构化 A/B：忽略 metrics 中的时延后，10 个请求的 token ids、文本、token
数量和 finish reason 全部一致。

完整 62 序列质量作业 `180064` 评测 20,418 个 token：

| 指标 | 结果 |
| --- | ---: |
| mean NLL | 2.3948107879 |
| reference mean NLL | 2.3084555301 |
| delta NLL | **0.0863552578** |
| 质量门槛 | **通过，< 0.1** |

## 微基准

旧布局基准作业为 `179579`，最终 blocked kernel 作业为 `179936`。所有时间均为同一类
H800 PCIe 10 GiB MIG 上的 CUDA Event 时间。

| 形状 | 旧 kernel | 新 kernel | 加速 |
| --- | ---: | ---: | ---: |
| decode gate, `1x3840 -> 15360` | 2.185824 ms | 0.396904 ms | **5.51x** |
| decode down, `1x15360 -> 3840` | 3.441714 ms | 0.569146 ms | **6.05x** |
| prefill gate, `M=128` | 3.990707 ms | 0.604909 ms | **6.60x** |

大 M 是本轮新增覆盖。`128x64` 相对本轮初版 `64x128` 的结果为：

| 形状 | `64x128` | `128x64` | tile 加速 | 相对物化反量化 reference |
| --- | ---: | ---: | ---: | ---: |
| gate, `M=1500` | 10.720032 ms | 6.581216 ms | 1.63x | 1.37x |
| gate, `M=2000` | 13.772032 ms | 8.666229 ms | 1.59x | 1.28x |
| down, `M=1500` | - | 6.484843 ms | - | 1.37x |
| down, `M=2000` | - | 8.543840 ms | - | 1.36x |

reference 每次物化约 355.7 MB BF16 权重；融合 kernel 的额外峰值只包含输出，例如
decode gate 为 30,720 B，`M=2000` gate 为 61,440,000 B。

## 完整 10 请求正式成绩

最终作业 `179945` 直接读取仓库 `config.yaml`，完整运行 10 个请求：

```text
requests=10
generated_tokens=320
elapsed_s=49.23441375599941
generated_tokens_per_s=6.499518844397872
```

| 版本 | 作业 | elapsed_s | tokens/s | 相对 continuous 基线 |
| --- | ---: | ---: | ---: | ---: |
| continuous 基线 | 178631 | 111.395758 | 2.872641 | - |
| blocked 初版 `64x128` | 179795 | 55.622783 | 5.753038 | 2.00x |
| blocked 最终 `128x64` | 179945 | **49.234414** | **6.499519** | **2.26x** |

最终版本节省 `62.161344 s`，总时间下降 **55.80%**，吞吐增加 **126.26%**。本轮已
超过路线图 Round 1 的 `<=65 s` 阶段门槛，但仍比最终 `<=36 s` 目标多 13.234 秒。

## 完整 10 请求 Profile

作业 `179995` 先在正式计时外预热真实形状，再对完整 10 请求做 CUDA-only profile。
原因是集群单作业主机内存上限为 24 GiB；同时收集全部 CPU ATen 事件会在聚合 full10
trace 时超限。CUDA-only 仍完整保留 kernel、Memcpy、调用次数、wall time 和峰值显存。

```text
wall_time_s=43.34748497401597
generated_tokens_per_s=7.382204531400597
peak_allocated_bytes=6071380992
peak_reserved_bytes=6639583232
Self CUDA time total=52.061s
```

| CUDA 热点 | Self CUDA | 占比 | 调用数 |
| --- | ---: | ---: | ---: |
| pinned H2D | **23.554 s** | **45.24%** | 2,544 |
| `_w4a16_gemm_kernel` | **20.557 s** | **39.49%** | 17,384 |
| `_flash_attention_kernel` | 2.908 s | 5.59% | 2,544 |
| 两类主要 elementwise kernel | 1.908 s | 3.67% | 84,217 |
| embedding/LM head GEMM | 0.464 s | 0.89% | 53 |

Self CUDA 总和高于 profile wall time，说明双 buffer 仍在覆盖一部分 H2D 与 W4 计算。
优化后 pinned H2D 已成为第一热点。当前每次完整模型遍历仍搬运约 5.8 GB，6 次 prefill
和 47 次 decode 共约 307 GB；只继续调 W4 或 attention 都无法跨过 36 秒。

## 下一轮决策

下一轮按既定路线同时解决显存容量与搬运下界：

1. 将 Paged KV cache 改为所有 request slot 共享的物理 block pool，按 workload 总 block
   budget 分配，保留 block size 16；
2. 用释放的显存让 48 层 MLP packed INT4 权重常驻 GPU，只异步 offload attention
   projection；
3. 把双 staging buffer 从整层 payload 缩到 attention payload，并在当前层 MLP 计算时
   预取下一层 attention；
4. 完整 10 请求验收 H2D bytes、峰值显存和 `elapsed_s`，目标先降到 38--42 秒，再根据
   profile 决定 layer-major prefill 或算子融合的优先级。

## 本轮产物

- `src/hpc101_infer/quantization/packing.py`：blocked layout 双向转换；
- `src/hpc101_infer/layers/linear.py`：旧 checkpoint 的运行时无损 bridge；
- `src/hpc101_infer/kernels/w4a16.py`：N-contiguous W4A16 Triton kernel；
- `tests/test_w4a16_layout.py`：布局和 nibble 回归；
- `scripts/profile_w4a16.py`：decode/prefill gate/down 微基准；
- `scripts/profile_continuous_batching.py`：24 GiB 可运行的 full10 CUDA-only profile；
- `results/w4-blocked-public-generation.jsonl` 与 summary/quality 文件。
