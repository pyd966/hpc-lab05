# Lab5 Gemma4-12B 端到端优化报告参考稿

日期：2026-08-27

最终实现基线：`a722d7e`

> 更易读的重写版见 `reports/lab5-final-optimization-report-plain-2026-08-28.md`。
> 当前文件保留完整技术细节、作业号和 profile 证据，适合作为数据附录。

实验目标：在 1/7 张 H800、约 10 GiB 可见显存上，使公开 10 请求的正式
`elapsed_s < 36 s`，同时保持 `delta_nll < 0.1`。

> 本文不是按 Git 提交时间排列的开发流水账，而是按瓶颈迁移重新组织的报告参考稿。
> 主线是：先保证 W4A16 质量，再解决 OOM；随后消除大临时张量并提高跨请求复用；
> 当 W4A16 成为热点后修正权重布局；最后把节省出的 KV 显存转化为权重常驻空间，
> 从根本上减少 PCIe 搬运。

## 1. 摘要

本实验实现了 GPTQ W4A16 量化、异步权重 Offloading、Ring/Paged KV Cache、自写
Triton 反量化-GEMM、自写 Triton Flash/Paged Attention、Continuous Batching、
N-contiguous blocked INT4 布局、共享物理 KV pool 和混合权重常驻。

初始 g64 版本在完整 `performance_public.jsonl` 上会在长 prompt prefill 阶段 OOM。
双 GPU buffer 版本首次完成全部 10 个请求，但耗时 `425.020 s`，且仍出现可恢复的
allocator warning。最终稳定配置连续运行三次，结果分别为：

| 正式作业 | `elapsed_s` | 生成吞吐 |
| ---: | ---: | ---: |
| `181028` | 31.907938 s | 10.028853 token/s |
| `181162` | 31.679216 s | 10.101260 token/s |
| `181188` | 31.238719 s | 10.243698 token/s |

三次中位数为 **31.679 s**，最大值为 **31.908 s**，均小于 36 秒。相对第一个可运行
版本，最终中位数约加速 **13.42x**；相对 Continuous Batching 基线 `111.396 s`，
总时间下降 **71.56%**，加速 **3.52x**。最终质量结果为：

```text
mean_nll           = 2.394810787939536
reference_mean_nll = 2.3084555301012055
delta_nll          = 0.08635525783833042
```

因此任务一进入 `delta_nll < 0.1` 的满分区间，任务二也进入 `elapsed_s < 36 s` 的目标
区间。最终 28 项测试全部通过；三次正式生成的 token id、文本、token 数和结束原因完全
一致，且没有 allocator OOM warning。

## 2. 实验约束、平台与测量口径

### 2.1 评测约束

实验文档规定：

- 性能入口必须是 `scripts/run_generation_queue.py`；
- 设备为 H800 MIG `1g.10gb`，只能使用约 10 GiB 显存；
- 所有请求在推理开始前进入等待队列，推理期间无新请求到达，允许组成 batch；
- 不得修改模型结构、计时区间或跳过必要计算；
- 融合算子必须使用 Triton 或 TileLang 自行编写，不得调用 FlashAttention、xFormers、
  bitsandbytes、cuBLASLt 封装或 PyTorch SDPA；
- 质量主指标为 `delta_nll`，性能主指标为完整 10 请求 summary 中的 `elapsed_s`。

公开性能数据的实际形状为：

| 项目 | 分布 | 实际总量 |
| --- | --- | ---: |
| prompt | 250/500/1000/1500/2000，各 2 条 | 10,500 token |
| output | 16 x 3、32 x 4、48 x 3 | 320 token |

最终配置使用 batch 10、最大序列长度 2048、BF16 activation、GPTQ INT4 group 64、
seed 42 和同步指标计时。

### 2.2 远程硬件与优化含义

硬件由作业 `167992` 在 `lab5` 分区实测，不使用开发机信息代替：

| 资源 | 实测信息 | 对优化的直接影响 |
| --- | --- | --- |
| CPU | 2 x Xeon Gold 5418Y；任务配额 4 vCPU | CPU 线程数不能按节点可见的 96 线程设置 |
| CPU cache | L1d 48 KiB/core，L2 2 MiB/core，L3 45 MiB/socket | 打包和量化数据应连续并避免跨 NUMA |
| CPU ISA | AVX2、AVX-512 BF16/VNNI/FP16、AMX BF16/INT8 | 离线量化有向量化潜力，但本轮热点在 GPU 推理 |
| GPU | H800 PCIe，Compute Capability 9.0 | 自写 Triton kernel 面向 Hopper 路径 |
| MIG | `1g.10gb`，14 SM，9984 MiB 可见显存 | 不能按完整 H800 的 SM 数和显存做预算 |
| GPU L2 | 6 MiB | decode 很难缓存完整权重，连续访存和减少重复读更重要 |
| Copy Engine | 1 个 | 增加 H2D stream 不会增加物理传输带宽 |
| PCIe | 当前 Gen4 x16 | Offloading 必须批量、pinned、异步并尽量减少总字节数 |
| 软件 | PyTorch 2.13.0+cu132，Triton 3.7.1 | 所有核心融合算子由项目内 Triton 实现 |

这里最重要的硬件事实不是“它是一张 H800”，而是当前进程只看到 **14 个 SM、6 MiB
L2 和一个 copy engine**。这解释了为什么 decode 小矩阵容易被权重读取和 launch 开销
限制，也解释了为什么最终选择减少 H2D 字节，而不是继续增加传输 stream。

### 2.3 证据口径

本文把数据分成四类，避免混用：

1. **正式成绩**：无 profiler，完整 10 请求、320 个输出 token，使用 summary 的
   `elapsed_s`。只有这一项可用于判断是否达到 36 秒。
2. **完整 10 请求 profile**：先预热真实 shape，再采集 CUDA-only 事件，用于归因。
   预热后的 wall time 不等于正式成绩。
3. **small/单请求 profile**：用于早期定位热点，不能代替正式 10 请求结果。
4. **microbenchmark**：使用 CUDA Event 比较同一算子和同一形状，只说明局部 kernel
   收益，必须再由端到端结果验证。

不同 CUDA stream 可以并行，因此 profiler 中各条目的 Self CUDA 时间之和可能大于
wall time。本文不会把 H2D 和 kernel Self CUDA 简单相加，也不会用 warmed profile 的
`26.382 s` 代替最终正式中位数 `31.679 s`。

### 2.4 固定量化 checkpoint，避免重复 GPTQ 干扰实验

**触发证据。** 一次真实 GPTQ 构建作业 `175438` 需要约 18 分 8 秒。后续每轮只改
Offloading、KV cache、scheduler 或 kernel 时，重新量化既浪费集群时间，也会引入
checkpoint 不一致的实验变量。

**目的与作用对象。** 这不是计时区间内的推理优化，而是实验可复现性和开发效率优化。

**实现。** 长期 checkpoint 固定在：

```text
/home/h3250106394/quantized/gemma-4-12b-gptq-group64
```

缓存键覆盖源模型文件签名、完整量化配置、校准 token 哈希、micro batch、token 上限和
分片配置；输出侧校验 manifest、配置、safetensors header 和分片签名。命中时不加载原始
12B 权重、不捕获激活、不运行 GPTQ；`--force` 才强制重算。

**结果与证据。** 完整 CLI 命中作业 `175606` 用时 `11.737 s`，并直接报告 328 个
Linear 已准备好。小 checkpoint cProfile 中，缓存键、读取和写入函数均只有约 1 至 3 ms
累计时间。OJ 仍会按实验文档重新执行量化，因此缓存没有绕过提交要求。

## 3. 总体策略：根据瓶颈迁移确定优化顺序

本文第 4 至 9 节的二级章节都先描述一个系统级目的；其下三级标题才是为达到该目的采用的
具体手段。每项手段均按“触发证据 -> 目的 -> 实现 -> 结果/Profile 验证”展开，而不是按
Git 提交时间罗列。

### 3.1 从“先可运行”到“稳定小于 36 秒”

整个过程可以抽象为连续的瓶颈迁移：

```text
BF16 权重放不下
  -> GPTQ W4A16 保证容量和质量
完整请求 prefill OOM
  -> 异步 Offloading + 双 buffer 首次恢复完整队列可运行
KV 容量与请求回收仍不适合 batch
  -> Ring/Paged KV 建立稳定容量与分页生命周期
reference 反量化和 attention 大临时张量
  -> 自写 fused W4A16 + Flash/Paged Attention
batch 1 重复执行 48 层并反复搬权重
  -> Continuous Batching + 稳定 KV slot + 去同步
W4A16 kernel 成为第一计算热点
  -> blocked INT4 layout + prefill/decode 专用调度
W4 变快后 PCIe H2D 成为第一系统热点
  -> 共享 KV pool 释放显存 + 44/48 层 MLP 常驻
  -> 正式 10 请求稳定小于 36 秒
```

这个顺序体现了三个原则：

- **先可运行，再谈加速**：OOM 时任何 kernel 加速都没有正式成绩。
- **每轮跟随新 profile 热点**：不能一直优化已经降到 1% 以下的算子。
- **把容量优化转化为吞吐收益**：省下显存后要用于更大 batch 或权重常驻，否则只是
  “显存数字更小”，不会自动缩短端到端时间。

### 3.2 基线证据：确定容量、算子与调度的优先级

初始 g64 基线使用 batch 1、静态调度、静态 KV、eager attention 和
`int4_reference`。small4 可以完成，耗时 `76.138967 s`、吞吐 `0.788033 token/s`；
完整 public 的 g64/g128 均在长 prompt prefill OOM，因此没有合法 baseline
`elapsed_s`。

small 单请求 profile 作业 `169188` 的 Self CUDA 为 `40.419 s`：

| 热点 | Self CUDA | 现象解释 |
| --- | ---: | --- |
| `aten::copy_` | 18.943 s | unpack、dtype 转换和临时权重 copy 很多 |
| 两类 elementwise | 9.538/9.280 s | INT4 位运算、scale/zero 反量化被拆成小 kernel |
| `aten::mul` | 9.073 s | 完整权重反量化的逐元素乘法 |
| `aten::mm` | 3.524 s | GEMM 反而不是第一热点 |

代码观察同时发现 eager attention 会物化 `scores`、mask 和 probability，静态 KV 又按
最大 batch/最大长度预留。这给出三条初始假设：

1. 先降低权重和 KV 常驻量，使完整请求不再 OOM；
2. 融合反量化与 GEMM，消除约数百 MiB 的临时高精度权重；
3. 长期必须批量推进请求，否则每生成一个 token 都重复搬运和执行全部 48 层。

### 3.3 正式性能演进

| 阶段 | 完整 10 请求 `elapsed_s` | tokens/s | 结论 |
| --- | ---: | ---: | --- |
| 初始 g64 baseline | OOM | - | 静态 KV 与 eager 临时张量无法通过长 prompt |
| 双 buffer + 合并 Offload | 425.019842 s | 0.752907 | 首次跑完，但有 allocator warning |
| 自写 fused W4A16 | 400.511 s | 0.79898 | 反量化临时权重消失，仍为串行 batch 1 |
| 自写 Flash/Paged Attention | 366.652 s | 0.87276 | 不再物化 attention scores，仍被 W4/H2D 主导 |
| Continuous Batching | 111.395758 s | 2.872641 | 相对串行下降 69.62% |
| Blocked W4A16 | 49.234414 s | 6.499519 | 相对 continuous 再下降 55.80% |
| 共享 KV + 44 层 MLP 常驻 | 31.679216 s（中位数） | 10.101260 | 三次全部通过 `<36 s` |

Ring KV、第一版 Paged KV 等容量优化没有独立 full10 A/B，因此不在表中虚构加速；其
small 结果和显存结果在后文单独报告。

后文按系统目的而不是开发时间组织。下表汇总每个目标内部使用的触发证据，不表示严格的
开发时间顺序；真实依赖顺序以 3.1 节为准：

| 系统目的 | 进入该阶段的主要证据 | 采用手段 |
| --- | --- | --- |
| 建立显存容量基础 | public full10 在长 prompt prefill OOM | GPTQ、异步 Offloading、Ring/Paged KV、共享物理 pool |
| 消除大临时张量和无效计算 | baseline 中 copy/反量化 elementwise 高于 `mm`，并会物化 attention scores | fused W4A16、Flash/Paged Attention、last-token logits |
| 提高跨请求复用 | batch 1 约需 320 次模型遍历；`nonzero` 2,880 次、CPU total 30.186 s | Continuous Batching、稳定 slot、确定性 token range |
| 针对真实 batch shape 优化 W4 | Continuous 后 W4A16 占 59.91%，FlashAttention 仅占 0.15% | blocked INT4、group64 scale 广播、prefill/decode 专用 tile |
| 减少 PCIe 搬运 | 初版 Offload H2D/D2H 各 3.3 万次；blocked W4 后 H2D 占 45.24% | 双 buffer、合并 payload、RoPE 去重、44 层 MLP 常驻 |
| 稳定达到 36 秒 | 48 层常驻约 28 秒但出现 allocator warning | 保留显存余量、提前初始化 cuBLAS、三次 full10 验收 |

## 4. 目标一：压缩权重与 KV，建立 10 GiB 显存容量基础

**阶段目的。** 初始 public full10 在长 prompt prefill 阶段 OOM，因此容量优化不以秒数为
首要指标，而是压缩权重、KV 和分配峰值，为完整负载、后续 batching 与权重常驻建立
容量基础。

### 4.0 先统一显存测量口径

本节同时出现三类数字，必须分开解释：

- **进程峰值 allocated** 是一次运行中 PyTorch 实际分配过的最高显存；
- **KV 物理池容量** 是初始化时真实创建的 K/V tensor 与 metadata 大小；
- **活跃 block 数** 只表示请求当前占用了 pool 中多少页。`release()` 把页面放回内部
  free list，不等于把底层 tensor 归还 CUDA allocator。

显存演进如下。表中 Offloading 与最终峰值使用的 workload 不同，因此不能把它们直接
相减归因；组件级 KV 数字则使用相同模型配置，可以做严格 A/B。

| 优化 | 测量对象 | 优化前 | 优化后 | 变化与结论 |
| --- | --- | ---: | ---: | --- |
| 异步 Offloading | 同一短输入的进程峰值 allocated | `8,252,107,776 B` | `2,673,564,672 B` | -67.60% |
| Ring KV | batch 1、2048 token 的 KV tensor | `704,643,456 B` | `369,099,136 B` | -320 MiB，-47.62% |
| 第一版 Paged+Ring | batch 1，与连续 Ring 比较 | `369,099,136 B` | `374,371,008 B` | **+5,271,872 B；没有省物理显存** |
| 共享 Paged pool | batch 10 的 KV tensor+metadata | `3,743,710,080 B` | `2,982,443,904 B` | -761,266,176 B，约 -726 MiB |
| 最终 44 层常驻 | full10 进程峰值 | - | allocated `9,742,262,784 B`；reserved `10,039,066,624 B` | 节省的 KV 空间被用于权重常驻，三次运行无告警 |

这张表也说明显存优化不是单调追求更低的最终 allocated。最终配置主动把空出的容量用于
44 层 MLP 常驻，使进程峰值高于 Continuous Batching 阶段，但显著减少 PCIe H2D。

### 4.1 使用 GPTQ W4A16 group64 压缩权重并控制量化误差

**触发证据。** Gemma4-12B 的 BF16 权重本身已经超过 10 GiB，无法在本 MIG 实例上全量
加载。普通 RTN 只最小化权重数值误差，没有根据校准激活区分输入通道的重要性，INT4
下质量损失偏大。早期 group 128 虽稍省 scale 并略快，但实测 `delta_nll` 约 0.121，
没有进入用户要求的 `<0.1` 区间。

**目的与作用对象。** 这项优化针对模型权重容量和量化误差，不直接针对在线推理
kernel。目标是在 W4A16 下保留足够精度，同时为后续 Offloading 和融合 GEMM 提供 packed
INT4、scale 和 zero point。

**实现。** 对每个 Linear 收集校准激活，以 FP32 构造 Hessian；处理 dead column，加入
`damp_percent=0.01` 的对角阻尼，使用 Cholesky 得到逆 Hessian 因子。随后以 128 列为
计算 block、以 64 列为量化 group，逐列量化并把当前误差传播到 block 内剩余列和后续
block。量化采用 4 bit、对称量化、FP16 scale，并让当前层量化后的输出继续作为下一层
校准输入。

**结果。** 最初 reference 路径的 g64 结果为 `delta_nll=0.0865803564`。更换自写
Triton attention 的数值归约路径后，最终完整 62 序列、20,418 token 结果为
`0.0863552578`。两者都小于 0.1，且后续布局优化只做可逆字节重排，没有重新求量化值。

**证据判断。** NLL 是这项优化的主证据，在线 profile 不适合评价离线 GPTQ。最终报告
应明确区分“GPTQ 决定量化质量”和“后续 kernel 决定如何高效使用同一组量化值”。

本阶段接着压缩运行时权重与 KV 的持久占用；判断成功的首要标准是完整负载不再 OOM，
容量型手段不强求在短请求上直接加速。

### 4.2 按 DecoderLayer 执行异步权重 Offloading

**触发证据。** 全量 GPU 权重的短输入峰值 allocated 为 `8,252,107,776 B`，再叠加长
prompt 的 KV、attention 和 logits 临时量后无法留在 10 GiB 内。

**目的与作用对象。** 针对 DecoderLayer 的持久权重占用，用 PCIe 带宽换 GPU 容量，
使长请求有机会运行。它的首要目标是“能跑”，不是单独保证变快。

**实现与时序。** CPU 上保存每层 packed INT4 权重和量化参数，并启用 pinned memory。
Pinned memory 是不会被操作系统换出的页锁定主存，CUDA DMA 可以直接访问，因此
`non_blocking=True` 的 H2D 才能真正与 GPU 计算异步。实现使用：

- 1 条当前 PyTorch compute stream；
- 1 条独立 transfer stream；
- 每层 1 个 ready event 和 buffer 复用所需的 compute-done event。

时序为：transfer stream 先搬 layer 0；compute stream 使用 layer `i` 前等待其 ready
event；在发射 layer `i` 计算前先向 transfer stream 提交 layer `i+1` 的预取；layer `i` 完成后
记录 compute-done event。若下一层搬运来不及，compute stream 会在 ready event 上等待，
但不会发生 CPU 端主动同步。第一版释放层时仍把权重 D2H 回 CPU。

**结果。** 短输入峰值从 `8.252 GB` 降到 `2.674 GB`，下降约 67.6%，输出
`max_abs_diff=0`。这证明容量目标达成。

**Profile 证据。** small 单请求 profile 中 H2D 和 D2H 各约 `192.864 GB`，Self CUDA
分别为 14.387 s 和 14.136 s；`aten::copy_` 占 68.55%。这说明异步 Offloading 把原来的
显存问题转化成了 PCIe 搬运问题，端到端并未加速。

### 4.3 使用 Ring KV Cache 限制滑动层的 KV 容量

**触发证据。** Gemma4 有 40 个 sliding attention layer，窗口固定为 1024；这些层保存
更早 KV 没有任何后续用途。原实现却和 8 个 global layer 一样按 2048 保存完整前缀。

**目的与作用对象。** 针对滑动层 KV 的持久显存，把序列长度因子从 `S` 限制为
`min(S,1024)`，同时保持 8 个 global layer 的长程语义不变。

**实现。** sliding 层用 `position % capacity` 写环形物理槽；读取时按绝对逻辑位置恢复
正确顺序。causal、padding 和 sliding mask 全部基于绝对位置，而不是环形下标。超过窗口
的 prefill 只保留最新 1024 个 KV。

**结果。** batch 1、max length 2048 时，KV allocation 从 `704,643,456 B` 降到
`369,099,136 B`，节省 320 MiB，即 47.62%。prefill 最后位置误差为 0，追加 decode
token 最大误差为 `1.19e-7`。

**Profile 证据。** 短请求没有越过 1024 窗口，Ring off/on 为 `52.065 s/53.467 s`，
开启后慢约 2.7%；profile 中 Ring gather 没进入主要热点，H2D 和 reference 反量化仍居前。
这不是失败，而是说明 Ring 是容量型优化，短上下文本来就没有计算量收益。

### 4.4 使用 16-token Paged KV Cache 实现按页分配与回收

**触发证据。** 连续 KV 会为每个 slot 预留最大序列长度；请求长度从 250 到 2000，静态
预留会造成大量内部空洞，也无法在请求完成后把物理空间交给其他请求。

**目的与作用对象。** 针对不同长度请求的 KV 分配和回收，为后续 Continuous Batching
建立稳定 slot、按需 page 和完成即释放的内存管理基础。

**Block size 选择。** 采用 16 token/block。vLLM 默认使用 16，常见 Paged Attention
后端支持 16/32；TensorRT-LLM 要求 tokens-per-block 为大于 1 的 2 次幂。16 还能整除
1024 窗口，每请求末页最多浪费 15 token，页表开销和内部碎片较平衡。

**实现。** 每层 K/V 为
`[physical_blocks, block_size, kv_heads, head_dim]`，请求用 block table 完成逻辑页到
物理页的映射，首次访问时分配、完成时回收。Ring 与 Paged 同时开启时保留一个边界页，
解决未对齐窗口最多跨 65 页的问题。第一版为了先验证功能，attention 前仍用
`index_select` 把离散页收集成连续 K/V。

**这里必须区分管理粒度和物理容量。** 第一版的物理 tensor 仍按
`max_batch_size * max_blocks_per_sequence` 一次性预分配。完成请求时，`release()` 只是
把物理页号放回该 layer 的 free list，供后续请求复用；它不会缩小 K/V tensor，也不会
把显存归还 PyTorch allocator。因此 Paged 本身这一轮没有降低进程峰值。

**结果。** batch 1 的连续 Ring cache 为 `369,099,136 B`，Paged+Ring 为
`374,371,008 B`，后者因 block table 和未对齐 Ring 的边界页反而多 `5,271,872 B`
（约 5.03 MiB）。若与完全关闭 Ring 的 `704,692,608 B` 比，Paged+Ring 确实低
46.875%，但这个差值主要来自上一节 Ring 对 40 个 sliding layer 的容量截断，不能归因
于分页。跨 3 页且发生 Ring page 复用的测试中最大误差为 `5.96e-8`。

**Profile 证据。** small 单请求 profile 的端到端时间从 Ring 版本的 `51.413 s` 变为
`53.973 s`，慢 4.98%；Self CUDA 从 `54.762 s` 增到 `56.691 s`。这符合“先 gather
再计算”的功能版会增加整理开销。它的价值是建立 block table、pool 内页面复用和请求
完成释放的语义；真正的容量下降要等下一节缩小物理 pool，速度收益则要等 attention
kernel 直接读取 block table。

### 4.5 使用跨请求共享的物理 KV block pool 缩小真实预留

**触发证据。** 第一版 Paged allocator 虽然按需分配逻辑 page，但物理 tensor 仍按
`max_batch_size * max_blocks_per_sequence` 初始化。batch 10 时 KV tensor 和 metadata
仍占 `3,743,710,080 B`。也就是说，分页语义正确，却没有把未使用物理页归还给
PyTorch allocator。

**目的与作用对象。** 针对 KV 的物理容量，把 workload 分布允许的共享空间释放出来，
并将这部分显存用于 packed MLP 权重常驻，间接降低 H2D。

**实现。** block size 继续为 16，固定 block table 形状不变，只缩小每层物理 K/V pool。
根据题目给出的输入/输出区间上界，每个 global layer 配 776 blocks，
每个 sliding layer 配 530 blocks。公开数据实际只需 680/495 blocks，因此仍分别保留
96/35 blocks 余量，而不是硬编码公开样本的精确长度。

**结果。** 新 pool 含 metadata 为 `2,982,443,904 B`，释放 `761,266,176 B`，约
726 MiB，即物理 KV 预留下降 20.33%。这才是 Paged 机制第一次转化为真实的 allocated
容量下降。请求完成后的页面仍只归还内部 pool；之所以能省进程显存，是因为初始化时创建
的 pool tensor 已经从 3.744 GB 缩到 2.982 GB。该子项和权重常驻在同一最终提交中，
没有独立 full10 A/B；可以证明容量按预期下降，但不能声称 17.555 秒的最终差值中有多少
由 pool 自身直接贡献。

### 4.6 使用生命周期 credit 与原子 reserve 保证小 pool 可用

**触发证据。** 缩小物理 pool 后，如果只在 decode 跨页时临时申请，可能运行到一半才
耗尽；旧 `reserve()` 还可能在前几层分配成功、后层失败，留下半提交状态。

**目的与作用对象。** 这项优化主要针对正确性和可预测性，使小物理池可以安全支持
Continuous Batching，而不是直接减少 kernel 时间。

**实现。** 请求进入 stable slot 前，按 `prompt_length + max_new_tokens` 一次预留完整
生命周期 credit。global 需求为 `ceil(max_length/16)`；sliding 需求上限为 65 页，因为
1024 窗口在未对齐时可能跨 65 页。容量不足时暂停新 prefill、继续 active decode；请求
完成后归还实际页和未来 credit。跨 48 层 reserve 先完整规划并检查，再原子提交。

因此真正的因果链是：Paged 提供逻辑页和 free list；lifecycle credit 防止较小 pool
过量接纳请求；原子 reserve 防止运行中半提交；请求完成 release 让页面在 pool 内复用；
最后由较小的预分配物理 tensor 把这种复用能力转化为 726 MiB 的真实显存下降。

**结果与证据。** 测试覆盖 1024/1025 边界、请求容量竞争、完成释放、slot refill 和
跨层失败不产生部分分配。它没有独立性能数字，但消除了最终配置在运行中途 OOM 或页表
损坏的风险。

## 5. 目标二：消除大临时张量和无效计算

**阶段目的。** 容量问题缓解后，baseline profile 显示 copy、INT4 位运算和逐元素反量化
远高于 `mm`，代码又会物化完整 BF16 权重与 attention scores。因此本阶段围绕“少写一次
大张量、少读一次无效数据”重写核心算子。

### 5.1 融合 INT4 反量化与 GEMM

**触发证据。** baseline profile 中 copy、位运算、`mul/sub` 明显高于 `mm`。代码中
`QuantizedLinear.forward()` 每次都执行
`packed INT4 -> 完整 BF16 权重 -> F.linear`。一个真实投影会产生约 355.9 MB 临时
高精度权重，decode 的 M 很小时，这次物化比矩阵乘本身更贵。

**目的与作用对象。** 针对所有 328 个量化 Linear，消除完整反量化权重的 global memory
写回/读回和小 kernel 发射，同时降低峰值显存。

**Kernel 设计。** 自写 Triton kernel 在 K tile 内完成 nibble 解包、scale/zero 读取、
BF16/FP16 转换和 `tl.dot`，使用 FP32 accumulator，最后融合 bias 并一次写回输出。
持久参数仍只有 packed INT4、scale、zero 和元数据，没有缓存完整 BF16 权重。初版根据 M
分成 decode、小 prefill 和大 prefill 三档 tile，使小 M 优先减少空线程，大 M 优先提高
Tensor Core 利用率。

**Microbenchmark。** 在真实 Gemma4 形状上：

| 形状 | reference | 初版 fused | 加速 | 临时显存变化 |
| --- | ---: | ---: | ---: | ---: |
| decode gate, `M=1,N=15360,K=3840` | 6.633 ms | 2.211 ms | 3.00x | 355.9 MB -> 30,720 B |
| decode down, `M=1,N=3840,K=15360` | 6.597 ms | 3.449 ms | 1.91x | 355.9 MB -> 7,680 B |
| prefill gate, `M=128` | 6.621 ms | 4.024 ms | 1.65x | 355.9 MB -> 3.93 MB |

**端到端结果。** 同一单请求从 `50.331 s` 降到 `31.866 s`，其中 Mean TPOT 下降 38.4%，
但 TTFT 反而增加 15.8%，说明初版 tile 对 decode 更有效。
完整 10 请求 batch 1 的跨版本观察值为 `400.511 s`，
相对最早可运行的双 buffer 版本 `425.020 s` 下降 5.77%。两者之间还加入了
Ring/Paged KV，且远程作业存在波动，因此不能把全部差值归因于 fused W4。
更直接的因果证据是上面的同请求后端 A/B、microbenchmark 和临时显存下降。

**Profile 证据。** 原来的反量化 elementwise 热点不再主导，计算集中为一个
`_w4a16_gemm_kernel`：17.972 s、48.53%；pinned H2D 为 18.141 s、48.99%。融合假设被
证实，同时暴露出两个新瓶颈：kernel 内部权重访存仍不理想，且每 token 仍搬全部层。

### 5.2 使用自写 Triton Flash/Paged Attention 避免物化 scores

**触发证据。** eager attention 依次物化 `QK^T` scores、mask 和 probability，空间和
访存均为 `O(L^2)`，这正是长 prompt OOM 的主要嫌疑之一。第一版 Paged KV 又需要先
gather 连续 K/V，尚未兑现分页寻址的计算收益。

**目的与作用对象。** 针对 full/sliding attention 的 prefill 和 decode：不在 global
memory 物化完整 scores/probability，直接从 block table 读取 Paged KV，并在 kernel 内
处理 GQA 和 mask。

**Kernel 设计。** 每次只计算 `16 x 32` QK tile，FP32 保存 row max、指数和与输出累加，
用 online softmax 跨 key tile 更新；GQA 直接计算 query head 到 KV head 的映射，不做
`repeat_kv`。causal、padding、global/sliding window 条件全部融合到 score tile，并跳过
理论上不需要的 key block。

**为降低寄存器占用进行 kernel 拆分。** sliding head dimension 为 256，global head
dimension 为 512。若一个 program 同时保存 512 维 FP32 output accumulator，寄存器压力
过高，容易降低 occupancy 或无法编译。因此输出维按 256 拆成两个 program：每个 program
只维护一半 accumulator。代价是 512 维 global head 会重复一次 QK 计算，但换来了可控的
寄存器 live range 和稳定执行。现有 profile 没有采集硬件 register/occupancy counter，
所以报告应把它表述为设计约束与成功运行证据，而不能声称“profile 证明寄存器下降了某个
百分比”。

**结果。** 相邻版本的完整 10 请求从 `400.511 s` 降到 `366.652 s`，观察到 8.45% 的
下降；但远程作业间 H2D 存在明显波动，不能把全部差值归因于 attention kernel。质量保持
`delta_nll=0.08635526`。峰值显存没有下降，是因为剩余 1 GiB warning 实际来自
`2000 x 262144 x 2 = 1,048,576,000 B` 的全 prompt BF16 LM-head logits，而不是 attention
scores。

**Profile 证据。** 单请求 warmed profile 中 Flash kernel 只有 `79.552 ms`，占 Self
CUDA 0.24%；W4A16 为 54.82%，H2D 为 42.86%。因此“不物化 scores”已经成功，继续调
attention tile 不再是当时的最高优先级。

### 5.3 Prefill 仅对最后有效 token 计算 LM-head logits

**触发证据。** FlashAttention 后仍出现恰好 `1,048,576,000 B` 的分配失败提示，其大小
等于 `2000 prompt x 262144 vocab x 2 bytes`。生成首 token 只需要每条 prompt 最后有效
位置的 logits，却先计算了所有 prompt token 的词表投影。

**目的与作用对象。** 针对 LM head 的长 prompt 临时显存和无效 GEMM，释放 batch 10
所需容量。

**实现。** 在最终 RMSNorm 后先按每行有效长度 gather 最后一个 hidden state，再执行
LM head，只生成 `[batch,1,vocab]` logits。

**结果与证据。** Continuous 版本正式峰值为 `6,297,783,808 B`，不再出现此前 1 GiB
long-prompt logits warning。该修改与 continuous scheduler 同轮完成，没有独立 full10
A/B，因此只能严谨地声称“精确消除了该 1 GiB 临时张量”，不能把某个端到端秒数全部
归给它。Continuous 轮 warmed small4 profile 中 embedding/LM head 只占 0.73%，说明
它已不是主要热点。


## 6. 目标三：提高跨请求复用，消除调度中的 CPU-GPU 气泡

**阶段目的。** 自写算子后 full10 仍需 `366.652 s`；batch 1 会为 320 个输出 token 反复
遍历 48 层并搬运权重。本阶段让一次模型遍历服务多个请求，同时删除 profiler 暴露的
host 同步点。

### 6.1 使用 Continuous Batching 合并请求并压紧 active rows

**触发证据。** batch 1 的完整 10 请求需要大约 320 次完整模型遍历；每次都执行 48 层
并重新搬权重。输出长度又有 16/32/48 三档，静态 batch 会让短请求在结束后继续占槽，
产生拖尾。

**目的与作用对象。** 针对请求调度和 decode 的有效 M，提高一次权重加载服务的请求数，
让完成请求立即释放 KV page，并使 W4A16 从 GEMV 向小 GEMM 转变。

**实现。** `ContinuousBatchScheduler` 维护 FIFO pending queue、稳定 slot 和紧凑 active
rows。配置 `max_batch_size=10`、`scheduler_batch_size=10`、prefill token budget 2048。
预算按 `batch_size * max_prompt_length` 计算，因此公开数据形成 `4,2,1,1,1,1` 六个
prefill batch。所有请求获得稳定 KV slot 后，decode 每步只包含尚未完成的请求；完成后
立即释放 page 和 slot。

在本公开评测中，所有 10 个请求在开始前已经到达，因此主要收益不是在线到达请求补位，
而是：把 10 个请求共同推进、在输出长度从 48 降到 32/16 时压紧 active rows，并把
完整模型遍历数降到 6 次 prefill 加 47 次 decode，即 53 次。

**结果。** 首个可运行版本为 `133.088 s`；进一步去同步后为 `111.395758 s`，相对串行
Flash 基线 `366.652 s` 下降 69.62%，吞吐提高 3.291x。

**为什么收益不是 batch size 的 10 倍。** 自回归 decode 仍要逐 token 串行；六个不同
长度 prefill 不能合成一次；active batch 会从 10 逐步降到 7、再降到 3；大 M W4 kernel
本身计算更多；H2D 只能被摊薄而不会完全消失。因此 batch 10 表示最大并发槽位，不等于
端到端必然加速 10 倍。

### 6.2 使用确定性 token range 消除每层 `nonzero` 同步

**触发证据。** 初版 continuous 路径每层用 GPU 布尔索引删除 padding，profile 中
`aten::nonzero` 调用 2,880 次，CPU total 达 30.186 s。该操作需要把动态索引信息返回
host，造成 CPU-GPU 同步，也破坏 H2D 与 W4 计算重叠。

**目的与作用对象。** 针对 scheduler/KV 写入路径的外部 bubble，消除数据依赖型 host
同步。

**实现。** scheduler 在 CPU 已经知道每个请求的 `(start,end)` token range，因此把这个
范围一直传到 KV reserve/write，用确定性 range 构造 source index，不再从 GPU mask 反推。

**结果与 profile。** `aten::nonzero` 从最终热点表消失。同口径 warmed small4 profile
wall time 从 `37.745 s` 降到 `25.809 s`，下降 31.62%；完整 10 请求从初版 continuous
`133.088 s` 进一步降到 `111.396 s`。两组变化还包含同轮其他整理，因此不能把全部差值
只归因于 `nonzero`，但调用消失直接证明同步点已被移除。

## 7. 目标四：针对真实 batch shape 提高 W4A16 的访存与计算效率

**阶段目的。** Continuous Batching 改变了 W4A16 的真实 M 分布；此时 small4 profile 中
W4A16 占 `59.91%`，已经成为第一计算热点，而 FlashAttention 只有 `0.15%`。因此本阶段
不再继续优化 attention，而是针对新的 prefill/decode shape 重排权重并拆分调度。

### 7.1 将 canonical INT4 重排为 N-contiguous blocked layout

**触发证据。** 初版 qweight 为 canonical `[N,K/2]`。一个 CTA 同时计算 BN 个输出通道
时，相邻 N lane 地址相隔 `K/2` 字节，无法形成合并访问。MIG 只有 6 MiB L2，decode
也不可能依赖缓存掩盖这种跨行读取。

**目的与作用对象。** 针对 fused W4A16 的 global memory 访问，把 CTA 需要的 N 维
权重和 scale 排成连续地址，减少 memory transaction 和重复反量化。

**实现。** 运行时把长期 canonical checkpoint 无损重排为：

```text
qweight: [K/64, N/128, 32, 128]
Qb[k//64, n//128, (k%64)//2, n%128]

scales/zeros: [K/group_size, N/128, 128]
Sb[k//group_size, n//128, n%128]
```

低 nibble 仍表示偶数 K，高 nibble 仍表示奇数 K。全部 328 个正式矩阵的 K 都是 64 的
倍数、N 都是 128 的倍数，因此没有 padding 增长。转换发生在正式计时和 pinned payload
建立之前，不改变 GPTQ 数值，也不缓存 BF16 权重。

**结果与证据。** canonical 与 blocked layout 往返逐字节一致，全部正式矩阵都无需
padding，且最终质量仍为 `delta_nll=0.0863552578`。布局重排和下一节的 kernel 调度属于
同一性能轮，没有独立 full10 A/B；因此端到端收益在下一节按组合结果报告。

### 7.2 对齐 group64 scale，并为 prefill/decode 使用专用调度

**触发证据。** 初版 kernel 即使完成融合，仍为同一 group 重复加载 scale，而且一个 tile
配置同时服务 `M=1` decode 和 `M=2000` prefill，无法兼顾空线程、权重复用和 accumulator
规模。

**目的与作用对象。** 让 `BLOCK_K=64` 与量化 group 完全对齐，每个 K tile 只加载 BN
个 scale 后沿 K 广播；同时按 M 拆分执行配置，控制寄存器/线程块规模并提高不同阶段的
有效并行度。

**最终配置。** decode 使用 `16x128x64`；中等 M 使用 `32x128x64`；大 prefill 使用
N-fast `128x64x64`。相对 `64x128x64`，大 prefill tile 保持每 CTA 输出元素数不变，
但让一次解包后的权重服务两倍 M 行，权重反量化次数减半。`GROUP_M=1` 还让相邻 CTA
优先遍历 N，以复用比 packed B tile 更大的 activation A tile。

**Microbenchmark。**

| 形状 | 旧 fused kernel | blocked kernel | 加速 |
| --- | ---: | ---: | ---: |
| decode gate, `M=1` | 2.185824 ms | 0.396904 ms | 5.51x |
| decode down, `M=1` | 3.441714 ms | 0.569146 ms | 6.05x |
| prefill gate, `M=128` | 3.990707 ms | 0.604909 ms | 6.60x |

大 prefill 从 `64x128` 改为 `128x64` 后，gate `M=1500` 从 10.720 ms 降到
6.581 ms，`M=2000` 从 13.772 ms 降到 8.666 ms。布局往返逐字节一致，三个独立形状
对 reference 的最大绝对误差仍为 0.015625、0.0078125 和 0.03125。

**端到端结果。** 初版 blocked 为 `55.622783 s`，最终 tile 为 `49.234414 s`；相对
continuous `111.395758 s` 下降 55.80%，吞吐提高 126.26%。

**Profile 证据与瓶颈迁移。** 完整 10 请求 warmed CUDA-only profile 中：

| 热点 | Self CUDA | 占比 |
| --- | ---: | ---: |
| pinned H2D | 23.554 s | 45.24% |
| W4A16 | 20.557 s | 39.49% |
| FlashAttention | 2.908 s | 5.59% |

W4 已显著下降；在 blocked full10 profile 中，H2D 明确成为剩余第一系统热点。53 次
完整模型遍历仍各搬约 5.799 GB，总量约 307.37 GB。这个 profile 直接决定最后一轮
不再只调 kernel，而是先减少必须搬的字节。

## 8. 目标五：减少 PCIe 搬运，并把节省的显存用于权重常驻

**阶段目的。** Offloading 解决容量却制造了 H2D 下界。这个系统目标在开发早期和最终阶段
分别处理两类问题：第一版 Offloading profile 暴露了大量小 copy 与无意义 D2H；
最终阶段的 blocked W4 full10 profile 则显示 pinned H2D 已以 `23.554 s/45.24%`
成为剩余第一热点。8.1 节降低 copy 次数，8.2 至 8.3 节减少每轮真正需要搬运的字节；
这里按共同目的归类，不表示三项连续开发。

### 8.1 使用双 GPU buffer、合并 payload 并消除只读权重 D2H

**触发证据。** 第一版为每个 qweight、scale、zero、norm 和 RoPE tensor 单独发射 copy，
profile 中 H2D 33,280 次、D2H 33,311 次。大量小 copy 带来 launch 和同步开销；同时推理
权重只读，D2H 没有语义必要。

**目的与作用对象。** 针对 Offloading 的外部 bubble 和临时分配，把“每层许多小 copy”
变成“每层一次大 copy”，并用固定地址支持稳定复用。

**实现。** 每层 tensor 被打包成一块连续 pinned `uint8` host payload，并保存 dtype、shape
和 byte offset。GPU 只分配两个最大层大小的 byte buffer，当前层和下一层交替占用。
释放时只把 module 参数切回 CPU view，transfer stream 在复用 buffer 前等待
compute-done event，不再执行 D2H。仍然只有一条 transfer stream，因为硬件只有一个
copy engine。

**结果。** 两个 buffer 总容量为 `257,531,912 B`，单次完整 prefill H2D 为
`5,799,469,248 B`，D2H 为 0。短测试峰值为 `2,682,359,808 B`，输出仍逐位一致。
完整 10 请求首次跑完，`elapsed_s=425.019842 s`，但出现一次可恢复 allocator warning。

**Profile 证据。** 同口径 small profile 从：

| 指标 | 第一版异步 Offload | 双 buffer + 合并 payload |
| --- | ---: | ---: |
| Self CUDA | 69.980 s | 55.307 s |
| H2D 次数 | 33,280 | 1,536 |
| D2H 次数 | 33,311 | 0 |
| H2D 时间 | 14.387 s | 14.029 s |
| D2H 时间 | 14.136 s | 0 |

CUDA 时间下降约 21%，说明合并和去 D2H 成功；但 small4 端到端从 `76.139 s` 变为
`78.900 s`，仍慢 3.63%。原因是无法隐藏的 H2D 字节没有消失，这项优化主要解决 copy
粒度和显存上界。

### 8.2 让 44/48 层 MLP 常驻，仅 Offload attention 子 payload

**触发证据。** Blocked W4 profile 中 pinned H2D 为第一热点，53 次遍历总计约
307.37 GB。硬件只有一个 copy engine，增加 stream 无法减少带宽下界；必须减少 payload
字节。共享 KV pool 刚释放约 726 MiB，使常驻更多 packed 权重成为可能。

**目的与作用对象。** 针对 PCIe H2D 总量，把高复用、占 payload 大头的 MLP/norm 放在
GPU，只继续异步搬较小的 attention 子 payload。

**实现。** embedding、final norm 和 44 层 MLP/norm 常驻；48 层 attention projection
仍 offload；另有 4 个 sliding layer 的 MLP 留在 payload 中作为显存安全余量。Offloader
仍使用一个 transfer stream、compute stream 和两个固定 GPU buffer。时序仍是当前层开始
计算前立即预取下一层，使当前 attention 和 resident MLP 尽量覆盖下一层 H2D。

**为什么不常驻全部 48 层。** 全 48 层常驻曾得到 `28.149 s`；RoPE 去重后的探索版本为
`28.813 s`，但两者都在申请 64 MiB activation 时出现 allocator warning，峰值 allocated
接近 9.95 GB。最终选择 44 层，用约 3 秒换取约 207 MB allocated 余量和可重复通过。

**搬运结果。** 最终 profile 区间：

```text
host_to_device_bytes = 87,695,737,256
device_to_host_bytes = 0
prefetch_calls        = 2,544
merged_payload_bytes  = 1,654,636,552
gpu_buffer_slots      = 2
gpu_buffer_bytes      = 238,204,932
```

H2D 从约 307.37 GB 降到 87.696 GB，下降 **71.47%**。调用数仍是 53 次遍历乘 48 层，
说明收益来自每次 payload 变小，而不是隐藏统计口径或跳过层。

### 8.3 共享 RoPE cache，去除重复存储与 H2D

**触发证据。** 代码检查发现 48 层 payload 各有一份 FP32 RoPE cache，合计约 224 MiB；
这些内容只分为 sliding/global 两种，却在每次模型遍历中被重复搬运。

**目的与作用对象。** 针对与层号无关的只读 RoPE 数据，消除重复 host/pinned 存储和
H2D 字节。

**实现。** 40 个 sliding layer 共享一份 GPU RoPE cache，8 个 global layer 共享另一份；
offloader tensor filter 将这两份常驻 buffer 排除在 payload 之外。

**结果与证据。** 代码结构将 48 份同类 cache 收敛为 2 份，并从逐层 payload 中移除。
该项与共享 pool 和 MLP 常驻同轮完成，没有独立 full10 A/B，因此只把去重事实计为证据，
不虚构单项秒数。

## 9. 目标六：固定显存分配顺序，并验收稳定配置

**阶段目的。** 上一节已根据 48 层全常驻的 allocator warning 回退到 44 层，并保留约
207 MB allocated 余量。本阶段进一步固定运行库的分配顺序，并用重复运行验证
“三次 full10 都小于 36 秒、无显存告警、token 一致且 `delta_nll < 0.1`”，
而不是一次偶然的最低值。

### 9.1 提前初始化 cuBLAS workspace，固定显存分配顺序

**触发证据。** 失败作业表明，若最大 prefill 后才首次调用 LM head，cuBLAS handle/workspace
的惰性创建可能在持久显存已接近上限时报 `CUBLAS_STATUS_ALLOC_FAILED`。

**目的与作用对象。** 这项手段不减少模型计算量，而是把小块运行库工作区的申请移到
大块 MLP/KV 分配之前，避免碎片和分配顺序导致的偶发失败。

**实现。** engine 在正式计时前用一个 `16x16` BF16 矩阵乘初始化 cuBLAS handle/workspace，
随后再建立大块常驻权重与 KV pool。

**结果与证据。** 最终三次正式运行和一次 full10 profile 均无 allocator warning。该项
属于最终组合轮，没有独立性能 A/B，因此只将稳定性现象作为证据。

### 9.2 使用三次 full10 与最终 Profile 验收稳定配置

**正式结果。** 作业 `181028/181162/181188` 分别为 `31.907938/31.679216/31.238719 s`；
中位数为 `31.679216 s`，三次都小于 36 秒。

最终正式峰值 allocated 为 `9,742,262,784 B`，reserved 为 `10,039,066,624 B`。三次
正式成绩最大值为 31.908 s，没有 allocator warning。

以正式 run 1 的逐请求 metrics 进一步汇总：六个 prefill batch 的计算时间合计
`18.237 s`；310 个 per-request decode 间隔的加权 Mean TPOT 为 `0.2463 s`；最短和最长
请求的总延迟分别为 `25.695 s` 和 `31.879 s`。这里的 prefill batch 时间不包含请求在
FIFO 中等待前序 prefill batch 的时间，因此不把它错误标成所有请求的
arrival-to-first-token TTFT。

最终 warmed full10 CUDA-only profile 作业 `181123`：

```text
wall_time_s          = 26.38207982800668
peak_allocated_bytes = 9,516,695,552
peak_reserved_bytes  = 10,039,066,624
Self CUDA total      = 31.678 s
```

| 热点 | Self CUDA | 占比 | 调用数 | Blocked W4 轮对照 |
| --- | ---: | ---: | ---: | ---: |
| W4A16 | 13.590 s | 42.90% | 17,384 | 20.557 s |
| pinned H2D | 10.168 s | 32.10% | 2,544 | 23.554 s |
| FlashAttention | 2.925 s | 9.23% | 2,544 | 2.908 s |
| 两类 elementwise | 1.897 s | 5.99% | 84,217 | 1.908 s |
| embedding/LM head | 0.465 s | 1.47% | 53 | 0.464 s |

profile wall time 从 43.347 s 降到 26.382 s，下降 39.14%；其中 H2D Self CUDA 从
23.554 s 降到 10.168 s，直接支持“减少 payload 字节成功”的假设。W4 和 H2D 仍有异步
重叠，因此二者不能直接相加为端到端时间。

## 10. 负优化、失败实验与工程取舍

这些结果应保留在正式报告中，因为它们构成了优化决策证据：

| 尝试 | 观察 | 原因与后续决策 |
| --- | --- | --- |
| 异步 Offloading 第一版 | 显存降 67.6%，但 H2D/D2H 成为主热点 | Offload 是容量交换；随后去 D2H、合并 copy |
| 双 buffer Offload | small 比无 Offload 慢 3.63% | 必须搬的 5.8 GB/遍历仍在；后续做 batching/residency |
| Ring KV | 短请求慢约 2.7% | 请求未越过窗口，收益在容量而非短请求计算 |
| 第一版 Paged KV | 比连续 Ring 多约 5.03 MiB，small 慢 4.98% | 只建立页表与池内复用语义；后续缩小物理 pool 并让 kernel 直读页表 |
| 初版 fused W4 prefill | TTFT 回退 15.8% | decode tile 不能代表大 M；后续单独优化 prefill tile |
| FlashAttention | 相邻版本 full10 观察到 8.45% 下降，不可全部归因 | profile 显示 attention 已仅 0.24%，W4/H2D 才是主瓶颈 |
| Continuous batch 10 | 只获得 3.29x，不是 10x | 自回归、六组 prefill、active batch 衰减和 W4/H2D 成本 |
| 48 层 MLP 全常驻 | 28.1 至 28.8 s，但有 allocator warning | 放弃最快点，采用 44 层稳定配置 |

“优化失败”并不等于工作无效。Ring/Paged 和 Offloading 提供了后续 batch 10、稳定 slot
和权重常驻所需的容量与接口；它们的价值要在完整系统组合中评价。

## 11. 正确性、规则符合性与证据索引

### 11.1 正确性

- 最终 GPU 回归：作业 `181333`，`28 passed, 5 warnings in 9.99s`；
- warning 仅为 Triton 对未来 Python 3.15 的 `AnnAssign` 弃用提示；
- canonical/blocked packing 往返逐字节一致，nibble 顺序有 sentinel 测试；
- Ring/Paged 跨窗口、跨页、回收和容量竞争均有测试；
- 三次正式生成以及 blocked W4 轮的 token 结果逐字段一致；
- 最终完整质量 `delta_nll=0.0863552578 < 0.1`。

### 11.2 算子规则

W4A16 和 Flash/Paged Attention 都由仓库内 Triton kernel 实现。没有调用实验禁止的
FlashAttention、xFormers、bitsandbytes、cuBLASLt 封装或 PyTorch SDPA。W4A16 的持久
参数仍为 packed INT4、FP16 scale 和必要元数据，没有用完整 BF16 权重缓存绕过约束。

### 11.3 最终配置

```yaml
engine:
  dtype: bfloat16
  max_batch_size: 10
  scheduler_batch_size: 10
  max_sequence_length: 2048
  attention_backend: triton_flash
  linear_backend: int4_triton
  scheduler_backend: continuous
  prefill_token_budget: 2048
  weight_offloading: true
  weight_offloading_prefetch: true
  weight_offloading_pin_memory: true
  weight_resident_mlp: true
  weight_resident_mlp_layers: 44
  ring_kv_cache: true
  paged_kv_cache: true
  paged_kv_block_size: 16
  paged_kv_global_pool_blocks: 776
  paged_kv_sliding_pool_blocks: 530

quantization:
  algorithm: gptq
  bits: 4
  group_size: 64
  symmetric: true
  scale_dtype: float16
  packing: uint8_little_nibble
```

### 11.4 主要证据文件

- 实验要求：`docs/Lab5-Gemma4/index.md`
- 硬件：`docs/Lab5-Gemma4/hardware.md`
- 正式质量：`results/w4-blocked-public-quality.json`
- Blocked W4 正式结果：`results/w4-blocked-public-summary.json`
- 最终三次结果：`results/hybrid-resident-public-summary.json`、
  `results/hybrid-resident-public-summary-run2.json`、
  `results/hybrid-resident-public-summary-run3.json`
- 分轮报告：`reports/baseline-2026-08-26.md` 至
  `reports/shared-kv-hybrid-residency-2026-08-27.md`

主要实现位置如下：

| 子系统 | 文件 |
| --- | --- |
| GPTQ | `src/hpc101_infer/quantization/methods/gptq.py` |
| 量化缓存 | `src/hpc101_infer/quantization/checkpoint.py` |
| 异步 Offloading | `src/hpc101_infer/runtime/offloading.py` |
| Ring/Paged/共享 KV | `src/hpc101_infer/runtime/kv_cache.py` |
| W4A16 Triton kernel | `src/hpc101_infer/kernels/w4a16.py` |
| Flash/Paged Triton kernel | `src/hpc101_infer/kernels/flash_attention.py` |
| Continuous scheduler | `src/hpc101_infer/scheduler/continuous.py` |
| blocked packing bridge | `src/hpc101_infer/quantization/packing.py`、`src/hpc101_infer/layers/linear.py` |
| MLP 常驻与 RoPE 共享 | `src/hpc101_infer/models/gemma4.py` |

远程作业证据索引：

| 阶段 | 正式/正确性作业 | Profile/Microbenchmark 作业 |
| --- | --- | --- |
| 初始 baseline | public OOM `168756/168885` | small profile `169188` |
| 异步 Offloading | correctness `172187` | small profile `172187` |
| 双 buffer/合并 payload | full10 `174926` | small profile `175102` |
| Ring KV | 容量/正确性 `175769`，A/B `175837/175864` | small profile `175773` |
| 第一版 Paged KV | 正确性/容量 `176608` | small profile `176617` |
| fused W4A16 | full10 `177213`，质量 `176961` | micro `177115`，profile `177051` |
| Flash/Paged Attention | full10 `177681`，质量 `177656` | profile `177742` |
| Continuous Batching | full10 `178631`，质量 `178579` | small profile `178661` |
| blocked W4A16 | full10 `179945`，质量 `180064` | micro `179936`，full10 profile `179995` |
| shared KV + hybrid residency | full10 `181028/181162/181188`，测试 `181333` | full10 profile `181123` |

## 12. 结论：以瓶颈迁移串联全部优化

本实验的关键不是某一个“神奇 kernel”，而是让容量、计算、调度和搬运四条优化线形成
闭环。GPTQ、Offloading、Ring/Paged KV 先让 10 GiB 环境具备运行和批处理条件；自写
Triton W4A16 与 FlashAttention 消除完整反量化权重和 `O(L^2)` attention 中间张量；
Continuous Batching 把多请求合并为 53 次模型遍历；blocked layout 及配套
W4 kernel 调度让代表性 W4 shape 的 microbenchmark 加速 5.5 至 6.6 倍，并让
full10 总体加速 2.26x。此后 profile 明确显示 H2D 成为第一系统瓶颈，于是进一步缩小
共享 KV 物理池，把释放的容量换成 44 层 MLP 常驻，使 H2D 字节下降 71.47%。最终
三次完整 10 请求均稳定小于 36 秒，同时保持 `delta_nll < 0.1`。

如果继续优化，应优先考虑 W4 小 M 专用 kernel，以及 gate/up 与 GELU(tanh) × up
epilogue 融合；其次才是固定 batch 形状的 CUDA Graph。最终 profile 中
FlashAttention 仅占 9.23%，继续优先微调 attention 已不符合基于 profile 选择热点的原则。
路线图中的 layer-major prefill、gate/up 融合和 CUDA Graph 当前均未实现，
最终 31.679 秒不依赖这些计划项。
任何后续方案都应保留“三次 full10 最大值小于 36 秒、无 allocator warning、
token 一致和 `delta_nll < 0.1`”四项验收条件。
