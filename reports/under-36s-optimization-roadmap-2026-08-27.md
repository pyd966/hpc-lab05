# Gemma4 完整 10 请求进入 36 秒的优化路线图

## 结论

当前正式成绩是 **111.396 s**，要进入评分公式的满分区间，需要做到严格的
`elapsed_s < 36`，不是只把某个 kernel 或单请求延迟降到 36 秒。考虑集群波动，
本文把工程验收线设为 **33--34 s**。

仅继续调 FlashAttention、Paged Attention block size 或 batch size 无法达到目标。
当前代表性 profile 中：

| 项目 | 当前数据 | 结论 |
| --- | ---: | --- |
| 自写 W4A16 kernel | 22.833 s，59.91% Self CUDA | 第一计算瓶颈 |
| Pinned H2D | 14.280 s，37.47% Self CUDA | 第一系统瓶颈 |
| 自写 FlashAttention | 57.222 ms，0.15% Self CUDA | 已不是主要矛盾 |
| embedding / LM head | 278.233 ms，0.73% Self CUDA | 暂不优先 |
| 完整 10 请求峰值 allocated | 6.298 GB | 有空间换取部分权重常驻 |

要稳定进入 36 秒，必须组合实施以下五项，而不是押注单项：

1. **无损重排 packed INT4，并重写 SM90 专用 W4A16 kernel**；
2. **把 Paged KV 从“逻辑按需、物理全量预留”改成真正的共享物理 block pool**；
3. **利用腾出的显存常驻全部 MLP packed 权重，只 offload attention 子 payload**；
4. **把六次 prefill 改为 layer-major，使一层权重只搬一次**；
5. **融合 QKV、gate/up/activation、Paged KV scatter，并对 batch 10/7/3 的
   decode 建立 CUDA Graph，消除逐 token Python 发射和同步**。

这条路径的目标预算是：

| 阶段 | 当前方向性拆分 | 工程目标 |
| --- | ---: | ---: |
| Prefill | 约 63.0 s | 18 s，最迟不得超过 20 s |
| Decode | 约 32.8 s | 10--11 s |
| 调度、采样、同步及剩余开销 | 约 15.6 s | 3--4 s |
| 总计 | 111.396 s | **31--34 s** |

这些阶段数字用于制定预算，不能当作互相独立的精确 CUDA 时间。当前
`measure_operation()` 只在计时前同步，GPU 尾部可能在采样的 `.tolist()` 或下一次
同步中结算。因此第一轮 profile 必须补充 CUDA Event/NVTX 的阶段测量，但不得修改官方
`elapsed_s` 的计时区间。

## 评测边界

- 硬件是 H800 PCIe MIG `1g.10gb`：14 SM、9984 MiB 显存、6 MiB L2、一个
  copy engine、PCIe Gen4 x16。
- 正式入口必须是 `scripts/run_generation_queue.py`，数据必须是
  `datasets/performance_public.jsonl` 的全部 10 个请求。
- 10 个请求在计时开始前均已入队；可以 batch，但不能改计时区间、少生成 token、
  跳过 LM head 或改变模型结构。
- 评分文档实际写的是严格 `x < 36`。恰好 36 秒没有安全余量。
- `delta_nll` 必须保持小于 0.1；当前是 **0.0863552578**，只剩约 0.0136 的余量。
- 优化算子必须用 Triton 或 TileLang 自写。不得直接调用 FlashAttention、xFormers、
  bitsandbytes、cuBLASLt 封装、PyTorch SDPA、Marlin 或 AWQ kernel。
- 可以借鉴开源设计，但持久权重必须仍是 packed INT4；不得缓存完整 BF16/FP16
  反量化权重。
- OJ 会重新执行量化。长期量化缓存只用于本地迭代，新 packed layout 必须由我们的
  量化/保存代码生成，而不能只存在于手工制作的 checkpoint 中。

## 为什么当前实现慢

### 1. W4A16 的根因首先是布局，不只是 tile

当前 `qweight` 是 `[N, K/2]` 行主序，而
`_w4a16_gemm_kernel` 的一个 tile 按 `[K, N]` 读取。相邻 N lane 之间跨越很大的
`stride_qn`，读取不能理想合并；scale 也按跨行方式加载，同一个 group 的 scale
还会被每个 K lane 重复读取。

此外，当前只有三档静态配置：

- `M <= 4`；
- `4 < M <= 64`；
- `M > 64`。

正式 decode 的 `M=7/10` 因而落入 `BLOCK_M=32` 档，而不是 skinny decode 专用
路径。Prefill 的配置还使用 `BLOCK_K=32`，与 group size 64 不对齐。

这解释了为什么简单融合虽然消除了完整反量化临时张量，却仍然需要约 2--3 ms 完成
一个 decode projection。完整模型每个 forward 有 328 次 W4 调用，47 个 decode
step 会把这个延迟放大。

### 2. 每个 forward 都重新搬 5.799 GB

48 层合并后的 packed payload 总计 **5,799,469,248 B**。完整 workload 有六次
prefill forward，加上 47 次 decode forward，共 53 次；当前策略理论上要发起约
**307 GB H2D**。

代表性 profile 有 32 次完整层遍历，理论 payload 约 185.6 GB，H2D stream 时间为
14.280 s，对应约 **13.0 GB/s** 有效传输率。按这个实测带宽，53 次遍历的传输流时间
约 23.6 s，其中约 21.0 s 来自 decode。部分传输已经和 W4 重叠，因此不能把 23.6 s
直接从 E2E 相减；它说明 W4 变快后，若不做 residency，PCIe 会立刻成为新下界。

现有 offloader 的拓扑本身是合理的：一条 transfer stream、一条 compute stream、
双 GPU buffer、pinned host memory 和 CUDA Event。硬件只有一个 copy engine，再增加
H2D stream 不会增加 PCIe 带宽。真正的问题是必须搬的字节太多。

当前实现还在每次 prefetch/release 时重新绑定 tensor、创建 `nn.Parameter`，并在
`release()` 中新建 CUDA Event。这些动态对象和地址变化也阻碍 CUDA Graph。

### 3. Paged KV 还没有真正缩小物理池

当前 Paged KV 在逻辑上按需分配 page，但物理 tensor 仍按
`max_batch_size * max_blocks_per_sequence` 全量预留，batch 10 约占 **3.744 GB**。

按公开 workload 的 prompt、最大输出长度和 block size 16 计算，实际物理容量约为：

- sliding 层：7920 token slots；
- global 层：10880 token slots；
- 合计约 **2.773 GB**。

因此公开 workload 可无损释放约 **970 MB**。若按题目各长度区间上界做保守预算，
共享池约为 2.982 GB，仍可释放约 **761 MB**。实现应根据计时前已经入队的请求长度和
`max_new_tokens` 计算总 block budget，并保留额外 page watermark，而不是把公开数据
的固定容量写死。这部分显存不只是“降低峰值”，而是要直接换成 packed 权重常驻空间。

### 4. 当前 prefill 为同一套权重搬了六遍

`prefill_token_budget=2048` 形成 `4、2、1、1、1、1` 六个 batch。每个 batch
独立遍历 48 层，因此每一层被搬六次。

真实 prompt 共 10500 token，当前六组 padding 后约 11000 token。把六个 hidden-state
张量同时保存只需要约：

`11000 * 3840 * 2 B = 84.48 MB`

所以无需构造 `10 x 2000` 的大 padded batch，也能把执行顺序改成：

`for layer -> for prefill_microbatch`

一层在 GPU 上时依次处理六个 microbatch，最后才释放该层。模型数学和每条序列的
attention 语义均不变。

### 5. Decode 形状已经稳定，但仍逐 kernel 由 Python 发射

完整 workload 的 decode 只有三种稳定 batch：

| batch | step 数 |
| ---: | ---: |
| 10 | 15 |
| 7 | 16 |
| 3 | 16 |

当前仍逐层、逐 projection 从 Python 发射 kernel，并且每步把采样结果
`.tolist()` 回 CPU 更新 scheduler。这是 CUDA Graph 和 GPU 端采样最适合处理的场景。

## 实施路线

## Round 0：先修正测量，不改变正式路径

这一项可与 Round 1 同一提交完成，但 profile 报告必须先给出以下数据：

1. 保持官方 `perf_counter` 计时原样；
2. 用 CUDA Event 分别测六组 prefill、decode batch 10/7/3；
3. 记录每层 H2D bytes、H2D duration、compute duration 以及
   `wait_event` 真正暴露到 wall time 的等待；
4. 分别记录 prefill 结束、decode 开始和全程峰值显存；
5. 记录 CPU enqueue、采样、`.tolist()` 和 scheduler 的 wall time；
6. full10 profile 禁用不必要的 stack/shape 记录，避免之前聚合 trace 时超过
   24 GiB 主机内存。

以后每一轮都必须报告完整 10 请求的 `elapsed_s`；微基准和小数据 profile 只能解释
原因，不能替代正式成绩。

## Round 1：重构 packed INT4 layout 和 W4A16

### 1.1 离线、无损、kernel-native 的 layout

在量化保存阶段把当前格式转换成 SM90 kernel 使用的 blocked layout：

- qweight 以 K-major、N-contiguous 为基本原则，按实际 MMA tile 对 N/K 做
  interleave；
- 每个 byte 仍只保存两个 INT4 code，不增加精度或物化 BF16 权重；
- scale 改为 `[K / 64, N]` 的 N-contiguous blocked layout；
- `BLOCK_K=64` 时每个 K tile 只加载一次 BN 个 scale，然后沿 K 广播，而不是加载
  `BK * BN` 份重复 scale；
- checkpoint 只持久化新 packed layout，增加 `layout_version`，长期缓存按
  模型、量化配置和 layout version 校验。

本地长期缓存需要在 layout 改动后重建一次；之后继续复用。OJ 的量化流程也必须自动
生成同一格式。

### 1.2 Prefill/decode 分离

不能再用只看 M 的三档配置。为实际矩阵建立静态 LUT：

- decode：`M in {1, 3, 7, 10, 16}`；
- prefill：重点覆盖正式 workload 的 `M=1500/2000`，同时保留质量测试形状；
- 分别覆盖 gate/up `3840 -> 15360`、down `15360 -> 3840` 和 Q/K/V/O
  projection。

Decode 候选设计：

- grid 先扫 14/28 个 persistent CTA，匹配 14 SM MIG；
- striped N 分工，避免输出列数不能均匀铺满 14 SM；
- down projection 单独评估 split-K，并在输出/L2 中做低开销归约；
- 最大宽度向量加载、静态展开地址、双缓冲 weight/activation tile；
- 以寄存器数、shared memory 和实际 occupancy 决定是否保留 WGMMA/TMA，不以
  “生成了 SM90 代码”代替性能验证。

Prefill 候选设计：

- `BLOCK_K=64` 对齐 group64；
- grouped program ordering 改善 6 MiB L2 上的局部复用；
- 对 exact `(M,N,K)` 扫 BM/BN/warps/stages；
- 尝试 TMA data-movement warp 与 compute warp 分工；
- 配置离线跑完后固化，正式评测不得在线 autotune。

### Round 1 硬门槛

- 代表性 profile 的 W4 Self CUDA：`22.833 s -> <= 7.5 s`，至少 3x；
- decode gate/down、prefill gate/down 四类微基准都必须正确，不能只优化一个形状；
- full10 `elapsed_s <= 80--85 s`；
- layout 转换后逐层反量化抽检、logits A/B 和完整质量均通过。

若 W4 profile 仍大于 9 s，应继续做 layout/访存和占用率分析，不应提前投入
Attention 微调。没有至少 3x 的 W4 收益，36 秒基本不可达。

## Round 2：真实 Paged block pool 和 MLP 权重常驻

### 2.1 共享物理 KV pool

- 物理 K/V tensor 按总 block budget 分配，不再为每个 slot 预留完整 2048；
- block table 仍保持固定地址，以便后续 CUDA Graph；
- scheduler admission 同时检查空闲 request slot 和空闲 KV block；
- 保留 block size 16。它是 vLLM 的常见默认值，且当前 attention 不是瓶颈，没有证据
  支持为少量 kernel 收益增加尾部碎片；
- page 分配失败必须回退为分批接纳，不能 OOM 或覆盖活跃 page。

### 2.2 Phase-aware hybrid residency

量化 manifest 中的 projection payload 可进一步拆分为：

| Payload | 字节数 | decoder projection 占比 |
| --- | ---: | ---: |
| 48 层 MLP packed INT4 + scales | 4,512,153,600 | 77.9% |
| attention packed INT4 + scales | 1,278,443,520 | 22.1% |
| 合计 | 5,790,597,120 | 100% |

完整 offloader payload 的 5,799,469,248 B 还包含 norm、Rotary buffer 等小 tensor。
这些小 tensor 应与 MLP 一起常驻。

最合理的初版不是随机挑 36 个整层，而是：

- 48 层 gate/up/down 和小 norm 参数全部常驻；
- 每层只为 attention projection 建 pinned host payload；
- 双 staging buffer 从最大整层约 128.8 MB/slot 缩到最大 attention 约
  34.5 MB/slot；
- 当前层 attention 结束并释放 buffer 后，在当前层 MLP 计算期间预取下一层
  attention；
- resident tensor 和两个 staging buffer 的地址必须固定；
- 所有 ready/reuse Event 在初始化时预创建；
- 禁止每步新建 `nn.Parameter`、tensor view 或 CUDA Event。

按题目分布上界而非最好公开样例估算：

| 项目 | 字节 |
| --- | ---: |
| 当前 full10 峰值 | 6,297,783,808 |
| 减去旧 KV pool | -3,743,416,320 |
| 加入共享 KV pool 上界 | +2,982,150,144 |
| 减去旧双整层 buffer | -257,531,912 |
| 加入双 attention buffer | 约 +69,000,000 |
| 加入全部 MLP resident | +4,512,153,600 |
| 预计峰值 | **约 9.86 GB / 9.18 GiB** |

9984 MiB MIG 上还剩约 580 MiB。公开 workload 的实际 KV 需求更小，可留下约
0.75 GiB；最终以 allocator 和驱动侧实测为准。若低于 512 MiB 安全余量，则按
“暴露 H2D 等待 / byte”从少数 MLP projection 开始回退到 CPU，而不是 OOM。

一个 copy stream 加一个 compute stream 已足够。只有 profile 证明两 buffer 的预取距离
不足，才考虑第三 buffer；不要增加 H2D stream。

MLP 常驻后，每次遍历只搬约 1.278 GB attention payload。完成 layer-major prefill 后，
一次 prefill 加 47 次 decode 共约 **61.4 GB H2D**，相对当前 307 GB 下降约 80%。

### Round 2 硬门槛

- KV pool 按实际队列或分布上界分配，公开 workload allocated `<= 2.9 GB`；
- 48 层 MLP packed weight 全部常驻；若回退，必须在报告中给出精确 projection；
- 峰值 allocated `<= 9.25 GiB`，驱动侧剩余显存至少 512 MiB；
- 本轮尚未 layer-major 时 full10 H2D bytes `<= 70 GB`；
- exposed H2D wait 相对 Round 1 至少下降 65%；
- full10 `elapsed_s <= 45--50 s`。

如果全部 MLP 不能常驻，先查 KV pool、旧整层 buffer、临时张量和 allocator 碎片，
不要用更激进的 prefetch 逃避显存问题。整层粒度的约 36/48 层常驻只作为回退方案。

## Round 3：Layer-major prefill

执行结构改为：

1. 一次准备六个 prefill microbatch 的 input、positions、lengths、cache slots；
2. 分别执行 embedding，保留六个 hidden-state tensor；
3. 外层循环 48 层；
4. 每层加载/等待一次，在内层依次处理六个 microbatch；
5. 该层六组计算全部结束后再释放 offloaded layer；
6. 最后分别执行 final norm 和每行最后一个 token 的 LM head。

Paged/Ring KV 必须继续按各自 slot 写入，不能把六个 microbatch 的序列混成一条。
当前约 84.5 MB 的额外 hidden-state 常驻开销远低于重复 H2D 的代价。

MLP 常驻后，layer-major prefill 只需要把全模型的 attention 子 payload 搬一次，
prefill H2D 约 1.278 GB；完整 48 次遍历的 H2D 约 61.4 GB。

若 layer-major 后 prefill 仍超过 20 秒，再实现 packed/ragged prefill，并扫描
`4096/6144/8192/10500` token budget；不要直接构造 `10 x 2000` padded batch。
vLLM 的吞吐调优建议更大的 chunked-prefill token budget，但这是方向性依据，最终值
必须由本机 10 GiB full10 实测决定。

### Round 3 硬门槛

- prefill 每个 offloaded layer 只 H2D 一次；
- full10 H2D bytes `<= 65 GB`；
- prefill 真实 CUDA/wall 阶段时间 `<= 18--20 s`；
- 峰值 allocated 不超过 Round 2 预算；
- full10 `elapsed_s <= 38--42 s`。

## Round 4：Projection/KV 融合

按收益与风险依次实现：

1. **gate + up + exact GELU + multiply**

   两组 packed weight 在同一个 kernel 中共享 activation load，分别累加 gate/up，
   epilogue 必须精确实现当前
   `F.gelu(..., approximate="tanh") * up`，直接只写一个 MLP 中间结果。
   不能擅自改为 SiLU/SwiGLU。

2. **Q + K + V projection**

   离线把三组 packed layout 组织成一个 projection payload。一个 kernel 共享输入，
   产生 Q/K/V；随后把 Q/K norm、RoPE 和 layout transform 尽量并入 epilogue。

两项 projection 融合的结构目标是把每次模型遍历的 W4 launch 从 328 次压到约
192 次；最终是否保留仍由 full10 wall time 决定。

3. **Paged KV scatter/commit**

   用一个 Triton kernel 根据 block table 直接写 K/V，替代逐 batch row 创建索引和两次
   `index_copy_`。长度与 page metadata 每步只更新一次，不能在 48 层重复 commit
   相同 sequence length。

4. **低风险小融合**

   最后再评估 residual + post norm、final softcap 和 sampler。每个融合都以 full10
   profile 证明收益；不要因为“kernel 数少了”就假定 wall time必然下降。

### Round 4 硬门槛

- gate/up 融合和 QKV 融合分别有独立 A/B，不允许一次改完后无法归因；
- W4/MLP/QKV 总 CUDA 时间再下降 15% 以上；
- KV scatter/metadata 的 CPU 与 CUDA 调用数显著下降；
- `delta_nll < 0.1`，10 请求 token id 与上一正确版本一致；
- full10 `elapsed_s <= 35--38 s`。

## Round 5：三个 decode CUDA Graph 和 GPU 端采样

在 Round 2 固定权重/staging 地址、Round 4 固定 KV metadata 后，分别 capture
batch 10、7、3 的完整单步 decode graph。

所有真实 kernel 配置应在 engine 初始化阶段预编译，graph 也用固定 dummy buffer
完成 capture；不得处理、缓存或预生成正式请求的输出，也不得移动官方 timer。

每张 graph 包含：

- 每层需要的 attention 子 payload H2D memcpy；
- transfer/compute event dependency；
- 48 层自写 W4、attention、融合 epilogue；
- LM head、采样、token 写回和 device sequence-length 更新。

静态 input/output、KV pool、block table、weight staging buffer 的地址都不得改变。
EOS/stop token 状态保存在 device mask 中；已结束行保持 masked，不能为了固定 graph
而继续把无效 token 写进输出。三个长度 bucket 结束时再批量把 token/状态搬回 CPU，
消除每 token 的 `.tolist()` 同步。

当前逐 forward 的 `torch.cuda.synchronize()` 和
`torch.cuda.reset_peak_memory_stats()` 也不能保留在 replay 热路径。改用 CUDA
Event 延迟读取阶段指标，并在完整 generation 结束后统一同步和读取峰值。

如果整步 capture 被当前 module 参数重绑定阻塞，应先让 QuantizedLinear 直接接收固定
staging slab 的 base pointer + offset；不能通过跳过 offload 或捕获错误地址来“跑通”。

CUDA Graph 可以包含 kernel、memcpy、event 和 memset，但它只消除 host launch gap，
不会减少 W4 计算量或 PCIe 字节数，所以必须放在 W4 和 residency 之后。

### Round 5 验收

- 47 次 decode 只发生 47 次 graph replay，而不是上万次 Python kernel dispatch；
- 每 token CPU round-trip 消失；
- “其他/host/launch”降到 3--4 s；
- 连续三次 full10：最大值 `< 36 s`，中位数 `<= 33--34 s`；
- 最终完整质量 `delta_nll < 0.1`。

## 预计收敛过程

以下是工程预估，不是把重叠的 profiler 百分比直接相加：

| 完成项 | 预计 full10 | 是否足够 |
| --- | ---: | --- |
| 当前版本 | 111.4 s | 否 |
| 新 packed layout + SM90 W4 | 70--85 s | 否 |
| + 真实 KV pool + 全部 MLP 权重常驻 | 42--50 s | 否 |
| + layer-major prefill + projection/KV 融合 | 34--41 s | 接近 |
| + decode CUDA Graph/GPU sampler | **29--34 s** | 目标区间 |

任何一轮没有达到门槛，都应依据 full10 profile 修正该轮，而不是继续叠加下一项。尤其是：

- W4 未达到 3x：继续修 layout、scale broadcast、SM 分工和 occupancy；
- 全部 MLP 无法常驻：继续修 KV 物理池、旧 buffer 和临时内存；
- full10 已低于 40 s 但 host gap 仍大：优先 CUDA Graph；
- CUDA Graph 后仍受 H2D 限制：重新按 exposed wait/byte 选择 resident set。

## 只作为兜底的方案

如果完成主路线后仍在 34--38 秒，可按以下顺序评估：

1. **FP8 KV cache**

   对 K/V 做 per-head 或 per-block scale，在自写 attention 中即时反量化。2.773 GB
   KV 理论上可再释放约 1.39 GB，让更多 packed layer 常驻。H800 有 FP8 能力，但这是
   数值改动，当前质量余量只有 0.0136，必须完整重跑质量。

2. **INT8 tied embedding/LM head**

   BF16 tied embedding 约
   `262144 * 3840 * 2 B = 2.013 GB`。INT8 可释放约 1 GB 给 resident weights，
   但 LM head 当前只占 0.73%，直接算子收益很小，且量化 vocab head 的质量风险高于
   单纯 layout 优化。只有 residency 被显存卡住时才考虑。

3. **更激进的 ragged/chunked prefill**

   先扫 token budget，再决定是否值得为约 4.8% 的当前 padding 差额引入复杂
   ragged kernel。不要照搬大 GPU 上的 vLLM 默认调参。

## 不应继续优先投入

- **继续调 FlashAttention tile**：当前只占 0.15%，即使完全消失也救不了 75 秒差距。
- **增加 batch size**：正式 workload 只有 10 个请求，当前已经 batch 10。
- **增加 H2D stream**：单 copy engine 不会并行执行两路 H2D。
- **盲目加 buffer**：只有现有双 buffer 的预取距离不足时才有收益，还会减少 resident
  权重空间。
- **单纯增大 `prefill_token_budget` 到 10500**：会形成过大的 padded activation，
  在 10 GiB 上有 OOM 风险；先做 layer-major。
- **优先优化 LM head**：当前 0.73%，除非为了释放 tied embedding 显存。
- **直接接入 Marlin/AWQ/vLLM kernel**：违反自写 Triton/TileLang 的实验规则。
- **缓存完整反量化权重**：违反 INT4 持久存储约束，也无法在 10 GiB 中稳定运行。

## 每轮交付与回归

每个 Round 单独完成、单独 profile、单独写中文报告并单独 git commit。每轮至少包含：

1. 正式完整 10 请求的命令、job id、`elapsed_s`、tokens/s；
2. full10 的阶段时间、H2D bytes/calls/exposed wait、主要 kernel；
3. prefill/decode 分阶段峰值 allocated/reserved；
4. 与上一提交的 A/B 和是否达到本轮硬门槛；
5. layout/数学发生变化时的 kernel 单测、logits A/B、10 请求 token A/B；
6. 每个数学改动后的完整 `delta_nll`；
7. 失败实验和回退原因，不能只报告最好结果。

最终验收不是单次偶然的 35.9 秒，而是：

- 正式配置、正式 10 请求；
- 连续三次均 `elapsed_s < 36`；
- 中位数不高于 33--34 秒；
- `delta_nll < 0.1`；
- 峰值显存至少保留 512 MiB，目标保留约 0.7 GiB 安全余量；
- OJ 从代码重新量化后能够生成同一 packed layout。

## 一手资料

- 实验规则与评分口径：[Lab5 文档](../docs/Lab5-Gemma4/index.md)
- 本机硬件实测：[硬件报告](../docs/Lab5-Gemma4/hardware.md)
- 当前完整 10 请求与 profile：[Continuous Batching 报告](continuous-batching-2026-08-27.md)
- Marlin 论文：[MARLIN: Mixed-Precision Auto-Regressive Parallel Inference](https://arxiv.org/abs/2408.11743)
- Marlin 官方实现与设计说明：[IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)
- AWQ 论文：[Activation-aware Weight Quantization](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)
- TinyChat 官方说明：[llm-awq/tinychat](https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/README.md)
- Triton GEMM/autotune：[Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- Triton persistent GEMM：[Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
- Triton grouped GEMM：[Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- NVIDIA Hopper 调优指南：[Hopper Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.2/hopper-tuning-guide/index.html)
- NVIDIA async copy/copy engine 说明：[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- NVIDIA CUDA Graph：[CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- vLLM chunked prefill 调优：[Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- vLLM block size 16 默认值：[CacheConfig](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)
- PagedAttention 论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- ZeRO-Inference offload/prefetch 经验：[DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)

上述项目只作为算法和工程设计依据；本实验实现仍必须使用仓库内自写
Triton/TileLang kernel。
