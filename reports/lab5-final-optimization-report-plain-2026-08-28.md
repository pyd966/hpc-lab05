# Lab5 Gemma4-12B 端到端优化报告：通俗重写版

日期：2026-08-28

最终代码版本：a722d7e

目标是在约 10 GiB 可见显存的 H800 MIG 上处理题目给出的 10 个请求，并允许框架
自行组成 batch。最终要求同时满足：

- 完整 10 请求总时间不超过 36 秒；
- 量化精度 delta_nll 小于 0.1；
- 不调用现成 FlashAttention、xFormers、bitsandbytes、PyTorch SDPA 等禁用算子。

这份报告只保留理解优化所需的术语。更细的 kernel 参数、作业号和完整 profile 表仍可在
详细参考稿与各轮报告中查找。

## 1. 先说明几个名词

- **W4A16**：权重以 4 bit 保存，输入和输出激活使用 16 bit。它不是一个具体 kernel 的
  名字，而是一种数据精度组合。
- **Prefill**：一次处理整段 prompt，矩阵通常较大。
- **Decode**：每轮只生成一个新 token，矩阵的行数通常很小，但会重复很多轮。
- **H2D**：把数据从 CPU 内存搬到 GPU 显存。
- **Peak allocated**：一次运行中，PyTorch 实际分配过的最高显存。
- **KV 物理池容量**：初始化时真正创建的 K/V tensor 与页表大小。它和“当前有多少页正在
  使用”不是同一个指标。

## 2. 总体思路

整个优化过程不是一直修改同一个 kernel，而是跟着瓶颈迁移：

1. 模型和长请求放不下，先做量化、Offloading、Ring KV。
2. 模型能运行后，删除完整反量化权重、attention score 和全 prompt logits 等大临时量。
3. 单请求已经较快，但 10 个请求串行仍太慢，于是做 Continuous Batching。
4. Batch 改变了矩阵形状，W4A16 又成为第一计算热点，于是重排权重并分别优化 prefill 和
   decode。
5. W4A16 变快后，H2D 成为第一热点，于是缩小共享 KV pool，并把空出的显存用于 MLP
   权重常驻。

性能演进如下：

| 阶段 | 完整 10 请求时间 | 这一轮主要解决的问题 |
| --- | ---: | --- |
| 初始 g64 baseline | OOM | 长 prompt 无法运行 |
| 双 buffer Offloading | 425.020 s | 首次跑完整队列，但仍有显存告警 |
| 融合 W4A16 后 | 400.511 s | 不再生成完整 BF16 临时权重 |
| 自写 Flash/Paged Attention | 366.652 s | 不再生成完整 attention score |
| Continuous Batching | 111.396 s | 一次权重加载同时服务多个请求 |
| Blocked W4A16 | 49.234 s | 让 GPU 连续读取权重，并适配真实 batch 形状 |
| 共享 KV + 44 层 MLP 常驻 | 31.679 s 中位数 | 大幅减少 H2D 总字节数 |

## 3. 先把模型放进 10 GiB 显存

### 3.1 GPTQ 量化

**为什么想到。** Gemma4-12B 的 BF16 权重本身已经超过 10 GiB，无法全量加载。RTN
只看权重数值，INT4 下精度不够稳定，因此采用会结合校准激活的 GPTQ。

**做了什么。** 对每个 Linear 收集校准激活，以 128 列为计算块、64 列为量化 group，
加入 0.01 对角阻尼，并把当前层量化后的输出继续传给下一层做校准。

**结果。** 最终 delta_nll 为 0.0863552578，小于 0.1。后续权重重排只改变字节位置，
没有重新计算量化值，因此最终质量保持不变。

量化 checkpoint 的长期缓存只用于避免开发时重复等待约 18 分钟，不属于正式推理优化；
OJ 在干净环境中仍会从头执行 GPTQ。

### 3.2 异步权重 Offloading

**原来的问题。** INT4 已经缩小权重，但所有 DecoderLayer 同时放在 GPU 时，短输入峰值
仍有 8,252,107,776 B。长 prompt 再加入 KV 和临时激活就会 OOM。

**做法。** CPU 保存各层权重，GPU 只保留当前层和下一层。实现使用一条计算 stream、
一条传输 stream 和两个 GPU buffer：

    传输 layer 0
          |
    计算 layer i 的同时，预取 layer i+1
          |
    layer i 计算完成后，buffer 才允许复用

CPU 权重使用 pinned memory，DMA 才能真正异步读取。若下一层还没搬完，计算 stream 会
等待该层的 ready event；代码不会假装搬运已经被完全隐藏。

**显存结果。** 同一短输入峰值从 8,252,107,776 B 降到 2,673,564,672 B，下降 67.60%。

**性能结果。** Offloading 没有直接加速。第一版 profile 中 H2D 和 D2H 都成为大热点，
说明它只是把“显存放不下”换成了“PCIe 搬运很多”。

### 3.3 Ring KV Cache

**原来的问题。** 40 个 sliding attention layer 只看最近 1024 个 token，旧实现却为它们
保留完整 2048-token 历史。

**做法。** sliding layer 使用环形槽位覆盖 1024 token 以前的 KV；8 个 global layer
仍保留完整前缀。逻辑位置保持绝对 token 位置，因此不会破坏 causal 和 sliding mask。

**显存结果。** batch 1、max length 2048 时，KV 从 704,643,456 B 降到
369,099,136 B，准确节省 320 MiB，下降 47.62%。

短请求没有超过 1024 token，Ring A/B 反而慢约 2.7%。这不是计算优化失败，而是说明
Ring 的主要价值是限制容量上界。

### 3.4 Paged KV 到底有没有省显存

第一版 Paged KV **没有直接省显存**。它完成的是管理机制：

- 逻辑 token 页通过 block table 找到物理页；
- 写入新 token 时从 free list 取得页面；
- 请求完成时把页面放回 free list；
- Continuous Batching 可以让后续请求复用这些页。

但第一版底层 K/V tensor 仍按“batch 大小乘每个请求的最大页数”一次性创建。页面释放后
只是回到内部 free list，并没有把 tensor 归还给 PyTorch allocator。

因此 batch 1 的连续 Ring KV 为 369,099,136 B，第一版 Paged+Ring 为
374,371,008 B，反而多约 5.03 MiB。这部分额外空间来自页表和未对齐 sliding window 的
边界页。Paged+Ring 若与完全关闭 Ring 的 704,692,608 B 比显得更小，主要原因是 Ring，
不能把这 46.9% 归因给 Paged。

Paged 真正转化为显存下降发生在后续共享物理池：

| batch 10 KV 配置 | 物理 KV+metadata |
| --- | ---: |
| 第一版：每个 slot 预留最大容量 | 3,743,710,080 B |
| 最终：跨请求共享较小物理池 | 2,982,443,904 B |
| 实际释放 | 761,266,176 B，约 726 MiB |

较小 pool 能安全运行，还依赖完整生命周期管理：

1. 请求进入 GPU 前，按 prompt 长度加最大输出长度预留未来 block credit。
2. 一次 KV 写入先检查全部 48 层，所有层都有页后再统一提交。
3. 请求结束立即归还实际页面和未来 credit；放不下的新 prefill 暂停，先让已有 decode
   完成并释放页面。

所以准确的因果关系是：

    Paged 提供页表与复用机制
        + 生命周期 admission / reserve / release
        + 初始化时真正缩小物理 tensor
        = KV 真实少占约 726 MiB

## 4. 删除不必要的大临时张量

### 4.1 反量化与 GEMM 融合：W4A16 到底优化了什么

**原来的做法。**

    packed INT4 权重
        -> 展开整张 BF16 权重
        -> 把 BF16 权重写入显存
        -> GEMM 再从显存读回来
        -> 得到输出

一个真实投影会临时生成约 355.9 MB 的 BF16 权重。Decode 每次只有很少的输入行，却仍要
展开整张权重；profile 因此显示位运算、乘减和 copy 比真正的矩阵乘还贵。

**融合后的做法。**

    读取一小块 INT4
        -> 在 Triton kernel 内解包和乘 scale
        -> 立刻与对应 activation 相乘并累加
        -> 丢弃这一小块反量化结果
        -> 只写最终输出

也就是说，反量化结果只在 kernel 内短暂存在，不再生成完整 BF16 权重 tensor。持久参数
仍然是 INT4、scale 和必要元数据，没有用 BF16 权重缓存绕过实验要求。

**局部结果。**

| 真实 Gemma4 形状 | 旧实现 | 初版融合 | 临时显存 |
| --- | ---: | ---: | ---: |
| decode gate | 6.633 ms | 2.211 ms | 355.9 MB -> 30,720 B |
| decode down | 6.597 ms | 3.449 ms | 355.9 MB -> 7,680 B |
| prefill gate，M=128 | 6.621 ms | 4.024 ms | 355.9 MB -> 3.93 MB |

同一单请求总时间从 50.331 s 降到 31.866 s，峰值 allocated 从 3,051,874,816 B 降到
2,703,799,296 B。完整 10 请求相邻版本为 400.511 s，但这一差值还包含 Ring/Paged 等
改动，因此不能全部算给 W4A16 融合。

**Profile 是否证明成功。** 原来的反量化逐元素热点消失，计算集中到一个 W4A16 kernel。
这说明融合确实删除了中间步骤；同时 profile 暴露出下一个问题：W4A16 自身与 H2D 各占
约一半时间。

### 4.2 自写 Flash/Paged Attention

普通 attention 会先创建一张“每个 query token 对每个 key token”的 score 矩阵。长度
翻倍时，这张矩阵的元素数约变成四倍。实际上 softmax 和最终加权和可以分块完成，不需要
把整张 score 保存在显存。

自写 Triton kernel 每次只处理一个小 score tile，用 online softmax 保存当前最大值、指数
和与输出累加值；处理完所有 key tile 后才写最终输出。它还直接读取 Paged KV 的 block
table，不先把离散 K/V gather 成一份完整连续副本。

global attention 的 head dimension 为 512，一次保存完整累加值会带来过高寄存器压力。
实现把输出维拆成两个 256 维程序。这样会重复一部分 QK 计算，但 kernel 能稳定编译和运行。
我们没有采集硬件寄存器计数，因此只把它写成设计取舍，不声称寄存器下降了某个百分比。

相邻完整 10 请求从 400.511 s 降到 366.652 s。峰值显存没有下降，不代表 Flash 没有删除
score；当时真正决定峰值的是下一节约 1 GiB 的 LM-head logits。单请求 profile 中 Flash
只占 0.24%，W4A16 与 H2D 已成为更大的热点，所以没有继续优先调整 attention。

### 4.3 Prefill 只计算最后一个位置的 logits

生成第一个新 token 时，每条 prompt 只需要最后一个有效位置的词表 logits。旧代码却为
2000 个 prompt token 全部计算 262144 维 logits，临时 tensor 大小恰好为：

    2000 x 262144 x 2 bytes = 1,048,576,000 B

修改后先选出每条请求最后一个有效 hidden state，再执行 LM head，只产生
batch x 1 x vocab 的 logits。Continuous 版本完整 10 请求峰值为 6,297,783,808 B，原来的
1 GiB 分配告警消失。该修改与 scheduler 同轮完成，没有独立端到端 A/B，因此只把精确
删除这 1 GiB 临时量作为证据。

## 5. 让一次模型计算同时服务多个请求

### 5.1 Continuous Batching

**串行为什么慢。** 10 个请求一共有 320 个输出 token。batch 1 时，几乎每生成一个 token
都要重新遍历 48 层并重新搬权重，大约需要 320 次模型遍历。

**怎样合并。** Scheduler 为请求分配稳定 KV slot。Prefill 受 2048-token budget 限制，
形成 4、2、1、1、1、1 六组；进入 decode 后，一次 48 层遍历同时为所有未完成请求生成
下一个 token。短请求结束后立即移出 batch 并释放 KV 页，因此后续 batch 从 10 行缩到
7 行，再缩到 3 行。

这样完整模型遍历从约 320 次降为：

    6 次 prefill + 47 次 decode = 53 次

完整 10 请求先降到 133.088 s，去除同步点后达到 111.396 s，相对串行 Flash 版本下降
69.62%。

**为什么 batch 10 没有加速 10 倍。** Decode 的不同 token 之间仍然串行；六组 prefill
不能合成一次；短请求完成后 active batch 会变小；较大的 batch 也会让单次矩阵乘计算量
增加。因此 batch size 10 是最大并发请求数，不是固定的 10 倍加速。

### 5.2 删除 nonzero 引起的 CPU-GPU 等待

初版 scheduler 每层都在 GPU 上寻找哪些 token 不是 padding，再把结果交给 CPU。这个
nonzero 操作调用 2,880 次，CPU total 达 30.186 s；CPU 必须等待 GPU 返回动态索引，
破坏了权重预取与计算的重叠。

实际上 scheduler 在 CPU 上本来就知道每条请求本轮要处理的 token 起止位置，因此直接把
这些范围传给 KV reserve/write，不再让 GPU 重新寻找。同口径 small4 profile wall time
从 37.745 s 降到 25.809 s，nonzero 也从最终热点表消失。

## 6. 第二次优化 W4A16：让权重读取更连续

融合反量化只是删除了完整 BF16 临时量，但 INT4 权重在显存里的排列仍不适合 kernel。
Continuous Batching 后，profile 显示 W4A16 占 59.91%，而 FlashAttention 只有 0.15%，
所以此时应该继续优化 W4A16，而不是 attention。

### 6.1 为什么要重排权重

旧权重按输出通道逐行存储：

    输出通道 0：这一行的全部 K
    输出通道 1：这一行的全部 K
    输出通道 2：这一行的全部 K

但 kernel 会同时计算一组输出通道，并在同一个 K 小块上读取它们。旧布局导致 GPU 为了
读取相邻输出通道，要跨过整行 K/2 字节，访存不连续。

新布局把“同一个 K 小块、连续 128 个输出通道”放在一起：

    K 小块 0：[通道 0..127]
    K 小块 1：[通道 0..127]
    ...

这只是无损字节重排，发生在正式计时和 pinned payload 构建之前。INT4 nibble、scale 和
GPTQ 数值都不变，全部 328 个矩阵也不需要 padding。

### 6.2 Decode 和 Prefill 使用不同 tile

Decode 通常只处理 1 到 10 行，重点是避免启动大量空线程；Prefill 可能处理上千行，重点
是让同一块权重服务更多输入行。用同一个 tile 同时服务两者会顾此失彼。

下表中的 M 就是一次矩阵乘同时处理的 token 行数。M=1 是典型单请求 decode，M 较大则
更接近 prefill 或多请求 batch。

另外量化 group size 是 64，因此 kernel 也按 64 个 K 元素处理。这样每块 scale 只需读取
一次，再用于对应 64 个权重，而不是重复加载。

**局部结果。**

| 形状 | 旧融合 kernel | 重排后 kernel | 加速 |
| --- | ---: | ---: | ---: |
| decode gate，M=1 | 2.186 ms | 0.397 ms | 5.51x |
| decode down，M=1 | 3.442 ms | 0.569 ms | 6.05x |
| prefill gate，M=128 | 3.991 ms | 0.605 ms | 6.60x |

大 prefill 的 M=2000 gate 也从 13.772 ms 降到 8.666 ms。完整 10 请求从 Continuous 的
111.396 s 降到 49.234 s，下降 55.80%。

此时 full10 profile 中 H2D 为 23.554 s、45.24%，W4A16 为 20.557 s、39.49%。
W4A16 已明显变快，H2D 变成新的第一热点，所以下一轮不再继续只调 tile。

## 7. 最后减少真正需要搬运的字节

### 7.1 双 buffer 和合并 payload

第一版 Offloading 把每层的 qweight、scale、norm 等 tensor 分开搬运，还把只读权重从
GPU 搬回 CPU。改进后，每层先在 CPU 拼成一块连续 pinned payload，每层只发射一次 H2D；
GPU 只使用两个固定 buffer，释放层时只切回 CPU view，不再 D2H。

| 指标 | 第一版 | 双 buffer |
| --- | ---: | ---: |
| H2D 次数 | 33,280 | 1,536 |
| D2H 次数 | 33,311 | 0 |
| Self CUDA | 69.980 s | 55.307 s |

但 H2D 时间只从 14.387 s 变为 14.029 s，因为需要搬运的总字节数没有下降。因此这一步
主要删除小 copy 和无意义 D2H，还没有解决 PCIe 带宽下界。

### 7.2 用省下的 KV 显存让 44 层 MLP 常驻

Blocked W4 版本只需 53 次模型遍历，但每次仍搬约 5.799 GB，累计约 307.37 GB。硬件只有
一个 copy engine，继续增加 stream 不能减少总字节数。

共享 KV pool 刚释放约 726 MiB，因此把占 payload 大头、每轮都会使用的 MLP/norm 留在
GPU，只继续 Offload attention projection。最终常驻 embedding、final norm 和 44 层
MLP/norm；另有 4 层 MLP 继续 Offload 作为显存安全余量。

最终 H2D 为 87,695,737,256 B，相比约 307.37 GB 下降 71.47%。H2D Self CUDA 也从
23.554 s 降到 10.168 s。这说明收益来自每次 payload 变小，而不是减少模型层数或改变
计时口径。

### 7.3 合并重复 RoPE cache

原来 48 层 payload 各带一份 FP32 RoPE cache，合计约 224 MiB，但实际只有 sliding 和
global 两种内容。最终让 40 个 sliding layer 共享一份、8 个 global layer 共享一份，并
把它们留在 GPU，不再逐层重复 H2D。

该改动与共享 KV 和 MLP 常驻同轮完成，没有独立端到端 A/B，因此报告只声明 48 份变为
2 份，不虚构单项加速秒数。

## 8. 为什么最终选择 44 层，而不是最快的 48 层

全部 48 层 MLP 常驻曾达到 28.149 s 和 28.813 s，但峰值 allocated 接近 9.95 GB。
申请一个 64 MiB activation 时出现 allocator warning，清理缓存重试后才完成。这种结果
虽然更快，却不适合作为稳定提交。

最终退回 44 层常驻：

- 正式峰值 allocated：9,742,262,784 B；
- 正式峰值 reserved：10,039,066,624 B；
- 相对 48 层常驻减少约 207 MB allocated；
- 三次正式运行都没有 allocator warning。

另外在正式计时前先执行一次很小的 BF16 矩阵乘，让 cuBLAS 提前创建 handle/workspace。
这样运行库的小块内存先分配，避免大块 MLP/KV 已占满显存后才因碎片创建失败。它不减少
计算量，只改善分配顺序和稳定性。

## 9. 最终结果和 Profile

三次完整 10 请求：

| 作业 | elapsed_s | 生成吞吐 |
| ---: | ---: | ---: |
| 181028 | 31.907938 s | 10.028853 token/s |
| 181162 | 31.679216 s | 10.101260 token/s |
| 181188 | 31.238719 s | 10.243698 token/s |

三次都小于 36 秒，中位数为 31.679216 s。最终完整质量为：

- mean_nll：2.3948107879；
- reference_mean_nll：2.3084555301；
- delta_nll：0.0863552578。

最终 warmed full10 profile：

| 热点 | Self CUDA | 占比 |
| --- | ---: | ---: |
| W4A16 | 13.590 s | 42.90% |
| pinned H2D | 10.168 s | 32.10% |
| FlashAttention | 2.925 s | 9.23% |
| 两类 elementwise | 1.897 s | 5.99% |
| embedding/LM head | 0.465 s | 1.47% |

从 profile 可以看到：

1. 反量化 elementwise 已经不再单独成为热点，说明 W4A16 融合生效。
2. H2D 从 blocked 版本的 23.554 s 降到 10.168 s，说明混合常驻生效。
3. FlashAttention 只有 9.23%，继续优先优化 attention 已经不符合热点顺序。
4. 最终剩余主要瓶颈是 W4A16 小 M kernel 和不可完全隐藏的 H2D。

## 10. 哪些结果不能过度归因

- 第一版 Paged KV 没有降低物理池，反而比连续 Ring 多约 5.03 MiB；真实的 726 MiB
  降幅来自后续共享小池。
- FlashAttention 删除了完整 score，但进程峰值当时没有下降，因为峰值由 1 GiB logits
  决定。
- 共享 KV、44 层常驻和 RoPE 去重在同一轮完成，不能把 49.234 s 到 31.679 s 的全部差值
  单独算给其中一项。
- 初版 fused W4 的 full10 相邻版本还包含 Ring/Paged 改动，因此最可靠证据是同请求 A/B、
  microbenchmark、临时显存下降和 profile 热点消失。
- 28 秒的 48 层常驻不是最终成绩，因为它出现 allocator warning；正式结论使用三次稳定
  的 44 层配置。

## 11. 结论

最终达到 31.679 s，不是因为某一个 kernel 突然快了十倍，而是连续消除了五类浪费：

1. GPTQ、Offloading 和 Ring KV 解决“放不下”。
2. Paged KV 加生命周期管理，让请求可以安全共享较小物理池。
3. 融合 W4A16、FlashAttention 和 last-token logits 删除大临时 tensor。
4. Continuous Batching 把约 320 次模型遍历压到 53 次。
5. Blocked W4 提高权重读取效率，再把省下的显存用于 44 层 MLP 常驻，最终减少 71.47%
   的 H2D 字节。

这条主线的核心是：每次先用 profile 找到新的第一瓶颈，再决定下一轮优化；节省出的显存
也不是闲置，而是重新投入到能减少 PCIe 搬运的权重常驻中。
