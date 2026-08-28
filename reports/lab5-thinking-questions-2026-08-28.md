# Lab 5 思考题

日期：2026-08-28

本文中的 `GB` 表示 $10^9$ byte，`GiB` 表示 $2^{30}$ byte。显存计算均以当前仓库的
Gemma4-12B 实现和 `config.yaml` 为准，而不是用“12B 参数”作粗略估计。

## 1. W4A16 权重和 KV Cache 显存推导

### 1.1 计算口径

当前配置的 W4A16 含义是：Linear 权重为 INT4，Linear 输入和输出激活为 BF16。它并不
表示模型中的所有 tensor 都是 4 bit：embedding、RMSNorm 和 layer scalar 仍为 BF16；
每 64 个 INT4 权重还要保存一个 FP16 scale。当前使用对称量化，因此不保存 zero point。

Gemma4-12B 共有 48 个 DecoderLayer，其中：

- 40 层为 sliding attention；
- 8 层为 global attention；
- global attention 中 $W_K$ 与 $W_V$ 共享，因此没有独立的 `v_proj`；
- LM head 直接复用 `embed_tokens.weight`，没有第二份输出矩阵。

记

$$
\begin{gathered}
D=3840,\quad F=15360,\quad V=262144,\\
H_q=16,\quad H_{s,kv}=8,\quad d_s=256,\\
H_{g,kv}=1,\quad d_g=512.
\end{gathered}
$$

### 1.2 被量化 Linear 的大小

所有 MLP 的三个矩阵共有

$$
P_{\mathrm{MLP}}
=48(DF+DF+FD)
=48\times3DF
=8,493,465,600
$$

个权重。

40 个 sliding attention 层都有 Q、K、V、O 四个投影：

$$
\begin{aligned}
P_{\mathrm{slide}}
&=40D(H_qd_s+H_{s,kv}d_s+H_{s,kv}d_s+H_qd_s)\\
&=1,887,436,800.
\end{aligned}
$$

8 个 global attention 层的 K/V 共享，只需 Q、K、O 三个投影：

$$
\begin{aligned}
P_{\mathrm{global}}
&=8D(H_qd_g+H_{g,kv}d_g+H_qd_g)\\
&=519,045,120.
\end{aligned}
$$

所以被量化权重总数为

$$
P_Q=P_{\mathrm{MLP}}+P_{\mathrm{slide}}+P_{\mathrm{global}}
=10,899,947,520.
$$

这也能从模块数量侧面检查：40 个 sliding layer 每层量化 7 个 Linear，8 个 global layer
每层量化 6 个 Linear，共 $40\times7+8\times6=328$ 个 Linear，与量化 manifest 一致。

对于 group size 为 64 的对称 INT4：

$$
M_{\mathrm{code}}=\frac{P_Q}{2},\qquad
M_{\mathrm{scale}}=\frac{P_Q}{64}\times2=\frac{P_Q}{32}.
$$

即每个权重平均占

$$
\frac12+\frac1{32}=\frac{17}{32}\ \mathrm{byte}=4.25\ \mathrm{bit}.
$$

具体结果为：

| 内容 | 字节数 | GiB |
| --- | ---: | ---: |
| Packed INT4 code | 5,449,973,760 | 5.076 |
| FP16 scale | 340,623,360 | 0.317 |
| 全部量化 Linear | 5,790,597,120 | 5.393 |

### 1.3 未量化权重和权重总量

未量化部分包括：

- 一份 $V\times D$ 的 BF16 embedding，同时也是 LM head；
- 每层 4 个长度为 $D$ 的 RMSNorm scale；
- 每个 attention 的 Q-Norm 和 K-Norm；V-Norm 没有可学习 scale；
- 最终长度为 $D$ 的 RMSNorm；
- 每层一个 layer scalar。

其 BF16 元素数为

$$
\begin{aligned}
P_{\mathrm{BF16}}
={}&VD+48(4D)+2(40d_s+8d_g)+D+48\\
={}&1,007,402,800.
\end{aligned}
$$

因此未量化权重占

$$
M_{\mathrm{BF16}}=2P_{\mathrm{BF16}}
=2,014,805,600\ \mathrm{B}.
$$

全量加载 W4A16 模型权重的理论 tensor payload 为

$$
\boxed{
M_{\mathrm{weight}}
=5,790,597,120+2,014,805,600
=7,805,402,720\ \mathrm{B}
}
$$

即 **7.805 GB = 7.269 GiB**。

长期量化 checkpoint 的 7 个 Safetensors 分片实测合计为 `7,805,505,952 B`，只比理论
payload 多 `103,232 B`，差额来自 Safetensors header 和对齐。这个结果说明推导与实际
checkpoint 一致。

不能用“12B × 0.5 byte = 6 GB”代替上述计算，原因是约 10.90B 个 Linear 权重才是 INT4，
约 0.94 GiB 参数的 embedding 仍以 BF16 保存，group-64 scale 还额外占约 325 MiB。

### 1.4 普通 KV Cache 与 $B,L$ 的关系

设预分配 batch capacity 为 $B$，`max_seq_len` 为 $L$。KV Cache 使用 BF16，K 和 V
各保存一份，所以每个 token、每个 KV 通道共占 $2\times2=4$ byte。虽然 global layer
共享 K/V 的投影权重，但经过 K-Norm、RoPE 和 V-Norm 后，实际 K 与 V 不相同，因此
cache 中仍必须保存两份。

不使用 Ring 时：

$$
\begin{aligned}
M_{\mathrm{KV}}(B,L)
={}&4BL\left(40\times8\times256+8\times1\times512\right)
+48B\times8\\
={}&\boxed{344,064BL+384B\ \mathrm{B}}.
\end{aligned}
$$

最后的 $384B$ 是 48 层的 int64 `lengths` metadata。这里的 $B$ 是实际分配容量；若实现
按 `max_batch_size` 静态预分配，即使当前活跃请求少于 $B$，这部分 tensor 也不会缩小。

当 $B=1,L=2048$ 时，理论值为：

$$
M_{\mathrm{KV}}(1,2048)=704,643,456\ \mathrm{B}.
$$

远程作业 `175769` 实测恰好也是 `704,643,456 B`，逐字节一致。若静态扩展到
$B=10$，则为 `7,046,434,560 B`，仅 KV 就约 6.56 GiB。

### 1.5 Ring KV Cache

滑动层最多访问最近 $W=1024$ 个 token，因此 40 个 sliding layer 只需保留
$\min(L,W)$，8 个 global layer 仍需保留完整的 $L$：

$$
\boxed{
M_{\mathrm{Ring}}(B,L)
=4B\left[81,920\min(L,1024)+4,096L\right]+384B.
}
$$

当 $B=1,L=2048$ 时：

$$
M_{\mathrm{Ring}}=369,099,136\ \mathrm{B}.
$$

远程实测同样为 `369,099,136 B`。它比普通连续 KV 少 `335,544,320 B`，即 320 MiB、
47.62%。这部分节省来自滑动层旧 KV 的生命周期已经确定结束，所以能够被环形槽覆盖。

### 1.6 Paged KV 为什么不会自动省显存

当前 Paged KV 的 block size 为 $P=16$。对于 $L=2048$：

$$
b_g=\left\lceil\frac{L}{P}\right\rceil=128.
$$

Ring sliding window 的起点可能落在 page 中间，因此长度为 1024 的逻辑窗口最坏会跨
65 个 page，即 $b_s=65$。如果仍然为每个 batch slot 预留最坏容量，则：

$$
\begin{aligned}
M_{\mathrm{PagedRing}}(B,L)
={}&4BP\left(81,920b_s+4,096b_g\right)\\
&+8B(40b_s+8b_g)+384B.
\end{aligned}
$$

第二行是 int64 block table 和 `lengths`。代入 $B=1,L=2048$ 得
`374,371,008 B`，与远程实测完全一致。它反而比连续 Ring 多 `5,271,872 B`：

- 40 个 sliding layer 各多一个边界 page，共 `5,242,880 B`；
- block table 占 `28,992 B`。

因此，Paged Attention 只提供“逻辑页映射、分配、回收”的机制，并不会凭空把已经
预分配的物理 pool 还给 PyTorch。只有把 page 生命周期真正用于较小的共享物理池，才会
降低 allocated 显存：请求进入时按最大生命周期预留 block，请求结束时立即 `release()`，
后续请求复用这些 block，同时物理 pool 不再按“每个 slot 的最坏值之和”创建。

本项目第一版 batch-10 Paged+Ring pool 为：

$$
10\times374,371,008=3,743,710,080\ \mathrm{B}.
$$

最终让同类 layer 在 10 个 slot 间共享物理页，并把每层 sliding/global pool 分别设为
530/776 blocks，实测占 `2,982,443,904 B`，减少 `761,266,176 B`，约 726 MiB。请求
结束后释放页面是“允许跨请求复用”的必要条件；初始化时确实创建更小的物理 tensor，
才是进程 allocated 显存实际下降的直接原因。

### 1.7 为什么进程峰值比理论权重大

全量 GPU、`int4_reference`、batch 1、`max_sequence_length=64` 的短输入实验中，远程
实测峰值 allocated 为 `8,252,107,776 B`。它比纯权重理论值多：

$$
8,252,107,776-7,805,402,720
=446,705,056\ \mathrm{B}
=426.01\ \mathrm{MiB}.
$$

已知额外部分包括：

| 额外内容 | 字节数 |
| --- | ---: |
| 64-token、48 份 FP32 RoPE cache | 7,340,032 |
| batch 1、64-token 普通 KV Cache | 22,020,480 |
| Reference W4A16 单次反量化路径的实测临时量 | 355,860,480 |
| 其余激活、logits、attention 临时量及分配对齐 | 约 61,484,064 |

RoPE cache 是运行时 buffer，不是学习到的模型权重；reference W4A16 还会临时展开 INT4
code 并生成完整 BF16 权重。二者都不应塞进“理论权重 payload”，但会进入端到端峰值。
当前自写 fused Triton W4A16 已经删除完整反量化临时权重；最终配置还把相同 RoPE 配置
从 48 份去重为 sliding/global 各一份。

最终 full-10 的 `9,742,262,784 B` 峰值也不能直接与 7.269 GiB 的全量权重相减：最终
运行使用混合常驻与 Offloading，并同时包含 batch-10 KV pool、激活、双 GPU 权重 buffer
和库 workspace。比较显存时必须先统一 workload 和统计边界。

## 2. 不预设掩码形状的稀疏注意力

### 2.1 不能先算完整 $QK^T$ 再取 Top-k

最直接的想法是先计算完整 attention score，再为每个 query 保留 Top-k 位置。这样得到的
掩码确实是动态的，但最昂贵的 $QK^T$ 已经完成，计算量仍为 $O(L^2d)$；如果 score 被
写入显存，中间张量仍为 $O(L^2)$。它只让 softmax/PV 的一部分结果为零，不能真正获得
稀疏注意力的主要效率收益。

有效的动态稀疏实现必须在精确 QK 之前，用比完整 head dimension 更便宜的表示筛选候选。

### 2.2 内容感知的块级两阶段方案

可以直接复用本项目 Paged KV 的 16-token block，设计如下：

1. 每个 KV block 写满时，把块内 K 投影到较小维度 $d_r$，例如 32 或 64，再做 pooling，
   得到一个或少量摘要向量 $s_b$。摘要只计算一次。
2. 对 query 计算低维路由向量 $r(q_t)$，用
   $a_{t,b}=\langle r(q_t),s_b\rangle$ 对因果范围内的 block 做廉价近似打分。
3. 候选集合同时包含最近若干 block、BOS/system/attention-sink block，以及超过阈值或
   Top-p 的远程 block。再设置 `K_max`，约束最坏计算量和尾延迟。
4. 用 `row_ptr + block_ids` 这样的 CSR 结构保存每个 query tile 的候选 block。扩展自写
   Paged FlashAttention kernel，使其按 block table 直接读取这些物理页，以 online
   softmax 计算精确 QK、softmax 和 PV，不生成 dense mask 或 dense scores。

可写为：

$$
\mathcal C_t=\mathcal C_{\mathrm{recent}}
\cup\mathcal C_{\mathrm{sink}}
\cup\operatorname{TopP}_b(a_{t,b}).
$$

候选集合由内容和 query 共同决定，不再是预先画好的固定条带。为减少 GPU 线程分歧，
prefill 时可让一个 query tile 共用候选集合；GQA 中共享同一个 KV head 的 query heads
也可以共用候选集合。

### 2.3 路由器如何获得可靠性

未经训练的低维路由器容易漏掉关键位置。可在校准或微调阶段周期性运行稠密 attention
作为 teacher，训练路由器覆盖 teacher 中注意力质量最大的 block，再用稀疏前向微调模型。

如果被删除位置的稠密 attention 概率质量为 $\epsilon$，并且
$\lVert v_i\rVert\le V_{\max}$，对保留概率重新归一化后的单头输出误差可以粗略约束为
$2\epsilon V_{\max}$。因此训练目标应提高保留的 attention-mass recall，而不只是猜中
score 最大的一个位置。

### 2.4 与滑动窗口注意力的模型效果比较

动态内容路由的潜在优点是：在相同 token 预算下，它能找到窗口外的人名、指令、证据和
重复主题；滑动窗口则必然丢弃所有窗口外信息。因此在超长文本检索和跨段依赖任务上，
动态方案可能优于固定窗口。

风险也更大：

- 路由漏召回会直接删除关键上下文，错误不再只由距离决定；
- Top-k/阈值选择是离散的，训练和负载均衡更困难；
- Gemma4 原本按“40 个 sliding layer + 8 个 global layer”的固定结构训练，直接替换
  掩码会产生分布偏移，未经稀疏微调时 NLL 可能恶化；
- Gemma4 已有 global layer，动态化 sliding layer 的额外质量收益可能没有纯滑窗模型大。

一个保守方案是保留原来的 1024-token 局部窗口，再动态增加少量远程 block。这样质量风险
较小，但它只增加长程能力，并不会减少原有局部 attention 的计算量。若目标是加速，就必须
用更小的候选预算替换一部分窗口，此时需要重新评测 `delta_nll`。

### 2.5 与滑动窗口注意力的推理效率比较

设每个 query 选择 $K$ 个 block，每块 $P$ 个 token。两种方案的主要差异为：

| 项目 | 固定滑动窗口 | 动态块稀疏 |
| --- | --- | --- |
| 精确 attention | $O(Wd)$ | $O(KPd)$ |
| 候选选择 | 无 | 线性扫描时 $O((L/P)d_r)$ |
| KV 读取 | 连续、规则 | 分页 gather、较不规则 |
| GPU 控制流 | 边界固定，线程一致 | Top-k、索引构造和线程分歧 |
| Kernel 发射 | 单一路径 | 可能多出路由和索引 kernel |

若路由线性扫描全部摘要，prefill 的低维筛选仍可能达到 $O(L^2d_r/P)$。也可用 LSH 或树形
索引降低查询复杂度，但 GPU 上的索引构建、更新和批量查询更加复杂。只有当 $KP\ll W$、
$d_r\ll d$ 且上下文足够长时，精确 attention 的减少才可能覆盖路由与随机访存开销。

本实验 $L\le2048,W=1024,P=16$，固定窗口最多读取 64 个 block。即使动态方案只选择
16 个 block，精确 attention 理论上少约 75%，也不能直接推导为 4 倍端到端加速，因为
还有路由、Top-k、block table、非连续读取以及 W4A16、H2D 等其他热点。当前上下文较短，
规则滑窗很可能更快；动态方案更适合 $L\gg W$ 的场景。

### 2.6 对 KV 显存的影响

这是动态稀疏与滑动窗口最重要的区别之一。滑动层能使用 Ring KV，是因为超过窗口的旧
token 此后永远不会再被访问。内容感知路由可能在未来重新选中任意历史 block，因此默认
必须保留完整历史 KV：

$$
O(Wd)\quad\longrightarrow\quad O(Ld)+O((L/P)d_r).
$$

后一项是 block 摘要。也就是说，动态稀疏首先减少 attention 计算和 KV 读取带宽，**不会
自动减少 KV Cache 显存**，甚至会破坏 Ring 带来的容量收益。

若还要降低显存，必须额外设计生命周期策略，例如按累计重要性淘汰旧 block、把冷 KV
量化/压缩，或只把摘要和热点 KV 留在 GPU、将冷 KV offload 到 CPU。淘汰会带来不可恢复
的信息损失，CPU offload 则可能导致候选命中后等待 PCIe 传输。

因此对当前项目更稳妥的落点是：保留 40 个 sliding layer 及其 Ring KV；如果要验证动态
稀疏，先用于本来就保存完整历史的 8 个 global layer。验收时应同时报告
attention-mass recall、`delta_nll`、每个 query 的候选 block 数、路由耗时、KV 读取字节、
full-10 端到端时间和峰值显存。

## 3. 分层量化与推理 Offloading 的异同

### 3.1 分层量化的权重生命周期

分层量化不是把完整 BF16 模型先加载到 GPU 再逐层处理，而是一条离线流式转换流水线：

$$
\text{磁盘 BF16 checkpoint}
\rightarrow\text{当前层 GPU BF16}
\rightarrow\text{GPU GPTQ}
\rightarrow\text{CPU INT4 writer buffer}
\rightarrow\text{磁盘 INT4 checkpoint}.
$$

当前实现的步骤为：

1. 在 `meta` device 上只构造第 $i$ 层的形状，再从 Safetensors 读取并物化该层权重；
2. 校准 hidden states 长期保存在 CPU，只按 micro-batch 搬到 GPU；Linear 输入由 hook
   捕获后放回 CPU，并由 `max_calibration_tokens` 限制数量；
3. GPTQ 为当前 Linear 构造 FP32 Hessian、Cholesky 因子和误差补偿工作区，生成 INT4
   `qweight`、FP16 `scale` 和可选 zero point；
4. `CheckpointWriter` 把结果转到 CPU，以不超过约 1 GiB 的 shard buffer 流式写盘；
5. 当前配置启用 `propagate_quantized`，所以还会把量化权重临时反量化，重放当前层，并
   把量化后的层输出存回 CPU，作为下一层的校准输入；
6. `_quantize_layer` 返回后不再持有当前层，GPU 存储由后续层复用。

因此量化峰值主要由“最大单层 BF16 权重 + 当前 GPTQ FP32 工作区 + 当前 micro-batch”决定，
而不是 48 层权重之和。PyTorch caching allocator 可能继续保留 reserved block，因此
`nvidia-smi` 不立即下降不代表当前层仍是活跃 tensor。

### 3.2 推理 Offloading 的权重生命周期

推理使用的是已经生成好的 INT4 checkpoint。被卸载的权重长期放在 CPU pinned memory，
每层的 `qweight`、`scale`、Norm 等 tensor 被合并成一个连续 byte payload。GPU 上有两个
可复用权重 buffer、一条 transfer stream 和默认 compute stream：

1. 前向开始前，在 transfer stream 上把第 0 层异步复制到第一个 GPU buffer；
2. compute stream 使用第 $i$ 层前等待该层的 ready event；
3. 计算第 $i$ 层时，transfer stream 把第 $i+1$ 层预取到另一个 buffer；
4. 第 $i$ 层计算结束后记录 compute-done event，并把模块 tensor 重新绑定到 CPU payload；
5. transfer stream 在覆盖旧 buffer 前等待 compute-done event；
6. 权重只读，GPU 没有产生需要保存的新版本，所以释放时不执行 D2H 回拷。

数据流为：

$$
\text{CPU pinned INT4}
\xrightarrow[\text{每次前向重复}]{\text{异步 H2D}}
\text{GPU 双缓冲}
\rightarrow\text{执行当前层}
\rightarrow\text{复用 buffer}.
$$

最终配置还是混合常驻：embedding、final norm、共享 RoPE cache 和 44 层 MLP/Norm 常驻
GPU；48 层 attention 权重参加 Offloading，另有 4 个 sliding layer 的 MLP 也保留在
Offloading payload 中。这是在 PCIe 流量和 10 GiB 显存余量之间的折中。

### 3.3 相同点和不同点

| 比较项 | 分层量化 | 推理 Offloading |
| --- | --- | --- |
| 共同基础 | 利用 DecoderLayer 严格按顺序执行，只让少数层进入 GPU 工作集 | 同左 |
| 主要目的 | 在有限显存中完成一次 GPTQ 转换 | 为 KV、激活和 batch 腾出在线推理显存 |
| 后备存储 | 输入 BF16 与输出 INT4 都在磁盘 checkpoint | 卸载后的 INT4 权重在 CPU pinned memory |
| GPU 工作集 | 当前层、GPTQ 工作区和校准 micro-batch | 常驻部分、当前层与下一层双 buffer |
| 权重是否改变 | BF16 有损转换为 INT4、scale/zero | 只读搬运，权重内容不变 |
| 数据方向 | BF16 H2D，INT4 D2H 后写盘 | 只做 H2D；释放不做无意义 D2H |
| 执行频率 | 一个 checkpoint 通常只量化一次，可长期缓存 | 每次 forward 都发生，decode 每个 token 都会再次遍历 |
| 跨层重叠 | 当前实现逐层同步处理，没有下一层双缓冲预取 | transfer/compute stream、双 buffer、event 异步重叠 |
| 激活管理 | 校准激活放 CPU，按 micro-batch 搬入并将层输出放回 CPU | 请求 hidden states 留在 GPU，主要移动权重 |
| 释放方式 | 层对象生命周期结束后由 allocator 复用 | `release()` 显式重绑 CPU view，并用 event 保护 buffer |
| 精度影响 | INT4 转换本身产生误差，逐层传播量化输出以适配累计误差 | Offloading 本身不改变数值，不产生额外模型误差 |
| 主要代价 | Hessian/Cholesky/误差补偿和磁盘 I/O，但只做一次 | 重复 PCIe H2D；预取不及时会直接形成在线等待 |

二者的共同本质是按层缩小 GPU 工作集，但不能把分层量化简单称为推理 Offloading：前者
管理的是一次性、会改变权重的离线转换生命周期，后者管理的是高频、只读权重的在线驻留
生命周期。二者是互补关系：W4 将每次 Offloading 的传输字节数降到约 4.25 bit/weight，
Offloading 则让量化模型在 10 GiB 上仍能给 KV Cache、激活和更大 batch 留出空间。

## 证据来源

- 模型结构与实验要求：`docs/Lab5-Gemma4/index.md`
- Linear 存储格式：`src/hpc101_infer/layers/linear.py`
- K/V 共享和 attention 投影形状：`src/hpc101_infer/layers/attention.py`
- LM head 权重共享：`src/hpc101_infer/models/gemma4.py`
- KV、Ring 和 Paged 分配：`src/hpc101_infer/runtime/kv_cache.py`
- 分层量化生命周期：`src/hpc101_infer/quantization/pipeline.py`
- 异步双缓冲 Offloading：`src/hpc101_infer/runtime/offloading.py`
- 远程显存实测：`reports/offloading-2026-08-26.md`、
  `reports/ring-kv-cache-2026-08-27.md`、`reports/paged-attention-2026-08-27.md`、
  `reports/shared-kv-hybrid-residency-2026-08-27.md`
