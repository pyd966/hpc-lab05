# Lab5 远程硬件参考

本文档记录通过 `hpc submit -p lab5` 在评测分区实际探测到的硬件和软件环境，供量化、offloading、CUDA kernel 和请求调度优化时参考。

## 探测信息

- 探测时间：2026-08-26 UTC
- 分区：`lab5`
- 探测任务：`167992`
- 任务申请资源：4 CPU、1 GPU、24 GiB 内存
- 节点：`j167992-bsp9b`
- 探测命令：`lscpu`、`lscpu -C`、CPU cache sysfs、`nvidia-smi`、PyTorch CUDA device properties

## CPU

### 拓扑和频率

| 项目 | 测量值 |
| --- | --- |
| 架构 | x86_64 |
| 处理器 | Intel Xeon Gold 5418Y，Family 6，Model 143，Stepping 8 |
| 插槽 | 2 |
| 每插槽物理核心 | 24 |
| 总物理核心 | 48 |
| 每核心线程数 | 2（SMT） |
| 逻辑 CPU | 96（`0-95`） |
| 基频/最高频率 | 2.0 GHz / 3.8 GHz |
| 可观测最低频率 | 0.8 GHz |
| NUMA 节点 | 2 |
| NUMA node 0 CPU | `0-23,48-71` |
| NUMA node 1 CPU | `24-47,72-95` |

任务申请的是 4 个 CPU；容器中的 `lscpu` 会显示完整节点的 96 个逻辑 CPU，因此线程池应以任务实际申请数为上限，避免无意中在 4 核配额上启动 96 个线程。

### Cache 层级

缓存行大小为 64 B，测量结果如下：

| 层级 | 类型 | 单实例 | 实例数 | 总容量 | 相联度 | 共享范围 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| L1d | Data | 48 KiB | 48 | 2.25 MiB | 12-way | 每物理核心（SMT 两线程共享） |
| L1i | Instruction | 32 KiB | 48 | 1.50 MiB | 8-way | 每物理核心（SMT 两线程共享） |
| L2 | Unified | 2 MiB | 48 | 96 MiB | 16-way | 每物理核心（SMT 两线程共享） |
| L3 | Unified | 45 MiB | 2 | 90 MiB | 15-way | 每插槽一个，跨该插槽核心共享 |

优化含义：CPU 侧量化和权重打包的内层循环应尽量让工作集落在 L1/L2；跨线程共享数据要注意 64 B 对齐和 false sharing。跨 NUMA 节点访问 L3/内存可能增加延迟，CPU 预处理线程和其数据应尽量绑定在同一 NUMA 节点。

### 指令集和 CPU 特性

远程 `lscpu` 的 `Flags` 包含以下与本实验相关的能力：

- 基础 SIMD：SSE、SSE2、SSSE3、SSE4.1、SSE4.2、AVX、AVX2、FMA、F16C
- AVX-512：`avx512f`、`avx512dq`、`avx512ifma`、`avx512cd`、`avx512bw`、`avx512vl`、`avx512vbmi`、`avx512vbmi2`、`avx512vnni`、`avx512bitalg`、`avx512vpopcntdq`、`avx512bf16`、`avx512fp16`
- 矩阵/低精度：`amx_tile`、`amx_int8`、`amx_bf16`、`avx_vnni`
- 加密/位操作/访存辅助：AES、SHA-NI、VAES、VPCLMULQDQ、BMI1、BMI2、ADX、GFNI、ERMS、CLWB、CLFLUSHOPT、MOVDIRI、MOVDIR64B
- 其他：`xsave`/`xsavec`/`xsaves`、`pclmulqdq`、`rdtscp`、`serialize`、`enqcmd`

这些能力可用于 CPU 端 BF16/INT8 校准、反量化和权重打包；实际启用前仍应检查所用 PyTorch/编译器 kernel 的 dispatch 路径，不要仅凭 CPU flag 假设算子会自动使用 AMX。

## GPU

### 可见设备和 MIG 配置

| 项目 | 测量值 |
| --- | --- |
| GPU | NVIDIA H800 PCIe（Hopper） |
| MIG | Enabled，实例配置 `1g.10gb` |
| Compute Capability | 9.0（PyTorch `major=9, minor=0`） |
| 可见 GPU 数 | 1 |
| MIG SM 数 | 14 multiprocessors/SM |
| 可见显存 | 9984 MiB（约 10 GiB） |
| 探测时显存 | 15 MiB used，9970 MiB free |
| 可见 L2 Cache | 6 MiB（PyTorch device properties） |
| GPU UUID | `GPU-126df02f-669c-88b4-3204-1e2d04d0fc55` |
| MIG UUID | `MIG-cea30b35-d10b-53c0-b357-0c0941217181` |
| PCIe 地址 | `00000000:B8:00.0` |

`nvidia-smi` 对整卡 framebuffer 的权限受限，但 MIG device 的 `Shared FB Memory Usage` 明确报告总量 9984 MiB；因此显存预算应按约 10 GiB MIG 实例计算，而不是按完整 H800 计算。

### 复制、编解码和链路

| 项目 | 测量值 |
| --- | --- |
| Copy Engine | 1 |
| Decoder | 1 |
| Encoder | 0 |
| JPEG/OFA | JPEG 1，OFA 0 |
| PCIe 当前链路 | Gen4 x16 |
| PCIe 设备/主机最大能力 | Gen5 / x16 |
| GPU/SM 时钟（探测瞬时值） | 1755 MHz |
| 显存时钟（探测瞬时值） | 1593 MHz |
| 功耗上限 | 350 W |
| ECC | Enabled；探测时无可纠正或不可纠正错误 |

PCIe 当前运行在 Gen4 x16，权重 offloading 的 Host-to-Device 传输应使用 pinned memory、批量传输和独立 CUDA stream；双缓冲预取只有在传输与计算确实重叠时才有收益。时钟和功耗是探测时状态，不能作为固定性能保证。

### CUDA/PyTorch 环境

| 项目 | 测量值 |
| --- | --- |
| NVIDIA 驱动/KMD | 610.43.02 |
| CUDA 驱动报告 | 13.3 |
| `nvcc` | CUDA 13.3，V13.3.33 |
| PyTorch | 2.13.0+cu132 |
| `torch.version.cuda` | 13.2 |
| `torch.cuda.is_available()` | `True` |

PyTorch 识别的设备字符串为 `NVIDIA H800 PCIe MIG 1g.10gb`。MIG 实例只有 14 个 SM 和 6 MiB 可见 L2，decode 阶段的小矩阵/矩阵向量乘法容易受 kernel launch 和权重读取限制；应分别针对 prefill 与 decode 测试 block size、融合程度和 batch size，不能用完整 H800 的 SM 数来估算吞吐量。

## 对本实验的直接建议

1. **线程数**：任务申请 4 CPU 时，将 OpenMP/线程池限制在 4（必要时再做 1/2/4 的基准），并设置合理的 CPU affinity；不要依据容器显示的 96 个逻辑 CPU 创建线程。
2. **CPU 量化**：优先验证 AVX-512 BF16、AVX-512 VNNI 或 AMX BF16/INT8 的实际 dispatch；INT4 打包和 scale/zero-point 读取按 64 B cache line 对齐。
3. **显存预算**：模型常驻和临时 buffer 总量必须低于 9984 MiB；MIG 上给 kernel 的并行度按 14 SM 设计。
4. **L2/访存**：6 MiB L2 较小，decode 时应减少重复读权重和中间张量；融合反量化、矩阵乘和必要的激活有助于降低显存流量。
5. **Offloading**：PCIe Gen4 x16 是当前链路，使用 pinned host memory、批量 H2D、异步 stream 和双缓冲，并用 CUDA event 验证是否真正重叠。
6. **测量方法**：固定 warmup 后使用 CUDA Event 分别测量 prefill/decode；同时记录显存峰值、H2D 时间和 GPU kernel 间空隙，避免只看单个 kernel 加速比。

