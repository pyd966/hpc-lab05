# 双 GPU Buffer 与合并权重 Payload 报告

日期：2026-08-27
代码提交：本轮提交前的工作区（最终提交见 Git 历史）
远程作业：`174907`（轻量 CUDA 单元）、`174926`（模型 correctness、small/public）、`175102`（合并后 profile）

## 本轮目标

上一版异步 Offloading 为每个量化 tensor 单独执行 H2D 和 D2H。profile 显示大量小 copy 是主要瓶颈。本轮实现两个改动：

1. 使用两个固定大小、可复用的 GPU byte buffer，当前层和预取的下一层分别占用一个 buffer。
2. 将每层的 `qweight`、`scales`、`zeros`、norm 参数和 Rotary buffer 合并为一个连续的 CPU byte payload，每层只发射一次 H2D copy。

推理权重是只读的，所以释放层时不再把相同权重 D2H 拷回 CPU，而是记录计算完成 event，恢复 CPU 视图，之后复用 buffer 前等待该 event。

## 实现细节

- `_pack_layer()` 为每层建立连续的 pinned `uint8` payload，并为每个原始 tensor 保存 dtype、shape、byte offset 描述。
- GPU 上只分配两个大小为“最大层 payload”的 `uint8` buffer，而不是为 48 层永久分配 GPU 权重。
- `prefetch(i)` 在 transfer stream 上等待被复用 buffer 的 `compute_done` event，然后执行一次 payload H2D；随后把参数和 buffer 的 typed view 绑定到对应 module。
- `wait(i)` 在计算流上等待该层 ready event。
- `release(i)` 在计算流记录完成 event，并立即把 module 参数切回 CPU payload view；后续 buffer 复用由 transfer stream 等待该 event 保护。
- 该设计仍使用一个 transfer stream 和当前计算 stream；H2D 与计算可以重叠，但 H2D 之间仍在同一个 transfer stream 上串行。

核心代码位于 [offloading.py](/home/h3250106394/lab05/src/hpc101_infer/runtime/offloading.py:29)。

## 轻量 CUDA 单元测试

作业 `174907` 使用混合 `bfloat16`、`float32`、`uint8` 的两个 dummy layer，验证了 payload 打包、typed view、两个 buffer 交替复用、event 保护和释放后 CPU 权重保持不变，结果为 `unit_ok`。

## 模型正确性与显存

作业 `174926` 使用 g64 GPTQ INT4 checkpoint、CUDA/bfloat16、`int4_reference`、batch 1，输入 token `[1, 2, 3, 4]`，`max_sequence_length=64`：

| 模式 | 峰值 GPU allocated |
| --- | ---: |
| 无 Offloading | `8,252,107,776` B（约 7.69 GiB） |
| 双 buffer + 合并 payload | `2,682,359,808` B（约 2.50 GiB） |

- `max_abs_diff = 0.0`
- `torch.allclose(..., atol=1e-4, rtol=1e-4) = True`
- 合并后总 CPU payload：`5,799,469,248` B
- 两个 GPU buffer 总容量：`257,531,912` B
- prefill H2D：`5,799,469,248` B
- prefill D2H：`0` B
- correctness 测试中预取 48 层，结束时没有驻留层。

## 端到端结果

短测试使用 `performance_small.jsonl`、4 个请求、60 个输出 token、batch 1、seed 0：

| 配置 | 总耗时 | 生成 token/s |
| --- | ---: | ---: |
| g64 baseline（提交 `956e507`） | `76.138967 s` | `0.788033` |
| 双 buffer + 合并 payload | `78.900482 s` | `0.760452` |

相对 baseline，当前实现慢 `3.6269%`。但它比上一版逐 tensor D2H 的 Offloading 明显更接近 baseline：D2H 已完全消除，H2D copy 数量大幅下降。

完整 `performance_public.jsonl` 队列在同一作业中完成了 10 个请求和 320 个输出 token，summary 为：

- `elapsed_s = 425.019842 s`
- `generated_tokens_per_s = 0.752907`
- `requests_per_s = 0.023528`

运行中曾出现 allocator 警告：尝试申请 `1,048,576,000` B 时只剩 `306,184,192` B；进程随后仍完成并输出 summary。因此该结果说明 Offloading 恢复了队列可运行性，但仍有明显显存压力，不能视为无 OOM 的干净验收结果，更不能满足 `<=36 s` 目标。

本轮没有改变 GPTQ 量化参数，因此没有重新计算 `delta_nll`；质量结果沿用 g64 量化版本，后续只需在量化 checkpoint 不变时复用原质量评测。

## Profile 对比

profile 使用 `performance_small.jsonl` 第一条请求，单请求、最大输出 32 token，g64、batch 1、最大序列长度 2048。作业 `175102` 成功完成 profile，未导出大型 trace。

| 指标 | 上一版异步 Offloading | 双 buffer + 合并 payload |
| --- | ---: | ---: |
| Self CUDA 总时间 | `69.980 s` | `55.307 s` |
| `aten::copy_` self CUDA | `47.973 s` | `33.509 s` |
| H2D copy 次数 | `33,280` | `1,536` |
| D2H copy 次数 | `33,311` | `0` |
| H2D CUDA 时间 | `14.387 s` | `14.029 s` |
| D2H CUDA 时间 | `14.136 s` | `0` |
| `aten::copy_` 占比 | `68.55%` | `60.59%` |

合并后 profile 的主要热点为：

- `aten::copy_`：`33.509 s`，`60.59%`，`98,734` 次；其中还包括 INT4 反量化产生的内部 copy。
- H2D（Pinned -> Device）：`14.029 s`，`1,536` 次。
- unrolled elementwise kernel：`9.846 s`，`20,992` 次。
- 128-thread elementwise kernel：`9.504 s`，`20,992` 次。
- `aten::mul`：`9.161 s`，`41,344` 次。
- `aten::mm`：`3.607 s`，`10,528` 次。
- `Command Buffer Full`：`7.018 s`，`62,085` 次。

H2D 仍约占 14 秒，说明合并主要消除了小 copy 和 D2H，无法消除“每个 decode token 都重新加载 48 层权重”的带宽成本。profile 统计中总 H2D payload 为 `192,864,327,680` B，约 179.7 GiB。

## 结论与下一步

本轮达成了两个局部目标：

1. 由两个固定 GPU buffer 控制驻留空间，避免为每层动态创建 GPU 权重。
2. 将每层数十个权重 tensor 合并为一次 H2D，并消除只读推理中不必要的 D2H。

结果是 profile CUDA 时间下降约 21.0%，但普通 small 端到端仍比全量 GPU baseline 慢 3.6%。public 队列可以完成但耗时 425 s，远未达到 36 s；长 prompt 的 eager attention 和静态 KV cache 仍是显存瓶颈。

下一步应在本提交基础上实现 Ring KV cache，再实现 Paged Attention；随后再考虑 fused W4A16 GEMM 和减少每 token 的层权重搬运。
