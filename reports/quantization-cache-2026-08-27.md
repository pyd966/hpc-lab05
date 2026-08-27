# 量化结果缓存报告（2026-08-27）

## 结论

量化入口现在默认复用已经完成且仍然有效的量化 checkpoint。首次运行成功后，重复执行同一条量化命令不会再次加载 Gemma4-12B 权重、捕获校准激活或执行 GPTQ；只进行文件签名、JSON 元数据和 safetensors header 校验。需要改变结果时使用 `--force`。

## 实现

- `quantization_cache.json` 写入量化输出目录，与 packed INT4 分片、`manifest.json` 和 `quantization_config.json` 一起保存。
- 缓存键包含源模型目录绝对路径、源目录下所有文件的路径/大小/mtime/ctime/inode、完整 `QuantizationConfig`、校准 token 的 shape/dtype/SHA-256、校准 micro batch、最大校准 token 数和最大分片大小。
- 缓存元数据同时保存输出 `config.json`、manifest、量化配置、index 和所有 `model-*.safetensors` 的大小/mtime/ctime/inode。任一分片被替换、截断或缺失时自动失效。
- 命中后通过 `QuantizedCheckpointSource` 读取 manifest，不读取权重 tensor；缓存损坏、格式版本不匹配或 checkpoint 不完整都会退回正常量化流程。
- `quantize_checkpoint(..., reuse_cache=True)` 默认开启复用；CLI 的 `--force` 映射为 `reuse_cache=False`。

缓存格式版本当前为 `2`。加入新影响量化结果的参数时，应同步加入缓存键并递增格式版本。

## 验证

远程 `lab5` 环境执行：

```text
pytest -q tests/test_quantization_cache.py
2 passed in 3.09s
```

测试覆盖：

1. 缓存写入后可以读取原 manifest；
2. 改变量化配置会失效；
3. 原地修改输出分片会失效；
4. 两次调用量化入口时，假流水线只执行一次。

## Profile

远程任务 `175362` 使用 `python3 -m cProfile -s cumulative -m pytest -q tests/test_quantization_cache.py`，测试本身耗时 `3.76 s`，profile 总耗时 `4.942 s`（包含 Python/torch/pytest 导入）。缓存相关累计开销为：

| 函数 | 调用次数 | 累计时间 |
| --- | ---: | ---: |
| `quantization_cache_key` | 4 | 约 `0.002 s` |
| `load_quantization_cache` | 5 | 约 `0.002 s` |
| `write_quantization_cache` | 2 | 约 `0.001 s` |
| `quantize_checkpoint` | 2 | 约 `0.003 s` |

该 profile 使用极小测试 checkpoint，不能代表 12B 权重首次量化时间；它验证的是命中路径不会进入实际量化流水线。命中时仍需读取源目录的 stat 信息和输出 safetensors header，但不会扫描权重数据。

## 使用方式

默认复用：

```bash
python3 scripts/quantize.py \
  --config config.yaml \
  --model "$MODEL_DIR" \
  --output "$QUANT_DIR" \
  --device cuda
```

强制重算：

```bash
python3 scripts/quantize.py \
  --config config.yaml \
  --model "$MODEL_DIR" \
  --output "$QUANT_DIR" \
  --device cuda \
  --force
```
