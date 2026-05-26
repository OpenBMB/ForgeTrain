# Environment Requirements

TrainingEngine (8B) 的完整环境清单。`README.md` / `README_zh.md` 顶部
的「Quick Start → 环境要求」段以一行链接指向本文档，避免在主 README
里堆细节。

---

## 1. 硬件


| 项      | 要求                                                                     |
| ------ | ---------------------------------------------------------------------- |
| GPU 架构 | **SM90a (Hopper)** — NVIDIA H100 SXM5（仅 H100 SXM5 完成单机环境验证）            |
| 单卡显存   | ≥ 80 GB（H100 SXM5 80GB 验证过；40GB SKU 未测试）                               |
| 多卡互联   | NVLink 单机 8 卡（DP=4 + TP=2）；多节点 NCCL allreduce / reduce_scatter 支持但非主用例 |
| 推荐规模   | **8× H100 单机**（TP=2 / DP=4，MFU 50.9% / GAS=8）                          |


> ❌ **不支持**：SM89 及以下（A100 / RTX 4090 等）。`flash_attn_dsl/` 与
> 自研 GEMM kernel 依赖 SM90a 的 WGMMA + TMA 指令；GEMM 算子的 kernel
> template 也对 H100 做了 shape 特化。

---

## 2. 系统 / 驱动


| 项             | 版本要求                                             | 备注                                |
| ------------- | ------------------------------------------------ | --------------------------------- |
| OS            | Linux x86_64（Ubuntu 22.04 / NGC PyTorch image 等） | macOS / Windows 不支持               |
| NVIDIA Driver | ≥ R535（CUDA 12.x 兼容）                             | —                                 |
| CUDA Toolkit  | ≥ 12.0（推荐 12.4+）                                 | `nvcc` 用于 CuTeDSL JIT/AOT 出 `.so` |
| cuDNN         | 随 CUDA 安装                                        | attention baseline 走 cuDNN        |


---

## 3. Python 运行时


| 包                                       | 版本要求     | 用途                                                           |
| --------------------------------------- | -------- | ------------------------------------------------------------ |
| Python                                  | ≥ 3.11   | 框架使用 `@dataclass(slots=True)` / `tomllib` 等 3.11+ 特性         |
| PyTorch                                 | ≥ 2.4    | CUDA 12.x build，需含 `torch.cuda.graphs` 与 `torch.distributed` |
| `triton`                                | ≥ 3.3    | Triton 融合 kernel（RMSNorm、SwiGLU bwd 等）                       |
| `transformer-engine`                    | ≥ 1.0    | 多 tensor fused Adam（`te.optimizers`） + 部分 baseline 算子路径      |
| `nvidia-cutlass` · `nvidia-cutlass-dsl` | == 4.5.0 | 自研 GEMM + flash-attention fwd 的 CuTeDSL runtime（详见 §4）       |
| `numpy`                                 | —        | tensor 互操作（HF data → `torch.Tensor`）                         |
| `tomli`                                 | 可选       | 仅 Python < 3.11 才需要兜底；3.11+ 用内置 `tomllib`                    |


> 上述 import-time 依赖已在 `pyproject.toml` 的 `[project] dependencies` 中固定，
> 执行 `pip install -e .` 会一次性拉齐。`tomli` 仅在 Python < 3.11 时手动安装。

### 数据通路


| 路径                   | 依赖                                            | 用途                                   |
| -------------------- | --------------------------------------------- | ------------------------------------ |
| HuggingFace datasets | `transformers` · `datasets` · `sentencepiece` | 唯一数据通路；HF Hub 或本地 Parquet/Arrow/JSON |


> 8B 支持 HuggingFace `datasets`。`scripts/entry_hf_pretrain.sh`
> 启动时若发现 `datasets` 未安装，会自动 `pip install` 到
> `${ENGINE_ROOT}/.hf_site` 目录（不污染系统 Python）。

---

## 4. CuTeDSL Toolchain（自研 GEMM / FA fwd 必装）

以下 5 个 wheel 来自 NVIDIA CUTLASS-DSL 发行版，**在启用
`CUSTOM_GEMM=1` 或 `OP_ATTENTION=v1` 时必装**。本地手动安装：

```bash
pip install --no-deps \
    nvidia_cutlass_dsl-4.5.0-py3-none-any.whl \
    nvidia_cutlass_dsl_libs_base-4.5.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    cuda_bindings-12.9.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    cuda_pathfinder-1.5.4-py3-none-any.whl \
    apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

> 也可指向已解压的 CUTLASS-DSL 根目录：
> `export CUTLASS_DSL_PATH=/path/to/nvidia_cutlass_dsl/python_packages`，
> 框架运行时会把它自动加到 `sys.path`。

---

## 5. 安装

```bash
git clone https://github.com/<org>/training_engine_8b.git
cd training_engine_8b
pip install -e .

# HuggingFace 数据通路（必装）
pip install transformers datasets sentencepiece

# 可选：CuTeDSL（启用自研 kernel 时必装，见上一节）
```

### 安装验证

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

### 算子压力测试（推荐首次训练前过一遍）

```bash
pytest tests/ -v
```

详见 `tests/` 下三个 `test_stress_*.py` 文件。

---

## 6. 环境变量

框架内部禁止裸读 `os.environ`，所有配置走 `EngineConfig`。仅以下变量
是「process-external 契约」，由 torchrun / 容器运行时注入：


| 变量                                                    | 来源               | 用途                                        |
| ----------------------------------------------------- | ---------------- | ----------------------------------------- |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE`                  | torchrun         | 分布式拓扑                                     |
| `MASTER_ADDR` / `MASTER_PORT`                         | torchrun         | rendezvous                                |
| `CUDA_DEVICE_MAX_CONNECTIONS`                         | 用户               | CUDA 连接数上限                                |
| `PYTORCH_CUDA_ALLOC_CONF` / `CUBLAS_WORKSPACE_CONFIG` | PyTorch / cuBLAS | 分配器与确定性                                   |
| `TORCH_NCCL_USE_COMM_NONBLOCKING`                     | NCCL             | NCCL 非阻塞通信                                |
| `CUTLASS_DSL_PATH`                                    | 用户               | CuTeDSL 解压目录 `sys.path` 提示（wheel 安装时无需设置） |
| `CUSTOM_GEMM` / `OP_ATTENTION` / `OP_GEMM_*`          | 用户               | 算子分发                                      |


完整白名单：`[src/training_engine_tensor/ENV_WHITELIST.md](../src/training_engine_tensor/ENV_WHITELIST.md)`

---

## 7. 已知不兼容场景

- ❌ **SM80 及以下卡**（A100 / V100 / RTX 30 系等）：CuTeDSL kernel 依赖 SM90 WGMMA + TMA。
- ❌ **AMD ROCm / Intel GPU**：仅 NVIDIA + CUDA。
- ❌ **macOS / Windows**：CuTeDSL + cpp_extension + cuda.bindings 全栈仅 Linux。
- ⚠️ **CUDA 11.x**：未适配；CuTeDSL wheel 要求 CUDA 12.x。
- ⚠️ **Python 3.10 及以下**：`tomllib` 不存在，且部分 `@dataclass(slots=True)` / `BooleanOptionalAction` 特性缺失。

