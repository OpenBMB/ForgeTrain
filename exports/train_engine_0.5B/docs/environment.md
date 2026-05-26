# Environment Requirements

TrainingEngine 的完整环境清单。`README.md` 顶部的「Quick Start → 环境要求」段以一行链接指向本文档，避免在主 README 里堆细节。

---

## 1. 硬件


| 项      | 要求                                                                     |
| ------ | ---------------------------------------------------------------------- |
| GPU 架构 | **SM90 (Hopper)** — NVIDIA H100 / H200 / H800（仅 H100 完成 production 验证） |
| 单卡显存   | ≥ 80 GB（H100 80GB 验证过；40GB SKU 未测试）                                    |
| 多卡互联   | NVLink + IB / RoCE（DP-only，跨节点用 NCCL allreduce / reduce_scatter）       |
| 推荐规模   | 8× H100（smoke / 对齐测试）· 64× H100（production，三阶段 30 万 step ≈ 数日）         |


> ❌ **不支持**：SM89 及以下（A100 / RTX 4090 等）。FA4 + CuTeDSL kernels 依赖 SM90 的 WGMMA + TMA 指令；同时 GEMM 算子做了针对 H100 的 AOT shape 特化。

---

## 2. 系统 / 驱动


| 项             | 版本要求                                             | 备注                         |
| ------------- | ------------------------------------------------ | -------------------------- |
| OS            | Linux x86_64（Ubuntu 22.04 / NGC PyTorch image 等） | macOS / Windows 不支持        |
| NVIDIA Driver | ≥ R535（CUDA 12.x 兼容）                             | —                          |
| CUDA Toolkit  | ≥ 12.0（推荐 12.4+）                                 | 用于 `nvcc` 编译自研 CUDA kernel |
| cuDNN         | 随 CUDA 安装                                        | —                          |


---

## 3. Python 运行时


| 包                 | 版本要求               | 用途                                                           |
| ----------------- | ------------------ | ------------------------------------------------------------ |
| Python            | ≥ 3.11             | 框架使用 `@dataclass(slots=True)` / `tomllib` 等 3.11+ 特性         |
| PyTorch           | ≥ 2.4              | CUDA 12.x build，需含 `torch.cuda.graphs` 与 `torch.distributed` |
| TransformerEngine | ≥ 1.x              | RMSNorm fwd/bwd、attention TE 接口                              |
| Triton            | ≥ 3.3              | Fused CE / SwiGLU / RMSNorm+residual / RoPE / Adam-sync      |
| FlashAttention    | 可选                 | 仅作为 FA4 不可用时的 debug fallback（`ATTN_V1_ALLOW_NON_FA4_FWD=1`）  |
| `tomli`           | 仅 Python < 3.11 兜底 | 一般用不到                                                        |


### 数据通路（二选一）


| 路径                   | 依赖                                         | 用途                                      |
| -------------------- | ------------------------------------------ | --------------------------------------- |
| HuggingFace datasets | `transformers` · `datasets` · `tokenizers` | 通用开源数据集，推荐起步使用                          |
| modelbest_sdk        | `modelbest_sdk`                            | 预 tokenize 流式数据，性能更优但需配合 ModelBest 数据资产 |


---

## 4. CuTeDSL Toolchain（自研 GEMM / FA4 必装）

以下 5 个 wheel 来自 NVIDIA CUTLASS-DSL 发行版，**在启用 `CUSTOM_GEMM=1` 或 `OP_ATTENTION=v1` 时必装**。`scripts/entry_train.sh` 会按需自动安装，本地手动安装时：

```bash
pip install --no-deps \
    nvidia_cutlass_dsl-4.4.2-py3-none-any.whl \
    nvidia_cutlass_dsl_libs_base-4.4.2-cp312-cp312-manylinux_2_28_x86_64.whl \
    cuda_bindings-12.9.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    cuda_python-12.9.6-py3-none-any.whl \
    nvidia_cutlass-4.4.2-py3-none-any.whl
```

> 也可指向已解压的 CUTLASS-DSL 根目录：`export CUTLASS_DSL_FALLBACK_DIR=/path/to/nvidia_cutlass_dsl`，框架运行时会把 `python_packages/` 与 `lib/` 自动加到 `sys.path` / 链接路径。

---

## 5. 安装

```bash
git clone https://github.com/<org>/training_engine.git
cd training_engine
pip install -e .

# 可选：安装 HuggingFace 数据通路
pip install transformers datasets

# 可选：CuTeDSL（启用自研 kernel 时必装，见上一节）
```

### 安装验证

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

---

## 6. 环境变量

框架内部禁止裸读 `os.environ`，所有配置走 `EngineConfig`。仅以下变量是「process-external 契约」，由 torchrun / 容器运行时注入：


| 变量                                                     | 来源                | 用途         |
| ------------------------------------------------------ | ----------------- | ---------- |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE`                   | torchrun          | 分布式拓扑      |
| `MASTER_ADDR` / `MASTER_PORT`                          | torchrun          | rendezvous |
| `CUDA_DEVICE_MAX_CONNECTIONS`                          | 用户                | CUDA 连接数上限 |
| `NVTE_FUSED_ATTN` / `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | TransformerEngine | TE 后端选择    |
| `PYTORCH_CUDA_ALLOC_CONF` / `CUBLAS_WORKSPACE_CONFIG`  | PyTorch / cuBLAS  | 分配器与确定性    |


完整白名单：`[src/training_engine_tensor/ENV_WHITELIST.md](../src/training_engine_tensor/ENV_WHITELIST.md)`

---

## 7. 已知不兼容场景

- ❌ **SM80 及以下卡**（A100 / V100 / RTX 30 系等）：FA4 + AOT GEMM 都依赖 SM90 指令。
- ❌ **AMD ROCm / Intel GPU**：仅 NVIDIA + CUDA。
- ❌ **macOS / Windows**：Triton + TransformerEngine + cpp_extension 全栈仅 Linux。
- ⚠️ **CUDA 11.x**：未适配；CuTeDSL wheel 要求 CUDA 12.x。
- ⚠️ **Python 3.10 及以下**：`tomllib` 不存在，且部分 dataclass 特性缺失。

