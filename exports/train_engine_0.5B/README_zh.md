# ForgeTrain Engine

### 全程由 AI Agent Loop 自动构建的 LLM 预训练框架

**🤖 100% AI 自动编写 · 🚀 H100 上 MFU 44.13% · 📈 较 Megatron-LM 提升约 10% · ✅ 生产环境验证**

> [ForgeTrain](../../README.md) monorepo 的子项目。

[English](./README.md) | [中文](./README_zh.md)

[License](../../LICENSE)
[Python](https://www.python.org)
[CUDA](https://developer.nvidia.com/cuda-toolkit)
[PyTorch](https://pytorch.org)
[GPU](https://www.nvidia.com/en-us/data-center/h100/)
[MFU](#性能)

[HuggingFace](https://huggingface.co/openbmb/MiniCPM4-ForgeTrain-0.5B)
[ModelScope](https://modelscope.cn/<org>)
[Docs](./docs)
[Discord](https://discord.gg/<invite>)

⭐ 如果觉得有用，欢迎 Star · 🤝 站在 CUTLASS、FlashAttention-4、TransformerEngine 等开源工作的肩膀上

---

**TrainingEngine** 是一套面向 H100 (SM90) 的 LLM 预训练框架，**框架代码（未计入基于 Dao-AILab FlashAttention-4 / NVIDIA CuTeDSL 上游移植的部分）由一个 AI Agent Loop 端到端自动撰写、调试与优化，全程无人为代码改动**。框架以 MiniCPM4-0.5B 为目标负载，在 H100 上稳定训练，**相对 Megatron-LM baseline 实现约 10% 的 MFU 提升**，并已完成预训练并产出可用的模型权重。

---

## ✨ 亮点

### 🤖 AI 构建，生产验证

- **100% Agent Loop 自动产出** —— 整个框架由 AI Agent 在 auto-loop 模式下自主产出。人类角色仅提供训练目标与硬件资源，**无任何手动改码、手动调参、手动修 bug**。
- **可自诊断的 Agent Loop** —— Agent 自主完成「读 baseline 脚本 / 里程碑 → 实现 → 跑任务 → 读日志 → 定位根因 → 改代码」的完整闭环，**调试过程无需人类介入**。
- **端到端生产验证** —— 不是 demo，**真训了模型**：在 64× H100 上完成 MiniCPM4-0.5B 预训练并产出模型权重；checkpoint round-trip / 异步保存 / cursor-resume 均有单元测试覆盖。以 decay 阶段为例，最终 loss 差 0.001（见下方训练曲线）。

  <p align="center"><img src="./asset/decay_phase_training_curve.png" alt="Decay-phase training curve" width="720"></p>

### 🚀 性能超越 Megatron-LM

- **MFU 44.13%，相对 Megatron-LM baseline（MFU ~40%）提升约 10%**（测试条件：64× H100，未启用 MTP）。
- **自研 GEMM 与 FlashAttention** —— GEMM 完全自研并已全量接入主路径，性能优于 cuBLAS；FlashAttention 完全自研，性能超过 Transformer Engine / FA3，对标 FA4。
- **自主探索全部优化空间** —— Agent 在 auto-loop 中自行枚举并实测了 CuTeDSL / cuBLAS / Triton / TransformerEngine 等多条算子路径，以及 wgrad-overlap / sharded-optimizer / step-graph / accum-stream 等数十种通信与图捕获组合，每条都跑分布式任务实测 MFU 与 loss 对齐，**最终的生产默认值是 Agent 从全部组合中自行挑出的最优解**。

---

## 🤖 面向 Agent 的一键部署

> 本仓库本身就是 AI Agent 产出的，对 AI Agent 也最友好。**把下面的 prompt 直接粘贴给 Cursor / Claude Code / Codex / Cline**，它会自己读完 README、安装依赖、跑通冒烟测试并报告 MFU，无需你逐条命令操作。

### 🟢 5-step 最小预训练 demo（最快验证安装）

```text
请按 https://raw.githubusercontent.com/<org>/training_engine/main/README_zh.md
的指引，在当前节点跑一次 5-step 最小预训练 demo：

1. 检查环境（Python ≥ 3.11、CUDA ≥ 12.x、H100、PyTorch ≥ 2.4），缺则装；
2. 安装本仓库：pip install -e . 并安装 HF 依赖：pip install datasets transformers；
3. 导入冒烟：
   PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
4. 跑 5 step HF GSM8K demo：
   torchrun --standalone --nproc-per-node=1 \
     -m training_engine_tensor pretrain \
     --num-steps 5 --global-batch-size 1 --micro-batch-size 1 \
     --seq-length 4096 \
     --hf-dataset openai/gsm8k --hf-dataset-config main \
     --hf-text-template "Question: {question}\nAnswer: {answer}" \
     --tokenizer-path openbmb/MiniCPM4-0.5B \
     --save-dir ./checkpoints/demo
5. 输出最终 loss、step time、MFU。

如有步骤失败，自主排查源码，不要询问我。
```

> 单节点 8× H100 与多节点训练的完整命令见下文「快速开始」。

---

## 目录结构

```
train_engine_0.5B/
  src/
    training_engine_tensor/        # 框架核心（Python + CUDA + Triton）
      __main__.py                  # CLI 入口（pretrain / train 子命令）
      train_loop.py                # 主训练循环
      forward.py / backward.py     # 前向 / 反向
      forward_graph.py             # 前向 CUDA Graph 捕获
      step_graph*.py               # 训练步 CUDA Graph（4 种粒度）
      cuda_graph_utils.py          # CUDA Graph 工具
      optimizer.py / parameters.py # Adam + 参数管理
      kernels.py                   # 算子分发桥接
      triton_kernels.py            # Triton 融合算子
      custom_gemm.py               # 自研 GEMM 路径
      data.py / hf_dataloader.py   # 数据加载（HuggingFace datasets）
      nccl.py                      # NCCL 通信
      profiling.py                 # 性能采集
      engine_config.py             # 配置中枢（EngineConfig frozen dataclass）
      config.py / op_dispatcher.py # model_spec 加载 + 算子版本分发
      ENV_WHITELIST.md             # 环境变量白名单（进程外契约）
      ops/                         # 自研算子子包
        attention/                 # 自研 FlashAttention
          fa4_cute/                # 基于 FA4 CuTeDSL 的 fwd / bwd (SM90)
          flash_attn_dsl/          # CuTeDSL 端到端 fwd / bwd
          kernel.py / register.toml / env_vars.toml
        gemm_qkv_proj/ ... gemm_output/   # 5 个 CuTeDSL 自定义 GEMM
        _gemm_inhouse_jit.py / _gemm_inhouse_kernel.py
    quack/                         # CuTeDSL 工具库（attention 依赖）

  tests/                           # 单元测试

  scripts/                         # 训练入口与工具
    entry_hf_pretrain.sh           # HuggingFace dataset 训练入口
    entry_train.sh / entry_precompile.sh
    precompile_ops.py              # 算子预编译
    bench_attention_bwd.py         # attention bwd 基准
    convert_to_hf.py               # ckpt → HuggingFace 格式

  job_specs/smoke/                 # 分布式训练 job spec 示例
    hf_gsm8k_8gpu_p1.json          # 8× H100 + HuggingFace GSM8K demo
    stable_16gpu_gbs320_p1.json    # 16× H100 稳定预训练

  docs/environment.md              # 完整环境与依赖清单
  model_spec.toml                  # 模型架构规格（由 config.py 读取）
  pyproject.toml / ruff.toml       # 包定义 + lint 配置
  remote-run.sh                    # 远程调度入口
```

---

## 🚀 快速开始

> 完整环境与依赖清单见 [`docs/environment.md`](./docs/environment.md)。简要版：`Python ≥ 3.11` · `CUDA ≥ 12.x` · `PyTorch ≥ 2.4` · `H100 (SM90)`。

### 1. 安装

```bash
git clone https://github.com/<org>/training_engine.git
cd training_engine
pip install -e .

# HuggingFace 数据通路（必装）
pip install datasets transformers
```

### 2. 安装验证

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

### 3. 算子预编译（首次运行；后续复用 cache）

```bash
PYTHONPATH=src CUSTOM_GEMM=1 OP_ATTENTION=v1 \
    python scripts/precompile_ops.py
```

预热 5 个 CuTeDSL GEMM 的 AOT export + cpp_extension 编译，产物落盘到 `${ENGINE_ROOT}/.persist_cache/`；后续训练任务复用 cache，dlopen 仅需几秒。

### 4. 单节点训练（8× H100，使用 HuggingFace 数据集）

```bash
torchrun --standalone --nproc-per-node=8 \
    -m training_engine_tensor pretrain \
    --num-steps 200 \
    --global-batch-size 1280 --micro-batch-size 10 \
    --seq-length 4096 \
    --hf-dataset <YOUR_HF_DATASET> \
    --tokenizer-path <YOUR_TOKENIZER> \
    --save-dir ./checkpoints/run1
```

`--hf-dataset` 接 **HuggingFace Hub 名**（如 `openai/gsm8k`）或**本地数据集目录**（Parquet / Arrow / JSON / JSONL）。文本字段两种声明方式（二选一）：

- `--hf-text-field <COLUMN>` —— 单列直接取文本（如 `text`）
- `--hf-text-template "..."` —— 多列拼接，Python `.format` 语法（如 `"Question: {question}\nAnswer: {answer}"`）

### 5. 多节点训练（任意调度器）

每个节点跑同一条命令，仅 `NODE_RANK` / `MASTER_ADDR` 不同。`torchrun` / Kubernetes PyTorchJob / Slurm / Ray 任一分布式调度器都可：

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc-per-node=8 \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    -m training_engine_tensor pretrain \
    --num-steps 100000 \
    --global-batch-size 1280 --micro-batch-size 10 --seq-length 4096 \
    --hf-dataset <YOUR_HF_DATASET> \
    --tokenizer-path <YOUR_TOKENIZER> \
    --save-interval 5000 \
    --save-dir <SHARED_FS_DIR>/run1
```

### 6. 断点续训

```bash
training_engine_tensor train \
    --resume <SHARED_FS_DIR>/run1/step_NNNNNNN \
    --num-steps 200000 \
    --hf-dataset <YOUR_HF_DATASET> \
    --tokenizer-path <YOUR_TOKENIZER> \
    --save-dir <SHARED_FS_DIR>/run1
```

单个 `training_state.pt` 文件打包 BF16 模型 + FP32 优化器 + dataloader cursor，由 rank 0 异步落盘。

### 7. 导出为 HuggingFace 格式

```bash
python scripts/convert_to_hf.py \
    --input <SHARED_FS_DIR>/run1/step_NNNNNNN \
    --output ./hf_model
```

> 可直接 fork 改用的 PyTorchJob spec 示例：[`job_specs/smoke/hf_gsm8k_8gpu_p1.json`](./job_specs/smoke/hf_gsm8k_8gpu_p1.json)（HuggingFace GSM8K demo，8 卡 H100，50 step）。

---

## 📦 支持的模型与硬件

### 适配模型


| 模型                | 参数量   | 架构                                               | HuggingFace                             | ModelScope                          |
| ----------------- | ----- | ------------------------------------------------ | --------------------------------------- | ----------------------------------- |
| **MiniCPM4-0.5B** | 0.5 B | 24 层 Transformer · GQA (16Q/2KV) · SwiGLU · RoPE | [🤗 link](https://huggingface.co/openbmb/MiniCPM4-ForgeTrain-0.5B) | [link](https://modelscope.cn/<org>) |


### 硬件要求


| GPU                    | 架构            | 状态             |
| ---------------------- | ------------- | -------------- |
| **NVIDIA H100 (80GB)** | SM90 (Hopper) | ✅ 已适配并完成生产环境验证 |


---

## 算子组合

生产默认配置启用全部 6 个自研算子加上 fused RMSNorm backward：


| 环境变量                    | 生产默认 | 说明                             |
| ----------------------- | ---- | ------------------------------ |
| `CUSTOM_GEMM`           | `1`  | 启用自研 GEMM 路径分发                 |
| `OP_ATTENTION`          | `v1` | FA4 CuTeDSL forward + backward |
| `OP_GEMM_QKV_PROJ`      | `v1` | AOT C-export QKV GEMM          |
| `OP_GEMM_ATTN_OUT_PROJ` | `v1` | AOT C-export attn out GEMM     |
| `OP_GEMM_FC1`           | `v1` | AOT C-export FC1               |
| `OP_GEMM_FC2`           | `v1` | AOT C-export FC2               |
| `OP_GEMM_OUTPUT`        | `v1` | AOT C-export output / LM head  |
| `USE_FUSED_NORM_BWD`    | `1`  | fused Triton RMSNorm backward  |


---

## 命令行用法

```
python -m training_engine_tensor {pretrain,train,checkpoint-train-loop} ...
```

- `pretrain` —— 从零开始训练
- `train` —— 从 checkpoint 续训，支持 `--resume <dir>`
- `checkpoint-train-loop` —— 微基准模式，加载固定权重 + tokenized inputs 跑指定 step indices

环境变量配置详见 `src/training_engine_tensor/ENV_WHITELIST.md`，所有非白名单变量必须通过 `EngineConfig` 显式传入。

