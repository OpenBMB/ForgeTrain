# ForgeTrain Engine

### 全程由 AI Agent Loop 自动构建的 LLM 预训练框架

**🤖 100% AI 自动编写 · 🚀 H100 上 MFU 50.9% · 📈 较 Megatron-LM 提升约 8% · 🧪 8× H100 单机已跑通**

> [ForgeTrain](../../README.md) monorepo 的子项目。

[English](./README.md) | [中文](./README_zh.md)

[License](../../LICENSE)
[Python](https://www.python.org)
[CUDA](https://developer.nvidia.com/cuda-toolkit)
[PyTorch](https://pytorch.org)
[GPU](https://www.nvidia.com/en-us/data-center/h100/)
[MFU](#-性能超越-megatron-lm)

[HuggingFace](https://huggingface.co/openbmb/MiniCPM4-8B)
[ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B)

⭐ 如果觉得有用，欢迎 Star · 🤝 站在 CUTLASS、FlashAttention、TransformerEngine 等开源工作的肩膀上

---

**TrainingEngine (8B)** 是一套单机版的 LLM 预训练框架，专为 8× H100 SXM5（SM90a）打造。**框架代码（Python + CuTeDSL，未计入基于 NVIDIA CuTeDSL 上游移植的 helper 部分）由一个 AI Agent Loop 端到端自动撰写、调试与优化，全程无人为代码改动**。框架以 MiniCPM4-8B 为目标负载，在 8× H100 上以 `**tensor_model_parallel_size = 2` / `data_parallel_size = 4`** 的拓扑跑通了端到端训练循环，**在 GAS=8 的节拍下相对 Megatron-LM baseline 实现约 8% 的 MFU 提升**；未来会尝试扩展节点。数据加载走 HuggingFace `datasets`，任意 Hub 数据集或本地 Parquet / Arrow / JSON 都可一行 CLI 接入。

---

## ✨ 亮点

### 🤖 全栈 AI 自动产出

- **100% Agent Loop 自动产出** —— 框架代码（Python + CuTeDSL）、压力测试、算子包装层**全部由 AI Agent 在 auto-loop 模式下自主产出**。人类角色仅提供训练目标与硬件资源，**无任何手动改码、手动调参、手动修 bug**。
- **可自诊断的 Agent Loop** —— Agent 在 harness 驱动下自主完成「读 baseline 脚本 / 里程碑 → 实现 → 跑任务 → 读日志 → 定位根因 → 改代码」的完整闭环，**调试过程无需人类介入**。

### 🚀 性能超越 Megatron-LM

- **MFU 50.9%，相对 Megatron-LM baseline（MFU ~47%）提升约 8%**（`micro_batch_size = 2`、`grad_accum_steps = 8`、`seq_length = 4096`，8× H100 上 TP=2 / DP=4）。
- **自研 CuTeDSL 算子** —— 针对调用最热的三个位点手写 SM90a kernel：`gemm_fc1`（SwiGLU 列并行 GEMM）、`gemm_output`（LM-head 列并行 GEMM）、以及 flash-attention 前向 DSL（`flash_attn_dsl`）。三个算子覆盖了内层循环的瓶颈，其余位点继续走 baseline `torch.matmul`，未来会尝试接入剩余 GEMM 和 flash-attention 后向 DSL 。
- **自主探索全部优化空间** —— Agent 在 auto-loop 中自行枚举并实测了 CuTeDSL / cuBLAS / SDPA 等多条算子路径，以及 kernel template 的逐 shape 参数组合（stage 数、swizzle、cluster mode、epilogue overlap），**最终的生产默认值是 Agent 从全部组合中自行挑出的最优解**。

---

## 🤖 面向 Agent 的一键部署

> 本仓库本身就是 AI Agent 产出的，对 AI Agent 也最友好。**把下面的 prompt 直接粘贴给 Cursor / Claude Code / Codex / Cline**，它会自己读完 README、安装依赖、跑通冒烟测试并报告 MFU，无需你逐条命令操作。

### 🟢 5-step 最小预训练 demo（最快验证安装）

```text
请按 README_zh.md
的指引，在当前 8× H100 节点跑一次 5-step 最小预训练 demo：

1. 检查环境（Python ≥ 3.11、CUDA ≥ 12.x、H100 SXM5、PyTorch ≥ 2.4），缺则装；
2. 安装本仓库：pip install -e . 并安装 HF 依赖：
   pip install datasets transformers sentencepiece；
3. 导入冒烟：
   PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
4. 跑 5 step HF FineWeb demo（FineWeb 是开放可下载的英文 webtext
   预训练语料，`sample-10BT` 子集约 10 B token，适合 8B 预训练的真实
   数据分布）：
   export CUSTOM_GEMM=1 OP_ATTENTION=v1
   torchrun --standalone --nproc-per-node=8 \
     -m training_engine_tensor pretrain \
     --num-steps 5 --global-batch-size 64 \
     --micro-batch-size 2 --grad-accum-steps 8 \
     --seq-length 4096 \
     --hf-dataset HuggingFaceFW/fineweb \
     --hf-dataset-config sample-10BT \
     --hf-text-field text \
     --tokenizer-path openbmb/MiniCPM4-8B
5. 输出最终 loss、step time、MFU。

如有步骤失败，自主排查源码，不要询问我。
```

---

## 目录结构

```
train_engine/
  src/
    training_engine_tensor/        # 框架核心（Python + CuTeDSL）
      __main__.py                  # CLI 入口（pretrain 子命令）
      entry.py                     # 训练主循环驱动
      forward.py / backward.py     # 前向 / 反向
      optimizer.py / parameters.py # Adam + 参数管理
      nccl.py                      # NCCL 通信
      kernels.py / custom_gemm.py  # 算子分发桥接
      engine_config.py             # 配置中枢（EngineConfig frozen dataclass）
      op_dispatcher.py             # 算子版本分发
      ENV_WHITELIST.md             # 环境变量白名单（进程外契约）
      ops/                         # 自研算子子包
        gemm_fc1/                  # SwiGLU 列并行 GEMM（CuTeDSL v1）
          kernel.py / _cute_kernel.py / register.toml
        gemm_output/               # LM-head 列并行 GEMM（CuTeDSL v1）
          kernel.py / register.toml
        gemm_qkv_proj/             # 仅 baseline（register.toml stub）
        gemm_attn_out_proj/        # 仅 baseline
        gemm_fc2/                  # 仅 baseline
    flash_attn_dsl/                # SM90a flash-attention 前向 DSL kernel
    quack/                         # flash_attn_dsl 依赖的 CuTeDSL 工具库

  tests/                           # 算子压力测试（裸机 pytest）

  scripts/                         # 训练入口与工具
    entry_hf_pretrain.sh           # HuggingFace dataset 训练入口

  model_spec.toml                  # 模型 & 训练参数（L1 SSOT）
  pyproject.toml                   # 包定义
```

---

## 🚀 快速开始

> 简要环境：`Python ≥ 3.11` · `CUDA ≥ 12.x` · `PyTorch ≥ 2.4` · `H100 SXM5 (SM90a)` · `nvidia-cutlass-dsl ≥ 4.4`。
> 完整环境清单（硬件 / 驱动 / 依赖 wheel / 环境变量白名单）见
> [`docs/environment.md`](./docs/environment.md)。

### 1. 安装

```bash
git clone https://github.com/OpenBMB/ForgeTrain.git
cd ForgeTrain/exports/train_engine_8b
pip install -e .

# HuggingFace 数据通路（必装）
pip install datasets transformers sentencepiece
```

### 2. 安装验证

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

### 3. 运行算子压力测试（首次训练前推荐执行一遍）

```bash
pytest tests/ -v
```

`tests/` 下的三个冒烟测试会在「并发邻居负载」下跑生产 GEMM / attention
kernel（cuBLAS GEMM、HBM memcpy、TMA-heavy GEMM、Hopper cluster-mode GEMM、
跨 stream event 抖动、allocator churn）。单测默认 ~30 s；长跑曲线通过
`STRESS_DURATION_S` 拉长即可。

### 4. 单节点训练（8× H100，使用 HuggingFace 数据集）

推荐的开放预训练语料是 [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
（`sample-10BT` 子集约 10 B token 的 web text）。

```bash
export HF_DATASET=HuggingFaceFW/fineweb
export HF_DATASET_CONFIG=sample-10BT
export HF_TEXT_FIELD=text
export TOKENIZER_PATH=<YOUR_TOKENIZER>

# 可选：启用自研 CuTeDSL 算子（gemm_fc1 + gemm_output + FA fwd）。
export CUSTOM_GEMM=1
export OP_ATTENTION=v1

scripts/entry_hf_pretrain.sh
```

`HF_DATASET` 接 **HuggingFace Hub 名**（如 `HuggingFaceFW/fineweb`）
或**本地数据集目录**（Parquet / Arrow / JSON / JSONL）。文本字段两种
声明方式（二选一）：

- `HF_TEXT_FIELD=<COLUMN>` —— 单列直接取文本（如 fineweb 的 `text`）
- `HF_TEXT_TEMPLATE="..."` —— 多列拼接，Python `.format` 语法（如
`"{title}\n\n{content}"`，适合需要把多列拼成单段文本的数据集）

如需更细粒度控制，可直接调用 CLI：

```bash
PYTHONPATH=src \
torchrun --standalone --nproc-per-node=8 \
    -m training_engine_tensor pretrain \
    --num-steps 200 \
    --global-batch-size 64 \
    --micro-batch-size 2 --grad-accum-steps 8 \
    --seq-length 4096 \
    --hf-dataset HuggingFaceFW/fineweb \
    --hf-dataset-config sample-10BT \
    --hf-text-field text \
    --tokenizer-path <YOUR_TOKENIZER>
```

`python -m training_engine_tensor pretrain --help` 可看到完整 flag 列表；  
所有 `EngineConfig` 字段都会自动镜像为对应 CLI flag。

---

## 📦 支持的模型与硬件

### 适配模型


| 模型              | 参数量 | 架构                                               | HuggingFace                                           | ModelScope                                               |
| --------------- | --- | ------------------------------------------------ | ----------------------------------------------------- | -------------------------------------------------------- |
| **MiniCPM4-8B** | 8 B | 32 层 Transformer · GQA (32Q/2KV) · SwiGLU · RoPE | [🤗 link](https://huggingface.co/openbmb/MiniCPM4-8B) | [link](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B) |


架构常量（L1 SSOT 在 `model_spec.toml`）：
`hidden_size = 4096`、`num_layers = 32`、`num_attention_heads = 32`、
`num_query_groups = 2`、`head_dim = 128`、`ffn_hidden_size = 16384`、
`seq_length = 4096`、`vocab_size = 73448`。

### 硬件要求


| GPU                          | 架构             | 状态                                     |
| ---------------------------- | -------------- | -------------------------------------- |
| **NVIDIA H100 SXM5 (80 GB)** | SM90a (Hopper) | ✅ 8× H100 单机已跑通端到端训练 + MFU 验证（≠ 完整预训练） |


---

## 算子组合

框架包含 **三个自研 CuTeDSL kernel**；其余 GEMM 位点无条件走 baseline
`torch.matmul`，保证 `CUSTOM_GEMM=0` 时仍能正确跑：


| 算子                   | 默认版本           | 生产来源                                          | 替代                        |
| -------------------- | -------------- | --------------------------------------------- | ------------------------- |
| `gemm_fc1`           | `v1` (CuTeDSL) | `src/training_engine_tensor/ops/gemm_fc1/`    | SwiGLU 列并行 GEMM           |
| `gemm_output`        | `v1` (CuTeDSL) | `src/training_engine_tensor/ops/gemm_output/` | LM-head 列并行 GEMM          |
| flash-attention fwd  | `v1` (DSL)     | `src/flash_attn_dsl/`                         | SM90a flash-attention fwd |
| `gemm_qkv_proj`      | `baseline`     | `torch.matmul`                                | —                         |
| `gemm_attn_out_proj` | `baseline`     | `torch.matmul`                                | —                         |
| `gemm_fc2`           | `baseline`     | `torch.matmul`                                | —                         |


分发器（`op_dispatcher.get_op_version`）在 import 时读取每个算子目录下的
`register.toml`，再按 env 变量解析当前激活版本：


| 环境变量                    | 生产默认            | 说明                                                 |
| ----------------------- | --------------- | -------------------------------------------------- |
| `CUSTOM_GEMM`           | `1`             | 总开关 —— 同时强制 `OP_GEMM_FC1=v1` + `OP_GEMM_OUTPUT=v1` |
| `OP_ATTENTION`          | `v1`            | 启用 SM90a flash-attn DSL forward                    |
| `OP_GEMM_FC1`           | `v1`            | `gemm_fc1` CuTeDSL kernel                          |
| `OP_GEMM_OUTPUT`        | `v1`            | `gemm_output` CuTeDSL kernel                       |
| `OP_GEMM_FC2`           | `baseline` (固定) | `torch.matmul` 兜底                                  |
| `OP_GEMM_QKV_PROJ`      | `baseline` (固定) | `torch.matmul` 兜底                                  |
| `OP_GEMM_ATTN_OUT_PROJ` | `baseline` (固定) | `torch.matmul` 兜底                                  |


任意 `OP_<NAME>=baseline` 都是安全 fallback，即便所有自研算子关闭，
框架依然可以正常启动。

---

## 命令行用法

```
python -m training_engine_tensor pretrain ...
```

- `pretrain` —— HF dataloader 下从随机初始化预训练
（`--checkpoint-root` 可从 Megatron 格式 checkpoint 热启；
 `--save-checkpoint-dir` 在最后一步写入 resume-able 分片）。

环境变量配置详见 `src/training_engine_tensor/ENV_WHITELIST.md`，
所有非白名单变量必须通过 `EngineConfig` 显式传入。

---

## 🛡️ 代码质量

Agent Loop 产出的框架代码遵循一套强约束，仓库内压力测试 + AST guard
把它们落成**可执行**的门禁——CI 红线、端到端训练任务挂掉就是违约。

### 🚨 红线 —— 零容忍


| 红线                         | 一句话                                     | 仓库落地                                                    |
| -------------------------- | --------------------------------------- | ------------------------------------------------------- |
| **SSOT**                   | 每条知识只有一个权威来源                            | `EngineConfig` 收口每个行为开关；`model_spec.toml` L1→L2→L3 单向引用 |
| **DAG**                    | 模块依赖必须无环、下层不依赖上层                        | `engine_config → config → kernels → fwd/bwd → entry` 单向 |
| **Fail Fast**              | 错误立刻 raise，禁止多层 try-catch / 静默 fallback | `ops/gemm_*/kernel.py` 拒绝吞掉 CuTeDSL JIT 异常              |
| **Minimal Public Surface** | 默认私有，每个 public 必须有真实 caller             | 全模块 `__all__`；`ENV_WHITELIST.md` 限定唯一一组对外 env           |
| **TDD**                    | 先写失败测试，再写最小实现让它通过                       | `tests/` 提供算子压力测试，全部 H100 实测                            |


### 开发流程纪律

- **根因修复，禁用规避方案** —— 每次修复打到根因，禁止 band-aid / 对症治疗
- **先设计后实现** —— 改动先设计后实现，无论多简单
- **一次跑完** —— 一次执行内跑完实现 + 测试 + 验证，**不在中途停下问要不要继续**
- **默认不向后兼容** —— 假设接口未公开、快速迭代，不写 migration code

