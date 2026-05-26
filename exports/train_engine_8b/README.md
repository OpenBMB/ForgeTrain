# ForgeTrain Engine

### An LLM Pretraining Framework Built End-to-End by an Autonomous Agent Loop

**🤖 100% AI-Authored · 🚀 50.9% MFU on H100 · 📈 +8% over Megatron-LM · 🧪 8× H100 single-host bring-up validated**

> Subproject of the [ForgeTrain](../../README.md) monorepo.

[English](./README.md) | [中文](./README_zh.md)

[License](../../LICENSE)
[Python](https://www.python.org)
[CUDA](https://developer.nvidia.com/cuda-toolkit)
[PyTorch](https://pytorch.org)
[GPU](https://www.nvidia.com/en-us/data-center/h100/)
[MFU](#-faster-than-megatron-lm)

[HuggingFace](https://huggingface.co/openbmb/MiniCPM4-8B)
[ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B)

⭐ Star this repo if you find it useful · 🤝 Built on the shoulders of CUTLASS, FlashAttention & TransformerEngine

---

**TrainingEngine (8B)** is a single-host LLM pretraining framework targeting eight H100 SXM5 GPUs (SM90a). **The framework code (Python + CuTeDSL, excluding helpers ported from upstream NVIDIA CuTeDSL) was written, debugged, and optimized end-to-end by an AI Agent Loop, with zero manual code edits.** Using MiniCPM4-8B as the target workload, it runs the end-to-end training loop on 8× H100 in a **`tensor_model_parallel_size = 2` / `data_parallel_size = 4`** layout and delivers **a ~8% MFU lift over the Megatron-LM baseline** at GAS=8; multi-node scale-out is future work. Data loading goes through *HuggingFace `datasets`* — any Hub dataset or local Parquet / Arrow / JSON works with a one-line CLI flag.

---

## ✨ Highlights

### 🤖 Fully AI-Authored

- **100% Agent-Loop Authored** — every line of framework code (Python + CuTeDSL), every stress test, and every operator wrapper **was produced by an AI Agent running in auto-loop mode**. Humans only supplied the training objective and the hardware budget; **no manual code edits, no manual hyperparameter tuning, no manual bug fixes**.
- **Self-Diagnosing Agent Loop** — Driven by a harness, the Agent autonomously executes the full loop: read baseline scripts / milestones → implement → launch a job → read logs → locate root cause → patch code — **all without human intervention for debugging**.

### 🚀 Faster than Megatron-LM

- **MFU 50.9%, ~8% above the Megatron-LM baseline (MFU ~47%)** at the production cadence (`micro_batch_size = 2`, `grad_accum_steps = 8`, `seq_length = 4096`, TP=2 / DP=4 on 8× H100).
- **Custom CuTeDSL kernels** — hand-written SM90a kernels for the three hottest call sites: `gemm_fc1` (SwiGLU column-parallel GEMM), `gemm_output` (LM-head column-parallel GEMM), and a flash-attention forward DSL (`flash_attn_dsl`). Together they cover the inner-loop bottleneck and free the remaining call sites to run the baseline `torch.matmul` path unchanged， other GEMM and flash-attention backward dsl are future work.
- **Self-explored optimization space** — In auto-loop the Agent enumerated and benchmarked CuTeDSL / cuBLAS / SDPA operator variants plus per-shape kernel-template parameters (stage count, swizzle, cluster mode, epilogue overlap), measuring both MFU and loss alignment; **the production defaults are the optimum the Agent picked from the full grid**.

---

## 🤖 Agent-Friendly Quick Deploy

> This repo was produced by an AI Agent and is friendliest to AI Agents. **Paste the prompt below into Cursor / Claude Code / Codex / Cline** — it will read the README, install dependencies, run the smoke test and report the MFU, without you typing commands one at a time.

### 🟢 5-step minimal pretraining demo (fastest install check)

```text
Following https://raw.githubusercontent.com/OpenBMB/ForgeTrain/main/exports/train_engine_8b/README.md,
run a 5-step minimal pretraining demo on the current 8× H100 node:

1. Check the environment (Python ≥ 3.11, CUDA ≥ 12.x, H100 SXM5, PyTorch ≥ 2.4)
   and install anything missing;
2. Install the repo: pip install -e . and HF deps:
   pip install datasets transformers sentencepiece;
3. Import smoke test:
   PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
4. Run 5 steps on HF FineWeb (an open, downloadable English webtext
   pretraining corpus; the ``sample-10BT`` subset is ~10 B tokens,
   suitable for the 8B pretraining distribution):
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
5. Print the final loss, step time, and MFU.

If anything fails, dig into the source on your own — do not ask me.
```

---

## Directory Layout

```
train_engine/
  src/
    training_engine_tensor/        # Framework core (Python + CuTeDSL)
      __main__.py                  # CLI entry (pretrain subcommand)
      entry.py                     # Training loop driver
      forward.py / backward.py     # Forward / backward
      optimizer.py / parameters.py # Adam + parameter management
      nccl.py                      # NCCL collectives
      kernels.py / custom_gemm.py  # Operator dispatch shims
      engine_config.py             # Config hub (EngineConfig frozen dataclass)
      op_dispatcher.py             # Per-operator version dispatch
      ENV_WHITELIST.md             # Process-external env var whitelist
      ops/                         # Custom operator subpackage
        gemm_fc1/                  # SwiGLU column-parallel GEMM (CuTeDSL v1)
          kernel.py / _cute_kernel.py / register.toml
        gemm_output/               # LM-head column-parallel GEMM (CuTeDSL v1)
          kernel.py / register.toml
        gemm_qkv_proj/             # baseline-only (register.toml stub)
        gemm_attn_out_proj/        # baseline-only
        gemm_fc2/                  # baseline-only
    flash_attn_dsl/                # SM90a flash-attention fwd DSL kernel
    quack/                         # CuTeDSL helpers used by flash_attn_dsl

  tests/                           # Operator stress tests (bare-metal pytest)

  scripts/                         # Training entries & tools
    entry_hf_pretrain.sh           # HuggingFace dataset training entry

  model_spec.toml                  # Model & training spec (L1 SSOT)
  pyproject.toml                   # Package definition
```

---

## 🚀 Quick Start

> Short form: `Python ≥ 3.11` · `CUDA ≥ 12.x` · `PyTorch ≥ 2.4` · `H100 SXM5 (SM90a)` · `nvidia-cutlass-dsl ≥ 4.4`.
> Full environment manifest (hardware / driver / dep wheels / env var
> whitelist) lives in [`docs/environment.md`](./docs/environment.md).

### 1. Install

```bash
git clone https://github.com/OpenBMB/ForgeTrain.git
cd ForgeTrain/exports/train_engine_8b
pip install -e .

# HuggingFace data path (required)
pip install datasets transformers sentencepiece
```

### 2. Verify install

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

### 3. Run the operator stress tests (recommended before the first training job)

```bash
pytest tests/ -v
```

The three smoke tests under `tests/` exercise the production GEMM /
attention kernels under concurrent neighbour workloads (cuBLAS GEMMs,
HBM memcpy, TMA-heavy GEMMs, Hopper cluster-mode GEMMs, cross-stream
event chaos, alloc churn). Each takes ~30 s by default; raise
`STRESS_DURATION_S` for the long-run profile.

### 4. Single-node training (8× H100, with a HuggingFace dataset)

The recommended open pretraining corpus is
[`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
(its `sample-10BT` config is a ~10 B-token webtext subset).

```bash
export HF_DATASET=HuggingFaceFW/fineweb
export HF_DATASET_CONFIG=sample-10BT
export HF_TEXT_FIELD=text
export TOKENIZER_PATH=<YOUR_TOKENIZER>

# Optional: enable the engine's CuTeDSL kernels (gemm_fc1 + gemm_output + FA fwd).
export CUSTOM_GEMM=1
export OP_ATTENTION=v1

scripts/entry_hf_pretrain.sh
```

`HF_DATASET` accepts either a **HuggingFace Hub name** (e.g.
`HuggingFaceFW/fineweb`) or a **local dataset directory** (Parquet /
Arrow / JSON / JSONL). Pick one of two ways to declare the text field:

- `HF_TEXT_FIELD=<COLUMN>` — take a single column directly (e.g. fineweb's `text`)
- `HF_TEXT_TEMPLATE="..."` — concatenate multiple columns with Python `.format` syntax (e.g. `"{title}\n\n{content}"`, suitable for datasets whose text is split across multiple columns)

For finer control, invoke the CLI directly:

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

Run `python -m training_engine_tensor pretrain --help` to see the full
flag surface; every `EngineConfig` field is auto-mirrored as a CLI flag.

---

## 📦 Models & Versions

### Supported models

| Model           | Params | Architecture                                            | HuggingFace                             | ModelScope                          |
| --------------- | ------ | ------------------------------------------------------- | --------------------------------------- | ----------------------------------- |
| **MiniCPM4-8B** | 8 B    | 32-layer Transformer · GQA (32Q/2KV) · SwiGLU · RoPE    | [🤗 link](https://huggingface.co/openbmb/MiniCPM4-8B) | [link](https://modelscope.cn/models/OpenBMB/MiniCPM4-8B) |

Architectural constants (L1 SSOT in `model_spec.toml`):
`hidden_size = 4096`, `num_layers = 32`, `num_attention_heads = 32`,
`num_query_groups = 2`, `head_dim = 128`, `ffn_hidden_size = 16384`,
`seq_length = 4096`, `vocab_size = 73448`.

### Hardware

| GPU                          | Arch           | Status                                                  |
| ---------------------------- | -------------- | ------------------------------------------------------- |
| **NVIDIA H100 SXM5 (80 GB)** | SM90a (Hopper) | ✅ 8× H100 single-host bring-up + MFU validated (≠ full pretrain) |

---

## Operator Stack

The engine ships **three self-developed CuTeDSL kernels**; the remaining
GEMM call sites run the baseline `torch.matmul` path unconditionally so
the engine stays correct even when `CUSTOM_GEMM=0`:

| Op                      | Default       | Production source                                      | Replaces                              |
| ----------------------- | ------------- | ------------------------------------------------------ | ------------------------------------- |
| `gemm_fc1`              | `v1` (CuTeDSL)| `src/training_engine_tensor/ops/gemm_fc1/`             | SwiGLU column-parallel GEMM           |
| `gemm_output`           | `v1` (CuTeDSL)| `src/training_engine_tensor/ops/gemm_output/`          | LM-head column-parallel GEMM          |
| flash-attention fwd     | `v1` (DSL)    | `src/flash_attn_dsl/`                                  | SM90a flash-attention fwd             |
| `gemm_qkv_proj`         | `baseline`    | `torch.matmul`                                         | —                                     |
| `gemm_attn_out_proj`    | `baseline`    | `torch.matmul`                                         | —                                     |
| `gemm_fc2`              | `baseline`    | `torch.matmul`                                         | —                                     |

The dispatcher (`op_dispatcher.get_op_version`) reads each operator's
`register.toml` at import time and resolves the active version from the
declared env var:

| Env var             | Production default | Description                              |
| ------------------- | ------------------ | ---------------------------------------- |
| `CUSTOM_GEMM`       | `1`                | Master switch — also forces `OP_GEMM_FC1=v1` + `OP_GEMM_OUTPUT=v1` |
| `OP_ATTENTION`      | `v1`               | Enables the SM90a flash-attn DSL forward |
| `OP_GEMM_FC1`       | `v1`               | `gemm_fc1` CuTeDSL kernel                |
| `OP_GEMM_OUTPUT`    | `v1`               | `gemm_output` CuTeDSL kernel             |
| `OP_GEMM_FC2`       | `baseline` (fixed) | `torch.matmul` fallback                  |
| `OP_GEMM_QKV_PROJ`  | `baseline` (fixed) | `torch.matmul` fallback                  |
| `OP_GEMM_ATTN_OUT_PROJ` | `baseline` (fixed) | `torch.matmul` fallback              |

Setting any `OP_<NAME>=baseline` is always a safe fall-back to the eager
PyTorch path; the engine boots correctly even with every custom op
disabled.

---

## CLI

```
python -m training_engine_tensor pretrain ...
```

- `pretrain` — random-init pretraining with the HF dataloader
  (`--checkpoint-root` warm-starts from a Megatron-format checkpoint;
  `--save-checkpoint-dir` writes a resume-able shard set after the
  final step).

For env-var configuration see `src/training_engine_tensor/ENV_WHITELIST.md`.
Anything not on the whitelist must flow through `EngineConfig`.

---

## 🛡️ Code Quality

The framework code produced by the Agent Loop follows a strict contract.
The repo's stress tests + AST guard turn it into an **executable** gate
— a red CI run or a failing end-to-end training job counts as a violation.

### 🚨 RED LINE — Zero Tolerance

| Red line                   | One-liner                                                              | Repo evidence                                                                  |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **SSOT**                   | Every fact has exactly one authoritative source                        | `EngineConfig` owns every behavioural knob; `model_spec.toml` L1→L2→L3 one-way |
| **DAG**                    | Module deps are acyclic; lower layers never depend on upper layers     | `engine_config → config → kernels → fwd/bwd → entry` is one-way                |
| **Fail Fast**              | Errors raise immediately; no layered try/catch or silent fallback      | `ops/gemm_*/kernel.py` refuse to swallow CuTeDSL JIT errors                    |
| **Minimal Public Surface** | Hidden by default; every public must justify itself with a real caller | All modules declare `__all__`; `ENV_WHITELIST.md` is the only outward env contract |
| **TDD**                    | Failing test first, then minimum code to pass                          | `tests/` ships operator stress tests gated on real H100 hardware               |

### Development process

- **Root Cause, No Workaround** — every fix targets the root cause; no band-aids or symptomatic patches
- **Design Before Implementation** — design first, then implement, no matter how trivial the change
- **Development Completion** — finish implementation + verification + tests in a single execution pass; **do not stop mid-flight to ask whether to continue**
- **No Backward Compatibility by Default** — assume unpublished, fast-iteration; no migration code
