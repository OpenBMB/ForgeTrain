<p align="center">
  <img src="./assets/logo.png" alt="ForgeTrain" width="90%">
</p>

# ForgeTrain

### An LLM Pretraining Framework Built End-to-End by an Autonomous Agent Loop + a matching Harness scaffolding *(coming soon)*

**🤖 100% AI-Authored · 🚀 44.13% MFU on H100 · 📈 +10% over Megatron-LM · ✅ Production-Validated · 📄 [Paper](./assets/forgetrain_paper.pdf)**

[English](./README.md) | [中文](./README_zh.md)

<p align="center">
  <a href="./assets/forgetrain_paper.pdf"><img src="https://img.shields.io/badge/Paper-ForgeTrain%20Technical%20Report-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="ForgeTrain Paper (PDF)"></a>
</p>

---

> **An LLM pretraining framework written end-to-end by an AI Agent Loop with zero human edits — plus the Harness that produced the pretraining framework *(coming soon)*.**
>
> **Current release: v0.1.0** (NVIDIA H100 · MiniCPM4-0.5B / MiniCPM4-8B training frameworks; matching Harness *coming soon*)

---

## ✨ Highlights

- 🤖 **100% Agent-Loop Authored** — the entire framework produced by an AI Agent running in auto-loop mode, with zero manual edits
- 🔄 **Self-Diagnosing Agent Loop** — read reference → implement → launch job → parse logs → root-cause → patch → pass gate → commit, fully autonomous
- 🚀 **44.13% MFU on H100** — ~10% above the Megatron-LM baseline (~40%), validated on 64× H100 with BF16, DP-only
- ✅ **Production-Validated** — MiniCPM4-0.5B fully pretrained, real model weights produced (not a demo)
- 🛠️ **GEMM + Attention kernels authored by the agent loop** — **per-op MFU up to 90%**; FlashAttention written from scratch, outperforms Transformer Engine / FA3, on par with FA4

---

## 🗺️ Roadmap

- **Reproduction live demo**
- **Huawei MiniCPM5-1B training framework**
- **Training framework self-generates the Harness scaffolding**

---

## Feature Comparison

| Feature | **ForgeTrain** | Megatron-LM |
|---------|:-:|:-:|
| MFU on H100 (MiniCPM4-0.5B, BF16, DP) | **44.13%** | ~40% |
| 100% AI-Authored Code | ✅ | ❌ |
| CuTeDSL custom GEMMs (AOT C-export) | ✅ (5 GEMMs) | ❌ |
| Custom FlashAttention (on par with FA4) | ✅ (self-built CuTeDSL impl) | ❌ (uses upstream TE / FA) |
| Checkpoint → HuggingFace export | ✅ (one script) | Manual |

<sub>Also supports CUDA Graph, Triton fused kernels, and comm-compute overlap out of the box.</sub>

> Comparison based on Megatron-LM v0.15 on the same hardware (H100, SM90). ForgeTrain v1 is scoped to MiniCPM4-0.5B (DP-only) and MiniCPM4-8B (TP=2) × BF16; Megatron-LM supports broader model families and parallelism strategies.

---

## 📢 News

- 🎉 **[2026-08] Paper released!** — our technical report *“ForgeTrain: Forging Production-Grade Training Frameworks via Harness-Driven AI Development”* is now available — **[Read the PDF](./assets/forgetrain_paper.pdf)**
- 📌 **[2026-05] ForgeTrain v0.1.0 released** — first public release of the training engine; the Harness that produced it is *coming soon*. MiniCPM4-0.5B pretrained on 64× H100, achieving **44.13% MFU**.

---

## Table of Contents

- [Highlights](#-highlights)
- [Roadmap](#-roadmap)
- [Feature Comparison](#feature-comparison)
- [News](#-news)
- [Agent-Friendly Quick Deploy](#-agent-friendly-quick-deploy)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Core Technology](#core-technology)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## 🤖 Agent-Friendly Quick Deploy

> This repo was produced by an AI Agent and is friendliest to AI Agents. **Paste the prompt below into Cursor / Claude Code / Codex / Cline** — it will read the README, install dependencies, run the smoke test and report the MFU, without you typing commands one at a time.

**🟢 5-step minimal pretraining demo (paste into your Coding Agent)**

```text
Following this project's exports/train_engine_0.5B/README.md,
run a 5-step minimal pretraining demo on the current node:

1. Check the environment (Python ≥ 3.11, CUDA ≥ 12.x, H100, PyTorch ≥ 2.4)
   and install anything missing;
2. Install the repo: pip install -e . and HF deps: pip install datasets transformers;
3. Import smoke test:
   PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
4. Run 5 steps on HF GSM8K:
   torchrun --standalone --nproc-per-node=1 \
     -m training_engine_tensor pretrain \
     --num-steps 5 --global-batch-size 1 --micro-batch-size 1 \
     --seq-length 4096 \
     --hf-dataset openai/gsm8k --hf-dataset-config main \
     --hf-text-template "Question: {question}\nAnswer: {answer}" \
     --tokenizer-path openbmb/MiniCPM4-0.5B \
     --save-dir ./checkpoints/demo
5. Print the final loss, step time, and MFU.

If anything fails, dig into the source on your own — do not ask me.
```

> Full single-node 8× H100 and multi-node commands are in the [Quick Start](#quick-start) section below.

---

## Repository Layout

This repo bundles a family of subprojects in a strict **producer / product** relationship:

| Subdirectory | Role |
|---|---|
| `harness/` *(coming soon)* | **Harness** — the scaffolding that drives an Agent Loop to autonomously build a training framework |
| `exports/train_engine_0.5B/` | **TrainingEngine (0.5B)** — produced end-to-end by `harness/` *(coming soon)*; targets MiniCPM4-0.5B at 44.13% MFU on 8× H100 |
| `exports/train_engine_8b/`   | **TrainingEngine (8B)** — also produced by `harness/` *(coming soon)*; targets MiniCPM4-8B with TP=2 / DP=4 at 50.9% MFU on a single 8× H100 host |

```
 harness/  ──(bash agent-loop.sh, zero human input)──▶  exports/train_engine_0.5B/
  Harness  (coming soon)                               exports/train_engine_8b/
  producer (gates + prompts + control plane)           product (a runnable training framework)
```

Each subdirectory has its own README with full CLI docs, config reference, layout, performance baselines, and limitations.

---

## Quick Start

> **Environment**: Python ≥ 3.11 · CUDA 12.x · **PyTorch ≥ 2.4** · NVIDIA H100 80GB (SM90). Full pretraining requires 8× H100; early alignment stages run on a single GPU.

Use the training framework directly → `exports/train_engine_0.5B/`

---

### Use the training framework

**Goal**: take the ready-made framework and run pretraining on your H100s.

#### 1. Install

```bash
git clone https://github.com/OpenBMB/ForgeTrain.git
cd ForgeTrain/exports/train_engine_0.5B
pip install -e .
pip install datasets transformers   # HuggingFace data path (required)
```

#### 2. Verify install

```bash
PYTHONPATH=src python -c "from training_engine_tensor import config; print('OK')"
```

Expected output: `OK`

#### 3. Precompile operators (first run only; subsequent runs reuse the cache)

```bash
PYTHONPATH=src CUSTOM_GEMM=1 OP_ATTENTION=v1 \
    python scripts/precompile_ops.py
```

Warms up AOT export + `cpp_extension` builds for the 5 CuTeDSL GEMMs, persisting under `${ENGINE_ROOT}/.persist_cache/`. Subsequent jobs reuse the cache; only a few seconds of `dlopen` cost remains.

#### 4. Single-node 8× H100, bring your own HF dataset

```bash
torchrun --standalone --nproc-per-node=8 \
    -m training_engine_tensor pretrain \
    --num-steps 200 \
    --global-batch-size 1280 --micro-batch-size 10 \
    --seq-length 4096 \
    --hf-dataset openai/gsm8k \
    --hf-dataset-config main \
    --hf-text-template "Question: {question}\nAnswer: {answer}" \
    --tokenizer-path openbmb/MiniCPM4-0.5B \
    --save-dir ./checkpoints/run1
```

`--hf-dataset` accepts a HuggingFace Hub name (e.g. `openai/gsm8k`) or a local Parquet / Arrow / JSON directory.

**Expected output (200 steps on 8× H100)**

Each step logs a line like:

```
[STEP 200] loss=X.XXX | step_time=XXXms | mfu=44.XX%
```

On 8× H100 with BF16, expect **MFU ~44%** and step time ~XXXms for GBS=1280 / MBS=10 / seq=4096.

> Full documentation including multi-node training, checkpoint resume, and HuggingFace export → [`exports/train_engine_0.5B/README.md`](./exports/train_engine_0.5B/README.md) · [`exports/train_engine_8b/README.md`](./exports/train_engine_8b/README.md)

---

## Core Technology

### `harness/` — the scaffolding that lets an Agent produce a training framework *(coming soon)*

- **Zero-touch Agent Loop** — `bash agent-loop.sh` runs a Coding Agent in a loop against the prompt files, with no human in the loop
- **Two-stage gate-driven convergence**:
  - **Stage 1 (M1-M6)** — bitwise forward / backward alignment → DP=8 multi-step training → long-train statistical gate (loss rel diff < 1%, MFU(standard) ≥ 36%)
  - **Stage 2** — per-operator CUDA kernel optimization, 30 rounds per op with best-MFU election, with a DP=8 long-train integration gate on every merge
- **Portable to any reference training stack** — Megatron-LM v0.15 / torch in this reproduction, but any working stack (DeepSpeed, custom, …) honoring the same env contract drops in unchanged

### `exports/train_engine_0.5B/` — the training framework the Agent wrote (0.5B)

- **CUDA Graph × 5 capture granularities** — forward / step / step_full / step_optimizer / step_nccl_opt, freely composable with BucketedGradReducer / sharded optimizer / wgrad-overlap
- **Triton fused kernels** — one kernel each for CE fwd+bwd / SwiGLU / RMSNorm+residual / RoPE / fused Adam+param sync
- **Self-explored optimization space** — the Agent enumerated and benchmarked CuTeDSL / cuBLAS / Triton / TransformerEngine operator variants and dozens of comm + CUDA-Graph capture combinations in real distributed jobs, scoring both MFU and loss alignment; production defaults are the optimum the Agent picked

### `exports/train_engine_8b/` — the training framework the Agent wrote (8B)

- **MiniCPM4-8B on a single 8× H100 host** — `tensor_model_parallel_size=2`, **50.9% MFU**, ~8% above the Megatron-LM baseline

---

## Performance

| Metric | ForgeTrain | Megatron-LM (baseline) |
|---|---|---|
| **MFU** | **44.13%** | ~40% |
| **MFU improvement** | **+10%** | — |

> Test conditions: MiniCPM4-0.5B · 64× H100 · BF16 · DP-only.

> Full performance guide → [`exports/train_engine_0.5B/README.md`](./exports/train_engine_0.5B/README.md) · [`exports/train_engine_8b/README.md`](./exports/train_engine_8b/README.md)

---

## Contributing

Contributions are welcome! Here are some ways to help:

- 🐛 **Bug reports** — [open an issue](https://github.com/OpenBMB/ForgeTrain/issues)
- 💡 **Feature requests** — [open an issue](https://github.com/OpenBMB/ForgeTrain/issues) with `[feature]` in the title
- 📝 **Reproducibility reports** — share your experience reproducing the Agent Loop on different setups
- 🔧 **Pull requests** — code improvements, documentation fixes, and new model support

---

## License

Licensed under the [Apache License 2.0](./LICENSE).

The vendored `exports/train_engine_0.5B/src/quack/` and `exports/train_engine_8b/src/quack/` snapshots retain their upstream copyright headers; see [`exports/train_engine_0.5B/src/quack/NOTICE.md`](./exports/train_engine_0.5B/src/quack/NOTICE.md) and [`exports/train_engine_8b/src/quack/NOTICE.md`](./exports/train_engine_8b/src/quack/NOTICE.md) for provenance. Built on a reference training stack (Megatron-LM v0.15 in this reproduction) and the Cursor Coding Agent; data and tokenizers follow MiniCPM4-0.5B / MiniCPM4-8B upstream.

---

## Acknowledgments

ForgeTrain builds on the work of several outstanding open-source projects:

- [CUTLASS](https://github.com/NVIDIA/cutlass) / [CuTeDSL](https://github.com/NVIDIA/cutlass) — CuTeDSL GEMMs and helper utilities
- [FlashAttention-4](https://github.com/Dao-AILab/flash-attention) (Dao-AILab) — FA4 CuTeDSL SM90 attention kernels
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) (NVIDIA) — reference operator implementations
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (NVIDIA) — reference training stack for gate verification
- [Cursor](https://cursor.com) — the Coding Agent that authored the engine
- [MiniCPM4](https://github.com/OpenBMB/MiniCPM) (OpenBMB) — target model architecture and tokenizer

---

## Citation

If you find this project useful, please consider citing:

```bibtex
@misc{forgetrain2026,
  title  = {ForgeTrain: Forging Production-Grade Training Frameworks via Harness-Driven AI Development},
  author = {Zhu, Zhui and He, Qingfeng and Li, Shangzhan and Chen, Yaojian and Sun, Haojun and Li, Zhen and Li, Yuxuan and Liu, Zhiyuan},
  year   = {2026},
  url    = {https://github.com/OpenBMB/ForgeTrain}
}
```

---

## Hardware / Software Baseline

| Item | Requirement |
|---|---|
| GPU | NVIDIA H100 80GB (SM90, Hopper) |
| GPU count | 8× H100 for full gates / pretraining; 1× H100 for early alignment |
| CUDA | 12.x |
| PyTorch | ≥ 2.4 |
| Python | ≥ 3.11 |
| Validated scope | MiniCPM4-0.5B (DP-only) / MiniCPM4-8B (TP=2) × H100 × BF16 |
