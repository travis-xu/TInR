# TInR

Code for our paper [TInR: Exploring Tool-Internalized Reasoning in Large Language Models](https://arxiv.org/abs/2604.10788).

---

## 1. Environment Setup

- **Python**: 3.10+
- **CUDA / GPUs**: The provided scripts assume a multi-GPU environment.

Install dependencies:

```bash
cd TInR
pip install -r requirements.txt
```

If you only need the basic training pipeline (without web / retrieval tools), most experiments will work with the core dependencies section of `requirements.txt`.

---

## 2. Data Preparation

### 2.1 Tool definitions

Tool definitions (names, descriptions, parameters, and virtual tokens) are stored in:

- `dataset/tools.json`

This file is used both for:

- Constructing **virtual token** vocabularies.
- Evaluating tool-use behavior (via `training/test.py` and `verl/interactions`).

### 2.2 Training datasets

This repository assumes preprocessed datasets in the following locations:

- `dataset/token_call_token/` – raw JSON / parquet data for token-level tool calls.
- `training/dataset/token_call_token/` – processed JSONs for model training (e.g. `train_memorization_use.json`, `train_memorization_recall_use.json`, `train_think.json`).

If you need to regenerate processed datasets from `tools.json` and external raw data, you can write your own preprocessing script following the format of files in `training/dataset/token_call_token/`.

---

## 3. Training Pipeline

### 3.1 Stage 1 – Memorization / Recall / Use

Usage:

```bash
cd TInR
bash scripts/train_memorization_recall_use.sh
```

This script:

- Trains on `training/dataset/token_call_token/train_memorization_use.json` (and related data) to:
  - Memorize mappings from tool docs to virtual tokens.
  - Recall docs from tokens.
  - Learn basic tool-use patterns.

Key hyperparameters and paths can be edited directly in `scripts/train_memorization_recall_use.sh`.

### 3.2 Stage 2 – Think-augmented Tool Use

Script:

- `scripts/train_think.sh`

Usage:

```bash
cd TInR
bash scripts/train_think.sh
```

This script:

- Continues from a Stage-1 checkpoint:
  - `checkpoints/qwen-2.5-7b-memorization-recall-token-call-token-5e-5`
- Trains on:
  - `training/dataset/token_call_token/train_think.json`
- Optimizes the model to generate explicit `<think>...</think>` reasoning traces before tool calls.

You can change:

- `model_name_or_path` – to point to your own Stage-1 checkpoint.
- `datasets` – to another think-style dataset in the same format.

### 3.3 Stage 3 – GRPO Training

Usage:

```bash
cd TInR
bash scripts/train_grpo.sh
```

This script:

- Sets environment variables and then launches a GRPO multi-turn trainer via:
  - `examples/sglang_multiturn/run_rlla_multiturn_w_interaction.sh`
- Uses:
  - `DATA_DIR="dataset/token_call_token"`
  - `EXPERIMENT_NAME` under `saves/`
  - A base checkpoint specified by `BASE_MODEL`

To adapt to your environment:

- Set `BASE_MODEL` to a Stage-2 checkpoint saved under `checkpoints/`.
- Point `DATA_DIR` to your own tool-use dataset directory under `dataset/`.
- Adjust `CUDA_VISIBLE_DEVICES`, `N_GPUS`, and batch-size parameters as needed.

---

## 4. Evaluation

Usage:

```bash
cd TInR
bash scripts/test.sh
```

You can override default paths with environment variables:

```bash
MODEL_PATH=checkpoints/your_model \
EVAL_FILE=dataset/token_call_token/eval.json \
VIRTUAL_TOKENS_FILE=training/src/configs/virtual_tokens.txt \
TOOL_FILE=dataset/tools.json \
bash scripts/test.sh
```
