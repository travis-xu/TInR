#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-checkpoints/qwen-2.5-7b-token-call-token-mr-use-5e-5-think-5e-6-bs64-ep8-grpo-2e-6}"
EVAL_FILE="${EVAL_FILE:-dataset/token_call_token/eval.json}"
VIRTUAL_TOKENS_FILE="${VIRTUAL_TOKENS_FILE:-training/src/configs/virtual_tokens_token_call_token.txt}"
TOOL_FILE="${TOOL_FILE:-dataset/tools.json}"

python -m training.test \
  --model_path "${MODEL_PATH}" \
  --eval_file "${EVAL_FILE}" \
  --virtual_tokens_file "${VIRTUAL_TOKENS_FILE}" \
  --tool_file "${TOOL_FILE}" \
  --write_results True \
  --batch_size 32


