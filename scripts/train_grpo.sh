#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1,2,3,7
export N_GPUS=4
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export NCCL_P2P_LEVEL=NVL

export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

export DATA_DIR="dataset/token_call_token"
export EXPERIMENT_NAME="saves/qwen-2.5-7b-token-call-token-mr-use-5e-5-think-5e-6-bs64-ep8-grpo-2e-6"
export BASE_MODEL="checkpoints/qwen-2.5-7b-token-call-token-mr-use-5e-5-think-5e-6-bs64-ep8"
export LEARNING_RATE="2e-6"
export MICRO_BATCH_SIZE_PER_GPU=32
export TENSOR_MODEL_PARALLEL_SIZE=1
export USE_DYNAMIC_BSZ=True
OFFLOAD=${OFFLOAD:-False}

bash ./examples/sglang_multiturn/run_rlla_multiturn_w_interaction.sh


