#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export NCCL_P2P_LEVEL=NVL
master_port=25025
GPU_ID=4,5,6,7

cd training

add_virtual_tokens="False"
model_name_or_path="checkpoints/qwen-2.5-7b-memorization-recall-token-call-token-5e-5"
run_name="qwen-2.5-7b-token-call-token-mr-use-5e-5-think-5e-6-bs64-ep8"
output_dir="checkpoints/${run_name}"
virtual_tokens_file=None
template="qwen-7b-chat"
flash_attention="True"
datasets="dataset/token_call_token/train_think.json"
dataset_nums="10000000"
max_length="4096"
per_device_train_batch_size="1"
lr="5e-6"
accumulation_steps="16"
epochs="8"
save_strategy="steps"
save_steps="1000"
zero="z3_offload" # z3_offload, z3
chat="True"

deepspeed --include=localhost:${GPU_ID} --master_port ${master_port} train.py \
  --model_name_or_path ${model_name_or_path} \
  --add_virtual_tokens ${add_virtual_tokens} \
  --virtual_tokens_file ${virtual_tokens_file} \
  --flash_attention ${flash_attention} \
  --deepspeed src/configs/ds_${zero}_config.json \
  --chat ${chat} \
  --template ${template} \
  --architecture causal \
  --output_dir ${output_dir} \
  --save_strategy ${save_strategy} \
  --gather_weights True \
  --learning_rate ${lr} \
  --warmup_ratio 0.03 \
  --datasets ${datasets} \
  --dataset_nums ${dataset_nums} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --gradient_accumulation_steps ${accumulation_steps} \
  --max_length ${max_length} \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name ${run_name} \
  --num_train_epochs ${epochs} \
  --save_steps ${save_steps}


