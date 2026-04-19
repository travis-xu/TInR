#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export NCCL_P2P_LEVEL=NVL
master_port=25027
GPU_ID=0,1,2,3

cd training

model_name_or_path="/data/models/Qwen2.5-3B-Instruct"
output_dir="checkpoints/qwen-2.5-3b-token-call-token-mr-use-2e-5"
virtual_tokens_file="src/configs/virtual_tokens_token_call_token.txt"
add_virtual_tokens="True"
template="qwen-7b-chat"
flash_attention="True"
run_name="qwen-2.5-3b-token-call-token-mr-use-2e-5"
datasets="dataset/token_call_token/train_memorization_recall_use.json"
dataset_nums="10000000"
max_length="2048"
per_device_train_batch_size="2"
lr="2e-5"
accumulation_steps="8"
epochs="8"
save_strategy="steps"
save_steps="2000"
zero="z2" # z3_offload, z3
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


