#!/bin/bash
# GRPO training script for table parsing (table image to HTML)
# Model: Qwen/Qwen3-VL-4B-Thinking
# Dataset: Local JSONL (run examples/table_grpo/prepare_dataset.py to generate)
# Reward functions: TEDS and GriTS
# Hardware: 8 x H100 GPUs (2 for rollout, 6 for training)

# =============================================================================
# Step 1: Start the rollout server on GPU 0,1 (run this first in a separate terminal)
# =============================================================================
# MAX_PIXELS=1003520 \
# CUDA_VISIBLE_DEVICES=0,1 \
# swift rollout \
#     --model Qwen/Qwen3-VL-4B-Thinking \
#     --vllm_tensor_parallel_size 2 \
#     --vllm_max_model_len 16384

# =============================================================================
# Step 2: Run GRPO training on GPU 2-7 (run this after the rollout server is ready)
# =============================================================================
export MAX_PIXELS=1003520

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-4B-Thinking \
    --dataset examples/table_grpo/train_table_grpo.jsonl \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --system "You are an expert at parsing table images into HTML code. Given a table image, output the corresponding HTML table with accurate structure and cell content. Output only the raw HTML table using <table>, <thead>, <tbody>, <tr>, <th>, <td> tags with colspan and rowspan attributes where needed. Do not include any CSS styles, class attributes, or inline styling. Do not add any explanation." \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --output_dir output/grpo_table_qwen3vl \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 8192 \
    --reward_funcs teds grits table_format \
    --reward_weights 0.5 0.3 0.2 \
    --num_generations 8 \
    --sleep_level 1 \
    --temperature 0.7 \
    --top_p 0.9 \
    --beta 0.04 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard wandb
