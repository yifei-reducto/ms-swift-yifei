# doppler run -- modal run --env=dev --detach swift_vl_table_grpo.py::train

import modal

image = (
    modal.Image.from_registry("modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py311-torch2.8.0-vllm0.11.0-modelscope1.32.0-swift3.11.1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install uv",
    )
    .pip_install(
        "fastapi==0.117.1",
        "pydantic==2.11.9",
    )
    .pip_install(
        "liger-kernel", "hf_transfer"
    ).env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    ).pip_install(
        "qwen-vl-utils",
        "causal-conv1d",
        "flash-linear-attention"
    ).run_commands(
        "pip install https://github.com/windreamer/flash-attention3-wheels/releases/download/2025.09.11/flash_attn_3-3.0.0b1%2B20250911.cu129torch280cxx11abitrue.dfb664-cp39-abi3-linux_x86_64.whl"
    ).pip_install(
        "ms-swift==3.12.1"
    )
)

app = modal.App("table_grpo_length_control")

DATA = modal.Volume.from_name("v2_data_yifei", create_if_missing=True)
CKPT = modal.Volume.from_name("v2_checkpoints", create_if_missing=False)

@app.function(
    image=image,
    gpu="H200:8",
    timeout=60*60*12,
    volumes={"/data": DATA, "/checkpoints": CKPT},
)
def train() -> None:
    import subprocess, textwrap

    # NOTE: This must NOT be an f-string because the bash command contains JSON like
    # --columns '{"image": "images"}' which includes braces that f-strings try to parse.
    cmd = textwrap.dedent("""

    git clone https://github.com/yifei-reducto/ms-swift-yifei.git
    cd ms-swift-yifei
    pip install -e .
    
    cd

    CUDA_VISIBLE_DEVICES=0,1 \
    swift rollout \
        --model /checkpoints/Qwen/Qwen3-VL-2B-Thinking \
        --vllm_tensor_parallel_size 2 \
        --vllm_max_model_len 16384 &

    sleep 180

    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200 \
    TORCH_NCCL_ENABLE_MONITORING=0 \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
    NPROC_PER_NODE=6 \
    WANDB_API_KEY=fa6c7eb8aa9047ba4211276eea43d561730faf34 \
    WANDB_PROJECT=table_grpo \
    swift rlhf \
        --rlhf_type grpo \
        --use_hf true \
        --model /checkpoints/Qwen/Qwen3-VL-2B-Thinking \
        --dataset /data/train_table_grpo.jsonl \
        --load_from_cache_file true \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host 127.0.0.1 \
        --vllm_server_port 8000 \
        --tuner_type full \
        --torch_dtype bfloat16 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --learning_rate 1e-5 \
        --save_total_limit 3 \
        --logging_steps 1 \
        --output_dir /checkpoints/grpo_table_qwen3vl_2b_0120_length_control \
        --gradient_accumulation_steps 4 \
        --warmup_ratio 0.05 \
        --dataloader_num_workers 8 \
        --max_completion_length 12288 \
        --vllm_max_model_len 16384 \
        --reward_funcs teds grits table_format thinking_length_penalty table_length_penalty \
        --reward_weights 0.3 0.3 0.1 0.15 0.15 \
        --scale_rewards gdpo \
        --num_generations 8 \
        --temperature 0.7 \
        --top_p 0.9 \
        --deepspeed zero3 \
        --log_completions true \
        --report_to wandb
    
    """).strip()
    subprocess.run(["bash","-lc",cmd], check=True)