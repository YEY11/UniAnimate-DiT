#!/usr/bin/env bash
set -euo pipefail

# ========= GPU 设置（单卡/多卡通用）=========
# 传入示例：GPU_IDS="0" 或 GPU_IDS="0,1" 或 "0,1,2,3"
GPU_IDS="${GPU_IDS:-0}"
IFS=',' read -r -a GPU_ARR <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARR[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# 训练策略：默认 auto；多卡时如未显式指定，则自动切到 deepspeed_stage_2
TRAINING_STRATEGY="${TRAINING_STRATEGY:-auto}"
if (( NUM_GPUS > 1 )); then
  if [[ "$TRAINING_STRATEGY" == "auto" ]]; then
    TRAINING_STRATEGY="deepspeed_stage_2"
  fi
fi


# 推荐的 NCCL/通信环境
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=INFO

# 单机多卡常见稳妥设置（如环境没有 InfiniBand）
export NCCL_IB_DISABLE=1
# 若机器上有多网卡，指定实际可用网卡名称（一般是 eth0），避免走 lo/docker0
export NCCL_SOCKET_IFNAME=eth0


# ========= 时间戳，用于检查点目录和日志文件命名 ========= 
TIMESTAMP="$(date +%Y%m%d%H%M%S)"

# ========= 数据与模型路径（按你的环境修改）=========
DATASET_PATH="data/example_dataset"
OUTPUT_PATH="./models_out_lora"
CHECKPOINT_DIR="${OUTPUT_PATH}/checkpoints_${TIMESTAMP}"

DIT_PATH="./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,./Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"
TEXT_ENCODER_PATH="./Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="./Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"
IMAGE_ENCODER_PATH="./Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# ========= 训练超参 =========
MAX_EPOCHS=10
LR=1e-4
ACCUM_STEPS=1
LORA_RANK=64
LORA_ALPHA=64

# ========= 保存策略（方案D参数，对应你已经加到 parse_args 的四个参数）=========
SAVE_EVERY_N_EPOCHS=3          # 按 epoch 保存频率；若使用按步保存将被覆盖
SAVE_LAST_FLAG="--save_last"   # 另存 last.ckpt
# 按步保存：例如 "--save_every_n_train_steps 10"；为空则使用按 epoch 策略
SAVE_EVERY_N_TRAIN_STEPS=""

# ========= 日志设置 =========
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_lora_${TIMESTAMP}.log"

# ========= 运行前准备 =========
mkdir -p "$OUTPUT_PATH" "$CHECKPOINT_DIR"

echo "可见 GPU: $GPU_IDS (共 ${NUM_GPUS} 张)"
echo "使用训练策略: $TRAINING_STRATEGY"
echo "日志将保存到: $LOG_FILE"
echo "检查点将保存到: $CHECKPOINT_DIR"

# ========= 启动训练 =========
python examples/unianimate_wan/train_unianimate_wan.py \
  --task train \
  --train_architecture lora \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --dataset_path "$DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --dit_path "$DIT_PATH" \
  --text_encoder_path "$TEXT_ENCODER_PATH" \
  --vae_path "$VAE_PATH" \
  --image_encoder_path "$IMAGE_ENCODER_PATH" \
  --max_epochs "$MAX_EPOCHS" \
  --learning_rate "$LR" \
  --accumulate_grad_batches "$ACCUM_STEPS" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS" \
  $SAVE_LAST_FLAG \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --training_strategy "$TRAINING_STRATEGY" \
  $SAVE_EVERY_N_TRAIN_STEPS \
  2>&1 | tee -a "$LOG_FILE"

# 若希望在后台运行且只写日志，不在终端输出，可改用：
# nohup python examples/unianimate_wan/train_unianimate_wan.py \
#   ...同上参数... \
#   > "$LOG_FILE" 2>&1 &


######################################## 使用说明 ########################################
# 1. 单卡：直接运行 bash 脚本，或指定 GPU_IDS="0" bash train_lora.sh。
# 2. 多卡：GPU_IDS="0,1" bash train_lora.sh 或 GPU_IDS="0,1,2,3" bash train_lora.sh。脚本会自动把训练策略切换为 deepspeed_stage_2（可通过 TRAINING_STRATEGY 环境变量覆盖为 auto、deepspeed_stage_1/2/3）。
#   2.1 默认：多卡时自动用 ZeRO-2（模型复制，优化器/梯度分片）
#       GPU_IDS="0,1" TRAINING_STRATEGY="deepspeed_stage_2" bash train_lora.sh
#   2.2 模型参数也分片（ZeRO-3），适合显存紧张的大模型
#       GPU_IDS="0,1" TRAINING_STRATEGY="deepspeed_stage_3" bash train_lora.sh
#   2.3 纯 DDP（完全复制模型），最简单的多卡
#       GPU_IDS="0,1" TRAINING_STRATEGY="ddp" bash train_lora.sh
#   2.4 FSDP（完全分片并按需聚合），同样是模型分片的方案
#       GPU_IDS="0,1,2,3" TRAINING_STRATEGY="fsdp" bash train_lora.sh
# 3. 日志：训练输出同时打印到终端并写入 logs/train_lora_{时间戳}.log。
# 4. 检查点：默认存到 ${OUTPUT_PATH}/checkpoints；可用 --checkpoint_dir 覆盖。
##########################################################################################
