export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5-12b-zh-InP"
export TRAIN_PROMPT_PATH="MovieGenVideoBench_train.txt"
# Performing validation simultaneously with training will increase time and GPU memory usage.
export VALIDATION_PROMPT_PATH="MovieGenVideoBench_val.txt"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --num_processes=8 --mixed_precision="bf16" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/train_reward_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/easyanimate_video_v5_magvit_multi_text_encoder.yaml" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --checkpointing_steps=100 \
  --learning_rate=1e-05 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=0.3 \
  --low_vram \
  --use_deepspeed \
  --prompt_path=$TRAIN_PROMPT_PATH \
  --train_sample_height=256 \
  --train_sample_width=256 \
  --video_length=49 \
  --validation_prompt_path=$VALIDATION_PROMPT_PATH \
  --validation_steps=100 \
  --validation_batch_size=8 \
  --num_decoded_latents=1 \
  --reward_fn="HPSReward" \
  --reward_fn_kwargs='{"version": "v2.1"}' \
  --backprop

# For V5.1
# export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
# export TRAIN_PROMPT_PATH="MovieGenVideoBench_train.txt"
# # Performing validation simultaneously with training will increase time and GPU memory usage.
# export VALIDATION_PROMPT_PATH="MovieGenVideoBench_val.txt"

# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# # When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
# accelerate launch --num_processes=8 --mixed_precision="bf16" --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/train_reward_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --config_path="config/easyanimate_video_v5.1_magvit_qwen.yaml" \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --max_train_steps=10000 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-05 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --max_grad_norm=0.3 \
#   --low_vram \
#   --use_deepspeed \
#   --prompt_path=$TRAIN_PROMPT_PATH \
#   --train_sample_height=256 \
#   --train_sample_width=256 \
#   --video_length=49 \
#   --num_decoded_latents=1 \
#   --reward_fn="HPSReward" \
#   --reward_fn_kwargs='{"version": "v2.1"}' \
#   --backprop_strategy "tail" \
#   --backprop_num_steps 10