export MODEL_NAME="models/Diffusion_Transformer/PixArt-XL-2-512x512"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
# vae_mode can be choosen in "normal" and "magvit"
# transformer_mode can be choosen in "normal" and "kvcompress"
accelerate launch --mixed_precision="bf16" scripts/train_t2iv.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_motion_module_v1.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=16 \
  --train_batch_size=2 \
  --video_repeat=1 \
  --image_repeat_in_forward=4 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --validation_prompts="A girl with delicate face is smiling." \
  --validation_epochs=1 \
  --validation_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=1 \
  --vae_mini_batch=16 \
  --use_ema \
  --trainable_modules "attn_temporal"