export MODEL_NAME="models/EasyAnimateV4-XL-2-InP-512x512"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/image_video.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train_lcm_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_slicevae_multi_text_encoder_v4.yaml" \
  --image_sample_size=256 \
  --video_sample_size=256 \
  --token_sample_size=256 \
  --video_sample_stride=1 \
  --video_sample_n_frames=32 \
  --train_batch_size=8 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_hunyuan_256_InP_slice_vae_all_attention_f32" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=0.0 \
  --max_grad_norm=1 \
  --vae_mini_batch=1 \
  --random_frame_crop \
  --enable_bucket \
  --low_vram \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --motion_sub_loss \
  --not_sigma_loss \
  --resume_from_checkpoint="latest" \
  --train_mode="inpaint" \
  --trainable_modules "." \
  --validation_prompts "a girl is smiling" \
  --validation_steps 100 \
  --checkpoints_total_limit 2