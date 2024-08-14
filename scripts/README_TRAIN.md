## Training Code

EasyAnimate currently maintains several versions, each with different training commands. The default training commands for the different versions are as follows:

We can choose whether to use deep speed in EasyAnimateV4, which can save a lot of video memory. 

EasyAnimateV4 without deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_slicevae_multi_text_encoder_v4.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
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
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --motion_sub_loss \
  --not_sigma_loss \
  --random_frame_crop \
  --enable_bucket \
  --train_mode="inpaint" \
  --trainable_modules "."
```

EasyAnimateV4 with deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_slicevae_multi_text_encoder_v4.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
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
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --motion_sub_loss \
  --not_sigma_loss \
  --random_frame_crop \
  --enable_bucket \
  --use_deepspeed \
  --train_mode="inpaint" \
  --trainable_modules "."
```

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

EasyAnimateV3's training requires a larger batch size; otherwise, the training may become unstable. If the batch size is small, you can choose to use LoRA training or freeze some layers during training. 

For example, we can set trainable_modules to "transformer_blocks.6", "transformer_blocks.7", "transformer_blocks.8", "transformer_blocks.9", "transformer_blocks.10", "transformer_blocks.11", "transformer_blocks.12", which means only these modules will be trained.

```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_slicevae_motion_module_v3.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=3e-2 \
  --vae_mini_batch=1 \
  --random_frame_crop \
  --enable_bucket \
  --train_mode="inpaint" \
  --trainable_modules "transformer_blocks" "proj_out" "long_connect_fc"
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>

```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_magvit_motion_module_v2.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=500 \
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
  --max_grad_norm=3e-2 \
  --vae_mini_batch=1 \
  --random_frame_crop \
  --enable_bucket \
  --trainable_modules "transformer_blocks" "proj_out" "long_connect_fc"
```
</details>