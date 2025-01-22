## Lora Training Code

The default training commands for the different versions are as follows:

We can choose whether to use deep speed in EasyAnimate, which can save a lot of video memory. 

Some parameters in the sh file can be confusing, and they are explained in this document:

- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the images and videos at the center, but instead, it trains the entire images and videos after grouping them into buckets based on resolution.
- `random_frame_crop` is used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt` is used to enable automatic height and width scaling for images and videos. When `random_hw_adapt` is enabled, the training images will have their height and width set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum. For training videos, the height and width will be set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=1024`, and `image_sample_size=1024`, the resolution of image inputs for training is `512x512` to `1024x1024`, and the resolution of video inputs for training is `512x512x49` to `1024x1024x49`.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49`.
- `training_with_video_token_length` specifies training the model according to token length. For training images and videos, the height and width will be set to `image_sample_size` as the maximum and `video_sample_size` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=1024`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x49`.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x9`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `train_mode` is used to specify the training mode, which can be either normal or inpaint. Since EasyAnimate uses the Inpaint model to achieve image-to-video generation, the default is set to inpaint mode. If you only wish to achieve text-to-video generation, you can remove this line, and it will default to the text-to-video mode.
- `uniform_sampling` is used to ensure that each batch can be uniformly sampled from 0 to 1000.
- The default parameter for training is the Inpaint model. If you only want to train the T2V model, please set train_made="normal" and use the EasyAnimateV5-12b-zh model.
- `loss_type`: The loss type for training. Currently, flow is used in v5.1, ddpm is used in v5 and v4, sigma is used in v3, v2 and v1. 

EasyAnimateV5.1-InP without deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5.1_magvit_qwen.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --loss_type="flow" \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="inpaint"
```

EasyAnimateV5.1-InP with deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5.1_magvit_qwen.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --loss_type="flow" \
  --enable_bucket \
  --use_deepspeed \
  --uniform_sampling \
  --train_mode="inpaint"
```

EasyAnimateV5-InP without deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5-12b-zh-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5_magvit_multi_text_encoder.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --loss_type="ddpm" \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="inpaint"
```

EasyAnimateV5 with deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5-12b-zh-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5_magvit_multi_text_encoder.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --loss_type="ddpm" \
  --enable_bucket \
  --use_deepspeed \
  --uniform_sampling \
  --train_mode="inpaint"
```

<details>
  <summary>(Obsolete) EasyAnimateV4:</summary>

EasyAnimateV4 without deepspeed:
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v4_slicevae_multi_text_encoder.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --motion_sub_loss \
  --loss_type="ddpm" \
  --enable_bucket \
  --train_mode="inpaint"
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
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v4_slicevae_multi_text_encoder.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --motion_sub_loss \
  --loss_type="ddpm" \
  --enable_bucket \
  --use_deepspeed \
  --train_mode="inpaint"
```
</details>

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>
  
```sh
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch --mixed_precision="bf16" scripts/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v3_slicevae_motion_module.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=1 \
  --vae_mini_batch=1 \
  --enable_bucket \
  --train_mode="inpaint"
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
accelerate launch --mixed_precision="bf16" scripts/train_t2iv_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v2_magvit_motion_module.yaml" \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=144 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_steps=100 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --max_grad_norm=1 \
  --vae_mini_batch=1 \
  --enable_bucket
```
</details>