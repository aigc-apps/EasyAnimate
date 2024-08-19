import os
import numpy as np
import torch
import cv2
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPVisionModelWithProjection,  CLIPImageProcessor

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.models.transformer3d import Transformer3DModel, HunyuanTransformer3DModel
from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder_inpaint import EasyAnimatePipeline_Multi_Text_Encoder_Inpaint
from easyanimate.pipeline.pipeline_easyanimate_inpaint import \
    EasyAnimateInpaintPipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid, get_image_to_video_latent

# Low gpu memory mode, this is used when the GPU memory is under 16GB
low_gpu_memory_mode = False

# Config and model path
config_path         = "config/easyanimate_video_slicevae_multi_text_encoder_v4.yaml"
model_name          = "models/Diffusion_Transformer/EasyAnimateV4-XL-2-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "Euler"

# Load pretrained model if need
transformer_path    = None
# V2 and V3 does not need a motion module
motion_module_path  = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [384, 672]
# In EasyAnimateV1, the video_length of video is 40 ~ 80.
# In EasyAnimateV2 and V3, the video_length of video is 1 ~ 144. If u want to generate a image, please set the video_length = 1.
video_length        = 144
fps                 = 24

# If you want to generate ultra long videos, please set partial_video_length as the length of each sub video segment
partial_video_length = None
overlap_video_length = 4

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "asset/1.png"
validation_image_end    = None

# We support English and Chinese in V4
prompt                  = "一条狗看着屏幕。质量高、杰作、最佳品质、高分辨率、超精细、梦幻般。"
negative_prompt         = "低质量，不清晰，突变，变形，失真。"
# prompt                  = "The dog is looking at camera and smiling. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."
# negative_prompt         = "The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion. "
guidance_scale          = 7.0
seed                    = 43
num_inference_steps     = 25
lora_weight             = 0.60
save_path               = "samples/easyanimate-videos_i2v"

config = OmegaConf.load(config_path)

# Get Transformer
if config.get('enable_multi_text_encoder', False):
    Choosen_Transformer3DModel = HunyuanTransformer3DModel
else:
    Choosen_Transformer3DModel = Transformer3DModel

transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
if weight_dtype == torch.float16:
    transformer_additional_kwargs["upcast_attention"] = True

transformer = Choosen_Transformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if motion_module_path is not None:
    print(f"From Motion Module: {motion_module_path}")
    if motion_module_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(motion_module_path)
    else:
        state_dict = torch.load(motion_module_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

# Get Vae
if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
    Choosen_AutoencoderKL = AutoencoderKLMagvit
else:
    Choosen_AutoencoderKL = AutoencoderKL
vae = Choosen_AutoencoderKL.from_pretrained(
    model_name, 
    subfolder="vae", 
).to(weight_dtype)
if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit'] and weight_dtype == torch.float16:
    vae.upcast_vae = True

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    model_name, subfolder="image_encoder"
).to("cuda", weight_dtype)
clip_image_processor = CLIPImageProcessor.from_pretrained(
    model_name, subfolder="image_encoder"
)

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}[sampler_name]

if config.get('enable_multi_text_encoder', False):
    scheduler = Choosen_Scheduler.from_pretrained(
        model_name, 
        subfolder="scheduler"
    )
    pipeline = EasyAnimatePipeline_Multi_Text_Encoder_Inpaint.from_pretrained(
        model_name,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )
else:
    scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

    pipeline = EasyAnimateInpaintPipeline.from_pretrained(
        model_name,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )

if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, "cuda")

if partial_video_length is not None:
    init_frames = 0
    last_frames = init_frames + partial_video_length
    while init_frames < video_length:
        if last_frames >= video_length:
            if pipeline.vae.quant_conv.weight.ndim==5:
                mini_batch_encoder = pipeline.vae.mini_batch_encoder
                _partial_video_length = video_length - init_frames
                _partial_video_length = int(_partial_video_length // mini_batch_encoder * mini_batch_encoder)
            else:
                _partial_video_length = video_length - init_frames
            
            if _partial_video_length <= 0:
                break
        else:
            _partial_video_length = partial_video_length

        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image, None, video_length=_partial_video_length, sample_size=sample_size)
        
        with torch.no_grad():
            sample = pipeline(
                prompt + ". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic. ", 
                video_length = _partial_video_length,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,

                video        = input_video,
                mask_video   = input_video_mask,
                clip_image   = clip_image, 
            ).videos
        
        if init_frames != 0:
            mix_ratio = torch.from_numpy(
                np.array([float(_index) / float(overlap_video_length) for _index in range(overlap_video_length)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            new_sample[:, :, -overlap_video_length:] = new_sample[:, :, -overlap_video_length:] * (1 - mix_ratio) + \
                sample[:, :, :overlap_video_length] * mix_ratio
            new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim = 2)

            sample = new_sample
        else:
            new_sample = sample

        if last_frames >= video_length:
            break

        validation_image = [
            Image.fromarray(
                (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
            ) for _index in range(-overlap_video_length, 0)
        ]

        init_frames = init_frames + _partial_video_length - overlap_video_length
        last_frames = init_frames + _partial_video_length
else:
    video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1
    input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)

    with torch.no_grad():
        sample = pipeline(
            prompt, 
            video_length = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,

            video        = input_video,
            mask_video   = input_video_mask,
            clip_image   = clip_image, 
        ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, "cuda")
    
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)
    
if video_length == 1:
    save_sample_path = os.path.join(save_path, prefix + f".png")

    image = sample[0, :, 0]
    image = image.transpose(0, 1).transpose(1, 2)
    image = (image * 255).numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_sample_path)
else:
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_videos_grid(sample, video_path, fps=fps)