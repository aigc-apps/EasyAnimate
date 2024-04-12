import os

import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer3d import Transformer3DModel
from easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import save_videos_grid

# Config and model path
config_path         = "config/easyanimate_video_motion_module_v1.yaml"
model_name          = "models/Diffusion_Transformer/PixArt-XL-2-512x512"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "Euler A"

# Load pretrained model if need
transformer_path    = None
motion_module_path  = "models/Motion_Module/easyanimate_v1_mm.safetensors" 
vae_path            = None
lora_path           = None

# other params
sample_size         = [512, 512]
video_length        = 80
fps                 = 12

weight_dtype        = torch.float16
prompt              = "1girl, bangs, blue eyes, blunt bangs, 4k, best quality, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt, (best quality), (realistic, photo-realistic:1.3), (beautiful eyes:1.3), (sparkling eyes:1.3), (beautiful mouth:1.3), finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup."# "1girl, bangs, blue eyes, blunt bangs, 4k, best quality, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt, (best quality), (realistic, photo-realistic:1.3), (beautiful eyes:1.3), (sparkling eyes:1.3), (beautiful mouth:1.3), finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup."
negative_prompt     = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry"
guidance_scale      = 7.0
seed                = 42
num_inference_steps = 30
lora_weight         = 0.55
save_path           = "samples/easyanimate-videos"

config = OmegaConf.load(config_path)

# Get Transformer
transformer = Transformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
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
    torch_dtype=weight_dtype
)

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

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

pipeline = EasyAnimatePipeline.from_pretrained(
    model_name,
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=weight_dtype
)
pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

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
    ).videos

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)
video_path = os.path.join(save_path, prefix + ".gif")
save_videos_grid(sample, video_path, fps=fps)