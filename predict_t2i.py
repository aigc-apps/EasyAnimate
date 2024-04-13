import os

import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer2d import Transformer2DModel
from easyanimate.pipeline.pipeline_pixart_magvit import PixArtAlphaMagvitPipeline
from easyanimate.utils.lora_utils import merge_lora

# Config and model path
config_path         = "config/easyanimate_image_normal_v1.yaml"
model_name          = "models/Diffusion_Transformer/PixArt-XL-2-512x512"
# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
sampler_name        = "DPM++"

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size     = [512, 512]
weight_dtype    = torch.float16
prompt          = "1girl, bangs, blue eyes, blunt bangs, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt."
negative_prompt = "bad detailed"
guidance_scale  = 6.0
seed            = 43
lora_weight     = 0.55
save_path       = "samples/easyanimate-images"

config = OmegaConf.load(config_path)

# Get Transformer
transformer = Transformer2DModel.from_pretrained(
    model_name, 
    subfolder="transformer"
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
    assert len(u) == 0

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

# PixArtAlphaMagvitPipeline is compatible with PixArtAlphaPipeline
pipeline = PixArtAlphaMagvitPipeline.from_pretrained(
    model_name,
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=weight_dtype
)
pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

generator = torch.Generator(device="cuda").manual_seed(seed)

with torch.no_grad():
    sample = pipeline(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
    ).images[0]

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

index = len([path for path in os.listdir(save_path)]) + 1
prefix = str(index).zfill(8)
image_path = os.path.join(save_path, prefix + ".png")
sample.save(image_path)