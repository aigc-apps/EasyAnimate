import os

import numpy as np
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       FlowMatchEulerDiscreteScheduler, PNDMScheduler)
from omegaconf import OmegaConf
from PIL import Image
from transformers import (BertModel, BertTokenizer, CLIPImageProcessor,
                          CLIPVisionModelWithProjection, Qwen2Tokenizer,
                          Qwen2VLForConditionalGeneration, T5EncoderModel,
                          T5Tokenizer)

from easyanimate.models import (name_to_autoencoder_magvit,
                                name_to_transformer3d)
from easyanimate.models.transformer3d import get_teacache_coefficients
from easyanimate.pipeline.pipeline_easyanimate_inpaint import \
    EasyAnimateInpaintPipeline
from easyanimate.utils.fp8_optimization import (convert_model_weight_to_float8,
                                                convert_weight_dtype_wrapper)
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import get_video_to_video_latent, save_videos_grid

# GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
# 
# EasyAnimateV3 support "model_cpu_offload" "sequential_cpu_offload"
# EasyAnimateV4, V5 and V5.1 support "model_cpu_offload" "model_cpu_offload_and_qfloat8" "sequential_cpu_offload"
GPU_memory_mode     = "model_cpu_offload_and_qfloat8"
# EasyAnimateV5.1 support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.1. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
teacache_threshold  = 0.08

# Config and model path
config_path         = "config/easyanimate_video_v5.1_magvit_qwen.yaml"
model_name          = "models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"

# Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" "DDIM" "Flow"
# EasyAnimateV3 support "Euler" "Euler A" "DPM++" "PNDM"
# EasyAnimateV4 and V5 support "Euler" "Euler A" "DPM++" "PNDM" "DDIM".
# EasyAnimateV5.1 supports Flow.
sampler_name        = "Flow"

# Load pretrained model if need
transformer_path    = None
# Only V1 does need a motion module
motion_module_path  = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [384, 672]
# In EasyAnimateV3, V4, the video_length of video is 1 ~ 144. 
# In EasyAnimateV5, V5.1, the video_length of video is 1 ~ 49.
# If u want to generate a image, please set the video_length = 1.
video_length        = 49
fps                 = 8

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you are preparing to redraw the reference video, set validation_video and validation_video_mask. 
# If you do not use validation_video_mask, the entire video will be redrawn; 
# if you use validation_video_mask, as shown in asset/mask.jpg, only a portion of the video will be redrawn.
# Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70
validation_video        = "asset/1.mp4"
validation_video_mask   = None 
denoise_strength        = 0.70

# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
prompt                  = "一只穿着小外套的猫咪正安静地坐在花园的秋千上弹吉他。它的小外套精致而合身，增添了几分俏皮与可爱。晚霞的余光洒在它柔软的毛皮上，给它的毛发镀上了一层温暖的金色光辉。和煦的微风轻轻拂过，带来阵阵花香和草木的气息，令人心旷神怡。周围斑驳的光影随着音乐的旋律轻轻摇曳，仿佛整个花园都在为这只小猫咪的演奏伴舞。阳光透过树叶间的缝隙，投下一片片光影交错的图案，与悠扬的吉他声交织在一起，营造出一种梦幻而宁静的氛围。猫咪专注而投入地弹奏着，每一个音符都似乎充满了魔力，让这个傍晚变得更加美好。"
negative_prompt         = "扭曲的身体，肢体残缺，文本字幕，漫画，静止，丑陋，错误，乱码。"
# 
# Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
# prompt                  = "A cute cat is playing the guitar. "
# negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55
save_path               = "samples/easyanimate-videos_v2v"

config = OmegaConf.load(config_path)

# Get Transformer
Choosen_Transformer3DModel = name_to_transformer3d[
    config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')
]

transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
if weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
    transformer_additional_kwargs["upcast_attention"] = True

transformer = Choosen_Transformer3DModel.from_pretrained_2d(
    model_name, 
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs,
    torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
    low_cpu_mem_usage=True,
)

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
Choosen_AutoencoderKL = name_to_autoencoder_magvit[
    config['vae_kwargs'].get('vae_type', 'AutoencoderKL')
]
vae = Choosen_AutoencoderKL.from_pretrained(
    model_name, 
    subfolder="vae",
    vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])
).to(weight_dtype)
if weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
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

if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
    tokenizer = BertTokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )
    if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
        tokenizer_2 = Qwen2Tokenizer.from_pretrained(
            os.path.join(model_name, "tokenizer_2")
        )
    else:
        tokenizer_2 = T5Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer_2"
        )
else:
    if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            os.path.join(model_name, "tokenizer")
        )
    else:
        tokenizer = T5Tokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
    tokenizer_2 = None

if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
    text_encoder = BertModel.from_pretrained(
        model_name, subfolder="text_encoder"
    ).to(weight_dtype)
    if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
        text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
            os.path.join(model_name, "text_encoder_2"),
            torch_dtype=weight_dtype,
        )
    else:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder_2"
        ).to(weight_dtype)
else:
    if config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
        text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
            os.path.join(model_name, "text_encoder"),
            torch_dtype=weight_dtype,
        )
    else:
        text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(weight_dtype)
    text_encoder_2 = None

if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_name, subfolder="image_encoder"
    ).to("cuda", weight_dtype)
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        model_name, subfolder="image_encoder"
    )
else:
    clip_image_encoder = None
    clip_image_processor = None

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "Flow": FlowMatchEulerDiscreteScheduler,
}[sampler_name]

scheduler = Choosen_Scheduler.from_pretrained(
    model_name, 
    subfolder="scheduler"
)

pipeline = EasyAnimateInpaintPipeline(
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
    clip_image_processor=clip_image_processor,
)

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline._manual_cpu_offload_in_sequential_cpu_offload = []
    for name, _text_encoder in zip(["text_encoder", "text_encoder_2"], [pipeline.text_encoder, pipeline.text_encoder_2]):
        if isinstance(_text_encoder, Qwen2VLForConditionalGeneration):
            if hasattr(_text_encoder, "visual"):
                del _text_encoder.visual
            convert_model_weight_to_float8(_text_encoder)
            convert_weight_dtype_wrapper(_text_encoder, weight_dtype)
            pipeline._manual_cpu_offload_in_sequential_cpu_offload = [name]
    pipeline.enable_sequential_cpu_offload()
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    for _text_encoder in [pipeline.text_encoder, pipeline.text_encoder_2]:
        if hasattr(_text_encoder, "visual"):
            del _text_encoder.visual
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

coefficients = get_teacache_coefficients(model_name)
if coefficients is not None and enable_teacache:
    print(f"Enable TeaCache with threshold: {teacache_threshold}.")
    pipeline.transformer.enable_teacache(num_inference_steps, teacache_threshold, coefficients=coefficients)

generator = torch.Generator(device="cuda").manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device="cuda", dtype=weight_dtype)

if vae.cache_mag_vae:
    video_length = int((video_length - 1) // vae.mini_batch_encoder * vae.mini_batch_encoder) + 1 if video_length != 1 else 1
else:
    video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1
input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, video_length=video_length, fps=fps, validation_video_mask=validation_video_mask, sample_size=sample_size)

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
        strength     = denoise_strength
    ).frames

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device="cuda", dtype=weight_dtype)
    
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