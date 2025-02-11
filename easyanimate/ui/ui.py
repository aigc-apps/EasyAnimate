"""Modified from https://github.com/guoyww/AnimateDiff/blob/main/app.py
"""
import base64
import gc
import json
import os
import random
from datetime import datetime
from glob import glob

import cv2
import gradio as gr
import numpy as np
import pkg_resources
import requests
import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       FlowMatchEulerDiscreteScheduler, PNDMScheduler)
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from transformers import (BertModel, BertTokenizer, CLIPImageProcessor,
                          CLIPVisionModelWithProjection, Qwen2Tokenizer,
                          Qwen2VLForConditionalGeneration, T5EncoderModel,
                          T5Tokenizer)

from ..data.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from ..models import name_to_autoencoder_magvit, name_to_transformer3d
from ..models.transformer3d import get_teacache_coefficients
from ..pipeline.pipeline_easyanimate import EasyAnimatePipeline
from ..pipeline.pipeline_easyanimate_control import EasyAnimateControlPipeline
from ..pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from ..utils.fp8_optimization import (convert_model_weight_to_float8,
                                      convert_weight_dtype_wrapper)
from ..utils.lora_utils import merge_lora, unmerge_lora
from ..utils.utils import (get_image_to_video_latent,
                           get_video_to_video_latent,
                           get_width_and_height_from_image_and_base_resolution,
                           save_videos_grid)

ddpm_scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}
flow_scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
}
all_cheduler_dict = {**ddpm_scheduler_dict, **flow_scheduler_dict}

gradio_version = pkg_resources.get_distribution("gradio").version
gradio_version_is_above_4 = True if int(gradio_version.split('.')[0]) >= 4 else False

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class EasyAnimateController:
    def __init__(self, GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype):
        # config dirs
        self.basedir                    = os.getcwd()
        self.config_dir                 = os.path.join(self.basedir, "config")
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.savedir                    = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample             = os.path.join(self.savedir, "sample")
        self.model_type                 = "Inpaint"
        os.makedirs(self.savedir, exist_ok=True)

        self.diffusion_transformer_list = []
        self.motion_module_list      = []
        self.personalized_model_list = []
        
        self.refresh_diffusion_transformer()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.transformer           = None
        self.pipeline              = None
        self.motion_module_path    = "none"
        self.base_model_path       = "none"
        self.lora_model_path       = "none"
        self.GPU_memory_mode       = GPU_memory_mode
        self.enable_teacache       = enable_teacache
        self.teacache_threshold    = teacache_threshold
        
        self.weight_dtype          = weight_dtype
        self.edition               = "v5.1"
        self.inference_config      = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v5.1_magvit_qwen.yaml"))

    def refresh_diffusion_transformer(self):
        self.diffusion_transformer_list = sorted(glob(os.path.join(self.diffusion_transformer_dir, "*/")))

    def refresh_motion_module(self):
        motion_module_list = sorted(glob(os.path.join(self.motion_module_dir, "*.safetensors")))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = sorted(glob(os.path.join(self.personalized_model_dir, "*.safetensors")))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_model_type(self, model_type):
        self.model_type = model_type
    
    def update_edition(self, edition):
        print("Update edition of EasyAnimate")
        self.edition = edition
        if edition == "v1":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v1_motion_module.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=True), gr.update(visible=True), \
                gr.update(choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]), \
                gr.update(value=512, minimum=384, maximum=704, step=32), \
                gr.update(value=512, minimum=384, maximum=704, step=32), gr.update(value=80, minimum=40, maximum=80, step=1)
        elif edition == "v2":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v2_magvit_motion_module.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=144, minimum=9, maximum=144, step=9)
        elif edition == "v3":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v3_slicevae_motion_module.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=144, minimum=8, maximum=144, step=8)
        elif edition == "v4":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v4_slicevae_multi_text_encoder.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=144, minimum=8, maximum=144, step=8)
        elif edition == "v5":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v5_magvit_multi_text_encoder.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=49, minimum=1, maximum=49, step=4)
        elif edition == "v5.1":
            self.inference_config = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_v5.1_magvit_qwen.yaml"))
            return gr.update(), gr.update(value="none"), gr.update(visible=False), gr.update(visible=False), \
                gr.update(choices=list(flow_scheduler_dict.keys()), value=list(flow_scheduler_dict.keys())[0]), \
                gr.update(value=672, minimum=128, maximum=1344, step=16), \
                gr.update(value=384, minimum=128, maximum=1344, step=16), gr.update(value=49, minimum=1, maximum=49, step=4)

    def update_diffusion_transformer(self, diffusion_transformer_dropdown):
        print("Update diffusion transformer")
        if diffusion_transformer_dropdown == "none":
            return gr.update()
        Choosen_AutoencoderKL = name_to_autoencoder_magvit[
            self.inference_config['vae_kwargs'].get('vae_type', 'AutoencoderKL')
        ]
        self.vae = Choosen_AutoencoderKL.from_pretrained(
            diffusion_transformer_dropdown, 
            subfolder="vae", 
        ).to(self.weight_dtype)
        if self.weight_dtype == torch.float16 and "v5.1" not in diffusion_transformer_dropdown.lower():
            self.vae.upcast_vae = True
            
        transformer_additional_kwargs = OmegaConf.to_container(self.inference_config['transformer_additional_kwargs'])
        if self.weight_dtype == torch.float16 and "v5.1" not in diffusion_transformer_dropdown.lower():
            transformer_additional_kwargs["upcast_attention"] = True

        # Get Transformer
        Choosen_Transformer3DModel = name_to_transformer3d[
            self.inference_config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')
        ]

        self.transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            diffusion_transformer_dropdown, 
            subfolder="transformer", 
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=torch.float8_e4m3fn if self.GPU_memory_mode == "model_cpu_offload_and_qfloat8" else self.weight_dtype,
            low_cpu_mem_usage=True,
        )
        
        if self.inference_config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            tokenizer = BertTokenizer.from_pretrained(
                diffusion_transformer_dropdown, subfolder="tokenizer"
            )
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer_2 = Qwen2Tokenizer.from_pretrained(
                    os.path.join(diffusion_transformer_dropdown, "tokenizer_2")
                )
            else:
                tokenizer_2 = T5Tokenizer.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="tokenizer_2"
                )
        else:
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    os.path.join(diffusion_transformer_dropdown, "tokenizer")
                )
            else:
                tokenizer = T5Tokenizer.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="tokenizer"
                )
            tokenizer_2 = None

        if self.inference_config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            text_encoder = BertModel.from_pretrained(
                diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype=self.weight_dtype
            )
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(diffusion_transformer_dropdown, "text_encoder_2"),
                    torch_dtype=self.weight_dtype,
                )
            else:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="text_encoder_2", torch_dtype=self.weight_dtype
                )
        else:  
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(diffusion_transformer_dropdown, "text_encoder"),
                    torch_dtype=self.weight_dtype,
                )
            else:
                text_encoder = T5EncoderModel.from_pretrained(
                    diffusion_transformer_dropdown, subfolder="text_encoder", torch_dtype=self.weight_dtype
                )
            text_encoder_2 = None

        # Get pipeline
        if self.transformer.config.in_channels != self.vae.config.latent_channels and self.inference_config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                diffusion_transformer_dropdown, subfolder="image_encoder"
            ).to(self.weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                diffusion_transformer_dropdown, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None

        # Get Scheduler
        if self.edition in ["v5.1"]:
            Choosen_Scheduler = all_cheduler_dict["Flow"]
        else:
            Choosen_Scheduler = all_cheduler_dict["Euler"]
        scheduler = Choosen_Scheduler.from_pretrained(
            diffusion_transformer_dropdown, 
            subfolder="scheduler"
        )

        if self.model_type == "Inpaint":
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                self.pipeline = EasyAnimateInpaintPipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=scheduler,
                )
        else:
            self.pipeline = EasyAnimateControlPipeline(
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=self.vae,
                transformer=self.transformer,
                scheduler=scheduler,
            )

        if self.GPU_memory_mode == "sequential_cpu_offload":
            self.pipeline._manual_cpu_offload_in_sequential_cpu_offload = []
            for name, _text_encoder in zip(["text_encoder", "text_encoder_2"], [self.pipeline.text_encoder, self.pipeline.text_encoder_2]):
                if isinstance(_text_encoder, Qwen2VLForConditionalGeneration):
                    if hasattr(_text_encoder, "visual"):
                        del _text_encoder.visual
                    convert_model_weight_to_float8(_text_encoder)
                    convert_weight_dtype_wrapper(_text_encoder, self.weight_dtype)
                    self.pipeline._manual_cpu_offload_in_sequential_cpu_offload = [name]
            self.pipeline.enable_sequential_cpu_offload()
        elif self.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            for _text_encoder in [self.pipeline.text_encoder, self.pipeline.text_encoder_2]:
                if hasattr(_text_encoder, "visual"):
                    del _text_encoder.visual
            self.pipeline.enable_model_cpu_offload()
            convert_weight_dtype_wrapper(self.pipeline.transformer, self.weight_dtype)
        else:
            self.pipeline.enable_model_cpu_offload()
        print("Update diffusion transformer done")
        return gr.update()

    def update_motion_module(self, motion_module_dropdown):
        self.motion_module_path = motion_module_dropdown
        print("Update motion module")
        if motion_module_dropdown == "none":
            return gr.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.update(value=None)
        else:
            motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
            if motion_module_dropdown.endswith(".safetensors"):
                from safetensors.torch import load_file, safe_open
                motion_module_state_dict = load_file(motion_module_dropdown)
            else:
                if not os.path.isfile(motion_module_dropdown):
                    raise RuntimeError(f"{motion_module_dropdown} does not exist")
                motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
            missing, unexpected = self.transformer.load_state_dict(motion_module_state_dict, strict=False)
            print("Update motion module done.")
            return gr.update()

    def update_base_model(self, base_model_dropdown):
        self.base_model_path = base_model_dropdown
        print("Update base model")
        if base_model_dropdown == "none":
            return gr.update()
        if self.transformer is None:
            gr.Info(f"Please select a pretrained model path.")
            return gr.update(value=None)
        else:
            base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
            print(f"From checkpoint: {base_model_dropdown}")
            if base_model_dropdown.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(base_model_dropdown)
            else:
                state_dict = torch.load(base_model_dropdown, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = self.transformer.load_state_dict(state_dict, strict=False)
            print(f"Update base done. Missing keys: {len(m)}, unexpected keys: {len(u)}")
            return gr.update()

    def update_lora_model(self, lora_model_dropdown):
        print("Update lora model")
        if lora_model_dropdown == "none":
            self.lora_model_path = "none"
            return gr.update()
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.update()

    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        validation_video_mask,
        control_video,
        denoise_strength, 
        seed_textbox,
        is_api = False,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if self.base_model_path != base_model_dropdown:
            self.update_base_model(base_model_dropdown)

        if self.motion_module_path != motion_module_dropdown:
            self.update_motion_module(motion_module_dropdown)

        if self.lora_model_path != lora_model_dropdown:
            self.update_lora_model(lora_model_dropdown)

        if control_video is not None and self.model_type == "Inpaint":
            if is_api:
                return "", f"If specifying the control video, please set the model_type == \"Control\". "
            else:
                raise gr.Error(f"If specifying the control video, please set the model_type == \"Control\". ")

        if control_video is None and self.model_type == "Control":
            if is_api:
                return "", f"If set the model_type == \"Control\", please specifying the control video. "
            else:
                raise gr.Error(f"If set the model_type == \"Control\", please specifying the control video. ")

        if resize_method == "Resize according to Reference":
            if start_image is None and validation_video is None and control_video is None:
                if is_api:
                    return "", f"Please upload an image when using \"Resize according to Reference\"."
                else:
                    raise gr.Error(f"Please upload an image when using \"Resize according to Reference\".")

            aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            if self.model_type == "Inpaint":
                if validation_video is not None:
                    original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
                else:
                    original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
            else:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
            closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
            height_slider, width_slider = [int(x / 16) * 16 for x in closest_size]

        if self.transformer.config.in_channels == self.vae.config.latent_channels and start_image is not None:
            if is_api:
                return "", f"Please select an image to video pretrained model while using image to video."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using image to video.")

        if self.transformer.config.in_channels == self.vae.config.latent_channels and generation_method == "Long Video Generation":
            if is_api:
                return "", f"Please select an image to video pretrained model while using long video generation."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using long video generation.")
        
        if start_image is None and end_image is not None:
            if is_api:
                return "", f"If specifying the ending image of the video, please specify a starting image of the video."
            else:
                raise gr.Error(f"If specifying the ending image of the video, please specify a starting image of the video.")

        fps = {"v1": 12, "v2": 24, "v3": 24, "v4": 24, "v5": 8, "v5.1": 8}[self.edition]
        is_image = True if generation_method == "Image Generation" else False

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))

        if is_xformers_available() \
            and self.inference_config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel') == 'Transformer3DModel':
            self.transformer.enable_xformers_memory_efficient_attention()

        self.pipeline.scheduler = all_cheduler_dict[sampler_dropdown].from_config(self.pipeline.scheduler.config)
        if self.lora_model_path != "none":
            # lora part
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
        
        coefficients = get_teacache_coefficients(self.base_model_path)
        if coefficients is not None and self.enable_teacache:
            print(f"Enable TeaCache with threshold: {self.teacache_threshold}.")
            self.pipeline.transformer.enable_teacache(sample_step_slider, self.teacache_threshold, coefficients=coefficients)
        
        try:
            if self.model_type == "Inpaint":
                if self.transformer.config.in_channels != self.vae.config.latent_channels:
                    if generation_method == "Long Video Generation":
                        if validation_video is not None:
                            raise gr.Error(f"Video to Video is not Support Long Video Generation now.")
                        init_frames = 0
                        last_frames = init_frames + partial_video_length
                        while init_frames < length_slider:
                            if last_frames >= length_slider:
                                _partial_video_length = length_slider - init_frames
                                if self.vae.cache_mag_vae:
                                    _partial_video_length = int((_partial_video_length - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                                else:
                                    _partial_video_length = int(_partial_video_length // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)
                                
                                if _partial_video_length <= 0:
                                    break
                            else:
                                _partial_video_length = partial_video_length

                            if last_frames >= length_slider:
                                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, video_length=_partial_video_length, sample_size=(height_slider, width_slider))
                            else:
                                input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, None, video_length=_partial_video_length, sample_size=(height_slider, width_slider))

                            with torch.no_grad():
                                sample = self.pipeline(
                                    prompt_textbox, 
                                    negative_prompt     = negative_prompt_textbox,
                                    num_inference_steps = sample_step_slider,
                                    guidance_scale      = cfg_scale_slider,
                                    width               = width_slider,
                                    height              = height_slider,
                                    video_length        = _partial_video_length,
                                    generator           = generator,

                                    video        = input_video,
                                    mask_video   = input_video_mask,
                                    strength     = 1,
                                ).frames
                            
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

                            if last_frames >= length_slider:
                                break

                            start_image = [
                                Image.fromarray(
                                    (sample[0, :, _index].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                                ) for _index in range(-overlap_video_length, 0)
                            ]

                            init_frames = init_frames + _partial_video_length - overlap_video_length
                            last_frames = init_frames + _partial_video_length
                    else:
                        if self.vae.cache_mag_vae:
                            length_slider = int((length_slider - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                        else:
                            length_slider = int(length_slider // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)
                        if validation_video is not None:
                            input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), validation_video_mask=validation_video_mask, fps=fps)
                            strength = denoise_strength
                        else:
                            input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                            strength = 1

                        sample = self.pipeline(
                            prompt_textbox,
                            negative_prompt     = negative_prompt_textbox,
                            num_inference_steps = sample_step_slider,
                            guidance_scale      = cfg_scale_slider,
                            width               = width_slider,
                            height              = height_slider,
                            video_length        = length_slider if not is_image else 1,
                            generator           = generator,

                            video        = input_video,
                            mask_video   = input_video_mask,
                            strength     = strength,
                        ).frames
                else:
                    if self.vae.cache_mag_vae:
                        length_slider = int((length_slider - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                    else:
                        length_slider = int(length_slider // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)
                    
                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        video_length        = length_slider if not is_image else 1,
                        generator           = generator
                    ).frames
            else:
                if self.vae.cache_mag_vae:
                    length_slider = int((length_slider - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                else:
                    length_slider = int(length_slider // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)
                input_video, input_video_mask, clip_image = get_video_to_video_latent(control_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), fps=fps)

                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    video_length        = length_slider if not is_image else 1,
                    generator           = generator,

                    control_video = input_video,
                ).frames
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.lora_model_path != "none":
                self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider, device="cuda", dtype=self.weight_dtype)
            if is_api:
                return "", f"Error. error information is {str(e)}"
            else:
                return gr.update(), gr.update(), f"Error. error information is {str(e)}"

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # lora part
        if self.lora_model_path != "none":
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider, device="cuda", dtype=self.weight_dtype)

        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed_textbox": seed_textbox
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_sample_path)

            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
                else:
                    return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            save_videos_grid(sample, save_sample_path, fps=fps)

            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
                else:
                    return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui(GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype):
    controller = EasyAnimateController(GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. EasyAnimate Model Type (EasyAnimate模型的种类，正常模型还是控制模型).
                """
            )
            with gr.Row():
                model_type = gr.Dropdown(
                    label="The model type of EasyAnimate (EasyAnimate模型的种类，正常模型还是控制模型)",
                    choices=["Inpaint", "Control"],
                    value="Inpaint",
                    interactive=True,
                )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. EasyAnimate Edition (EasyAnimate版本).
                """
            )
            with gr.Row():
                easyanimate_edition_dropdown = gr.Dropdown(
                    label="The config of EasyAnimate Edition (EasyAnimate版本配置)",
                    choices=["v1", "v2", "v3", "v4", "v5", "v5.1"],
                    value="v5.1",
                    interactive=True,
                )
            gr.Markdown(
                """
                ### 3. Model checkpoints (模型路径).
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path (预训练模型路径)",
                    choices=controller.diffusion_transformer_list,
                    value="none",
                    interactive=True,
                )
                diffusion_transformer_dropdown.change(
                    fn=controller.update_diffusion_transformer, 
                    inputs=[diffusion_transformer_dropdown], 
                    outputs=[diffusion_transformer_dropdown]
                )
                
                diffusion_transformer_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def refresh_diffusion_transformer():
                    controller.refresh_diffusion_transformer()
                    return gr.update(choices=controller.diffusion_transformer_list)
                diffusion_transformer_refresh_button.click(fn=refresh_diffusion_transformer, inputs=[], outputs=[diffusion_transformer_dropdown])

            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module (选择运动模块[非必需])",
                    choices=controller.motion_module_list,
                    value="none",
                    interactive=True,
                    visible=False
                )

                motion_module_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton", visible=False)
                def update_motion_module():
                    controller.refresh_motion_module()
                    return gr.update(choices=controller.motion_module_list)
                motion_module_refresh_button.click(fn=update_motion_module, inputs=[], outputs=[motion_module_dropdown])
                
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (选择基模型[非必需])",
                    choices=controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (选择LoRA模型[非必需])",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )

                lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_personalized_model()
                    return [
                        gr.update(choices=controller.personalized_model_list),
                        gr.update(choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 3. Configs for Generation (生成参数配置).
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt (正向提示词)", lines=2, value="A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical.")
            gr.Markdown(
                """
                Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability. Adding words such as "quiet, solid" to the neg prompt can increase dynamism.   
                使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性。在neg prompt中添加"安静，固定"等词语可以增加动态性。
                """
            )
            negative_prompt_textbox = gr.Textbox(label="Negative prompt (负向提示词)", lines=2, value="Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(
                            label="Sampling method (采样器种类)", 
                            choices=list(flow_scheduler_dict.keys()), value=list(flow_scheduler_dict.keys())[0]
                        )
                        sample_step_slider = gr.Slider(label="Sampling steps (生成步数)", value=50, minimum=10, maximum=100, step=1)
                        
                    resize_method = gr.Radio(
                        ["Generate by", "Resize according to Reference"],
                        value="Generate by",
                        show_label=False,
                    )
                    width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1344, step=16)
                    height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1344, step=16)
                    base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], visible=False)

                    with gr.Group():
                        generation_method = gr.Radio(
                            ["Video Generation", "Image Generation", "Long Video Generation"],
                            value="Video Generation",
                            show_label=False,
                        )
                        with gr.Row():
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=49, minimum=1,   maximum=49,  step=4)
                            overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                            partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=25, minimum=5,   maximum=49,  step=4, visible=False)
                    
                    source_method = gr.Radio(
                        ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)", "Video Control (视频控制)"],
                        value="Text to Video (文本到视频)",
                        show_label=False,
                    )
                    with gr.Column(visible = False) as image_to_video_col:
                        start_image = gr.Image(
                            label="The image at the beginning of the video (图片到视频的开始图片)",  show_label=True, 
                            elem_id="i2v_start", sources="upload", type="filepath", 
                        )
                        
                        template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                        def select_template(evt: gr.SelectData):
                            text = {
                                "asset/1.png": "A brown dog is shaking its head and sitting on a light colored sofa in a comfortable room. Behind the dog, there is a framed painting on the shelf surrounded by pink flowers. The soft and warm lighting in the room creates a comfortable atmosphere.", 
                                "asset/2.png": "A sailboat navigates through moderately rough seas, with waves and ocean spray visible. The sailboat features a white hull and sails, accompanied by an orange sail catching the wind. The sky above shows dramatic, cloudy formations with a sunset or sunrise backdrop, casting warm colors across the scene. The water reflects the golden light, enhancing the visual contrast between the dark ocean and the bright horizon. The camera captures the scene with a dynamic and immersive angle, showcasing the movement of the boat and the energy of the ocean.", 
                                "asset/3.png": "A stunningly beautiful woman with flowing long hair stands gracefully, her elegant dress rippling and billowing in the gentle wind. Petals falling off. Her serene expression and the natural movement of her attire create an enchanting and captivating scene, full of ethereal charm.", 
                                "asset/4.png": "An astronaut, clad in a full space suit with a helmet, plays an electric guitar while floating in a cosmic environment filled with glowing particles and rocky textures. The scene is illuminated by a warm light source, creating dramatic shadows and contrasts. The background features a complex geometry, similar to a space station or an alien landscape, indicating a futuristic or otherworldly setting.", 
                                "asset/5.png": "Fireworks light up the evening sky over a sprawling cityscape with gothic-style buildings featuring pointed towers and clock faces. The city is lit by both artificial lights from the buildings and the colorful bursts of the fireworks. The scene is viewed from an elevated angle, showcasing a vibrant urban environment set against a backdrop of a dramatic, partially cloudy sky at dusk.", 
                            }[template_gallery_path[evt.index]]
                            return template_gallery_path[evt.index], text

                        template_gallery = gr.Gallery(
                            template_gallery_path,
                            columns=5, rows=1,
                            height=140,
                            allow_preview=False,
                            container=False,
                            label="Template Examples",
                        )
                        template_gallery.select(select_template, None, [start_image, prompt_textbox])
                        
                        with gr.Accordion("The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", open=False):
                            end_image   = gr.Image(label="The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", show_label=False, elem_id="i2v_end", sources="upload", type="filepath")

                    with gr.Column(visible = False) as video_to_video_col:
                        with gr.Row():
                            validation_video = gr.Video(
                                label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                                elem_id="v2v", sources="upload", 
                            )
                        with gr.Accordion("The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])", open=False):
                            gr.Markdown(
                                """
                                - Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70  
                                - (请设置更大的denoise_strength，当使用validation_video_mask的时候，比如1而不是0.70)
                                """
                            )
                            validation_video_mask = gr.Image(
                                label="The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])",
                                show_label=False, elem_id="v2v_mask", sources="upload", type="filepath"
                            )
                        denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=1.00, step=0.01)

                    with gr.Column(visible = False) as control_video_col:
                        gr.Markdown(
                            """
                            Demo pose control video can be downloaded here [URL](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4).
                            Only normal controls are supported in app.py; trajectory control and camera control need ComfyUI, as shown in https://github.com/aigc-apps/EasyAnimate/tree/main/comfyui.
                            """
                        )
                        control_video = gr.Video(
                            label="The control video (用于提供控制信号的video)",  show_label=True, 
                            elem_id="v2v_control", sources="upload", 
                        )

                    cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=6.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed (随机种子)", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image (生成图片)", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation (生成视频)", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info (生成信息)",
                        value="No task currently",
                        interactive=False
                    )

            model_type.change(
                fn=controller.update_model_type, 
                inputs=[model_type], 
                outputs=[]
            )

            def upload_generation_method(generation_method, easyanimate_edition_dropdown):
                if easyanimate_edition_dropdown == "v1":
                    f_maximum = 80
                    f_value = 80
                elif easyanimate_edition_dropdown in ["v2", "v3", "v4"]:
                    f_maximum = 144
                    f_value = 144
                else:
                    f_maximum = 49
                    f_value = 49

                if generation_method == "Video Generation":
                    return [gr.update(visible=True, maximum=f_maximum, value=f_value), gr.update(visible=False), gr.update(visible=False)]
                elif generation_method == "Image Generation":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
                else:
                    return [gr.update(visible=True, maximum=1200), gr.update(visible=True), gr.update(visible=True)]
            generation_method.change(
                upload_generation_method, generation_method, [length_slider, overlap_video_length, partial_video_length]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Video to Video (视频到视频)":
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [
                    image_to_video_col, video_to_video_col, control_video_col, start_image, end_image, 
                    validation_video, validation_video_mask, control_video
                ]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            easyanimate_edition_dropdown.change(
                fn=controller.update_edition, 
                inputs=[easyanimate_edition_dropdown], 
                outputs=[
                    easyanimate_edition_dropdown, 
                    diffusion_transformer_dropdown, 
                    motion_module_dropdown, 
                    motion_module_refresh_button, 
                    sampler_dropdown, 
                    width_slider, 
                    height_slider, 
                    length_slider, 
                ]
            )
            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown,
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    overlap_video_length, 
                    partial_video_length, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    validation_video_mask,
                    control_video,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


class EasyAnimateController_Modelscope:
    def __init__(self, model_type, edition, config_path, model_name, savedir_sample, GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype):
        # Basic dir
        self.basedir                    = os.getcwd()
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.lora_model_path            = "none"
        self.savedir_sample             = savedir_sample
        self.refresh_personalized_model()
        os.makedirs(self.savedir_sample, exist_ok=True)

        # Config and model path
        self.model_type = model_type
        self.edition = edition
        self.model_name = model_name
        self.enable_teacache = enable_teacache
        self.teacache_threshold = teacache_threshold
        self.weight_dtype = weight_dtype
        self.inference_config = OmegaConf.load(config_path)
        Choosen_AutoencoderKL = name_to_autoencoder_magvit[
            self.inference_config['vae_kwargs'].get('vae_type', 'AutoencoderKL')
        ]
        self.vae = Choosen_AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae", 
        ).to(self.weight_dtype)
        if self.weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
            self.vae.upcast_vae = True
            
        transformer_additional_kwargs = OmegaConf.to_container(self.inference_config['transformer_additional_kwargs'])
        if self.weight_dtype == torch.float16 and "v5.1" not in model_name.lower():
            transformer_additional_kwargs["upcast_attention"] = True

        # Get Transformer
        Choosen_Transformer3DModel = name_to_transformer3d[
            self.inference_config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')
        ]

        self.transformer = Choosen_Transformer3DModel.from_pretrained_2d(
            model_name, 
            subfolder="transformer", 
            transformer_additional_kwargs=transformer_additional_kwargs,
            torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
            low_cpu_mem_usage=True,
        )
        
        if self.inference_config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            tokenizer = BertTokenizer.from_pretrained(
                model_name, subfolder="tokenizer"
            )
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer_2 = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_name, "tokenizer_2")
                )
            else:
                tokenizer_2 = T5Tokenizer.from_pretrained(
                    model_name, subfolder="tokenizer_2"
                )
        else:
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    os.path.join(model_name, "tokenizer")
                )
            else:
                tokenizer = T5Tokenizer.from_pretrained(
                    model_name, subfolder="tokenizer"
                )
            tokenizer_2 = None

        if self.inference_config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
            text_encoder = BertModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=self.weight_dtype
            )
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder_2 = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_name, "text_encoder_2"), torch_dtype=self.weight_dtype
                )
            else:
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    model_name, subfolder="text_encoder_2", torch_dtype=self.weight_dtype
                )
        else:  
            if self.inference_config['text_encoder_kwargs'].get('replace_t5_to_llm', False):
                text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
                    os.path.join(model_name, "text_encoder"), torch_dtype=self.weight_dtype
                )
            else:
                text_encoder = T5EncoderModel.from_pretrained(
                    model_name, subfolder="text_encoder", torch_dtype=self.weight_dtype
                )
            text_encoder_2 = None

        # Get pipeline
        if self.transformer.config.in_channels != self.vae.config.latent_channels and self.inference_config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
            clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_name, subfolder="image_encoder"
            ).to(self.weight_dtype)
            clip_image_processor = CLIPImageProcessor.from_pretrained(
                model_name, subfolder="image_encoder"
            )
        else:
            clip_image_encoder = None
            clip_image_processor = None

        # Get Scheduler
        if self.edition in ["v5.1"]:
            Choosen_Scheduler = all_cheduler_dict["Flow"]
        else:
            Choosen_Scheduler = all_cheduler_dict["Euler"]
        scheduler = Choosen_Scheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )

        if model_type == "Inpaint":
            if self.transformer.config.in_channels != self.vae.config.latent_channels:
                self.pipeline = EasyAnimateInpaintPipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=scheduler,
                    clip_image_encoder=clip_image_encoder,
                    clip_image_processor=clip_image_processor,
                )
            else:
                self.pipeline = EasyAnimatePipeline(
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=self.vae,
                    transformer=self.transformer,
                    scheduler=scheduler
                )
        else:
            self.pipeline = EasyAnimateControlPipeline(
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                vae=self.vae,
                transformer=self.transformer,
                scheduler=scheduler,
            )

        if GPU_memory_mode == "sequential_cpu_offload":
            self.pipeline._manual_cpu_offload_in_sequential_cpu_offload = []
            for name, _text_encoder in zip(["text_encoder", "text_encoder_2"], [self.pipeline.text_encoder, self.pipeline.text_encoder_2]):
                if isinstance(_text_encoder, Qwen2VLForConditionalGeneration):
                    if hasattr(_text_encoder, "visual"):
                        del _text_encoder.visual
                    convert_model_weight_to_float8(_text_encoder)
                    convert_weight_dtype_wrapper(_text_encoder, weight_dtype)
                    self.pipeline._manual_cpu_offload_in_sequential_cpu_offload = [name]
            self.pipeline.enable_sequential_cpu_offload()
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            for _text_encoder in [self.pipeline.text_encoder, self.pipeline.text_encoder_2]:
                if hasattr(_text_encoder, "visual"):
                    del _text_encoder.visual
            convert_weight_dtype_wrapper(self.pipeline.transformer, weight_dtype)
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()
        print("Update diffusion transformer done")

    def refresh_personalized_model(self):
        personalized_model_list = sorted(glob(os.path.join(self.personalized_model_dir, "*.safetensors")))
        self.personalized_model_list = [os.path.basename(p) for p in personalized_model_list]

    def update_lora_model(self, lora_model_dropdown):
        print("Update lora model")
        if lora_model_dropdown == "none":
            self.lora_model_path = "none"
            return gr.update()
        lora_model_dropdown = os.path.join(self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_path = lora_model_dropdown
        return gr.update()
    
    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        overlap_video_length, 
        partial_video_length, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        validation_video_mask,
        control_video,
        denoise_strength, 
        seed_textbox,
        is_api = False,
    ):    
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.transformer is None:
            raise gr.Error(f"Please select a pretrained model path.")

        if self.lora_model_path != lora_model_dropdown:
            print("Update lora model")
            self.update_lora_model(lora_model_dropdown)
        
        if control_video is not None and self.model_type == "Inpaint":
            if is_api:
                return "", f"If specifying the control video, please set the model_type == \"Control\". "
            else:
                raise gr.Error(f"If specifying the control video, please set the model_type == \"Control\". ")

        if control_video is None and self.model_type == "Control":
            if is_api:
                return "", f"If set the model_type == \"Control\", please specifying the control video. "
            else:
                raise gr.Error(f"If set the model_type == \"Control\", please specifying the control video. ")

        if resize_method == "Resize according to Reference":
            if start_image is None and validation_video is None and control_video is None:
                if is_api:
                    return "", f"Please upload an image when using \"Resize according to Reference\"."
                else:
                    raise gr.Error(f"Please upload an image when using \"Resize according to Reference\".")
        
            aspect_ratio_sample_size    = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
            if self.model_type == "Inpaint":
                if validation_video is not None:
                    original_width, original_height = Image.fromarray(cv2.VideoCapture(validation_video).read()[1]).size
                else:
                    original_width, original_height = start_image[0].size if type(start_image) is list else Image.open(start_image).size
            else:
                original_width, original_height = Image.fromarray(cv2.VideoCapture(control_video).read()[1]).size
            closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
            height_slider, width_slider = [int(x / 16) * 16 for x in closest_size]

        if self.transformer.config.in_channels == self.vae.config.latent_channels and start_image is not None:
            if is_api:
                return "", f"Please select an image to video pretrained model while using image to video."
            else:
                raise gr.Error(f"Please select an image to video pretrained model while using image to video.")

        if start_image is None and end_image is not None:
            if is_api:
                return "", f"If specifying the ending image of the video, please specify a starting image of the video."
            else:
                raise gr.Error(f"If specifying the ending image of the video, please specify a starting image of the video.")

        fps = {"v1": 12, "v2": 24, "v3": 24, "v4": 24, "v5": 8, "v5.1": 8}[self.edition]
        is_image = True if generation_method == "Image Generation" else False

        if int(seed_textbox) != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: seed_textbox = np.random.randint(0, 1e10)
        generator = torch.Generator(device="cuda").manual_seed(int(seed_textbox))

        self.pipeline.scheduler = all_cheduler_dict[sampler_dropdown].from_config(self.pipeline.scheduler.config)
        if self.lora_model_path != "none":
            # lora part
            self.pipeline = merge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider)
        
        coefficients = get_teacache_coefficients(self.model_name)
        if coefficients is not None and self.enable_teacache:
            print(f"Enable TeaCache with threshold: {self.teacache_threshold}.")
            self.pipeline.transformer.enable_teacache(sample_step_slider, self.teacache_threshold, coefficients=coefficients)
        
        try:
            if self.model_type == "Inpaint":
                if self.vae.cache_mag_vae:
                    length_slider = int((length_slider - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                else:
                    length_slider = int(length_slider // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)
                    
                if self.transformer.config.in_channels != self.vae.config.latent_channels:
                    if validation_video is not None:
                        input_video, input_video_mask, clip_image = get_video_to_video_latent(validation_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), validation_video_mask=validation_video_mask, fps=fps)
                        strength = denoise_strength
                    else:
                        input_video, input_video_mask, clip_image = get_image_to_video_latent(start_image, end_image, length_slider if not is_image else 1, sample_size=(height_slider, width_slider))
                        strength = 1

                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        video_length        = length_slider if not is_image else 1,
                        generator           = generator,

                        video        = input_video,
                        mask_video   = input_video_mask,
                        strength     = strength,
                    ).frames
                else:
                    sample = self.pipeline(
                        prompt_textbox,
                        negative_prompt     = negative_prompt_textbox,
                        num_inference_steps = sample_step_slider,
                        guidance_scale      = cfg_scale_slider,
                        width               = width_slider,
                        height              = height_slider,
                        video_length        = length_slider if not is_image else 1,
                        generator           = generator
                    ).frames
            else:
                if self.vae.cache_mag_vae:
                    length_slider = int((length_slider - 1) // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder) + 1
                else:
                    length_slider = int(length_slider // self.vae.mini_batch_encoder * self.vae.mini_batch_encoder)

                input_video, input_video_mask, clip_image = get_video_to_video_latent(control_video, length_slider if not is_image else 1, sample_size=(height_slider, width_slider), fps=fps)

                sample = self.pipeline(
                    prompt_textbox,
                    negative_prompt     = negative_prompt_textbox,
                    num_inference_steps = sample_step_slider,
                    guidance_scale      = cfg_scale_slider,
                    width               = width_slider,
                    height              = height_slider,
                    video_length        = length_slider if not is_image else 1,
                    generator           = generator,

                    control_video = input_video,
                ).frames
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if self.lora_model_path != "none":
                self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider, device="cuda", dtype=self.weight_dtype)
            if is_api:
                return "", f"Error. error information is {str(e)}"
            else:
                return gr.update(), gr.update(), f"Error. error information is {str(e)}"

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # lora part
        if self.lora_model_path != "none":
            self.pipeline = unmerge_lora(self.pipeline, self.lora_model_path, multiplier=lora_alpha_slider, device="cuda", dtype=self.weight_dtype)

        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_sample_path)
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
                else:
                    return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            save_videos_grid(sample, save_sample_path, fps=fps)
            if is_api:
                return save_sample_path, "Success"
            else:
                if gradio_version_is_above_4:
                    return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
                else:
                    return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui_modelscope(model_type, edition, config_path, model_name, savedir_sample, GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype):
    controller = EasyAnimateController_Modelscope(model_type, edition, config_path, model_name, savedir_sample, GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints (模型路径).
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path (预训练模型路径)",
                    choices=[model_name],
                    value=model_name,
                    interactive=False,
                )
            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module (选择运动模块[非必需])",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (选择基模型[非必需])",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                with gr.Column(visible=False):
                    gr.Markdown(
                        """
                        ### Minimalism is an example portrait of Lora, triggered by specific prompt words. More details can be found on [Wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora).
                        """
                    )
                    with gr.Row():
                        lora_model_dropdown = gr.Dropdown(
                            label="Select LoRA model",
                            choices=["none"],
                            value="none",
                            interactive=False,
                        )

                        lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for Generation (生成参数配置).
                """
            )

            prompt_textbox = gr.Textbox(label="Prompt (正向提示词)", lines=2, value="A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical.")
            gr.Markdown(
                """
                Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability. Adding words such as "quiet, solid" to the neg prompt can increase dynamism.   
                使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性。在neg prompt中添加"安静，固定"等词语可以增加动态性。
                """
            )
            negative_prompt_textbox = gr.Textbox(label="Negative prompt (负向提示词)", lines=2, value="Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        if edition in ["v5.1"]:
                            sampler_dropdown   = gr.Dropdown(
                                label="Sampling method (采样器种类)", 
                                choices=list(flow_scheduler_dict.keys()), value=list(flow_scheduler_dict.keys())[0]
                            )
                        else:
                            sampler_dropdown   = gr.Dropdown(
                                label="Sampling method (采样器种类)", 
                                choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]
                            )
                        sample_step_slider = gr.Slider(label="Sampling steps (生成步数)", value=50, minimum=10, maximum=50, step=1, interactive=False)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=512, minimum=384, maximum=704, step=32)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=False,
                            )
                            length_slider = gr.Slider(label="Animation length (视频帧数)", value=80,  minimum=40,  maximum=96,   step=1)
                        overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                        partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=72, minimum=8,   maximum=144,  step=8, visible=False)
                        cfg_scale_slider = gr.Slider(label="CFG Scale (引导系数)",        value=6.0, minimum=0,   maximum=20)
                    else:
                        resize_method = gr.Radio(
                            ["Generate by", "Resize according to Reference"],
                            value="Generate by",
                            show_label=False,
                        )
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1344, step=16, interactive=False)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1344, step=16, interactive=False)
                        base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], interactive=False, visible=False)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=True,
                            )
                            if edition in ["v2", "v3", "v4"]:
                                length_slider = gr.Slider(label="Animation length (视频帧数)", value=144, minimum=8,   maximum=144,  step=8)
                            else:
                                length_slider = gr.Slider(label="Animation length (视频帧数)", value=49, minimum=5,   maximum=49,  step=4)
                            overlap_video_length = gr.Slider(label="Overlap length (视频续写的重叠帧数)", value=4, minimum=1,   maximum=4,  step=1, visible=False)
                            partial_video_length = gr.Slider(label="Partial video generation length (每个部分的视频生成帧数)", value=72, minimum=8,   maximum=144,  step=8, visible=False)
                        
                        source_method = gr.Radio(
                            ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)", "Video Control (视频控制)"],
                            value="Text to Video (文本到视频)",
                            show_label=False,
                        )
                        with gr.Column(visible = False) as image_to_video_col:
                            with gr.Row():
                                start_image = gr.Image(label="The image at the beginning of the video (图片到视频的开始图片)", show_label=True, elem_id="i2v_start", sources="upload", type="filepath")
                            
                            template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                            def select_template(evt: gr.SelectData):
                                text = {
                                    "asset/1.png": "A brown dog is shaking its head and sitting on a light colored sofa in a comfortable room. Behind the dog, there is a framed painting on the shelf surrounded by pink flowers. The soft and warm lighting in the room creates a comfortable atmosphere.", 
                                    "asset/2.png": "A sailboat navigates through moderately rough seas, with waves and ocean spray visible. The sailboat features a white hull and sails, accompanied by an orange sail catching the wind. The sky above shows dramatic, cloudy formations with a sunset or sunrise backdrop, casting warm colors across the scene. The water reflects the golden light, enhancing the visual contrast between the dark ocean and the bright horizon. The camera captures the scene with a dynamic and immersive angle, showcasing the movement of the boat and the energy of the ocean.", 
                                    "asset/3.png": "A stunningly beautiful woman with flowing long hair stands gracefully, her elegant dress rippling and billowing in the gentle wind. Petals falling off. Her serene expression and the natural movement of her attire create an enchanting and captivating scene, full of ethereal charm.", 
                                    "asset/4.png": "An astronaut, clad in a full space suit with a helmet, plays an electric guitar while floating in a cosmic environment filled with glowing particles and rocky textures. The scene is illuminated by a warm light source, creating dramatic shadows and contrasts. The background features a complex geometry, similar to a space station or an alien landscape, indicating a futuristic or otherworldly setting.", 
                                    "asset/5.png": "Fireworks light up the evening sky over a sprawling cityscape with gothic-style buildings featuring pointed towers and clock faces. The city is lit by both artificial lights from the buildings and the colorful bursts of the fireworks. The scene is viewed from an elevated angle, showcasing a vibrant urban environment set against a backdrop of a dramatic, partially cloudy sky at dusk.", 
                                }[template_gallery_path[evt.index]]
                                return template_gallery_path[evt.index], text

                            template_gallery = gr.Gallery(
                                template_gallery_path,
                                columns=5, rows=1,
                                height=140,
                                allow_preview=False,
                                container=False,
                                label="Template Examples",
                            )
                            template_gallery.select(select_template, None, [start_image, prompt_textbox])

                            with gr.Accordion("The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", open=False):
                                end_image   = gr.Image(label="The image at the ending of the video (图片到视频的结束图片[非必需, Optional])", show_label=False, elem_id="i2v_end", sources="upload", type="filepath")

                        with gr.Column(visible = False) as video_to_video_col:
                            with gr.Row():
                                validation_video = gr.Video(
                                    label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                                    elem_id="v2v", sources="upload", 
                                ) 
                            with gr.Accordion("The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])", open=False):
                                gr.Markdown(
                                    """
                                    - Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70  
                                    - (请设置更大的denoise_strength，当使用validation_video_mask的时候，比如1而不是0.70)
                                    """
                                )
                                validation_video_mask = gr.Image(
                                    label="The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])",
                                    show_label=False, elem_id="v2v_mask", sources="upload", type="filepath"
                                )
                            denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=1.00, step=0.01)

                        with gr.Column(visible = False) as control_video_col:
                            gr.Markdown(
                                """
                                Demo pose control video can be downloaded here [URL](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4).
                                Only normal controls are supported in app.py; trajectory control and camera control need ComfyUI, as shown in https://github.com/aigc-apps/EasyAnimate/tree/main/comfyui.
                                """
                            )
                            control_video = gr.Video(
                                label="The control video (用于提供控制信号的video)",  show_label=True, 
                                elem_id="v2v_control", sources="upload", 
                            )

                    cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=6.0, minimum=0,   maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed (随机种子)", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate (生成)", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image (生成图片)", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation (生成视频)", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info (生成信息)",
                        value="No task currently",
                        interactive=False
                    )

            def upload_generation_method(generation_method):
                if edition == "v1":
                    f_maximum = 80
                    f_value = 80
                elif edition in ["v2", "v3", "v4"]:
                    f_maximum = 144
                    f_value = 144
                else:
                    f_maximum = 49
                    f_value = 49

                if generation_method == "Video Generation":
                    return gr.update(visible=True, maximum=f_maximum, value=f_value)
                elif generation_method == "Image Generation":
                    return gr.update(visible=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Video to Video (视频到视频)":
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(), gr.update(), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update()]
            source_method.change(
                upload_source_method, source_method, [
                    image_to_video_col, video_to_video_col, control_video_col, start_image, end_image, 
                    validation_video, validation_video_mask, control_video
                ]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown, 
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    overlap_video_length, 
                    partial_video_length, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    validation_video_mask,
                    control_video,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller


def post_eas(
    diffusion_transformer_dropdown, motion_module_dropdown,
    base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
    prompt_textbox, negative_prompt_textbox, 
    sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
    base_resolution, generation_method, length_slider, cfg_scale_slider, 
    start_image, end_image, validation_video, validation_video_mask, denoise_strength, seed_textbox,
):
    if start_image is not None:
        with open(start_image, 'rb') as file:
            file_content = file.read()
            start_image_encoded_content = base64.b64encode(file_content)
            start_image = start_image_encoded_content.decode('utf-8')

    if end_image is not None:
        with open(end_image, 'rb') as file:
            file_content = file.read()
            end_image_encoded_content = base64.b64encode(file_content)
            end_image = end_image_encoded_content.decode('utf-8')

    if validation_video is not None:
        with open(validation_video, 'rb') as file:
            file_content = file.read()
            validation_video_encoded_content = base64.b64encode(file_content)
            validation_video = validation_video_encoded_content.decode('utf-8')

    if validation_video_mask is not None:
        with open(validation_video_mask, 'rb') as file:
            file_content = file.read()
            validation_video_mask_encoded_content = base64.b64encode(file_content)
            validation_video_mask = validation_video_mask_encoded_content.decode('utf-8')


    datas = {
        "base_model_path": base_model_dropdown,
        "motion_module_path": motion_module_dropdown,
        "lora_model_path": lora_model_dropdown, 
        "lora_alpha_slider": lora_alpha_slider, 
        "prompt_textbox": prompt_textbox, 
        "negative_prompt_textbox": negative_prompt_textbox, 
        "sampler_dropdown": sampler_dropdown, 
        "sample_step_slider": sample_step_slider, 
        "resize_method": resize_method,
        "width_slider": width_slider, 
        "height_slider": height_slider, 
        "base_resolution": base_resolution,
        "generation_method": generation_method,
        "length_slider": length_slider,
        "cfg_scale_slider": cfg_scale_slider,
        "start_image": start_image,
        "end_image": end_image,
        "validation_video": validation_video,
        "validation_video_mask": validation_video_mask,
        "denoise_strength": denoise_strength,
        "seed_textbox": seed_textbox,
    }

    session = requests.session()
    session.headers.update({"Authorization": os.environ.get("EAS_TOKEN")})

    response = session.post(url=f'{os.environ.get("EAS_URL")}/easyanimate/infer_forward', json=datas, timeout=300)

    outputs = response.json()
    return outputs


class EasyAnimateController_EAS:
    def __init__(self, edition, config_path, model_name, savedir_sample):
        self.savedir_sample = savedir_sample
        os.makedirs(self.savedir_sample, exist_ok=True)

    def generate(
        self,
        diffusion_transformer_dropdown,
        motion_module_dropdown,
        base_model_dropdown,
        lora_model_dropdown, 
        lora_alpha_slider,
        prompt_textbox, 
        negative_prompt_textbox, 
        sampler_dropdown, 
        sample_step_slider, 
        resize_method,
        width_slider, 
        height_slider, 
        base_resolution, 
        generation_method, 
        length_slider, 
        cfg_scale_slider, 
        start_image, 
        end_image, 
        validation_video,
        validation_video_mask,
        denoise_strength, 
        seed_textbox
    ):
        is_image = True if generation_method == "Image Generation" else False

        outputs = post_eas(
            diffusion_transformer_dropdown, motion_module_dropdown,
            base_model_dropdown, lora_model_dropdown, lora_alpha_slider,
            prompt_textbox, negative_prompt_textbox, 
            sampler_dropdown, sample_step_slider, resize_method, width_slider, height_slider,
            base_resolution, generation_method, length_slider, cfg_scale_slider, 
            start_image, end_image, validation_video, validation_video_mask, denoise_strength, 
            seed_textbox
        )
        try:
            base64_encoding = outputs["base64_encoding"]
        except:
            return gr.Image(visible=False, value=None), gr.Video(None, visible=True), outputs["message"]
            
        decoded_data = base64.b64decode(base64_encoding)

        if not os.path.exists(self.savedir_sample):
            os.makedirs(self.savedir_sample, exist_ok=True)
        index = len([path for path in os.listdir(self.savedir_sample)]) + 1
        prefix = str(index).zfill(3)
        
        if is_image or length_slider == 1:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".png")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(value=save_sample_path, visible=True), gr.Video(value=None, visible=False), "Success"
            else:
                return gr.Image.update(value=save_sample_path, visible=True), gr.Video.update(value=None, visible=False), "Success"
        else:
            save_sample_path = os.path.join(self.savedir_sample, prefix + f".mp4")
            with open(save_sample_path, "wb") as file:
                file.write(decoded_data)
            if gradio_version_is_above_4:
                return gr.Image(visible=False, value=None), gr.Video(value=save_sample_path, visible=True), "Success"
            else:
                return gr.Image.update(visible=False, value=None), gr.Video.update(value=save_sample_path, visible=True), "Success"


def ui_eas(edition, config_path, model_name, savedir_sample):
    controller = EasyAnimateController_EAS(edition, config_path, model_name, savedir_sample)

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # EasyAnimate: An End-to-End Solution for High-Resolution and Long Video Generation 
             
            Generate your videos easily.   

            EasyAnimate is an end-to-end solution for generating high-resolution and long videos. We can train transformer based diffusion generators, train VAEs for processing long videos, and preprocess metadata.   
            
            [Github](https://github.com/aigc-apps/EasyAnimate/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model checkpoints.
                """
            )
            with gr.Row():
                diffusion_transformer_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=[model_name],
                    value=model_name,
                    interactive=False,
                )
            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model",
                    choices=["none"],
                    value="none",
                    interactive=False,
                    visible=False
                )
                with gr.Column(visible=False):
                    gr.Markdown(
                        """
                        ### Minimalism is an example portrait of Lora, triggered by specific prompt words. More details can be found on [Wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora).
                        """
                    )
                    with gr.Row():
                        lora_model_dropdown = gr.Dropdown(
                            label="Select LoRA model",
                            choices=["none"],
                            value="none",
                            interactive=False,
                        )

                        lora_alpha_slider = gr.Slider(label="LoRA alpha (LoRA权重)", value=0.55, minimum=0, maximum=2, interactive=True)
                
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for Generation.
                """
            )
            
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical.")
            gr.Markdown(
                """
                Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability. Adding words such as "quiet, solid" to the neg prompt can increase dynamism.   
                使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性。在neg prompt中添加"安静，固定"等词语可以增加动态性。
                """
            )
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code." )
                
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        if edition in ["v5.1"]:
                            sampler_dropdown   = gr.Dropdown(
                                label="Sampling method (采样器种类)", 
                                choices=list(flow_scheduler_dict.keys()), value=list(flow_scheduler_dict.keys())[0]
                            )
                        else:
                            sampler_dropdown   = gr.Dropdown(
                                label="Sampling method (采样器种类)", 
                                choices=list(ddpm_scheduler_dict.keys()), value=list(ddpm_scheduler_dict.keys())[0]
                            )
                        sample_step_slider = gr.Slider(label="Sampling steps", value=50, minimum=10, maximum=50, step=1, interactive=False)
                    
                    if edition == "v1":
                        width_slider     = gr.Slider(label="Width",            value=512, minimum=384, maximum=704, step=32)
                        height_slider    = gr.Slider(label="Height",           value=512, minimum=384, maximum=704, step=32)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=False,
                            )
                            length_slider    = gr.Slider(label="Animation length", value=80,  minimum=40,  maximum=96,   step=1)
                        cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6.0, minimum=0,   maximum=20)
                    else:
                        resize_method = gr.Radio(
                            ["Generate by", "Resize according to Reference"],
                            value="Generate by",
                            show_label=False,
                        )
                        width_slider     = gr.Slider(label="Width (视频宽度)",            value=672, minimum=128, maximum=1344, step=16, interactive=False)
                        height_slider    = gr.Slider(label="Height (视频高度)",           value=384, minimum=128, maximum=1344, step=16, interactive=False)
                        base_resolution  = gr.Radio(label="Base Resolution of Pretrained Models", value=512, choices=[512, 768, 960], interactive=False, visible=False)

                        with gr.Group():
                            generation_method = gr.Radio(
                                ["Video Generation", "Image Generation"],
                                value="Video Generation",
                                show_label=False,
                                visible=True,
                            )
                            if edition in ["v2", "v3", "v4"]:
                                length_slider = gr.Slider(label="Animation length (视频帧数)", value=144, minimum=8,   maximum=144,  step=8)
                            else:
                                length_slider = gr.Slider(label="Animation length (视频帧数)", value=49, minimum=5,   maximum=49,  step=4)
                        
                        source_method = gr.Radio(
                            ["Text to Video (文本到视频)", "Image to Video (图片到视频)", "Video to Video (视频到视频)"],
                            value="Text to Video (文本到视频)",
                            show_label=False,
                        )
                        with gr.Column(visible = False) as image_to_video_col:
                            start_image = gr.Image(label="The image at the beginning of the video", show_label=True, elem_id="i2v_start", sources="upload", type="filepath")
                            
                            template_gallery_path = ["asset/1.png", "asset/2.png", "asset/3.png", "asset/4.png", "asset/5.png"]
                            def select_template(evt: gr.SelectData):
                                text = {
                                    "asset/1.png": "A brown dog is shaking its head and sitting on a light colored sofa in a comfortable room. Behind the dog, there is a framed painting on the shelf surrounded by pink flowers. The soft and warm lighting in the room creates a comfortable atmosphere.", 
                                    "asset/2.png": "A sailboat navigates through moderately rough seas, with waves and ocean spray visible. The sailboat features a white hull and sails, accompanied by an orange sail catching the wind. The sky above shows dramatic, cloudy formations with a sunset or sunrise backdrop, casting warm colors across the scene. The water reflects the golden light, enhancing the visual contrast between the dark ocean and the bright horizon. The camera captures the scene with a dynamic and immersive angle, showcasing the movement of the boat and the energy of the ocean.", 
                                    "asset/3.png": "A stunningly beautiful woman with flowing long hair stands gracefully, her elegant dress rippling and billowing in the gentle wind. Petals falling off. Her serene expression and the natural movement of her attire create an enchanting and captivating scene, full of ethereal charm.", 
                                    "asset/4.png": "An astronaut, clad in a full space suit with a helmet, plays an electric guitar while floating in a cosmic environment filled with glowing particles and rocky textures. The scene is illuminated by a warm light source, creating dramatic shadows and contrasts. The background features a complex geometry, similar to a space station or an alien landscape, indicating a futuristic or otherworldly setting.", 
                                    "asset/5.png": "Fireworks light up the evening sky over a sprawling cityscape with gothic-style buildings featuring pointed towers and clock faces. The city is lit by both artificial lights from the buildings and the colorful bursts of the fireworks. The scene is viewed from an elevated angle, showcasing a vibrant urban environment set against a backdrop of a dramatic, partially cloudy sky at dusk.", 
                                }[template_gallery_path[evt.index]]
                                return template_gallery_path[evt.index], text

                            template_gallery = gr.Gallery(
                                template_gallery_path,
                                columns=5, rows=1,
                                height=140,
                                allow_preview=False,
                                container=False,
                                label="Template Examples",
                            )
                            template_gallery.select(select_template, None, [start_image, prompt_textbox])

                            with gr.Accordion("The image at the ending of the video (Optional)", open=False):
                                end_image   = gr.Image(label="The image at the ending of the video (Optional)", show_label=True, elem_id="i2v_end", sources="upload", type="filepath")
                        
                        with gr.Column(visible = False) as video_to_video_col:
                            with gr.Row():
                                validation_video = gr.Video(
                                    label="The video to convert (视频转视频的参考视频)",  show_label=True, 
                                    elem_id="v2v", sources="upload", 
                                )
                            with gr.Accordion("The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])", open=False):
                                gr.Markdown(
                                    """
                                    - Please set a larger denoise_strength when using validation_video_mask, such as 1.00 instead of 0.70  
                                    - (请设置更大的denoise_strength，当使用validation_video_mask的时候，比如1而不是0.70)
                                    """
                                )
                                validation_video_mask = gr.Image(
                                    label="The mask of the video to inpaint (视频重新绘制的mask[非必需, Optional])",
                                    show_label=False, elem_id="v2v_mask", sources="upload", type="filepath"
                                )
                            denoise_strength = gr.Slider(label="Denoise strength (重绘系数)", value=0.70, minimum=0.10, maximum=1.00, step=0.01)

                        cfg_scale_slider  = gr.Slider(label="CFG Scale (引导系数)",        value=6.0, minimum=0,   maximum=20)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=43)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(
                            fn=lambda: gr.Textbox(value=random.randint(1, 1e8)) if gradio_version_is_above_4 else gr.Textbox.update(value=random.randint(1, 1e8)), 
                            inputs=[], 
                            outputs=[seed_textbox]
                        )

                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                with gr.Column():
                    result_image = gr.Image(label="Generated Image", interactive=False, visible=False)
                    result_video = gr.Video(label="Generated Animation", interactive=False)
                    infer_progress = gr.Textbox(
                        label="Generation Info",
                        value="No task currently",
                        interactive=False
                    )

            def upload_generation_method(generation_method):
                if edition == "v1":
                    f_maximum = 80
                    f_value = 80
                elif edition in ["v2", "v3", "v4"]:
                    f_maximum = 144
                    f_value = 144
                else:
                    f_maximum = 49
                    f_value = 49

                if generation_method == "Video Generation":
                    return gr.update(visible=True, maximum=f_maximum, value=f_value)
                elif generation_method == "Image Generation":
                    return gr.update(visible=False)
            generation_method.change(
                upload_generation_method, generation_method, [length_slider]
            )

            def upload_source_method(source_method):
                if source_method == "Text to Video (文本到视频)":
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)]
                elif source_method == "Image to Video (图片到视频)":
                    return [gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(value=None), gr.update(value=None)]
                else:
                    return [gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None), gr.update(), gr.update()]
            source_method.change(
                upload_source_method, source_method, [image_to_video_col, video_to_video_col, start_image, end_image, validation_video, validation_video_mask]
            )

            def upload_resize_method(resize_method):
                if resize_method == "Generate by":
                    return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]
                else:
                    return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
            resize_method.change(
                upload_resize_method, resize_method, [width_slider, height_slider, base_resolution]
            )

            generate_button.click(
                fn=controller.generate,
                inputs=[
                    diffusion_transformer_dropdown,
                    motion_module_dropdown,
                    base_model_dropdown,
                    lora_model_dropdown, 
                    lora_alpha_slider,
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    resize_method,
                    width_slider, 
                    height_slider, 
                    base_resolution, 
                    generation_method, 
                    length_slider, 
                    cfg_scale_slider, 
                    start_image, 
                    end_image, 
                    validation_video,
                    validation_video_mask,
                    denoise_strength, 
                    seed_textbox,
                ],
                outputs=[result_image, result_video, infer_progress]
            )
    return demo, controller